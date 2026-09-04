import Foundation
import MLX

/// One-launch 3×3 convolution for the frame generation blocks: bias, the
/// clamped LeakyReLU, an optional residual add and an optional 2×2 mean pool
/// happen in the epilogue, so a block's layer costs one kernel instead of the
/// four to six MLX dispatches of the operation chain. NHWC, fp16 in/out,
/// fp32 accumulation; channel counts are multiples of 8 (inputs are padded).
enum FrameGenerationFusedConv {
  /// `NRK_FG_FUSED=0` runs the MLX convolutions (diagnostics).
  nonisolated(unsafe) static var enabled: Bool = ProcessInfo.processInfo.environment["NRK_FG_FUSED"] != "0"

  static let kernel = MLXFast.metalKernel(
    name: "nrk_fg_conv3x3",
    inputNames: ["input", "weight", "bias", "residual", "params"],
    outputNames: ["out"],
    source: #"""
      // params = [height, width, cin, cout, flags, outHeight, outWidth]
      //   flags bit 0: activation (leaky 0.01, clamp ±6), bit 1: add residual, bit 2: 2x2 mean pool.
      // One thread per (output pixel, group of 8 output channels).
      // weight layout: [tap 9][cin/4][cout] half4  (4 input channels per half4).
      const uint index = thread_position_in_grid.x;
      const int height = int(params[0]);
      const int width = int(params[1]);
      const int cin = int(params[2]);
      const int cout = int(params[3]);
      const uint flags = params[4];
      const int outHeight = int(params[5]);
      const int outWidth = int(params[6]);
      const int groups = cout / 8;
      const uint total = uint(outHeight * outWidth * groups);
      if (index >= total) { return; }
      const int group = int(index % uint(groups));
      const int pixel = int(index / uint(groups));
      const int oy = pixel / outWidth;
      const int ox = pixel % outWidth;
      const bool pool = (flags & 4u) != 0u;
      const int span = pool ? 2 : 1;
      const int cin4 = cin / 4;
      const device half4* input4 = (const device half4*)input;
      const device half4* weight4 = (const device half4*)weight;
      float acc[8];
      for (int c = 0; c < 8; ++c) { acc[c] = 0.0f; }
      for (int sy = 0; sy < span; ++sy) {
        for (int sx = 0; sx < span; ++sx) {
          const int y = oy * span + sy;
          const int x = ox * span + sx;
          float part[8];
          for (int c = 0; c < 8; ++c) { part[c] = 0.0f; }
          for (int tap = 0; tap < 9; ++tap) {
            const int iy = y + tap / 3 - 1;
            const int ix = x + tap % 3 - 1;
            if (iy < 0 || iy >= height || ix < 0 || ix >= width) { continue; }
            const device half4* row = input4 + (iy * width + ix) * cin4;
            const device half4* w = weight4 + (tap * cin4) * cout + group * 8;
            for (int k = 0; k < cin4; ++k) {
              const float4 v = float4(row[k]);
              const device half4* wk = w + k * cout;
              for (int c = 0; c < 8; ++c) {
                part[c] += dot(v, float4(wk[c]));
              }
            }
          }
          for (int c = 0; c < 8; ++c) {
            float value = part[c] + float(bias[group * 8 + c]);
            if (flags & 1u) { value = clamp(value < 0.0f ? value * 0.01f : value, -6.0f, 6.0f); }
            if (flags & 2u) { value += float(residual[((y * width + x) * cout) + group * 8 + c]); }
            acc[c] += value;
          }
        }
      }
      const float scale = pool ? 0.25f : 1.0f;
      for (int c = 0; c < 8; ++c) {
        out[(pixel * cout) + group * 8 + c] = acc[c] * scale;
      }
      """#
  )

  /// simdgroup-matrix version: a threadgroup computes a run of 16 output pixels along a
  /// row for every output channel; each simdgroup owns 16 channels and accumulates
  /// 2x2 8x8 tiles over the 9 taps and the input channels in chunks of 8, so a weight
  /// tile is loaded once per 16 pixels instead of once per pixel. Input is the
  /// zero-padded image `[H+3][W+2][cin]` (one border row/column plus a slack row).
  static let simdKernel = MLXFast.metalKernel(
    name: "nrk_fg_conv3x3_sg",
    inputNames: ["input", "weight", "bias", "residual", "params"],
    outputNames: ["out"],
    source: #"""
      // params = [height, width, cin, cout, flags]; weight layout [tap*cin + ci][cout] halves.
      const int height = int(params[0]);
      const int width = int(params[1]);
      const int cin = int(params[2]);
      const int cout = int(params[3]);
      const uint flags = params[4];
      const int paddedWidth = width + 2;
      const uint simdgroupsPerTile = uint(cout / 16);
      const uint tile = threadgroup_position_in_grid.x;
      const uint sg = simdgroup_index_in_threadgroup;
      const uint lane = thread_index_in_simdgroup;
      const int tilesPerRow = (width + 15) / 16;
      const int ty = int(tile) / tilesPerRow;
      const int x0 = (int(tile) % tilesPerRow) * 16;
      if (ty >= height || sg >= simdgroupsPerTile) { return; }
      const int n0 = int(sg) * 16;
      threadgroup float scratch[6 * 16 * 16];
      simdgroup_float8x8 acc[2][2];
      for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
      simdgroup_half8x8 a[2];
      simdgroup_half8x8 b[2];
      const device half* in = (const device half*)input;
      const device half* w = (const device half*)weight;
      for (int tap = 0; tap < 9; ++tap) {
        const int dy = tap / 3, dx = tap % 3;
        const device half* rowBase = in + ((ty + dy) * paddedWidth + (x0 + dx)) * cin;
        const device half* wBase = w + (tap * cin) * cout + n0;
        for (int c0 = 0; c0 < cin; c0 += 8) {
          simdgroup_load(a[0], rowBase + c0, ulong(cin));
          simdgroup_load(a[1], rowBase + 8 * cin + c0, ulong(cin));
          simdgroup_load(b[0], wBase + c0 * cout, ulong(cout));
          simdgroup_load(b[1], wBase + c0 * cout + 8, ulong(cout));
          simdgroup_multiply_accumulate(acc[0][0], a[0], b[0], acc[0][0]);
          simdgroup_multiply_accumulate(acc[0][1], a[0], b[1], acc[0][1]);
          simdgroup_multiply_accumulate(acc[1][0], a[1], b[0], acc[1][0]);
          simdgroup_multiply_accumulate(acc[1][1], a[1], b[1], acc[1][1]);
        }
      }
      threadgroup float* mine = scratch + sg * 256;   // [16 pixels][16 channels]
      simdgroup_store(acc[0][0], mine, 16);
      simdgroup_store(acc[0][1], mine + 8, 16);
      simdgroup_store(acc[1][0], mine + 8 * 16, 16);
      simdgroup_store(acc[1][1], mine + 8 * 16 + 8, 16);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lane; e < 256; e += 32) {
        const int m = int(e) / 16, n = int(e) % 16;
        const int x = x0 + m;
        if (x >= width) { continue; }
        float value = mine[e] + float(bias[n0 + n]);
        if (flags & 1u) { value = clamp(value < 0.0f ? value * 0.01f : value, -6.0f, 6.0f); }
        const uint at = uint((ty * width + x) * cout + n0 + n);
        if (flags & 2u) { value += float(residual[at]); }
        out[at] = value;
      }
      """#,
    header: "#include <metal_simdgroup_matrix>\n"
  )

  /// Dense `[Cout, Cin, 3, 3]` -> `[9*cinPadded][cout]` halves for the simdgroup kernel.
  static func packRows(_ weight: MLXArray, cinPadded: Int) -> MLXArray {
    let cout = weight.dim(0), cin = weight.dim(1)
    var w = weight.asType(.float32)
    if cinPadded > cin { w = padded(w, widths: [[0, 0], [0, cinPadded - cin], [0, 0], [0, 0]]) }
    // [cout, cin, 3, 3] -> [3, 3, cin, cout] -> [9*cin, cout]
    return contiguous(w.transposed(2, 3, 1, 0).reshaped([9 * cinPadded, cout])).asType(.float16)
  }

  /// Packs a dense `[Cout, Cin, 3, 3]` weight into the kernel layout `[tap][cin/4][cout][4]`
  /// (channels padded to `cinPadded`, a multiple of 4; `cout` a multiple of 8).
  static func pack(_ weight: MLXArray, cinPadded: Int) -> MLXArray {
    let cout = weight.dim(0), cin = weight.dim(1)
    precondition(cout % 8 == 0 && cinPadded % 4 == 0 && cinPadded >= cin)
    var w = weight.asType(.float32)
    if cinPadded > cin {
      w = padded(w, widths: [[0, 0], [0, cinPadded - cin], [0, 0], [0, 0]])
    }
    // [cout, cin, kh, kw] -> [kh*kw, cin/4, cout, 4]
    let packed = w.reshaped([cout, cinPadded / 4, 4, 9]).transposed(3, 1, 0, 2)
    return contiguous(packed).asType(.float16)
  }

  struct Layer {
    let packed: MLXArray   // [9, cin/4, cout, 4] fp16 (per-pixel kernel)
    let rows: MLXArray     // [9*cin, cout] fp16 (simdgroup kernel)
    let bias: MLXArray     // [cout] fp16
    let cin: Int
    let cout: Int

    init(weight: MLXArray, bias: MLXArray, cinPadded: Int? = nil) {
      let cin = cinPadded ?? ((weight.dim(1) + 7) / 8 * 8)
      self.packed = FrameGenerationFusedConv.pack(weight, cinPadded: cin)
      self.rows = FrameGenerationFusedConv.packRows(weight, cinPadded: cin)
      self.bias = bias.asType(.float16)
      self.cin = cin
      self.cout = weight.dim(0)
      eval(self.packed, self.rows, self.bias)
    }

    var simdgroupCapable: Bool { cout % 16 == 0 && cin % 8 == 0 }
  }

  /// `NRK_FG_SIMD=0` keeps the per-pixel kernel for every layer (diagnostics).
  nonisolated(unsafe) static var simdEnabled: Bool = ProcessInfo.processInfo.environment["NRK_FG_SIMD"] == "1"

  /// The simdgroup-matrix convolution (no pooling; `cout % 16 == 0`, `cin % 8 == 0`).
  static func applySimd(_ x: MLXArray, _ layer: Layer, activation: Bool, residual: MLXArray? = nil) -> MLXArray {
    let (h, w) = (x.dim(1), x.dim(2))
    precondition(x.dim(3) == layer.cin && layer.simdgroupCapable)
    let paddedInput = padded(x.asType(.float16), widths: [[0, 0], [1, 2], [1, 1], [0, 0]])
    var flags: UInt32 = activation ? 1 : 0
    if residual != nil { flags |= 2 }
    let params = MLXArray([UInt32(h), UInt32(w), UInt32(layer.cin), UInt32(layer.cout), flags])
    let tiles = ((w + 15) / 16) * h
    let group = 32 * (layer.cout / 16)
    let res = residual ?? layer.bias
    return simdKernel(
      [contiguous(paddedInput), layer.rows, layer.bias, contiguous(res.asType(.float16)), params],
      grid: (tiles * group, 1, 1),
      threadGroup: (group, 1, 1),
      outputShapes: [[1, h, w, layer.cout]],
      outputDTypes: [.float16]
    )[0]
  }

  /// `x` [1,H,W,cin] fp16 (cin == layer.cin) -> [1,H,W,cout] or [1,H/2,W/2,cout] with `pool`.
  static func apply(_ x: MLXArray, _ layer: Layer, activation: Bool, residual: MLXArray? = nil, pool: Bool = false) -> MLXArray {
    if simdEnabled, layer.simdgroupCapable {
      let y = applySimd(x, layer, activation: activation, residual: residual)
      return pool ? FrameGenerator.meanPool2(y) : y
    }
    let (h, w) = (x.dim(1), x.dim(2))
    precondition(x.dim(3) == layer.cin, "fused conv expects \(layer.cin) input channels, got \(x.dim(3))")
    let outH = pool ? h / 2 : h, outW = pool ? w / 2 : w
    var flags: UInt32 = activation ? 1 : 0
    if residual != nil { flags |= 2 }
    if pool { flags |= 4 }
    let params = MLXArray([UInt32(h), UInt32(w), UInt32(layer.cin), UInt32(layer.cout), flags, UInt32(outH), UInt32(outW)])
    let count = outH * outW * (layer.cout / 8)
    let res = residual ?? layer.bias   // any array when unused; never read
    return kernel(
      [contiguous(x.asType(.float16)), layer.packed, layer.bias, contiguous(res.asType(.float16)), params],
      grid: (count, 1, 1),
      threadGroup: (min(count, 256), 1, 1),
      outputShapes: [[1, outH, outW, layer.cout]],
      outputDTypes: [.float16]
    )[0]
  }
}
