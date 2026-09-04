import Foundation
import MLX
import MLXNN

/// DLSS frame generation for plain video on MLX/Metal — the port of the
/// library's video path (two colour frames in, one interpolated frame out).
///
/// The graph, recovered from the library's kernels and verified against its
/// memory snapshots (see the Python reference `neuralrenderkit.framegen`):
///
///   a, b   = 2x2 box means of the frames, zero-padded to a multiple of 16
///   err    = [1/8, 3/4, 1/8]-blurred mean_rgb |a - b|
///   x      = [a, err, b, err, 0, t]                          (10 channels)
///   block0 : 3 stem convs (leaky-clamp, 2x2 mean pool), 8 residual convs,
///            3 activated heads on the x2-upsampled trunk, one linear head -> flow(4) mask(1) res(3)
///   block1 : the same on [warp(a, 2 f0A), warp(b, 2 f0B), f0, m0, 0, r0, t]  (18 channels)
///   export = [2 f0 + f1, m0 + m1, r0 + r1]                   (half resolution)
///   out    = m * warp(A, 2 flowA) + (1 - m) * warp(B, 2 flowB), m = sigmoid(mask) upsampled
///
/// Weights: the dense `[Cout, Cin, 3, 3]` fp16 tensors written by
/// `nrk-weights extract-fg` from the user's own `libnvidia-ngx-dlssg.so`.
public final class FrameGenerator {
  public enum Precision: String, Sendable { case float16, float32 }

  public struct Error: Swift.Error, CustomStringConvertible {
    public let description: String
  }

  private struct Block {
    var stem: [(MLXArray, MLXArray)]
    var residual: [(MLXArray, MLXArray)]
    var heads: [(MLXArray, MLXArray)]
    var outWeight: MLXArray
    var outBias: MLXArray
  }

  public let precision: Precision
  private let dtype: DType
  private let block0: Block
  private let block1: Block
  /// `NRK_FG_COMPILE=0` runs the eager graph (diagnostics).
  nonisolated(unsafe) static var compileEnabled: Bool = ProcessInfo.processInfo.environment["NRK_FG_COMPILE"] != "0"
  private lazy var compiledSynthesize: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: false) { [self] inputs in
    [self.synthesizeGraph(inputs[0], inputs[1], phase: inputs[2])]
  }

  /// Tensor names the dense weight file must contain.
  public static let tensorNames: [String] = {
    var names: [String] = []
    func pair(_ base: String) { names += ["\(base).weight", "\(base).bias"] }
    for i in 0..<3 { pair("block0.stem\(i)") }
    for i in 0..<8 { pair("block0.res\(i)") }
    for i in 0..<3 { pair("block0.bot0.head\(i)") }
    pair("block0.bot1")
    for i in 0..<2 { pair("block1.stem\(i)") }
    for i in 0..<8 { pair("block1.res\(i)") }
    for i in 0..<3 { pair("block1.bot0.head\(i)") }
    pair("block1.bot1")
    return names
  }()

  public convenience init(weightsURL: URL, precision: Precision = .float16) throws {
    let arrays = try loadArrays(url: weightsURL, stream: .cpu)
    try self.init(weights: arrays, precision: precision)
  }

  /// `weights` holds dense `[Cout, Cin, kh, kw]` tensors (any float dtype).
  public init(weights: [String: MLXArray], precision: Precision = .float16) throws {
    let missing = Self.tensorNames.filter { weights[$0] == nil }
    if !missing.isEmpty {
      throw Error(description: "frame generation weights are missing \(missing.count) tensors, e.g. \(missing.prefix(3).joined(separator: ", "))")
    }
    self.precision = precision
    let dtype: DType = precision == .float16 ? .float16 : .float32
    self.dtype = dtype
    func conv(_ name: String) throws -> (MLXArray, MLXArray) {
      let w = weights["\(name).weight"]!
      let b = weights["\(name).bias"]!
      guard w.ndim == 4, w.dim(2) == 3, w.dim(3) == 3, b.ndim == 1, b.dim(0) == w.dim(0) else {
        throw Error(description: "\(name): expected a [Cout, Cin, 3, 3] weight and a [Cout] bias, got \(w.shape) and \(b.shape)")
      }
      // MLX convolutions take [Cout, kh, kw, Cin] weights over NHWC inputs.
      return (w.transposed(0, 2, 3, 1).asType(dtype), b.asType(dtype))
    }
    func block(_ prefix: String, stems: Int) throws -> Block {
      Block(
        stem: try (0..<stems).map { try conv("\(prefix).stem\($0)") },
        residual: try (0..<8).map { try conv("\(prefix).res\($0)") },
        heads: try (0..<3).map { try conv("\(prefix).bot0.head\($0)") },
        outWeight: try conv("\(prefix).bot1").0,
        outBias: try conv("\(prefix).bot1").1
      )
    }
    block0 = try block("block0", stems: 3)
    block1 = try block("block1", stems: 2)
    for block in [block0, block1] {
      for (w, b) in block.stem + block.residual + block.heads { eval(w, b) }
      eval(block.outWeight, block.outBias)
    }
  }

  // MARK: - Building blocks

  private static func act(_ x: MLXArray) -> MLXArray {
    clip(leakyRelu(x, negativeSlope: 0.01), min: -6, max: 6)
  }

  private static func conv(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray) -> MLXArray {
    conv2d(x, w, stride: 1, padding: 1) + b
  }

  private static func padChannels(_ x: MLXArray, to channels: Int) -> MLXArray {
    let c = x.dim(3)
    if c == channels { return x }
    let zeros = MLXArray.zeros([x.dim(0), x.dim(1), x.dim(2), channels - c], dtype: x.dtype)
    return concatenated([x, zeros], axis: 3)
  }

  /// 2x2 mean pool of an NHWC array with even height and width.
  static func meanPool2(_ x: MLXArray) -> MLXArray {
    let (n, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    return x.reshaped([n, h / 2, 2, w / 2, 2, c]).mean(axes: [2, 4])
  }

  /// 2x2 box mean of a full-resolution frame, zero-padded to a multiple of 16 (the library's tile).
  static func box2(_ x: MLXArray) -> MLXArray {
    let (h, w) = (x.dim(1), x.dim(2))
    let pooled = meanPool2(x[0..., 0..<(h / 2 * 2), 0..<(w / 2 * 2), 0...])
    let hp = (pooled.dim(1) + 15) / 16 * 16
    let wp = (pooled.dim(2) + 15) / 16 * 16
    if hp == pooled.dim(1), wp == pooled.dim(2) { return pooled }
    return padded(pooled, widths: [[0, 0], [0, hp - pooled.dim(1)], [0, wp - pooled.dim(2)], [0, 0]])
  }

  /// mean_rgb |a - b| blurred with the separable [1/8, 3/4, 1/8] kernel, borders replicated.
  static func photometricError(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
    let e = abs(a - b).mean(axis: 3, keepDims: true)
    let h = e.dim(1), w = e.dim(2)
    let ep = padded(e, widths: [[0, 0], [1, 1], [1, 1], [0, 0]], mode: .edge)
    func tap(_ dy: Int, _ dx: Int) -> MLXArray { ep[0..., (1 + dy)..<(1 + dy + h), (1 + dx)..<(1 + dx + w), 0...] }
    let k: [Float] = [0.125, 0.75, 0.125]
    var acc = MLXArray.zeros(e.shape, dtype: e.dtype)
    for (i, dy) in [-1, 0, 1].enumerated() {
      for (j, dx) in [-1, 0, 1].enumerated() {
        acc = acc + tap(dy, dx) * (k[i] * k[j])
      }
    }
    return acc
  }

  private func run(_ block: Block, _ input: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    var x = input
    for (w, b) in block.stem {
      x = Self.meanPool2(Self.act(Self.conv(Self.padChannels(x, to: w.dim(3)), w, b)))
    }
    var skip = x
    for (i, (w, b)) in block.residual.enumerated() {
      let y = Self.act(Self.conv(x, w, b))
      if i % 2 == 0 {
        skip = x
        x = y
      } else {
        x = y + skip
      }
    }
    let trunk = Self.upsample2(x)
    let heads = block.heads.map { Self.act(Self.conv(trunk, $0.0, $0.1)) }
    let spans = [(0, 4), (4, 5), (5, 8)]
    let outs = spans.enumerated().map { (i, span) -> MLXArray in
      let w = block.outWeight[span.0..<span.1]
      let b = block.outBias[span.0..<span.1]
      return Self.conv(Self.upsample2(heads[i]), w, b)
    }
    return (outs[0], outs[1], outs[2])
  }

  // MARK: - Metal kernels

  /// Bilinear x2 upsample with the half-pixel convention (PyTorch align_corners=False), NHWC.
  private static let up2Kernel = MLXFast.metalKernel(
    name: "nrk_fg_up2",
    inputNames: ["input", "params"],
    outputNames: ["out"],
    source: #"""
      // One thread per output pixel: params = [inHeight, inWidth, channels].
      const uint index = thread_position_in_grid.x;
      const int ih = int(params[0]);
      const int iw = int(params[1]);
      const uint channels = params[2];
      const uint oh = uint(ih) * 2;
      const uint ow = uint(iw) * 2;
      if (index >= oh * ow) { return; }
      const uint y = index / ow;
      const uint x = index % ow;
      const float sx = max((float(x) + 0.5f) * 0.5f - 0.5f, 0.0f);
      const float sy = max((float(y) + 0.5f) * 0.5f - 0.5f, 0.0f);
      const int x0 = min(int(sx), iw - 1);
      const int y0 = min(int(sy), ih - 1);
      const int x1 = min(x0 + 1, iw - 1);
      const int y1 = min(y0 + 1, ih - 1);
      const float tx = sx - float(x0);
      const float ty = sy - float(y0);
      for (uint c = 0; c < channels; ++c) {
        const float v00 = float(input[(y0 * iw + x0) * channels + c]);
        const float v01 = float(input[(y0 * iw + x1) * channels + c]);
        const float v10 = float(input[(y1 * iw + x0) * channels + c]);
        const float v11 = float(input[(y1 * iw + x1) * channels + c]);
        out[index * channels + c] = (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v01 + (1 - tx) * ty * v10 + tx * ty * v11;
      }
      """#
  )

  /// `[1,h,w,C]` -> `[1,2h,2w,C]`, bilinear, half-pixel centres.
  static func upsample2(_ x: MLXArray) -> MLXArray {
    let (h, w, c) = (x.dim(1), x.dim(2), x.dim(3))
    let params = MLXArray([UInt32(h), UInt32(w), UInt32(c)])
    let count = 4 * h * w
    return up2Kernel(
      [contiguous(x), params],
      grid: (count, 1, 1),
      threadGroup: (min(count, 256), 1, 1),
      outputShapes: [[1, 2 * h, 2 * w, c]],
      outputDTypes: [x.dtype]
    )[0]
  }

  /// Backward bilinear warp of an NHWC image by per-pixel offsets (in pixels), borders clamped.
  private static let warpKernel = MLXFast.metalKernel(
    name: "nrk_fg_warp",
    inputNames: ["image", "flow", "params"],
    outputNames: ["out"],
    source: #"""
      // One thread per output pixel: params = [height, width, channels].
      const uint index = thread_position_in_grid.x;
      const uint height = params[0];
      const uint width = params[1];
      const uint channels = params[2];
      if (index >= height * width) { return; }
      const uint y = index / width;
      const uint x = index % width;
      const float px = float(x) + float(flow[index * 2]);
      const float py = float(y) + float(flow[index * 2 + 1]);
      const float fx0 = floor(px);
      const float fy0 = floor(py);
      const float tx = px - fx0;
      const float ty = py - fy0;
      const int x0 = clamp(int(fx0), 0, int(width) - 1);
      const int x1 = clamp(int(fx0) + 1, 0, int(width) - 1);
      const int y0 = clamp(int(fy0), 0, int(height) - 1);
      const int y1 = clamp(int(fy0) + 1, 0, int(height) - 1);
      for (uint c = 0; c < channels; ++c) {
        const float v00 = float(image[(y0 * width + x0) * channels + c]);
        const float v01 = float(image[(y0 * width + x1) * channels + c]);
        const float v10 = float(image[(y1 * width + x0) * channels + c]);
        const float v11 = float(image[(y1 * width + x1) * channels + c]);
        out[index * channels + c] = (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v01 + (1 - tx) * ty * v10 + tx * ty * v11;
      }
      """#
  )

  /// The output kernel: upsample the half-resolution export (plain x/scale
  /// sampling, index-clamped), warp both frames by the scaled flows, blend by the mask.
  private static let composeKernel = MLXFast.metalKernel(
    name: "nrk_fg_compose",
    inputNames: ["a", "b", "coarse", "params"],
    outputNames: ["out"],
    source: #"""
      // One thread per output pixel: params = [height, width, coarseHeight, coarseWidth, scale(bits)].
      const uint index = thread_position_in_grid.x;
      const uint height = params[0];
      const uint width = params[1];
      const int ch = int(params[2]);
      const int cw = int(params[3]);
      const float scale = as_type<float>(params[4]);
      if (index >= height * width) { return; }
      const uint y = index / width;
      const uint x = index % width;
      const float u = min(float(x) / scale, float(cw - 1));
      const float v = min(float(y) / scale, float(ch - 1));
      const float u0 = floor(u);
      const float v0 = floor(v);
      const float tu = u - u0;
      const float tv = v - v0;
      const int cx0 = clamp(int(u0), 0, cw - 1);
      const int cx1 = clamp(int(u0) + 1, 0, cw - 1);
      const int cy0 = clamp(int(v0), 0, ch - 1);
      const int cy1 = clamp(int(v0) + 1, 0, ch - 1);
      float up[5];
      for (uint c = 0; c < 5; ++c) {
        const float s00 = float(coarse[(cy0 * cw + cx0) * 5 + c]);
        const float s01 = float(coarse[(cy0 * cw + cx1) * 5 + c]);
        const float s10 = float(coarse[(cy1 * cw + cx0) * 5 + c]);
        const float s11 = float(coarse[(cy1 * cw + cx1) * 5 + c]);
        up[c] = (1 - tu) * (1 - tv) * s00 + tu * (1 - tv) * s01 + (1 - tu) * tv * s10 + tu * tv * s11;
      }
      const float m = up[4];
      float rgb[3] = {0, 0, 0};
      for (uint which = 0; which < 2; ++which) {
        const float px = float(x) + up[which * 2] * scale;
        const float py = float(y) + up[which * 2 + 1] * scale;
        const float fx0 = floor(px);
        const float fy0 = floor(py);
        const float tx = px - fx0;
        const float ty = py - fy0;
        const int x0 = clamp(int(fx0), 0, int(width) - 1);
        const int x1 = clamp(int(fx0) + 1, 0, int(width) - 1);
        const int y0 = clamp(int(fy0), 0, int(height) - 1);
        const int y1 = clamp(int(fy0) + 1, 0, int(height) - 1);
        const float weight = which == 0 ? m : (1 - m);
        for (uint c = 0; c < 3; ++c) {
          float v00, v01, v10, v11;
          if (which == 0) {
            v00 = float(a[(y0 * width + x0) * 3 + c]); v01 = float(a[(y0 * width + x1) * 3 + c]);
            v10 = float(a[(y1 * width + x0) * 3 + c]); v11 = float(a[(y1 * width + x1) * 3 + c]);
          } else {
            v00 = float(b[(y0 * width + x0) * 3 + c]); v01 = float(b[(y0 * width + x1) * 3 + c]);
            v10 = float(b[(y1 * width + x0) * 3 + c]); v11 = float(b[(y1 * width + x1) * 3 + c]);
          }
          rgb[c] += weight * ((1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v01 + (1 - tx) * ty * v10 + tx * ty * v11);
        }
      }
      for (uint c = 0; c < 3; ++c) { out[index * 3 + c] = clamp(rgb[c], 0.0f, 1.0f); }
      """#
  )

  /// `image` [1,H,W,C], `flow` [1,H,W,2] pixel offsets -> [1,H,W,C].
  static func warp(_ image: MLXArray, flow: MLXArray) -> MLXArray {
    let (h, w, c) = (image.dim(1), image.dim(2), image.dim(3))
    let params = MLXArray([UInt32(h), UInt32(w), UInt32(c)])
    let count = h * w
    return warpKernel(
      [contiguous(image), contiguous(flow), params],
      grid: (count, 1, 1),
      threadGroup: (min(count, 256), 1, 1),
      outputShapes: [image.shape],
      outputDTypes: [image.dtype]
    )[0]
  }

  /// `a`, `b` [1,H,W,3] full frames; `coarse` [1,h,w,5] = (flowA, flowB, sigmoid mask) -> [1,H,W,3].
  static func compose(_ a: MLXArray, _ b: MLXArray, coarse: MLXArray, scale: Float = 2) -> MLXArray {
    let (h, w) = (a.dim(1), a.dim(2))
    let params = MLXArray([UInt32(h), UInt32(w), UInt32(coarse.dim(1)), UInt32(coarse.dim(2)), scale.bitPattern])
    let count = h * w
    return composeKernel(
      [contiguous(a), contiguous(b), contiguous(coarse), params],
      grid: (count, 1, 1),
      threadGroup: (min(count, 256), 1, 1),
      outputShapes: [a.shape],
      outputDTypes: [a.dtype]
    )[0]
  }

  // MARK: - Public API

  /// Half-resolution export `[1, h, w, 8]`: flowA.xy, flowB.xy, mask logit, residual.rgb.
  public func synthesize(_ aFull: MLXArray, _ bFull: MLXArray, phase: Float) -> MLXArray {
    let phaseArray = MLXArray(phase)
    if Self.compileEnabled {
      return compiledSynthesize([aFull, bFull, phaseArray])[0]
    }
    return synthesizeGraph(aFull, bFull, phase: phaseArray)
  }

  private func synthesizeGraph(_ aFull: MLXArray, _ bFull: MLXArray, phase: MLXArray) -> MLXArray {
    let a = Self.box2(aFull.asType(dtype))
    let b = Self.box2(bFull.asType(dtype))
    let err = Self.photometricError(a, b)
    let zero = MLXArray.zeros(err.shape, dtype: dtype)
    let t = broadcast(phase.asType(dtype), to: err.shape)
    let candA = concatenated([a, err], axis: 3)
    let candB = concatenated([b, err], axis: 3)
    let (f0c, m0c, r0c) = run(block0, concatenated([candA, candB, zero, t], axis: 3))
    let f0 = Self.upsample2(f0c), m0 = Self.upsample2(m0c), r0 = Self.upsample2(r0c)
    let warpedA = Self.warp(candA, flow: f0[0..., 0..., 0..., 0..<2] * 2)
    let warpedB = Self.warp(candB, flow: f0[0..., 0..., 0..., 2..<4] * 2)
    let (f1, m1, r1) = run(block1, concatenated([warpedA, warpedB, f0, m0, zero, r0, t], axis: 3))
    let flow: MLXArray = f0 * 2 + f1
    let mask: MLXArray = m0 + m1
    let residual: MLXArray = r0 + r1
    return concatenated([flow, mask, residual], axis: 3)
  }

  /// The interpolated frame `[1, H, W, 3]` in [0, 1] between `a` and `b` (NHWC RGB in [0, 1]) at `phase`.
  public func interpolate(_ a: MLXArray, _ b: MLXArray, phase: Float = 0.5) throws -> MLXArray {
    guard a.ndim == 4, a.dim(0) == 1, a.dim(3) == 3 else { throw Error(description: "frames must be [1, H, W, 3], got \(a.shape)") }
    guard a.shape == b.shape else { throw Error(description: "frames differ in shape: \(a.shape) vs \(b.shape)") }
    let export = synthesize(a, b, phase: phase)
    let coarse = concatenated([export[0..., 0..., 0..., 0..<4], sigmoid(export[0..., 0..., 0..., 4..<5])], axis: 3)
    let out = Self.compose(a.asType(.float32), b.asType(.float32), coarse: coarse.asType(.float32))
    eval(out)
    return out
  }

  /// `factor - 1` intermediate frames at phases k/factor.
  public func generate(_ a: MLXArray, _ b: MLXArray, factor: Int) throws -> [MLXArray] {
    guard factor >= 2 else { throw Error(description: "factor must be at least 2") }
    return try (1..<factor).map { try interpolate(a, b, phase: Float($0) / Float(factor)) }
  }
}
