import Foundation
import MLX

/// Single-pass Metal kernels for the element-wise glue between blocks: the
/// block-0 publication with its 2×2 average pool, the pool + publication of
/// the downsample transitions and the nearest-neighbour upsample merges of
/// the upsample blocks, the decoder input and the post block. Each kernel
/// reproduces the roundings of the MLX operation chain it replaces (half
/// arithmetic per operation, the sequential half accumulation of MLX's
/// float16 mean for the pool, E4M3 publication where the chain publishes).
enum NeuralRenderingFusedGlue {
  /// `NRK_FUSED_GLUE=0` restores the per-operation chains (diagnostics).
  nonisolated(unsafe) static var enabled: Bool =
    ProcessInfo.processInfo.environment["NRK_FUSED_GLUE"] != "0"

  static func canPool(_ input: MLXArray) -> Bool {
    enabled && input.ndim == 4 && input.shape[0] == 1 && input.dtype == .float16
      && input.shape[1].isMultiple(of: 2) && input.shape[2].isMultiple(of: 2) && input.shape[3].isMultiple(of: 4)
  }

  static func canMerge(low: MLXArray, skip: MLXArray) -> Bool {
    enabled && low.ndim == 4 && skip.ndim == 4 && low.shape[0] == 1 && skip.shape[0] == 1
      && low.dtype == .float16 && skip.dtype == .float16 && low.shape[3] == skip.shape[3]
      && skip.shape[3].isMultiple(of: 4) && low.shape[1] * 2 >= skip.shape[1] && low.shape[2] * 2 >= skip.shape[2]
  }

  private static let poolKernel = MLXFast.metalKernel(
    name: "nrk_publish_pool2",
    inputNames: ["input", "params"],
    outputNames: ["full", "pooled"],
    source: #"""
      // One thread per pooled half4: params = [pooledHeight, pooledWidth, channels, emitFull].
      const uint index = thread_position_in_grid.x;
      const uint pooledHeight = params[0];
      const uint pooledWidth = params[1];
      const uint channels = params[2];
      const bool emitFull = params[3] != 0u;
      const uint vectorsPerPixel = channels / 4;
      const uint total = pooledHeight * pooledWidth * vectorsPerPixel;
      if (index >= total) { return; }
      const uint c4 = index % vectorsPerPixel;
      const uint pixel = index / vectorsPerPixel;
      const uint py = pixel / pooledWidth;
      const uint px = pixel % pooledWidth;
      const uint width = pooledWidth * 2;
      const device half4* input4 = (const device half4*)input;
      device half4* full4 = (device half4*)full;
      device half4* pooled4 = (device half4*)pooled;
      const uint base00 = ((py * 2) * width + px * 2) * vectorsPerPixel + c4;
      const uint base01 = base00 + vectorsPerPixel;
      const uint base10 = base00 + width * vectorsPerPixel;
      const uint base11 = base10 + vectorsPerPixel;
      half4 v00 = input4[base00];
      half4 v01 = input4[base01];
      half4 v10 = input4[base10];
      half4 v11 = input4[base11];
      if (emitFull) {
        full4[base00] = half4(half(float(nrk_e4m3(v00.x))), half(float(nrk_e4m3(v00.y))), half(float(nrk_e4m3(v00.z))), half(float(nrk_e4m3(v00.w))));
        full4[base01] = half4(half(float(nrk_e4m3(v01.x))), half(float(nrk_e4m3(v01.y))), half(float(nrk_e4m3(v01.z))), half(float(nrk_e4m3(v01.w))));
        full4[base10] = half4(half(float(nrk_e4m3(v10.x))), half(float(nrk_e4m3(v10.y))), half(float(nrk_e4m3(v10.z))), half(float(nrk_e4m3(v10.w))));
        full4[base11] = half4(half(float(nrk_e4m3(v11.x))), half(float(nrk_e4m3(v11.y))), half(float(nrk_e4m3(v11.z))), half(float(nrk_e4m3(v11.w))));
      }
      // MLX's float16 mean over the (2, 2) window accumulates in half, one
      // rounding per addition, in row-major order, then scales by 0.25.
      half4 sum = v00;
      sum = sum + v01;
      sum = sum + v10;
      sum = sum + v11;
      half4 mean = sum * half4(0.25h);
      pooled4[index] = half4(half(float(nrk_e4m3(mean.x))), half(float(nrk_e4m3(mean.y))), half(float(nrk_e4m3(mean.z))), half(float(nrk_e4m3(mean.w))));
      """#,
    header: NeuralRenderingTransformerOperations.e4m3MetalHeaderText
  )

  /// `(e4m3RoundTrip(input), e4m3RoundTrip(averagePool2(input)))` for a
  /// `[1, H, W, C]` half tensor with even H and W; `emitFull: false` skips the
  /// full-resolution output (it is then a placeholder of one element).
  static func publishAndPool(_ input: MLXArray, emitFull: Bool) -> (full: MLXArray, pooled: MLXArray) {
    precondition(input.ndim == 4 && input.shape[0] == 1 && input.dtype == .float16)
    let height = input.shape[1], width = input.shape[2], channels = input.shape[3]
    precondition(height.isMultiple(of: 2) && width.isMultiple(of: 2) && channels.isMultiple(of: 4))
    let pooledHeight = height / 2, pooledWidth = width / 2
    let count = pooledHeight * pooledWidth * channels / 4
    let params = NeuralRenderingKernelParameters.array([UInt32(pooledHeight), UInt32(pooledWidth), UInt32(channels), emitFull ? UInt32(1) : UInt32(0)])
    let outputs = poolKernel(
      [input, params],
      grid: (count, 1, 1),
      threadGroup: (min(count, 256), 1, 1),
      outputShapes: [emitFull ? input.shape : [1], [1, pooledHeight, pooledWidth, channels]],
      outputDTypes: [.float16, .float16]
    )
    return (outputs[0], outputs[1])
  }

  private static let upsampleMergeKernel = MLXFast.metalKernel(
    name: "nrk_upsample_merge",
    inputNames: ["low", "skip", "lowScale", "skipScale", "params"],
    outputNames: ["output"],
    source: #"""
      // One thread per output half4: params = [height, width, channels, lowWidth, hasLowScale, publish].
      const uint index = thread_position_in_grid.x;
      const uint height = params[0];
      const uint width = params[1];
      const uint channels = params[2];
      const uint lowWidth = params[3];
      const bool hasLowScale = params[4] != 0u;
      const bool publish = params[5] != 0u;
      const uint vectorsPerPixel = channels / 4;
      const uint total = height * width * vectorsPerPixel;
      if (index >= total) { return; }
      const uint c4 = index % vectorsPerPixel;
      const uint pixel = index / vectorsPerPixel;
      const uint y = pixel / width;
      const uint x = pixel % width;
      const device half4* low4 = (const device half4*)low;
      const device half4* skip4 = (const device half4*)skip;
      const device half4* lowScale4 = (const device half4*)lowScale;
      const device half4* skipScale4 = (const device half4*)skipScale;
      device half4* output4 = (device half4*)output;
      half4 upsampled = low4[((y / 2) * lowWidth + x / 2) * vectorsPerPixel + c4];
      if (hasLowScale) { upsampled = upsampled * lowScale4[c4]; }
      half4 scaledSkip = skip4[index] * skipScale4[c4];
      half4 value = upsampled + scaledSkip;
      if (publish) {
        value = half4(half(float(nrk_e4m3(value.x))), half(float(nrk_e4m3(value.y))), half(float(nrk_e4m3(value.z))), half(float(nrk_e4m3(value.w))));
      }
      output4[index] = value;
      """#,
    header: NeuralRenderingTransformerOperations.e4m3MetalHeaderText
  )

  /// `nearestUpsample2Crop(low) [* lowScale] + skip * skipScale`, optionally
  /// E4M3-published, in one pass; `low` is `[1, h, w, C]` with `2h >= H`,
  /// `2w >= W` for the `[1, H, W, C]` skip.
  static func upsampleMerge(low: MLXArray, skip: MLXArray, lowScale: MLXArray?, skipScale: MLXArray, publish: Bool) -> MLXArray {
    precondition(low.ndim == 4 && skip.ndim == 4 && low.shape[0] == 1 && skip.shape[0] == 1)
    precondition(low.dtype == .float16 && skip.dtype == .float16)
    let height = skip.shape[1], width = skip.shape[2], channels = skip.shape[3]
    precondition(low.shape[3] == channels && low.shape[1] * 2 >= height && low.shape[2] * 2 >= width)
    precondition(channels.isMultiple(of: 4) && skipScale.shape == [channels])
    if let lowScale { precondition(lowScale.shape == [channels]) }
    let count = height * width * channels / 4
    let params = NeuralRenderingKernelParameters.array([
      UInt32(height), UInt32(width), UInt32(channels), UInt32(low.shape[2]),
      lowScale == nil ? UInt32(0) : UInt32(1), publish ? UInt32(1) : UInt32(0),
    ])
    return upsampleMergeKernel(
      [low, skip, (lowScale ?? skipScale).asType(.float16), skipScale.asType(.float16), params],
      grid: (count, 1, 1),
      threadGroup: (min(count, 256), 1, 1),
      outputShapes: [skip.shape],
      outputDTypes: [.float16]
    )[0]
  }
}
