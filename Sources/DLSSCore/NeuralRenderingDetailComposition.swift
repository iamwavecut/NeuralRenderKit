import Accelerate
import Foundation

/// Optional composition around the first-frame path that reproduces the
/// community "photoreal" recipe on top of the faithful single pass: run the
/// network on a resampled (usually 2×) frame, then split the model's change
/// into a low-pass colour/tone part and a high-pass detail part with
/// independent strengths:
///
///     result = input + colourStrength · lowPass(change) + detailStrength · highPass(change)
///
/// where `change = output − input` and `lowPass` is a Gaussian of the given
/// radius (in output pixels). Strengths of `1`/`1` return the output unchanged.
/// Measured against the user's ComfyUI CG→photoreal reference (516×718): a
/// single pass at 1× has a detail correlation of 0.18 with the target, the
/// same model at 2× reaches 0.32–0.37, and `detail 2` with `colour 1` at 2×
/// gives the best full-frame agreement (MAE 0.040 versus 0.051 before).
public enum NeuralRenderingDetailComposition {
  public enum Error: Swift.Error, Equatable, Sendable {
    case unsupportedTensor(String)
    case shapeMismatch(input: [Int], output: [Int])
    case resamplingFailed(Int)
    case invalidStrength(String)
  }

  /// Lanczos-resamples a `[1, height, width, channels]` float32 NHWC tensor.
  public static func resample(
    _ tensor: HostTensor,
    width: Int,
    height: Int
  ) throws -> HostTensor {
    let (sourceHeight, sourceWidth, channels) = try planarShape(of: tensor)
    guard width > 0, height > 0 else {
      throw Error.unsupportedTensor("resample target must be positive")
    }
    if width == sourceWidth, height == sourceHeight {
      return tensor
    }
    let planes = deinterleave(tensor, height: sourceHeight, width: sourceWidth, channels: channels)
    if sourceWidth == width * (sourceWidth / width), sourceHeight == height * (sourceHeight / height),
      sourceWidth / width == sourceHeight / height, sourceWidth / width > 1
    {
      // Integer downscale: an exact box average keeps the recomposition free of
      // resampling ringing (the detail split would otherwise read it as change).
      let factor = sourceWidth / width
      let averaged = planes.map { plane -> [Float] in
        var result = [Float](repeating: 0, count: width * height)
        let normalization = 1 / Float(factor * factor)
        for y in 0..<height {
          for x in 0..<width {
            var total: Float = 0
            for dy in 0..<factor {
              let row = (y * factor + dy) * sourceWidth + x * factor
              for dx in 0..<factor {
                total += plane[row + dx]
              }
            }
            result[y * width + x] = total * normalization
          }
        }
        return result
      }
      return try interleave(
        averaged, height: height, width: width, channels: channels, name: tensor.descriptor.name
      )
    }
    var scaled: [[Float]] = []
    scaled.reserveCapacity(channels)
    for plane in planes {
      var source = plane
      var destination = [Float](repeating: 0, count: width * height)
      let status: vImage_Error = source.withUnsafeMutableBufferPointer { sourcePointer in
        destination.withUnsafeMutableBufferPointer { destinationPointer in
          var sourceBuffer = vImage_Buffer(
            data: sourcePointer.baseAddress!,
            height: vImagePixelCount(sourceHeight),
            width: vImagePixelCount(sourceWidth),
            rowBytes: sourceWidth * MemoryLayout<Float>.stride
          )
          var destinationBuffer = vImage_Buffer(
            data: destinationPointer.baseAddress!,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: width * MemoryLayout<Float>.stride
          )
          return vImageScale_PlanarF(
            &sourceBuffer,
            &destinationBuffer,
            nil,
            vImage_Flags(kvImageHighQualityResampling)
          )
        }
      }
      guard status == kvImageNoError else {
        throw Error.resamplingFailed(Int(status))
      }
      scaled.append(destination)
    }
    return try interleave(
      scaled,
      height: height,
      width: width,
      channels: channels,
      name: tensor.descriptor.name
    )
  }

  /// Splits `output − input` into Gaussian low-pass and residual high-pass
  /// parts and recombines them with independent strengths; the result is
  /// clamped to `[0, 1]` like the display output it replaces.
  public static func compose(
    input: HostTensor,
    output: HostTensor,
    detailStrength: Float,
    colourStrength: Float,
    radius: Float = 4
  ) throws -> HostTensor {
    guard detailStrength.isFinite, colourStrength.isFinite, radius.isFinite, radius > 0 else {
      throw Error.invalidStrength("detail/colour strengths and radius must be finite and positive")
    }
    let (height, width, channels) = try planarShape(of: input)
    guard input.descriptor.shape == output.descriptor.shape else {
      throw Error.shapeMismatch(input: input.descriptor.shape, output: output.descriptor.shape)
    }
    if detailStrength == 1, colourStrength == 1 {
      return output
    }
    let inputPlanes = deinterleave(input, height: height, width: width, channels: channels)
    let outputPlanes = deinterleave(output, height: height, width: width, channels: channels)
    let kernel = gaussianKernel(radius: radius)
    var composed: [[Float]] = []
    composed.reserveCapacity(channels)
    for channel in 0..<channels {
      let inputPlane = inputPlanes[channel]
      let change = zip(outputPlanes[channel], inputPlane).map { $0 - $1 }
      let lowPass = try blur(change, height: height, width: width, kernel: kernel)
      var result = [Float](repeating: 0, count: change.count)
      for index in 0..<change.count {
        let highPass = change[index] - lowPass[index]
        let value = inputPlane[index] + colourStrength * lowPass[index] + detailStrength * highPass
        result[index] = min(max(value, 0), 1)
      }
      composed.append(result)
    }
    return try interleave(
      composed,
      height: height,
      width: width,
      channels: channels,
      name: output.descriptor.name
    )
  }

  static func gaussianKernel(radius sigma: Float) -> [Float] {
    let extent = Int((3 * sigma).rounded(.up))
    var kernel = (-extent...extent).map { offset -> Float in
      let distance = Float(offset)
      return exp(-distance * distance / (2 * sigma * sigma))
    }
    let total = kernel.reduce(0, +)
    kernel = kernel.map { $0 / total }
    return kernel
  }

  static func blur(
    _ plane: [Float],
    height: Int,
    width: Int,
    kernel: [Float]
  ) throws -> [Float] {
    var horizontal = [Float](repeating: 0, count: plane.count)
    var vertical = [Float](repeating: 0, count: plane.count)
    var source = plane
    let size = UInt32(kernel.count)
    let horizontalStatus: vImage_Error = kernel.withUnsafeBufferPointer { kernelPointer in
      source.withUnsafeMutableBufferPointer { sourcePointer in
        horizontal.withUnsafeMutableBufferPointer { destinationPointer in
          var sourceBuffer = vImage_Buffer(
            data: sourcePointer.baseAddress!,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: width * MemoryLayout<Float>.stride
          )
          var destinationBuffer = vImage_Buffer(
            data: destinationPointer.baseAddress!,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: width * MemoryLayout<Float>.stride
          )
          return vImageConvolve_PlanarF(
            &sourceBuffer, &destinationBuffer, nil, 0, 0,
            kernelPointer.baseAddress!, 1, size, 0,
            vImage_Flags(kvImageEdgeExtend)
          )
        }
      }
    }
    guard horizontalStatus == kvImageNoError else {
      throw Error.resamplingFailed(Int(horizontalStatus))
    }
    let verticalStatus: vImage_Error = kernel.withUnsafeBufferPointer { kernelPointer in
      horizontal.withUnsafeMutableBufferPointer { sourcePointer in
        vertical.withUnsafeMutableBufferPointer { destinationPointer in
          var sourceBuffer = vImage_Buffer(
            data: sourcePointer.baseAddress!,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: width * MemoryLayout<Float>.stride
          )
          var destinationBuffer = vImage_Buffer(
            data: destinationPointer.baseAddress!,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: width * MemoryLayout<Float>.stride
          )
          return vImageConvolve_PlanarF(
            &sourceBuffer, &destinationBuffer, nil, 0, 0,
            kernelPointer.baseAddress!, size, 1, 0,
            vImage_Flags(kvImageEdgeExtend)
          )
        }
      }
    }
    guard verticalStatus == kvImageNoError else {
      throw Error.resamplingFailed(Int(verticalStatus))
    }
    return vertical
  }

  private static func planarShape(of tensor: HostTensor) throws -> (Int, Int, Int) {
    let descriptor = tensor.descriptor
    guard descriptor.dataType == .float32, descriptor.layout == .nhwc,
      descriptor.shape.count == 4, descriptor.shape[0] == 1
    else {
      throw Error.unsupportedTensor(
        "expected a [1, height, width, channels] float32 NHWC tensor, got \(descriptor.shape) \(descriptor.dataType)"
      )
    }
    return (descriptor.shape[1], descriptor.shape[2], descriptor.shape[3])
  }

  private static func deinterleave(
    _ tensor: HostTensor,
    height: Int,
    width: Int,
    channels: Int
  ) -> [[Float]] {
    let values = tensor.bytes.withUnsafeBytes { raw -> [Float] in
      Array(raw.bindMemory(to: Float.self))
    }
    let pixelCount = height * width
    var planes = [[Float]](repeating: [Float](repeating: 0, count: pixelCount), count: channels)
    for pixel in 0..<pixelCount {
      let base = pixel * channels
      for channel in 0..<channels {
        planes[channel][pixel] = values[base + channel]
      }
    }
    return planes
  }

  private static func interleave(
    _ planes: [[Float]],
    height: Int,
    width: Int,
    channels: Int,
    name: String
  ) throws -> HostTensor {
    let pixelCount = height * width
    var values = [Float](repeating: 0, count: pixelCount * channels)
    for pixel in 0..<pixelCount {
      let base = pixel * channels
      for channel in 0..<channels {
        values[base + channel] = planes[channel][pixel]
      }
    }
    let descriptor = try TensorDescriptor(
      name: name,
      shape: [1, height, width, channels],
      dataType: .float32,
      layout: .nhwc
    )
    let bytes = values.withUnsafeBufferPointer { Data(buffer: $0) }
    return try HostTensor(descriptor: descriptor, bytes: bytes)
  }
}
