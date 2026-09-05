import Foundation

public enum NeuralRenderingAlignedSubrectError: Error, Equatable, Sendable {
  case invalidOrigin(baseX: Int, baseY: Int)
  case invalidExtent(width: Int, height: Int)
  case expectedBatchOneNHWC(shape: [Int], layout: TensorLayout)
  case outsideResource(
    baseX: Int,
    baseY: Int,
    width: Int,
    height: Int,
    resourceWidth: Int,
    resourceHeight: Int
  )
}

/// Integer-aligned subset of the recovered texture-transform contract.
///
/// Extracting the compact logical view before preprocessing is equivalent to
/// the vendor affine transform when the declared extent equals the processing
/// size. Use `NeuralRenderingTextureTransform` for recovered extent-mismatched
/// history and motion sampling.
public struct NeuralRenderingAlignedSubrect: Equatable, Sendable {
  public let baseX: Int
  public let baseY: Int
  public let width: Int
  public let height: Int

  public init(baseX: Int, baseY: Int, width: Int, height: Int) throws {
    guard baseX >= 0, baseY >= 0 else {
      throw NeuralRenderingAlignedSubrectError.invalidOrigin(
        baseX: baseX,
        baseY: baseY
      )
    }
    guard width > 0, height > 0 else {
      throw NeuralRenderingAlignedSubrectError.invalidExtent(
        width: width,
        height: height
      )
    }
    self.baseX = baseX
    self.baseY = baseY
    self.width = width
    self.height = height
  }

  public func extract(from tensor: HostTensor) throws -> HostTensor {
    let descriptor = tensor.descriptor
    guard descriptor.shape.count == 4,
      descriptor.shape[0] == 1,
      descriptor.layout == .nhwc
    else {
      throw NeuralRenderingAlignedSubrectError.expectedBatchOneNHWC(
        shape: descriptor.shape,
        layout: descriptor.layout
      )
    }
    let resourceHeight = descriptor.shape[1]
    let resourceWidth = descriptor.shape[2]
    guard baseX <= resourceWidth,
      baseY <= resourceHeight,
      width <= resourceWidth - baseX,
      height <= resourceHeight - baseY
    else {
      throw NeuralRenderingAlignedSubrectError.outsideResource(
        baseX: baseX,
        baseY: baseY,
        width: width,
        height: height,
        resourceWidth: resourceWidth,
        resourceHeight: resourceHeight
      )
    }

    let bytesPerPixel = descriptor.shape[3] * descriptor.dataType.byteWidth
    let resourceRowBytes = resourceWidth * bytesPerPixel
    let rowBytes = width * bytesPerPixel
    var bytes = Data()
    bytes.reserveCapacity(height * rowBytes)
    for y in 0..<height {
      let offset = (baseY + y) * resourceRowBytes + baseX * bytesPerPixel
      bytes.append(contentsOf: tensor.bytes[offset..<(offset + rowBytes)])
    }
    return try HostTensor(
      descriptor: TensorDescriptor(
        name: descriptor.name,
        shape: [1, height, width, descriptor.shape[3]],
        dataType: descriptor.dataType,
        layout: .nhwc
      ),
      bytes: bytes
    )
  }
}
