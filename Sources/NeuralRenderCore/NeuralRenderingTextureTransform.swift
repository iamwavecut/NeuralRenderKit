import Foundation

public enum NeuralRenderingTextureTransformError: Error, Equatable, Sendable {
  case expectedBatchOneNHWC(name: String, shape: [Int], layout: TensorLayout)
  case invalidOrigin(baseX: Int, baseY: Int)
  case invalidExtent(width: Int, height: Int)
  case invalidResource(width: Int, height: Int)
  case outsideResource(
    baseX: Int,
    baseY: Int,
    extentWidth: Int,
    extentHeight: Int,
    resourceWidth: Int,
    resourceHeight: Int
  )
  case resourceShapeMismatch(
    name: String,
    expectedWidth: Int,
    expectedHeight: Int,
    actualWidth: Int,
    actualHeight: Int
  )
}

/// Recovered normalized CUDA texture transform.
///
/// Host fields are `(baseX, baseY, extentWidth, extentHeight,
/// 1/resourceWidth, 1/resourceHeight)`. Sampling policy is semantic:
/// history uses linear filtering; motion/current/control use point filtering.
public struct NeuralRenderingTextureTransform: Equatable, Sendable {
  public let baseX: Int
  public let baseY: Int
  public let extentWidth: Int
  public let extentHeight: Int
  public let resourceWidth: Int
  public let resourceHeight: Int

  public init(
    baseX: Int,
    baseY: Int,
    extentWidth: Int,
    extentHeight: Int,
    resourceWidth: Int,
    resourceHeight: Int
  ) throws {
    guard baseX >= 0, baseY >= 0 else {
      throw NeuralRenderingTextureTransformError.invalidOrigin(
        baseX: baseX,
        baseY: baseY
      )
    }
    guard extentWidth > 0, extentHeight > 0 else {
      throw NeuralRenderingTextureTransformError.invalidExtent(
        width: extentWidth,
        height: extentHeight
      )
    }
    guard resourceWidth > 0, resourceHeight > 0 else {
      throw NeuralRenderingTextureTransformError.invalidResource(
        width: resourceWidth,
        height: resourceHeight
      )
    }
    guard baseX <= resourceWidth,
      baseY <= resourceHeight,
      extentWidth <= resourceWidth - baseX,
      extentHeight <= resourceHeight - baseY
    else {
      throw NeuralRenderingTextureTransformError.outsideResource(
        baseX: baseX,
        baseY: baseY,
        extentWidth: extentWidth,
        extentHeight: extentHeight,
        resourceWidth: resourceWidth,
        resourceHeight: resourceHeight
      )
    }
    self.baseX = baseX
    self.baseY = baseY
    self.extentWidth = extentWidth
    self.extentHeight = extentHeight
    self.resourceWidth = resourceWidth
    self.resourceHeight = resourceHeight
  }

  public func pointSample(
    _ resource: HostTensor,
    logicalWidth: Int,
    logicalHeight: Int
  ) throws -> HostTensor {
    guard logicalWidth > 0, logicalHeight > 0 else {
      throw NeuralRenderingTextureTransformError.invalidExtent(
        width: logicalWidth,
        height: logicalHeight
      )
    }
    let descriptor = resource.descriptor
    guard descriptor.shape.count == 4,
      descriptor.shape[0] == 1,
      descriptor.layout == .nhwc
    else {
      throw NeuralRenderingTextureTransformError.expectedBatchOneNHWC(
        name: descriptor.name,
        shape: descriptor.shape,
        layout: descriptor.layout
      )
    }
    let actualHeight = descriptor.shape[1]
    let actualWidth = descriptor.shape[2]
    guard actualWidth == resourceWidth, actualHeight == resourceHeight else {
      throw NeuralRenderingTextureTransformError.resourceShapeMismatch(
        name: descriptor.name,
        expectedWidth: resourceWidth,
        expectedHeight: resourceHeight,
        actualWidth: actualWidth,
        actualHeight: actualHeight
      )
    }

    let bytesPerPixel = descriptor.shape[3] * descriptor.dataType.byteWidth
    var bytes = Data()
    bytes.reserveCapacity(logicalWidth * logicalHeight * bytesPerPixel)
    for y in 0..<logicalHeight {
      for x in 0..<logicalWidth {
        let coordinate = pointCoordinate(
          logicalX: x,
          logicalY: y,
          logicalWidth: logicalWidth,
          logicalHeight: logicalHeight
        )
        let offset = (
          coordinate.y * resourceWidth + coordinate.x
        ) * bytesPerPixel
        bytes.append(contentsOf: resource.bytes[offset..<(offset + bytesPerPixel)])
      }
    }
    return try HostTensor(
      descriptor: TensorDescriptor(
        name: descriptor.name,
        shape: [1, logicalHeight, logicalWidth, descriptor.shape[3]],
        dataType: descriptor.dataType,
        layout: descriptor.layout
      ),
      bytes: bytes
    )
  }

  func resourcePixelX(_ logicalPixel: Float, logicalWidth: Int) -> Float {
    Float(baseX) + logicalPixel * Float(extentWidth) / Float(logicalWidth)
  }

  func resourcePixelY(_ logicalPixel: Float, logicalHeight: Int) -> Float {
    Float(baseY) + logicalPixel * Float(extentHeight) / Float(logicalHeight)
  }

  func pointCoordinate(
    logicalX: Int,
    logicalY: Int,
    logicalWidth: Int,
    logicalHeight: Int
  ) -> (x: Int, y: Int) {
    let x = Int(floor(resourcePixelX(Float(logicalX) + 0.5, logicalWidth: logicalWidth)))
    let y = Int(floor(resourcePixelY(Float(logicalY) + 0.5, logicalHeight: logicalHeight)))
    return (
      min(resourceWidth - 1, max(0, x)),
      min(resourceHeight - 1, max(0, y))
    )
  }
}
