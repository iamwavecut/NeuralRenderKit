import Foundation

public enum NeuralRenderingNetworkGeometryError: Error, Equatable, Sendable {
  case invalidOutput(width: Int, height: Int)
  case networkSmallerThanOutput(
    outputWidth: Int,
    outputHeight: Int,
    networkWidth: Int,
    networkHeight: Int
  )
}

/// Output-to-network geometry surrounding the recovered 71-block graph.
public struct NeuralRenderingNetworkGeometry: Equatable, Sendable {
  public static let minimumExtent = 320
  public static let extentMultiple = 64

  public let outputWidth: Int
  public let outputHeight: Int
  public let networkWidth: Int
  public let networkHeight: Int

  public init(
    outputWidth: Int,
    outputHeight: Int,
    networkWidth: Int,
    networkHeight: Int
  ) throws {
    guard outputWidth > 0, outputHeight > 0 else {
      throw NeuralRenderingNetworkGeometryError.invalidOutput(
        width: outputWidth,
        height: outputHeight
      )
    }
    guard networkWidth >= outputWidth, networkHeight >= outputHeight else {
      throw NeuralRenderingNetworkGeometryError.networkSmallerThanOutput(
        outputWidth: outputWidth,
        outputHeight: outputHeight,
        networkWidth: networkWidth,
        networkHeight: networkHeight
      )
    }
    self.outputWidth = outputWidth
    self.outputHeight = outputHeight
    self.networkWidth = networkWidth
    self.networkHeight = networkHeight
  }

  public static func vendorAligned(
    outputWidth: Int,
    outputHeight: Int
  ) throws -> Self {
    try Self(
      outputWidth: outputWidth,
      outputHeight: outputHeight,
      networkWidth: alignedExtent(outputWidth),
      networkHeight: alignedExtent(outputHeight)
    )
  }

  public func sourceCoordinate(x: Int, y: Int) -> (x: Int, y: Int) {
    (
      Self.extendedCoordinate(x, extent: outputWidth),
      Self.extendedCoordinate(y, extent: outputHeight)
    )
  }

  public func cropOutput(_ tensor: HostTensor) throws -> HostTensor {
    try NeuralRenderingAlignedSubrect(
      baseX: 0,
      baseY: 0,
      width: outputWidth,
      height: outputHeight
    ).extract(from: tensor)
  }

  /// `true` when the network extent equals the logical output extent.
  public var isIdentity: Bool {
    networkWidth == outputWidth && networkHeight == outputHeight
  }

  /// Extends a logical `[1, outputHeight, outputWidth, 16]` feature tensor to
  /// the network extent.
  ///
  /// Every padded pixel copies channels 3–15 from its reflected-then-clamped
  /// source pixel and regenerates the three deterministic-noise channels from
  /// its own network coordinate, exactly as the first-frame preprocessor does
  /// when it is given this geometry directly.
  public func extendFeatureTensor(
    _ features: HostTensor,
    noiseFrameIndex: UInt32
  ) throws -> HostTensor {
    let descriptor = features.descriptor
    guard descriptor.shape == [1, outputHeight, outputWidth, 16],
      descriptor.dataType == .float32,
      descriptor.layout == .nhwc
    else {
      throw NeuralRenderingTemporalPreprocessorError.spatialShapeMismatch(
        name: descriptor.name,
        expected: [1, outputHeight, outputWidth, 16],
        actual: descriptor.shape
      )
    }
    guard !isIdentity else {
      return features
    }
    let channels = 16
    let logical = [Float](unsafeUninitializedCapacity: descriptor.elementCount) {
      buffer, count in
      features.bytes.copyBytes(to: buffer)
      count = descriptor.elementCount
    }
    var extended = [Float](repeating: 0, count: networkHeight * networkWidth * channels)
    for y in 0..<networkHeight {
      let sourceY = Self.extendedCoordinate(y, extent: outputHeight)
      for x in 0..<networkWidth {
        let sourceX = Self.extendedCoordinate(x, extent: outputWidth)
        let target = (y * networkWidth + x) * channels
        let origin = (sourceY * outputWidth + sourceX) * channels
        for channel in 0..<channels {
          extended[target + channel] = logical[origin + channel]
        }
        if x >= outputWidth || y >= outputHeight {
          let noise = NeuralRenderingFirstFramePreprocessor.deterministicNoise(
            x: x,
            y: y,
            frameIndex: noiseFrameIndex
          )
          extended[target] = noise.0
          extended[target + 1] = noise.1
          extended[target + 2] = noise.2
        }
      }
    }
    return try HostTensor(
      descriptor: TensorDescriptor(
        name: descriptor.name,
        shape: [1, networkHeight, networkWidth, channels],
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: extended.withUnsafeBytes { Data($0) }
    )
  }

  private static func alignedExtent(_ outputExtent: Int) -> Int {
    let minimum = max(minimumExtent, outputExtent)
    return (minimum + extentMultiple - 1) / extentMultiple * extentMultiple
  }

  private static func extendedCoordinate(_ coordinate: Int, extent: Int) -> Int {
    guard coordinate >= extent else { return coordinate }
    return max(0, 2 * extent - 2 - coordinate)
  }
}
