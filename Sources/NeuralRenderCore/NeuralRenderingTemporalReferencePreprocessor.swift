import Foundation

public enum NeuralRenderingTemporalPreprocessorError: Error, Equatable, Sendable {
  case expectedFloat32NHWC(
    name: String,
    channels: Int,
    shape: [Int],
    dataType: TensorDataType,
    layout: TensorLayout
  )
  case spatialShapeMismatch(name: String, expected: [Int], actual: [Int])
  case invalidMotionExtent(width: Int, height: Int)
  case nonFiniteMotionScale
  case nonFiniteJitterDelta
}

public enum NeuralRenderingDepthGuideMode: String, Equatable, Sendable {
  case observedZeroDescriptor = "observed-zero-descriptor"
  case closestDepth = "closest-depth"
}

public protocol NeuralRenderingTemporalFeaturePreprocessing: Sendable {
  var featureControls: NeuralRenderingFeatureControls { get }

  func makeFeatureTensor(
    currentColor: HostTensor,
    historyColor: HostTensor,
    controlMask: HostTensor?,
    normalizedMotion: HostTensor,
    depth: HostTensor,
    depthInverted: Bool,
    noiseFrameIndex: UInt32
  ) throws -> HostTensor
}

extension NeuralRenderingTemporalFeaturePreprocessing {
  public var featureControls: NeuralRenderingFeatureControls { .init() }
}

public struct NeuralRenderingCPUTemporalFeaturePreprocessor:
  NeuralRenderingTemporalFeaturePreprocessing
{
  private let depthGuideMode: NeuralRenderingDepthGuideMode
  private let historyTransform: NeuralRenderingTextureTransform?
  private let motionTransform: NeuralRenderingTextureTransform?
  public let featureControls: NeuralRenderingFeatureControls

  public init() {
    self.init(
      depthGuideMode: .observedZeroDescriptor,
      historyTransform: nil,
      motionTransform: nil
    )
  }

  public init(depthGuideMode: NeuralRenderingDepthGuideMode) {
    self.init(
      depthGuideMode: depthGuideMode,
      historyTransform: nil,
      motionTransform: nil
    )
  }

  public init(
    depthGuideMode: NeuralRenderingDepthGuideMode,
    historyTransform: NeuralRenderingTextureTransform?,
    motionTransform: NeuralRenderingTextureTransform?,
    featureControls: NeuralRenderingFeatureControls = .init()
  ) {
    self.depthGuideMode = depthGuideMode
    self.historyTransform = historyTransform
    self.motionTransform = motionTransform
    self.featureControls = featureControls
  }

  public func makeFeatureTensor(
    currentColor: HostTensor,
    historyColor: HostTensor,
    controlMask: HostTensor? = nil,
    normalizedMotion: HostTensor,
    depth: HostTensor,
    depthInverted: Bool,
    noiseFrameIndex: UInt32
  ) throws -> HostTensor {
    try NeuralRenderingTemporalReferencePreprocessor.makeFeatureTensor(
      currentColor: currentColor,
      historyColor: historyColor,
      historyTransform: historyTransform,
      controlMask: controlMask,
      normalizedMotion: normalizedMotion,
      motionTransform: motionTransform,
      depth: depth,
      depthInverted: depthInverted,
      depthGuideMode: depthGuideMode,
      noiseFrameIndex: noiseFrameIndex,
      normalizedStyle: featureControls.normalizedStyle,
      localToneStrength: featureControls.localToneStrength,
      localStructureStrength: featureControls.localStructureStrength,
      automaticMask: featureControls.automaticMask
    )
  }
}

/// CPU temporal reference with motion expressed as normalized history-UV offsets.
///
/// Vendor-specific engine motion units remain an adapter outside this type.
/// Positive normalized motion is added to the current pixel center; recovered
/// texture transforms map history and motion backing resources.
/// ponytail: scalar CPU oracle; replace with MLX/Metal only after golden parity.
public enum NeuralRenderingTemporalReferencePreprocessor {
  /// Converts current-pixel-to-history motion in pixels to normalized UV offsets.
  ///
  /// The scale sign is preserved. `effectiveWidth` and `effectiveHeight` are the
  /// active motion-vector extents, not necessarily the backing texture size.
  /// `jitterDeltaX` and `jitterDeltaY` are previous-minus-current jitter in
  /// pixels and are not multiplied by the engine motion-vector scale.
  public static func normalizePixelMotion(
    _ pixelMotion: HostTensor,
    scaleX: Float,
    scaleY: Float,
    effectiveWidth: Int,
    effectiveHeight: Int,
    jitterDeltaX: Float = 0,
    jitterDeltaY: Float = 0
  ) throws -> HostTensor {
    try require(pixelMotion, semanticName: "pixelMotion", channels: 2)
    guard effectiveWidth > 0, effectiveHeight > 0 else {
      throw NeuralRenderingTemporalPreprocessorError.invalidMotionExtent(
        width: effectiveWidth,
        height: effectiveHeight
      )
    }
    guard scaleX.isFinite, scaleY.isFinite else {
      throw NeuralRenderingTemporalPreprocessorError.nonFiniteMotionScale
    }
    guard jitterDeltaX.isFinite, jitterDeltaY.isFinite else {
      throw NeuralRenderingTemporalPreprocessorError.nonFiniteJitterDelta
    }
    let xScale = scaleX / Float(effectiveWidth)
    let yScale = scaleY / Float(effectiveHeight)
    var values = floatValues(in: pixelMotion)
    if jitterDeltaX == 0, jitterDeltaY == 0 {
      for offset in stride(from: 0, to: values.count, by: 2) {
        values[offset] *= xScale
        values[offset + 1] *= yScale
      }
    } else {
      let normalizedJitterX = jitterDeltaX / Float(effectiveWidth)
      let normalizedJitterY = jitterDeltaY / Float(effectiveHeight)
      for offset in stride(from: 0, to: values.count, by: 2) {
        values[offset] = values[offset] * xScale + normalizedJitterX
        values[offset + 1] = values[offset + 1] * yScale + normalizedJitterY
      }
    }
    return try HostTensor(
      descriptor: TensorDescriptor(
        name: "motion",
        shape: pixelMotion.descriptor.shape,
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: values.withUnsafeBytes { Data($0) }
    )
  }

  public static func makeFeatureTensor(
    currentColor: HostTensor,
    historyColor: HostTensor,
    historyTransform: NeuralRenderingTextureTransform? = nil,
    controlMask: HostTensor? = nil,
    normalizedMotion: HostTensor,
    motionTransform: NeuralRenderingTextureTransform? = nil,
    depth: HostTensor,
    depthInverted: Bool = false,
    depthGuideMode: NeuralRenderingDepthGuideMode = .observedZeroDescriptor,
    noiseFrameIndex: UInt32 = 1,
    normalizedStyle: Float = 0,
    localToneStrength: Float = 1,
    localStructureStrength: Float = 1,
    automaticMask: NeuralRenderingAutomaticMaskConfiguration? = nil
  ) throws -> HostTensor {
    try require(currentColor, semanticName: "color", channels: 3)
    try require(historyColor, semanticName: "history", channels: 3)
    try require(normalizedMotion, semanticName: "motion", channels: 2)
    try require(depth, semanticName: "depth", channels: 1)
    let spatialShape = Array(currentColor.descriptor.shape.prefix(3))
    let resolvedHistoryTransform = try resolveTransform(
      historyTransform,
      tensor: historyColor,
      name: "history",
      logicalShape: spatialShape
    )
    let resolvedMotionTransform = try resolveTransform(
      motionTransform,
      tensor: normalizedMotion,
      name: "motion",
      logicalShape: spatialShape
    )
    try requireSpatialShape(depth, name: "depth", expected: spatialShape)

    let firstFrame = try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(
      from: currentColor,
      noiseFrameIndex: noiseFrameIndex,
      normalizedStyle: normalizedStyle,
      localToneStrength: localToneStrength,
      localStructureStrength: localStructureStrength,
      automaticMask: automaticMask,
      controlMask: controlMask
    )
    var features = floatValues(in: firstFrame)
    let history = floatValues(in: historyColor)
    let motion = floatValues(in: normalizedMotion)
    let depthValues = floatValues(in: depth)
    let height = spatialShape[1]
    let width = spatialShape[2]
    let historyResourceWidth = historyColor.descriptor.shape[2]
    let historyResourceHeight = historyColor.descriptor.shape[1]
    let motionResourceWidth = normalizedMotion.descriptor.shape[2]

    for y in 0..<height {
      for x in 0..<width {
        let selected =
          switch depthGuideMode {
          case .observedZeroDescriptor:
            (x: x, y: y)
          case .closestDepth:
            closestDepthCoordinate(
              x: x,
              y: y,
              width: width,
              height: height,
              depth: depthValues,
              inverted: depthInverted
            )
          }
        let motionCoordinate = resolvedMotionTransform.pointCoordinate(
          logicalX: selected.x,
          logicalY: selected.y,
          logicalWidth: width,
          logicalHeight: height
        )
        let motionOffset = (
          motionCoordinate.y * motionResourceWidth + motionCoordinate.x
        ) * 2
        let u = (Float(x) + 0.5) / Float(width) + motion[motionOffset]
        let v = (Float(y) + 0.5) / Float(height) + motion[motionOffset + 1]
        let sampled = sampleFiveTapCatmullRom(
          history,
          resourceWidth: historyResourceWidth,
          resourceHeight: historyResourceHeight,
          logicalWidth: width,
          logicalHeight: height,
          transform: resolvedHistoryTransform,
          u: u,
          v: v
        )
        let featureOffset = (y * width + x) * 16 + 7
        features[featureOffset] = scaledColor(sampled.red)
        features[featureOffset + 1] = scaledColor(sampled.green)
        features[featureOffset + 2] = scaledColor(sampled.blue)
      }
    }

    return try HostTensor(
      descriptor: firstFrame.descriptor,
      bytes: features.withUnsafeBytes { Data($0) }
    )
  }

  private static func closestDepthCoordinate(
    x: Int,
    y: Int,
    width: Int,
    height: Int,
    depth: [Float],
    inverted: Bool
  ) -> (x: Int, y: Int) {
    var selected = (x: x, y: y)
    var selectedDepth = depth[y * width + x]
    for (dx, dy) in [(-1, -1), (1, -1), (-1, 1), (1, 1)] {
      let candidateX = min(width - 1, max(0, x + dx))
      let candidateY = min(height - 1, max(0, y + dy))
      let candidateDepth = depth[candidateY * width + candidateX]
      let closer =
        inverted
        ? candidateDepth > selectedDepth
        : candidateDepth < selectedDepth
      if closer {
        selected = (candidateX, candidateY)
        selectedDepth = candidateDepth
      }
    }
    return selected
  }

  private static func sampleFiveTapCatmullRom(
    _ image: [Float],
    resourceWidth: Int,
    resourceHeight: Int,
    logicalWidth: Int,
    logicalHeight: Int,
    transform: NeuralRenderingTextureTransform,
    u: Float,
    v: Float
  ) -> (red: Float, green: Float, blue: Float) {
    let x = catmullCoordinates(normalized: u, dimension: logicalWidth)
    let y = catmullCoordinates(normalized: v, dimension: logicalHeight)
    let leftWeight = x.w0 * y.g
    let topWeight = x.g * y.w0
    let middleWeight = x.g * y.g
    let bottomWeight = x.g * y.w3
    let rightWeight = x.w3 * y.g
    let left = sampleLinear(
      image,
      width: resourceWidth,
      height: resourceHeight,
      x: transform.resourcePixelX(x.outer0, logicalWidth: logicalWidth),
      y: transform.resourcePixelY(y.middle, logicalHeight: logicalHeight)
    )
    let top = sampleLinear(
      image,
      width: resourceWidth,
      height: resourceHeight,
      x: transform.resourcePixelX(x.middle, logicalWidth: logicalWidth),
      y: transform.resourcePixelY(y.outer0, logicalHeight: logicalHeight)
    )
    let middle = sampleLinear(
      image,
      width: resourceWidth,
      height: resourceHeight,
      x: transform.resourcePixelX(x.middle, logicalWidth: logicalWidth),
      y: transform.resourcePixelY(y.middle, logicalHeight: logicalHeight)
    )
    let bottom = sampleLinear(
      image,
      width: resourceWidth,
      height: resourceHeight,
      x: transform.resourcePixelX(x.middle, logicalWidth: logicalWidth),
      y: transform.resourcePixelY(y.outer3, logicalHeight: logicalHeight)
    )
    let right = sampleLinear(
      image,
      width: resourceWidth,
      height: resourceHeight,
      x: transform.resourcePixelX(x.outer3, logicalWidth: logicalWidth),
      y: transform.resourcePixelY(y.middle, logicalHeight: logicalHeight)
    )
    let weightSum =
      leftWeight + topWeight + middleWeight + bottomWeight
      + rightWeight
    return (
      weighted(
        left.red, top.red, middle.red, bottom.red, right.red,
        leftWeight, topWeight, middleWeight, bottomWeight, rightWeight
      ) / weightSum,
      weighted(
        left.green, top.green, middle.green, bottom.green, right.green,
        leftWeight, topWeight, middleWeight, bottomWeight, rightWeight
      ) / weightSum,
      weighted(
        left.blue, top.blue, middle.blue, bottom.blue, right.blue,
        leftWeight, topWeight, middleWeight, bottomWeight, rightWeight
      ) / weightSum
    )
  }

  private static func catmullCoordinates(
    normalized: Float,
    dimension: Int
  ) -> (outer0: Float, middle: Float, outer3: Float, w0: Float, w3: Float, g: Float) {
    let pixel = normalized * Float(dimension) - 0.5
    let baseIndex = floor(pixel)
    let t = min(1, max(0, pixel - baseIndex))
    let square = t * t
    let cube = square * t
    let w0 = -0.5 * t + square - 0.5 * cube
    let w1 = 1 - 2.5 * square + 1.5 * cube
    let w2 = 0.5 * t + 2 * square - 1.5 * cube
    let w3 = -0.5 * square + 0.5 * cube
    let g = w1 + w2
    let base = baseIndex + 0.5
    let lower = Float(0.5)
    let upper = Float(dimension) - 0.5
    return (
      min(upper, max(lower, base - 1)),
      min(upper, max(lower, base + w2 / g)),
      min(upper, max(lower, base + 2)),
      w0,
      w3,
      g
    )
  }

  private static func sampleLinear(
    _ image: [Float],
    width: Int,
    height: Int,
    x: Float,
    y: Float
  ) -> (red: Float, green: Float, blue: Float) {
    let pixelX = x - 0.5
    let pixelY = y - 0.5
    let x0 = min(width - 1, max(0, Int(floor(pixelX))))
    let y0 = min(height - 1, max(0, Int(floor(pixelY))))
    let x1 = min(width - 1, x0 + 1)
    let y1 = min(height - 1, y0 + 1)
    let tx = min(1, max(0, pixelX - Float(x0)))
    let ty = min(1, max(0, pixelY - Float(y0)))
    func channel(_ channel: Int) -> Float {
      let topLeft = image[(y0 * width + x0) * 3 + channel]
      let topRight = image[(y0 * width + x1) * 3 + channel]
      let bottomLeft = image[(y1 * width + x0) * 3 + channel]
      let bottomRight = image[(y1 * width + x1) * 3 + channel]
      let top = topLeft * (1 - tx) + topRight * tx
      let bottom = bottomLeft * (1 - tx) + bottomRight * tx
      return top * (1 - ty) + bottom * ty
    }
    return (channel(0), channel(1), channel(2))
  }

  private static func weighted(
    _ a: Float,
    _ b: Float,
    _ c: Float,
    _ d: Float,
    _ e: Float,
    _ wa: Float,
    _ wb: Float,
    _ wc: Float,
    _ wd: Float,
    _ we: Float
  ) -> Float {
    a * wa + b * wb + c * wc + d * wd + e * we
  }

  private static func scaledColor(_ value: Float) -> Float {
    let sampled = Float(Float16(value))
    let centered = Float(Float16(sampled - 0.5))
    return Float(Float16(centered * 0.125))
  }

  private static func require(
    _ tensor: HostTensor,
    semanticName: String,
    channels: Int
  ) throws {
    let descriptor = tensor.descriptor
    guard descriptor.shape.count == 4,
      descriptor.shape[0] == 1,
      descriptor.shape[3] == channels,
      descriptor.dataType == .float32,
      descriptor.layout == .nhwc
    else {
      throw NeuralRenderingTemporalPreprocessorError.expectedFloat32NHWC(
        name: semanticName,
        channels: channels,
        shape: descriptor.shape,
        dataType: descriptor.dataType,
        layout: descriptor.layout
      )
    }
  }

  private static func requireSpatialShape(
    _ tensor: HostTensor,
    name: String,
    expected: [Int]
  ) throws {
    let actual = Array(tensor.descriptor.shape.prefix(3))
    guard actual == expected else {
      throw NeuralRenderingTemporalPreprocessorError.spatialShapeMismatch(
        name: name,
        expected: expected,
        actual: actual
      )
    }
  }

  private static func resolveTransform(
    _ transform: NeuralRenderingTextureTransform?,
    tensor: HostTensor,
    name: String,
    logicalShape: [Int]
  ) throws -> NeuralRenderingTextureTransform {
    let resourceHeight = tensor.descriptor.shape[1]
    let resourceWidth = tensor.descriptor.shape[2]
    if let transform {
      guard transform.resourceWidth == resourceWidth,
        transform.resourceHeight == resourceHeight
      else {
        throw NeuralRenderingTextureTransformError.resourceShapeMismatch(
          name: name,
          expectedWidth: transform.resourceWidth,
          expectedHeight: transform.resourceHeight,
          actualWidth: resourceWidth,
          actualHeight: resourceHeight
        )
      }
      return transform
    }
    try requireSpatialShape(tensor, name: name, expected: logicalShape)
    return try NeuralRenderingTextureTransform(
      baseX: 0,
      baseY: 0,
      extentWidth: resourceWidth,
      extentHeight: resourceHeight,
      resourceWidth: resourceWidth,
      resourceHeight: resourceHeight
    )
  }

  private static func floatValues(in tensor: HostTensor) -> [Float] {
    tensor.bytes.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
  }
}
