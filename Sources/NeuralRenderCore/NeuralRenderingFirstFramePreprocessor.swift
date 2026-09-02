import Foundation

public enum NeuralRenderingPreprocessorError: Error, Equatable, Sendable {
  case expectedFloat32NHWCRGB(
    shape: [Int],
    dataType: TensorDataType,
    layout: TensorLayout
  )
}

public struct NeuralRenderingAutomaticMaskConfiguration: Equatable, Sendable {
  public let skinStructureStrength: Float
  public let automaticMaskStructureStrength: Float

  public init(
    skinStructureStrength: Float,
    automaticMaskStructureStrength: Float
  ) {
    self.skinStructureStrength = skinStructureStrength
    self.automaticMaskStructureStrength = automaticMaskStructureStrength
  }
}

/// Create-time model selection exposed by feature-18 hosts.
///
/// The official UI presents choices labeled Model A, B, and C. Public hosts
/// expose numeric slots 1 through 3 plus a shipping-default slot 0, but neither
/// the ordering nor whether the choices swap full weights, adapters, or graph
/// conditioning is proven. The current v19 package captures only slot-0
/// behavior.
public enum NeuralRenderingModelSelection: Int, CaseIterable, Sendable {
  case shippingDefault = 0
  case slot1 = 1
  case slot2 = 2
  case slot3 = 3
}

/// User-facing live-control bundles within one selected model checkpoint.
public enum NeuralRenderingControlProfile: String, CaseIterable, Sendable {
  /// Community "Default" / "Standard": the strongest nominal profile.
  case standard
  /// Gentler style 1 profile used as the default by still-image hosts.
  case natural
  /// Film-oriented style 2 profile.
  case cinematic
  /// Diagnostic profile with tone and structure disabled.
  case neutral

  public var checkpointModelSelection: NeuralRenderingModelSelection {
    .shippingDefault
  }

  public var styleIndex: Int {
    switch self {
    case .standard, .neutral: 0
    case .natural: 1
    case .cinematic: 2
    }
  }

  public var localToneStrength: Float {
    self == .neutral ? 0 : 1
  }

  public var localStructureStrength: Float {
    self == .neutral ? 0 : 1
  }

  public var skinStructureStrength: Float { -1 }
  public var automaticMaskEnabled: Bool { false }
  public var intensity: Float { 1 }

  public var featureControls: NeuralRenderingFeatureControls {
    NeuralRenderingFeatureControls(
      normalizedStyle: Float(styleIndex) / 128,
      localToneStrength: localToneStrength,
      localStructureStrength: localStructureStrength
    )
  }
}

public struct NeuralRenderingFeatureControls: Equatable, Sendable {
  public let normalizedStyle: Float
  public let localToneStrength: Float
  public let localStructureStrength: Float
  public let automaticMask: NeuralRenderingAutomaticMaskConfiguration?

  public init(
    normalizedStyle: Float = 0,
    localToneStrength: Float = 1,
    localStructureStrength: Float = 1,
    automaticMask: NeuralRenderingAutomaticMaskConfiguration? = nil
  ) {
    self.normalizedStyle = normalizedStyle
    self.localToneStrength = localToneStrength
    self.localStructureStrength = localStructureStrength
    self.automaticMask = automaticMask
  }
}

/// CPU reference for the recovered first-frame feature contract.
///
/// This covers the branch where temporal history is absent. Automatic-mask
/// strengths and a logical RGB ControlMask are optional; callers can use
/// `NeuralRenderingTextureTransform.pointSample` for recovered non-full current
/// and control resources. Motion, depth, and history reprojection belong to the
/// temporal preprocessor.
public enum NeuralRenderingFirstFramePreprocessor {
  public static func makeFeatureTensor(
    from color: HostTensor,
    noiseFrameIndex: UInt32 = 0,
    geometry requestedGeometry: NeuralRenderingNetworkGeometry? = nil,
    normalizedStyle: Float = 0,
    localToneStrength: Float = 1,
    localStructureStrength: Float = 1,
    automaticMask: NeuralRenderingAutomaticMaskConfiguration? = nil,
    controlMask: HostTensor? = nil
  ) throws -> HostTensor {
    let descriptor = color.descriptor
    guard descriptor.shape.count == 4,
      descriptor.shape[0] == 1,
      descriptor.shape[3] == 3,
      descriptor.dataType == .float32,
      descriptor.layout == .nhwc
    else {
      throw NeuralRenderingPreprocessorError.expectedFloat32NHWCRGB(
        shape: descriptor.shape,
        dataType: descriptor.dataType,
        layout: descriptor.layout
      )
    }
    if let controlMask {
      let maskDescriptor = controlMask.descriptor
      guard maskDescriptor.shape.count == 4,
        maskDescriptor.shape[0] == 1,
        maskDescriptor.shape[3] == 3,
        maskDescriptor.dataType == .float32,
        maskDescriptor.layout == .nhwc
      else {
        throw NeuralRenderingPreprocessorError.expectedFloat32NHWCRGB(
          shape: maskDescriptor.shape,
          dataType: maskDescriptor.dataType,
          layout: maskDescriptor.layout
        )
      }
      guard
        maskDescriptor.shape.prefix(3).elementsEqual(
          descriptor.shape.prefix(3)
        )
      else {
        throw NeuralRenderingPreprocessorError.expectedFloat32NHWCRGB(
          shape: maskDescriptor.shape,
          dataType: maskDescriptor.dataType,
          layout: maskDescriptor.layout
        )
      }
    }

    let outputHeight = descriptor.shape[1]
    let outputWidth = descriptor.shape[2]
    let geometry =
      try requestedGeometry
      ?? NeuralRenderingNetworkGeometry(
        outputWidth: outputWidth,
        outputHeight: outputHeight,
        networkWidth: outputWidth,
        networkHeight: outputHeight
      )
    guard geometry.outputWidth == outputWidth,
      geometry.outputHeight == outputHeight
    else {
      throw NeuralRenderingNetworkGeometryError.invalidOutput(
        width: geometry.outputWidth,
        height: geometry.outputHeight
      )
    }
    let height = geometry.networkHeight
    let width = geometry.networkWidth
    let colorValues = color.bytes.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
    let controlMaskValues = controlMask?.bytes.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
    let style = halfRounded(normalizedStyle)
    let tone = halfRounded(localToneStrength)
    let structure: Float
    let skinStructure: Float
    let automaticMaskStructure: Float
    if controlMask != nil {
      structure = 0
      skinStructure = 0
      automaticMaskStructure = 0
    } else if let automaticMask {
      let hasEnabledStrength =
        max(
          automaticMask.skinStructureStrength,
          automaticMask.automaticMaskStructureStrength
        ) >= 0
      structure = halfRounded(hasEnabledStrength ? 1 : localStructureStrength)
      skinStructure = halfRounded(
        hasEnabledStrength
          ? (automaticMask.skinStructureStrength >= 0
            ? automaticMask.skinStructureStrength
            : localStructureStrength)
          : -1
      )
      automaticMaskStructure = halfRounded(
        hasEnabledStrength
          ? (automaticMask.automaticMaskStructureStrength >= 0
            ? automaticMask.automaticMaskStructureStrength
            : localStructureStrength)
          : -1
      )
    } else {
      structure = halfRounded(localStructureStrength)
      skinStructure = -1
      automaticMaskStructure = -1
    }
    var features = [Float](repeating: 0, count: height * width * 16)

    for y in 0..<height {
      for x in 0..<width {
        let source = geometry.sourceCoordinate(x: x, y: y)
        let colorOffset = (source.y * outputWidth + source.x) * 3
        let red = scaledColor(colorValues[colorOffset])
        let green = scaledColor(colorValues[colorOffset + 1])
        let blue = scaledColor(colorValues[colorOffset + 2])
        let noise = gaussianFeatures(
          x: UInt32(truncatingIfNeeded: x),
          y: UInt32(truncatingIfNeeded: y),
          frameIndex: noiseFrameIndex
        )

        let featureOffset = (y * width + x) * 16
        let maskOffset = (source.y * outputWidth + source.x) * 3
        let pixelTone =
          controlMaskValues.map {
            halfRounded($0[maskOffset + 1] * localToneStrength)
          } ?? tone
        let pixelStructure =
          controlMaskValues.map {
            halfRounded($0[maskOffset + 2] * localStructureStrength)
          } ?? structure
        features[featureOffset] = noise.0
        features[featureOffset + 1] = noise.1
        features[featureOffset + 2] = noise.2
        features[featureOffset + 3] = 1
        features[featureOffset + 4] = red
        features[featureOffset + 5] = green
        features[featureOffset + 6] = blue
        features[featureOffset + 7] = red
        features[featureOffset + 8] = green
        features[featureOffset + 9] = blue
        features[featureOffset + 10] = style
        features[featureOffset + 11] = pixelTone
        features[featureOffset + 12] = pixelStructure
        features[featureOffset + 13] = skinStructure
        features[featureOffset + 14] = automaticMaskStructure
      }
    }

    let outputDescriptor = try TensorDescriptor(
      name: "color",
      shape: [1, height, width, 16],
      dataType: .float32,
      layout: .nhwc
    )
    return try HostTensor(
      descriptor: outputDescriptor,
      bytes: features.withUnsafeBytes { Data($0) }
    )
  }

  private static func scaledColor(_ value: Float) -> Float {
    let sampled = halfRounded(value)
    let centered = halfRounded(sampled - 0.5)
    return halfRounded(centered * 0.125)
  }

  /// The three recovered half-rounded Gaussian noise features for one network
  /// pixel and noise frame index.
  public static func deterministicNoise(
    x: Int,
    y: Int,
    frameIndex: UInt32
  ) -> (Float, Float, Float) {
    gaussianFeatures(
      x: UInt32(truncatingIfNeeded: x),
      y: UInt32(truncatingIfNeeded: y),
      frameIndex: frameIndex
    )
  }

  private static func gaussianFeatures(
    x: UInt32,
    y: UInt32,
    frameIndex: UInt32
  ) -> (Float, Float, Float) {
    var seed = y &* 0xd816_3841
    seed ^= x &* 0x8da6_b343
    seed ^= frameIndex &* 0x9e37_79b9
    seed ^= 0x243f_6a88
    let mixed = finalizedMix(seed)

    let radiusAUniform = uniform24(
      mixed &* 0xcaa5_b80d &+ 0x21dd_796b
    )
    let angleBUniform = uniform24(
      mixed &* 0x8323_2c31 &+ 0x3463_e0ac
    )
    let radiusBUniform = uniform24(
      mixed &* 0x2c92_77b5 &+ 0xac56_4b05
    )
    let angleAUniform = uniform24(
      mixed &* 0xfa6d_c5f9 &+ 0x4712_a88e
    )

    let radiusA = sqrt(-2 * log(radiusAUniform))
    let radiusB = sqrt(-2 * log(radiusBUniform))
    let tau = Float(6.283_185_482_025_146_5)
    let angleA = tau * angleAUniform
    let angleB = tau * angleBUniform
    return (
      halfRounded(radiusB * cos(angleA)),
      halfRounded(radiusB * sin(angleA)),
      halfRounded(radiusA * cos(angleB))
    )
  }

  private static func finalizedMix(_ input: UInt32) -> UInt32 {
    let multiplied = dynamicShiftMix(input)
    return multiplied ^ (multiplied >> 22)
  }

  private static func uniform24(_ input: UInt32) -> Float {
    let multiplied = dynamicShiftMix(input)
    let bits = (multiplied >> 30) ^ (multiplied >> 8)
    return Float(bits &+ 1) * 5.960_464_477_539_063e-8
  }

  /// PCG `RXS-M-XS` output step: the shift is `(input >> 28) + 4`, never
  /// zero. A zero shift would fold `input ^ input` to `0` for one pixel in
  /// sixteen and inflate the Box–Muller radius; the vendor pooled-noise capture
  /// matches this form at correlation `0.9997` on channels 0 and 1.
  private static func dynamicShiftMix(_ input: UInt32) -> UInt32 {
    let shift = Int(input >> 28) + 4
    return (input ^ (input >> shift)) &* 0x108e_f2d9
  }

  private static func halfRounded(_ value: Float) -> Float {
    Float(Float16(value))
  }
}
