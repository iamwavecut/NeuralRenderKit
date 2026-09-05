import Foundation

public enum NeuralRenderingDisplayCodecError: Error, Equatable, Sendable {
  case expectedFloat32NHWCRGB(
    shape: [Int],
    dataType: TensorDataType,
    layout: TensorLayout
  )
  case spatialShapeMismatch(expected: [Int], actual: [Int])
  case invalidConfiguration
}

public struct NeuralRenderingDisplayCodecConfiguration: Equatable, Sendable {
  public let whitePoint: Float
  public let transferStrength: Float
  public let colorStrength: Float
  public let maximumLuminanceRatio: Float
  public let inputIsDisplayReferred: Bool

  public init(
    whitePoint: Float = 1,
    transferStrength: Float = 1,
    colorStrength: Float = 1,
    maximumLuminanceRatio: Float = 2,
    inputIsDisplayReferred: Bool = false
  ) {
    self.whitePoint = whitePoint
    self.transferStrength = transferStrength
    self.colorStrength = colorStrength
    self.maximumLuminanceRatio = maximumLuminanceRatio
    self.inputIsDisplayReferred = inputIsDisplayReferred
  }
}

/// Portable reference for the display codec surrounding the recovered model.
///
/// The model sees an sRGB proxy rather than open-ended scene-linear HDR. Its
/// complete output is then rescaled onto the untouched original; no inverse
/// tone curve is used. The luminance-ratio composition and OkLab hue correction
/// are based on the MIT-licensed RenoDX design credited in `NOTICE`.
public enum NeuralRenderingDisplayCodec {
  public static func encode(
    _ original: HostTensor,
    configuration: NeuralRenderingDisplayCodecConfiguration = .init()
  ) throws -> HostTensor {
    try requireRGB(original)
    try validate(configuration)
    if configuration.inputIsDisplayReferred {
      return original
    }

    let source = values(in: original)
    var proxy = [Float]()
    proxy.reserveCapacity(source.count)
    for offset in stride(from: 0, to: source.count, by: 3) {
      let frame = nonnegative(
        SIMD3(source[offset], source[offset + 1], source[offset + 2])
      )
      var display = frame / configuration.whitePoint
      let displayLuminance = luminance(display)
      if displayLuminance > 0.75 {
        let rolled =
          0.75
          + 0.25 * (1 - exp(-(displayLuminance - 0.75) / 0.25))
        display *= rolled / displayLuminance
      }
      let encoded = linearToSRGB(display)
      proxy.append(encoded.x)
      proxy.append(encoded.y)
      proxy.append(encoded.z)
    }
    return try tensor(like: original, values: proxy)
  }

  public static func resolve(
    proxy: HostTensor,
    model: HostTensor,
    original: HostTensor,
    configuration: NeuralRenderingDisplayCodecConfiguration = .init()
  ) throws -> HostTensor {
    try requireRGB(proxy)
    try requireRGB(model)
    try requireRGB(original)
    try validate(configuration)
    for candidate in [proxy, model]
    where
      candidate.descriptor.shape != original.descriptor.shape
    {
      throw NeuralRenderingDisplayCodecError.spatialShapeMismatch(
        expected: original.descriptor.shape,
        actual: candidate.descriptor.shape
      )
    }
    if configuration.transferStrength == 0 {
      return original
    }

    let proxyValues = values(in: proxy)
    let modelValues = values(in: model)
    let originalValues = values(in: original)
    let normalization =
      configuration.inputIsDisplayReferred
      ? 1
      : configuration.whitePoint
    var output = [Float]()
    output.reserveCapacity(originalValues.count)

    for offset in stride(from: 0, to: originalValues.count, by: 3) {
      let proxyPixel = decodeIfNeeded(
        SIMD3(
          proxyValues[offset], proxyValues[offset + 1], proxyValues[offset + 2]
        ),
        configuration: configuration
      )
      let modelPixel = decodeIfNeeded(
        SIMD3(
          modelValues[offset], modelValues[offset + 1], modelValues[offset + 2]
        ),
        configuration: configuration
      )
      let originalPixel =
        SIMD3(
          originalValues[offset],
          originalValues[offset + 1],
          originalValues[offset + 2]
        ) / normalization

      let modelLuminance = luminance(modelPixel)
      if modelLuminance <= 0.000_01 {
        output.append(originalValues[offset])
        output.append(originalValues[offset + 1])
        output.append(originalValues[offset + 2])
        continue
      }

      let originalLuminance = luminance(originalPixel)
      let proxyLuminance = luminance(proxyPixel)
      let ratio: Float
      if originalLuminance < proxyLuminance {
        ratio = originalLuminance / max(proxyLuminance, 0.000_001)
      } else {
        ratio = (modelLuminance + max(0, originalLuminance - proxyLuminance)) / modelLuminance
      }

      let corrected = hueCorrect(
        modelPixel * ratio,
        toward: modelPixel
      )
      let upgraded = mix(
        originalPixel,
        corrected,
        amount: configuration.transferStrength
      )
      let upgradedLuminance = luminance(upgraded)
      let luminanceRatio =
        originalLuminance > 0.000_001
        ? min(
          configuration.maximumLuminanceRatio,
          max(0, upgradedLuminance / originalLuminance)
        )
        : 1
      let result =
        mix(
          originalPixel * luminanceRatio,
          upgraded,
          amount: configuration.colorStrength
        ) * normalization
      let bounded = nonnegative(result)
      output.append(bounded.x)
      output.append(bounded.y)
      output.append(bounded.z)
    }

    return try tensor(like: original, values: output)
  }

  private static let luma = SIMD3<Float>(0.2126, 0.7152, 0.0722)

  private static func validate(
    _ configuration: NeuralRenderingDisplayCodecConfiguration
  ) throws {
    guard configuration.whitePoint.isFinite,
      configuration.whitePoint > 0,
      configuration.transferStrength.isFinite,
      configuration.colorStrength.isFinite,
      configuration.maximumLuminanceRatio.isFinite,
      configuration.maximumLuminanceRatio >= 0
    else {
      throw NeuralRenderingDisplayCodecError.invalidConfiguration
    }
  }

  private static func requireRGB(_ tensor: HostTensor) throws {
    let descriptor = tensor.descriptor
    guard descriptor.shape.count == 4,
      descriptor.shape[0] == 1,
      descriptor.shape[3] == 3,
      descriptor.dataType == .float32,
      descriptor.layout == .nhwc
    else {
      throw NeuralRenderingDisplayCodecError.expectedFloat32NHWCRGB(
        shape: descriptor.shape,
        dataType: descriptor.dataType,
        layout: descriptor.layout
      )
    }
  }

  private static func values(in tensor: HostTensor) -> [Float] {
    tensor.bytes.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
  }

  private static func tensor(
    like reference: HostTensor,
    values: [Float]
  ) throws -> HostTensor {
    try HostTensor(
      descriptor: reference.descriptor,
      bytes: values.withUnsafeBytes { Data($0) }
    )
  }

  private static func decodeIfNeeded(
    _ value: SIMD3<Float>,
    configuration: NeuralRenderingDisplayCodecConfiguration
  ) -> SIMD3<Float> {
    configuration.inputIsDisplayReferred ? value : sRGBToLinear(value)
  }

  private static func luminance(_ value: SIMD3<Float>) -> Float {
    value.x * luma.x + value.y * luma.y + value.z * luma.z
  }

  private static func linearToSRGB(_ value: SIMD3<Float>) -> SIMD3<Float> {
    SIMD3(
      linearToSRGB(value.x),
      linearToSRGB(value.y),
      linearToSRGB(value.z)
    )
  }

  private static func linearToSRGB(_ value: Float) -> Float {
    let value = min(1, max(0, value))
    return value < 0.003_130_8
      ? value * 12.92
      : 1.055 * pow(max(value, 0.000_000_01), 1 / 2.4) - 0.055
  }

  private static func sRGBToLinear(_ value: SIMD3<Float>) -> SIMD3<Float> {
    SIMD3(
      sRGBToLinear(value.x),
      sRGBToLinear(value.y),
      sRGBToLinear(value.z)
    )
  }

  private static func sRGBToLinear(_ value: Float) -> Float {
    let value = min(1, max(0, value))
    return value < 0.040_45
      ? value / 12.92
      : pow((value + 0.055) / 1.055, 2.4)
  }

  private static func hueCorrect(
    _ incorrect: SIMD3<Float>,
    toward correct: SIMD3<Float>
  ) -> SIMD3<Float> {
    var incorrectLab = toOkLab(incorrect)
    let correctLab = toOkLab(correct)
    let incorrectChroma = sqrt(
      incorrectLab.y * incorrectLab.y + incorrectLab.z * incorrectLab.z
    )
    let correctChroma = sqrt(
      correctLab.y * correctLab.y + correctLab.z * correctLab.z
    )
    let scale = correctChroma == 0 ? 1 : incorrectChroma / correctChroma
    incorrectLab.y = correctLab.y * scale
    incorrectLab.z = correctLab.z * scale
    return clampAP1(fromOkLab(incorrectLab))
  }

  private static func toOkLab(_ value: SIMD3<Float>) -> SIMD3<Float> {
    let lms = multiply(
      (
        SIMD3(0.412_221_46, 0.536_332_55, 0.051_445_995),
        SIMD3(0.211_903_5, 0.680_699_5, 0.107_396_96),
        SIMD3(0.088_302_46, 0.281_718_85, 0.629_978_7)
      ),
      value
    )
    return multiply(
      (
        SIMD3(0.210_454_26, 0.793_617_8, -0.004_072_047),
        SIMD3(1.977_998_5, -2.428_592_2, 0.450_593_7),
        SIMD3(0.025_904_037, 0.782_771_77, -0.808_675_77)
      ),
      SIMD3(
        signedCubeRoot(lms.x),
        signedCubeRoot(lms.y),
        signedCubeRoot(lms.z)
      )
    )
  }

  private static func fromOkLab(_ value: SIMD3<Float>) -> SIMD3<Float> {
    let lms = multiply(
      (
        SIMD3(1, 0.396_337_78, 0.215_803_76),
        SIMD3(1, -0.105_561_346, -0.063_854_17),
        SIMD3(1, -0.089_484_18, -1.291_485_5)
      ),
      value
    )
    return multiply(
      (
        SIMD3(4.076_741_7, -3.307_711_6, 0.230_969_94),
        SIMD3(-1.268_438, 2.609_757_4, -0.341_319_4),
        SIMD3(-0.004_196_086_3, -0.703_418_6, 1.707_614_7)
      ),
      lms * lms * lms
    )
  }

  private static func clampAP1(_ value: SIMD3<Float>) -> SIMD3<Float> {
    let ap1 = nonnegative(
      multiply(
        (
          SIMD3(0.613_097, 0.339_523, 0.047_379),
          SIMD3(0.070_194, 0.916_354, 0.013_452),
          SIMD3(0.020_616, 0.109_57, 0.869_815)
        ),
        value
      )
    )
    return multiply(
      (
        SIMD3(1.705_051, -0.621_792, -0.083_259),
        SIMD3(-0.130_256, 1.140_805, -0.010_548),
        SIMD3(-0.024_003, -0.128_969, 1.152_972)
      ),
      ap1
    )
  }

  private static func multiply(
    _ matrix: (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>),
    _ value: SIMD3<Float>
  ) -> SIMD3<Float> {
    SIMD3(
      matrix.0.x * value.x + matrix.0.y * value.y + matrix.0.z * value.z,
      matrix.1.x * value.x + matrix.1.y * value.y + matrix.1.z * value.z,
      matrix.2.x * value.x + matrix.2.y * value.y + matrix.2.z * value.z
    )
  }

  private static func signedCubeRoot(_ value: Float) -> Float {
    value < 0 ? -pow(-value, 1 / 3) : pow(value, 1 / 3)
  }

  private static func mix(
    _ start: SIMD3<Float>,
    _ end: SIMD3<Float>,
    amount: Float
  ) -> SIMD3<Float> {
    start + (end - start) * amount
  }

  private static func nonnegative(_ value: SIMD3<Float>) -> SIMD3<Float> {
    SIMD3(max(0, value.x), max(0, value.y), max(0, value.z))
  }
}
