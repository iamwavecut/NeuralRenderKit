import MLX
import DLSSCore

/// Device-resident MLX/Metal implementation of `NeuralRenderingDisplayCodec`.
public final class MLXNeuralRenderingDisplayCodec: @unchecked Sendable {
  private let encodeKernel = MLXFast.metalKernel(
    name: "mlxdlss_display_codec_encode",
    inputNames: ["original"],
    outputNames: ["proxy"],
    source: #"""
      uint pixel = thread_position_in_grid.x;
      uint pixelCount = uint(original_shape[1] * original_shape[2]);
      if (pixel >= pixelCount) {
        return;
      }
      uint offset = pixel * 3;
      float whitePoint = as_type<float>(uint(whitePointBits));
      float3 frame = max(
        float3(original[offset], original[offset + 1], original[offset + 2]),
        float3(0.0f)
      );
      float3 display = frame / whitePoint;
      float displayLuminance = mlxdlssDisplayLuminance(display);
      if (displayLuminance > 0.75f) {
        float rolled = 0.75f + 0.25f * (
          1.0f - metal::precise::exp(-(displayLuminance - 0.75f) / 0.25f)
        );
        display *= rolled / displayLuminance;
      }
      float3 encoded = mlxdlssLinearToSRGB(display);
      proxy[offset] = encoded.x;
      proxy[offset + 1] = encoded.y;
      proxy[offset + 2] = encoded.z;
      """#,
    header: neuralRenderingDisplayCodecMetalHeader
  )

  private let resolveKernel = MLXFast.metalKernel(
    name: "mlxdlss_display_codec_resolve",
    inputNames: ["proxy", "model", "original"],
    outputNames: ["output"],
    source: #"""
      uint pixel = thread_position_in_grid.x;
      uint pixelCount = uint(original_shape[1] * original_shape[2]);
      if (pixel >= pixelCount) {
        return;
      }
      uint offset = pixel * 3;
      float whitePoint = as_type<float>(uint(whitePointBits));
      float transferStrength = as_type<float>(uint(transferStrengthBits));
      float colorStrength = as_type<float>(uint(colorStrengthBits));
      float maximumRatio = as_type<float>(uint(maximumRatioBits));
      float normalization = inputIsDisplayReferred ? 1.0f : whitePoint;

      float3 proxyPixel = float3(
        proxy[offset], proxy[offset + 1], proxy[offset + 2]
      );
      float3 modelPixel = float3(
        model[offset], model[offset + 1], model[offset + 2]
      );
      if (!inputIsDisplayReferred) {
        proxyPixel = mlxdlssSRGBToLinear(proxyPixel);
        modelPixel = mlxdlssSRGBToLinear(modelPixel);
      }
      float3 originalPixel = float3(
        original[offset], original[offset + 1], original[offset + 2]
      ) / normalization;

      float modelLuminance = mlxdlssDisplayLuminance(modelPixel);
      if (modelLuminance <= 0.00001f) {
        output[offset] = original[offset];
        output[offset + 1] = original[offset + 1];
        output[offset + 2] = original[offset + 2];
        return;
      }

      float originalLuminance = mlxdlssDisplayLuminance(originalPixel);
      float proxyLuminance = mlxdlssDisplayLuminance(proxyPixel);
      float ratio;
      if (originalLuminance < proxyLuminance) {
        ratio = originalLuminance / max(proxyLuminance, 0.000001f);
      } else {
        ratio = (
          modelLuminance + max(0.0f, originalLuminance - proxyLuminance)
        ) / modelLuminance;
      }

      float3 corrected = mlxdlssHueCorrect(modelPixel * ratio, modelPixel);
      float3 upgraded = mix(originalPixel, corrected, transferStrength);
      float upgradedLuminance = mlxdlssDisplayLuminance(upgraded);
      float luminanceRatio = originalLuminance > 0.000001f
        ? clamp(upgradedLuminance / originalLuminance, 0.0f, maximumRatio)
        : 1.0f;
      float3 result = mix(
        originalPixel * luminanceRatio,
        upgraded,
        colorStrength
      ) * normalization;
      result = max(result, float3(0.0f));
      output[offset] = result.x;
      output[offset + 1] = result.y;
      output[offset + 2] = result.z;
      """#,
    header: neuralRenderingDisplayCodecMetalHeader
  )

  public init() {}

  public func encode(
    _ original: MLXArray,
    configuration: NeuralRenderingDisplayCodecConfiguration = .init()
  ) -> MLXArray {
    validate(original)
    validate(configuration)
    if configuration.inputIsDisplayReferred {
      return original
    }
    return encodeKernel(
      [original],
      template: [
        ("whitePointBits", Int(configuration.whitePoint.bitPattern))
      ],
      grid: (original.shape[1] * original.shape[2], 1, 1),
      threadGroup: (256, 1, 1),
      outputShapes: [original.shape],
      outputDTypes: [.float32]
    )[0]
  }

  public func resolve(
    proxy: MLXArray,
    model: MLXArray,
    original: MLXArray,
    configuration: NeuralRenderingDisplayCodecConfiguration = .init()
  ) -> MLXArray {
    validate(proxy)
    validate(model)
    validate(original)
    validate(configuration)
    precondition(proxy.shape == original.shape && model.shape == original.shape)
    if configuration.transferStrength == 0 {
      return original
    }
    return resolveKernel(
      [proxy, model, original],
      template: [
        ("whitePointBits", Int(configuration.whitePoint.bitPattern)),
        ("transferStrengthBits", Int(configuration.transferStrength.bitPattern)),
        ("colorStrengthBits", Int(configuration.colorStrength.bitPattern)),
        ("maximumRatioBits", Int(configuration.maximumLuminanceRatio.bitPattern)),
        ("inputIsDisplayReferred", configuration.inputIsDisplayReferred),
      ],
      grid: (original.shape[1] * original.shape[2], 1, 1),
      threadGroup: (256, 1, 1),
      outputShapes: [original.shape],
      outputDTypes: [.float32]
    )[0]
  }

  private func validate(_ value: MLXArray) {
    precondition(
      value.shape.count == 4 && value.shape[0] == 1 && value.shape[3] == 3
        && value.dtype == .float32
    )
  }

  private func validate(
    _ configuration: NeuralRenderingDisplayCodecConfiguration
  ) {
    precondition(
      configuration.whitePoint.isFinite && configuration.whitePoint > 0
        && configuration.transferStrength.isFinite
        && configuration.colorStrength.isFinite
        && configuration.maximumLuminanceRatio.isFinite
        && configuration.maximumLuminanceRatio >= 0
    )
  }
}

private let neuralRenderingDisplayCodecMetalHeader = #"""
  using namespace metal;
  #pragma clang fp contract(off)
  #pragma clang fp reassociate(off)

  METAL_FUNC float mlxdlssDisplayLuminance(float3 value) {
    return value.x * 0.2126f + value.y * 0.7152f + value.z * 0.0722f;
  }

  METAL_FUNC float mlxdlssLinearToSRGBComponent(float value) {
    value = clamp(value, 0.0f, 1.0f);
    return value < 0.0031308f
      ? value * 12.92f
      : 1.055f * metal::precise::pow(max(value, 0.00000001f), 1.0f / 2.4f)
        - 0.055f;
  }

  METAL_FUNC float3 mlxdlssLinearToSRGB(float3 value) {
    return float3(
      mlxdlssLinearToSRGBComponent(value.x),
      mlxdlssLinearToSRGBComponent(value.y),
      mlxdlssLinearToSRGBComponent(value.z)
    );
  }

  METAL_FUNC float mlxdlssSRGBToLinearComponent(float value) {
    value = clamp(value, 0.0f, 1.0f);
    return value < 0.04045f
      ? value / 12.92f
      : metal::precise::pow((value + 0.055f) / 1.055f, 2.4f);
  }

  METAL_FUNC float3 mlxdlssSRGBToLinear(float3 value) {
    return float3(
      mlxdlssSRGBToLinearComponent(value.x),
      mlxdlssSRGBToLinearComponent(value.y),
      mlxdlssSRGBToLinearComponent(value.z)
    );
  }

  METAL_FUNC float mlxdlssSignedCubeRoot(float value) {
    return value < 0.0f
      ? -metal::precise::pow(-value, 1.0f / 3.0f)
      : metal::precise::pow(value, 1.0f / 3.0f);
  }

  METAL_FUNC float3 mlxdlssToOkLab(float3 value) {
    float3 lms = float3(
      0.41222146f * value.x + 0.53633255f * value.y + 0.051445995f * value.z,
      0.2119035f * value.x + 0.6806995f * value.y + 0.10739696f * value.z,
      0.08830246f * value.x + 0.28171885f * value.y + 0.6299787f * value.z
    );
    lms = float3(
      mlxdlssSignedCubeRoot(lms.x),
      mlxdlssSignedCubeRoot(lms.y),
      mlxdlssSignedCubeRoot(lms.z)
    );
    return float3(
      0.21045426f * lms.x + 0.7936178f * lms.y - 0.004072047f * lms.z,
      1.9779985f * lms.x - 2.4285922f * lms.y + 0.4505937f * lms.z,
      0.025904037f * lms.x + 0.78277177f * lms.y - 0.80867577f * lms.z
    );
  }

  METAL_FUNC float3 mlxdlssFromOkLab(float3 value) {
    float3 lms = float3(
      value.x + 0.39633778f * value.y + 0.21580376f * value.z,
      value.x - 0.105561346f * value.y - 0.06385417f * value.z,
      value.x - 0.08948418f * value.y - 1.2914855f * value.z
    );
    lms = lms * lms * lms;
    return float3(
      4.0767417f * lms.x - 3.3077116f * lms.y + 0.23096994f * lms.z,
      -1.268438f * lms.x + 2.6097574f * lms.y - 0.3413194f * lms.z,
      -0.0041960863f * lms.x - 0.7034186f * lms.y + 1.7076147f * lms.z
    );
  }

  METAL_FUNC float3 mlxdlssClampAP1(float3 value) {
    float3 ap1 = max(float3(
      0.613097f * value.x + 0.339523f * value.y + 0.047379f * value.z,
      0.070194f * value.x + 0.916354f * value.y + 0.013452f * value.z,
      0.020616f * value.x + 0.10957f * value.y + 0.869815f * value.z
    ), float3(0.0f));
    return float3(
      1.705051f * ap1.x - 0.621792f * ap1.y - 0.083259f * ap1.z,
      -0.130256f * ap1.x + 1.140805f * ap1.y - 0.010548f * ap1.z,
      -0.024003f * ap1.x - 0.128969f * ap1.y + 1.152972f * ap1.z
    );
  }

  METAL_FUNC float3 mlxdlssHueCorrect(float3 incorrect, float3 correct) {
    float3 incorrectLab = mlxdlssToOkLab(incorrect);
    float3 correctLab = mlxdlssToOkLab(correct);
    float incorrectChroma = metal::precise::sqrt(
      incorrectLab.y * incorrectLab.y + incorrectLab.z * incorrectLab.z
    );
    float correctChroma = metal::precise::sqrt(
      correctLab.y * correctLab.y + correctLab.z * correctLab.z
    );
    float scale = correctChroma == 0.0f ? 1.0f : incorrectChroma / correctChroma;
    incorrectLab.y = correctLab.y * scale;
    incorrectLab.z = correctLab.z * scale;
    return mlxdlssClampAP1(mlxdlssFromOkLab(incorrectLab));
  }
  """#
