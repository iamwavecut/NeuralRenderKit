import CoreGraphics
import Foundation
import ImageIO
import NeuralRenderCore
import NeuralRenderCoreML
import NeuralRenderMLX
import UniformTypeIdentifiers

/// Still-image processing: decode with ImageIO into float32 sRGB, run the
/// history-free pipeline, encode the result with ImageIO.
enum RenderImageCommand {
  static func run(arguments: [String]) async throws {
    let parsed = try parse(arguments: arguments)
    guard !FileManager.default.fileExists(atPath: parsed.outputURL.path) else {
      throw CLIError.destinationExists(parsed.outputURL)
    }
    let source = try ImageFile.read(parsed.inputURL)
    let width = source.descriptor.shape[2]
    let height = source.descriptor.shape[1]
    let processingWidth = Int((Float(width) * parsed.options.processingScale).rounded())
    let processingHeight = Int((Float(height) * parsed.options.processingScale).rounded())
    let geometry = try parsed.options.geometry.resolve(
      outputWidth: processingWidth,
      outputHeight: processingHeight
    )
    let backend: any NeuralRenderBackend
    let backendName: String
    switch parsed.backend {
    case .mlx:
      backend = try MLXNeuralRenderer(
        packageURL: parsed.modelURL,
        executionMode: parsed.executionMode,
        computePrecision: parsed.computePrecision
      )
      backendName = "mlx-metal"
    case .coreml:
      let coreML = try await CoreMLNeuralRenderer(
        modelURL: parsed.modelURL,
        configuration: CoreMLBackendConfiguration(computeUnits: parsed.computeUnits)
      )
      guard coreML.inputChannels == 16, coreML.outputChannels == 4,
        coreML.width == geometry.networkWidth,
        coreML.height == geometry.networkHeight
      else {
        throw CLIError.usage(
          "render-image Core ML head must be fixed 16-to-4 at the network extent "
            + "\(geometry.networkWidth)x\(geometry.networkHeight)"
        )
      }
      backend = coreML
      backendName = "coreml-\(parsed.computeUnits.rawValue)"
    }
    let rendered = try await FirstFramePipeline.render(
      source: source,
      controlMask: nil,
      options: parsed.options,
      backend: backend
    )
    try ImageFile.write(rendered.output, to: parsed.outputURL)
    var summary: [String: Any] = [
      "backend": backendName,
      "executionNanoseconds": rendered.executionNanoseconds,
      "preprocessingNanoseconds": rendered.preprocessingNanoseconds,
      "postprocessingNanoseconds": rendered.postprocessingNanoseconds,
      "input": parsed.inputURL.path,
      "output": parsed.outputURL.path,
      "shape": [height, width, 3],
      "outputShape": [height, width, 3],
      "networkShape": rendered.networkShape,
      "networkGeometry": parsed.options.geometry.rawValue,
      "profile": parsed.profile.rawValue,
      "styleIndex": parsed.styleIndex,
      "intensity": Double(parsed.options.intensity),
      "localToneStrength": Double(parsed.options.featureControls.localToneStrength),
      "localStructureStrength": Double(parsed.options.featureControls.localStructureStrength),
      "effectiveAutomaticMask": parsed.options.featureControls.automaticMask != nil,
      "processingScale": Double(parsed.options.processingScale),
      "detailStrength": Double(parsed.options.detailStrength),
      "colourStrength": Double(parsed.options.colourStrength),
      "detailRadius": Double(parsed.options.detailRadius),
      "noiseFrameIndex": Int(parsed.options.noiseFrameIndex),
    ]
    if parsed.backend == .mlx {
      summary["computePrecision"] = parsed.computePrecision.rawValue
      summary["executionMode"] = parsed.executionMode.rawValue
    }
    try CLIOutput.writeJSON(summary)
  }

  private static func parse(arguments: [String]) throws -> ParsedRenderImage {
    guard arguments.count >= 2 else {
      throw CLIError.usage(
        "render-image requires IMAGE and MODEL paths followed by --output PATH"
      )
    }
    let inputURL = URL(fileURLWithPath: arguments[0])
    let modelURL = URL(fileURLWithPath: arguments[1])
    let knownOptions = [
      "--output", "--backend", "--compute-units", "--execution", "--precision",
      "--profile", "--style-index", "--local-tone", "--local-structure",
      "--skin-structure", "--auto-mask", "--intensity", "--network-geometry",
      "--processing-scale", "--detail-strength", "--colour-strength", "--detail-radius",
      "--noise-frame-index",
    ]
    var values: [String: String] = [:]
    var index = 2
    while index < arguments.count {
      let option = arguments[index]
      guard knownOptions.contains(option) else {
        throw CLIError.usage("unknown render-image option '\(option)'")
      }
      guard index + 1 < arguments.count else {
        throw CLIError.usage("render-image option '\(option)' needs a value")
      }
      guard values[option] == nil else {
        throw CLIError.usage("duplicate render-image option '\(option)'")
      }
      values[option] = arguments[index + 1]
      index += 2
    }
    guard let outputPath = values["--output"] else {
      throw CLIError.usage("render-image requires --output PATH.png|jpg|tiff|heic")
    }
    let backend = try enumeration(
      CLIBackend.self, values["--backend"], default: .mlx,
      message: "render-image backend must be 'mlx' or 'coreml'"
    )
    let computeUnits = try enumeration(
      CoreMLComputeUnits.self, values["--compute-units"], default: .cpuAndGPU,
      message: "render-image compute units must be 'cpu', 'cpu-gpu', 'cpu-neural-engine', or 'all'"
    )
    let executionMode = try enumeration(
      MLXExecutionMode.self, values["--execution"], default: .metalFused,
      message: "render-image execution must be 'eager', 'block-compiled', 'int8-fast', or 'metal-fused'"
    )
    guard executionMode != .compiled else {
      throw CLIError.usage("render-image does not support whole-graph compilation")
    }
    let computePrecision = try enumeration(
      MLXComputePrecision.self, values["--precision"], default: .float16,
      message: "render-image precision must be 'float32' or 'float16'"
    )
    if backend == .coreml, values["--execution"] != nil || values["--precision"] != nil {
      throw CLIError.usage("render-image --execution and --precision apply only to the MLX backend")
    }
    if backend == .mlx, values["--compute-units"] != nil {
      throw CLIError.usage("render-image --compute-units applies only to the Core ML backend")
    }
    let profile = try enumeration(
      NeuralRenderingControlProfile.self, values["--profile"], default: .standard,
      message: "render-image profile must be 'standard', 'natural', 'cinematic', or 'neutral'"
    )
    let geometry = try enumeration(
      NeuralRenderingNetworkGeometryPolicy.self, values["--network-geometry"],
      default: .vendorAligned,
      message: "render-image network geometry must be 'vendor-aligned' or 'match-output'"
    )
    let styleIndex: Int
    if let text = values["--style-index"] {
      guard let parsed = UInt32(text) else {
        throw CLIError.usage("render-image style index must be an unsigned integer")
      }
      styleIndex = Int(parsed)
    } else {
      styleIndex = profile.styleIndex
    }
    let localTone = try floatOption("--local-tone", values: values, default: profile.localToneStrength)
    let localStructure = try floatOption(
      "--local-structure", values: values, default: profile.localStructureStrength
    )
    let skinStructure = try floatOption(
      "--skin-structure", values: values, default: profile.skinStructureStrength
    )
    let intensity = try floatOption("--intensity", values: values, default: profile.intensity)
    let processingScale = try floatOption("--processing-scale", values: values, default: 1)
    guard processingScale >= 1, processingScale <= 4 else {
      throw CLIError.usage("render-image processing scale must be within [1, 4]")
    }
    let detailStrength = try floatOption("--detail-strength", values: values, default: 1)
    let colourStrength = try floatOption("--colour-strength", values: values, default: 1)
    let detailRadius = try floatOption("--detail-radius", values: values, default: 4)
    guard detailStrength >= 0, detailStrength <= 8, colourStrength >= 0, colourStrength <= 4,
      detailRadius >= 0.5, detailRadius <= 64
    else {
      throw CLIError.usage(
        "render-image detail strength must be within [0, 8], colour strength within [0, 4], detail radius within [0.5, 64]"
      )
    }
    var noiseFrameIndex: UInt32 = 0
    if let text = values["--noise-frame-index"] {
      guard let parsed = UInt32(text) else {
        throw CLIError.usage("render-image noise frame index must be an unsigned integer")
      }
      noiseFrameIndex = parsed
    }
    let automaticMaskRequested: Bool
    if let text = values["--auto-mask"] {
      guard let mode = RunAutomaticMask(rawValue: text) else {
        throw CLIError.usage("render-image auto mask must be 'enabled' or 'disabled'")
      }
      automaticMaskRequested = mode == .enabled
    } else {
      automaticMaskRequested = profile.automaticMaskEnabled
    }
    let automaticMask: NeuralRenderingAutomaticMaskConfiguration? =
      automaticMaskRequested
      ? NeuralRenderingAutomaticMaskConfiguration(
        skinStructureStrength: skinStructure >= 0 ? skinStructure : localStructure,
        automaticMaskStructureStrength: localStructure
      )
      : nil
    let options = FirstFrameOptions(
      featureControls: NeuralRenderingFeatureControls(
        normalizedStyle: Float(styleIndex) / 128,
        localToneStrength: localTone,
        localStructureStrength: localStructure,
        automaticMask: automaticMask
      ),
      intensity: intensity,
      noiseFrameIndex: noiseFrameIndex,
      geometry: geometry,
      processingScale: processingScale,
      detailStrength: detailStrength,
      colourStrength: colourStrength,
      detailRadius: detailRadius
    )
    return ParsedRenderImage(
      inputURL: inputURL,
      modelURL: modelURL,
      outputURL: URL(fileURLWithPath: outputPath),
      backend: backend,
      computeUnits: computeUnits,
      executionMode: executionMode,
      computePrecision: computePrecision,
      profile: profile,
      styleIndex: styleIndex,
      options: options
    )
  }

  private static func enumeration<T: RawRepresentable>(
    _: T.Type,
    _ text: String?,
    default defaultValue: T,
    message: String
  ) throws -> T where T.RawValue == String {
    guard let text else { return defaultValue }
    guard let value = T(rawValue: text) else {
      throw CLIError.usage(message)
    }
    return value
  }

  private static func floatOption(
    _ name: String,
    values: [String: String],
    default defaultValue: Float
  ) throws -> Float {
    guard let text = values[name] else { return defaultValue }
    guard let value = Float(text), value.isFinite else {
      throw CLIError.usage("render-image \(name) must be a finite number")
    }
    return value
  }
}

private struct ParsedRenderImage {
  let inputURL: URL
  let modelURL: URL
  let outputURL: URL
  let backend: CLIBackend
  let computeUnits: CoreMLComputeUnits
  let executionMode: MLXExecutionMode
  let computePrecision: MLXComputePrecision
  let profile: NeuralRenderingControlProfile
  let styleIndex: Int
  let options: FirstFrameOptions
}

/// ImageIO decode/encode of 8-bit sRGB images as float32 NHWC RGB tensors.
enum ImageFile {
  static func read(_ url: URL) throws -> HostTensor {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
      throw CLIError.usage("cannot decode image \(url.path)")
    }
    let width = image.width
    let height = image.height
    var pixels = [UInt8](repeating: 0, count: width * height * 4)
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    let drawn = pixels.withUnsafeMutableBytes { buffer -> Bool in
      guard
        let context = CGContext(
          data: buffer.baseAddress,
          width: width,
          height: height,
          bitsPerComponent: 8,
          bytesPerRow: width * 4,
          space: colorSpace,
          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        )
      else { return false }
      context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
      return true
    }
    guard drawn else {
      throw CLIError.usage("cannot rasterize image \(url.path)")
    }
    var values = [Float](repeating: 0, count: width * height * 3)
    for pixel in 0..<(width * height) {
      values[pixel * 3] = Float(pixels[pixel * 4]) / 255
      values[pixel * 3 + 1] = Float(pixels[pixel * 4 + 1]) / 255
      values[pixel * 3 + 2] = Float(pixels[pixel * 4 + 2]) / 255
    }
    return try HostTensor(
      descriptor: TensorDescriptor(
        name: "color",
        shape: [1, height, width, 3],
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: values.withUnsafeBytes { Data($0) }
    )
  }

  static func write(_ tensor: HostTensor, to url: URL) throws {
    let shape = tensor.descriptor.shape
    guard shape.count == 4, shape[0] == 1, shape[3] == 3,
      tensor.descriptor.dataType == .float32
    else {
      throw CLIError.usage("image output expects a float32 NHWC RGB tensor")
    }
    let height = shape[1]
    let width = shape[2]
    let values = tensor.bytes.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
    var pixels = [UInt8](repeating: 255, count: width * height * 4)
    for pixel in 0..<(width * height) {
      for channel in 0..<3 {
        let value = min(max(values[pixel * 3 + channel], 0), 1)
        pixels[pixel * 4 + channel] = UInt8((value * 255).rounded())
      }
    }
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    let data = Data(pixels)
    guard let provider = CGDataProvider(data: data as CFData),
      let image = CGImage(
        width: width,
        height: height,
        bitsPerComponent: 8,
        bitsPerPixel: 32,
        bytesPerRow: width * 4,
        space: colorSpace,
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue),
        provider: provider,
        decode: nil,
        shouldInterpolate: false,
        intent: .defaultIntent
      )
    else {
      throw CLIError.usage("cannot build the output image")
    }
    let type = UTType(filenameExtension: url.pathExtension.lowercased()) ?? .png
    guard let destination = CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil) else {
      throw CLIError.usage("cannot create image \(url.path)")
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else {
      throw CLIError.usage("cannot write image \(url.path)")
    }
  }
}
