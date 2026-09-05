import Foundation
import DLSSCore
import DLSSCoreML
import DLSSMLX

enum CLIBackend: String {
  case mlx
  case coreml
}

enum RunInputFormat: String {
  case model
  case rgbFirstFrame = "rgb-first-frame"
}

enum RunAutomaticMask: String {
  case disabled
  case enabled
}

enum CLITextureTransform {
  static func parse(
    _ text: String?,
    command: String,
    label: String
  ) throws -> NeuralRenderingTextureTransform? {
    guard let text else { return nil }
    let fields = text.split(separator: ",", omittingEmptySubsequences: false)
    guard fields.count == 6,
      fields.allSatisfy({ Int($0) != nil })
    else {
      throw CLIError.usage(
        "\(command) \(label) transform must contain six integers"
      )
    }
    let integers = fields.map { Int($0)! }
    do {
      return try NeuralRenderingTextureTransform(
        baseX: integers[0],
        baseY: integers[1],
        extentWidth: integers[2],
        extentHeight: integers[3],
        resourceWidth: integers[4],
        resourceHeight: integers[5]
      )
    } catch {
      throw CLIError.usage(
        "\(command) \(label) transform is invalid: \(error)"
      )
    }
  }

  static func fields(_ transform: NeuralRenderingTextureTransform) -> [Int] {
    [
      transform.baseX,
      transform.baseY,
      transform.extentWidth,
      transform.extentHeight,
      transform.resourceWidth,
      transform.resourceHeight,
    ]
  }
}

enum RunCommand {
  static func run(arguments: [String]) async throws {
    let parsed = try parse(arguments: arguments)
    let sourceResource = try parsed.input.makeTensor()
    let sourceInput =
      try parsed.inputTransform?.pointSample(
        sourceResource,
        logicalWidth: parsed.logicalWidth,
        logicalHeight: parsed.logicalHeight
      ) ?? sourceResource
    let controlMaskResource = try parsed.controlMask?.makeTensor()
    let controlMask = try controlMaskResource.map { resource in
      try parsed.controlMaskTransform?.pointSample(
        resource,
        logicalWidth: parsed.logicalWidth,
        logicalHeight: parsed.logicalHeight
      ) ?? resource
    }
    let input: HostTensor
    let networkGeometry: NeuralRenderingNetworkGeometry?
    let preprocessingNanoseconds: UInt64
    var processingSource = sourceInput
    switch parsed.inputFormat {
    case .model:
      input = sourceInput
      networkGeometry = nil
      preprocessingNanoseconds = 0
    case .rgbFirstFrame:
      let start = DispatchTime.now().uptimeNanoseconds
      if parsed.processingScale != 1 {
        // Photoreal recipe: run the network on a resampled frame (the community
        // pipelines process at the source's native scale) and resample back after
        // composition.
        processingSource = try NeuralRenderingDetailComposition.resample(
          sourceInput,
          width: Int((Float(parsed.logicalWidth) * parsed.processingScale).rounded()),
          height: Int((Float(parsed.logicalHeight) * parsed.processingScale).rounded())
        )
      }
      let geometry = try NeuralRenderingNetworkGeometry.vendorAligned(
        outputWidth: processingSource.descriptor.shape[2],
        outputHeight: processingSource.descriptor.shape[1]
      )
      input = try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(
        from: processingSource,
        noiseFrameIndex: parsed.noiseFrameIndex,
        geometry: geometry,
        normalizedStyle: Float(parsed.styleIndex) / 128,
        localToneStrength: parsed.localToneStrength,
        localStructureStrength: parsed.localStructureStrength,
        automaticMask: parsed.automaticMask,
        controlMask: controlMask
      )
      networkGeometry = geometry
      preprocessingNanoseconds = DispatchTime.now().uptimeNanoseconds - start
    }
    if let featuresOutputURL = parsed.featuresOutputURL {
      // Diagnostic export of the exact network input tensor (16-channel NHWC float32).
      try input.bytes.write(to: featuresOutputURL, options: .atomic)
    }
    let request = try NeuralRenderRequest(sequenceID: 1, inputs: [input])
    let renderer: any NeuralRenderBackend
    let effectiveHostBridge: CoreMLHostBridge?
    switch parsed.backend {
    case .mlx:
      renderer = try MLXNeuralRenderer(
        packageURL: parsed.modelURL,
        executionMode: parsed.executionMode,
        computePrecision: parsed.computePrecision
      )
      effectiveHostBridge = nil
    case .coreml:
      let coreMLRenderer = try await CoreMLNeuralRenderer(
        modelURL: parsed.modelURL,
        configuration: CoreMLBackendConfiguration(
          computeUnits: parsed.computeUnits,
          hostBridge: parsed.hostBridge
        )
      )
      renderer = coreMLRenderer
      effectiveHostBridge = coreMLRenderer.effectiveHostBridge
    }
    let result = try await renderer.render(request)
    guard let backendOutput = result.output(named: "color") else {
      throw CLIError.missingOutput("color")
    }
    let output: HostTensor
    let postprocessingNanoseconds: UInt64
    switch parsed.inputFormat {
    case .model:
      output = backendOutput
      postprocessingNanoseconds = 0
    case .rgbFirstFrame:
      let start = DispatchTime.now().uptimeNanoseconds
      let head = try networkGeometry!.cropOutput(backendOutput)
      var composed = try NeuralRenderingFirstFramePostprocessor.compose(
        head: head,
        over: processingSource,
        controlMask: controlMask,
        intensity: parsed.intensity
      )
      if parsed.processingScale != 1 {
        composed = try NeuralRenderingDetailComposition.resample(
          composed,
          width: parsed.logicalWidth,
          height: parsed.logicalHeight
        )
      }
      output = try NeuralRenderingDetailComposition.compose(
        input: sourceInput,
        output: composed,
        detailStrength: parsed.detailStrength,
        colourStrength: parsed.colourStrength,
        radius: parsed.detailRadius
      )
      postprocessingNanoseconds = DispatchTime.now().uptimeNanoseconds - start
    }

    if let outputURL = parsed.outputURL {
      guard !FileManager.default.fileExists(atPath: outputURL.path) else {
        throw CLIError.destinationExists(outputURL)
      }
      try output.bytes.write(to: outputURL, options: .atomic)
      var summary: [String: Any] = [
        "backend": parsed.backendName(effectiveHostBridge: effectiveHostBridge),
        "elementCount": output.descriptor.elementCount,
        "executionNanoseconds": result.timing.executionNanoseconds,
        "shape": output.descriptor.shape,
      ]
      addBackendConfiguration(
        parsed,
        effectiveHostBridge: effectiveHostBridge,
        to: &summary
      )
      summary["inputFormat"] = parsed.inputFormat.rawValue
      if parsed.inputFormat == .rgbFirstFrame {
        addFirstFrameControls(parsed, to: &summary)
        summary["processingScale"] = Double(parsed.processingScale)
        summary["detailStrength"] = Double(parsed.detailStrength)
        summary["colourStrength"] = Double(parsed.colourStrength)
        summary["detailRadius"] = Double(parsed.detailRadius)
        summary["networkShape"] = input.descriptor.shape
        summary["preprocessingNanoseconds"] = preprocessingNanoseconds
        summary["postprocessingNanoseconds"] = postprocessingNanoseconds
      }
      try CLIOutput.writeJSON(summary)
    } else {
      var summary: [String: Any] = [
        "backend": parsed.backendName(effectiveHostBridge: effectiveHostBridge),
        "executionNanoseconds": result.timing.executionNanoseconds,
        "output": floatValues(in: output).map(Double.init),
      ]
      addBackendConfiguration(
        parsed,
        effectiveHostBridge: effectiveHostBridge,
        to: &summary
      )
      summary["inputFormat"] = parsed.inputFormat.rawValue
      if parsed.inputFormat == .rgbFirstFrame {
        addFirstFrameControls(parsed, to: &summary)
        summary["processingScale"] = Double(parsed.processingScale)
        summary["detailStrength"] = Double(parsed.detailStrength)
        summary["colourStrength"] = Double(parsed.colourStrength)
        summary["detailRadius"] = Double(parsed.detailRadius)
        summary["networkShape"] = input.descriptor.shape
        summary["preprocessingNanoseconds"] = preprocessingNanoseconds
        summary["postprocessingNanoseconds"] = postprocessingNanoseconds
      }
      try CLIOutput.writeJSON(summary)
    }
  }

  private static func parse(arguments: [String]) throws -> ParsedRun {
    guard let packagePath = arguments.first else {
      throw CLIError.usage("run requires a model-package path")
    }
    if arguments.count == 1 {
      return ParsedRun(
        modelURL: URL(fileURLWithPath: packagePath),
        input: try .literalDemo(),
        outputURL: nil,
        backend: .mlx,
        executionMode: .eager,
        computePrecision: .float32,
        computeUnits: .cpuAndGPU,
        hostBridge: .automatic,
        inputFormat: .model,
        controlProfile: .standard,
        styleIndex: 0,
        localToneStrength: 1,
        localStructureStrength: 1,
        automaticMask: nil,
        controlMask: nil,
        inputTransform: nil,
        controlMaskTransform: nil,
        logicalWidth: 2,
        logicalHeight: 1,
        intensity: 1,
        featuresOutputURL: nil,
        noiseFrameIndex: 0,
        processingScale: 1,
        detailStrength: 1,
        colourStrength: 1,
        detailRadius: 4
      )
    }

    var values: [String: String] = [:]
    var index = 1
    while index < arguments.count {
      guard index + 1 < arguments.count else {
        throw CLIError.usage("run option '\(arguments[index])' needs a value")
      }
      let option = arguments[index]
      guard
        [
          "--input", "--output", "--height", "--width", "--execution", "--precision",
          "--backend",
          "--compute-units",
          "--coreml-bridge",
          "--input-format",
          "--channels",
          "--profile",
          "--style-index",
          "--local-tone",
          "--local-structure",
          "--auto-mask",
          "--skin-structure",
          "--control-mask",
          "--input-transform",
          "--control-mask-transform",
          "--intensity",
          "--features-output",
          "--noise-frame-index",
          "--processing-scale",
          "--detail-strength",
          "--colour-strength",
          "--detail-radius",
        ]
        .contains(option)
      else {
        throw CLIError.usage("unknown run option '\(option)'")
      }
      guard values[option] == nil else {
        throw CLIError.usage("duplicate run option '\(option)'")
      }
      values[option] = arguments[index + 1]
      index += 2
    }
    let backend: CLIBackend
    if let value = values["--backend"] {
      guard let parsedBackend = CLIBackend(rawValue: value) else {
        throw CLIError.usage("run backend must be 'mlx' or 'coreml'")
      }
      backend = parsedBackend
    } else {
      backend = .mlx
    }
    let inputFormat: RunInputFormat
    if let value = values["--input-format"] {
      guard let parsedInputFormat = RunInputFormat(rawValue: value) else {
        throw CLIError.usage(
          "run input format must be 'model' or 'rgb-first-frame'"
        )
      }
      inputFormat = parsedInputFormat
    } else {
      inputFormat = .model
    }
    let firstFrameOptions = [
      "--profile", "--style-index", "--local-tone", "--local-structure", "--auto-mask",
      "--skin-structure", "--control-mask", "--input-transform",
      "--control-mask-transform", "--intensity",
      "--processing-scale", "--detail-strength", "--colour-strength", "--detail-radius",
    ]
    if inputFormat != .rgbFirstFrame,
      firstFrameOptions.contains(where: { values[$0] != nil })
    {
      throw CLIError.usage(
        "run first-frame controls require rgb-first-frame input"
      )
    }
    let inputTransform = try CLITextureTransform.parse(
      values["--input-transform"],
      command: "run",
      label: "input"
    )
    let controlMaskTransform = try CLITextureTransform.parse(
      values["--control-mask-transform"],
      command: "run",
      label: "control mask"
    )
    if controlMaskTransform != nil, values["--control-mask"] == nil {
      throw CLIError.usage(
        "run control mask transform requires --control-mask"
      )
    }
    let controlProfile: NeuralRenderingControlProfile
    if let text = values["--profile"] {
      guard let parsed = NeuralRenderingControlProfile(rawValue: text) else {
        throw CLIError.usage(
          "run profile must be 'standard', 'natural', 'cinematic', or 'neutral'"
        )
      }
      controlProfile = parsed
    } else {
      controlProfile = .standard
    }
    let styleIndex: Int
    if let text = values["--style-index"] {
      guard let parsed = UInt32(text) else {
        throw CLIError.usage("run style index must be an unsigned integer")
      }
      styleIndex = Int(parsed)
    } else {
      styleIndex = controlProfile.styleIndex
    }
    let localToneStrength = try floatOption(
      "--local-tone", values: values, default: controlProfile.localToneStrength
    )
    let localStructureStrength = try floatOption(
      "--local-structure", values: values,
      default: controlProfile.localStructureStrength
    )
    let skinStructureStrength = try floatOption(
      "--skin-structure", values: values,
      default: controlProfile.skinStructureStrength
    )
    let intensity = try floatOption(
      "--intensity", values: values, default: controlProfile.intensity
    )
    let automaticMaskRequested: Bool
    if let text = values["--auto-mask"] {
      guard let mode = RunAutomaticMask(rawValue: text) else {
        throw CLIError.usage("run auto mask must be 'enabled' or 'disabled'")
      }
      automaticMaskRequested = mode == .enabled
    } else {
      automaticMaskRequested = controlProfile.automaticMaskEnabled
    }
    let computeUnits: CoreMLComputeUnits
    if let value = values["--compute-units"] {
      guard let parsedComputeUnits = CoreMLComputeUnits(rawValue: value) else {
        throw CLIError.usage(
          "run compute units must be 'cpu', 'cpu-gpu', "
            + "'cpu-neural-engine', or 'all'"
        )
      }
      computeUnits = parsedComputeUnits
    } else {
      computeUnits = .cpuAndGPU
    }
    let hostBridge: CoreMLHostBridge
    if let value = values["--coreml-bridge"] {
      guard let parsedBridge = CoreMLHostBridge(rawValue: value) else {
        throw CLIError.usage(
          "run Core ML bridge must be 'automatic', 'multi-array', "
            + "or 'ml-tensor'"
        )
      }
      hostBridge = parsedBridge
    } else {
      hostBridge = .automatic
    }
    let executionMode: MLXExecutionMode
    if let mode = values["--execution"] {
      guard let parsedMode = MLXExecutionMode(rawValue: mode) else {
        throw CLIError.usage(
          "run execution must be 'eager', 'block-compiled', 'int8-fast', "
            + "'metal-fused', or 'compiled'"
        )
      }
      executionMode = parsedMode
    } else {
      executionMode = .eager
    }
    let computePrecision: MLXComputePrecision
    if let precision = values["--precision"] {
      guard let parsedPrecision = MLXComputePrecision(rawValue: precision) else {
        throw CLIError.usage("run precision must be 'float32' or 'float16'")
      }
      computePrecision = parsedPrecision
    } else {
      computePrecision = .float32
    }

    if backend == .coreml {
      if values["--execution"] != nil || values["--precision"] != nil {
        throw CLIError.usage(
          "run --execution and --precision apply only to the MLX backend"
        )
      }
    } else if values["--compute-units"] != nil
      || values["--coreml-bridge"] != nil
    {
      throw CLIError.usage(
        "run --compute-units and --coreml-bridge apply only to the Core ML backend"
      )
    }

    let tensorOptions = ["--input", "--output", "--height", "--width"]
    if !tensorOptions.contains(where: { values[$0] != nil }) {
      if inputFormat == .rgbFirstFrame {
        throw CLIError.usage(
          "run rgb-first-frame input requires --input, --output, "
            + "--height, and --width"
        )
      }
      guard backend == .mlx else {
        throw CLIError.usage(
          "Core ML run requires --input, --output, --height, and --width"
        )
      }
      return ParsedRun(
        modelURL: URL(fileURLWithPath: packagePath),
        input: try .literalDemo(),
        outputURL: nil,
        backend: backend,
        executionMode: executionMode,
        computePrecision: computePrecision,
        computeUnits: computeUnits,
        hostBridge: hostBridge,
        inputFormat: inputFormat,
        controlProfile: controlProfile,
        styleIndex: styleIndex,
        localToneStrength: localToneStrength,
        localStructureStrength: localStructureStrength,
        automaticMask: nil,
        controlMask: nil,
        inputTransform: nil,
        controlMaskTransform: nil,
        logicalWidth: 2,
        logicalHeight: 1,
        intensity: intensity,
      featuresOutputURL: values["--features-output"].map { URL(fileURLWithPath: $0) },
      noiseFrameIndex: values["--noise-frame-index"].flatMap { UInt32($0) } ?? 0,
      processingScale: 1,
      detailStrength: 1,
      colourStrength: 1,
      detailRadius: 4
    )
    }
    for option in tensorOptions where values[option] == nil {
      throw CLIError.usage("missing run option '\(option)'")
    }
    guard let inputPath = values["--input"],
      let outputPath = values["--output"],
      let heightText = values["--height"],
      let widthText = values["--width"]
    else {
      throw CLIError.usage("missing run tensor option")
    }
    guard let height = Int(heightText), let width = Int(widthText) else {
      throw CLIError.usage("run height and width must be integers")
    }
    let modelURL = URL(fileURLWithPath: packagePath)
    let channels: Int
    if let channelText = values["--channels"] {
      guard let parsed = Int(channelText), parsed > 0 else {
        throw CLIError.usage("run channels must be a positive integer")
      }
      channels = parsed
    } else {
      channels =
        inputFormat == .rgbFirstFrame
        ? 3
        : try inputChannels(modelURL: modelURL, backend: backend)
    }
    if inputFormat == .rgbFirstFrame, channels != 3 {
      throw CLIError.usage("run rgb-first-frame input requires three channels")
    }
    let input = try RawTensorInput(
      url: URL(fileURLWithPath: inputPath),
      height: inputTransform?.resourceHeight ?? height,
      width: inputTransform?.resourceWidth ?? width,
      channels: channels
    )
    let controlMask: RawTensorInput?
    if let path = values["--control-mask"] {
      controlMask = try RawTensorInput(
        url: URL(fileURLWithPath: path),
        height: controlMaskTransform?.resourceHeight ?? height,
        width: controlMaskTransform?.resourceWidth ?? width,
        channels: 3,
        name: "controlMask"
      )
    } else {
      controlMask = nil
    }
    let automaticMask: NeuralRenderingAutomaticMaskConfiguration?
    if automaticMaskRequested, controlMask == nil {
      automaticMask = NeuralRenderingAutomaticMaskConfiguration(
        skinStructureStrength:
          skinStructureStrength >= 0
          ? skinStructureStrength
          : localStructureStrength,
        automaticMaskStructureStrength: localStructureStrength
      )
    } else {
      automaticMask = nil
    }
    func parseStrength(_ option: String, default defaultValue: Float, minimum: Float, maximum: Float) throws -> Float {
      guard let text = values[option] else { return defaultValue }
      guard let value = Float(text), value.isFinite, value >= minimum, value <= maximum else {
        throw CLIError.usage("run \(option) must be a number in [\(minimum), \(maximum)]")
      }
      return value
    }
    let processingScale = try parseStrength("--processing-scale", default: 1, minimum: 1, maximum: 4)
    let detailStrength = try parseStrength("--detail-strength", default: 1, minimum: 0, maximum: 8)
    let colourStrength = try parseStrength("--colour-strength", default: 1, minimum: 0, maximum: 4)
    let detailRadius = try parseStrength("--detail-radius", default: 4, minimum: 0.5, maximum: 64)
    if processingScale != 1, values["--control-mask"] != nil {
      throw CLIError.usage("run processing scale does not support --control-mask")
    }
    return ParsedRun(
      modelURL: modelURL,
      input: input,
      outputURL: URL(fileURLWithPath: outputPath),
      backend: backend,
      executionMode: executionMode,
      computePrecision: computePrecision,
      computeUnits: computeUnits,
      hostBridge: hostBridge,
      inputFormat: inputFormat,
      controlProfile: controlProfile,
      styleIndex: styleIndex,
      localToneStrength: localToneStrength,
      localStructureStrength: localStructureStrength,
      automaticMask: automaticMask,
      controlMask: controlMask,
      inputTransform: inputTransform,
      controlMaskTransform: controlMaskTransform,
      logicalWidth: width,
      logicalHeight: height,
      intensity: intensity,
      featuresOutputURL: values["--features-output"].map { URL(fileURLWithPath: $0) },
      noiseFrameIndex: values["--noise-frame-index"].flatMap { UInt32($0) } ?? 0,
      processingScale: processingScale,
      detailStrength: detailStrength,
      colourStrength: colourStrength,
      detailRadius: detailRadius
    )
  }

  private static func floatOption(
    _ name: String,
    values: [String: String],
    default defaultValue: Float
  ) throws -> Float {
    guard let text = values[name] else {
      return defaultValue
    }
    guard let value = Float(text), value.isFinite else {
      throw CLIError.usage("run \(name) must be a finite number")
    }
    return value
  }

  private static func inputChannels(
    modelURL: URL,
    backend: CLIBackend
  ) throws -> Int {
    guard backend == .mlx else {
      return 3
    }
    let manifestData = try Data(
      contentsOf: modelURL.appendingPathComponent("manifest.json")
    )
    let manifest = try ModelPackageManifest.decode(data: manifestData)
    guard let color = manifest.inputs.first(where: { $0.name == "color" }),
      color.shape.count == 4,
      case .fixed(let channels) = color.shape[3]
    else {
      throw CLIError.usage(
        "run requires a fixed channel dimension for the color input"
      )
    }
    return channels
  }

  private static func addFirstFrameControls(
    _ parsed: ParsedRun,
    to summary: inout [String: Any]
  ) {
    summary["profile"] = parsed.controlProfile.rawValue
    summary["checkpointModelSelection"] =
      parsed.controlProfile.checkpointModelSelection.rawValue
    summary["checkpointModel"] = "shipping-default"
    summary["styleIndex"] = parsed.styleIndex
    summary["localToneStrength"] = Double(parsed.localToneStrength)
    summary["localStructureStrength"] = Double(parsed.localStructureStrength)
    summary["effectiveAutomaticMask"] = parsed.automaticMask != nil
    summary["controlMask"] =
      if parsed.controlMask == nil {
        "none"
      } else if parsed.controlMaskTransform == nil {
        "full-rect-rgb"
      } else {
        "point-transformed-rgb"
      }
    if let transform = parsed.inputTransform {
      summary["inputTransform"] = CLITextureTransform.fields(transform)
    }
    if let transform = parsed.controlMaskTransform {
      summary["controlMaskTransform"] = CLITextureTransform.fields(transform)
    }
    summary["intensity"] = Double(parsed.intensity)
  }

  private static func addBackendConfiguration(
    _ parsed: ParsedRun,
    effectiveHostBridge: CoreMLHostBridge?,
    to summary: inout [String: Any]
  ) {
    switch parsed.backend {
    case .mlx:
      summary["computePrecision"] = parsed.computePrecision.rawValue
      summary["executionMode"] = parsed.executionMode.rawValue
    case .coreml:
      summary["computeUnits"] = parsed.computeUnits.rawValue
      summary["hostBridge"] = effectiveHostBridge?.rawValue
    }
  }

  private static func floatValues(in tensor: HostTensor) -> [Float] {
    tensor.bytes.withUnsafeBytes { bytes in
      stride(
        from: 0,
        to: bytes.count,
        by: MemoryLayout<Float>.size
      ).map { offset in
        bytes.loadUnaligned(fromByteOffset: offset, as: Float.self)
      }
    }
  }
}

private struct ParsedRun {
  let modelURL: URL
  let input: RawTensorInput
  let outputURL: URL?
  let backend: CLIBackend
  let executionMode: MLXExecutionMode
  let computePrecision: MLXComputePrecision
  let computeUnits: CoreMLComputeUnits
  let hostBridge: CoreMLHostBridge
  let inputFormat: RunInputFormat
  let controlProfile: NeuralRenderingControlProfile
  let styleIndex: Int
  let localToneStrength: Float
  let localStructureStrength: Float
  let automaticMask: NeuralRenderingAutomaticMaskConfiguration?
  let controlMask: RawTensorInput?
  let inputTransform: NeuralRenderingTextureTransform?
  let controlMaskTransform: NeuralRenderingTextureTransform?
  let logicalWidth: Int
  let logicalHeight: Int
  let intensity: Float
  let featuresOutputURL: URL?
  let noiseFrameIndex: UInt32
  let processingScale: Float
  let detailStrength: Float
  let colourStrength: Float
  let detailRadius: Float

  func backendName(effectiveHostBridge: CoreMLHostBridge?) -> String {
    switch backend {
    case .mlx:
      "mlx-metal"
    case .coreml:
      "coreml-\(computeUnits.rawValue)-\((effectiveHostBridge ?? hostBridge).rawValue)"
    }
  }
}

struct RawTensorInput {
  private let storage: Storage
  let descriptor: TensorDescriptor

  static func literalDemo() throws -> Self {
    try RawTensorInput(
      literalValues: [0.2, 0.4, 0.25, 0.6, 2, 2],
      height: 1,
      width: 2
    )
  }

  init(
    url: URL,
    height: Int,
    width: Int,
    channels: Int = 3,
    name: String = "color"
  ) throws {
    self.storage = .file(url)
    self.descriptor = try TensorDescriptor(
      name: name,
      shape: [1, height, width, channels],
      dataType: .float32,
      layout: .nhwc
    )
  }

  private init(literalValues: [Float], height: Int, width: Int) throws {
    self.storage = .literal(literalValues)
    self.descriptor = try TensorDescriptor(
      name: "color",
      shape: [1, height, width, 3],
      dataType: .float32,
      layout: .nhwc
    )
  }

  func makeTensor() throws -> HostTensor {
    switch storage {
    case .literal(let values):
      return try HostTensor(
        descriptor: descriptor,
        bytes: values.withUnsafeBytes { Data($0) }
      )
    case .file(let url):
      let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
      let actualByteCount = (attributes[.size] as? NSNumber)?.intValue ?? -1
      guard actualByteCount == descriptor.byteCount else {
        throw TensorError.byteCount(
          expected: descriptor.byteCount,
          actual: actualByteCount
        )
      }
      return try HostTensor(
        descriptor: descriptor,
        bytes: Data(contentsOf: url, options: .mappedIfSafe)
      )
    }
  }

  private enum Storage {
    case literal([Float])
    case file(URL)
  }
}
