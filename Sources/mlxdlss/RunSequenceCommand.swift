import Foundation
import DLSSCore
import DLSSCoreML
import DLSSMLX

private enum SequenceInputFormat: String {
  case model
  case rgbTemporalReference = "rgb-temporal-reference"
}

private enum SequenceMotionFormat: String {
  case normalizedUV = "normalized-uv"
  case pixel
}

private enum SequencePreprocessor: String {
  case cpu
  case metal
}

private enum SequencePipeline: String {
  case portable
  case deviceResident = "device-resident"
}

enum RunSequenceCommand {
  static func run(arguments: [String]) async throws {
    let parsed = try parse(arguments: arguments)
    let renderer: any NeuralRenderBackend
    let modelInputChannels: Int
    var effectiveHostBridge: CoreMLHostBridge?
    switch parsed.inputFormat {
    case .model:
      guard parsed.backend == .mlx else {
        throw CLIError.usage(
          "run-sequence model input currently requires the MLX backend"
        )
      }
      let headRenderer = try MLXNeuralRenderer(
        packageURL: parsed.modelURL,
        executionMode: parsed.executionMode,
        computePrecision: parsed.computePrecision
      )
      renderer = headRenderer
      modelInputChannels = try fixedModelInputChannels(at: parsed.modelURL)
    case .rgbTemporalReference:
      modelInputChannels = 3
      let temporalPreprocessor: any NeuralRenderingTemporalFeaturePreprocessing =
        switch parsed.preprocessor {
        case .cpu:
          NeuralRenderingCPUTemporalFeaturePreprocessor(
            depthGuideMode: .observedZeroDescriptor,
            historyTransform: parsed.historyTransform,
            motionTransform: parsed.motionTransform,
            featureControls: parsed.featureControls
          )
        case .metal:
          try MetalNeuralRenderingTemporalFeaturePreprocessor(
            depthGuideMode: .observedZeroDescriptor,
            historyTransform: parsed.historyTransform,
            motionTransform: parsed.motionTransform,
            featureControls: parsed.featureControls
          )
        }
      switch parsed.backend {
      case .mlx:
        switch parsed.pipeline {
        case .portable:
          let headRenderer = try MLXNeuralRenderer(
            packageURL: parsed.modelURL,
            executionMode: parsed.executionMode,
            computePrecision: parsed.computePrecision
          )
          renderer = NeuralRenderingTemporalReferenceBackend(
            backend: headRenderer,
            depthInverted: parsed.depthInverted,
            controlMaskIntensity: parsed.controlMaskIntensity,
            temporalPreprocessor: temporalPreprocessor,
            geometry: parsed.networkGeometry
          )
        case .deviceResident:
          renderer = try MLXNeuralRenderingDeviceTemporalBackend(
            packageURL: parsed.modelURL,
            executionMode: parsed.executionMode,
            computePrecision: parsed.computePrecision,
            depthInverted: parsed.depthInverted,
            historyTransform: parsed.historyTransform,
            motionTransform: parsed.motionTransform,
            controlMaskIntensity: parsed.controlMaskIntensity,
            featureControls: parsed.featureControls,
            geometry: parsed.networkGeometry
          )
        }
      case .coreml:
        guard parsed.pipeline == .portable else {
          throw CLIError.usage(
            "run-sequence device-resident pipeline requires the MLX backend"
          )
        }
        let headRenderer = try await CoreMLNeuralRenderer(
          modelURL: parsed.modelURL,
          configuration: CoreMLBackendConfiguration(
            computeUnits: parsed.computeUnits,
            hostBridge: parsed.hostBridge
          )
        )
        let networkGeometry = try parsed.networkGeometry.resolve(
          outputWidth: parsed.width,
          outputHeight: parsed.height
        )
        guard headRenderer.inputChannels == 16,
          headRenderer.outputChannels == 4,
          headRenderer.width == networkGeometry.networkWidth,
          headRenderer.height == networkGeometry.networkHeight
        else {
          throw CLIError.usage(
            "run-sequence Core ML head must be fixed 16-to-4 at the network extent "
              + "\(networkGeometry.networkWidth)x\(networkGeometry.networkHeight)"
          )
        }
        effectiveHostBridge = headRenderer.effectiveHostBridge
        renderer = NeuralRenderingTemporalReferenceBackend(
          backend: headRenderer,
          depthInverted: parsed.depthInverted,
          controlMaskIntensity: parsed.controlMaskIntensity,
          temporalPreprocessor: temporalPreprocessor,
          geometry: parsed.networkGeometry
        )
      }
    }

    let fileManager = FileManager.default
    let outputDirectory = parsed.outputDirectory.standardizedFileURL
    guard !fileManager.fileExists(atPath: outputDirectory.path) else {
      throw CLIError.destinationExists(outputDirectory)
    }
    let stagingDirectory = outputDirectory.deletingLastPathComponent()
      .appendingPathComponent(
        ".\(outputDirectory.lastPathComponent).\(UUID().uuidString).tmp",
        isDirectory: true
      )
    try fileManager.createDirectory(
      at: stagingDirectory,
      withIntermediateDirectories: false
    )
    defer { try? fileManager.removeItem(at: stagingDirectory) }

    var executionNanoseconds: UInt64 = 0
    var endToEndNanoseconds: UInt64 = 0
    var outputShape: [Int] = []
    for (index, inputURL) in parsed.inputURLs.enumerated() {
      let endToEndStart = DispatchTime.now().uptimeNanoseconds
      let colorResource = try RawTensorInput(
        url: inputURL,
        height: parsed.inputTransform?.resourceHeight ?? parsed.height,
        width: parsed.inputTransform?.resourceWidth ?? parsed.width,
        channels: modelInputChannels
      ).makeTensor()
      let color = try parsed.inputTransform?.pointSample(
        colorResource,
        logicalWidth: parsed.width,
        logicalHeight: parsed.height
      ) ?? colorResource
      let inputs: [HostTensor]
      switch parsed.inputFormat {
      case .model:
        inputs = [color]
      case .rgbTemporalReference:
        let rawMotion = try RawTensorInput(
          url: parsed.motionURLs[index],
          height: parsed.motionTransform?.resourceHeight ?? parsed.height,
          width: parsed.motionTransform?.resourceWidth ?? parsed.width,
          channels: 2,
          name: parsed.motionFormat == .pixel ? "pixelMotion" : "motion"
        ).makeTensor()
        let motion =
          if parsed.motionFormat == .pixel {
            try NeuralRenderingTemporalReferencePreprocessor.normalizePixelMotion(
              rawMotion,
              scaleX: parsed.motionScaleX,
              scaleY: parsed.motionScaleY,
              effectiveWidth:
                parsed.motionTransform?.extentWidth ?? parsed.width,
              effectiveHeight:
                parsed.motionTransform?.extentHeight ?? parsed.height,
              jitterDeltaX: parsed.jitterDeltaXs[index],
              jitterDeltaY: parsed.jitterDeltaYs[index]
            )
          } else {
            rawMotion
          }
        let depth = try RawTensorInput(
          url: parsed.depthURLs[index],
          height: parsed.height,
          width: parsed.width,
          channels: 1,
          name: "depth"
        ).makeTensor()
        var frameInputs = [color, motion, depth]
        if !parsed.controlMaskURLs.isEmpty {
          let maskResource = try RawTensorInput(
            url: parsed.controlMaskURLs[index],
            height: parsed.controlMaskTransform?.resourceHeight ?? parsed.height,
            width: parsed.controlMaskTransform?.resourceWidth ?? parsed.width,
            channels: 3,
            name: "controlMask"
          ).makeTensor()
          let mask = try parsed.controlMaskTransform?.pointSample(
            maskResource,
            logicalWidth: parsed.width,
            logicalHeight: parsed.height
          ) ?? maskResource
          frameInputs.append(mask)
        }
        inputs = frameInputs
      }
      let frameIndex = UInt64(index)
      let request = try NeuralRenderRequest(
        sequenceID: frameIndex,
        temporalContext: NeuralRenderFrameContext(
          streamID: 1,
          frameIndex: frameIndex
        ),
        inputs: inputs
      )
      let result = try await renderer.render(request)
      guard let output = result.output(named: "color") else {
        throw CLIError.missingOutput("color")
      }
      outputShape = output.descriptor.shape
      executionNanoseconds += result.timing.executionNanoseconds
      try output.bytes.write(
        to: stagingDirectory.appendingPathComponent(
          String(format: "%06d.f32", index)
        ),
        options: .atomic
      )
      endToEndNanoseconds += DispatchTime.now().uptimeNanoseconds - endToEndStart
    }

    try fileManager.moveItem(at: stagingDirectory, to: outputDirectory)
    var summary: [String: Any] = [
      "backend": parsed.backendName(effectiveHostBridge: effectiveHostBridge),
      "cadence": renderer.temporalCadence.rawValue,
      "executionNanoseconds": executionNanoseconds,
      "endToEndNanoseconds": endToEndNanoseconds,
      "frameCount": parsed.inputURLs.count,
      "inputFormat": parsed.inputFormat.rawValue,
      "motionFormat": parsed.inputFormat == .rgbTemporalReference
        ? parsed.motionFormat.rawValue
        : "not-applicable",
      "pipeline": parsed.inputFormat == .rgbTemporalReference
        ? parsed.pipeline.rawValue
        : "not-applicable",
      "preprocessor": parsed.inputFormat == .rgbTemporalReference
        && parsed.pipeline == .portable
        ? parsed.preprocessor.rawValue
        : "not-applicable",
      "outputDirectory": outputDirectory.path,
      "shape": outputShape,
    ]
    switch parsed.backend {
    case .mlx:
      summary["computePrecision"] = parsed.computePrecision.rawValue
      summary["executionMode"] = parsed.executionMode.rawValue
    case .coreml:
      summary["computeUnits"] = parsed.computeUnits.rawValue
      summary["hostBridge"] = effectiveHostBridge?.rawValue
    }
    if let transform = parsed.inputTransform {
      summary["inputTransform"] = CLITextureTransform.fields(transform)
    }
    if let transform = parsed.historyTransform {
      summary["historyTransform"] = CLITextureTransform.fields(transform)
    }
    if let transform = parsed.motionTransform {
      summary["motionTransform"] = CLITextureTransform.fields(transform)
    }
    if let transform = parsed.controlMaskTransform {
      summary["controlMaskTransform"] = CLITextureTransform.fields(transform)
    }
    if parsed.inputFormat == .rgbTemporalReference {
      summary["networkGeometry"] = parsed.networkGeometry.rawValue
      if let geometry = try? parsed.networkGeometry.resolve(
        outputWidth: parsed.width,
        outputHeight: parsed.height
      ) {
        summary["networkShape"] = [1, geometry.networkHeight, geometry.networkWidth, 16]
      }
      summary["profile"] = parsed.controlProfile.rawValue
      summary["checkpointModelSelection"] =
        parsed.controlProfile.checkpointModelSelection.rawValue
      summary["checkpointModel"] = "shipping-default"
      summary["controlMask"] = parsed.controlMaskURLs.isEmpty
        ? "none"
        : parsed.controlMaskTransform == nil
          ? "full-rect-rgb"
          : "point-transformed-rgb"
      summary["intensity"] = Double(parsed.controlMaskIntensity)
      summary["styleIndex"] = parsed.styleIndex
      summary["localToneStrength"] = Double(
        parsed.featureControls.localToneStrength
      )
      summary["localStructureStrength"] = Double(
        parsed.featureControls.localStructureStrength
      )
      summary["effectiveAutomaticMask"] =
        parsed.featureControls.automaticMask != nil
      if parsed.motionFormat == .pixel {
        let deltas = zip(parsed.jitterDeltaXs, parsed.jitterDeltaYs).map {
          [Double($0), Double($1)]
        }
        summary["jitterDeltaPixelsPerFrame"] = deltas
        if let first = deltas.first, deltas.dropFirst().allSatisfy({ $0 == first }) {
          summary["jitterDeltaPixels"] = first
        }
      }
    }
    try CLIOutput.writeJSON(summary)
  }

  private static func fixedModelInputChannels(at packageURL: URL) throws -> Int {
    let package = try ModelPackageLoader.load(url: packageURL)
    guard
      let input = package.manifest.inputs.first(where: { $0.name == "color" }),
      let channelDimension = input.shape.last,
      case .fixed(let channels) = channelDimension,
      channels > 0
    else {
      throw CLIError.usage(
        "run-sequence model input requires a fixed positive channel count"
      )
    }
    return channels
  }

  private static func parse(arguments: [String]) throws -> ParsedSequenceRun {
    guard let packagePath = arguments.first else {
      throw CLIError.usage("run-sequence requires a model-package path")
    }
    var values: [String: String] = [:]
    var inputURLs: [URL] = []
    var motionURLs: [URL] = []
    var depthURLs: [URL] = []
    var controlMaskURLs: [URL] = []
    var jitterDeltaXValues: [String] = []
    var jitterDeltaYValues: [String] = []
    var index = 1
    while index < arguments.count {
      let option = arguments[index]
      guard index + 1 < arguments.count else {
        throw CLIError.usage("run-sequence option '\(option)' needs a value")
      }
      let value = arguments[index + 1]
      if option == "--input" {
        inputURLs.append(URL(fileURLWithPath: value))
      } else if option == "--motion" {
        motionURLs.append(URL(fileURLWithPath: value))
      } else if option == "--depth" {
        depthURLs.append(URL(fileURLWithPath: value))
      } else if option == "--control-mask" {
        controlMaskURLs.append(URL(fileURLWithPath: value))
      } else if option == "--jitter-delta-x" {
        jitterDeltaXValues.append(value)
      } else if option == "--jitter-delta-y" {
        jitterDeltaYValues.append(value)
      } else {
        guard
          [
            "--height", "--width", "--output-dir", "--execution", "--precision",
            "--input-format", "--depth-inverted", "--motion-format",
            "--motion-scale-x", "--motion-scale-y", "--preprocessor", "--pipeline",
            "--backend", "--compute-units", "--coreml-bridge",
            "--input-transform", "--history-transform", "--motion-transform",
            "--control-mask-transform", "--intensity", "--style-index",
            "--profile", "--network-geometry",
            "--local-tone", "--local-structure", "--auto-mask",
            "--skin-structure",
          ]
          .contains(option)
        else {
          throw CLIError.usage("unknown run-sequence option '\(option)'")
        }
        guard values[option] == nil else {
          throw CLIError.usage("duplicate run-sequence option '\(option)'")
        }
        values[option] = value
      }
      index += 2
    }

    guard !inputURLs.isEmpty else {
      throw CLIError.usage("run-sequence requires at least one --input")
    }
    let backend: CLIBackend
    if let value = values["--backend"] {
      guard let parsed = CLIBackend(rawValue: value) else {
        throw CLIError.usage("run-sequence backend must be 'mlx' or 'coreml'")
      }
      backend = parsed
    } else {
      backend = .mlx
    }
    let inputFormat: SequenceInputFormat
    if let value = values["--input-format"] {
      guard let parsed = SequenceInputFormat(rawValue: value) else {
        throw CLIError.usage(
          "run-sequence input format must be 'model' or 'rgb-temporal-reference'"
        )
      }
      inputFormat = parsed
    } else {
      inputFormat = .model
    }
    let inputTransform = try CLITextureTransform.parse(
      values["--input-transform"],
      command: "run-sequence",
      label: "input"
    )
    let historyTransform = try CLITextureTransform.parse(
      values["--history-transform"],
      command: "run-sequence",
      label: "history"
    )
    let motionTransform = try CLITextureTransform.parse(
      values["--motion-transform"],
      command: "run-sequence",
      label: "motion"
    )
    let controlMaskTransform = try CLITextureTransform.parse(
      values["--control-mask-transform"],
      command: "run-sequence",
      label: "control mask"
    )
    if inputFormat != .rgbTemporalReference,
      inputTransform != nil || historyTransform != nil || motionTransform != nil
        || controlMaskTransform != nil || !controlMaskURLs.isEmpty
        || values["--intensity"] != nil
    {
      throw CLIError.usage(
        "run-sequence texture transforms require rgb-temporal-reference input"
      )
    }
    let featureControlOptions = [
      "--profile", "--style-index", "--local-tone", "--local-structure", "--auto-mask",
      "--skin-structure",
    ]
    if inputFormat != .rgbTemporalReference,
      featureControlOptions.contains(where: { values[$0] != nil })
    {
      throw CLIError.usage(
        "run-sequence feature controls require rgb-temporal-reference input"
      )
    }
    if controlMaskTransform != nil, controlMaskURLs.isEmpty {
      throw CLIError.usage(
        "run-sequence control mask transform requires --control-mask"
      )
    }
    let motionFormat: SequenceMotionFormat
    if let value = values["--motion-format"] {
      guard let parsed = SequenceMotionFormat(rawValue: value) else {
        throw CLIError.usage(
          "run-sequence motion format must be 'normalized-uv' or 'pixel'"
        )
      }
      motionFormat = parsed
    } else {
      motionFormat = .normalizedUV
    }
    let preprocessor: SequencePreprocessor
    if let value = values["--preprocessor"] {
      guard let parsed = SequencePreprocessor(rawValue: value) else {
        throw CLIError.usage("run-sequence preprocessor must be 'cpu' or 'metal'")
      }
      preprocessor = parsed
    } else {
      preprocessor = .cpu
    }
    let pipeline: SequencePipeline
    if let value = values["--pipeline"] {
      guard let parsed = SequencePipeline(rawValue: value) else {
        throw CLIError.usage(
          "run-sequence pipeline must be 'portable' or 'device-resident'"
        )
      }
      pipeline = parsed
    } else {
      pipeline = .portable
    }
    let networkGeometry: NeuralRenderingNetworkGeometryPolicy
    if let value = values["--network-geometry"] {
      guard let parsed = NeuralRenderingNetworkGeometryPolicy(rawValue: value) else {
        throw CLIError.usage(
          "run-sequence network geometry must be 'vendor-aligned' or 'match-output'"
        )
      }
      networkGeometry = parsed
    } else {
      networkGeometry = .vendorAligned
    }
    let controlProfile: NeuralRenderingControlProfile
    if let value = values["--profile"] {
      guard let parsed = NeuralRenderingControlProfile(rawValue: value) else {
        throw CLIError.usage(
          "run-sequence profile must be 'standard', 'natural', 'cinematic', or 'neutral'"
        )
      }
      controlProfile = parsed
    } else {
      controlProfile = .standard
    }
    let styleIndex: Int
    if let value = values["--style-index"] {
      guard let parsed = UInt32(value) else {
        throw CLIError.usage(
          "run-sequence style index must be an unsigned integer"
        )
      }
      styleIndex = Int(parsed)
    } else {
      styleIndex = controlProfile.styleIndex
    }
    let localToneStrength = try finiteFloat(
      values["--local-tone"],
      option: "local tone",
      default: controlProfile.localToneStrength
    )
    let localStructureStrength = try finiteFloat(
      values["--local-structure"],
      option: "local structure",
      default: controlProfile.localStructureStrength
    )
    let skinStructureStrength = try finiteFloat(
      values["--skin-structure"],
      option: "skin structure",
      default: controlProfile.skinStructureStrength
    )
    let automaticMaskRequested: Bool
    if let value = values["--auto-mask"] {
      guard let mode = RunAutomaticMask(rawValue: value) else {
        throw CLIError.usage(
          "run-sequence auto mask must be 'enabled' or 'disabled'"
        )
      }
      automaticMaskRequested = mode == .enabled
    } else {
      automaticMaskRequested = controlProfile.automaticMaskEnabled
    }
    let automaticMask: NeuralRenderingAutomaticMaskConfiguration? =
      if automaticMaskRequested, controlMaskURLs.isEmpty {
        NeuralRenderingAutomaticMaskConfiguration(
          skinStructureStrength:
            skinStructureStrength >= 0
            ? skinStructureStrength
            : localStructureStrength,
          automaticMaskStructureStrength: localStructureStrength
        )
      } else {
        nil
      }
    let featureControls = NeuralRenderingFeatureControls(
      normalizedStyle: Float(styleIndex) / 128,
      localToneStrength: localToneStrength,
      localStructureStrength: localStructureStrength,
      automaticMask: automaticMask
    )
    let motionScaleX = try motionScale(values["--motion-scale-x"], option: "x")
    let motionScaleY = try motionScale(values["--motion-scale-y"], option: "y")
    let jitterDeltaXs = try jitterDeltas(
      jitterDeltaXValues,
      option: "x",
      frameCount: inputURLs.count
    )
    let jitterDeltaYs = try jitterDeltas(
      jitterDeltaYValues,
      option: "y",
      frameCount: inputURLs.count
    )
    let controlMaskIntensity = try finiteFloat(
      values["--intensity"],
      option: "intensity",
      default: controlProfile.intensity
    )
    if inputFormat == .rgbTemporalReference {
      guard motionURLs.count == inputURLs.count else {
        throw CLIError.usage(
          "run-sequence rgb-temporal-reference requires one --motion per --input"
        )
      }
      guard depthURLs.count == inputURLs.count else {
        throw CLIError.usage(
          "run-sequence rgb-temporal-reference requires one --depth per --input"
        )
      }
      guard controlMaskURLs.isEmpty || controlMaskURLs.count == inputURLs.count else {
        throw CLIError.usage(
          "run-sequence requires one --control-mask per --input or none"
        )
      }
    } else if !motionURLs.isEmpty || !depthURLs.isEmpty {
      throw CLIError.usage(
        "run-sequence --motion and --depth require rgb-temporal-reference input"
      )
    }
    if inputFormat != .rgbTemporalReference,
      values["--motion-format"] != nil
        || values["--motion-scale-x"] != nil
        || values["--motion-scale-y"] != nil
        || !jitterDeltaXValues.isEmpty
        || !jitterDeltaYValues.isEmpty
        || values["--preprocessor"] != nil
        || values["--pipeline"] != nil
        || values["--network-geometry"] != nil
    {
      throw CLIError.usage(
        "run-sequence motion options require rgb-temporal-reference input"
      )
    }
    if motionFormat != .pixel,
      values["--motion-scale-x"] != nil || values["--motion-scale-y"] != nil
    {
      throw CLIError.usage(
        "run-sequence motion scales require '--motion-format pixel'"
      )
    }
    if motionFormat != .pixel,
      !jitterDeltaXValues.isEmpty || !jitterDeltaYValues.isEmpty
    {
      throw CLIError.usage(
        "run-sequence jitter deltas require '--motion-format pixel'"
      )
    }
    for option in ["--height", "--width", "--output-dir"] where values[option] == nil {
      throw CLIError.usage("missing run-sequence option '\(option)'")
    }
    guard let height = Int(values["--height"]!),
      let width = Int(values["--width"]!)
    else {
      throw CLIError.usage("run-sequence height and width must be integers")
    }
    guard height > 0, width > 0 else {
      throw CLIError.usage("run-sequence height and width must be positive")
    }
    let executionMode: MLXExecutionMode
    if let value = values["--execution"] {
      guard let parsed = MLXExecutionMode(rawValue: value) else {
        throw CLIError.usage(
          "run-sequence execution must be 'eager', 'block-compiled', 'int8-fast', "
            + "'metal-fused', or 'compiled'"
        )
      }
      executionMode = parsed
    } else {
      executionMode = .eager
    }
    let computePrecision: MLXComputePrecision
    if let value = values["--precision"] {
      guard let parsed = MLXComputePrecision(rawValue: value) else {
        throw CLIError.usage("run-sequence precision must be 'float32' or 'float16'")
      }
      computePrecision = parsed
    } else {
      computePrecision = .float32
    }
    let computeUnits: CoreMLComputeUnits
    if let value = values["--compute-units"] {
      guard let parsed = CoreMLComputeUnits(rawValue: value) else {
        throw CLIError.usage(
          "run-sequence compute units must be 'cpu', 'cpu-gpu', "
            + "'cpu-neural-engine', or 'all'"
        )
      }
      computeUnits = parsed
    } else {
      computeUnits = .cpuAndGPU
    }
    let hostBridge: CoreMLHostBridge
    if let value = values["--coreml-bridge"] {
      guard let parsed = CoreMLHostBridge(rawValue: value) else {
        throw CLIError.usage(
          "run-sequence Core ML bridge must be 'automatic', 'multi-array', "
            + "or 'ml-tensor'"
        )
      }
      hostBridge = parsed
    } else {
      hostBridge = .automatic
    }
    if backend == .coreml {
      if values["--execution"] != nil || values["--precision"] != nil {
        throw CLIError.usage(
          "run-sequence --execution and --precision apply only to the MLX backend"
        )
      }
    } else if values["--compute-units"] != nil
      || values["--coreml-bridge"] != nil
    {
      throw CLIError.usage(
        "run-sequence --compute-units and --coreml-bridge apply only to Core ML"
      )
    }
    let depthInverted: Bool
    if let value = values["--depth-inverted"] {
      guard let parsed = Bool(value) else {
        throw CLIError.usage("run-sequence depth-inverted must be 'true' or 'false'")
      }
      depthInverted = parsed
    } else {
      depthInverted = false
    }

    return ParsedSequenceRun(
      modelURL: URL(fileURLWithPath: packagePath),
      backend: backend,
      inputURLs: inputURLs,
      motionURLs: motionURLs,
      depthURLs: depthURLs,
      controlMaskURLs: controlMaskURLs,
      outputDirectory: URL(fileURLWithPath: values["--output-dir"]!),
      height: height,
      width: width,
      executionMode: executionMode,
      computePrecision: computePrecision,
      computeUnits: computeUnits,
      hostBridge: hostBridge,
      inputFormat: inputFormat,
      depthInverted: depthInverted,
      motionFormat: motionFormat,
      motionScaleX: motionScaleX,
      motionScaleY: motionScaleY,
      jitterDeltaXs: jitterDeltaXs,
      jitterDeltaYs: jitterDeltaYs,
      preprocessor: preprocessor,
      pipeline: pipeline,
      networkGeometry: networkGeometry,
      inputTransform: inputTransform,
      historyTransform: historyTransform,
      motionTransform: motionTransform,
      controlMaskTransform: controlMaskTransform,
      controlMaskIntensity: controlMaskIntensity,
      controlProfile: controlProfile,
      styleIndex: styleIndex,
      featureControls: featureControls
    )
  }

  private static func motionScale(_ value: String?, option: String) throws -> Float {
    guard let value else { return 1 }
    guard let parsed = Float(value), parsed.isFinite else {
      throw CLIError.usage("run-sequence motion scale \(option) must be finite")
    }
    return parsed
  }

  private static func jitterDeltas(
    _ values: [String],
    option: String,
    frameCount: Int
  ) throws -> [Float] {
    let values = values.isEmpty ? ["0"] : values
    guard values.count == 1 || values.count == frameCount else {
      throw CLIError.usage(
        "run-sequence requires one jitter delta \(option) or one per input"
      )
    }
    let parsed = try values.map { value in
      guard let parsed = Float(value), parsed.isFinite else {
        throw CLIError.usage("run-sequence jitter delta \(option) must be finite")
      }
      return parsed
    }
    return parsed.count == 1
      ? [Float](repeating: parsed[0], count: frameCount)
      : parsed
  }

  private static func finiteFloat(
    _ value: String?,
    option: String,
    default defaultValue: Float
  ) throws -> Float {
    guard let value else { return defaultValue }
    guard let parsed = Float(value), parsed.isFinite else {
      throw CLIError.usage("run-sequence \(option) must be finite")
    }
    return parsed
  }
}

private struct ParsedSequenceRun {
  let modelURL: URL
  let backend: CLIBackend
  let inputURLs: [URL]
  let motionURLs: [URL]
  let depthURLs: [URL]
  let controlMaskURLs: [URL]
  let outputDirectory: URL
  let height: Int
  let width: Int
  let executionMode: MLXExecutionMode
  let computePrecision: MLXComputePrecision
  let computeUnits: CoreMLComputeUnits
  let hostBridge: CoreMLHostBridge
  let inputFormat: SequenceInputFormat
  let depthInverted: Bool
  let motionFormat: SequenceMotionFormat
  let motionScaleX: Float
  let motionScaleY: Float
  let jitterDeltaXs: [Float]
  let jitterDeltaYs: [Float]
  let preprocessor: SequencePreprocessor
  let pipeline: SequencePipeline
  let networkGeometry: NeuralRenderingNetworkGeometryPolicy
  let inputTransform: NeuralRenderingTextureTransform?
  let historyTransform: NeuralRenderingTextureTransform?
  let motionTransform: NeuralRenderingTextureTransform?
  let controlMaskTransform: NeuralRenderingTextureTransform?
  let controlMaskIntensity: Float
  let controlProfile: NeuralRenderingControlProfile
  let styleIndex: Int
  let featureControls: NeuralRenderingFeatureControls

  func backendName(effectiveHostBridge: CoreMLHostBridge?) -> String {
    switch backend {
    case .mlx:
      "mlx-metal"
    case .coreml:
      "coreml-\(computeUnits.rawValue)-\((effectiveHostBridge ?? hostBridge).rawValue)"
    }
  }
}
