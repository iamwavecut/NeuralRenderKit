import Foundation
import DLSSCore
import DLSSMLX

/// `mlxdlss stream MODEL --width W --height H [--mode temporal|first-frame] ...`
///
/// Frame server for hosts that decode and encode video themselves (the
/// Python `mlxdlss-video` converter): raw float32 frames arrive on stdin and the
/// rendered float32 RGB frames leave on stdout, one per input frame.
///
/// Per frame: a little-endian UInt32 flag word (bit 0: reset the history
/// before this frame), then `H*W*3` colour floats, and in temporal mode
/// `H*W*2` normalised history-UV motion floats and `H*W*1` depth floats.
/// End of input ends the stream.
enum StreamCommand {
  static let resetFlag: UInt32 = 1

  static func run(arguments: [String]) async throws {
    let parsed = try parse(arguments: arguments)
    let width = parsed.width
    let height = parsed.height
    let pixelCount = width * height
    let head = try MLXNeuralRenderer(
      packageURL: parsed.modelURL,
      executionMode: parsed.executionMode,
      computePrecision: parsed.computePrecision
    )
    let temporal: MLXNeuralRenderingDeviceTemporalBackend? =
      parsed.mode == .temporal
      ? try MLXNeuralRenderingDeviceTemporalBackend(
        packageURL: parsed.modelURL,
        executionMode: parsed.executionMode,
        computePrecision: parsed.computePrecision,
        depthInverted: parsed.depthInverted,
        historyTransform: nil,
        motionTransform: nil,
        controlMaskIntensity: parsed.options.intensity,
        featureControls: parsed.options.featureControls,
        geometry: parsed.options.geometry
      )
      : nil
    let input = FileHandle.standardInput
    let output = FileHandle.standardOutput
    var streamID: UInt64 = 1
    var frameIndex: UInt64 = 0
    var frames = 0
    let clock = ContinuousClock()
    let started = clock.now
    while let flags = try readWord(from: input) {
      guard let colorData = try readExactly(pixelCount * 3 * 4, from: input) else {
        throw CLIError.usage("stream: truncated colour frame \(frames)")
      }
      let color = try tensor(named: "color", data: colorData, height: height, width: width, channels: 3)
      if flags & resetFlag != 0, frameIndex > 0 {
        streamID += 1
        frameIndex = 0
      }
      let rendered: HostTensor
      if let temporal {
        guard let motionData = try readExactly(pixelCount * 2 * 4, from: input),
          let depthData = try readExactly(pixelCount * 4, from: input)
        else {
          throw CLIError.usage("stream: truncated motion or depth frame \(frames)")
        }
        let motion = try tensor(named: "motion", data: motionData, height: height, width: width, channels: 2)
        let depth = try tensor(named: "depth", data: depthData, height: height, width: width, channels: 1)
        let request = try NeuralRenderRequest(
          sequenceID: frameIndex,
          temporalContext: NeuralRenderFrameContext(streamID: streamID, frameIndex: frameIndex),
          inputs: [color, motion, depth]
        )
        guard let result = try await temporal.render(request).output(named: "color") else {
          throw CLIError.missingOutput("color")
        }
        rendered = result
      } else {
        var options = parsed.options
        options.noiseFrameIndex = UInt32(truncatingIfNeeded: frameIndex)
        rendered = try await FirstFramePipeline.render(
          source: color,
          controlMask: nil,
          options: options,
          backend: head
        ).output
      }
      output.write(rendered.bytes)
      frameIndex += 1
      frames += 1
    }
    let seconds = started.duration(to: clock.now)
    let summary: [String: Any] = [
      "frames": frames,
      "mode": parsed.mode.rawValue,
      "seconds": Double(seconds.components.seconds) + Double(seconds.components.attoseconds) / 1e18,
      "shape": [height, width, 3],
    ]
    let data = try JSONSerialization.data(withJSONObject: summary, options: [.sortedKeys])
    FileHandle.standardError.write(data)
    FileHandle.standardError.write(Data("\n".utf8))
  }

  private static func readWord(from handle: FileHandle) throws -> UInt32? {
    guard let data = try readExactly(4, from: handle) else { return nil }
    return data.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian
  }

  /// Reads exactly `count` bytes, or nil at a clean end of input.
  private static func readExactly(_ count: Int, from handle: FileHandle) throws -> Data? {
    var buffer = Data()
    buffer.reserveCapacity(count)
    while buffer.count < count {
      let chunk = handle.readData(ofLength: count - buffer.count)
      if chunk.isEmpty {
        if buffer.isEmpty { return nil }
        throw CLIError.usage("stream: unexpected end of input")
      }
      buffer.append(chunk)
    }
    return buffer
  }

  private static func tensor(named name: String, data: Data, height: Int, width: Int, channels: Int) throws -> HostTensor {
    try HostTensor(
      descriptor: TensorDescriptor(name: name, shape: [1, height, width, channels], dataType: .float32, layout: .nhwc),
      bytes: data
    )
  }

  private enum Mode: String {
    case temporal
    case firstFrame = "first-frame"
  }

  private struct ParsedStream {
    let modelURL: URL
    let width: Int
    let height: Int
    let mode: Mode
    let executionMode: MLXExecutionMode
    let computePrecision: MLXComputePrecision
    let depthInverted: Bool
    let options: FirstFrameOptions
  }

  private static func parse(arguments: [String]) throws -> ParsedStream {
    guard let modelPath = arguments.first else {
      throw CLIError.usage("stream requires a model-package path followed by --width W --height H")
    }
    let knownOptions = [
      "--width", "--height", "--mode", "--execution", "--precision", "--profile", "--style-index",
      "--local-tone", "--local-structure", "--skin-structure", "--auto-mask", "--intensity",
      "--network-geometry", "--depth-inverted", "--processing-scale", "--detail-strength",
      "--colour-strength", "--detail-radius",
    ]
    var values: [String: String] = [:]
    var index = 1
    while index < arguments.count {
      let option = arguments[index]
      guard knownOptions.contains(option) else { throw CLIError.usage("unknown stream option '\(option)'") }
      guard index + 1 < arguments.count else { throw CLIError.usage("stream option '\(option)' needs a value") }
      guard values[option] == nil else { throw CLIError.usage("duplicate stream option '\(option)'") }
      values[option] = arguments[index + 1]
      index += 2
    }
    guard let widthText = values["--width"], let heightText = values["--height"],
      let width = Int(widthText), let height = Int(heightText), width > 0, height > 0
    else {
      throw CLIError.usage("stream requires positive --width and --height")
    }
    let mode = try enumeration(Mode.self, values["--mode"], default: .temporal, message: "stream mode must be 'temporal' or 'first-frame'")
    let executionMode = try enumeration(
      MLXExecutionMode.self, values["--execution"], default: .metalFused,
      message: "stream execution must be 'eager', 'block-compiled', 'int8-fast', or 'metal-fused'"
    )
    guard executionMode != .compiled else { throw CLIError.usage("stream does not support whole-graph compilation") }
    let computePrecision = try enumeration(
      MLXComputePrecision.self, values["--precision"], default: .float16, message: "stream precision must be 'float32' or 'float16'"
    )
    let profile = try enumeration(
      NeuralRenderingControlProfile.self, values["--profile"], default: .standard,
      message: "stream profile must be 'standard', 'natural', 'cinematic', or 'neutral'"
    )
    let geometry = try enumeration(
      NeuralRenderingNetworkGeometryPolicy.self, values["--network-geometry"], default: .vendorAligned,
      message: "stream network geometry must be 'vendor-aligned' or 'match-output'"
    )
    let styleIndex: Int
    if let text = values["--style-index"] {
      guard let parsed = UInt32(text) else { throw CLIError.usage("stream style index must be an unsigned integer") }
      styleIndex = Int(parsed)
    } else {
      styleIndex = profile.styleIndex
    }
    let localTone = try floatOption("--local-tone", values: values, default: profile.localToneStrength)
    let localStructure = try floatOption("--local-structure", values: values, default: profile.localStructureStrength)
    let skinStructure = try floatOption("--skin-structure", values: values, default: profile.skinStructureStrength)
    let intensity = try floatOption("--intensity", values: values, default: profile.intensity)
    let processingScale = try floatOption("--processing-scale", values: values, default: 1)
    let detailStrength = try floatOption("--detail-strength", values: values, default: 1)
    let colourStrength = try floatOption("--colour-strength", values: values, default: 1)
    let detailRadius = try floatOption("--detail-radius", values: values, default: 4)
    if mode == .temporal, processingScale != 1 {
      throw CLIError.usage("stream temporal mode runs at the native scale")
    }
    var depthInverted = false
    if let text = values["--depth-inverted"] {
      guard let parsed = Bool(text) else { throw CLIError.usage("stream --depth-inverted must be true or false") }
      depthInverted = parsed
    }
    let automaticMaskRequested: Bool
    if let text = values["--auto-mask"] {
      guard let parsedMask = RunAutomaticMask(rawValue: text) else { throw CLIError.usage("stream auto mask must be 'enabled' or 'disabled'") }
      automaticMaskRequested = parsedMask == .enabled
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
      noiseFrameIndex: 0,
      geometry: geometry,
      processingScale: processingScale,
      detailStrength: detailStrength,
      colourStrength: colourStrength,
      detailRadius: detailRadius
    )
    return ParsedStream(
      modelURL: URL(fileURLWithPath: modelPath), width: width, height: height, mode: mode,
      executionMode: executionMode, computePrecision: computePrecision, depthInverted: depthInverted, options: options
    )
  }

  private static func enumeration<T: RawRepresentable>(_: T.Type, _ text: String?, default defaultValue: T, message: String) throws -> T
  where T.RawValue == String {
    guard let text else { return defaultValue }
    guard let value = T(rawValue: text) else { throw CLIError.usage(message) }
    return value
  }

  private static func floatOption(_ name: String, values: [String: String], default defaultValue: Float) throws -> Float {
    guard let text = values[name] else { return defaultValue }
    guard let value = Float(text), value.isFinite else { throw CLIError.usage("stream \(name) must be a finite number") }
    return value
  }
}
