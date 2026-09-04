import Foundation
import MLX
import NeuralRenderCore
import NeuralRenderMLX

/// `nrk framegen A.png B.png --weights framegen.safetensors --output OUT.png [--phase 0.5] [--factor N] [--precision float16|float32]`
///
/// Generates the frame between two consecutive video frames with the ported
/// DLSS frame generator.  With `--factor N` the output name gets a `-k` suffix
/// for the N-1 phases k/N; `--phase` picks a single phase.  Prints a JSON
/// summary with the timing (`--repeat N` averages N runs after a warm-up).
enum FrameGenCommand {
  struct Summary: Encodable {
    let outputs: [String]
    let width: Int
    let height: Int
    let phases: [Float]
    let precision: String
    let networkSeconds: Double
  }

  static func run(arguments: [String]) async throws {
    var positional: [String] = []
    var weights: URL?
    var output: URL?
    var phase: Float?
    var factor = 2
    var precision = FrameGenerator.Precision.float16
    var repeatCount = 1
    var index = 0
    while index < arguments.count {
      let argument = arguments[index]
      func value() throws -> String {
        index += 1
        guard index < arguments.count else { throw CLIError.usage("\(argument) needs a value") }
        return arguments[index]
      }
      switch argument {
      case "--weights": weights = URL(fileURLWithPath: try value())
      case "--output": output = URL(fileURLWithPath: try value())
      case "--phase":
        guard let parsed = Float(try value()), parsed > 0, parsed < 1 else { throw CLIError.usage("--phase must be in (0, 1)") }
        phase = parsed
      case "--factor":
        guard let parsed = Int(try value()), parsed >= 2 else { throw CLIError.usage("--factor must be at least 2") }
        factor = parsed
      case "--precision":
        guard let parsed = FrameGenerator.Precision(rawValue: try value()) else { throw CLIError.usage("--precision must be float16 or float32") }
        precision = parsed
      case "--repeat":
        guard let parsed = Int(try value()), parsed >= 1 else { throw CLIError.usage("--repeat must be at least 1") }
        repeatCount = parsed
      default:
        if argument.hasPrefix("--") { throw CLIError.usage("unknown option \(argument)") }
        positional.append(argument)
      }
      index += 1
    }
    guard positional.count == 2 else { throw CLIError.usage("framegen expects two input images: A.png B.png") }
    guard let weights else { throw CLIError.usage("framegen requires --weights framegen.safetensors (nrk-weights extract-fg)") }
    guard let output else { throw CLIError.missingOutput("--output OUT.png") }
    let a = try ImageFile.read(URL(fileURLWithPath: positional[0]))
    let b = try ImageFile.read(URL(fileURLWithPath: positional[1]))
    guard a.descriptor.shape == b.descriptor.shape else {
      throw CLIError.usage("frames differ in size: \(a.descriptor.shape) vs \(b.descriptor.shape)")
    }
    let generator = try FrameGenerator(weightsURL: weights, precision: precision)
    let aArray = MLXArray(a.bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }, a.descriptor.shape)
    let bArray = MLXArray(b.bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }, b.descriptor.shape)
    let phases: [Float] = phase.map { [$0] } ?? (1..<factor).map { Float($0) / Float(factor) }
    let clock = ContinuousClock()
    var outputs: [String] = []
    var seconds = 0.0
    for (k, value) in phases.enumerated() {
      if repeatCount > 1 { _ = try generator.interpolate(aArray, bArray, phase: value) }  // warm-up: kernel compilation
      let started = clock.now
      var frame = try generator.interpolate(aArray, bArray, phase: value)
      for _ in 1..<repeatCount { frame = try generator.interpolate(aArray, bArray, phase: value) }
      let elapsed = started.duration(to: clock.now)
      seconds += (Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18) / Double(repeatCount)
      let url: URL
      if phases.count == 1 {
        url = output
      } else {
        let stem = output.deletingPathExtension().lastPathComponent
        url = output.deletingLastPathComponent().appendingPathComponent("\(stem)-\(k + 1).\(output.pathExtension)")
      }
      let values = frame.asArray(Float.self)
      let tensor = try HostTensor(
        descriptor: TensorDescriptor(name: "color", shape: frame.shape, dataType: .float32, layout: .nhwc),
        bytes: values.withUnsafeBytes { Data($0) }
      )
      try ImageFile.write(tensor, to: url)
      outputs.append(url.path)
    }
    try CLIOutput.writeEncodable(Summary(
      outputs: outputs, width: a.descriptor.shape[2], height: a.descriptor.shape[1], phases: phases,
      precision: precision.rawValue, networkSeconds: seconds
    ))
  }
}
