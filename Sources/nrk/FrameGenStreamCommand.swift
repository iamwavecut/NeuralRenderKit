import Foundation
import MLX
import NeuralRenderMLX

/// `nrk framegen-stream --weights framegen.safetensors --width W --height H [--factor N] [--batch P] [--format u8|f32] [--precision float16|float32]`
///
/// Frame server for hosts that decode and encode video themselves (the Python
/// `nrk-video framegen --backend nrk`): RGB frames (`H*W*3` values; `--format u8`
/// bytes, the default, or `f32` floats in [0, 1]) arrive on stdin; the `factor - 1` generated frames of every consecutive pair
/// leave on stdout in stream order (pair by pair, phase by phase). Pairs are
/// computed `batch` at a time (one pass over `batch * (factor - 1)` samples), so
/// the output of a pair appears once its window of `batch` pairs is complete or
/// the input ends. End of input ends the stream; a JSON summary goes to stderr.
enum FrameGenStreamCommand {
  struct Summary: Encodable {
    let inputFrames: Int
    let generatedFrames: Int
    let seconds: Double
    let precision: String
    let batch: Int
    let format: String
  }

  static func run(arguments: [String]) async throws {
    var weights: URL?
    var width = 0
    var height = 0
    var factor = 2
    var batch = 4
    var bytesPerValue = 1
    var precision = FrameGenerator.Precision.float16
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
      case "--width": width = Int(try value()) ?? 0
      case "--height": height = Int(try value()) ?? 0
      case "--factor":
        guard let parsed = Int(try value()), parsed >= 2 else { throw CLIError.usage("--factor must be at least 2") }
        factor = parsed
      case "--batch":
        guard let parsed = Int(try value()), parsed >= 1 else { throw CLIError.usage("--batch must be at least 1") }
        batch = parsed
      case "--format":
        switch try value() {
        case "u8": bytesPerValue = 1
        case "f32": bytesPerValue = 4
        default: throw CLIError.usage("--format must be u8 or f32")
        }
      case "--precision":
        guard let parsed = FrameGenerator.Precision(rawValue: try value()) else { throw CLIError.usage("--precision must be float16 or float32") }
        precision = parsed
      default: throw CLIError.usage("unknown option \(argument)")
      }
      index += 1
    }
    guard let weights else { throw CLIError.usage("framegen-stream requires --weights") }
    guard width > 0, height > 0 else { throw CLIError.usage("framegen-stream requires --width and --height") }
    let generator = try FrameGenerator(weightsURL: weights, precision: precision)
    let input = FileHandle.standardInput
    let output = FileHandle.standardOutput
    let frameBytes = width * height * 3 * bytesPerValue
    var window: [MLXArray] = []   // the frames of the current window: up to batch + 1
    var inputFrames = 0
    var generated = 0
    let clock = ContinuousClock()
    let started = clock.now
    let phases = (1..<factor).map { Float($0) / Float(factor) }
    func flush() throws {
      guard window.count >= 2 else { return }
      let pairs = window.count - 1
      let a = concatenated(window.dropLast().flatMap { f in Array(repeating: f, count: phases.count) }, axis: 0)
      let b = concatenated(window.dropFirst().flatMap { f in Array(repeating: f, count: phases.count) }, axis: 0)
      let out = try generator.interpolate(a, b, phases: (0..<pairs).flatMap { _ in phases })   // [pairs * (factor - 1), H, W, 3]
      if bytesPerValue == 1 {
        let bytes = (out * 255 + 0.5).asType(.uint8)   // out is clamped to [0, 1] by the compose kernel
        output.write(bytes.asData(access: .copy).data)
      } else {
        output.write(out.asData(access: .copy).data)
      }
      generated += pairs * phases.count
      window = [window[window.count - 1]]
    }
    while let data = try readExactly(frameBytes, from: input) {
      let frame = bytesPerValue == 1
        ? MLXArray(data, [1, height, width, 3], dtype: .uint8).asType(.float32) * (1.0 / 255.0)
        : MLXArray(data, [1, height, width, 3], dtype: .float32)
      inputFrames += 1
      window.append(frame)
      if window.count == batch + 1 { try flush() }
    }
    try flush()
    let elapsed = started.duration(to: clock.now)
    let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
    let summary = Summary(inputFrames: inputFrames, generatedFrames: generated, seconds: seconds, precision: precision.rawValue, batch: batch, format: bytesPerValue == 1 ? "u8" : "f32")
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    var line = try encoder.encode(summary)
    line.append(0x0a)
    FileHandle.standardError.write(line)
  }

  private static func readExactly(_ count: Int, from handle: FileHandle) throws -> Data? {
    var buffer = Data()
    buffer.reserveCapacity(count)
    while buffer.count < count {
      let chunk = handle.readData(ofLength: count - buffer.count)
      if chunk.isEmpty {
        if buffer.isEmpty { return nil }
        throw CLIError.usage("framegen-stream: unexpected end of input")
      }
      buffer.append(chunk)
    }
    return buffer
  }
}
