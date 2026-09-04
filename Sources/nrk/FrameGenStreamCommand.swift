import Foundation
import MLX
import NeuralRenderMLX

/// `nrk framegen-stream --weights framegen.safetensors --width W --height H [--factor N] [--precision float16|float32]`
///
/// Frame server for hosts that decode and encode video themselves (the Python
/// `nrk-video framegen --backend nrk`): float32 RGB frames (`H*W*3` floats)
/// arrive on stdin; from the second frame on, the `factor - 1` generated frames
/// between the previous frame and the new one leave on stdout, in phase order.
/// End of input ends the stream; a JSON summary goes to stderr.
enum FrameGenStreamCommand {
  struct Summary: Encodable {
    let inputFrames: Int
    let generatedFrames: Int
    let seconds: Double
    let precision: String
  }

  static func run(arguments: [String]) async throws {
    var weights: URL?
    var width = 0
    var height = 0
    var factor = 2
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
    let frameBytes = width * height * 3 * MemoryLayout<Float>.size
    var previous: MLXArray?
    var inputFrames = 0
    var generated = 0
    let clock = ContinuousClock()
    let started = clock.now
    while let data = try readExactly(frameBytes, from: input) {
      let frame = MLXArray(data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }, [1, height, width, 3])
      inputFrames += 1
      if let previous {
        for phase in 1..<factor {
          let out = try generator.interpolate(previous, frame, phase: Float(phase) / Float(factor))
          let values = out.asArray(Float.self)
          output.write(values.withUnsafeBytes { Data($0) })
          generated += 1
        }
      }
      previous = frame
    }
    let elapsed = started.duration(to: clock.now)
    let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
    let summary = Summary(inputFrames: inputFrames, generatedFrames: generated, seconds: seconds, precision: precision.rawValue)
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
