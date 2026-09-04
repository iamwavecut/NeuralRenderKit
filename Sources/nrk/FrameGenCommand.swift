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
    if repeatCount > 1, ProcessInfo.processInfo.environment["NRK_FG_BENCH_STAGES"] == "1" {
      // stage timing: synthesis alone, then the composition alone, after a warm-up
      let export = generator.synthesize(aArray, bArray, phase: phases[0]); eval(export)
      var t0 = clock.now
      for _ in 0..<repeatCount { eval(generator.synthesize(aArray, bArray, phase: phases[0])) }
      var d = t0.duration(to: clock.now)
      let synth = (Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18) / Double(repeatCount)
      let coarse = concatenated([export[0..., 0..., 0..., 0..<4], sigmoid(export[0..., 0..., 0..., 4..<5])], axis: 3).asType(.float32)
      let a32 = aArray.asType(.float32), b32 = bArray.asType(.float32); eval(coarse, a32, b32)
      t0 = clock.now
      for _ in 0..<repeatCount { eval(FrameGenerator.compose(a32, b32, coarse: coarse)) }
      d = t0.duration(to: clock.now)
      let comp = (Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18) / Double(repeatCount)
      FileHandle.standardError.write(Data(String(format: "stages: synthesize %.2f ms, compose %.2f ms\n", synth * 1000, comp * 1000).utf8))
      // per-op overhead probe: chains of tiny kernels
      do {
        var x = MLXArray.zeros([1, 16, 16, 8], dtype: .float16)
        eval(x)
        for (name, op) in [("mlx add", { (v: MLXArray) -> MLXArray in v + 1 }),
                           ("custom up2+pool", { (v: MLXArray) -> MLXArray in FrameGenerator.meanPool2(FrameGenerator.upsample2(v)) }),
                           ("mlx conv 8->8", { (v: MLXArray) -> MLXArray in conv2d(v, MLXArray.zeros([8, 3, 3, 8], dtype: .float16), stride: 1, padding: 1) })] {
          var y = x
          for _ in 0..<20 { y = op(y) }
          eval(y)
          let p0 = clock.now
          for _ in 0..<5 {
            y = x
            for _ in 0..<100 { y = op(y) }
            eval(y)
          }
          let dp = p0.duration(to: clock.now)
          let perOp = (Double(dp.components.seconds) + Double(dp.components.attoseconds) / 1e18) / 500 * 1e6
          FileHandle.standardError.write(Data(String(format: "  probe %@: %.1f us per op\n", name as NSString, perOp).utf8))
        }
        x = x + 0
      }
      var previous = 0.0
      for stage in 0...3 {
        eval(generator.synthesizeUpTo(stage, aArray, bArray, phase: phases[0]))
        let s0 = clock.now
        for _ in 0..<repeatCount { eval(generator.synthesizeUpTo(stage, aArray, bArray, phase: phases[0])) }
        let ds = s0.duration(to: clock.now)
        let ms = (Double(ds.components.seconds) + Double(ds.components.attoseconds) / 1e18) / Double(repeatCount) * 1000
        FileHandle.standardError.write(Data(String(format: "  up to stage %d: %.2f ms (+%.2f)\n", stage, ms, ms - previous).utf8))
        previous = ms
      }
    }
    // all phases of the pair in one batch (N = phases.count samples)
    if repeatCount > 1 { _ = try generator.interpolate(aArray, bArray, phases: phases) }  // warm-up: kernel compilation
    let started = clock.now
    var frames = try generator.interpolate(aArray, bArray, phases: phases)
    for _ in 1..<repeatCount { frames = try generator.interpolate(aArray, bArray, phases: phases) }
    let elapsed = started.duration(to: clock.now)
    seconds = (Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18) / Double(repeatCount)
    for k in 0..<phases.count {
      let frame = frames[k..<(k + 1)]
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
