import Foundation
import MLX
import XCTest

@testable import NeuralRenderMLX

/// Opt-in stage-level timing of the recovered 71-block head on external weights.
///
/// Set `NRK_LOGICAL_WEIGHTS` to the logical safetensors and `NRK_STAGE_TIMING`
/// to `HEIGHTxWIDTH` (for example `1088x1920`) to print per-stage milliseconds
/// for the fused float16 path. The numbers are diagnostics for optimization
/// work, not a pass/fail gate.
final class NeuralRenderingStageTimingTests: XCTestCase {
  func testExternalStageTimingWhenConfigured() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let weightsPath = environment["NRK_LOGICAL_WEIGHTS"],
      let shape = environment["NRK_STAGE_TIMING"]
    else {
      throw XCTSkip("set NRK_LOGICAL_WEIGHTS and NRK_STAGE_TIMING=HEIGHTxWIDTH")
    }
    let parts = shape.split(separator: "x").compactMap { Int($0) }
    guard parts.count == 2, parts[0] > 0, parts[1] > 0 else {
      throw XCTSkip("NRK_STAGE_TIMING must look like 1088x1920")
    }
    let height = parts[0]
    let width = parts[1]
    let arrays = try loadArrays(url: URL(fileURLWithPath: weightsPath), stream: .cpu)
    let weights = ValidatedWeights(arrays: arrays).cast(to: .float16)
    let encoder = try NeuralRenderingEncoder(weights: weights, compileBlocks: true)
    let splitEncoder = try NeuralRenderingSplitEncoderStage(
      weights: weights,
      preciseAttention: true,
      fusedFeedForward: true
    )
    let globalStage = try NeuralRenderingGlobalStage(
      weights: weights,
      quantizeFFN: false,
      preciseAttention: true
    )
    let decoderInput = try NeuralRenderingDecoderInput(weights: weights)
    let decoder = try NeuralRenderingDecoder(weights: weights, compileBlocks: true)
    let post = try NeuralRenderingPostBlock(weights: weights, compileBlocks: true)
    let input = MLXArray(
      (0..<(height * width * 16)).map { sin(Float($0) * 0.001) },
      [1, height, width, 16]
    ).asType(.float16)

    let checksums = environment["NRK_STAGE_CHECKSUMS"] != nil
    func timed(_ label: String, _ body: () -> [MLXArray]) -> [MLXArray] {
      let clock = ContinuousClock()
      let start = clock.now
      let outputs = Device.withDefaultDevice(.gpu) { body() }
      eval(outputs)
      let duration = start.duration(to: clock.now)
      let milliseconds =
        Double(duration.components.seconds) * 1_000
        + Double(duration.components.attoseconds) / 1e15
      if checksums {
        let sums = outputs.map { abs($0.asType(.float32)).sum().item(Float.self) }
        print("stage-timing \(label): \(String(format: "%.1f", milliseconds)) ms checksum \(sums)")
      } else {
        print("stage-timing \(label): \(String(format: "%.1f", milliseconds)) ms")
      }
      return outputs
    }

    let iterations = Int(environment["NRK_STAGE_ITERATIONS"] ?? "2") ?? 2
    for iteration in 0..<iterations {
      if environment["NRK_STAGE_CLEAR"] != nil { GPU.clearCache() }
      print("stage-timing iteration \(iteration) shape \(height)x\(width)")
      let encoded = timed("encoder blocks 0-22") { [encoder(input).latent] }
      let encoderOutput = Device.withDefaultDevice(.gpu) { encoder(input) }
      eval(encoderOutput.latent, encoderOutput.fullResolutionSkip)
      eval(encoderOutput.skips)
      let split = timed("split encoder blocks 23-30") {
        let output = splitEncoder(encoded[0])
        return [output.latent, output.skip]
      }
      let latent = timed("global blocks 31-38") { [globalStage(split[0])] }
      let decoderStart = timed("decoder input block 39") {
        [decoderInput(latent[0], skip: split[1])]
      }
      let decoded = timed("decoder blocks 40-69") {
        [decoder(decoderStart[0], skips: encoderOutput.skips)]
      }
      _ = timed("post block 70") {
        [post(decoded[0], skip: encoderOutput.fullResolutionSkip)]
      }
      print("stage-timing iteration \(iteration) done")
      Memory.clearCache()
    }
  }
}
