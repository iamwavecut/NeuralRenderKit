import Foundation
import MLX
import XCTest

@testable import DLSSMLX

/// Opt-in per-family timing of the encoder on external weights:
/// `MLXDLSS_LOGICAL_WEIGHTS` and `MLXDLSS_FAMILY_TIMING=HEIGHTxWIDTH`.
final class NeuralRenderingFamilyTimingTests: XCTestCase {
  func testEncoderFamiliesWhenConfigured() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let weightsPath = environment["MLXDLSS_LOGICAL_WEIGHTS"], let shape = environment["MLXDLSS_FAMILY_TIMING"] else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS and MLXDLSS_FAMILY_TIMING=HEIGHTxWIDTH")
    }
    let parts = shape.split(separator: "x").compactMap { Int($0) }
    let height = parts[0], width = parts[1]
    let arrays = try loadArrays(url: URL(fileURLWithPath: weightsPath), stream: .cpu)
    let weights = ValidatedWeights(arrays: arrays).cast(to: .float16)
    let stem = try NeuralRenderingEncoderStem(weights: weights, compileBlocks: true)
    let second = try NeuralRenderingEncoderStage(weights: weights, regularBlocks: 5...7, transitionBlock: 8, channels: 64, hiddenChannels: 224, headCount: 2, compileBlocks: true)
    let third = try NeuralRenderingEncoderStage(weights: weights, regularBlocks: 9...13, transitionBlock: 14, channels: 128, hiddenChannels: 384, headCount: 4, compileBlocks: true)
    let fourth = try NeuralRenderingEncoderStage(weights: weights, regularBlocks: 15...21, transitionBlock: 22, channels: 256, hiddenChannels: 704, headCount: 8, compileBlocks: true)
    let input = MLXArray((0..<(height * width * 16)).map { sin(Float($0) * 0.001) }, [1, height, width, 16]).asType(.float16)
    eval(input)
    func best(_ label: String, iterations: Int = 3, _ body: () -> [MLXArray]) -> [MLXArray] {
      var bestMs = Double.infinity
      var outputs: [MLXArray] = []
      for _ in 0..<iterations {
        let clock = ContinuousClock(); let start = clock.now
        outputs = Device.withDefaultDevice(.gpu) { body() }
        eval(outputs)
        let d = start.duration(to: clock.now)
        bestMs = min(bestMs, Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15)
      }
      print("family-timing \(label): \(String(format: "%.1f", bestMs)) ms (best of \(iterations))")
      return outputs
    }
    let stemOut = best("stem block 0 (pre + pool)") { let o = stem(input); return [o.fullResolutionSkip, o.skip] }
    // stem.skip is the half-resolution tensor after blocks 1-3 and the block-4 downsample happens in the first stage;
    // time blocks 1-3 and the transition explicitly through the first stage pieces
    let firstStage = try NeuralRenderingFirstEncoderStage(weights: weights, compileBlocks: true)
    let firstOut = best("first stage: blocks 0-3 + downsample 4 (full+half res)") { let o = firstStage(input); return [o.skip, o.downsampled] }
    _ = stemOut
    let secondOut = best("second stage: blocks 5-7 + downsample 8 (quarter res, 64ch, 2 heads)") { let o = second(firstOut[1]); return [o.skip, o.downsampled] }
    let thirdOut = best("third stage: blocks 9-13 + downsample 14 (1/8, 128ch, 4 heads)") { let o = third(secondOut[1]); return [o.skip, o.downsampled] }
    _ = best("fourth stage: blocks 15-21 + downsample 22 (1/16, 256ch, 8 heads)") { let o = fourth(thirdOut[1]); return [o.skip, o.downsampled] }
  }
}
