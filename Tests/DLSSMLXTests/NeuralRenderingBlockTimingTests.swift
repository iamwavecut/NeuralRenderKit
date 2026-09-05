import Foundation
import MLX
import XCTest

@testable import DLSSMLX

/// Opt-in piece-by-piece timing of the fused float16 path on external weights:
/// `MLXDLSS_LOGICAL_WEIGHTS` and `MLXDLSS_BLOCK_TIMING=HEIGHTxWIDTH`. Diagnostics only.
final class NeuralRenderingBlockTimingTests: XCTestCase {
  func testExternalBlockTimingWhenConfigured() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let weightsPath = environment["MLXDLSS_LOGICAL_WEIGHTS"], let shape = environment["MLXDLSS_BLOCK_TIMING"] else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS and MLXDLSS_BLOCK_TIMING=HEIGHTxWIDTH")
    }
    let parts = shape.split(separator: "x").compactMap { Int($0) }
    let height = parts[0], width = parts[1]
    let arrays = try loadArrays(url: URL(fileURLWithPath: weightsPath), stream: .cpu)
    let weights = ValidatedWeights(arrays: arrays).cast(to: .float16)
    typealias Ops = NeuralRenderingTransformerOperations
    let pre = try NeuralRenderingPreBlock(weights: weights, compileBlocks: true)
    let seq1 = try NeuralRenderingWindowSequence(weights: weights, blockIndices: 1...3, channels: 32, hiddenChannels: 128, headCount: 1, compileSequence: true)
    let ds4 = try NeuralRenderingDownsampleBlock(weights: weights, blockIndex: 4, channels: 32, hiddenChannels: 128, headCount: 1, compileBlocks: true)
    let seq5 = try NeuralRenderingWindowSequence(weights: weights, blockIndices: 5...7, channels: 64, hiddenChannels: 224, headCount: 2, compileSequence: true)
    let ds8 = try NeuralRenderingDownsampleBlock(weights: weights, blockIndex: 8, channels: 64, hiddenChannels: 224, headCount: 2, compileBlocks: true)
    let seq9 = try NeuralRenderingWindowSequence(weights: weights, blockIndices: 9...13, channels: 128, hiddenChannels: 384, headCount: 4, compileSequence: true)
    let ds14 = try NeuralRenderingDownsampleBlock(weights: weights, blockIndex: 14, channels: 128, hiddenChannels: 384, headCount: 4, compileBlocks: true)
    let seq15 = try NeuralRenderingWindowSequence(weights: weights, blockIndices: 15...21, channels: 256, hiddenChannels: 704, headCount: 8, compileSequence: true)
    let ds22 = try NeuralRenderingDownsampleBlock(weights: weights, blockIndex: 22, channels: 256, hiddenChannels: 704, headCount: 8, compileBlocks: true)
    let splitEncoder = try NeuralRenderingSplitEncoderStage(weights: weights, preciseAttention: true, fusedFeedForward: true)
    let globalStage = try NeuralRenderingGlobalStage(weights: weights, quantizeFFN: false, preciseAttention: true)
    let decoderInput = try NeuralRenderingDecoderInput(weights: weights)
    let split40 = try (40...47).map { try NeuralRenderingSplitWindowBlock(weights: weights, blockIndex: $0, preciseAttention: true, fusedFeedForward: true) }
    let up48 = try NeuralRenderingUpsampleBlock(weights: weights, blockIndex: 48, channels: 256, hiddenChannels: 704, headCount: 8, compileBlocks: true)
    let seq49 = try NeuralRenderingWindowSequence(weights: weights, blockIndices: 49...55, channels: 256, hiddenChannels: 704, headCount: 8, compileSequence: true)
    let up56 = try NeuralRenderingUpsampleBlock(weights: weights, blockIndex: 56, channels: 128, hiddenChannels: 384, headCount: 4, compileBlocks: true)
    let seq57 = try NeuralRenderingWindowSequence(weights: weights, blockIndices: 57...61, channels: 128, hiddenChannels: 384, headCount: 4, compileSequence: true)
    let up62 = try NeuralRenderingUpsampleBlock(weights: weights, blockIndex: 62, channels: 64, hiddenChannels: 224, headCount: 2, compileBlocks: true)
    let seq63 = try NeuralRenderingWindowSequence(weights: weights, blockIndices: 63...65, channels: 64, hiddenChannels: 224, headCount: 2, compileSequence: true)
    let up66 = try NeuralRenderingUpsampleBlock(weights: weights, blockIndex: 66, channels: 32, hiddenChannels: 128, headCount: 1, compileBlocks: true)
    let seq67 = try NeuralRenderingWindowSequence(weights: weights, blockIndices: 67...69, channels: 32, hiddenChannels: 128, headCount: 1, compileSequence: true)
    let post = try NeuralRenderingPostBlock(weights: weights, compileBlocks: true)
    let input = MLXArray((0..<(height * width * 16)).map { sin(Float($0) * 0.001) }, [1, height, width, 16]).asType(.float16)
    eval(input)
    var total = 0.0
    func best(_ label: String, _ body: () -> [MLXArray]) -> [MLXArray] {
      var bestMs = Double.infinity
      var outputs: [MLXArray] = []
      for _ in 0..<3 {
        let clock = ContinuousClock(); let start = clock.now
        outputs = Device.withDefaultDevice(.gpu) { body() }
        eval(outputs)
        let d = start.duration(to: clock.now)
        bestMs = min(bestMs, Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15)
      }
      total += bestMs
      print("block-timing \(label): \(String(format: "%.1f", bestMs)) ms  shape \(outputs[0].shape)")
      return outputs
    }
    let projected = best("block 0 adapter matmul 16->32 (full res)") { [pre.project(input)] }
    let block0 = best("block 0 fused window block (full res)") { [pre.transform(projected[0])] }
    let fullSkip = best("block 0 e4m3 full-res skip") { [Ops.e4m3RoundTrip(block0[0])] }
    let pooled = best("block 0 avg-pool 2x2 + e4m3") { [Ops.e4m3RoundTrip(Ops.averagePool2(block0[0]))] }
    let s1 = best("blocks 1-3 (1h, half res)") { [seq1(pooled[0])] }
    let d4 = best("downsample 4") { [ds4(s1[0])] }
    let s5 = best("blocks 5-7 (2h)") { [seq5(d4[0])] }
    let d8 = best("downsample 8") { [ds8(s5[0])] }
    let s9 = best("blocks 9-13 (4h)") { [seq9(d8[0])] }
    let d14 = best("downsample 14") { [ds14(s9[0])] }
    let s15 = best("blocks 15-21 (8h)") { [seq15(d14[0])] }
    let d22 = best("downsample 22") { [ds22(s15[0])] }
    let split = best("split encoder 23-30") { let o = splitEncoder(d22[0]); return [o.latent, o.skip] }
    let latent = best("global 31-38") { [globalStage(split[0])] }
    let d39 = best("decoder input 39") { [decoderInput(latent[0], skip: split[1])] }
    var x = d39
    for (i, block) in split40.enumerated() { let y = x; x = best("split window block \(40 + i)") { [block(y[0])] } }
    let u48 = best("upsample 48") { [up48(x[0], skip: s15[0])] }
    let s49 = best("blocks 49-55 (8h)") { [seq49(u48[0])] }
    let u56 = best("upsample 56") { [up56(s49[0], skip: s9[0])] }
    let s57 = best("blocks 57-61 (4h)") { [seq57(u56[0])] }
    let u62 = best("upsample 62") { [up62(s57[0], skip: s5[0])] }
    let s63 = best("blocks 63-65 (2h)") { [seq63(u62[0])] }
    let u66 = best("upsample 66") { [up66(s63[0], skip: s1[0])] }
    let s67 = best("blocks 67-69 (1h, half res)") { [seq67(u66[0])] }
    _ = best("post block 70 (upsample+merge+fused block+head)") { [post(s67[0], skip: fullSkip[0])] }
    print("block-timing total of pieces: \(String(format: "%.1f", total)) ms at \(height)x\(width)")
  }
}
