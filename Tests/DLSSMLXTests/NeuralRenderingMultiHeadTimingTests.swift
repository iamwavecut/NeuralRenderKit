import Foundation
import MLX
import XCTest

@testable import DLSSMLX

/// Opt-in cost split of a multi-head window block on external weights:
/// `MLXDLSS_LOGICAL_WEIGHTS` and `MLXDLSS_MULTIHEAD_TIMING=BLOCK:CHANNELS:HIDDEN:HEADS:HEIGHTxWIDTH` (e.g. `15:256:704:8:68x120`).
final class NeuralRenderingMultiHeadTimingTests: XCTestCase {
  func testMultiHeadBlockCostSplitWhenConfigured() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let weightsPath = environment["MLXDLSS_LOGICAL_WEIGHTS"], let spec = environment["MLXDLSS_MULTIHEAD_TIMING"] else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS and MLXDLSS_MULTIHEAD_TIMING=BLOCK:CHANNELS:HIDDEN:HEADS:HxW")
    }
    let fields = spec.split(separator: ":").map(String.init)
    let block = Int(fields[0])!, channels = Int(fields[1])!, hidden = Int(fields[2])!, heads = Int(fields[3])!
    let shape = fields[4].split(separator: "x").compactMap { Int($0) }
    let height = shape[0], width = shape[1]
    let arrays = try loadArrays(url: URL(fileURLWithPath: weightsPath), stream: .cpu)
    let weights = ValidatedWeights(arrays: arrays).cast(to: .float16)
    let prefix = "block\(block).layer0"
    let windowBlock = try NeuralRenderingWindowBlock(weights: weights, blockIndex: block, channels: channels, hiddenChannels: hidden, headCount: heads, compileBlock: true)
    let expansion = try weights.required("\(prefix).ffn_expand_weight")
    let branchProjection = try weights.required("\(prefix).ffn_branch_projection_weight")
    let outputProjection = try weights.required("\(prefix).ffn_output_projection_weight")
    let qkv = try weights.required("\(prefix).qkv_weight")
    let scale = try weights.required("\(prefix).attn_scale")
    let bias = try weights.required("\(prefix).attn_bias")
    let projection = try weights.required("\(prefix).projection_weight")
    let input = (MLXRandom.normal([1, height, width, channels]) * 0.5).asType(.float16)
    let windows = input.reshaped([1, height / 8, 8, width / 8, 8, channels]).transposed(0, 1, 3, 2, 4, 5).reshaped([height / 8 * width / 8, 64, channels])
    eval(input, windows)
    func timed(_ label: String, _ body: () -> MLXArray) {
      var best = Double.infinity
      for _ in 0..<3 {
        let clock = ContinuousClock(); let start = clock.now
        let out = Device.withDefaultDevice(.gpu) { body() }; eval(out)
        let d = start.duration(to: clock.now)
        best = min(best, Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15)
      }
      print("multihead-timing block \(block) \(channels)ch \(heads)h \(height)x\(width) \(label): \(String(format: "%.2f", best)) ms")
    }
    timed("full window block") { windowBlock(input) }
    timed("branched FFN fused") { NeuralRenderingTransformerOperations.fusedBranchedFeedForward(input, expansionWeight: expansion, branchProjectionWeight: branchProjection, outputProjectionWeight: outputProjection) }
    timed("branched FFN unfused") { NeuralRenderingTransformerOperations.branchedFeedForward(input, expansionWeight: expansion, branchProjectionWeight: branchProjection, outputProjectionWeight: outputProjection) }
    timed("cosine attention on windows") { NeuralRenderingTransformerOperations.cosineAttention(windows, qkvWeight: qkv, attentionScale: scale, attentionBias: bias, projectionWeight: projection, headCount: heads, preciseSoftmax: true) }
    timed("qkv matmul only") { matmul(windows, qkv) }
    timed("e4m3 round trip") { NeuralRenderingTransformerOperations.e4m3RoundTrip(input) }
  }
}
