import Foundation
import MLX
import XCTest

@testable import DLSSMLX

/// Opt-in cost split of one 32-channel single-head window block on external
/// weights: whole block, fused FFN alone, cosine attention on the partitioned
/// windows alone, and the E4M3 publication kernel alone.
///
/// Set `MLXDLSS_LOGICAL_WEIGHTS` and `MLXDLSS_WINDOW_TIMING=HEIGHTxWIDTH` (the
/// post-pool stem extent, for example `544x960` for a `1088x1920` network).
final class NeuralRenderingWindowBlockTimingTests: XCTestCase {
  func testExternalWindowBlockCostSplitWhenConfigured() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let weightsPath = environment["MLXDLSS_LOGICAL_WEIGHTS"],
      let shape = environment["MLXDLSS_WINDOW_TIMING"]
    else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS and MLXDLSS_WINDOW_TIMING=HEIGHTxWIDTH")
    }
    let parts = shape.split(separator: "x").compactMap { Int($0) }
    guard parts.count == 2, parts[0] > 0, parts[1] > 0,
      parts[0].isMultiple(of: 8), parts[1].isMultiple(of: 8)
    else {
      throw XCTSkip("MLXDLSS_WINDOW_TIMING must look like 544x960 with multiples of 8")
    }
    let height = parts[0]
    let width = parts[1]
    let arrays = try loadArrays(url: URL(fileURLWithPath: weightsPath), stream: .cpu)
    let weights = ValidatedWeights(arrays: arrays).cast(to: .float16)
    let block = try NeuralRenderingWindowBlock(
      weights: weights,
      blockIndex: 1,
      channels: 32,
      hiddenChannels: 128,
      headCount: 1,
      compileBlock: true
    )
    let prefix = "block1.layer0"
    let expansion = try weights.required("\(prefix).weight1")
    let projection = try weights.required("\(prefix).weight2")
    let qkv = try weights.required("\(prefix).qkv_weight")
    let scale = try weights.required("\(prefix).attn_scale")
    let bias = NeuralRenderingAttentionBiasLayout.recoverFragmentSwizzle(
      try weights.required("\(prefix).attn_bias")
    )
    let attentionProjection = try weights.required("\(prefix).projection_weight")
    let input = MLXArray(
      (0..<(height * width * 32)).map { sin(Float($0) * 0.001) },
      [1, height, width, 32]
    ).asType(.float16)
    let windows = input.reshaped([1, height / 8, 8, width / 8, 8, 32])
      .transposed(0, 1, 3, 2, 4, 5)
      .reshaped([height / 8 * width / 8, 64, 32])
    eval(input, windows)

    func timed(_ label: String, iterations: Int = 3, _ body: () -> MLXArray) {
      var best = Double.infinity
      for _ in 0..<iterations {
        let clock = ContinuousClock()
        let start = clock.now
        let output = Device.withDefaultDevice(.gpu) { body() }
        eval(output)
        let duration = start.duration(to: clock.now)
        let milliseconds =
          Double(duration.components.seconds) * 1_000
          + Double(duration.components.attoseconds) / 1e15
        best = min(best, milliseconds)
      }
      print("window-timing \(label): \(String(format: "%.2f", best)) ms (best of \(iterations))")
    }

    print("window-timing shape \(height)x\(width)x32 windows=\(windows.shape[0])")
    timed("full window block (fused)") { block(input) }
    timed("fused simple FFN only") {
      NeuralRenderingTransformerOperations.fusedSimpleFeedForward(
        input,
        expansionWeight: expansion,
        projectionWeight: projection
      )
    }
    timed("e4m3 round trip only") {
      NeuralRenderingTransformerOperations.e4m3RoundTrip(input)
    }
    timed("quadratic gate activation only") {
      NeuralRenderingTransformerOperations.quadraticGateActivation(input)
    }
    timed("window partition + reverse only") {
      windows.reshaped([1, height / 8, width / 8, 8, 8, 32])
        .transposed(0, 1, 3, 2, 4, 5)
        .reshaped([1, height, width, 32])
    }
    timed("cosine attention on windows only") {
      NeuralRenderingTransformerOperations.cosineAttention(
        windows,
        qkvWeight: qkv,
        attentionScale: scale,
        attentionBias: bias,
        projectionWeight: attentionProjection,
        headCount: 1,
        preciseSoftmax: true
      )
    }
    timed("plain matmul 32x96 (qkv) only") { matmul(windows, qkv) }
  }
}
