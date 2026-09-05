import Foundation
import MLX
import XCTest

@testable import DLSSMLX

/// Parity and timing of the fused single-kernel window block against the
/// per-operation Metal path, on synthetic weights (always) and on external
/// weights (`MLXDLSS_LOGICAL_WEIGHTS`, optional `MLXDLSS_FUSED_TIMING=HEIGHTxWIDTH`).
final class NeuralRenderingFusedWindowBlockTests: XCTestCase {
  private func syntheticWeights(seed: UInt64 = 1) -> [String: MLXArray] {
    MLXRandom.seed(seed)
    let prefix = "block1.layer0"
    func normal(_ shape: [Int], scale: Float) -> MLXArray {
      (MLXRandom.normal(shape) * scale).asType(.float16)
    }
    return [
      "\(prefix).weight1": normal([32, 128], scale: 0.2),
      "\(prefix).weight2": normal([128, 32], scale: 0.2),
      "\(prefix).ffn_cos_skip": (MLXRandom.uniform(low: 0.5, high: 1.0, [32])).asType(.float16),
      "\(prefix).qkv_weight": normal([32, 96], scale: 0.2),
      "\(prefix).attn_scale": MLXArray([Float(3.0)]).asType(.float16),
      "\(prefix).attn_bias": normal([1, 64, 64], scale: 0.3),
      "\(prefix).projection_weight": normal([32, 32], scale: 0.2),
      "\(prefix).attn_cos_skip": (MLXRandom.uniform(low: 0.5, high: 1.0, [32])).asType(.float16),
    ]
  }

  private func compare(
    weights: [String: MLXArray], blockIndex: Int, height: Int, width: Int, publish: Bool, label: String
  ) throws -> (maxAbs: Float, meanAbs: Float, referenceMs: Double, fusedMs: Double) {
    let prefix = "block\(blockIndex).layer0"
    let origin = NeuralRenderingGraphContract.windowOrigin(for: blockIndex)
    let bias = NeuralRenderingAttentionBiasLayout.recoverFragmentSwizzle(weights["\(prefix).attn_bias"]!)
    MLXRandom.seed(7)
    let input = (MLXRandom.normal([1, height, width, 32]) * 0.5).asType(.float16)
    eval(input)
    func reference() -> MLXArray {
      let out = NeuralRenderingTransformerOperations.windowBlock(
        input,
        expansionWeight: weights["\(prefix).weight1"]!,
        feedForwardProjectionWeight: weights["\(prefix).weight2"]!,
        feedForwardCosine: weights["\(prefix).ffn_cos_skip"]!,
        qkvWeight: weights["\(prefix).qkv_weight"]!,
        attentionScale: weights["\(prefix).attn_scale"]!,
        attentionBias: bias,
        attentionProjectionWeight: weights["\(prefix).projection_weight"]!,
        attentionCosine: weights["\(prefix).attn_cos_skip"]!,
        headCount: 1,
        windowSize: 8,
        windowOrigin: origin,
        preciseSoftmax: true,
        fusedFeedForward: true
      )
      return publish ? NeuralRenderingTransformerOperations.e4m3RoundTrip(out) : out
    }
    func fused() -> MLXArray {
      NeuralRenderingFusedWindowBlock.apply(
        input,
        expansionWeight: weights["\(prefix).weight1"]!,
        feedForwardProjectionWeight: weights["\(prefix).weight2"]!,
        feedForwardCosine: weights["\(prefix).ffn_cos_skip"]!,
        qkvWeight: weights["\(prefix).qkv_weight"]!,
        attentionScale: weights["\(prefix).attn_scale"]!,
        attentionBias: bias,
        attentionProjectionWeight: weights["\(prefix).projection_weight"]!,
        attentionCosine: weights["\(prefix).attn_cos_skip"]!,
        windowOrigin: origin,
        publish: publish
      )
    }
    func timed(_ body: () -> MLXArray) -> (MLXArray, Double) {
      var best = Double.infinity
      var result: MLXArray = MLXArray(0)
      for _ in 0..<3 {
        let clock = ContinuousClock()
        let start = clock.now
        result = Device.withDefaultDevice(.gpu) { body() }
        eval(result)
        let d = start.duration(to: clock.now)
        best = min(best, Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15)
      }
      return (result, best)
    }
    let (expected, referenceMs) = timed(reference)
    let (actual, fusedMs) = timed(fused)
    let difference = abs(actual.asType(.float32) - expected.asType(.float32))
    let maxAbs = difference.max().item(Float.self)
    let meanAbs = difference.mean().item(Float.self)
    print("fused-window-block \(label) \(height)x\(width) block \(blockIndex) origin (\(origin.y),\(origin.x)) publish \(publish): max |Δ| \(maxAbs) mean |Δ| \(meanAbs) | reference \(String(format: "%.2f", referenceMs)) ms fused \(String(format: "%.2f", fusedMs)) ms")
    return (maxAbs, meanAbs, referenceMs, fusedMs)
  }

  func testFusedBlockMatchesPerOperationPathOnSyntheticWeights() throws {
    let weights = syntheticWeights()
    for (height, width) in [(64, 96), (72, 104)] {
      for publish in [true, false] {
        let result = try compare(weights: weights, blockIndex: 1, height: height, width: width, publish: publish, label: "synthetic")
        XCTAssertLessThan(result.meanAbs, 0.002, "mean difference too large at \(height)x\(width)")
        XCTAssertLessThan(result.maxAbs, 0.25, "max difference too large at \(height)x\(width)")
      }
    }
  }

  func testShiftedWindowOriginMatchesOnSyntheticWeights() throws {
    var weights = syntheticWeights()
    for block in [2, 3, 4] where NeuralRenderingGraphContract.windowOrigin(for: block) != .zero {
      for key in Array(weights.keys) where key.hasPrefix("block1.") {
        weights[key.replacingOccurrences(of: "block1.", with: "block\(block).")] = weights[key]
      }
      let result = try compare(weights: weights, blockIndex: block, height: 72, width: 104, publish: true, label: "shifted")
      XCTAssertLessThan(result.meanAbs, 0.002)
    }
  }

  func testExternalWeightsParityAndTimingWhenConfigured() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let weightsPath = environment["MLXDLSS_LOGICAL_WEIGHTS"] else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS (and MLXDLSS_FUSED_TIMING=HEIGHTxWIDTH)")
    }
    let shape = environment["MLXDLSS_FUSED_TIMING"] ?? "544x960"
    let parts = shape.split(separator: "x").compactMap { Int($0) }
    let arrays = try loadArrays(url: URL(fileURLWithPath: weightsPath), stream: .cpu)
    let weights = ValidatedWeights(arrays: arrays).cast(to: .float16)
    var dict: [String: MLXArray] = [:]
    for block in [1, 2, 3, 4, 70] {
      let prefix = "block\(block).layer0"
      for name in ["weight1", "weight2", "ffn_cos_skip", "qkv_weight", "attn_scale", "attn_bias", "projection_weight", "attn_cos_skip"] {
        dict["\(prefix).\(name)"] = try weights.required("\(prefix).\(name)")
      }
    }
    for block in [1, 2, 3, 4, 70] {
      let result = try compare(weights: dict, blockIndex: block, height: parts[0], width: parts[1], publish: block != 70, label: "external")
      XCTAssertLessThan(result.meanAbs, 0.003)
    }
  }
}
