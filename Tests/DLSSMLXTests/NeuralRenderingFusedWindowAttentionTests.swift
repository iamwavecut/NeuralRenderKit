import Foundation
import MLX
import XCTest

@testable import DLSSMLX

/// The fused (window, head) attention core and the fused residual must match
/// the per-operation path bit for bit, including shifted origins and ragged
/// edges, for every multi-head window family.
final class NeuralRenderingFusedWindowAttentionTests: XCTestCase {
  private func synthetic(headCount: Int, height: Int, width: Int, origin: NeuralRenderingWindowOrigin, seed: UInt64) -> (fused: MLXArray, reference: MLXArray) {
    let channels = headCount * 32
    MLXRandom.seed(seed)
    let input = (MLXRandom.normal([1, height, width, channels]) * 0.5).asType(.float16)
    let qkvWeight = (MLXRandom.normal([channels, channels * 3]) * 0.08).asType(.float16)
    let projectionWeight = (MLXRandom.normal([channels, channels]) * 0.08).asType(.float16)
    let scale = (MLXRandom.uniform(low: 0.5, high: 2.0, [headCount])).asType(.float16)
    let bias = (MLXRandom.normal([headCount, 64, 64]) * 2.0).asType(.float16)
    let cosine = MLXRandom.uniform(low: 0.5, high: 1.0, [channels]).asType(.float16)
    eval(input, qkvWeight, projectionWeight, scale, bias, cosine)
    func run(_ fused: Bool) -> MLXArray {
      let attended = NeuralRenderingTransformerOperations.windowAttention(
        input, qkvWeight: qkvWeight, attentionScale: scale, attentionBias: bias, projectionWeight: projectionWeight,
        headCount: headCount, windowSize: 8, windowOrigin: origin, preciseSoftmax: true, fusedAttention: fused)
      let residual = fused
        ? NeuralRenderingFusedWindowAttention.cosineResidual(skip: input, branch: attended, cosine: cosine, publish: true)
        : NeuralRenderingTransformerOperations.e4m3RoundTrip(
          NeuralRenderingTransformerOperations.cosineResidual(skip: input, branch: attended, cosine: cosine))
      eval(residual)
      return residual
    }
    return (run(true), run(false))
  }

  func testFusedAttentionMatchesPerOperationPath() {
    for (headCount, height, width) in [(2, 24, 40), (4, 24, 40), (8, 16, 24), (16, 16, 24), (2, 19, 37), (4, 13, 27), (8, 11, 21), (16, 9, 13)] {
      for origin in [NeuralRenderingWindowOrigin.zero, NeuralRenderingWindowOrigin(y: -4, x: -4), NeuralRenderingWindowOrigin(y: 0, x: -4), NeuralRenderingWindowOrigin(y: -4, x: 0)] {
        let (fused, reference) = synthetic(headCount: headCount, height: height, width: width, origin: origin, seed: UInt64(headCount * 1000 + height * 10 + width))
        let delta = abs(fused.asType(.float32) - reference.asType(.float32))
        let maxDelta = delta.max().item(Float.self)
        let meanDelta = delta.mean().item(Float.self)
        print("fused-window-attention \(headCount)h \(height)x\(width) origin (\(origin.x),\(origin.y)): max |Δ| \(maxDelta) mean |Δ| \(meanDelta)")
        XCTAssertEqual(maxDelta, 0, "fused attention must be bit-exact for \(headCount) heads at \(height)x\(width) origin \(origin)")
      }
    }
  }

  /// The dense arrangement of the branched feed-forward must agree with the
  /// per-group reference up to accumulation order: the E4M3 publication makes
  /// individual elements flip by one E4M3 step, so the gate is on the mean.
  func testDenseBranchedFeedForwardMatchesReferenceArrangement() {
    for groupCount in [2, 4, 8] {
      let channels = groupCount * 32
      MLXRandom.seed(UInt64(11 + groupCount))
      let input = (MLXRandom.normal([1, 16, 24, channels]) * 0.5).asType(.float16)
      let expansion = (MLXRandom.normal([groupCount, 4, groupCount, 32, 32]) * 0.1).asType(.float16)
      let branchProjection = (MLXRandom.normal([groupCount, 4, 32, 32]) * 0.1).asType(.float16)
      let outputProjection = (MLXRandom.normal([channels, channels]) * 0.08).asType(.float16)
      eval(input, expansion, branchProjection, outputProjection)
      let reference = NeuralRenderingTransformerOperations.branchedFeedForward(
        input, expansionWeight: expansion, branchProjectionWeight: branchProjection, outputProjectionWeight: outputProjection)
      let dense = NeuralRenderingTransformerOperations.denseBranchedFeedForward(
        input,
        denseExpansionWeight: NeuralRenderingTransformerOperations.denseExpansionWeight(expansion),
        groupedProjectionWeight: NeuralRenderingTransformerOperations.groupedProjectionWeight(branchProjection),
        outputProjectionWeight: outputProjection)
      eval(reference, dense)
      let delta = abs(reference.asType(.float32) - dense.asType(.float32))
      let meanDelta = delta.mean().item(Float.self)
      let meanMagnitude = abs(reference.asType(.float32)).mean().item(Float.self)
      print("dense-ffn \(groupCount) groups: mean |Δ| \(meanDelta) vs mean |ref| \(meanMagnitude), max |Δ| \(delta.max().item(Float.self))")
      XCTAssertLessThan(meanDelta, 0.02 * meanMagnitude, "dense feed-forward diverges from the reference arrangement for \(groupCount) groups")
    }
  }

  /// Opt-in stress loop: `MLXDLSS_ATTENTION_STRESS=ITERATIONS` runs the fused
  /// attention core repeatedly at the 1080p family shapes to expose faults.
  func testFusedAttentionStressWhenConfigured() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let iterationsText = environment["MLXDLSS_ATTENTION_STRESS"], let iterations = Int(iterationsText) else {
      throw XCTSkip("set MLXDLSS_ATTENTION_STRESS=ITERATIONS")
    }
    let shapes: [(Int, Int, Int)] = [(2, 272, 480), (4, 136, 240), (8, 68, 120), (16, 36, 60)]
    let origins = [NeuralRenderingWindowOrigin.zero, NeuralRenderingWindowOrigin(y: -4, x: -4), NeuralRenderingWindowOrigin(y: 0, x: -4), NeuralRenderingWindowOrigin(y: -4, x: 0)]
    for (headCount, height, width) in shapes {
      let channels = headCount * 32
      MLXRandom.seed(UInt64(headCount))
      let qkvWeight = (MLXRandom.normal([channels, channels * 3]) * 0.08).asType(.float16)
      let projectionWeight = (MLXRandom.normal([channels, channels]) * 0.08).asType(.float16)
      let scale = (MLXRandom.uniform(low: 0.5, high: 2.0, [headCount])).asType(.float16)
      let bias = (MLXRandom.normal([headCount, 64, 64]) * 2.0).asType(.float16)
      eval(qkvWeight, projectionWeight, scale, bias)
      var checksum: Float = 0
      for iteration in 0..<iterations {
        let input = (MLXRandom.normal([1, height, width, channels]) * 0.5).asType(.float16)
        let origin = origins[iteration % origins.count]
        let out = NeuralRenderingTransformerOperations.windowAttention(
          input, qkvWeight: qkvWeight, attentionScale: scale, attentionBias: bias, projectionWeight: projectionWeight,
          headCount: headCount, windowSize: 8, windowOrigin: origin, preciseSoftmax: true, fusedAttention: true)
        checksum += abs(out).asType(.float32).mean().item(Float.self)
        if environment["MLXDLSS_ATTENTION_STRESS_CLEAR"] != nil { GPU.clearCache() }
      }
      print("attention-stress \(headCount)h \(height)x\(width): \(iterations) iterations ok, checksum \(checksum)")
    }
  }

  /// Opt-in: `MLXDLSS_ATTENTION_LARGE=1` compares both paths on synthetic weights
  /// at the 1080p family shapes and counts non-finite outputs.
  func testFusedAttentionLargeSyntheticWhenConfigured() throws {
    guard ProcessInfo.processInfo.environment["MLXDLSS_ATTENTION_LARGE"] != nil else { throw XCTSkip("set MLXDLSS_ATTENTION_LARGE=1") }
    for (headCount, height, width) in [(2, 272, 480), (4, 136, 240), (8, 68, 120), (16, 36, 60)] {
      for origin in [NeuralRenderingWindowOrigin.zero, NeuralRenderingWindowOrigin(y: -4, x: -4)] {
        let (fused, reference) = synthetic(headCount: headCount, height: height, width: width, origin: origin, seed: 99)
        let f32 = fused.asType(.float32), r32 = reference.asType(.float32)
        let finiteF = isFinite(f32).asType(.int32).sum().item(Int32.self), finiteR = isFinite(r32).asType(.int32).sum().item(Int32.self)
        let delta = abs(f32 - r32)
        print("attention-large \(headCount)h \(height)x\(width) origin (\(origin.x),\(origin.y)): non-finite fused \(f32.size - Int(finiteF)) reference \(f32.size - Int(finiteR)) | max |Δ| \(delta.max().item(Float.self)) mean |Δ| \(delta.mean().item(Float.self)) | mean |ref| \(abs(r32).mean().item(Float.self))")
      }
    }
  }

  func testFusedAttentionOnExternalWeightsWhenConfigured() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let weightsPath = environment["MLXDLSS_LOGICAL_WEIGHTS"], let spec = environment["MLXDLSS_FUSED_ATTENTION_TIMING"] else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS and MLXDLSS_FUSED_ATTENTION_TIMING=BLOCK:HEADS:HxW[;...]")
    }
    let arrays = try loadArrays(url: URL(fileURLWithPath: weightsPath), stream: .cpu)
    let weights = ValidatedWeights(arrays: arrays).cast(to: .float16)
    for entry in spec.split(separator: ";") {
      let fields = entry.split(separator: ":").map(String.init)
      let block = Int(fields[0])!, headCount = Int(fields[1])!
      let shape = fields[2].split(separator: "x").compactMap { Int($0) }
      let channels = headCount * 32
      // Window families keep attention in layer0; the split families in layer2/layer3.
      let split = headCount == 16
      let attentionPrefix = split ? "block\(block).layer2" : "block\(block).layer0"
      let projectionPrefix = split ? "block\(block).layer3" : "block\(block).layer0"
      let qkvWeight = try weights.required("\(attentionPrefix).qkv_weight")
      let projectionWeight = try weights.required("\(projectionPrefix).projection_weight")
      let scale = try weights.required("\(attentionPrefix).attn_scale")
      let storedBias = try weights.required("\(attentionPrefix).attn_bias")
      let bias = NeuralRenderingAttentionBiasLayout.usesFragmentSwizzle(blockIndex: block, headCount: headCount)
        ? NeuralRenderingAttentionBiasLayout.recoverFragmentSwizzle(storedBias) : storedBias
      let origin = NeuralRenderingGraphContract.windowOrigin(for: block)
      MLXRandom.seed(3)
      let input = (MLXRandom.normal([1, shape[0], shape[1], channels]) * 0.5).asType(.float16)
      eval(input)
      var outputs: [MLXArray] = []
      var times: [Double] = []
      for fused in [true, false] {
        var best = Double.infinity
        var out = input
        for _ in 0..<3 {
          let clock = ContinuousClock(); let start = clock.now
          out = Device.withDefaultDevice(.gpu) {
            NeuralRenderingTransformerOperations.windowAttention(
              input, qkvWeight: qkvWeight, attentionScale: scale, attentionBias: bias, projectionWeight: projectionWeight,
              headCount: headCount, windowSize: 8, windowOrigin: origin, preciseSoftmax: true, fusedAttention: fused)
          }
          eval(out)
          let d = start.duration(to: clock.now)
          best = min(best, Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15)
        }
        outputs.append(out); times.append(best)
      }
      let maxDelta = abs(outputs[0].asType(.float32) - outputs[1].asType(.float32)).max().item(Float.self)
      print("fused-window-attention external block \(block) \(headCount)h \(shape[0])x\(shape[1]): max |Δ| \(maxDelta) | fused \(String(format: "%.2f", times[0])) ms reference \(String(format: "%.2f", times[1])) ms")
      XCTAssertEqual(maxDelta, 0)
    }
  }
}
