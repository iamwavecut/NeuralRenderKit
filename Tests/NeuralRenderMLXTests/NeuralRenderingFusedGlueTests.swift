import Foundation
import MLX
import XCTest

@testable import NeuralRenderMLX

/// The single-pass glue kernels must reproduce their MLX operation chains bit for bit.
final class NeuralRenderingFusedGlueTests: XCTestCase {
  private func maxDelta(_ a: MLXArray, _ b: MLXArray) -> Float {
    abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
  }

  func testPublishAndPoolMatchesOperationChain() {
    typealias Ops = NeuralRenderingTransformerOperations
    for (height, width, channels) in [(16, 24, 32), (34, 60, 64), (68, 120, 256), (272, 480, 64)] {
      MLXRandom.seed(UInt64(height * width))
      let input = (MLXRandom.normal([1, height, width, channels]) * 2.0).asType(.float16)
      eval(input)
      let reference = (Ops.e4m3RoundTrip(input), Ops.e4m3RoundTrip(Ops.averagePool2(input)))
      let fused = NeuralRenderingFusedGlue.publishAndPool(input, emitFull: true)
      let pooledOnly = NeuralRenderingFusedGlue.publishAndPool(input, emitFull: false)
      eval(reference.0, reference.1, fused.full, fused.pooled, pooledOnly.pooled)
      XCTAssertEqual(maxDelta(fused.full, reference.0), 0, "full publication \(height)x\(width)x\(channels)")
      XCTAssertEqual(maxDelta(fused.pooled, reference.1), 0, "pooled publication \(height)x\(width)x\(channels)")
      XCTAssertEqual(maxDelta(pooledOnly.pooled, reference.1), 0, "pooled-only publication \(height)x\(width)x\(channels)")
    }
  }

  func testUpsampleMergeMatchesOperationChains() {
    typealias Ops = NeuralRenderingTransformerOperations
    for (lowHeight, lowWidth, height, width, channels) in [(8, 12, 16, 24, 32), (34, 60, 68, 120, 256), (17, 30, 33, 59, 64), (136, 240, 272, 480, 64)] {
      MLXRandom.seed(UInt64(height * 7 + width))
      let low = (MLXRandom.normal([1, lowHeight, lowWidth, channels]) * 2.0).asType(.float16)
      let skip = (MLXRandom.normal([1, height, width, channels]) * 2.0).asType(.float16)
      let sine = MLXRandom.uniform(low: -1.0, high: 1.0, [channels]).asType(.float16)
      let cosine = MLXRandom.uniform(low: -1.0, high: 1.0, [channels]).asType(.float16)
      eval(low, skip, sine, cosine)
      let upsampled = Ops.nearestUpsample2Crop(low, height: height, width: width)
      // upsample block: E4M3(up(low) + skip * sine)
      let referenceUp = Ops.e4m3RoundTrip(upsampled + skip * sine)
      let fusedUp = NeuralRenderingFusedGlue.upsampleMerge(low: low, skip: skip, lowScale: nil, skipScale: sine, publish: true)
      // decoder input: E4M3(decoderInputMerge)
      let referenceInput = Ops.e4m3RoundTrip(Ops.decoderInputMerge(low, skip: skip, skipSine: sine))
      // post merge: up(low) * sine + skip * cosine
      let referencePost = Ops.postMerge(upsampled, skip: skip, sine: sine, cosine: cosine)
      let fusedPost = NeuralRenderingFusedGlue.upsampleMerge(low: low, skip: skip, lowScale: sine, skipScale: cosine, publish: false)
      eval(referenceUp, fusedUp, referenceInput, referencePost, fusedPost)
      XCTAssertEqual(maxDelta(fusedUp, referenceUp), 0, "upsample merge \(height)x\(width)x\(channels)")
      XCTAssertEqual(maxDelta(fusedUp, referenceInput), 0, "decoder input merge \(height)x\(width)x\(channels)")
      XCTAssertEqual(maxDelta(fusedPost, referencePost), 0, "post merge \(height)x\(width)x\(channels)")
    }
  }
}
