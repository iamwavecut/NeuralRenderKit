import MLX
import XCTest

@testable import NeuralRenderMLX

final class NeuralRenderingAttentionBiasLayoutTests: XCTestCase {
  func testSourceIndexIsAPermutationOfAllFourThousandNinetySixEntries() {
    let indices = NeuralRenderingAttentionBiasLayout.fragmentSwizzleIndices
    XCTAssertEqual(indices.count, 4096)
    XCTAssertEqual(Set(indices).count, 4096)
    XCTAssertEqual(indices.min(), 0)
    XCTAssertEqual(indices.max(), 4095)
  }

  func testSourceIndexPlacesEachTokenBitAtItsFragmentPosition() {
    // (query, key) -> stored offset, one token bit at a time.
    let expectations: [(query: Int, key: Int, source: Int)] = [
      (0, 0, 0),
      (0, 1, 1 << 0),  // key x bit 0
      (16, 0, 1 << 1),  // query y bit 1
      (0, 16, 1 << 2),  // key y bit 1
      (0, 2, 1 << 3),  // key x bit 1
      (0, 8, 1 << 4),  // key y bit 0
      (1, 0, 1 << 5),  // query x bit 0
      (2, 0, 1 << 6),  // query x bit 1
      (8, 0, 1 << 7),  // query y bit 0
      (0, 4, 1 << 8),  // key x bit 2
      (0, 32, 1 << 9),  // key y bit 2
      (4, 0, 1 << 10),  // query x bit 2
      (32, 0, 1 << 11),  // query y bit 2
      (63, 63, 4095),
    ]
    for expectation in expectations {
      XCTAssertEqual(
        NeuralRenderingAttentionBiasLayout.sourceIndex(
          query: expectation.query,
          key: expectation.key
        ),
        expectation.source,
        "query \(expectation.query) key \(expectation.key)"
      )
    }
  }

  func testRecoverFragmentSwizzleGathersStoredEntries() {
    let stored = MLXArray((0..<(2 * 4096)).map { Float($0) }, [2, 64, 64])

    let recovered = NeuralRenderingAttentionBiasLayout.recoverFragmentSwizzle(stored)
    eval(recovered)

    XCTAssertEqual(recovered.shape, [2, 64, 64])
    let values = recovered.asArray(Float.self)
    for (query, key) in [(0, 1), (9, 3), (37, 58), (63, 63)] {
      let source = NeuralRenderingAttentionBiasLayout.sourceIndex(query: query, key: key)
      XCTAssertEqual(values[query * 64 + key], Float(source))
      XCTAssertEqual(values[4096 + query * 64 + key], Float(4096 + source))
    }
  }

  func testSingleHeadAndSplitBlocksUseTheSwizzle() {
    for block in [0, 1, 2, 3, 4, 66, 67, 68, 69, 70] {
      XCTAssertTrue(
        NeuralRenderingAttentionBiasLayout.usesFragmentSwizzle(blockIndex: block, headCount: 1),
        "block \(block)"
      )
    }
    for block in [23, 27, 30, 40, 45, 47] {
      XCTAssertTrue(
        NeuralRenderingAttentionBiasLayout.usesFragmentSwizzle(blockIndex: block, headCount: 16),
        "block \(block)"
      )
    }
    for (block, heads) in [(5, 2), (9, 4), (15, 8), (49, 8), (57, 4), (63, 2)] {
      XCTAssertFalse(
        NeuralRenderingAttentionBiasLayout.usesFragmentSwizzle(
          blockIndex: block,
          headCount: heads
        ),
        "block \(block)"
      )
    }
  }
}
