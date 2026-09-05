import MLX

/// Physical-to-logical layout recovery for window-attention bias tables.
///
/// The single-head window blocks of the recovered checkpoint store their
/// `[1, 64, 64]` relative bias in the fused kernel's `mma` fragment order rather
/// than logical `[query token, key token]` order. Read row-major, such a table
/// mixes unrelated token pairs; teacher-forced vendor stage captures put blocks
/// 1–4 at MAE 0.08–0.36 with the raw table and at 0.005–0.019 after this
/// permutation, so the affected blocks apply it at load time. The layout was
/// fitted on blocks 1 and 3 and confirmed out of sample on blocks 2 and 4; on
/// vendor-pooled block-0 inputs it also halves block 0's error (MAE `0.051` to
/// `0.027`). The 2/4/8-head blocks are already logical and must not be
/// remapped; the 16-head split blocks use the fragment order again.
enum NeuralRenderingAttentionBiasLayout {
  static let tokenCount = 64

  /// `true` for blocks whose stored bias needs `recoverFragmentSwizzle`: the
  /// single-head window blocks and the 16-head split blocks (23–30, 40–47),
  /// whose teacher-forced error on DLL captures drops from 0.03–0.30 raw to
  /// 0.001–0.01 remapped. The 2/4/8-head blocks stay raw (remapping them
  /// costs 0.15–0.33).
  static func usesFragmentSwizzle(blockIndex _: Int, headCount: Int) -> Bool {
    headCount == 1 || headCount == 16
  }

  /// Flattened source index of logical entry `(query, key)` inside the stored
  /// `64 × 64` table: a pure permutation of the twelve token-index bits.
  static func sourceIndex(query: Int, key: Int) -> Int {
    let queryY = query / 8
    let queryX = query % 8
    let keyY = key / 8
    let keyX = key % 8
    func bit(_ value: Int, _ position: Int) -> Int {
      (value >> position) & 1
    }
    return bit(queryY, 2) << 11
      | bit(queryX, 2) << 10
      | bit(keyY, 2) << 9
      | bit(keyX, 2) << 8
      | bit(queryY, 0) << 7
      | bit(queryX, 1) << 6
      | bit(queryX, 0) << 5
      | bit(keyY, 0) << 4
      | bit(keyX, 1) << 3
      | bit(keyY, 1) << 2
      | bit(queryY, 1) << 1
      | bit(keyX, 0)
  }

  static let fragmentSwizzleIndices: [Int32] = (0..<(tokenCount * tokenCount)).map { entry in
    Int32(sourceIndex(query: entry / tokenCount, key: entry % tokenCount))
  }

  /// Reorders a stored `[heads, 64, 64]` bias into logical `[heads, query, key]`.
  static func recoverFragmentSwizzle(_ bias: MLXArray) -> MLXArray {
    precondition(bias.ndim == 3 && bias.shape[1] == tokenCount && bias.shape[2] == tokenCount)
    let heads = bias.shape[0]
    let flat = bias.reshaped([heads, tokenCount * tokenCount])
    return flat.take(MLXArray(fragmentSwizzleIndices), axis: 1)
      .reshaped([heads, tokenCount, tokenCount])
  }
}
