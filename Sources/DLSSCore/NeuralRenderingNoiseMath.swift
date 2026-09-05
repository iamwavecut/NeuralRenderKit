import Foundation

/// Single-precision `log`, `sin` and `cos` built from multiplications and
/// additions only (the Cephes polynomials), so the CPU preprocessor and the
/// Metal feature kernel evaluate the noise features with the same operation
/// sequence and agree bit for bit. Library functions differ between the
/// host and the GPU by an ulp at some inputs, which the half rounding of the
/// noise turns into a visible difference for a few pixels per frame, and the
/// network amplifies. The Metal side is `mlxdlssLog`, `mlxdlssSin`, `mlxdlssCos` in
/// `MLXFirstFrameFeatureProcessor.header`; keep the two in step.
public enum NeuralRenderingNoiseMath {
  /// Natural logarithm of a positive normal float.
  public static func log(_ x: Float) -> Float {
    let bits = x.bitPattern
    var e = Float(Int32((bits >> 23) & 0xff) - 126)
    var m = Float(bitPattern: (bits & 0x807f_ffff) | 0x3f00_0000)   // [0.5, 1)
    if m < 0.707_106_781_186_547_524 {
      e -= 1
      m = m + m - 1
    } else {
      m = m - 1
    }
    let z = m * m
    var y = 7.037_683_629_2e-2 * m - 1.151_461_031_0e-1
    y = y * m + 1.167_699_874_0e-1
    y = y * m - 1.242_014_084_6e-1
    y = y * m + 1.424_932_278_7e-1
    y = y * m - 1.666_805_766_5e-1
    y = y * m + 2.000_071_476_5e-1
    y = y * m - 2.499_999_399_3e-1
    y = y * m + 3.333_333_117_4e-1
    y = y * m * z
    y = y + -2.121_944_40e-4 * e
    y = y + -0.5 * z
    var result = m + y
    result = result + 0.693_359_375 * e
    return result
  }

  private static let dp1: Float = 0.785_156_25
  private static let dp2: Float = 2.418_756_484_985_351_562_5e-4
  private static let dp3: Float = 3.774_894_977_445_941_08e-8
  private static let fourOverPi: Float = 1.273_239_544_735_16

  private static func sinPoly(_ z: Float, _ x: Float) -> Float {
    var y = -1.951_529_589_1e-4 * z + 8.332_160_873_6e-3
    y = y * z - 1.666_665_461_1e-1
    y = y * z * x
    return y + x
  }

  private static func cosPoly(_ z: Float) -> Float {
    var y = 2.443_315_711_809_948e-5 * z - 1.388_731_625_493_765e-3
    y = y * z + 4.166_664_568_298_827e-2
    y = y * z * z
    y = y - 0.5 * z
    return y + 1
  }

  /// Sine of a non-negative angle below 2^13 (the noise angles lie in [0, 2π]).
  public static func sin(_ input: Float) -> Float {
    var sign: Float = 1
    var x = input
    if x < 0 { x = -x; sign = -1 }
    var j = Int32(fourOverPi * x)
    var y = Float(j)
    if j & 1 != 0 { j += 1; y += 1 }
    j &= 7
    if j > 3 { sign = -sign; j -= 4 }
    x = ((x - y * dp1) - y * dp2) - y * dp3
    let z = x * x
    let result = (j == 1 || j == 2) ? cosPoly(z) : sinPoly(z, x)
    return sign < 0 ? -result : result
  }

  /// Cosine, same domain as `sin`.
  public static func cos(_ input: Float) -> Float {
    var x = input < 0 ? -input : input
    var j = Int32(fourOverPi * x)
    var y = Float(j)
    if j & 1 != 0 { j += 1; y += 1 }
    j &= 7
    var sign: Float = 1
    if j > 3 { j -= 4; sign = -sign }
    if j > 1 { sign = -sign }
    x = ((x - y * dp1) - y * dp2) - y * dp3
    let z = x * x
    let result = (j == 1 || j == 2) ? sinPoly(z, x) : cosPoly(z)
    return sign < 0 ? -result : result
  }
}
