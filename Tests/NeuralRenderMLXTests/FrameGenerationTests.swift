import Foundation
import MLX
import MLXNN
import XCTest
@testable import NeuralRenderMLX

final class FrameGenerationTests: XCTestCase {
  private func synthetic(seed: UInt64 = 1, scale: Float = 0.05) -> [String: MLXArray] {
    var generator = SplitMix(seed: seed)
    var shapes: [(String, Int, Int)] = [("block0.stem0", 32, 16), ("block0.stem1", 32, 32), ("block0.stem2", 64, 32), ("block1.stem0", 16, 18), ("block1.stem1", 32, 16)]
    for i in 0..<8 { shapes += [("block0.res\(i)", 64, 64), ("block1.res\(i)", 32, 32)] }
    for i in 0..<3 { shapes += [("block0.bot0.head\(i)", 32, 64), ("block1.bot0.head\(i)", 16, 32)] }
    shapes += [("block0.bot1", 8, 32), ("block1.bot1", 8, 16)]
    var weights: [String: MLXArray] = [:]
    for (name, cout, cin) in shapes {
      let w = (0..<(cout * cin * 9)).map { _ in generator.normal() * scale / Float(9 * cin).squareRoot() }
      let b = (0..<cout).map { _ in generator.normal() * scale }
      weights["\(name).weight"] = MLXArray(w, [cout, cin, 3, 3]).asType(.float16)
      weights["\(name).bias"] = MLXArray(b, [cout]).asType(.float16)
    }
    return weights
  }

  private func random(_ shape: [Int], seed: UInt64) -> MLXArray {
    var generator = SplitMix(seed: seed)
    let count = shape.reduce(1, *)
    return MLXArray((0..<count).map { _ in generator.uniform() }, shape)
  }

  /// CPU bilinear fetch with index clamping, the contract of both kernels.
  private func fetch(_ image: [Float], height: Int, width: Int, channels: Int, x: Float, y: Float, c: Int) -> Float {
    let x0f = x.rounded(.down), y0f = y.rounded(.down)
    let tx = x - x0f, ty = y - y0f
    func idx(_ yy: Int, _ xx: Int) -> Float {
      let cy = min(max(yy, 0), height - 1), cx = min(max(xx, 0), width - 1)
      return image[(cy * width + cx) * channels + c]
    }
    let x0 = Int(x0f), y0 = Int(y0f)
    return (1 - tx) * (1 - ty) * idx(y0, x0) + tx * (1 - ty) * idx(y0, x0 + 1) + (1 - tx) * ty * idx(y0 + 1, x0) + tx * ty * idx(y0 + 1, x0 + 1)
  }

  func testWarpKernelMatchesCPUReference() {
    let (h, w, c) = (7, 11, 4)
    let image = random([1, h, w, c], seed: 3)
    let flow = (random([1, h, w, 2], seed: 4) - 0.5) * 6
    let out = FrameGenerator.warp(image, flow: flow).asArray(Float.self)
    let img = image.asArray(Float.self), fl = flow.asArray(Float.self)
    for y in 0..<h {
      for x in 0..<w {
        let px = Float(x) + fl[(y * w + x) * 2], py = Float(y) + fl[(y * w + x) * 2 + 1]
        for ch in 0..<c {
          XCTAssertEqual(out[(y * w + x) * c + ch], fetch(img, height: h, width: w, channels: c, x: px, y: py, c: ch), accuracy: 1e-5)
        }
      }
    }
  }

  func testComposeKernelMatchesCPUReference() {
    let (h, w) = (9, 14)
    let (ch, cw) = (5, 7)
    let a = random([1, h, w, 3], seed: 5), b = random([1, h, w, 3], seed: 6)
    let coarse = concatenated([(random([1, ch, cw, 4], seed: 7) - 0.5) * 3, random([1, ch, cw, 1], seed: 8)], axis: 3)
    let out = FrameGenerator.compose(a, b, coarse: coarse, scale: 2).asArray(Float.self)
    let aa = a.asArray(Float.self), bb = b.asArray(Float.self), cc = coarse.asArray(Float.self)
    for y in 0..<h {
      for x in 0..<w {
        let u = min(Float(x) / 2, Float(cw - 1)), v = min(Float(y) / 2, Float(ch - 1))
        let up = (0..<5).map { fetch(cc, height: ch, width: cw, channels: 5, x: u, y: v, c: $0) }
        for c in 0..<3 {
          let wa = fetch(aa, height: h, width: w, channels: 3, x: Float(x) + up[0] * 2, y: Float(y) + up[1] * 2, c: c)
          let wb = fetch(bb, height: h, width: w, channels: 3, x: Float(x) + up[2] * 2, y: Float(y) + up[3] * 2, c: c)
          let expected = min(max(up[4] * wa + (1 - up[4]) * wb, 0), 1)
          XCTAssertEqual(out[(y * w + x) * 3 + c], expected, accuracy: 1e-5)
        }
      }
    }
  }

  func testBox2AndPhotometricError() {
    let a = random([1, 38, 44, 3], seed: 9), b = random([1, 38, 44, 3], seed: 10)
    let pa = FrameGenerator.box2(a), pb = FrameGenerator.box2(b)
    XCTAssertEqual(pa.shape, [1, 32, 32, 3])
    let aa = a.asArray(Float.self), pp = pa.asArray(Float.self)
    let mean = (aa[(0 * 44 + 0) * 3] + aa[(0 * 44 + 1) * 3] + aa[(1 * 44 + 0) * 3] + aa[(1 * 44 + 1) * 3]) / 4
    XCTAssertEqual(pp[0], mean, accuracy: 1e-6)
    XCTAssertEqual(pp[(20 * 32 + 5) * 3], 0, "rows past the pooled height are zero padding")
    let err = FrameGenerator.photometricError(pa, pb)
    XCTAssertEqual(err.shape, [1, 32, 32, 1])
    // interior tap: 0.75*0.75 of the centre plus the eight neighbours
    let d = abs(pa - pb).mean(axis: 3).asArray(Float.self)
    let k: [Float] = [0.125, 0.75, 0.125]
    var expected: Float = 0
    for (i, dy) in [-1, 0, 1].enumerated() { for (j, dx) in [-1, 0, 1].enumerated() { expected += k[i] * k[j] * d[(10 + dy) * 32 + 10 + dx] } }
    XCTAssertEqual(err.asArray(Float.self)[10 * 32 + 10], expected, accuracy: 1e-5)
  }

  func testUpsampleMatchesHalfPixelConvention() {
    // PyTorch bilinear x2 with align_corners=False: [0, 1] -> [0, 0.25, 0.75, 1]
    let x = MLXArray([Float(0), 1], [1, 1, 2, 1])
    let up = MLXNN.Upsample(scaleFactor: 2.0, mode: .linear(alignCorners: false))(x).asArray(Float.self)
    XCTAssertEqual(up.count, 8)
    XCTAssertEqual(up[0], 0, accuracy: 1e-6)
    XCTAssertEqual(up[1], 0.25, accuracy: 1e-6)
    XCTAssertEqual(up[2], 0.75, accuracy: 1e-6)
    XCTAssertEqual(up[3], 1, accuracy: 1e-6)
  }

  func testSyntheticEndToEndShapesAndConstantInvariance() throws {
    let generator = try FrameGenerator(weights: synthetic(), precision: .float32)
    let a = random([1, 48, 80, 3], seed: 11)
    let b = random([1, 48, 80, 3], seed: 12)
    let export = generator.synthesize(a, b, phase: 0.5)
    XCTAssertEqual(export.shape, [1, 32, 48, 8])
    let out = try generator.interpolate(a, b, phase: 0.5)
    XCTAssertEqual(out.shape, [1, 48, 80, 3])
    let frames = try generator.generate(a, b, factor: 4)
    XCTAssertEqual(frames.count, 3)
    let constant = MLXArray.full([1, 48, 80, 3], values: MLXArray(Float(0.37)))
    let same = try generator.interpolate(constant, constant, phase: 0.5).asArray(Float.self)
    XCTAssertEqual(same.max()!, 0.37, accuracy: 1e-5)
    XCTAssertEqual(same.min()!, 0.37, accuracy: 1e-5)
    XCTAssertThrowsError(try generator.interpolate(a, random([1, 48, 64, 3], seed: 1)))
    var missing = synthetic()
    missing.removeValue(forKey: "block1.bot1.bias")
    XCTAssertThrowsError(try FrameGenerator(weights: missing))
  }

  /// Against the PyTorch port on real weights: `NRK_FG_WEIGHTS` (dense safetensors) and
  /// `NRK_FG_REFERENCE` (a file of H, W as UInt32 followed by three float32 NHWC frames a, b, out).
  func testMatchesPythonReferenceWhenConfigured() throws {
    guard let weightsPath = ProcessInfo.processInfo.environment["NRK_FG_WEIGHTS"],
      let referencePath = ProcessInfo.processInfo.environment["NRK_FG_REFERENCE"]
    else { throw XCTSkip("set NRK_FG_WEIGHTS and NRK_FG_REFERENCE") }
    let data = try Data(contentsOf: URL(fileURLWithPath: referencePath))
    let header = data.prefix(8).withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
    let (h, w) = (Int(header[0]), Int(header[1]))
    let floats = data.dropFirst(8).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    let n = h * w * 3
    XCTAssertEqual(floats.count, 3 * n)
    let a = MLXArray(Array(floats[0..<n]), [1, h, w, 3])
    let b = MLXArray(Array(floats[n..<(2 * n)]), [1, h, w, 3])
    let expected = Array(floats[(2 * n)..<(3 * n)])
    for precision in [FrameGenerator.Precision.float32, .float16] {
      let generator = try FrameGenerator(weightsURL: URL(fileURLWithPath: weightsPath), precision: precision)
      let out = try generator.interpolate(a, b, phase: 0.5).asArray(Float.self)
      let mae = zip(out, expected).map { abs($0 - $1) }.reduce(0, +) / Float(n)
      let maxDiff = zip(out, expected).map { abs($0 - $1) }.max()!
      print("framegen \(precision): MAE \(mae) max \(maxDiff)")
      XCTAssertLessThan(mae, precision == .float32 ? 1e-3 : 4e-3)
    }
  }
}

/// Small deterministic generator for test data.
private struct SplitMix {
  var state: UInt64
  init(seed: UInt64) { state = seed &+ 0x9E37_79B9_7F4A_7C15 }
  mutating func next() -> UInt64 {
    state &+= 0x9E37_79B9_7F4A_7C15
    var z = state
    z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
    z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
    return z ^ (z >> 31)
  }
  mutating func uniform() -> Float { Float(next() >> 40) / Float(1 << 24) }
  mutating func normal() -> Float {
    let u1 = max(uniform(), 1e-7), u2 = uniform()
    return (-2 * log(u1)).squareRoot() * cos(2 * .pi * u2)
  }
}
