import Foundation
import MLX
import DLSSCore
import XCTest

@testable import DLSSMLX

/// The GPU base-feature kernel and the CPU preprocessor must agree bit for bit,
/// noise channels included: they share the software log/sin/cos of
/// `NeuralRenderingNoiseMath`, and the network turns a one-ulp noise difference
/// at a single pixel into a visible output difference.
final class MLXFeatureNoiseParityTests: XCTestCase {
  func testDeviceFeaturesEqualPortableFeaturesBitForBit() throws {
    let width = 128, height = 128
    var seed: UInt64 = 42
    func next() -> Float {
      seed = seed &* 6364136223846793005 &+ 1442695040888963407
      return Float(seed >> 40) / Float(1 << 24)
    }
    let color = (0..<(width * height * 3)).map { _ in next() }
    let controls = NeuralRenderingFeatureControls(
      normalizedStyle: 0.5, localToneStrength: 0.25, localStructureStrength: 0.75,
      automaticMask: NeuralRenderingAutomaticMaskConfiguration(skinStructureStrength: -1, automaticMaskStructureStrength: 0.125))
    let tensor = try HostTensor(
      descriptor: TensorDescriptor(name: "color", shape: [1, height, width, 3], dataType: .float32, layout: .nhwc),
      bytes: color.withUnsafeBytes { Data($0) })
    for frameIndex: UInt32 in [0, 31, 1000] {
      let expected = try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(
        from: tensor, noiseFrameIndex: frameIndex, geometry: nil,
        normalizedStyle: controls.normalizedStyle, localToneStrength: controls.localToneStrength,
        localStructureStrength: controls.localStructureStrength, automaticMask: controls.automaticMask, controlMask: nil)
      let reference = expected.bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
      let output = MLXFirstFrameFeatureProcessor()(
        color: MLXArray(color, [1, height, width, 3]), controlMask: nil, noiseFrameIndex: frameIndex, featureControls: controls)
      let actual = output.asArray(Float.self)
      XCTAssertEqual(actual.count, reference.count)
      var mismatches = 0
      for i in 0..<actual.count where actual[i] != reference[i] { mismatches += 1 }
      XCTAssertEqual(mismatches, 0, "frame \(frameIndex): \(mismatches) feature values differ between the GPU kernel and the CPU preprocessor")
    }
  }

  /// The software functions stay within a few ulps of the library ones over the noise domain.
  func testNoiseMathTracksTheLibrary() {
    var worstLog: Float = 0, worstSin: Float = 0, worstCos: Float = 0
    for i in 1...20000 {
      let u = Float(i) / 20000
      worstLog = max(worstLog, abs(NeuralRenderingNoiseMath.log(u) - Foundation.log(u)) / max(abs(Foundation.log(u)), 1e-6))
      let angle = Float(6.283_185_482_025_146_5) * u
      worstSin = max(worstSin, abs(NeuralRenderingNoiseMath.sin(angle) - Foundation.sin(angle)))
      worstCos = max(worstCos, abs(NeuralRenderingNoiseMath.cos(angle) - Foundation.cos(angle)))
    }
    XCTAssertLessThan(worstLog, 4e-7, "log relative error \(worstLog)")
    XCTAssertLessThan(worstSin, 4e-7, "sin absolute error \(worstSin)")
    XCTAssertLessThan(worstCos, 4e-7, "cos absolute error \(worstCos)")
    XCTAssertEqual(NeuralRenderingNoiseMath.log(Float(5.960_464_477_539_063e-8)), Foundation.log(Float(5.960_464_477_539_063e-8)), accuracy: 2e-6)
  }
}
