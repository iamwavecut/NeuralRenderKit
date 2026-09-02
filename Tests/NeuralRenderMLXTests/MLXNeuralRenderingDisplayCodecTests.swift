import Foundation
import MLX
import NeuralRenderCore
import XCTest

@testable import NeuralRenderMLX

final class MLXNeuralRenderingDisplayCodecTests: XCTestCase {
  func testDeviceEncodeMatchesPortableReference() throws {
    let original = try tensor(
      pattern(count: 8 * 8 * 3, multiplier: 7, modulus: 41, divisor: 10),
      width: 8,
      height: 8
    )
    let configuration = NeuralRenderingDisplayCodecConfiguration(
      whitePoint: 1.25
    )
    let expected = try NeuralRenderingDisplayCodec.encode(
      original,
      configuration: configuration
    )

    let output = MLXNeuralRenderingDisplayCodec().encode(
      array(original),
      configuration: configuration
    )
    eval(output)

    assertClose(output.asArray(Float.self), values(expected), tolerance: 0.000_002)
  }

  func testDeviceResolveMatchesPortableReference() throws {
    let proxy = try tensor(
      pattern(count: 8 * 8 * 3, multiplier: 3, modulus: 37, divisor: 36),
      width: 8,
      height: 8
    )
    let model = try tensor(
      pattern(
        count: 8 * 8 * 3,
        multiplier: 5,
        addend: 1,
        modulus: 39,
        divisor: 38
      ),
      width: 8,
      height: 8
    )
    let original = try tensor(
      pattern(
        count: 8 * 8 * 3,
        multiplier: 11,
        addend: 2,
        modulus: 53,
        divisor: 9
      ),
      width: 8,
      height: 8
    )
    let configuration = NeuralRenderingDisplayCodecConfiguration(
      whitePoint: 1.5,
      transferStrength: 0.8,
      colorStrength: 0.65,
      maximumLuminanceRatio: 1.75
    )
    let expected = try NeuralRenderingDisplayCodec.resolve(
      proxy: proxy,
      model: model,
      original: original,
      configuration: configuration
    )

    let output = MLXNeuralRenderingDisplayCodec().resolve(
      proxy: array(proxy),
      model: array(model),
      original: array(original),
      configuration: configuration
    )
    eval(output)

    assertClose(output.asArray(Float.self), values(expected), tolerance: 0.000_02)
  }

  func testZeroStrengthReturnsOriginalWithoutADeviceRoundTrip() throws {
    let proxy = MLXArray([Float(0.1), 0.2, 0.3], [1, 1, 1, 3])
    let model = MLXArray([Float(0.8), 0.7, 0.6], [1, 1, 1, 3])
    let original = MLXArray([Float(4), 2, 1], [1, 1, 1, 3])

    let output = MLXNeuralRenderingDisplayCodec().resolve(
      proxy: proxy,
      model: model,
      original: original,
      configuration: .init(transferStrength: 0)
    )
    eval(output)

    XCTAssertEqual(output.asArray(Float.self), [4, 2, 1])
  }

  private func tensor(
    _ values: [Float],
    width: Int,
    height: Int
  ) throws -> HostTensor {
    try HostTensor(
      descriptor: TensorDescriptor(
        name: "color",
        shape: [1, height, width, 3],
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: values.withUnsafeBytes { Data($0) }
    )
  }

  private func pattern(
    count: Int,
    multiplier: Int,
    addend: Int = 0,
    modulus: Int,
    divisor: Float
  ) -> [Float] {
    (0..<count).map {
      Float(($0 * multiplier + addend) % modulus) / divisor
    }
  }

  private func array(_ tensor: HostTensor) -> MLXArray {
    MLXArray(tensor.bytes, tensor.descriptor.shape, dtype: .float32)
  }

  private func values(_ tensor: HostTensor) -> [Float] {
    tensor.bytes.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
  }

  private func assertClose(
    _ actual: [Float],
    _ expected: [Float],
    tolerance: Float,
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    XCTAssertEqual(actual.count, expected.count, file: file, line: line)
    let maximum = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
    XCTAssertLessThanOrEqual(maximum, tolerance, file: file, line: line)
  }
}
