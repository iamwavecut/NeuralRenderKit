import Foundation
import XCTest

@testable import DLSSCore

final class NeuralRenderingDisplayCodecTests: XCTestCase {
  func testEncodesLinearMidGreyAsSRGB() throws {
    let original = try tensor([0.25, 0.25, 0.25])

    let proxy = try NeuralRenderingDisplayCodec.encode(
      original,
      configuration: .init(whitePoint: 1)
    )

    for value in values(proxy) {
      XCTAssertEqual(value, 0.537_098_7, accuracy: 0.000_001)
    }
  }

  func testEncodePassthroughPreservesExactBytes() throws {
    let original = try tensor([0.25, 0.5, 0.75])

    let proxy = try NeuralRenderingDisplayCodec.encode(
      original,
      configuration: .init(inputIsDisplayReferred: true)
    )

    XCTAssertEqual(proxy.bytes, original.bytes)
  }

  func testEncodeUsesSoftKneeInsteadOfClippingBrightInput() throws {
    let original = try tensor([1, 1, 1])

    let proxy = try NeuralRenderingDisplayCodec.encode(
      original,
      configuration: .init(whitePoint: 1)
    )

    for value in values(proxy) {
      XCTAssertGreaterThan(value, 0.88)
      XCTAssertLessThan(value, 1)
    }
  }

  func testResolveShadowBranchRescalesModelPicture() throws {
    let proxy = try tensor([0.5, 0.5, 0.5])
    let model = try tensor([0.75, 0.75, 0.75])
    let original = try tensor([0.25, 0.25, 0.25])

    let output = try NeuralRenderingDisplayCodec.resolve(
      proxy: proxy,
      model: model,
      original: original,
      configuration: .init(inputIsDisplayReferred: true)
    )

    for value in values(output) {
      XCTAssertEqual(value, 0.375, accuracy: 0.000_001)
    }
  }

  func testResolveHighlightBranchReturnsUnrepresentedHeadroom() throws {
    let proxy = try tensor([0.25, 0.25, 0.25])
    let model = try tensor([0.5, 0.5, 0.5])
    let original = try tensor([0.75, 0.75, 0.75])

    let output = try NeuralRenderingDisplayCodec.resolve(
      proxy: proxy,
      model: model,
      original: original,
      configuration: .init(inputIsDisplayReferred: true)
    )

    for value in values(output) {
      XCTAssertEqual(value, 1, accuracy: 0.000_003)
    }
  }

  func testResolveZeroStrengthIsByteExactNoOp() throws {
    let proxy = try tensor([0.1, 0.2, 0.3])
    let model = try tensor([0.8, 0.7, 0.6])
    let original = try tensor([4, 2, 1])

    let output = try NeuralRenderingDisplayCodec.resolve(
      proxy: proxy,
      model: model,
      original: original,
      configuration: .init(transferStrength: 0)
    )

    XCTAssertEqual(output, original)
  }

  func testResolveEmptyModelIsByteExactNoOp() throws {
    let proxy = try tensor([0.1, 0.2, 0.3])
    let model = try tensor([0, 0, 0])
    let original = try tensor([4, 2, 1])

    let output = try NeuralRenderingDisplayCodec.resolve(
      proxy: proxy,
      model: model,
      original: original,
      configuration: .init()
    )

    XCTAssertEqual(output, original)
  }

  func testRejectsInvalidConfigurationAndMismatchedShape() throws {
    let rgb = try tensor([0.1, 0.2, 0.3])
    let wider = try tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], width: 2)

    XCTAssertThrowsError(
      try NeuralRenderingDisplayCodec.encode(
        rgb,
        configuration: .init(whitePoint: 0)
      )
    ) {
      XCTAssertEqual(
        $0 as? NeuralRenderingDisplayCodecError,
        .invalidConfiguration
      )
    }
    XCTAssertThrowsError(
      try NeuralRenderingDisplayCodec.resolve(
        proxy: rgb,
        model: wider,
        original: rgb
      )
    ) {
      XCTAssertEqual(
        $0 as? NeuralRenderingDisplayCodecError,
        .spatialShapeMismatch(
          expected: [1, 1, 1, 3],
          actual: [1, 1, 2, 3]
        )
      )
    }
  }

  private func tensor(_ values: [Float], width: Int = 1) throws -> HostTensor {
    try HostTensor(
      descriptor: TensorDescriptor(
        name: "color",
        shape: [1, 1, width, 3],
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: values.withUnsafeBytes { Data($0) }
    )
  }

  private func values(_ tensor: HostTensor) -> [Float] {
    tensor.bytes.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
  }
}
