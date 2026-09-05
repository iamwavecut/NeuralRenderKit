import Foundation
import XCTest

@testable import DLSSCore

final class NeuralRenderingNetworkGeometryTests: XCTestCase {
  func testIdentityGeometryReturnsFeaturesUnchanged() throws {
    let geometry = try NeuralRenderingNetworkGeometry(
      outputWidth: 3, outputHeight: 2, networkWidth: 3, networkHeight: 2
    )
    let features = try featureTensor(width: 3, height: 2)

    let extended = try geometry.extendFeatureTensor(features, noiseFrameIndex: 4)

    XCTAssertTrue(geometry.isIdentity)
    XCTAssertEqual(extended, features)
  }

  func testExtensionReflectsThenClampsAndRegeneratesPaddedNoise() throws {
    let geometry = try NeuralRenderingNetworkGeometry(
      outputWidth: 3, outputHeight: 2, networkWidth: 5, networkHeight: 4
    )
    let features = try featureTensor(width: 3, height: 2)

    let extended = try geometry.extendFeatureTensor(features, noiseFrameIndex: 4)

    XCTAssertFalse(geometry.isIdentity)
    XCTAssertEqual(extended.descriptor.shape, [1, 4, 5, 16])
    let logical = floats(in: features)
    let values = floats(in: extended)
    for y in 0..<4 {
      for x in 0..<5 {
        let source = geometry.sourceCoordinate(x: x, y: y)
        let expectedSource = (x: [0, 1, 2, 1, 0][x], y: [0, 1, 0, 0][y])
        XCTAssertEqual(source.x, expectedSource.x, "x=\(x)")
        XCTAssertEqual(source.y, expectedSource.y, "y=\(y)")
        let target = (y * 5 + x) * 16
        let origin = (source.y * 3 + source.x) * 16
        XCTAssertEqual(
          Array(values[(target + 3)..<(target + 16)]),
          Array(logical[(origin + 3)..<(origin + 16)]),
          "channels 3...15 at (\(x), \(y))"
        )
        let noise: (Float, Float, Float)
        if x < 3, y < 2 {
          noise = (logical[origin], logical[origin + 1], logical[origin + 2])
        } else {
          noise = NeuralRenderingFirstFramePreprocessor.deterministicNoise(
            x: x, y: y, frameIndex: 4
          )
        }
        XCTAssertEqual(values[target], noise.0, "noise 0 at (\(x), \(y))")
        XCTAssertEqual(values[target + 1], noise.1, "noise 1 at (\(x), \(y))")
        XCTAssertEqual(values[target + 2], noise.2, "noise 2 at (\(x), \(y))")
      }
    }
  }

  func testExtendedFirstFrameFeaturesMatchDirectGeometryPreprocessing() throws {
    let geometry = try NeuralRenderingNetworkGeometry.vendorAligned(
      outputWidth: 70, outputHeight: 33
    )
    let color = try rgbTensor(width: 70, height: 33) { x, y in
      [Float(x) / 70, Float(y) / 33, Float((x * 7 + y * 3) % 11) / 11]
    }
    let logical = try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(
      from: color,
      noiseFrameIndex: 9,
      normalizedStyle: 1 / 128,
      localToneStrength: 0.5,
      localStructureStrength: 0.25
    )
    let direct = try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(
      from: color,
      noiseFrameIndex: 9,
      geometry: geometry,
      normalizedStyle: 1 / 128,
      localToneStrength: 0.5,
      localStructureStrength: 0.25
    )

    let extended = try geometry.extendFeatureTensor(logical, noiseFrameIndex: 9)

    XCTAssertEqual(extended.descriptor, direct.descriptor)
    XCTAssertEqual(extended.bytes, direct.bytes)
  }

  func testExtensionRejectsMismatchedLogicalShape() throws {
    let geometry = try NeuralRenderingNetworkGeometry(
      outputWidth: 3, outputHeight: 2, networkWidth: 5, networkHeight: 4
    )
    let features = try featureTensor(width: 4, height: 2)

    XCTAssertThrowsError(
      try geometry.extendFeatureTensor(features, noiseFrameIndex: 0)
    )
  }

  func testDeterministicNoiseMatchesFirstFrameFeatures() throws {
    let color = try rgbTensor(width: 4, height: 3) { _, _ in [0.5, 0.5, 0.5] }
    let features = floats(
      in: try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(
        from: color, noiseFrameIndex: 2
      )
    )

    for y in 0..<3 {
      for x in 0..<4 {
        let noise = NeuralRenderingFirstFramePreprocessor.deterministicNoise(
          x: x, y: y, frameIndex: 2
        )
        let offset = (y * 4 + x) * 16
        XCTAssertEqual([features[offset], features[offset + 1], features[offset + 2]], [
          noise.0, noise.1, noise.2,
        ])
      }
    }
  }

  private func featureTensor(width: Int, height: Int) throws -> HostTensor {
    let values = (0..<(width * height * 16)).map { Float($0) * 0.001 + 0.1 }
    return try HostTensor(
      descriptor: TensorDescriptor(
        name: "color",
        shape: [1, height, width, 16],
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: values.withUnsafeBytes { Data($0) }
    )
  }

  private func rgbTensor(
    width: Int,
    height: Int,
    pixel: (Int, Int) -> [Float]
  ) throws -> HostTensor {
    var values: [Float] = []
    for y in 0..<height {
      for x in 0..<width {
        values += pixel(x, y)
      }
    }
    return try HostTensor(
      descriptor: TensorDescriptor(
        name: "color",
        shape: [1, height, width, 3],
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: values.withUnsafeBytes { Data($0) }
    )
  }

  private func floats(in tensor: HostTensor) -> [Float] {
    tensor.bytes.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
  }
}
