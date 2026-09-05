import Foundation
import XCTest

@testable import DLSSCore

final class NeuralRenderingFirstFramePreprocessorTests: XCTestCase {
  func testControlProfilesStaySeparateFromCreateTimeModelSelection() {
    XCTAssertEqual(NeuralRenderingModelSelection.shippingDefault.rawValue, 0)
    XCTAssertEqual(NeuralRenderingModelSelection.slot1.rawValue, 1)
    XCTAssertEqual(NeuralRenderingModelSelection.slot2.rawValue, 2)
    XCTAssertEqual(NeuralRenderingModelSelection.slot3.rawValue, 3)
    XCTAssertEqual(
      NeuralRenderingControlProfile.standard.checkpointModelSelection,
      .shippingDefault
    )
    XCTAssertEqual(NeuralRenderingControlProfile.standard.styleIndex, 0)
    XCTAssertEqual(NeuralRenderingControlProfile.standard.localToneStrength, 1)
    XCTAssertEqual(NeuralRenderingControlProfile.standard.localStructureStrength, 1)
    XCTAssertEqual(NeuralRenderingControlProfile.standard.skinStructureStrength, -1)
    XCTAssertFalse(NeuralRenderingControlProfile.standard.automaticMaskEnabled)
    XCTAssertEqual(NeuralRenderingControlProfile.natural.styleIndex, 1)
    XCTAssertEqual(NeuralRenderingControlProfile.cinematic.styleIndex, 2)
    XCTAssertEqual(NeuralRenderingControlProfile.neutral.localToneStrength, 0)
    XCTAssertEqual(NeuralRenderingControlProfile.neutral.localStructureStrength, 0)
    XCTAssertEqual(
      NeuralRenderingControlProfile.standard.featureControls,
      NeuralRenderingFeatureControls()
    )
  }

  func testComposesDisplayRGBFromRecoveredFourChannelHead() throws {
    let colorValues: [Float] = [0.25, 0.5, 0.75]
    let headValues: [Float] = [1, -1, 2, 99]
    let color = try makeTensor(values: colorValues, channels: 3)
    let head = try makeTensor(values: headValues, channels: 4)

    let output = try NeuralRenderingFirstFramePostprocessor.compose(
      head: head,
      over: color
    )

    XCTAssertEqual(output.descriptor.shape, [1, 1, 1, 3])
    XCTAssertEqual(
      output.bytes.withUnsafeBytes { bytes in
        stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
          bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
        }
      }, [0.5, 0.25, 1])
  }

  func testPostprocessorRejectsNonFourChannelHead() throws {
    let color = try makeTensor(values: [0, 0, 0], channels: 3)
    let head = try makeTensor(values: [0, 0, 0], channels: 3)

    XCTAssertThrowsError(
      try NeuralRenderingFirstFramePostprocessor.compose(head: head, over: color)
    ) {
      XCTAssertEqual(
        $0 as? NeuralRenderingPostprocessorError,
        .expectedFloat32NHWC(
          channels: 4,
          shape: [1, 1, 1, 3],
          dataType: .float32,
          layout: .nhwc
        )
      )
    }
  }

  func testBuildsRecoveredSixteenChannelFirstFrameFeatures() throws {
    let values: [Float] = [
      0.25, 0.5, 0.75,
      1, 0, 0.5,
    ]
    let color = try HostTensor(
      descriptor: TensorDescriptor(
        name: "color",
        shape: [1, 1, 2, 3],
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: values.withUnsafeBytes { Data($0) }
    )

    let features = try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(
      from: color
    )

    XCTAssertEqual(features.descriptor.name, "color")
    XCTAssertEqual(features.descriptor.shape, [1, 1, 2, 16])
    XCTAssertEqual(features.descriptor.dataType, .float32)
    XCTAssertEqual(features.descriptor.layout, .nhwc)
    let actual = features.bytes.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
    let expected: [Float] = [
      -0.219_604_492_187_5, 1.028_320_312_5, 0.127_319_335_937_5, 1,
      -0.031_25, 0, 0.031_25,
      -0.031_25, 0, 0.031_25,
      0, 1, 1, -1, -1, 0,
      0.170_166_015_625, 1.937_5, 0.325_439_453_125, 1,
      0.062_5, -0.062_5, 0,
      0.062_5, -0.062_5, 0,
      0, 1, 1, -1, -1, 0,
    ]
    XCTAssertEqual(actual, expected)
  }

  func testVendorGeometryExtendsColorButRegeneratesNetworkNoise() throws {
    let color = try HostTensor(
      descriptor: TensorDescriptor(
        name: "color",
        shape: [1, 1, 2, 3],
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: [Float](arrayLiteral: 0, 0, 0, 1, 1, 1).withUnsafeBytes { Data($0) }
    )
    let geometry = try NeuralRenderingNetworkGeometry(
      outputWidth: 2,
      outputHeight: 1,
      networkWidth: 5,
      networkHeight: 1
    )

    let features = try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(
      from: color,
      geometry: geometry
    )
    let values = floatValues(in: features)

    XCTAssertEqual(features.descriptor.shape, [1, 1, 5, 16])
    XCTAssertEqual(
      (0..<5).map { values[$0 * 16 + 4] },
      [
        -0.0625, 0.0625, -0.0625, -0.0625, -0.0625,
      ])
    XCTAssertNotEqual(values[0], values[2 * 16])
  }

  func testVendorGeometryUsesObservedMinimumAndGraphAlignment() throws {
    let geometry = try NeuralRenderingNetworkGeometry.vendorAligned(
      outputWidth: 128,
      outputHeight: 129
    )

    XCTAssertEqual(geometry.networkWidth, 320)
    XCTAssertEqual(geometry.networkHeight, 320)
    XCTAssertEqual(geometry.sourceCoordinate(x: 128, y: 129).x, 126)
    XCTAssertEqual(geometry.sourceCoordinate(x: 128, y: 129).y, 127)
  }

  func testAutomaticMaskEncodesRecoveredStrengthChannels() throws {
    let color = try makeTensor(values: [0.5, 0.5, 0.5], channels: 3)

    let features = try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(
      from: color,
      normalizedStyle: 0.5,
      localToneStrength: 0.75,
      localStructureStrength: 0.25,
      automaticMask: NeuralRenderingAutomaticMaskConfiguration(
        skinStructureStrength: -1,
        automaticMaskStructureStrength: 0.125
      )
    )

    XCTAssertEqual(
      Array(floatValues(in: features)[10...15]),
      [
        0.5, 0.75, 1, 0.25, 0.125, 0,
      ])
  }

  func testAutomaticMaskPreservesNegativeStrengthSentinels() throws {
    let color = try makeTensor(values: [0.5, 0.5, 0.5], channels: 3)

    let features = try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(
      from: color,
      localToneStrength: 0.75,
      localStructureStrength: 0.25,
      automaticMask: NeuralRenderingAutomaticMaskConfiguration(
        skinStructureStrength: -1,
        automaticMaskStructureStrength: -1
      )
    )

    XCTAssertEqual(
      Array(floatValues(in: features)[10...15]),
      [
        0, 0.75, 0.25, -1, -1, 0,
      ])
  }

  func testFullRectControlMaskScalesToneAndStructureFromGreenAndBlue() throws {
    let color = try makeTensor(values: [0.5, 0.5, 0.5], channels: 3)
    let controlMask = try makeTensor(values: [0.5, 0.25, 0.75], channels: 3)

    let features = try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(
      from: color,
      normalizedStyle: 0.5,
      localToneStrength: 0.75,
      localStructureStrength: 0.25,
      controlMask: controlMask
    )

    XCTAssertEqual(
      Array(floatValues(in: features)[10...15]),
      [
        0.5, 0.1875, 0.1875, 0, 0, 0,
      ])
  }

  func testFullRectControlMaskRedBlendsOriginalAndClampedPrediction() throws {
    let color = try makeTensor(values: [0.25, 0.5, 0.75], channels: 3)
    let head = try makeTensor(values: [1, -1, 2, 99], channels: 4)
    let controlMask = try makeTensor(values: [0.5, 0.25, 0.75], channels: 3)

    let output = try NeuralRenderingFirstFramePostprocessor.compose(
      head: head,
      over: color,
      controlMask: controlMask,
      intensity: 0.8
    )

    XCTAssertEqual(floatValues(in: output)[0], 0.35, accuracy: 0.000_001)
    XCTAssertEqual(floatValues(in: output)[1], 0.4, accuracy: 0.000_001)
    XCTAssertEqual(floatValues(in: output)[2], 0.85, accuracy: 0.000_001)
  }

  func testRejectsAnythingExceptFloat32NHWCRGB() throws {
    let color = try HostTensor(
      descriptor: TensorDescriptor(
        name: "color",
        shape: [1, 1, 1, 4],
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: Data(count: 4 * MemoryLayout<Float>.size)
    )

    XCTAssertThrowsError(
      try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(from: color)
    ) {
      XCTAssertEqual(
        $0 as? NeuralRenderingPreprocessorError,
        .expectedFloat32NHWCRGB(
          shape: [1, 1, 1, 4],
          dataType: .float32,
          layout: .nhwc
        )
      )
    }
  }

  private func makeTensor(values: [Float], channels: Int) throws -> HostTensor {
    try HostTensor(
      descriptor: TensorDescriptor(
        name: "color",
        shape: [1, 1, 1, channels],
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: values.withUnsafeBytes { Data($0) }
    )
  }

  private func floatValues(in tensor: HostTensor) -> [Float] {
    tensor.bytes.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
  }
}
