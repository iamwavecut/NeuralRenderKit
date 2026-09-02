import Foundation
import XCTest

@testable import NeuralRenderCore

final class NeuralRenderingTemporalPreprocessorTests: XCTestCase {
  func testPixelMotionUsesSignedScaleAndEffectiveExtent() throws {
    let pixelMotion = try tensor(
      name: "pixelMotion",
      shape: [1, 1, 2, 2],
      values: [2, -4, -1, 3]
    )

    let normalized = try NeuralRenderingTemporalReferencePreprocessor.normalizePixelMotion(
      pixelMotion,
      scaleX: -2,
      scaleY: 0.5,
      effectiveWidth: 8,
      effectiveHeight: 4
    )

    XCTAssertEqual(normalized.descriptor.name, "motion")
    XCTAssertEqual(
      normalized.bytes.withUnsafeBytes { bytes in
        stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
          bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
        }
      },
      [-0.5, -0.5, 0.25, 0.375]
    )
  }

  func testPixelMotionAddsPreviousMinusCurrentJitterInPixelSpace() throws {
    let pixelMotion = try tensor(
      name: "pixelMotion",
      shape: [1, 1, 2, 2],
      values: [2, -4, -1, 3]
    )

    let normalized = try NeuralRenderingTemporalReferencePreprocessor.normalizePixelMotion(
      pixelMotion,
      scaleX: -2,
      scaleY: 0.5,
      effectiveWidth: 8,
      effectiveHeight: 4,
      jitterDeltaX: 1,
      jitterDeltaY: -2
    )

    XCTAssertEqual(
      normalized.bytes.withUnsafeBytes { bytes in
        stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
          bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
        }
      },
      [-0.375, -1, 0.375, -0.125]
    )
  }

  func testPixelMotionRejectsNonPositiveEffectiveExtent() throws {
    let pixelMotion = try tensor(
      name: "pixelMotion",
      shape: [1, 1, 1, 2],
      values: [0, 0]
    )

    XCTAssertThrowsError(
      try NeuralRenderingTemporalReferencePreprocessor.normalizePixelMotion(
        pixelMotion,
        scaleX: 1,
        scaleY: 1,
        effectiveWidth: 0,
        effectiveHeight: 4
      )
    ) {
      XCTAssertEqual(
        $0 as? NeuralRenderingTemporalPreprocessorError,
        .invalidMotionExtent(width: 0, height: 4)
      )
    }
  }

  func testPixelMotionRejectsNonFiniteScale() throws {
    let pixelMotion = try tensor(
      name: "pixelMotion",
      shape: [1, 1, 1, 2],
      values: [0, 0]
    )

    XCTAssertThrowsError(
      try NeuralRenderingTemporalReferencePreprocessor.normalizePixelMotion(
        pixelMotion,
        scaleX: .infinity,
        scaleY: 1,
        effectiveWidth: 1,
        effectiveHeight: 1
      )
    ) {
      XCTAssertEqual(
        $0 as? NeuralRenderingTemporalPreprocessorError,
        .nonFiniteMotionScale
      )
    }
  }

  func testPixelMotionRejectsNonFiniteJitterDelta() throws {
    let pixelMotion = try tensor(
      name: "pixelMotion",
      shape: [1, 1, 1, 2],
      values: [0, 0]
    )

    for (jitterDeltaX, jitterDeltaY) in [
      (Float.nan, Float.zero),
      (Float.zero, Float.infinity),
    ] {
      XCTAssertThrowsError(
        try NeuralRenderingTemporalReferencePreprocessor.normalizePixelMotion(
          pixelMotion,
          scaleX: 1,
          scaleY: 1,
          effectiveWidth: 1,
          effectiveHeight: 1,
          jitterDeltaX: jitterDeltaX,
          jitterDeltaY: jitterDeltaY
        )
      ) {
        XCTAssertEqual(
          $0 as? NeuralRenderingTemporalPreprocessorError,
          .nonFiniteJitterDelta
        )
      }
    }
  }

  func testTemporalPostprocessorBlendsPredictionWithReprojectedHistory() throws {
    let color = try tensor(
      name: "color",
      shape: [1, 1, 1, 3],
      values: [0.2, 0.4, 0.6]
    )
    let head = try tensor(
      name: "head",
      shape: [1, 1, 1, 4],
      values: [1, -1, 0, 0]
    )
    var featureValues = [Float](repeating: 0, count: 16)
    featureValues[7] = 0.0625
    featureValues[8] = -0.0625
    let features = try tensor(
      name: "features",
      shape: [1, 1, 1, 16],
      values: featureValues
    )

    let output = try NeuralRenderingTemporalReferencePostprocessor.compose(
      head: head,
      currentColor: color,
      features: features,
      blendScale: 0.5
    )

    let actual = output.bytes.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
    XCTAssertEqual(actual.count, 3)
    XCTAssertEqual(actual[0], 0.5875, accuracy: 0.000_001)
    XCTAssertEqual(actual[1], 0.1125, accuracy: 0.000_001)
    XCTAssertEqual(actual[2], 0.575, accuracy: 0.000_001)
  }

  func testTemporalControlMaskBlendsCurrentTowardCompletedTemporalRGB() throws {
    let color = try tensor(
      name: "color",
      shape: [1, 1, 1, 3],
      values: [0.2, 0.4, 0.6]
    )
    let head = try tensor(
      name: "head",
      shape: [1, 1, 1, 4],
      values: [1, -1, 0, 0]
    )
    let controlMask = try tensor(
      name: "controlMask",
      shape: [1, 1, 1, 3],
      values: [0.25, 0.5, 0.75]
    )
    var featureValues = [Float](repeating: 0, count: 16)
    featureValues[7] = 0.0625
    featureValues[8] = -0.0625
    let features = try tensor(
      name: "features",
      shape: [1, 1, 1, 16],
      values: featureValues
    )

    let output = try NeuralRenderingTemporalReferencePostprocessor.compose(
      head: head,
      currentColor: color,
      features: features,
      blendScale: 0.5,
      controlMask: controlMask,
      intensity: 0.8
    )

    let actual = output.bytes.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
    XCTAssertEqual(actual[0], 0.2775, accuracy: 0.000_001)
    XCTAssertEqual(actual[1], 0.3425, accuracy: 0.000_001)
    XCTAssertEqual(actual[2], 0.595, accuracy: 0.000_001)
  }

  func testTemporalIntensityBlendsWithoutControlMask() throws {
    let color = try tensor(
      name: "color",
      shape: [1, 1, 1, 3],
      values: [0.2, 0.4, 0.6]
    )
    let head = try tensor(
      name: "head",
      shape: [1, 1, 1, 4],
      values: [1, -1, 0, 0]
    )
    var featureValues = [Float](repeating: 0, count: 16)
    featureValues[7] = 0.0625
    featureValues[8] = -0.0625
    let features = try tensor(
      name: "features",
      shape: [1, 1, 1, 16],
      values: featureValues
    )

    let output = try NeuralRenderingTemporalReferencePostprocessor.compose(
      head: head,
      currentColor: color,
      features: features,
      blendScale: 0.5,
      intensity: 0.8
    )

    let actual = floatValues(in: output)
    XCTAssertEqual(actual[0], 0.51, accuracy: 0.000_001)
    XCTAssertEqual(actual[1], 0.17, accuracy: 0.000_001)
    XCTAssertEqual(actual[2], 0.58, accuracy: 0.000_001)
  }

  func testTemporalPostprocessorRejectsNonSixteenChannelFeatures() throws {
    let color = try tensor(
      name: "color",
      shape: [1, 1, 1, 3],
      values: [0, 0, 0]
    )
    let head = try tensor(
      name: "head",
      shape: [1, 1, 1, 4],
      values: [0, 0, 0, 0]
    )

    XCTAssertThrowsError(
      try NeuralRenderingTemporalReferencePostprocessor.compose(
        head: head,
        currentColor: color,
        features: color
      )
    ) {
      XCTAssertEqual(
        $0 as? NeuralRenderingPostprocessorError,
        .expectedFloat32NHWC(
          channels: 16,
          shape: [1, 1, 1, 3],
          dataType: .float32,
          layout: .nhwc
        )
      )
    }
  }

  func testObservedVendorDepthDescriptorKeepsCurrentPixelMotionByDefault() throws {
    let current = try tensor(
      name: "color",
      shape: [1, 3, 3, 3],
      values: [Float](repeating: 0.5, count: 27)
    )
    var historyValues = [Float](repeating: 0.5, count: 27)
    historyValues[(1 * 3 + 2) * 3..<(1 * 3 + 2) * 3 + 3] = [1, 0.5, 0]
    let history = try tensor(
      name: "history",
      shape: [1, 3, 3, 3],
      values: historyValues
    )
    var motionValues = [Float](repeating: 0, count: 18)
    motionValues[(2 * 3 + 2) * 2] = 1 / 3
    let motion = try tensor(
      name: "motion",
      shape: [1, 3, 3, 2],
      values: motionValues
    )
    let depth = try tensor(
      name: "depth",
      shape: [1, 3, 3, 1],
      values: [
        0.9, 0.9, 0.9,
        0.9, 0.8, 0.9,
        0.9, 0.9, 0.2,
      ]
    )

    let output = try NeuralRenderingTemporalReferencePreprocessor.makeFeatureTensor(
      currentColor: current,
      historyColor: history,
      normalizedMotion: motion,
      depth: depth,
      depthInverted: false,
      noiseFrameIndex: 1
    )

    XCTAssertEqual(historyFeatures(atX: 1, y: 1, in: output), [0, 0, 0])
  }

  func testTemporalControlMaskRoutesGreenAndBlueIntoBaseFeatures() throws {
    let current = try tensor(
      name: "color",
      shape: [1, 1, 1, 3],
      values: [0.5, 0.5, 0.5]
    )
    let history = try tensor(
      name: "history",
      shape: [1, 1, 1, 3],
      values: [0.5, 0.5, 0.5]
    )
    let controlMask = try tensor(
      name: "controlMask",
      shape: [1, 1, 1, 3],
      values: [0.25, 0.5, 0.75]
    )
    let motion = try tensor(
      name: "motion",
      shape: [1, 1, 1, 2],
      values: [0, 0]
    )
    let depth = try tensor(
      name: "depth",
      shape: [1, 1, 1, 1],
      values: [1]
    )

    let output = try NeuralRenderingCPUTemporalFeaturePreprocessor()
      .makeFeatureTensor(
        currentColor: current,
        historyColor: history,
        controlMask: controlMask,
        normalizedMotion: motion,
        depth: depth,
        depthInverted: false,
        noiseFrameIndex: 1
      )

    XCTAssertEqual(Array(floatValues(in: output)[10...15]), [
      0, 0.5, 0.75, 0, 0, 0,
    ])
  }

  func testTemporalPreprocessorCarriesStyleAndAutomaticMaskControls() throws {
    let color = try tensor(
      name: "color",
      shape: [1, 1, 1, 3],
      values: [0.5, 0.5, 0.5]
    )
    let motion = try tensor(
      name: "motion",
      shape: [1, 1, 1, 2],
      values: [0, 0]
    )
    let depth = try tensor(
      name: "depth",
      shape: [1, 1, 1, 1],
      values: [1]
    )
    let controls = NeuralRenderingFeatureControls(
      normalizedStyle: 0.5,
      localToneStrength: 0.25,
      localStructureStrength: 0.75,
      automaticMask: NeuralRenderingAutomaticMaskConfiguration(
        skinStructureStrength: -1,
        automaticMaskStructureStrength: 0.125
      )
    )

    let output = try NeuralRenderingCPUTemporalFeaturePreprocessor(
      depthGuideMode: .observedZeroDescriptor,
      historyTransform: nil,
      motionTransform: nil,
      featureControls: controls
    ).makeFeatureTensor(
      currentColor: color,
      historyColor: color,
      normalizedMotion: motion,
      depth: depth,
      depthInverted: false,
      noiseFrameIndex: 1
    )

    XCTAssertEqual(Array(floatValues(in: output)[10...15]), [
      0.5, 0.25, 1, 0.75, 0.125, 0,
    ])
  }

  func testExtentMismatchedHistoryUsesRecoveredLinearTextureFilter() throws {
    let current = try tensor(
      name: "color",
      shape: [1, 1, 2, 3],
      values: [Float](repeating: 0.5, count: 6)
    )
    let history = try tensor(
      name: "history",
      shape: [1, 1, 4, 3],
      values: [
        0, 0.5, 0.5,
        0.5, 0.5, 0.5,
        1, 0.5, 0.5,
        0.5, 0.5, 0.5,
      ]
    )
    let motion = try tensor(
      name: "motion",
      shape: [1, 1, 2, 2],
      values: [0, 0, 0, 0]
    )
    let depth = try tensor(
      name: "depth",
      shape: [1, 1, 2, 1],
      values: [1, 1]
    )
    let historyTransform = try NeuralRenderingTextureTransform(
      baseX: 0,
      baseY: 0,
      extentWidth: 4,
      extentHeight: 1,
      resourceWidth: 4,
      resourceHeight: 1
    )

    let preprocessor = NeuralRenderingCPUTemporalFeaturePreprocessor(
      depthGuideMode: .observedZeroDescriptor,
      historyTransform: historyTransform,
      motionTransform: nil
    )
    let output = try preprocessor.makeFeatureTensor(
      currentColor: current,
      historyColor: history,
      normalizedMotion: motion,
      depth: depth,
      depthInverted: false,
      noiseFrameIndex: 1
    )

    XCTAssertEqual(historyFeatures(atX: 0, y: 0, in: output), [-0.03125, 0, 0])
    XCTAssertEqual(historyFeatures(atX: 1, y: 0, in: output), [0.03125, 0, 0])
  }

  func testExtentMismatchedMotionUsesRecoveredPointTextureFilter() throws {
    let current = try tensor(
      name: "color",
      shape: [1, 1, 2, 3],
      values: [Float](repeating: 0.5, count: 6)
    )
    let history = try tensor(
      name: "history",
      shape: [1, 1, 2, 3],
      values: [
        0, 0.5, 0.5,
        1, 0.5, 0.5,
      ]
    )
    let motion = try tensor(
      name: "motion",
      shape: [1, 1, 4, 2],
      values: [
        0, 0,
        0.5, 0,
        0, 0,
        -0.5, 0,
      ]
    )
    let depth = try tensor(
      name: "depth",
      shape: [1, 1, 2, 1],
      values: [1, 1]
    )
    let motionTransform = try NeuralRenderingTextureTransform(
      baseX: 0,
      baseY: 0,
      extentWidth: 4,
      extentHeight: 1,
      resourceWidth: 4,
      resourceHeight: 1
    )

    let preprocessor = NeuralRenderingCPUTemporalFeaturePreprocessor(
      depthGuideMode: .observedZeroDescriptor,
      historyTransform: nil,
      motionTransform: motionTransform
    )
    let output = try preprocessor.makeFeatureTensor(
      currentColor: current,
      historyColor: history,
      normalizedMotion: motion,
      depth: depth,
      depthInverted: false,
      noiseFrameIndex: 1
    )

    XCTAssertEqual(historyFeatures(atX: 0, y: 0, in: output), [0.0625, 0, 0])
    XCTAssertEqual(historyFeatures(atX: 1, y: 0, in: output), [-0.0625, 0, 0])
  }

  func testDepthInversionSelectsClosestDiagonalMotion() throws {
    let current = try tensor(
      name: "color",
      shape: [1, 3, 3, 3],
      values: [Float](repeating: 0.5, count: 27)
    )
    var historyValues = [Float](repeating: 0.5, count: 27)
    historyValues[(1 * 3 + 0) * 3..<(1 * 3 + 0) * 3 + 3] = [0.25, 1, 0]
    historyValues[(1 * 3 + 2) * 3..<(1 * 3 + 2) * 3 + 3] = [1, 0.5, 0]
    let history = try tensor(
      name: "history",
      shape: [1, 3, 3, 3],
      values: historyValues
    )
    var motionValues = [Float](repeating: 0, count: 18)
    motionValues[(0 * 3 + 0) * 2] = -1 / 3
    motionValues[(2 * 3 + 2) * 2] = 1 / 3
    let motion = try tensor(
      name: "motion",
      shape: [1, 3, 3, 2],
      values: motionValues
    )
    let depth = try tensor(
      name: "depth",
      shape: [1, 3, 3, 1],
      values: [
        0.9, 0.4, 0.7,
        0.4, 0.8, 0.4,
        0.7, 0.4, 0.2,
      ]
    )

    let regular = try NeuralRenderingTemporalReferencePreprocessor.makeFeatureTensor(
      currentColor: current,
      historyColor: history,
      normalizedMotion: motion,
      depth: depth,
      depthInverted: false,
      depthGuideMode: .closestDepth,
      noiseFrameIndex: 1
    )
    let inverted = try NeuralRenderingTemporalReferencePreprocessor.makeFeatureTensor(
      currentColor: current,
      historyColor: history,
      normalizedMotion: motion,
      depth: depth,
      depthInverted: true,
      depthGuideMode: .closestDepth,
      noiseFrameIndex: 1
    )

    XCTAssertEqual(historyFeatures(atX: 1, y: 1, in: regular), [0.0625, 0, -0.0625])
    XCTAssertEqual(historyFeatures(atX: 1, y: 1, in: inverted), [-0.03125, 0.0625, -0.0625])
  }

  func testFiveTapCatmullRomChangesFractionalHistorySample() throws {
    let current = try tensor(
      name: "color",
      shape: [1, 1, 4, 3],
      values: [Float](repeating: 0.5, count: 12)
    )
    let history = try tensor(
      name: "history",
      shape: [1, 1, 4, 3],
      values: [
        0, 0.5, 0.5,
        0, 0.5, 0.5,
        1, 0.5, 0.5,
        0, 0.5, 0.5,
      ]
    )
    let motion = try tensor(
      name: "motion",
      shape: [1, 1, 4, 2],
      values: [
        0.125, 0,
        0.125, 0,
        0.125, 0,
        0.125, 0,
      ]
    )
    let depth = try tensor(
      name: "depth",
      shape: [1, 1, 4, 1],
      values: [1, 1, 1, 1]
    )

    let output = try NeuralRenderingTemporalReferencePreprocessor.makeFeatureTensor(
      currentColor: current,
      historyColor: history,
      normalizedMotion: motion,
      depth: depth,
      depthInverted: false,
      noiseFrameIndex: 1
    )

    XCTAssertEqual(historyFeatures(atX: 1, y: 0, in: output), [0.0078125, 0, 0])
  }

  func testRejectsMotionWithoutTwoChannels() throws {
    let color = try tensor(
      name: "color",
      shape: [1, 1, 1, 3],
      values: [0, 0, 0]
    )
    let badMotion = try tensor(
      name: "motion",
      shape: [1, 1, 1, 3],
      values: [0, 0, 0]
    )
    let depth = try tensor(
      name: "depth",
      shape: [1, 1, 1, 1],
      values: [1]
    )

    XCTAssertThrowsError(
      try NeuralRenderingTemporalReferencePreprocessor.makeFeatureTensor(
        currentColor: color,
        historyColor: color,
        normalizedMotion: badMotion,
        depth: depth
      )
    ) {
      XCTAssertEqual(
        $0 as? NeuralRenderingTemporalPreprocessorError,
        .expectedFloat32NHWC(
          name: "motion",
          channels: 2,
          shape: [1, 1, 1, 3],
          dataType: .float32,
          layout: .nhwc
        )
      )
    }
  }

  private func historyFeatures(atX x: Int, y: Int, in tensor: HostTensor) -> [Float] {
    let values = tensor.bytes.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
    let width = tensor.descriptor.shape[2]
    let offset = (y * width + x) * 16 + 7
    return Array(values[offset..<offset + 3])
  }

  private func floatValues(in tensor: HostTensor) -> [Float] {
    tensor.bytes.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
  }

  private func tensor(
    name: String,
    shape: [Int],
    values: [Float]
  ) throws -> HostTensor {
    try HostTensor(
      descriptor: TensorDescriptor(
        name: name,
        shape: shape,
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: values.withUnsafeBytes { Data($0) }
    )
  }
}
