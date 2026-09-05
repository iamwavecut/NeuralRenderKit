import Foundation
import DLSSCore
import XCTest

@testable import DLSSCore

final class MetalTemporalFeaturePreprocessorTests: XCTestCase {
  func testMetalHistorySamplingMatchesCPUReferenceBytesForBothDepthModes() throws {
    let width = 8
    let height = 8
    let current = try tensor(
      name: "color",
      channels: 3,
      width: width,
      height: height,
      values: (0..<(width * height * 3)).map { Float($0 % 17) / 16 }
    )
    let history = try tensor(
      name: "history",
      channels: 3,
      width: width,
      height: height,
      values: (0..<(width * height * 3)).map { Float(($0 * 7) % 19) / 18 }
    )
    let motion = try tensor(
      name: "motion",
      channels: 2,
      width: width,
      height: height,
      values: (0..<(width * height * 2)).map { index in
        let pixel = index / 2
        if index.isMultiple(of: 2) {
          return Float((pixel * 3) % 11 - 5) / 128
        }
        return Float((pixel * 5) % 13 - 6) / 128
      }
    )
    let depth = try tensor(
      name: "depth",
      channels: 1,
      width: width,
      height: height,
      values: (0..<(width * height)).map { Float(($0 * 11) % 23) / 22 }
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
    for mode in [
      NeuralRenderingDepthGuideMode.observedZeroDescriptor,
      .closestDepth,
    ] {
      let cpu = try NeuralRenderingTemporalReferencePreprocessor.makeFeatureTensor(
        currentColor: current,
        historyColor: history,
        normalizedMotion: motion,
        depth: depth,
        depthInverted: true,
        depthGuideMode: mode,
        noiseFrameIndex: 7,
        normalizedStyle: controls.normalizedStyle,
        localToneStrength: controls.localToneStrength,
        localStructureStrength: controls.localStructureStrength,
        automaticMask: controls.automaticMask
      )
      let metal = try MetalNeuralRenderingTemporalFeaturePreprocessor(
        depthGuideMode: mode,
        historyTransform: nil,
        motionTransform: nil,
        featureControls: controls
      ).makeFeatureTensor(
        currentColor: current,
        historyColor: history,
        normalizedMotion: motion,
        depth: depth,
        depthInverted: true,
        noiseFrameIndex: 7
      )

      XCTAssertEqual(metal.descriptor, cpu.descriptor)
      XCTAssertEqual(metal.bytes, cpu.bytes, "depth mode: \(mode.rawValue)")
    }
  }

  func testMetalExtentMismatchedTransformsMatchCPUBytes() throws {
    let current = try tensor(
      name: "color",
      channels: 3,
      width: 2,
      height: 1,
      values: [Float](repeating: 0.5, count: 6)
    )
    let history = try tensor(
      name: "history",
      channels: 3,
      width: 4,
      height: 1,
      values: [
        0, 0.5, 0.5,
        0.5, 0.5, 0.5,
        1, 0.5, 0.5,
        0.5, 0.5, 0.5,
      ]
    )
    let motion = try tensor(
      name: "motion",
      channels: 2,
      width: 4,
      height: 1,
      values: [
        0, 0,
        0.5, 0,
        0, 0,
        -0.5, 0,
      ]
    )
    let depth = try tensor(
      name: "depth",
      channels: 1,
      width: 2,
      height: 1,
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
    let motionTransform = try NeuralRenderingTextureTransform(
      baseX: 0,
      baseY: 0,
      extentWidth: 4,
      extentHeight: 1,
      resourceWidth: 4,
      resourceHeight: 1
    )
    let cpu = try NeuralRenderingTemporalReferencePreprocessor.makeFeatureTensor(
      currentColor: current,
      historyColor: history,
      historyTransform: historyTransform,
      normalizedMotion: motion,
      motionTransform: motionTransform,
      depth: depth,
      noiseFrameIndex: 1
    )

    let metal = try MetalNeuralRenderingTemporalFeaturePreprocessor(
      depthGuideMode: .observedZeroDescriptor,
      historyTransform: historyTransform,
      motionTransform: motionTransform
    ).makeFeatureTensor(
      currentColor: current,
      historyColor: history,
      normalizedMotion: motion,
      depth: depth,
      depthInverted: false,
      noiseFrameIndex: 1
    )

    XCTAssertEqual(metal.descriptor, cpu.descriptor)
    XCTAssertEqual(metal.bytes, cpu.bytes)
  }

  private func tensor(
    name: String,
    channels: Int,
    width: Int,
    height: Int,
    values: [Float]
  ) throws -> HostTensor {
    try HostTensor(
      descriptor: TensorDescriptor(
        name: name,
        shape: [1, height, width, channels],
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: values.withUnsafeBytes { Data($0) }
    )
  }
}
