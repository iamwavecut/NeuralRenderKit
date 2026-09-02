import Foundation
import XCTest

@testable import NeuralRenderCore

final class NeuralRenderingTemporalReferenceBackendTests: XCTestCase {
  func testInjectedTemporalPreprocessorFeedsSecondHeadRequest() async throws {
    let headBackend = RecordingHeadBackend()
    let backend = NeuralRenderingTemporalReferenceBackend(
      backend: headBackend,
      temporalPreprocessor: StubTemporalPreprocessor()
    )
    _ = try await backend.render(
      request(frameIndex: 0, color: [0.25, 0.5, 0.75])
    )

    _ = try await backend.render(
      request(frameIndex: 1, color: [0.75, 0.5, 0.25])
    )

    let recorded = await headBackend.inputs
    XCTAssertEqual(featureChannels(7..<10, in: recorded[1]), [0.125, 0.25, 0.375])
  }

  func testInjectedFeatureControlsFeedFirstHeadRequest() async throws {
    let headBackend = RecordingHeadBackend()
    let controls = NeuralRenderingFeatureControls(
      normalizedStyle: 0.5,
      localToneStrength: 0.25,
      localStructureStrength: 0.75,
      automaticMask: NeuralRenderingAutomaticMaskConfiguration(
        skinStructureStrength: -1,
        automaticMaskStructureStrength: 0.125
      )
    )
    let backend = NeuralRenderingTemporalReferenceBackend(
      backend: headBackend,
      temporalPreprocessor: NeuralRenderingCPUTemporalFeaturePreprocessor(
        depthGuideMode: .observedZeroDescriptor,
        historyTransform: nil,
        motionTransform: nil,
        featureControls: controls
      )
    )

    _ = try await backend.render(
      request(frameIndex: 0, color: [0.25, 0.5, 0.75])
    )

    let recorded = await headBackend.inputs
    XCTAssertEqual(featureChannels(10..<15, in: recorded[0]), [
      0.5, 0.25, 1, 0.75, 0.125,
    ])
  }

  func testSecondFrameUsesFirstDisplayResultAsReprojectedHistory() async throws {
    let headBackend = RecordingHeadBackend()
    let backend = NeuralRenderingTemporalReferenceBackend(backend: headBackend)

    let first = try await backend.render(
      request(frameIndex: 0, color: [0.25, 0.5, 0.75])
    )
    let second = try await backend.render(
      request(frameIndex: 1, color: [0.75, 0.5, 0.25])
    )

    XCTAssertEqual(rgb(in: first), [0.25, 0.5, 0.75])
    let secondRGB = rgb(in: second)
    XCTAssertEqual(secondRGB[0], 0.565_063_5, accuracy: 0.000_001)
    XCTAssertEqual(secondRGB[1], 0.5, accuracy: 0.000_001)
    XCTAssertEqual(secondRGB[2], 0.434_936_52, accuracy: 0.000_001)
    let recorded = await headBackend.inputs
    XCTAssertEqual(recorded.count, 2)
    XCTAssertEqual(
      featureChannels(4..<10, in: recorded[1]),
      [
        0.03125, 0, -0.03125,
        -0.03125, 0, 0.03125,
      ])
  }

  func testControlMaskAppearsWithoutResetAndGatesCompletedTemporalRGB() async throws {
    let headBackend = RecordingHeadBackend()
    let backend = NeuralRenderingTemporalReferenceBackend(backend: headBackend)
    _ = try await backend.render(
      request(frameIndex: 0, color: [0.25, 0.5, 0.75])
    )

    let second = try await backend.render(
      request(
        frameIndex: 1,
        color: [0.75, 0.5, 0.25],
        controlMask: [0.25, 0.5, 0.75]
      )
    )

    let secondRGB = rgb(in: second)
    XCTAssertEqual(secondRGB[0], 0.703_765_87, accuracy: 0.000_001)
    XCTAssertEqual(secondRGB[1], 0.5, accuracy: 0.000_001)
    XCTAssertEqual(secondRGB[2], 0.296_234_13, accuracy: 0.000_001)
    let recorded = await headBackend.inputs
    XCTAssertEqual(featureChannels(11..<15, in: recorded[1]), [0.5, 0.75, 0, 0])
    let resets = await headBackend.resets
    XCTAssertEqual(resets, [
      NeuralRenderResetRequest(streamID: 9, reason: .streamStarted),
    ])
  }

  func testFrameGapResetsHistoryBeforeRendering() async throws {
    let headBackend = RecordingHeadBackend()
    let backend = NeuralRenderingTemporalReferenceBackend(backend: headBackend)
    _ = try await backend.render(
      request(frameIndex: 0, color: [0.25, 0.5, 0.75])
    )

    let afterGap = try await backend.render(
      request(frameIndex: 2, color: [0.75, 0.5, 0.25])
    )

    XCTAssertEqual(rgb(in: afterGap), [0.75, 0.5, 0.25])
    let recorded = await headBackend.inputs
    XCTAssertEqual(
      featureChannels(4..<10, in: recorded[1]),
      [
        0.03125, 0, -0.03125,
        0.03125, 0, -0.03125,
      ])
    let resets = await headBackend.resets
    XCTAssertEqual(
      resets,
      [
        NeuralRenderResetRequest(streamID: 9, reason: .streamStarted),
        NeuralRenderResetRequest(
          streamID: 9,
          reason: .frameGap(expected: 1, actual: 2)
        ),
      ]
    )
  }

  func testMissingDepthIsRejectedBeforeHeadBackend() async throws {
    let headBackend = RecordingHeadBackend()
    let backend = NeuralRenderingTemporalReferenceBackend(backend: headBackend)
    let request = try NeuralRenderRequest(
      sequenceID: 0,
      temporalContext: NeuralRenderFrameContext(streamID: 1, frameIndex: 0),
      inputs: [
        try tensor(name: "color", channels: 3, values: [0, 0, 0]),
        try tensor(name: "motion", channels: 2, values: [0, 0]),
      ]
    )

    do {
      _ = try await backend.render(request)
      XCTFail("Expected missing depth")
    } catch {
      XCTAssertEqual(
        error as? NeuralRenderingTemporalReferenceBackendError,
        .missingInput("depth")
      )
    }
    let inputs = await headBackend.inputs
    XCTAssertTrue(inputs.isEmpty)
  }

  func testVendorGeometryPadsHeadInputAndCropsDisplayOutputByDefault() async throws {
    let headBackend = RecordingHeadBackend()
    let backend = NeuralRenderingTemporalReferenceBackend(backend: headBackend)

    let first = try await backend.render(
      request(frameIndex: 0, color: [0.25, 0.5, 0.75])
    )
    let second = try await backend.render(
      request(frameIndex: 1, color: [0.75, 0.5, 0.25])
    )

    let recorded = await headBackend.inputs
    XCTAssertEqual(
      recorded.map(\.descriptor.shape),
      [[1, 320, 320, 16], [1, 320, 320, 16]]
    )
    XCTAssertEqual(first.output(named: "color")?.descriptor.shape, [1, 1, 1, 3])
    XCTAssertEqual(second.output(named: "color")?.descriptor.shape, [1, 1, 1, 3])
    XCTAssertEqual(rgb(in: first), [0.25, 0.5, 0.75])
    let paddedPixel = 16 * (5 * 320 + 7)
    XCTAssertEqual(
      featureChannels(paddedPixel + 3..<paddedPixel + 10, in: recorded[1]),
      [1, 0.03125, 0, -0.03125, -0.03125, 0, 0.03125]
    )
    let noise = NeuralRenderingFirstFramePreprocessor.deterministicNoise(
      x: 7, y: 5, frameIndex: 1
    )
    XCTAssertEqual(
      featureChannels(paddedPixel..<paddedPixel + 3, in: recorded[1]),
      [noise.0, noise.1, noise.2]
    )
  }

  func testMatchOutputGeometryFeedsLogicalFrameToHead() async throws {
    let headBackend = RecordingHeadBackend()
    let backend = NeuralRenderingTemporalReferenceBackend(
      backend: headBackend,
      geometry: .matchOutput
    )

    _ = try await backend.render(request(frameIndex: 0, color: [0.25, 0.5, 0.75]))
    _ = try await backend.render(request(frameIndex: 1, color: [0.75, 0.5, 0.25]))

    let recorded = await headBackend.inputs
    XCTAssertEqual(recorded.map(\.descriptor.shape), [[1, 1, 1, 16], [1, 1, 1, 16]])
  }

  private func request(
    frameIndex: UInt64,
    color: [Float],
    controlMask: [Float]? = nil
  ) throws -> NeuralRenderRequest {
    var inputs = [
      try tensor(name: "color", channels: 3, values: color),
      try tensor(name: "depth", channels: 1, values: [1]),
      try tensor(name: "motion", channels: 2, values: [0, 0]),
    ]
    if let controlMask {
      inputs.append(
        try tensor(name: "controlMask", channels: 3, values: controlMask)
      )
    }
    return try NeuralRenderRequest(
      sequenceID: frameIndex,
      temporalContext: NeuralRenderFrameContext(streamID: 9, frameIndex: frameIndex),
      inputs: inputs
    )
  }

  private func tensor(
    name: String,
    channels: Int,
    values: [Float]
  ) throws -> HostTensor {
    try HostTensor(
      descriptor: TensorDescriptor(
        name: name,
        shape: [1, 1, 1, channels],
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: values.withUnsafeBytes { Data($0) }
    )
  }

  private func rgb(in result: NeuralRenderResult) -> [Float] {
    let output = result.output(named: "color")!
    return output.bytes.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
  }

  private func featureChannels(
    _ channels: Range<Int>,
    in tensor: HostTensor
  ) -> [Float] {
    let values = tensor.bytes.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
    return Array(values[channels])
  }
}

private struct StubTemporalPreprocessor: NeuralRenderingTemporalFeaturePreprocessing {
  func makeFeatureTensor(
    currentColor _: HostTensor,
    historyColor _: HostTensor,
    controlMask _: HostTensor?,
    normalizedMotion _: HostTensor,
    depth _: HostTensor,
    depthInverted _: Bool,
    noiseFrameIndex _: UInt32
  ) throws -> HostTensor {
    var values = [Float](repeating: 0, count: 16)
    values[7] = 0.125
    values[8] = 0.25
    values[9] = 0.375
    return try HostTensor(
      descriptor: TensorDescriptor(
        name: "color",
        shape: [1, 1, 1, 16],
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: values.withUnsafeBytes { Data($0) }
    )
  }
}

private actor RecordingHeadBackend: NeuralRenderBackend {
  private(set) var inputs: [HostTensor] = []
  private(set) var resets: [NeuralRenderResetRequest] = []

  func render(_ request: NeuralRenderRequest) async throws -> NeuralRenderResult {
    let input = request.input(named: "color")!
    inputs.append(input)
    let height = input.descriptor.shape[1]
    let width = input.descriptor.shape[2]
    let descriptor = try TensorDescriptor(
      name: "color",
      shape: [1, height, width, 4],
      dataType: .float32,
      layout: .nhwc
    )
    let values = [Float](repeating: 0, count: height * width * 4)
    return try NeuralRenderResult(
      outputs: [
        HostTensor(
          descriptor: descriptor,
          bytes: values.withUnsafeBytes { Data($0) }
        )
      ],
      timing: NeuralRenderTiming(executionNanoseconds: 1)
    )
  }

  func reset(sequenceID _: UInt64?) async {}

  func reset(_ request: NeuralRenderResetRequest) async {
    resets.append(request)
  }
}
