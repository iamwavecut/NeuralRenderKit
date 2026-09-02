import Foundation
import XCTest

@testable import NeuralRenderCore

final class TemporalLifecycleTests: XCTestCase {
  func testConsecutiveBackendResetsAtStartGapAndStreamChange() async throws {
    let backend = RecordingTemporalBackend(cadence: .consecutiveFrames)
    let lifecycle = TemporalLifecycleBackend(backend: backend)

    _ = try await lifecycle.render(request(streamID: 7, frameIndex: 0))
    _ = try await lifecycle.render(request(streamID: 7, frameIndex: 1))
    _ = try await lifecycle.render(request(streamID: 7, frameIndex: 3))
    _ = try await lifecycle.render(request(streamID: 8, frameIndex: 0))

    let resets = await backend.resets
    XCTAssertEqual(
      resets,
      [
        NeuralRenderResetRequest(streamID: 7, reason: .streamStarted),
        NeuralRenderResetRequest(
          streamID: 7,
          reason: .frameGap(expected: 2, actual: 3)
        ),
        NeuralRenderResetRequest(
          streamID: 8,
          reason: .streamChanged(previous: 7, current: 8)
        ),
      ]
    )
  }

  func testOrderedBackendAllowsFrameGapWithoutReset() async throws {
    let backend = RecordingTemporalBackend(cadence: .orderedFrames)
    let lifecycle = TemporalLifecycleBackend(backend: backend)

    _ = try await lifecycle.render(request(streamID: 2, frameIndex: 10))
    _ = try await lifecycle.render(request(streamID: 2, frameIndex: 15))

    let resets = await backend.resets
    XCTAssertEqual(
      resets,
      [NeuralRenderResetRequest(streamID: 2, reason: .streamStarted)]
    )
  }

  func testExplicitDiscontinuityAndInputShapeChangeResetBeforeRendering() async throws {
    let backend = RecordingTemporalBackend(cadence: .consecutiveFrames)
    let lifecycle = TemporalLifecycleBackend(backend: backend)

    _ = try await lifecycle.render(request(streamID: 1, frameIndex: 0))
    _ = try await lifecycle.render(
      request(
        streamID: 1,
        frameIndex: 1,
        discontinuity: .sceneCut
      )
    )
    _ = try await lifecycle.render(
      request(streamID: 1, frameIndex: 2, width: 2)
    )

    let resets = await backend.resets
    XCTAssertEqual(
      resets,
      [
        NeuralRenderResetRequest(streamID: 1, reason: .streamStarted),
        NeuralRenderResetRequest(
          streamID: 1,
          reason: .discontinuity(.sceneCut)
        ),
        NeuralRenderResetRequest(streamID: 1, reason: .inputShapeChanged),
      ]
    )
  }

  func testTemporalBackendRejectsMissingOrNonIncreasingFrameContext() async throws {
    let backend = RecordingTemporalBackend(cadence: .consecutiveFrames)
    let lifecycle = TemporalLifecycleBackend(backend: backend)

    do {
      _ = try await lifecycle.render(requestWithoutTemporalContext())
      XCTFail("Expected missing temporal context")
    } catch {
      XCTAssertEqual(error as? TemporalLifecycleError, .missingFrameContext)
    }

    _ = try await lifecycle.render(request(streamID: 4, frameIndex: 5))
    do {
      _ = try await lifecycle.render(request(streamID: 4, frameIndex: 5))
      XCTFail("Expected non-increasing frame rejection")
    } catch {
      XCTAssertEqual(
        error as? TemporalLifecycleError,
        .nonIncreasingFrameIndex(previous: 5, actual: 5)
      )
    }
  }

  func testFailureForcesResetBeforeNextFrame() async throws {
    let backend = RecordingTemporalBackend(
      cadence: .consecutiveFrames,
      failingFrameIndex: 1
    )
    let lifecycle = TemporalLifecycleBackend(backend: backend)
    _ = try await lifecycle.render(request(streamID: 9, frameIndex: 0))

    do {
      _ = try await lifecycle.render(request(streamID: 9, frameIndex: 1))
      XCTFail("Expected synthetic failure")
    } catch {
      XCTAssertEqual(error as? TemporalTestError, .synthetic)
    }
    _ = try await lifecycle.render(request(streamID: 9, frameIndex: 2))

    let resets = await backend.resets
    XCTAssertEqual(
      resets,
      [
        NeuralRenderResetRequest(streamID: 9, reason: .streamStarted),
        NeuralRenderResetRequest(
          streamID: 9,
          reason: .inferenceFailure(frameIndex: 1)
        ),
      ]
    )
  }

  private func request(
    streamID: UInt64,
    frameIndex: UInt64,
    width: Int = 1,
    discontinuity: NeuralRenderDiscontinuity? = nil
  ) throws -> NeuralRenderRequest {
    try NeuralRenderRequest(
      sequenceID: frameIndex,
      temporalContext: NeuralRenderFrameContext(
        streamID: streamID,
        frameIndex: frameIndex,
        discontinuity: discontinuity
      ),
      inputs: [tensor(width: width)]
    )
  }

  private func requestWithoutTemporalContext() throws -> NeuralRenderRequest {
    try NeuralRenderRequest(sequenceID: 0, inputs: [tensor(width: 1)])
  }

  private func tensor(width: Int) throws -> HostTensor {
    let descriptor = try TensorDescriptor(
      name: "color",
      shape: [1, 1, width, 3],
      dataType: .float32,
      layout: .nhwc
    )
    let values = [Float](repeating: 0, count: width * 3)
    return try HostTensor(
      descriptor: descriptor,
      bytes: values.withUnsafeBytes { Data($0) }
    )
  }
}

private enum TemporalTestError: Error, Equatable {
  case synthetic
}

private actor RecordingTemporalBackend: NeuralRenderBackend {
  nonisolated let temporalCadence: NeuralRenderTemporalCadence
  private let failingFrameIndex: UInt64?
  private(set) var resets: [NeuralRenderResetRequest] = []

  init(
    cadence: NeuralRenderTemporalCadence,
    failingFrameIndex: UInt64? = nil
  ) {
    self.temporalCadence = cadence
    self.failingFrameIndex = failingFrameIndex
  }

  func render(_ request: NeuralRenderRequest) async throws -> NeuralRenderResult {
    if request.temporalContext?.frameIndex == failingFrameIndex {
      throw TemporalTestError.synthetic
    }
    return try NeuralRenderResult(
      outputs: request.inputs,
      timing: NeuralRenderTiming(executionNanoseconds: 1)
    )
  }

  func reset(sequenceID _: UInt64?) async {}

  func reset(_ request: NeuralRenderResetRequest) async {
    resets.append(request)
  }
}
