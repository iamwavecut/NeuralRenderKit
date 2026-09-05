public enum NeuralRenderTemporalCadence: String, Codable, Equatable, Sendable {
  case frameIndependent
  case orderedFrames
  case consecutiveFrames
}

public enum NeuralRenderDiscontinuity: String, Codable, Equatable, Sendable {
  case explicit
  case seek
  case sceneCut
  case modelReload
}

public struct NeuralRenderFrameContext: Codable, Equatable, Sendable {
  public let streamID: UInt64
  public let frameIndex: UInt64
  public let discontinuity: NeuralRenderDiscontinuity?

  public init(
    streamID: UInt64,
    frameIndex: UInt64,
    discontinuity: NeuralRenderDiscontinuity? = nil
  ) {
    self.streamID = streamID
    self.frameIndex = frameIndex
    self.discontinuity = discontinuity
  }
}

public enum NeuralRenderResetReason: Equatable, Sendable {
  case explicit
  case streamStarted
  case streamChanged(previous: UInt64, current: UInt64)
  case discontinuity(NeuralRenderDiscontinuity)
  case inputShapeChanged
  case frameGap(expected: UInt64, actual: UInt64)
  case inferenceFailure(frameIndex: UInt64)
}

public struct NeuralRenderResetRequest: Equatable, Sendable {
  public let streamID: UInt64?
  public let reason: NeuralRenderResetReason

  public init(streamID: UInt64?, reason: NeuralRenderResetReason) {
    self.streamID = streamID
    self.reason = reason
  }
}

public enum TemporalLifecycleError: Error, Equatable, Sendable {
  case missingFrameContext
  case nonIncreasingFrameIndex(previous: UInt64, actual: UInt64)
}

public struct TemporalLifecycleTracker: Sendable {
  private var activeStreamID: UInt64?
  private var lastSuccessfulFrameIndex: UInt64?
  private var inputDescriptors: [TensorDescriptor]?
  private var failedFrameIndex: UInt64?

  public init() {}

  public mutating func prepare(
    cadence: NeuralRenderTemporalCadence,
    context: NeuralRenderFrameContext,
    inputDescriptors currentDescriptors: [TensorDescriptor]
  ) throws -> NeuralRenderResetRequest? {
    guard cadence != .frameIndependent else {
      return nil
    }
    if let failedFrameIndex {
      return NeuralRenderResetRequest(
        streamID: context.streamID,
        reason: .inferenceFailure(frameIndex: failedFrameIndex)
      )
    }
    guard let activeStreamID else {
      return NeuralRenderResetRequest(
        streamID: context.streamID,
        reason: .streamStarted
      )
    }
    if activeStreamID != context.streamID {
      return NeuralRenderResetRequest(
        streamID: context.streamID,
        reason: .streamChanged(
          previous: activeStreamID,
          current: context.streamID
        )
      )
    }
    if let discontinuity = context.discontinuity {
      return NeuralRenderResetRequest(
        streamID: context.streamID,
        reason: .discontinuity(discontinuity)
      )
    }
    if inputDescriptors != currentDescriptors {
      return NeuralRenderResetRequest(
        streamID: context.streamID,
        reason: .inputShapeChanged
      )
    }
    guard let lastSuccessfulFrameIndex else {
      return NeuralRenderResetRequest(
        streamID: context.streamID,
        reason: .streamStarted
      )
    }
    guard context.frameIndex > lastSuccessfulFrameIndex else {
      throw TemporalLifecycleError.nonIncreasingFrameIndex(
        previous: lastSuccessfulFrameIndex,
        actual: context.frameIndex
      )
    }
    if cadence == .consecutiveFrames {
      let expected = lastSuccessfulFrameIndex + 1
      if context.frameIndex != expected {
        return NeuralRenderResetRequest(
          streamID: context.streamID,
          reason: .frameGap(expected: expected, actual: context.frameIndex)
        )
      }
    }
    return nil
  }

  public mutating func commit(
    context: NeuralRenderFrameContext,
    inputDescriptors: [TensorDescriptor]
  ) {
    activeStreamID = context.streamID
    lastSuccessfulFrameIndex = context.frameIndex
    self.inputDescriptors = inputDescriptors
    failedFrameIndex = nil
  }

  public mutating func markInferenceFailure(frameIndex: UInt64) {
    failedFrameIndex = frameIndex
  }

  public mutating func reset() {
    activeStreamID = nil
    lastSuccessfulFrameIndex = nil
    inputDescriptors = nil
    failedFrameIndex = nil
  }
}

public actor TemporalLifecycleBackend: NeuralRenderBackend {
  public nonisolated let temporalCadence: NeuralRenderTemporalCadence

  private let backend: any NeuralRenderBackend
  private var tracker = TemporalLifecycleTracker()

  public init(backend: any NeuralRenderBackend) {
    self.backend = backend
    self.temporalCadence = backend.temporalCadence
  }

  public func render(
    _ request: NeuralRenderRequest
  ) async throws -> NeuralRenderResult {
    guard temporalCadence != .frameIndependent else {
      return try await backend.render(request)
    }
    guard let context = request.temporalContext else {
      throw TemporalLifecycleError.missingFrameContext
    }

    let descriptors = request.inputs.map(\.descriptor)
    if let resetRequest = try tracker.prepare(
      cadence: temporalCadence,
      context: context,
      inputDescriptors: descriptors
    ) {
      await backend.reset(resetRequest)
    }

    do {
      let result = try await backend.render(request)
      tracker.commit(context: context, inputDescriptors: descriptors)
      return result
    } catch {
      tracker.markInferenceFailure(frameIndex: context.frameIndex)
      throw error
    }
  }

  public func reset(sequenceID: UInt64?) async {
    await reset(
      NeuralRenderResetRequest(streamID: sequenceID, reason: .explicit)
    )
  }

  public func reset(_ request: NeuralRenderResetRequest) async {
    tracker.reset()
    await backend.reset(request)
  }
}
