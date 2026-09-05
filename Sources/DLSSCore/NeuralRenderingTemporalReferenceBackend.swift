public enum NeuralRenderingTemporalReferenceBackendError: Error, Equatable, Sendable {
  case missingInput(String)
  case missingOutput(String)
}

/// Pluggable temporal reference around a backend that executes the four-channel head.
///
/// Requests contain `color` RGB, `motion` normalized UV offsets, and `depth`.
/// Display output is retained as history and reset with consecutive-frame lifecycle
/// boundaries. Vendor-specific motion units remain an adapter outside this type.
public actor NeuralRenderingTemporalReferenceBackend: NeuralRenderBackend {
  public nonisolated let temporalCadence: NeuralRenderTemporalCadence = .consecutiveFrames

  private let backend: any NeuralRenderBackend
  private let temporalPreprocessor: any NeuralRenderingTemporalFeaturePreprocessing
  private let depthInverted: Bool
  private let blendScale: Float
  private let controlMaskIntensity: Float
  private let geometryPolicy: NeuralRenderingNetworkGeometryPolicy
  private var tracker = TemporalLifecycleTracker()
  private var history: HostTensor?
  private var noiseFrameIndex: UInt32 = 0

  /// - Parameter geometry: how logical frames map onto the network extent.
  ///   `vendorAligned` (default) pads every frame to the recovered minimum
  ///   `320` / multiple-of-`64` extent before the head and crops afterwards;
  ///   `matchOutput` feeds the logical frame to the head unchanged.
  public init(
    backend: any NeuralRenderBackend,
    depthInverted: Bool = false,
    blendScale: Float = 0.739_746_093_75,
    controlMaskIntensity: Float = 1,
    temporalPreprocessor: any NeuralRenderingTemporalFeaturePreprocessing =
      NeuralRenderingCPUTemporalFeaturePreprocessor(),
    geometry: NeuralRenderingNetworkGeometryPolicy = .vendorAligned
  ) {
    self.backend = backend
    self.temporalPreprocessor = temporalPreprocessor
    self.depthInverted = depthInverted
    self.blendScale = blendScale
    self.controlMaskIntensity = controlMaskIntensity
    self.geometryPolicy = geometry
  }

  public func render(_ request: NeuralRenderRequest) async throws -> NeuralRenderResult {
    guard let context = request.temporalContext else {
      throw TemporalLifecycleError.missingFrameContext
    }
    let color = try requiredInput("color", in: request)
    let motion = try requiredInput("motion", in: request)
    let depth = try requiredInput("depth", in: request)
    let controlMask = request.input(named: "controlMask")
    let descriptors = [color.descriptor, motion.descriptor, depth.descriptor]
    if let resetRequest = try tracker.prepare(
      cadence: temporalCadence,
      context: context,
      inputDescriptors: descriptors
    ) {
      clearReferenceState()
      await backend.reset(resetRequest)
    }

    let previousHistory = history
    do {
      let colorShape = color.descriptor.shape
      guard colorShape.count == 4 else {
        throw NeuralRenderingPreprocessorError.expectedFloat32NHWCRGB(
          shape: colorShape,
          dataType: color.descriptor.dataType,
          layout: color.descriptor.layout
        )
      }
      let geometry = try geometryPolicy.resolve(
        outputWidth: colorShape[2],
        outputHeight: colorShape[1]
      )
      let features: HostTensor
      let networkFeatures: HostTensor
      if let previousHistory {
        features = try temporalPreprocessor.makeFeatureTensor(
          currentColor: color,
          historyColor: previousHistory,
          controlMask: controlMask,
          normalizedMotion: motion,
          depth: depth,
          depthInverted: depthInverted,
          noiseFrameIndex: noiseFrameIndex
        )
        networkFeatures = try geometry.extendFeatureTensor(
          features,
          noiseFrameIndex: noiseFrameIndex
        )
      } else {
        let controls = temporalPreprocessor.featureControls
        networkFeatures = try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(
          from: color,
          noiseFrameIndex: noiseFrameIndex,
          geometry: geometry,
          normalizedStyle: controls.normalizedStyle,
          localToneStrength: controls.localToneStrength,
          localStructureStrength: controls.localStructureStrength,
          automaticMask: controls.automaticMask,
          controlMask: controlMask
        )
        features = networkFeatures
      }
      let headRequest = try NeuralRenderRequest(
        sequenceID: request.sequenceID,
        inputs: [networkFeatures]
      )
      let headResult = try await backend.render(headRequest)
      guard let networkHead = headResult.output(named: "color") else {
        throw NeuralRenderingTemporalReferenceBackendError.missingOutput("color")
      }
      let head = try geometry.cropOutput(networkHead)
      let output: HostTensor
      if previousHistory == nil {
        output = try NeuralRenderingFirstFramePostprocessor.compose(
          head: head,
          over: color,
          controlMask: controlMask,
          intensity: controlMaskIntensity
        )
      } else {
        output = try NeuralRenderingTemporalReferencePostprocessor.compose(
          head: head,
          currentColor: color,
          features: features,
          blendScale: blendScale,
          controlMask: controlMask,
          intensity: controlMaskIntensity
        )
      }
      history = output
      noiseFrameIndex &+= 1
      tracker.commit(context: context, inputDescriptors: descriptors)
      return try NeuralRenderResult(
        outputs: [output],
        timing: headResult.timing
      )
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
    clearReferenceState()
    await backend.reset(request)
  }

  private func requiredInput(
    _ name: String,
    in request: NeuralRenderRequest
  ) throws -> HostTensor {
    guard let input = request.input(named: name) else {
      throw NeuralRenderingTemporalReferenceBackendError.missingInput(name)
    }
    return input
  }

  private func clearReferenceState() {
    history = nil
    noiseFrameIndex = 0
  }
}
