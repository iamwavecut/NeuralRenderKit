import Foundation

public enum NeuralRenderingFirstFrameBackendError: Error, Equatable, Sendable {
  case missingInput(String)
  case missingOutput(String)
}

/// How the stateless image path maps a logical frame onto the network extent.
public enum NeuralRenderingNetworkGeometryPolicy: String, CaseIterable, Equatable, Sendable {
  /// Recovered vendor alignment: minimum `320`, multiples of `64`, extended
  /// right/bottom by reflection then clamp, cropped back after the head.
  case vendorAligned = "vendor-aligned"
  /// Feed the logical frame to the head unchanged; the head must accept it.
  case matchOutput = "match-output"

  /// Resolves the concrete network extent for one logical frame size.
  public func resolve(
    outputWidth: Int,
    outputHeight: Int
  ) throws -> NeuralRenderingNetworkGeometry {
    switch self {
    case .vendorAligned:
      try .vendorAligned(outputWidth: outputWidth, outputHeight: outputHeight)
    case .matchOutput:
      try NeuralRenderingNetworkGeometry(
        outputWidth: outputWidth,
        outputHeight: outputHeight,
        networkWidth: outputWidth,
        networkHeight: outputHeight
      )
    }
  }
}

/// Complete configuration of the stateless (no-history) rendering path.
public struct NeuralRenderingFirstFrameConfiguration: Equatable, Sendable {
  public var featureControls: NeuralRenderingFeatureControls
  /// Final effect intensity in `0...1`; multiplied by an explicit ControlMask red
  /// channel when one is supplied.
  public var intensity: Float
  /// Deterministic-noise frame index; stills use `0`.
  public var noiseFrameIndex: UInt32
  public var geometry: NeuralRenderingNetworkGeometryPolicy

  public init(
    featureControls: NeuralRenderingFeatureControls = .init(),
    intensity: Float = 1,
    noiseFrameIndex: UInt32 = 0,
    geometry: NeuralRenderingNetworkGeometryPolicy = .vendorAligned
  ) {
    self.featureControls = featureControls
    self.intensity = intensity
    self.noiseFrameIndex = noiseFrameIndex
    self.geometry = geometry
  }

  public init(profile: NeuralRenderingControlProfile) {
    self.init(
      featureControls: profile.featureControls,
      intensity: profile.intensity
    )
  }
}

/// Stateless single-image neural rendering around any four-channel head backend.
///
/// The request carries `color` (`[1, H, W, 3]` float32 NHWC in `0...1`) and an
/// optional `controlMask` of the same shape. The backend builds the recovered
/// sixteen-feature tensor on the network extent, runs the head, crops the result
/// back to the logical frame, and composes the first-frame display RGB. It is
/// frame independent, so it plugs into `mlxdlss render-image` and
/// `LatestFrameScheduler` for still images or history-free per-frame video.
public struct NeuralRenderingFirstFrameBackend: NeuralRenderBackend {
  public let temporalCadence: NeuralRenderTemporalCadence = .frameIndependent
  public let configuration: NeuralRenderingFirstFrameConfiguration
  private let head: any NeuralRenderBackend

  public init(
    head: any NeuralRenderBackend,
    configuration: NeuralRenderingFirstFrameConfiguration = .init(profile: .standard)
  ) {
    self.head = head
    self.configuration = configuration
  }

  public init(head: any NeuralRenderBackend, profile: NeuralRenderingControlProfile) {
    self.init(head: head, configuration: .init(profile: profile))
  }

  public func render(_ request: NeuralRenderRequest) async throws -> NeuralRenderResult {
    guard let color = request.input(named: "color") else {
      throw NeuralRenderingFirstFrameBackendError.missingInput("color")
    }
    let controlMask = request.input(named: "controlMask")
    let shape = color.descriptor.shape
    guard shape.count == 4,
      shape[0] == 1,
      shape[3] == 3,
      color.descriptor.dataType == .float32,
      color.descriptor.layout == .nhwc
    else {
      throw NeuralRenderingPreprocessorError.expectedFloat32NHWCRGB(
        shape: shape,
        dataType: color.descriptor.dataType,
        layout: color.descriptor.layout
      )
    }
    let geometry = try configuration.geometry.resolve(
      outputWidth: shape[2],
      outputHeight: shape[1]
    )
    let controls = configuration.featureControls
    let features = try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(
      from: color,
      noiseFrameIndex: configuration.noiseFrameIndex,
      geometry: geometry,
      normalizedStyle: controls.normalizedStyle,
      localToneStrength: controls.localToneStrength,
      localStructureStrength: controls.localStructureStrength,
      automaticMask: controls.automaticMask,
      controlMask: controlMask
    )
    let headResult = try await head.render(
      NeuralRenderRequest(sequenceID: request.sequenceID, inputs: [features])
    )
    guard let headOutput = headResult.output(named: "color") else {
      throw NeuralRenderingFirstFrameBackendError.missingOutput("color")
    }
    let cropped = try geometry.cropOutput(headOutput)
    let output = try NeuralRenderingFirstFramePostprocessor.compose(
      head: cropped,
      over: color,
      controlMask: controlMask,
      intensity: configuration.intensity
    )
    return try NeuralRenderResult(outputs: [output], timing: headResult.timing)
  }

  public func reset(sequenceID: UInt64?) async {
    await head.reset(sequenceID: sequenceID)
  }

  public func reset(_ request: NeuralRenderResetRequest) async {
    await head.reset(request)
  }
}
