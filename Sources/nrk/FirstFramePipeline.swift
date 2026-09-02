import Foundation
import NeuralRenderCore

/// Options of the history-free RGB path shared by `run --input-format rgb-first-frame`
/// and `render-image`.
struct FirstFrameOptions {
  var featureControls: NeuralRenderingFeatureControls
  var intensity: Float = 1
  var noiseFrameIndex: UInt32 = 0
  var geometry: NeuralRenderingNetworkGeometryPolicy = .vendorAligned
  var processingScale: Float = 1
  var detailStrength: Float = 1
  var colourStrength: Float = 1
  var detailRadius: Float = 4
}

struct FirstFrameRender {
  let output: HostTensor
  let features: HostTensor
  let networkShape: [Int]
  let preprocessingNanoseconds: UInt64
  let executionNanoseconds: UInt64
  let postprocessingNanoseconds: UInt64
}

/// Resample to the processing scale, build the 16-channel features on the
/// vendor-aligned extent, run the head, crop, compose the residual over the
/// frame, resample back and apply the detail/colour split against the source.
enum FirstFramePipeline {
  static func render(
    source: HostTensor,
    controlMask: HostTensor?,
    options: FirstFrameOptions,
    backend: any NeuralRenderBackend
  ) async throws -> FirstFrameRender {
    let logicalHeight = source.descriptor.shape[1]
    let logicalWidth = source.descriptor.shape[2]
    if controlMask != nil, options.processingScale != 1 {
      throw CLIError.usage("a control mask requires --processing-scale 1")
    }
    let preprocessingStart = DispatchTime.now().uptimeNanoseconds
    var processing = source
    if options.processingScale != 1 {
      processing = try NeuralRenderingDetailComposition.resample(
        source,
        width: Int((Float(logicalWidth) * options.processingScale).rounded()),
        height: Int((Float(logicalHeight) * options.processingScale).rounded())
      )
    }
    let geometry = try options.geometry.resolve(
      outputWidth: processing.descriptor.shape[2],
      outputHeight: processing.descriptor.shape[1]
    )
    let features = try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(
      from: processing,
      noiseFrameIndex: options.noiseFrameIndex,
      geometry: geometry,
      normalizedStyle: options.featureControls.normalizedStyle,
      localToneStrength: options.featureControls.localToneStrength,
      localStructureStrength: options.featureControls.localStructureStrength,
      automaticMask: options.featureControls.automaticMask,
      controlMask: controlMask
    )
    let preprocessingNanoseconds = DispatchTime.now().uptimeNanoseconds - preprocessingStart
    let request = try NeuralRenderRequest(sequenceID: 1, inputs: [features])
    let result = try await backend.render(request)
    guard let head = result.output(named: "color") else {
      throw CLIError.missingOutput("color")
    }
    let postprocessingStart = DispatchTime.now().uptimeNanoseconds
    var composed = try NeuralRenderingFirstFramePostprocessor.compose(
      head: try geometry.cropOutput(head),
      over: processing,
      controlMask: controlMask,
      intensity: options.intensity
    )
    if options.processingScale != 1 {
      composed = try NeuralRenderingDetailComposition.resample(
        composed,
        width: logicalWidth,
        height: logicalHeight
      )
    }
    let output = try NeuralRenderingDetailComposition.compose(
      input: source,
      output: composed,
      detailStrength: options.detailStrength,
      colourStrength: options.colourStrength,
      radius: options.detailRadius
    )
    let postprocessingNanoseconds = DispatchTime.now().uptimeNanoseconds - postprocessingStart
    return FirstFrameRender(
      output: output,
      features: features,
      networkShape: features.descriptor.shape,
      preprocessingNanoseconds: preprocessingNanoseconds,
      executionNanoseconds: result.timing.executionNanoseconds,
      postprocessingNanoseconds: postprocessingNanoseconds
    )
  }
}
