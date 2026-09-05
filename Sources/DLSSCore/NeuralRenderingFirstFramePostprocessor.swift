import Foundation

public enum NeuralRenderingPostprocessorError: Error, Equatable, Sendable {
  case expectedFloat32NHWC(
    channels: Int,
    shape: [Int],
    dataType: TensorDataType,
    layout: TensorLayout
  )
  case spatialShapeMismatch(head: [Int], color: [Int])
}

/// Converts the recovered four-channel neural head into first-frame display RGB.
public enum NeuralRenderingFirstFramePostprocessor {
  public static func compose(
    head: HostTensor,
    over color: HostTensor,
    controlMask: HostTensor? = nil,
    intensity: Float = 1
  ) throws -> HostTensor {
    try require(head, channels: 4)
    try require(color, channels: 3)
    if let controlMask {
      try require(controlMask, channels: 3)
    }
    guard head.descriptor.shape[0..<3].elementsEqual(
      color.descriptor.shape[0..<3]
    ) else {
      throw NeuralRenderingPostprocessorError.spatialShapeMismatch(
        head: head.descriptor.shape,
        color: color.descriptor.shape
      )
    }
    if let controlMask,
       !controlMask.descriptor.shape[0..<3].elementsEqual(
        color.descriptor.shape[0..<3]
       ) {
      throw NeuralRenderingPostprocessorError.spatialShapeMismatch(
        head: controlMask.descriptor.shape,
        color: color.descriptor.shape
      )
    }

    let headValues = floatValues(in: head)
    let colorValues = floatValues(in: color)
    let controlMaskValues = controlMask.map(floatValues(in:))
    var output: [Float] = []
    output.reserveCapacity(colorValues.count)
    for pixel in 0..<(colorValues.count / 3) {
      let colorOffset = pixel * 3
      let headOffset = pixel * 4
      let blend = min(
        1,
        max(0, (controlMaskValues?[colorOffset] ?? 1) * intensity)
      )
      for channel in 0..<3 {
        let residual = Float(Float16(headValues[headOffset + channel])) * 0.25
        let original = colorValues[colorOffset + channel]
        let predicted = min(1, max(0, original + residual))
        output.append(
          min(1, max(0, original + blend * (predicted - original)))
        )
      }
    }

    let descriptor = try TensorDescriptor(
      name: "color",
      shape: color.descriptor.shape,
      dataType: .float32,
      layout: .nhwc
    )
    return try HostTensor(
      descriptor: descriptor,
      bytes: output.withUnsafeBytes { Data($0) }
    )
  }

  fileprivate static func require(_ tensor: HostTensor, channels: Int) throws {
    let descriptor = tensor.descriptor
    guard descriptor.shape.count == 4,
          descriptor.shape[0] == 1,
          descriptor.shape[3] == channels,
          descriptor.dataType == .float32,
          descriptor.layout == .nhwc else {
      throw NeuralRenderingPostprocessorError.expectedFloat32NHWC(
        channels: channels,
        shape: descriptor.shape,
        dataType: descriptor.dataType,
        layout: descriptor.layout
      )
    }
  }

  fileprivate static func floatValues(in tensor: HostTensor) -> [Float] {
    tensor.bytes.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
  }
}

/// CPU temporal composition using the reprojected history encoded in features 7–9.
public enum NeuralRenderingTemporalReferencePostprocessor {
  public static func compose(
    head: HostTensor,
    currentColor: HostTensor,
    features: HostTensor,
    blendScale: Float = 0.739_746_093_75,
    controlMask: HostTensor? = nil,
    intensity: Float = 1
  ) throws -> HostTensor {
    try NeuralRenderingFirstFramePostprocessor.require(head, channels: 4)
    try NeuralRenderingFirstFramePostprocessor.require(currentColor, channels: 3)
    try NeuralRenderingFirstFramePostprocessor.require(features, channels: 16)
    if let controlMask {
      try NeuralRenderingFirstFramePostprocessor.require(controlMask, channels: 3)
    }
    let spatialShape = Array(currentColor.descriptor.shape.prefix(3))
    for tensor in [head, features] + [controlMask].compactMap({ $0 }) where
      !tensor.descriptor.shape.prefix(3).elementsEqual(spatialShape) {
      throw NeuralRenderingPostprocessorError.spatialShapeMismatch(
        head: tensor.descriptor.shape,
        color: currentColor.descriptor.shape
      )
    }

    let headValues = NeuralRenderingFirstFramePostprocessor.floatValues(in: head)
    let colorValues = NeuralRenderingFirstFramePostprocessor.floatValues(
      in: currentColor
    )
    let featureValues = NeuralRenderingFirstFramePostprocessor.floatValues(
      in: features
    )
    let controlMaskValues = controlMask.map(
      NeuralRenderingFirstFramePostprocessor.floatValues(in:)
    )
    let appliesEffectBlend = controlMaskValues != nil || intensity != 1
    let recoveredBlendScale = Float(Float16(blendScale))
    var output: [Float] = []
    output.reserveCapacity(colorValues.count)
    for pixel in 0..<(colorValues.count / 3) {
      let headOffset = pixel * 4
      let colorOffset = pixel * 3
      let featureOffset = pixel * 16
      let logit = Float(Float16(headValues[headOffset + 3]))
      let alpha = min(1, max(0, sigmoid(logit) * recoveredBlendScale))
      for channel in 0..<3 {
        let residual = Float(Float16(headValues[headOffset + channel])) * 0.25
        let predicted = min(
          1,
          max(0, colorValues[colorOffset + channel] + residual)
        )
        let history = featureValues[featureOffset + 7 + channel] * 8 + 0.5
        let temporal = predicted + alpha * (history - predicted)
        if appliesEffectBlend {
          let blend = min(
            1,
            max(0, (controlMaskValues?[colorOffset] ?? 1) * intensity)
          )
          output.append(
            min(
              1,
              max(0, colorValues[colorOffset + channel]
                + blend * (temporal - colorValues[colorOffset + channel]))
            )
          )
        } else {
          output.append(temporal)
        }
      }
    }

    let descriptor = try TensorDescriptor(
      name: "color",
      shape: currentColor.descriptor.shape,
      dataType: .float32,
      layout: .nhwc
    )
    return try HostTensor(
      descriptor: descriptor,
      bytes: output.withUnsafeBytes { Data($0) }
    )
  }

  private static func sigmoid(_ value: Float) -> Float {
    1 / (1 + exp(-value))
  }
}
