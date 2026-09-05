@preconcurrency import CoreML
import Foundation
import DLSSCore

public enum CoreMLComputeUnits: String, CaseIterable, Sendable {
  case cpuOnly = "cpu"
  case cpuAndGPU = "cpu-gpu"
  case cpuAndNeuralEngine = "cpu-neural-engine"
  case all

  var coreMLValue: MLComputeUnits {
    switch self {
    case .cpuOnly:
      .cpuOnly
    case .cpuAndGPU:
      .cpuAndGPU
    case .cpuAndNeuralEngine:
      .cpuAndNeuralEngine
    case .all:
      .all
    }
  }
}

public enum CoreMLHostBridge: String, CaseIterable, Sendable {
  case automatic
  case multiArray = "multi-array"
  case mlTensor = "ml-tensor"
}

public struct CoreMLBackendConfiguration: Equatable, Sendable {
  public let computeUnits: CoreMLComputeUnits
  public let hostBridge: CoreMLHostBridge
  public let modelInputName: String
  public let modelOutputName: String
  public let hostTensorName: String

  public init(
    computeUnits: CoreMLComputeUnits = .cpuAndGPU,
    hostBridge: CoreMLHostBridge = .automatic,
    modelInputName: String = "color",
    modelOutputName: String = "restored",
    hostTensorName: String = "color"
  ) {
    self.computeUnits = computeUnits
    self.hostBridge = hostBridge
    self.modelInputName = modelInputName
    self.modelOutputName = modelOutputName
    self.hostTensorName = hostTensorName
  }
}

public struct CoreMLModelMetadata: Equatable, Sendable {
  public let identifier: String?
  public let architecture: String?
  public let checkpointSHA256: String?
  public let computePrecision: String?
  public let reductionPrecision: String?
  public let ioType: String?

  public init(
    identifier: String? = nil,
    architecture: String? = nil,
    checkpointSHA256: String? = nil,
    computePrecision: String? = nil,
    reductionPrecision: String? = nil,
    ioType: String? = nil
  ) {
    self.identifier = identifier
    self.architecture = architecture
    self.checkpointSHA256 = checkpointSHA256
    self.computePrecision = computePrecision
    self.reductionPrecision = reductionPrecision
    self.ioType = ioType
  }
}

public enum CoreMLBackendError: Error, Equatable, Sendable {
  case modelNotFound(URL)
  case unsupportedModelExtension(String)
  case missingModelInput(String)
  case missingModelOutput(String)
  case modelFeatureIsNotMultiArray(String)
  case modelFeatureIsNotImage(String)
  case modelFeatureDataType(name: String, expected: String, actual: String)
  case modelFeatureShape(name: String, shape: [Int])
  case modelInputOutputShapeMismatch(input: [Int], output: [Int])
  case modelImageShape(name: String, width: Int, height: Int)
  case modelInputOutputImageShapeMismatch(
    inputWidth: Int,
    inputHeight: Int,
    outputWidth: Int,
    outputHeight: Int
  )
  case inputCount(expected: Int, actual: Int)
  case missingInput(String)
  case inputDataType(expected: TensorDataType, actual: TensorDataType)
  case inputLayout(expected: TensorLayout, actual: TensorLayout)
  case inputShape(expected: [Int], actual: [Int])
  case predictionMissingOutput(String)
  case pixelBufferFormat(expected: OSType, actual: OSType)
  case pixelBufferShape(
    expectedWidth: Int,
    expectedHeight: Int,
    actualWidth: Int,
    actualHeight: Int
  )
  case predictionOutputIsNotImage(String)
  case predictionOutputDataType(name: String, expected: String, actual: String)
  case predictionOutputShape(name: String, expected: [Int], actual: [Int])
  case nonFiniteOutput(String)
  case tensorBridgeUnavailable
}

struct CoreMLTensorSpecification: Equatable, Sendable {
  let name: String
  let shape: [Int]
  let dataType: MLMultiArrayDataType
}

protocol CoreMLPredicting: Sendable {
  var inputSpecification: CoreMLTensorSpecification { get }
  var outputSpecification: CoreMLTensorSpecification { get }
  var metadata: CoreMLModelMetadata { get }
  func predict(_ input: MLMultiArray) async throws -> MLMultiArray
}

@available(macOS 15.0, *)
protocol CoreMLTensorPredicting: Sendable {
  func predict(_ input: MLTensor) async throws -> MLTensor
}

private final class CoreMLModelPredictor: CoreMLPredicting, CoreMLTensorPredicting,
  @unchecked Sendable
{
  let inputSpecification: CoreMLTensorSpecification
  let outputSpecification: CoreMLTensorSpecification
  let metadata: CoreMLModelMetadata
  private let model: MLModel

  init(modelURL: URL, configuration: CoreMLBackendConfiguration) async throws {
    guard FileManager.default.fileExists(atPath: modelURL.path) else {
      throw CoreMLBackendError.modelNotFound(modelURL)
    }

    let loadURL: URL
    switch modelURL.pathExtension.lowercased() {
    case "mlmodelc":
      loadURL = modelURL
    case "mlmodel", "mlpackage":
      loadURL = try await MLModel.compileModel(at: modelURL)
    default:
      throw CoreMLBackendError.unsupportedModelExtension(modelURL.pathExtension)
    }

    let modelConfiguration = MLModelConfiguration()
    modelConfiguration.computeUnits = configuration.computeUnits.coreMLValue
    self.model = try await MLModel.load(
      contentsOf: loadURL,
      configuration: modelConfiguration
    )
    self.inputSpecification = try Self.specification(
      named: configuration.modelInputName,
      in: model.modelDescription.inputDescriptionsByName,
      missing: .missingModelInput(configuration.modelInputName)
    )
    self.outputSpecification = try Self.specification(
      named: configuration.modelOutputName,
      in: model.modelDescription.outputDescriptionsByName,
      missing: .missingModelOutput(configuration.modelOutputName)
    )
    let creatorDefined =
      model.modelDescription.metadata[.creatorDefinedKey]
      as? [String: String]
    self.metadata = CoreMLModelMetadata(
      identifier: creatorDefined?["com.mlxdlss.identifier"],
      architecture: creatorDefined?["com.mlxdlss.architecture"],
      checkpointSHA256: creatorDefined?["com.mlxdlss.checkpoint_sha256"],
      computePrecision: creatorDefined?["com.mlxdlss.compute_precision"],
      reductionPrecision: creatorDefined?[
        "com.mlxdlss.reduction_precision"
      ],
      ioType: creatorDefined?["com.mlxdlss.io_type"]
    )
  }

  func predict(_ input: MLMultiArray) async throws -> MLMultiArray {
    let provider = try MLDictionaryFeatureProvider(dictionary: [
      inputSpecification.name: MLFeatureValue(multiArray: input)
    ])
    let result = try await model.prediction(from: provider)
    guard let output = result.featureValue(for: outputSpecification.name)?.multiArrayValue else {
      throw CoreMLBackendError.predictionMissingOutput(outputSpecification.name)
    }
    return output
  }

  @available(macOS 15.0, *)
  func predict(_ input: MLTensor) async throws -> MLTensor {
    let result = try await model.prediction(from: [inputSpecification.name: input])
    guard let output = result[outputSpecification.name] else {
      throw CoreMLBackendError.predictionMissingOutput(outputSpecification.name)
    }
    return output
  }

  private static func specification(
    named name: String,
    in descriptions: [String: MLFeatureDescription],
    missing: CoreMLBackendError
  ) throws -> CoreMLTensorSpecification {
    guard let description = descriptions[name] else {
      throw missing
    }
    guard description.type == .multiArray,
      let constraint = description.multiArrayConstraint
    else {
      throw CoreMLBackendError.modelFeatureIsNotMultiArray(name)
    }
    guard constraint.dataType == .float32 else {
      throw CoreMLBackendError.modelFeatureDataType(
        name: name,
        expected: "float32",
        actual: String(describing: constraint.dataType)
      )
    }
    let shape = constraint.shape.map(\.intValue)
    return CoreMLTensorSpecification(name: name, shape: shape, dataType: .float32)
  }
}

public actor CoreMLNeuralRenderer: NeuralRenderBackend {
  private let predictor: any CoreMLPredicting
  private let configuration: CoreMLBackendConfiguration
  public nonisolated let height: Int
  public nonisolated let width: Int
  public nonisolated let inputChannels: Int
  public nonisolated let outputChannels: Int
  public nonisolated let modelMetadata: CoreMLModelMetadata
  public nonisolated let effectiveHostBridge: CoreMLHostBridge

  public init(
    modelURL: URL,
    configuration: CoreMLBackendConfiguration = CoreMLBackendConfiguration()
  ) async throws {
    let predictor = try await CoreMLModelPredictor(
      modelURL: modelURL,
      configuration: configuration
    )
    try self.init(predictor: predictor, configuration: configuration)
  }

  init(
    predictor: any CoreMLPredicting,
    configuration: CoreMLBackendConfiguration = CoreMLBackendConfiguration()
  ) throws {
    let inputShape = predictor.inputSpecification.shape
    let outputShape = predictor.outputSpecification.shape
    guard inputShape.count == 4,
      inputShape[0] == 1,
      inputShape[1] > 0,
      inputShape[2] > 0,
      inputShape[3] > 0
    else {
      throw CoreMLBackendError.modelFeatureShape(
        name: predictor.inputSpecification.name,
        shape: inputShape
      )
    }
    guard outputShape.count == 4,
      outputShape[0] == 1,
      outputShape[1] > 0,
      outputShape[2] > 0,
      outputShape[3] > 0
    else {
      throw CoreMLBackendError.modelFeatureShape(
        name: predictor.outputSpecification.name,
        shape: outputShape
      )
    }
    guard inputShape[0] == outputShape[0],
      inputShape[2] == outputShape[2],
      inputShape[3] == outputShape[3]
    else {
      throw CoreMLBackendError.modelInputOutputShapeMismatch(
        input: inputShape,
        output: outputShape
      )
    }
    guard predictor.inputSpecification.dataType == .float32 else {
      throw CoreMLBackendError.modelFeatureDataType(
        name: predictor.inputSpecification.name,
        expected: "float32",
        actual: String(describing: predictor.inputSpecification.dataType)
      )
    }
    guard predictor.outputSpecification.dataType == .float32 else {
      throw CoreMLBackendError.modelFeatureDataType(
        name: predictor.outputSpecification.name,
        expected: "float32",
        actual: String(describing: predictor.outputSpecification.dataType)
      )
    }
    let effectiveHostBridge: CoreMLHostBridge
    switch configuration.hostBridge {
    case .automatic:
      effectiveHostBridge = .multiArray
    case .multiArray:
      effectiveHostBridge = .multiArray
    case .mlTensor:
      if #available(macOS 15.0, *), predictor is any CoreMLTensorPredicting {
        effectiveHostBridge = .mlTensor
      } else {
        throw CoreMLBackendError.tensorBridgeUnavailable
      }
    }

    self.predictor = predictor
    self.configuration = configuration
    self.height = inputShape[2]
    self.width = inputShape[3]
    self.inputChannels = inputShape[1]
    self.outputChannels = outputShape[1]
    self.modelMetadata = predictor.metadata
    self.effectiveHostBridge = effectiveHostBridge
  }

  public func render(_ request: NeuralRenderRequest) async throws -> NeuralRenderResult {
    guard request.inputs.count == 1 else {
      throw CoreMLBackendError.inputCount(expected: 1, actual: request.inputs.count)
    }
    guard let input = request.input(named: configuration.hostTensorName) else {
      throw CoreMLBackendError.missingInput(configuration.hostTensorName)
    }
    guard input.descriptor.dataType == .float32 else {
      throw CoreMLBackendError.inputDataType(
        expected: .float32,
        actual: input.descriptor.dataType
      )
    }
    guard input.descriptor.layout == .nhwc else {
      throw CoreMLBackendError.inputLayout(
        expected: .nhwc,
        actual: input.descriptor.layout
      )
    }
    let expectedShape = [1, height, width, inputChannels]
    guard input.descriptor.shape == expectedShape else {
      throw CoreMLBackendError.inputShape(
        expected: expectedShape,
        actual: input.descriptor.shape
      )
    }

    let output: HostTensor
    let start: UInt64
    if shouldUseMLTensorBridge {
      if #available(macOS 15.0, *),
        let tensorPredictor = predictor as? any CoreMLTensorPredicting
      {
        let modelInput = MLTensor(
          shape: expectedShape,
          data: input.bytes,
          scalarType: Float.self
        ).transposed(permutation: [0, 3, 1, 2])
        start = DispatchTime.now().uptimeNanoseconds
        let modelOutput = try await tensorPredictor.predict(modelInput)
        output = try await makeNHWCOutput(from: modelOutput)
      } else {
        throw CoreMLBackendError.tensorBridgeUnavailable
      }
    } else {
      let modelInput = try makeNCHWInput(from: input)
      start = DispatchTime.now().uptimeNanoseconds
      let modelOutput = try await predictor.predict(modelInput)
      output = try makeNHWCOutput(from: modelOutput)
    }
    let end = DispatchTime.now().uptimeNanoseconds
    return try NeuralRenderResult(
      outputs: [output],
      timing: NeuralRenderTiming(executionNanoseconds: end - start)
    )
  }

  public func reset(sequenceID _: UInt64?) async {}

  private var shouldUseMLTensorBridge: Bool {
    effectiveHostBridge == .mlTensor
  }

  private func makeNCHWInput(from tensor: HostTensor) throws -> MLMultiArray {
    let result = try MLMultiArray(
      shape: [
        1, NSNumber(value: inputChannels), NSNumber(value: height),
        NSNumber(value: width),
      ],
      dataType: .float32
    )
    let destination = result.dataPointer.assumingMemoryBound(to: Float.self)
    let strides = result.strides.map(\.intValue)
    tensor.bytes.withUnsafeBytes { source in
      for y in 0..<height {
        for x in 0..<width {
          let sourceBase = (y * width + x) * inputChannels
          for channel in 0..<inputChannels {
            let destinationIndex =
              channel * strides[1]
              + y * strides[2]
              + x * strides[3]
            destination[destinationIndex] = source.loadUnaligned(
              fromByteOffset: (sourceBase + channel) * MemoryLayout<Float>.size,
              as: Float.self
            )
          }
        }
      }
    }
    return result
  }

  private func makeNHWCOutput(from array: MLMultiArray) throws -> HostTensor {
    let actualShape = array.shape.map(\.intValue)
    let expectedShape = [1, outputChannels, height, width]
    guard actualShape == expectedShape else {
      throw CoreMLBackendError.predictionOutputShape(
        name: predictor.outputSpecification.name,
        expected: expectedShape,
        actual: actualShape
      )
    }
    guard array.dataType == .float32 else {
      throw CoreMLBackendError.predictionOutputDataType(
        name: predictor.outputSpecification.name,
        expected: "float32",
        actual: String(describing: array.dataType)
      )
    }

    let source = array.dataPointer.assumingMemoryBound(to: Float.self)
    let strides = array.strides.map(\.intValue)
    var values = [Float](repeating: 0, count: height * width * outputChannels)
    var allFinite = true
    for y in 0..<height {
      for x in 0..<width {
        let destinationBase = (y * width + x) * outputChannels
        for channel in 0..<outputChannels {
          let sourceIndex =
            channel * strides[1]
            + y * strides[2]
            + x * strides[3]
          let value = source[sourceIndex]
          values[destinationBase + channel] = value
          allFinite = allFinite && value.isFinite
        }
      }
    }
    guard allFinite else {
      throw CoreMLBackendError.nonFiniteOutput(predictor.outputSpecification.name)
    }

    return try makeOutputTensor(values: values)
  }

  @available(macOS 15.0, *)
  private func makeNHWCOutput(from tensor: MLTensor) async throws -> HostTensor {
    let expectedShape = [1, outputChannels, height, width]
    guard tensor.shape == expectedShape else {
      throw CoreMLBackendError.predictionOutputShape(
        name: predictor.outputSpecification.name,
        expected: expectedShape,
        actual: tensor.shape
      )
    }
    let channelsLast = tensor.transposed(permutation: [0, 2, 3, 1])
    let shaped = await channelsLast.shapedArray(of: Float.self)
    let values = Array(shaped.scalars)
    guard values.allSatisfy(\.isFinite) else {
      throw CoreMLBackendError.nonFiniteOutput(predictor.outputSpecification.name)
    }
    return try makeOutputTensor(values: values)
  }

  private func makeOutputTensor(values: [Float]) throws -> HostTensor {
    let descriptor = try TensorDescriptor(
      name: configuration.hostTensorName,
      shape: [1, height, width, outputChannels],
      dataType: .float32,
      layout: .nhwc
    )
    return try HostTensor(
      descriptor: descriptor,
      bytes: values.withUnsafeBytes { Data($0) }
    )
  }
}
