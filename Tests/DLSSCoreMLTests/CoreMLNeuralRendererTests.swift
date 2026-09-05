import CoreML
import Foundation
import DLSSCore
import XCTest

@testable import DLSSCoreML

final class CoreMLNeuralRendererTests: XCTestCase {
  func testExternalNeuralRenderingCoreMLPackageRunsSixteenToFour() async throws {
    guard
      let path = ProcessInfo.processInfo.environment[
        "MLXDLSS_NEURAL_RENDERING_COREML_PACKAGE"
      ]
    else {
      throw XCTSkip(
        "set MLXDLSS_NEURAL_RENDERING_COREML_PACKAGE to run the Core ML probe"
      )
    }
    let renderer = try await CoreMLNeuralRenderer(
      modelURL: URL(fileURLWithPath: path),
      configuration: CoreMLBackendConfiguration(computeUnits: .cpuAndGPU)
    )
    let values = (0..<(128 * 128 * 16)).map {
      sin(Float($0) * 0.001)
    }
    let request = try makeRequest(
      values: values,
      height: 128,
      width: 128,
      channels: 16
    )

    let result = try await renderer.render(request)
    let output = try XCTUnwrap(result.output(named: "color"))

    XCTAssertEqual(renderer.inputChannels, 16)
    XCTAssertEqual(renderer.outputChannels, 4)
    XCTAssertEqual(output.descriptor.shape, [1, 128, 128, 4])
    XCTAssertTrue(floatValues(output).allSatisfy(\.isFinite))
    XCTAssertGreaterThan(result.timing.executionNanoseconds, 0)
  }

  func testRendererSupportsDifferentInputAndOutputChannels() async throws {
    let predictor = RecordingPredictor(
      inputShape: [1, 16, 1, 1],
      outputShape: [1, 4, 1, 1]
    ) { input in
      XCTAssertEqual(input.values(), Array(0..<16).map(Float.init))
      let output = try MLMultiArray(
        shape: [1, 4, 1, 1],
        dataType: .float32
      )
      for index in 0..<4 {
        output[index] = NSNumber(value: Float(index) + 0.5)
      }
      return output
    }
    let renderer = try CoreMLNeuralRenderer(predictor: predictor)
    let request = try makeRequest(
      values: Array(0..<16).map(Float.init),
      height: 1,
      width: 1,
      channels: 16
    )

    let result = try await renderer.render(request)
    let output = try XCTUnwrap(result.output(named: "color"))

    XCTAssertEqual(renderer.inputChannels, 16)
    XCTAssertEqual(renderer.outputChannels, 4)
    XCTAssertEqual(output.descriptor.shape, [1, 1, 1, 4])
    XCTAssertEqual(floatValues(output), [0.5, 1.5, 2.5, 3.5])
  }

  func testRendererConvertsNHWCToNCHWAndBack() async throws {
    let predictor = RecordingPredictor(shape: [1, 3, 2, 2]) { input in
      XCTAssertEqual(input.values(), [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11])
      return try input.mapValues { $0 + 0.5 }
    }
    let renderer = try CoreMLNeuralRenderer(predictor: predictor)
    let request = try makeRequest(
      values: Array(0..<12).map(Float.init),
      height: 2,
      width: 2
    )

    let result = try await renderer.render(request)

    XCTAssertEqual(
      result.output(named: "color").map(floatValues),
      [
        0.5, 1.5, 2.5,
        3.5, 4.5, 5.5,
        6.5, 7.5, 8.5,
        9.5, 10.5, 11.5,
      ])
    XCTAssertGreaterThan(result.timing.executionNanoseconds, 0)
    XCTAssertEqual(predictor.predictionCount, 1)
  }

  func testRendererRejectsMismatchedStaticShapeBeforePrediction() async throws {
    let predictor = RecordingPredictor(shape: [1, 3, 2, 2]) { $0 }
    let renderer = try CoreMLNeuralRenderer(predictor: predictor)
    let request = try makeRequest(values: Array(repeating: 0, count: 18), height: 2, width: 3)

    await XCTAssertThrowsErrorAsync(try await renderer.render(request)) {
      XCTAssertEqual(
        $0 as? CoreMLBackendError,
        .inputShape(expected: [1, 2, 2, 3], actual: [1, 2, 3, 3])
      )
    }
    XCTAssertEqual(predictor.predictionCount, 0)
  }

  func testRendererRejectsNonFiniteModelOutput() async throws {
    let predictor = RecordingPredictor(shape: [1, 3, 1, 1]) { input in
      try input.mapValues { _ in .nan }
    }
    let renderer = try CoreMLNeuralRenderer(predictor: predictor)
    let request = try makeRequest(values: [0, 0, 0], height: 1, width: 1)

    await XCTAssertThrowsErrorAsync(try await renderer.render(request)) {
      XCTAssertEqual($0 as? CoreMLBackendError, .nonFiniteOutput("restored"))
    }
  }

  func testConfigurationDefaultsToCPUGPU() {
    XCTAssertEqual(CoreMLBackendConfiguration().computeUnits, .cpuAndGPU)
    XCTAssertEqual(CoreMLBackendConfiguration().hostBridge, .automatic)
  }

  @available(macOS 15.0, *)
  func testExplicitMLTensorBridgeUsesTensorPredictor() async throws {
    let predictor = TensorRecordingPredictor(shape: [1, 3, 1, 2])
    let renderer = try CoreMLNeuralRenderer(
      predictor: predictor,
      configuration: CoreMLBackendConfiguration(hostBridge: .mlTensor)
    )
    let request = try makeRequest(
      values: [0, 1, 2, 3, 4, 5],
      height: 1,
      width: 2
    )

    let result = try await renderer.render(request)

    XCTAssertEqual(
      result.output(named: "color").map(floatValues),
      [
        0.25, 1.25, 2.25, 3.25, 4.25, 5.25,
      ])
    XCTAssertEqual(predictor.tensorPredictionCount, 1)
    XCTAssertEqual(predictor.multiArrayPredictionCount, 0)
    XCTAssertEqual(renderer.effectiveHostBridge, .mlTensor)
  }

  @available(macOS 15.0, *)
  func testAutomaticBridgeKeepsMaterializedHostPathOnMultiArray() throws {
    let predictor = TensorRecordingPredictor(shape: [1, 3, 1, 2])
    let renderer = try CoreMLNeuralRenderer(predictor: predictor)

    XCTAssertEqual(renderer.effectiveHostBridge, .multiArray)
  }

  private func makeRequest(
    values: [Float],
    height: Int,
    width: Int,
    channels: Int = 3
  ) throws -> NeuralRenderRequest {
    let descriptor = try TensorDescriptor(
      name: "color",
      shape: [1, height, width, channels],
      dataType: .float32,
      layout: .nhwc
    )
    let tensor = try HostTensor(
      descriptor: descriptor,
      bytes: values.withUnsafeBytes { Data($0) }
    )
    return try NeuralRenderRequest(sequenceID: 1, inputs: [tensor])
  }

  private func floatValues(_ tensor: HostTensor) -> [Float] {
    tensor.bytes.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: 4).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
  }
}

private final class RecordingPredictor: CoreMLPredicting, @unchecked Sendable {
  let inputSpecification: CoreMLTensorSpecification
  let outputSpecification: CoreMLTensorSpecification
  let metadata = CoreMLModelMetadata()
  private let operation: (MLMultiArray) throws -> MLMultiArray
  private(set) var predictionCount = 0

  init(
    shape: [Int],
    operation: @escaping (MLMultiArray) throws -> MLMultiArray
  ) {
    self.inputSpecification = CoreMLTensorSpecification(
      name: "color",
      shape: shape,
      dataType: .float32
    )
    self.outputSpecification = CoreMLTensorSpecification(
      name: "restored",
      shape: shape,
      dataType: .float32
    )
    self.operation = operation
  }

  init(
    inputShape: [Int],
    outputShape: [Int],
    operation: @escaping (MLMultiArray) throws -> MLMultiArray
  ) {
    self.inputSpecification = CoreMLTensorSpecification(
      name: "color",
      shape: inputShape,
      dataType: .float32
    )
    self.outputSpecification = CoreMLTensorSpecification(
      name: "restored",
      shape: outputShape,
      dataType: .float32
    )
    self.operation = operation
  }

  func predict(_ input: MLMultiArray) async throws -> MLMultiArray {
    predictionCount += 1
    return try operation(input)
  }
}

@available(macOS 15.0, *)
private final class TensorRecordingPredictor: CoreMLPredicting, CoreMLTensorPredicting,
  @unchecked Sendable
{
  let inputSpecification: CoreMLTensorSpecification
  let outputSpecification: CoreMLTensorSpecification
  let metadata = CoreMLModelMetadata()
  private(set) var tensorPredictionCount = 0
  private(set) var multiArrayPredictionCount = 0

  init(shape: [Int]) {
    inputSpecification = CoreMLTensorSpecification(
      name: "color", shape: shape, dataType: .float32
    )
    outputSpecification = CoreMLTensorSpecification(
      name: "restored", shape: shape, dataType: .float32
    )
  }

  func predict(_ input: MLMultiArray) async throws -> MLMultiArray {
    multiArrayPredictionCount += 1
    return input
  }

  func predict(_ input: MLTensor) async throws -> MLTensor {
    tensorPredictionCount += 1
    XCTAssertEqual(input.shape, inputSpecification.shape)
    let values = await input.shapedArray(of: Float.self)
    XCTAssertEqual(
      Array(values.scalars),
      [0, 3, 1, 4, 2, 5]
    )
    return input + Float(0.25)
  }
}

extension MLMultiArray {
  fileprivate func values() -> [Float] {
    (0..<count).map { self[$0].floatValue }
  }

  fileprivate func mapValues(_ transform: (Float) -> Float) throws -> MLMultiArray {
    let result = try MLMultiArray(shape: shape, dataType: .float32)
    for index in 0..<count {
      result[index] = NSNumber(value: transform(self[index].floatValue))
    }
    return result
  }
}

private func XCTAssertThrowsErrorAsync<T>(
  _ expression: @autoclosure () async throws -> T,
  _ errorHandler: (any Error) -> Void = { _ in }
) async {
  do {
    _ = try await expression()
    XCTFail("expected error")
  } catch {
    errorHandler(error)
  }
}
