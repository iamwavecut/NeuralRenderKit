import Foundation
import XCTest

@testable import DLSSCore

final class NeuralRenderingFirstFrameBackendTests: XCTestCase {
  func testBackendIsFrameIndependent() {
    let backend = NeuralRenderingFirstFrameBackend(head: ConstantHeadBackend())
    XCTAssertEqual(backend.temporalCadence, .frameIndependent)
  }

  func testZeroHeadReturnsInputCroppedToLogicalSizeThroughVendorGeometry() async throws {
    let head = ConstantHeadBackend()
    let backend = NeuralRenderingFirstFrameBackend(head: head)
    let color = try rgbTensor(width: 100, height: 60) { x, y in
      [Float(x) / 100, Float(y) / 60, 0.5]
    }

    let result = try await backend.render(
      NeuralRenderRequest(sequenceID: 3, inputs: [color])
    )

    let output = try XCTUnwrap(result.output(named: "color"))
    XCTAssertEqual(output.descriptor.shape, [1, 60, 100, 3])
    XCTAssertEqual(output.bytes, color.bytes)
    let headInputs = await head.inputs
    XCTAssertEqual(headInputs.map(\.descriptor.shape), [[1, 320, 320, 16]])
  }

  func testMatchOutputGeometryKeepsNetworkExtent() async throws {
    let head = ConstantHeadBackend()
    var configuration = NeuralRenderingFirstFrameConfiguration(profile: .standard)
    configuration.geometry = .matchOutput
    let backend = NeuralRenderingFirstFrameBackend(head: head, configuration: configuration)
    let color = try rgbTensor(width: 128, height: 128) { _, _ in [0.25, 0.5, 0.75] }

    _ = try await backend.render(NeuralRenderRequest(sequenceID: 1, inputs: [color]))

    let headInputs = await head.inputs
    XCTAssertEqual(headInputs.map(\.descriptor.shape), [[1, 128, 128, 16]])
  }

  func testProfileControlsReachHeadFeatures() async throws {
    let head = ConstantHeadBackend()
    let backend = NeuralRenderingFirstFrameBackend(head: head, profile: .natural)
    let color = try rgbTensor(width: 2, height: 1) { _, _ in [0.25, 0.5, 0.75] }

    _ = try await backend.render(NeuralRenderRequest(sequenceID: 1, inputs: [color]))

    let headInputs = await head.inputs
    let features = try XCTUnwrap(headInputs.first)
    let values = floats(in: features)
    XCTAssertEqual(Array(values[3..<7]), [1, -0.03125, 0, 0.03125])
    XCTAssertEqual(Array(values[10..<13]), [Float(Float16(1.0 / 128)), 1, 1])
  }

  func testNeutralProfileDisablesToneAndStructure() async throws {
    let head = ConstantHeadBackend()
    let backend = NeuralRenderingFirstFrameBackend(head: head, profile: .neutral)
    let color = try rgbTensor(width: 2, height: 1) { _, _ in [0.25, 0.5, 0.75] }

    _ = try await backend.render(NeuralRenderRequest(sequenceID: 1, inputs: [color]))

    let headInputs = await head.inputs
    let features = try XCTUnwrap(headInputs.first)
    XCTAssertEqual(Array(floats(in: features)[10..<13]), [0, 0, 0])
  }

  func testConstantHeadResidualIsScaledByIntensity() async throws {
    let head = ConstantHeadBackend(value: 1)
    var configuration = NeuralRenderingFirstFrameConfiguration(profile: .standard)
    configuration.intensity = 0.5
    let backend = NeuralRenderingFirstFrameBackend(head: head, configuration: configuration)
    let color = try rgbTensor(width: 1, height: 1) { _, _ in [0.2, 0.5, 0.9] }

    let result = try await backend.render(
      NeuralRenderRequest(sequenceID: 1, inputs: [color])
    )

    let output = floats(in: try XCTUnwrap(result.output(named: "color")))
    XCTAssertEqual(output[0], 0.325, accuracy: 0.000_001)
    XCTAssertEqual(output[1], 0.625, accuracy: 0.000_001)
    XCTAssertEqual(output[2], 0.95, accuracy: 0.000_001)
  }

  func testControlMaskRedChannelGatesTheEffect() async throws {
    let head = ConstantHeadBackend(value: 1)
    let backend = NeuralRenderingFirstFrameBackend(head: head)
    let color = try rgbTensor(width: 1, height: 1) { _, _ in [0.2, 0.5, 0.6] }
    let mask = try rgbTensor(name: "controlMask", width: 1, height: 1) { _, _ in
      [0, 1, 1]
    }

    let result = try await backend.render(
      NeuralRenderRequest(sequenceID: 1, inputs: [color, mask])
    )

    let output = floats(in: try XCTUnwrap(result.output(named: "color")))
    XCTAssertEqual(output, [0.2, 0.5, 0.6])
  }

  func testMissingColorInputIsRejected() async throws {
    let backend = NeuralRenderingFirstFrameBackend(head: ConstantHeadBackend())
    let mask = try rgbTensor(name: "controlMask", width: 1, height: 1) { _, _ in
      [1, 1, 1]
    }

    do {
      _ = try await backend.render(NeuralRenderRequest(sequenceID: 1, inputs: [mask]))
      XCTFail("expected a missing color input error")
    } catch let error as NeuralRenderingFirstFrameBackendError {
      XCTAssertEqual(error, .missingInput("color"))
    }
  }

  func testResetForwardsToHead() async {
    let head = ConstantHeadBackend()
    let backend = NeuralRenderingFirstFrameBackend(head: head)

    await backend.reset(sequenceID: 7)

    let resets = await head.resets
    XCTAssertEqual(resets, [NeuralRenderResetRequest(streamID: 7, reason: .explicit)])
  }

  private func rgbTensor(
    name: String = "color",
    width: Int,
    height: Int,
    pixel: (Int, Int) -> [Float]
  ) throws -> HostTensor {
    var values: [Float] = []
    values.reserveCapacity(width * height * 3)
    for y in 0..<height {
      for x in 0..<width {
        values += pixel(x, y)
      }
    }
    return try HostTensor(
      descriptor: TensorDescriptor(
        name: name,
        shape: [1, height, width, 3],
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: values.withUnsafeBytes { Data($0) }
    )
  }

  private func floats(in tensor: HostTensor) -> [Float] {
    tensor.bytes.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
  }
}

private actor ConstantHeadBackend: NeuralRenderBackend {
  private let value: Float
  private(set) var inputs: [HostTensor] = []
  private(set) var resets: [NeuralRenderResetRequest] = []

  init(value: Float = 0) {
    self.value = value
  }

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
    let values = [Float](repeating: value, count: height * width * 4)
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

  func reset(sequenceID: UInt64?) async {
    resets.append(NeuralRenderResetRequest(streamID: sequenceID, reason: .explicit))
  }

  func reset(_ request: NeuralRenderResetRequest) async {
    resets.append(request)
  }
}
