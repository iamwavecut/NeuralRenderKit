import Foundation
import XCTest
@testable import NeuralRenderCore

final class RenderContractsTests: XCTestCase {
    func testRequestRejectsDuplicateTensorNames() throws {
        let first = try makeTensor(name: "color", value: 1)
        let second = try makeTensor(name: "color", value: 2)

        XCTAssertThrowsError(try NeuralRenderRequest(sequenceID: 7, inputs: [first, second])) {
            XCTAssertEqual($0 as? RenderContractError, .duplicateTensor("color"))
        }
    }

    func testRequestRejectsEmptyInputs() {
        XCTAssertThrowsError(try NeuralRenderRequest(sequenceID: 7, inputs: [])) {
            XCTAssertEqual($0 as? RenderContractError, .emptyInputs)
        }
    }

    func testResultRejectsEmptyOutputs() {
        XCTAssertThrowsError(
            try NeuralRenderResult(outputs: [], timing: .init(executionNanoseconds: 10))
        ) {
            XCTAssertEqual($0 as? RenderContractError, .emptyOutputs)
        }
    }

    func testResultRejectsDuplicateTensorNames() throws {
        let first = try makeTensor(name: "color", value: 1)
        let second = try makeTensor(name: "color", value: 2)

        XCTAssertThrowsError(
            try NeuralRenderResult(
                outputs: [first, second],
                timing: .init(executionNanoseconds: 10)
            )
        ) {
            XCTAssertEqual($0 as? RenderContractError, .duplicateTensor("color"))
        }
    }

    func testNamedTensorsAreExposedInDeterministicOrder() throws {
        let zeta = try makeTensor(name: "zeta", value: 2)
        let alpha = try makeTensor(name: "alpha", value: 1)

        let request = try NeuralRenderRequest(sequenceID: 8, inputs: [zeta, alpha])
        let result = try NeuralRenderResult(
            outputs: [zeta, alpha],
            timing: .init(executionNanoseconds: 10)
        )

        XCTAssertEqual(request.inputs.map(\.descriptor.name), ["alpha", "zeta"])
        XCTAssertEqual(result.outputs.map(\.descriptor.name), ["alpha", "zeta"])
        XCTAssertEqual(request.input(named: "alpha"), alpha)
        XCTAssertEqual(result.output(named: "zeta"), zeta)
    }

    func testActorBackendWorksThroughPublicExistential() async throws {
        let input = try makeTensor(name: "color", value: 42)
        let request = try NeuralRenderRequest(sequenceID: 9, inputs: [input])
        let backend: any NeuralRenderBackend = EchoBackend()

        let result = try await backend.render(request)

        XCTAssertEqual(result.output(named: "color"), input)
        XCTAssertEqual(result.timing.executionNanoseconds, 123)
    }

    private func makeTensor(name: String, value: UInt8) throws -> HostTensor {
        let descriptor = try TensorDescriptor(
            name: name,
            shape: [1],
            dataType: .float32,
            layout: .vector
        )
        return try HostTensor(
            descriptor: descriptor,
            bytes: Data([value, 0, 0, 0])
        )
    }
}

private actor EchoBackend: NeuralRenderBackend {
    func render(_ request: NeuralRenderRequest) async throws -> NeuralRenderResult {
        try NeuralRenderResult(
            outputs: request.inputs,
            timing: .init(executionNanoseconds: 123)
        )
    }

    func reset(sequenceID: UInt64?) async {}
}
