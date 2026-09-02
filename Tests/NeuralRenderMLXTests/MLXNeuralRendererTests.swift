import CryptoKit
import Foundation
import MLX
import XCTest
@testable import NeuralRenderCore
@testable import NeuralRenderMLX

final class MLXNeuralRendererTests: XCTestCase {
    func testExternalPackageProducesHandDerivedGoldenOutput() async throws {
        let packageURL = try makePackage(
            scale: [2, 0.5, -1],
            bias: [0.1, 0.2, 1.0]
        )
        defer { try? FileManager.default.removeItem(at: packageURL) }
        let renderer = try MLXNeuralRenderer(packageURL: packageURL)
        let input = try makeHostTensor(
            name: "color",
            values: [0.2, 0.4, 0.25, 0.6, 2.0, 2.0],
            shape: [1, 1, 2, 3]
        )
        let request = try NeuralRenderRequest(sequenceID: 1, inputs: [input])

        let result = try await renderer.render(request)

        let output = try XCTUnwrap(result.output(named: "color"))
        XCTAssertEqual(output.descriptor.shape, [1, 1, 2, 3])
        XCTAssertEqual(output.descriptor.dataType, .float32)
        XCTAssertEqual(output.descriptor.layout, .nhwc)
        let expected: [Float] = [0.5, 0.4, 0.75, 1.0, 1.0, 0.0]
        for (actual, expected) in zip(floatValues(in: output), expected) {
            XCTAssertEqual(actual, expected, accuracy: 0.000_001)
        }
        XCTAssertGreaterThan(result.timing.executionNanoseconds, 0)
    }

    func testRejectsUnknownArchitectureBeforeRendering() throws {
        let packageURL = try makePackage(
            architecture: "unknown.v1",
            scale: [1, 1, 1],
            bias: [0, 0, 0]
        )
        defer { try? FileManager.default.removeItem(at: packageURL) }

        XCTAssertThrowsError(try MLXNeuralRenderer(packageURL: packageURL)) {
            XCTAssertEqual($0 as? MLXBackendError, .unsupportedArchitecture("unknown.v1"))
        }
    }

    func testStatelessArchitectureRejectsManifestClaimingRecurrentState() throws {
        let stateJSON = #"{"kind":"recurrent","cadence":"consecutiveFrames","tensors":[{"name":"history","dataType":"float16","layout":"vector","shape":[8]}]}"#
        let packageURL = try makePackage(
            stateJSON: stateJSON,
            scale: [1, 1, 1],
            bias: [0, 0, 0]
        )
        defer { try? FileManager.default.removeItem(at: packageURL) }

        XCTAssertThrowsError(try MLXNeuralRenderer(packageURL: packageURL)) {
            guard let backendError = $0 as? MLXBackendError else {
                return XCTFail("unexpected error: \(String(reflecting: $0))")
            }
            XCTAssertEqual(
                backendError,
                .modelStateKindMismatch(
                    architecture: "nrk.pixel-affine.v1",
                    expected: .stateless,
                    actual: .recurrent
                )
            )
        }
    }

    func testCompiledExecutionModePreservesGoldenOutput() async throws {
        let packageURL = try makePackage(
            scale: [2, 0.5, -1],
            bias: [0.1, 0.2, 1.0]
        )
        defer { try? FileManager.default.removeItem(at: packageURL) }
        let renderer = try MLXNeuralRenderer(
            packageURL: packageURL,
            executionMode: .compiled
        )
        let input = try makeHostTensor(
            name: "color",
            values: [0.2, 0.4, 0.25, 0.6, 2.0, 2.0],
            shape: [1, 1, 2, 3]
        )
        let request = try NeuralRenderRequest(sequenceID: 5, inputs: [input])

        let result = try await renderer.render(request)

        let output = try XCTUnwrap(result.output(named: "color"))
        let expected: [Float] = [0.5, 0.4, 0.75, 1.0, 1.0, 0.0]
        for (actual, expected) in zip(floatValues(in: output), expected) {
            XCTAssertEqual(actual, expected, accuracy: 0.000_001)
        }
    }

    func testFloat16ComputeModeReturnsFloat32HostTensorWithinDeclaredTolerance() async throws {
        let packageURL = try makePackage(
            scale: [2, 0.5, -1],
            bias: [0.1, 0.2, 1.0]
        )
        defer { try? FileManager.default.removeItem(at: packageURL) }
        let renderer = try MLXNeuralRenderer(
            packageURL: packageURL,
            computePrecision: .float16
        )
        let input = try makeHostTensor(
            name: "color",
            values: [0.2, 0.4, 0.25, 0.6, 2.0, 2.0],
            shape: [1, 1, 2, 3]
        )
        let request = try NeuralRenderRequest(sequenceID: 6, inputs: [input])

        let result = try await renderer.render(request)

        let output = try XCTUnwrap(result.output(named: "color"))
        XCTAssertEqual(output.descriptor.dataType, .float32)
        let expected: [Float] = [0.5, 0.4, 0.75, 1.0, 1.0, 0.0]
        for (actual, expected) in zip(floatValues(in: output), expected) {
            XCTAssertEqual(actual, expected, accuracy: 0.001)
        }
    }

    func testRejectsRequestMissingDeclaredColorInput() async throws {
        let packageURL = try makePackage(scale: [1, 1, 1], bias: [0, 0, 0])
        defer { try? FileManager.default.removeItem(at: packageURL) }
        let renderer = try MLXNeuralRenderer(packageURL: packageURL)
        let depth = try makeHostTensor(
            name: "depth",
            values: [1],
            shape: [1, 1, 1, 1]
        )
        let request = try NeuralRenderRequest(sequenceID: 2, inputs: [depth])

        do {
            _ = try await renderer.render(request)
            XCTFail("Expected a missing input error")
        } catch {
            XCTAssertEqual(error as? ManifestError, .missingInput("color"))
        }
    }

    func testRejectsUnsupportedFloat16DemoInput() async throws {
        let packageURL = try makePackage(
            inputDataType: "float16",
            scale: [1, 1, 1],
            bias: [0, 0, 0]
        )
        defer { try? FileManager.default.removeItem(at: packageURL) }
        let renderer = try MLXNeuralRenderer(packageURL: packageURL)
        let descriptor = try TensorDescriptor(
            name: "color",
            shape: [1, 1, 1, 3],
            dataType: .float16,
            layout: .nhwc
        )
        let values: [Float16] = [0, 0, 0]
        let input = try HostTensor(
            descriptor: descriptor,
            bytes: values.withUnsafeBytes { Data($0) }
        )
        let request = try NeuralRenderRequest(sequenceID: 3, inputs: [input])

        do {
            _ = try await renderer.render(request)
            XCTFail("Expected an unsupported data type error")
        } catch {
            XCTAssertEqual(error as? MLXBackendError, .unsupportedTensorDataType(.float16))
        }
    }

    func testRejectsDemoPackageWithoutColorOutput() async throws {
        let packageURL = try makePackage(
            outputName: "restored",
            scale: [1, 1, 1],
            bias: [0, 0, 0]
        )
        defer { try? FileManager.default.removeItem(at: packageURL) }
        let renderer = try MLXNeuralRenderer(packageURL: packageURL)
        let input = try makeHostTensor(
            name: "color",
            values: [0, 0, 0],
            shape: [1, 1, 1, 3]
        )
        let request = try NeuralRenderRequest(sequenceID: 4, inputs: [input])

        do {
            _ = try await renderer.render(request)
            XCTFail("Expected a missing output error")
        } catch {
            XCTAssertEqual(error as? MLXBackendError, .missingOutput("color"))
        }
    }

    private func makePackage(
        architecture: String = "nrk.pixel-affine.v1",
        inputDataType: String = "float32",
        outputName: String = "color",
        stateJSON: String = #"{"kind":"stateless"}"#,
        scale: [Float],
        bias: [Float]
    ) throws -> URL {
        let packageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("nrkmodel")
        try FileManager.default.createDirectory(at: packageURL, withIntermediateDirectories: false)

        let weightsURL = packageURL.appendingPathComponent("weights.safetensors")
        try save(
            arrays: ["scale": MLXArray(scale), "bias": MLXArray(bias)],
            url: weightsURL,
            stream: .cpu
        )
        let digest = SHA256.hash(data: try Data(contentsOf: weightsURL))
            .map { String(format: "%02x", $0) }
            .joined()
        try Data(
            manifestJSON(
                architecture: architecture,
                inputDataType: inputDataType,
                outputName: outputName,
                stateJSON: stateJSON,
                digest: digest
            ).utf8
        )
            .write(to: packageURL.appendingPathComponent("manifest.json"))
        return packageURL
    }

    private func makeHostTensor(
        name: String,
        values: [Float],
        shape: [Int]
    ) throws -> HostTensor {
        let descriptor = try TensorDescriptor(
            name: name,
            shape: shape,
            dataType: .float32,
            layout: .nhwc
        )
        let bytes = values.withUnsafeBytes { Data($0) }
        return try HostTensor(descriptor: descriptor, bytes: bytes)
    }

    private func floatValues(in tensor: HostTensor) -> [Float] {
        tensor.bytes.withUnsafeBytes { bytes in
            stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map { offset in
                bytes.loadUnaligned(fromByteOffset: offset, as: Float.self)
            }
        }
    }

    private func manifestJSON(
        architecture: String,
        inputDataType: String,
        outputName: String,
        stateJSON: String,
        digest: String
    ) -> String {
        return #"""
        {
          "schemaVersion": 1,
          "identifier": "org.neuralrenderkit.demo.pixel-affine",
          "architecture": "\#(architecture)",
          "inputs": [
            {"name": "color", "dataType": "\#(inputDataType)", "layout": "nhwc", "shape": [1, "height", "width", 3]}
          ],
          "outputs": [
            {"name": "\#(outputName)", "dataType": "\#(inputDataType)", "layout": "nhwc", "shape": [1, "height", "width", 3]}
          ],
          "state": \#(stateJSON),
          "weights": {
            "file": "weights.safetensors",
            "sha256": "\#(digest)",
            "tensors": [
              {"name": "scale", "dataType": "float32", "shape": [3]},
              {"name": "bias", "dataType": "float32", "shape": [3]}
            ]
          }
        }
        """#
    }
}
