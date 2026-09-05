import CryptoKit
import Foundation
import MLX
import XCTest
@testable import DLSSCore
@testable import DLSSMLX

final class SafeTensorValidationTests: XCTestCase {
    func testAcceptsExactDeclaredWeightsFromRealSafeTensorFile() throws {
        let package = try makePackage(arrays: [
            "scale": MLXArray([Float(1), 2, 3]),
            "bias": MLXArray([Float(0), 0, 0]),
        ])
        defer { try? FileManager.default.removeItem(at: package.url) }

        let weights = try SafeTensorValidation.loadAndValidate(package: package)

        XCTAssertEqual(weights.names, ["bias", "scale"])
    }

    func testRejectsMissingDeclaredWeight() throws {
        let package = try makePackage(arrays: [
            "scale": MLXArray([Float(1), 2, 3]),
        ])
        defer { try? FileManager.default.removeItem(at: package.url) }

        XCTAssertThrowsError(try SafeTensorValidation.loadAndValidate(package: package)) {
            XCTAssertEqual($0 as? MLXBackendError, .missingWeight("bias"))
        }
    }

    func testRejectsExtraWeight() throws {
        let package = try makePackage(arrays: [
            "scale": MLXArray([Float(1), 2, 3]),
            "bias": MLXArray([Float(0), 0, 0]),
            "unused": MLXArray([Float(9)]),
        ])
        defer { try? FileManager.default.removeItem(at: package.url) }

        XCTAssertThrowsError(try SafeTensorValidation.loadAndValidate(package: package)) {
            XCTAssertEqual($0 as? MLXBackendError, .extraWeight("unused"))
        }
    }

    func testRejectsWeightShapeMismatch() throws {
        let package = try makePackage(arrays: [
            "scale": MLXArray([Float(1), 2, 3], [1, 3]),
            "bias": MLXArray([Float(0), 0, 0]),
        ])
        defer { try? FileManager.default.removeItem(at: package.url) }

        XCTAssertThrowsError(try SafeTensorValidation.loadAndValidate(package: package)) {
            XCTAssertEqual(
                $0 as? MLXBackendError,
                .weightShapeMismatch(name: "scale", expected: [3], actual: [1, 3])
            )
        }
    }

    func testRejectsWeightDataTypeMismatch() throws {
        let package = try makePackage(arrays: [
            "scale": MLXArray([Float16(1), 2, 3]),
            "bias": MLXArray([Float(0), 0, 0]),
        ])
        defer { try? FileManager.default.removeItem(at: package.url) }

        XCTAssertThrowsError(try SafeTensorValidation.loadAndValidate(package: package)) {
            XCTAssertEqual(
                $0 as? MLXBackendError,
                .weightDataTypeMismatch(
                    name: "scale",
                    expected: .float32,
                    actual: "float16"
                )
            )
        }
    }

    private func makePackage(arrays: [String: MLXArray]) throws -> LoadedModelPackage {
        let packageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("dlssmodel")
        try FileManager.default.createDirectory(at: packageURL, withIntermediateDirectories: false)

        let weightsURL = packageURL.appendingPathComponent("weights.safetensors")
        try save(arrays: arrays, url: weightsURL, stream: .cpu)
        let digest = SHA256.hash(data: try Data(contentsOf: weightsURL))
            .map { String(format: "%02x", $0) }
            .joined()
        try Data(manifestJSON(digest: digest).utf8)
            .write(to: packageURL.appendingPathComponent("manifest.json"))
        return try ModelPackageLoader.load(url: packageURL)
    }

    private func manifestJSON(digest: String) -> String {
        return #"""
        {
          "schemaVersion": 1,
          "identifier": "org.mlxdlss.demo.pixel-affine",
          "architecture": "mlxdlss.pixel-affine.v1",
          "inputs": [
            {"name": "color", "dataType": "float32", "layout": "nhwc", "shape": [1, "height", "width", 3]}
          ],
          "outputs": [
            {"name": "color", "dataType": "float32", "layout": "nhwc", "shape": [1, "height", "width", 3]}
          ],
          "state": {"kind": "stateless"},
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

