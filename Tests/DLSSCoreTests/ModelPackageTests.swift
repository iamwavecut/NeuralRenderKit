import Foundation
import XCTest
@testable import DLSSCore

final class ModelPackageTests: XCTestCase {
    func testLoadsPackageWithHandDerivedDigest() throws {
        let packageURL = try makePackage(
            weightBytes: Data([0x01, 0x02, 0x03]),
            digest: "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81"
        )
        defer { try? FileManager.default.removeItem(at: packageURL) }

        let package = try ModelPackageLoader.load(url: packageURL)

        XCTAssertEqual(package.url, packageURL)
        XCTAssertEqual(package.manifest.identifier, "org.mlxdlss.demo.pixel-affine")
        XCTAssertEqual(package.weightsURL, packageURL.appendingPathComponent("weights.safetensors"))
    }

    func testRejectsUnsafeWeightPathsBeforeOpeningThem() throws {
        let candidates = [
            "/tmp/weights.safetensors",
            "nested/weights.safetensors",
            "nested\\weights.safetensors",
            ".",
            "..",
        ]

        for candidate in candidates {
            let packageURL = try makePackage(weightFile: candidate, writeWeightFile: false)
            defer { try? FileManager.default.removeItem(at: packageURL) }

            XCTAssertThrowsError(try ModelPackageLoader.load(url: packageURL), candidate) {
                XCTAssertEqual($0 as? ModelPackageError, .unsafeWeightFile(candidate))
            }
        }
    }

    func testRejectsDigestMismatch() throws {
        let packageURL = try makePackage(
            weightBytes: Data([0x01, 0x02, 0x03]),
            digest: "0000000000000000000000000000000000000000000000000000000000000000"
        )
        defer { try? FileManager.default.removeItem(at: packageURL) }

        XCTAssertThrowsError(try ModelPackageLoader.load(url: packageURL)) {
            XCTAssertEqual(
                $0 as? ModelPackageError,
                .digestMismatch(
                    expected: "0000000000000000000000000000000000000000000000000000000000000000",
                    actual: "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81"
                )
            )
        }
    }

    func testRejectsInvalidDigestEncoding() throws {
        let packageURL = try makePackage(
            weightBytes: Data([0x01, 0x02, 0x03]),
            digest: "NOT-A-SHA256"
        )
        defer { try? FileManager.default.removeItem(at: packageURL) }

        XCTAssertThrowsError(try ModelPackageLoader.load(url: packageURL)) {
            XCTAssertEqual($0 as? ModelPackageError, .invalidDigest("NOT-A-SHA256"))
        }
    }

    func testRejectsMissingManifest() throws {
        let packageURL = try makeEmptyDirectory()
        defer { try? FileManager.default.removeItem(at: packageURL) }

        XCTAssertThrowsError(try ModelPackageLoader.load(url: packageURL)) {
            XCTAssertEqual($0 as? ModelPackageError, .missingManifest)
        }
    }

    func testRejectsManifestLargerThanOneMiB() throws {
        let packageURL = try makeEmptyDirectory()
        defer { try? FileManager.default.removeItem(at: packageURL) }
        try Data(count: 1_048_577).write(to: packageURL.appendingPathComponent("manifest.json"))

        XCTAssertThrowsError(try ModelPackageLoader.load(url: packageURL)) {
            XCTAssertEqual($0 as? ModelPackageError, .manifestTooLarge(maxBytes: 1_048_576))
        }
    }

    func testRejectsMissingWeightFile() throws {
        let packageURL = try makePackage(writeWeightFile: false)
        defer { try? FileManager.default.removeItem(at: packageURL) }

        XCTAssertThrowsError(try ModelPackageLoader.load(url: packageURL)) {
            XCTAssertEqual($0 as? ModelPackageError, .missingWeightFile("weights.safetensors"))
        }
    }

    func testRejectsFileInsteadOfPackageDirectory() throws {
        let fileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("dlssmodel")
        try Data().write(to: fileURL)
        defer { try? FileManager.default.removeItem(at: fileURL) }

        XCTAssertThrowsError(try ModelPackageLoader.load(url: fileURL)) {
            XCTAssertEqual($0 as? ModelPackageError, .packageIsNotDirectory)
        }
    }

    private func makePackage(
        weightFile: String = "weights.safetensors",
        weightBytes: Data = Data([0x01, 0x02, 0x03]),
        digest: String = "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81",
        writeWeightFile: Bool = true
    ) throws -> URL {
        let packageURL = try makeEmptyDirectory()
        try Data(manifestJSON(weightFile: weightFile, digest: digest).utf8)
            .write(to: packageURL.appendingPathComponent("manifest.json"))
        if writeWeightFile {
            try weightBytes.write(to: packageURL.appendingPathComponent("weights.safetensors"))
        }
        return packageURL
    }

    private func makeEmptyDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("dlssmodel")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        return url
    }

    private func manifestJSON(weightFile: String, digest: String) -> String {
        let escapedWeightFile = weightFile
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
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
            "file": "\#(escapedWeightFile)",
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
