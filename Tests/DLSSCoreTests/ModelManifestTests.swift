import Foundation
import XCTest
@testable import DLSSCore

final class ModelManifestTests: XCTestCase {
    func testDecodesFixedAndSymbolicDimensions() throws {
        let manifest = try ModelPackageManifest.decode(data: validManifestJSON())

        XCTAssertEqual(manifest.schemaVersion, 1)
        XCTAssertEqual(manifest.inputs[0].shape, [
            .fixed(1), .symbol("height"), .symbol("width"), .fixed(3),
        ])
        XCTAssertEqual(manifest.state.kind, .stateless)
        XCTAssertEqual(manifest.weights.tensors.map(\.name), ["scale", "bias"])
    }

    func testDecodesRecurrentStateContractWithNamedHistory() throws {
        let json = validManifestJSONString().replacingOccurrences(
            of: #""state": {"kind": "stateless"}"#,
            with: #""state": {"kind": "recurrent", "cadence": "consecutiveFrames", "resetPolicy": "explicitAndSceneCut", "tensors": [{"name": "history", "dataType": "float16", "layout": "nhwc", "shape": [1, "height", "width", 8]}]}"#
        )

        let manifest = try ModelPackageManifest.decode(data: Data(json.utf8))

        XCTAssertEqual(manifest.state.kind, .recurrent)
        XCTAssertEqual(manifest.state.cadence, .consecutiveFrames)
        XCTAssertEqual(manifest.state.resetPolicy, .explicitAndSceneCut)
        XCTAssertEqual(manifest.state.tensors.map(\.name), ["history"])
        XCTAssertEqual(
            manifest.state.tensors[0].shape,
            [.fixed(1), .symbol("height"), .symbol("width"), .fixed(8)]
        )
    }

    func testRejectsContradictoryTemporalStateContracts() {
        let statelessWithHistory = validManifestJSONString().replacingOccurrences(
            of: #""state": {"kind": "stateless"}"#,
            with: #""state": {"kind": "stateless", "tensors": [{"name": "history", "dataType": "float16", "layout": "vector", "shape": [8]}]}"#
        )
        XCTAssertThrowsError(
            try ModelPackageManifest.decode(data: Data(statelessWithHistory.utf8))
        ) {
            XCTAssertEqual($0 as? ManifestError, .statelessStateDeclaresTensors)
        }

        let recurrentWithoutHistory = validManifestJSONString().replacingOccurrences(
            of: #""state": {"kind": "stateless"}"#,
            with: #""state": {"kind": "recurrent", "cadence": "consecutiveFrames"}"#
        )
        XCTAssertThrowsError(
            try ModelPackageManifest.decode(data: Data(recurrentWithoutHistory.utf8))
        ) {
            XCTAssertEqual($0 as? ManifestError, .recurrentStateRequiresTensors)
        }

        let recurrentIndependent = validManifestJSONString().replacingOccurrences(
            of: #""state": {"kind": "stateless"}"#,
            with: #""state": {"kind": "recurrent", "cadence": "frameIndependent", "tensors": [{"name": "history", "dataType": "float16", "layout": "vector", "shape": [8]}]}"#
        )
        XCTAssertThrowsError(
            try ModelPackageManifest.decode(data: Data(recurrentIndependent.utf8))
        ) {
            XCTAssertEqual($0 as? ManifestError, .recurrentStateRequiresTemporalCadence)
        }
    }

    func testRejectsDuplicateStateTensorName() {
        let json = validManifestJSONString().replacingOccurrences(
            of: #""state": {"kind": "stateless"}"#,
            with: #""state": {"kind": "recurrent", "cadence": "orderedFrames", "tensors": [{"name": "history", "dataType": "float16", "layout": "vector", "shape": [8]}, {"name": "history", "dataType": "float16", "layout": "vector", "shape": [8]}]}"#
        )

        XCTAssertThrowsError(try ModelPackageManifest.decode(data: Data(json.utf8))) {
            XCTAssertEqual($0 as? ManifestError, .duplicateTensorName("history"))
        }
    }

    func testRejectsUnsupportedSchemaVersion() {
        let json = validManifestJSONString().replacingOccurrences(
            of: "\"schemaVersion\": 1",
            with: "\"schemaVersion\": 2"
        )

        XCTAssertThrowsError(try ModelPackageManifest.decode(data: Data(json.utf8))) {
            XCTAssertEqual($0 as? ManifestError, .unsupportedSchemaVersion(2))
        }
    }

    func testRejectsInvalidShapeSymbol() {
        let json = validManifestJSONString().replacingOccurrences(
            of: "\"height\"",
            with: "\"9height\""
        )

        XCTAssertThrowsError(try ModelPackageManifest.decode(data: Data(json.utf8))) {
            XCTAssertEqual($0 as? ManifestError, .invalidShapeSymbol("9height"))
        }
    }

    func testRejectsNonPositiveFixedInputDimension() {
        let json = validManifestJSONString().replacingOccurrences(
            of: "[1, \"height\", \"width\", 3]",
            with: "[0, \"height\", \"width\", 3]"
        )

        XCTAssertThrowsError(try ModelPackageManifest.decode(data: Data(json.utf8))) {
            XCTAssertEqual($0 as? ManifestError, .invalidShapeDimension(0))
        }
    }

    func testRejectsNonPositiveWeightDimension() {
        let json = validManifestJSONString().replacingOccurrences(
            of: "{\"name\": \"scale\", \"dataType\": \"float32\", \"shape\": [3]}",
            with: "{\"name\": \"scale\", \"dataType\": \"float32\", \"shape\": [0]}"
        )

        XCTAssertThrowsError(try ModelPackageManifest.decode(data: Data(json.utf8))) {
            XCTAssertEqual($0 as? ManifestError, .invalidShapeDimension(0))
        }
    }

    func testRejectsDuplicateInputName() {
        let json = validManifestJSONString().replacingOccurrences(
            of: "\"inputs\": [\n    {\"name\": \"color\", \"dataType\": \"float32\", \"layout\": \"nhwc\", \"shape\": [1, \"height\", \"width\", 3]}\n  ]",
            with: "\"inputs\": [\n    {\"name\": \"color\", \"dataType\": \"float32\", \"layout\": \"nhwc\", \"shape\": [1, \"height\", \"width\", 3]},\n    {\"name\": \"color\", \"dataType\": \"float32\", \"layout\": \"nhwc\", \"shape\": [1, \"height\", \"width\", 3]}\n  ]"
        )

        XCTAssertThrowsError(try ModelPackageManifest.decode(data: Data(json.utf8))) {
            XCTAssertEqual($0 as? ManifestError, .duplicateTensorName("color"))
        }
    }

    func testResolvesInputSymbolsToLiteralDimensions() throws {
        let manifest = try ModelPackageManifest.decode(data: validManifestJSON())
        let color = try makeTensor(
            name: "color",
            shape: [1, 2, 3, 3],
            dataType: .float32,
            layout: .nhwc
        )

        let bindings = try manifest.resolveInputs([color])

        XCTAssertEqual(bindings, ["height": 2, "width": 3])
    }

    func testResolvesOutputDescriptorFromLiteralBindings() throws {
        let manifest = try ModelPackageManifest.decode(data: validManifestJSON())

        let descriptors = try manifest.resolveOutputDescriptors(
            bindings: ["height": 2, "width": 3]
        )

        XCTAssertEqual(descriptors.count, 1)
        XCTAssertEqual(descriptors[0].name, "color")
        XCTAssertEqual(descriptors[0].shape, [1, 2, 3, 3])
        XCTAssertEqual(descriptors[0].dataType, .float32)
        XCTAssertEqual(descriptors[0].layout, .nhwc)
    }

    func testScaledOutputDimensionsResolveFromInputBindings() throws {
        let json = validManifestJSONString().replacingOccurrences(
            of: #""shape": [1, "height", "width", 3]"#,
            with: #""shape": [1, {"symbol": "height", "multiplier": 4}, {"symbol": "width", "multiplier": 4}, 3]"#
        )
        let manifest = try ModelPackageManifest.decode(data: Data(json.utf8))

        XCTAssertEqual(manifest.outputs[0].shape, [
            .fixed(1),
            .scaled(symbol: "height", multiplier: 4),
            .scaled(symbol: "width", multiplier: 4),
            .fixed(3),
        ])
        XCTAssertEqual(
            try manifest.resolveOutputDescriptors(
                bindings: ["height": 2, "width": 3]
            )[0].shape,
            [1, 8, 12, 3]
        )
    }

    func testRejectsNonPositiveScaledDimensionMultiplier() {
        let json = validManifestJSONString().replacingOccurrences(
            of: #""shape": [1, "height", "width", 3]"#,
            with: #""shape": [1, {"symbol": "height", "multiplier": 0}, "width", 3]"#
        )

        XCTAssertThrowsError(
            try ModelPackageManifest.decode(data: Data(json.utf8))
        ) {
            XCTAssertEqual($0 as? ManifestError, .invalidShapeMultiplier(0))
        }
    }

    func testRejectsUnboundOutputSymbol() throws {
        let manifest = try ModelPackageManifest.decode(data: validManifestJSON())

        XCTAssertThrowsError(
            try manifest.resolveOutputDescriptors(bindings: ["width": 3])
        ) {
            XCTAssertEqual($0 as? ManifestError, .unboundShapeSymbol("height"))
        }
    }

    func testRejectsMissingInput() throws {
        let manifest = try ModelPackageManifest.decode(data: validManifestJSON())

        XCTAssertThrowsError(try manifest.resolveInputs([])) {
            XCTAssertEqual($0 as? ManifestError, .missingInput("color"))
        }
    }

    func testRejectsExtraInput() throws {
        let manifest = try ModelPackageManifest.decode(data: validManifestJSON())
        let color = try makeTensor(
            name: "color", shape: [1, 2, 3, 3], dataType: .float32, layout: .nhwc
        )
        let unused = try makeTensor(
            name: "unused", shape: [1], dataType: .float32, layout: .vector
        )

        XCTAssertThrowsError(try manifest.resolveInputs([color, unused])) {
            XCTAssertEqual($0 as? ManifestError, .extraInput("unused"))
        }
    }

    func testRejectsWrongInputLayout() throws {
        let manifest = try ModelPackageManifest.decode(data: validManifestJSON())
        let color = try makeTensor(
            name: "color", shape: [1, 2, 3, 3], dataType: .float32, layout: .nchw
        )

        XCTAssertThrowsError(try manifest.resolveInputs([color])) {
            XCTAssertEqual(
                $0 as? ManifestError,
                .layoutMismatch(name: "color", expected: .nhwc, actual: .nchw)
            )
        }
    }

    func testRejectsWrongInputDataType() throws {
        let manifest = try ModelPackageManifest.decode(data: validManifestJSON())
        let color = try makeTensor(
            name: "color", shape: [1, 2, 3, 3], dataType: .float16, layout: .nhwc
        )

        XCTAssertThrowsError(try manifest.resolveInputs([color])) {
            XCTAssertEqual(
                $0 as? ManifestError,
                .dataTypeMismatch(name: "color", expected: .float32, actual: .float16)
            )
        }
    }

    func testRejectsWrongInputRank() throws {
        let manifest = try ModelPackageManifest.decode(data: validManifestJSON())
        let color = try makeTensor(
            name: "color", shape: [2, 3, 3], dataType: .float32, layout: .nhwc
        )

        XCTAssertThrowsError(try manifest.resolveInputs([color])) {
            XCTAssertEqual(
                $0 as? ManifestError,
                .rankMismatch(name: "color", expected: 4, actual: 3)
            )
        }
    }

    func testRejectsWrongFixedDimension() throws {
        let manifest = try ModelPackageManifest.decode(data: validManifestJSON())
        let color = try makeTensor(
            name: "color", shape: [2, 2, 3, 3], dataType: .float32, layout: .nhwc
        )

        XCTAssertThrowsError(try manifest.resolveInputs([color])) {
            XCTAssertEqual(
                $0 as? ManifestError,
                .fixedDimensionMismatch(name: "color", axis: 0, expected: 1, actual: 2)
            )
        }
    }

    func testRejectsConflictingSymbolBindingsAcrossInputs() throws {
        let json = validManifestJSONString().replacingOccurrences(
            of: "\"inputs\": [\n    {\"name\": \"color\", \"dataType\": \"float32\", \"layout\": \"nhwc\", \"shape\": [1, \"height\", \"width\", 3]}\n  ]",
            with: "\"inputs\": [\n    {\"name\": \"color\", \"dataType\": \"float32\", \"layout\": \"nhwc\", \"shape\": [1, \"height\", \"width\", 3]},\n    {\"name\": \"depth\", \"dataType\": \"float32\", \"layout\": \"nhwc\", \"shape\": [1, \"height\", \"width\", 1]}\n  ]"
        )
        let manifest = try ModelPackageManifest.decode(data: Data(json.utf8))
        let color = try makeTensor(
            name: "color", shape: [1, 2, 3, 3], dataType: .float32, layout: .nhwc
        )
        let depth = try makeTensor(
            name: "depth", shape: [1, 4, 3, 1], dataType: .float32, layout: .nhwc
        )

        XCTAssertThrowsError(try manifest.resolveInputs([color, depth])) {
            XCTAssertEqual(
                $0 as? ManifestError,
                .symbolMismatch(symbol: "height", expected: 2, actual: 4)
            )
        }
    }

    private func makeTensor(
        name: String,
        shape: [Int],
        dataType: TensorDataType,
        layout: TensorLayout
    ) throws -> HostTensor {
        let descriptor = try TensorDescriptor(
            name: name,
            shape: shape,
            dataType: dataType,
            layout: layout
        )
        return try HostTensor(descriptor: descriptor, bytes: Data(count: descriptor.byteCount))
    }

    private func validManifestJSON() -> Data {
        Data(validManifestJSONString().utf8)
    }

    private func validManifestJSONString() -> String {
        #"""
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
            "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "tensors": [
              {"name": "scale", "dataType": "float32", "shape": [3]},
              {"name": "bias", "dataType": "float32", "shape": [3]}
            ]
          }
        }
        """#
    }
}
