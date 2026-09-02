import Foundation
import XCTest
@testable import NeuralRenderCore

final class TensorTypesTests: XCTestCase {
    func testNTHWCLayoutRoundTripsThroughPublicJSONContract() throws {
        let encoded = try JSONEncoder().encode(TensorLayout.nthwc)

        XCTAssertEqual(String(decoding: encoded, as: UTF8.self), "\"nthwc\"")
        XCTAssertEqual(
            try JSONDecoder().decode(TensorLayout.self, from: encoded),
            .nthwc
        )
    }

    func testFloat32NHWCDescriptorComputesLiteralByteCount() throws {
        let descriptor = try TensorDescriptor(
            name: "color",
            shape: [1, 2, 3, 4],
            dataType: .float32,
            layout: .nhwc
        )

        XCTAssertEqual(descriptor.elementCount, 24)
        XCTAssertEqual(descriptor.byteCount, 96)
    }

    func testDescriptorRejectsZeroDimension() {
        XCTAssertThrowsError(
            try TensorDescriptor(
                name: "color",
                shape: [1, 0, 3],
                dataType: .float16,
                layout: .nhwc
            )
        ) {
            XCTAssertEqual($0 as? TensorError, .invalidDimension(0))
        }
    }

    func testDescriptorRejectsEmptyShape() {
        XCTAssertThrowsError(
            try TensorDescriptor(
                name: "color",
                shape: [],
                dataType: .float32,
                layout: .vector
            )
        ) {
            XCTAssertEqual($0 as? TensorError, .emptyShape)
        }
    }

    func testDescriptorRejectsEmptyName() {
        XCTAssertThrowsError(
            try TensorDescriptor(
                name: "",
                shape: [1],
                dataType: .float32,
                layout: .vector
            )
        ) {
            XCTAssertEqual($0 as? TensorError, .emptyName)
        }
    }

    func testDescriptorRejectsElementCountOverflow() {
        XCTAssertThrowsError(
            try TensorDescriptor(
                name: "oversized",
                shape: [Int.max, 2],
                dataType: .float16,
                layout: .vector
            )
        ) {
            XCTAssertEqual($0 as? TensorError, .elementCountOverflow)
        }
    }

    func testDescriptorRejectsByteCountOverflow() {
        XCTAssertThrowsError(
            try TensorDescriptor(
                name: "oversized",
                shape: [Int.max],
                dataType: .float32,
                layout: .vector
            )
        ) {
            XCTAssertEqual($0 as? TensorError, .byteCountOverflow)
        }
    }

    func testHostTensorRejectsTruncatedBytes() throws {
        let descriptor = try TensorDescriptor(
            name: "scale",
            shape: [3],
            dataType: .float32,
            layout: .vector
        )

        XCTAssertThrowsError(
            try HostTensor(descriptor: descriptor, bytes: Data(count: 8))
        ) {
            XCTAssertEqual($0 as? TensorError, .byteCount(expected: 12, actual: 8))
        }
    }
}
