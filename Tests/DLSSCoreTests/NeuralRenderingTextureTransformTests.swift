import Foundation
import XCTest

@testable import DLSSCore

final class NeuralRenderingTextureTransformTests: XCTestCase {
  func testPointSamplesLogicalPixelsFromBackingResource() throws {
    let resource = try tensor(
      name: "controlMask",
      shape: [1, 2, 4, 3],
      values: (0..<24).map(Float.init)
    )
    let transform = try NeuralRenderingTextureTransform(
      baseX: 1,
      baseY: 0,
      extentWidth: 3,
      extentHeight: 2,
      resourceWidth: 4,
      resourceHeight: 2
    )

    let sampled = try transform.pointSample(
      resource,
      logicalWidth: 2,
      logicalHeight: 1
    )

    XCTAssertEqual(sampled.descriptor.name, "controlMask")
    XCTAssertEqual(sampled.descriptor.shape, [1, 1, 2, 3])
    XCTAssertEqual(floatValues(in: sampled), [15, 16, 17, 21, 22, 23])
  }

  func testPointSamplingRejectsMismatchedBackingResource() throws {
    let resource = try tensor(
      name: "color",
      shape: [1, 2, 3, 3],
      values: [Float](repeating: 0, count: 18)
    )
    let transform = try NeuralRenderingTextureTransform(
      baseX: 0,
      baseY: 0,
      extentWidth: 4,
      extentHeight: 2,
      resourceWidth: 4,
      resourceHeight: 2
    )

    XCTAssertThrowsError(
      try transform.pointSample(resource, logicalWidth: 2, logicalHeight: 1)
    ) {
      XCTAssertEqual(
        $0 as? NeuralRenderingTextureTransformError,
        .resourceShapeMismatch(
          name: "color",
          expectedWidth: 4,
          expectedHeight: 2,
          actualWidth: 3,
          actualHeight: 2
        )
      )
    }
  }

  func testPointSamplingRejectsNonNHWCBackingResource() throws {
    let resource = try HostTensor(
      descriptor: TensorDescriptor(
        name: "color",
        shape: [1, 3, 2, 4],
        dataType: .float32,
        layout: .nchw
      ),
      bytes: Data(count: 24 * MemoryLayout<Float>.size)
    )
    let transform = try NeuralRenderingTextureTransform(
      baseX: 0,
      baseY: 0,
      extentWidth: 4,
      extentHeight: 2,
      resourceWidth: 4,
      resourceHeight: 2
    )

    XCTAssertThrowsError(
      try transform.pointSample(resource, logicalWidth: 2, logicalHeight: 1)
    ) {
      XCTAssertEqual(
        $0 as? NeuralRenderingTextureTransformError,
        .expectedBatchOneNHWC(
          name: "color",
          shape: [1, 3, 2, 4],
          layout: .nchw
        )
      )
    }
  }

  func testPointSamplingRejectsNonPositiveLogicalExtent() throws {
    let resource = try tensor(
      name: "color",
      shape: [1, 2, 4, 3],
      values: [Float](repeating: 0, count: 24)
    )
    let transform = try NeuralRenderingTextureTransform(
      baseX: 0,
      baseY: 0,
      extentWidth: 4,
      extentHeight: 2,
      resourceWidth: 4,
      resourceHeight: 2
    )

    XCTAssertThrowsError(
      try transform.pointSample(resource, logicalWidth: 0, logicalHeight: 1)
    ) {
      XCTAssertEqual(
        $0 as? NeuralRenderingTextureTransformError,
        .invalidExtent(width: 0, height: 1)
      )
    }
  }

  private func tensor(
    name: String,
    shape: [Int],
    values: [Float]
  ) throws -> HostTensor {
    try HostTensor(
      descriptor: TensorDescriptor(
        name: name,
        shape: shape,
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: values.withUnsafeBytes { Data($0) }
    )
  }

  private func floatValues(in tensor: HostTensor) -> [Float] {
    tensor.bytes.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
  }
}
