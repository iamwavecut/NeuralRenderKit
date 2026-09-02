import Foundation
import XCTest

@testable import NeuralRenderCore

final class NeuralRenderingAlignedSubrectTests: XCTestCase {
  func testExtractsNHWCRowsUsingBackingResourceStride() throws {
    let values = (0..<24).map(Float.init)
    let tensor = try HostTensor(
      descriptor: TensorDescriptor(
        name: "motion",
        shape: [1, 3, 4, 2],
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: values.withUnsafeBytes { Data($0) }
    )
    let subrect = try NeuralRenderingAlignedSubrect(
      baseX: 1,
      baseY: 1,
      width: 2,
      height: 2
    )

    let extracted = try subrect.extract(from: tensor)

    XCTAssertEqual(extracted.descriptor.shape, [1, 2, 2, 2])
    XCTAssertEqual(floatValues(in: extracted), [10, 11, 12, 13, 18, 19, 20, 21])
  }

  func testRejectsSubrectOutsideBackingResource() throws {
    let tensor = try HostTensor(
      descriptor: TensorDescriptor(
        name: "history",
        shape: [1, 2, 3, 3],
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: Data(count: 2 * 3 * 3 * MemoryLayout<Float>.size)
    )
    let subrect = try NeuralRenderingAlignedSubrect(
      baseX: 2,
      baseY: 0,
      width: 2,
      height: 1
    )

    XCTAssertThrowsError(try subrect.extract(from: tensor)) {
      XCTAssertEqual(
        $0 as? NeuralRenderingAlignedSubrectError,
        .outsideResource(
          baseX: 2,
          baseY: 0,
          width: 2,
          height: 1,
          resourceWidth: 3,
          resourceHeight: 2
        )
      )
    }
  }

  private func floatValues(in tensor: HostTensor) -> [Float] {
    tensor.bytes.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
  }
}
