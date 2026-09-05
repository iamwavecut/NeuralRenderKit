import Darwin
import MLX
import XCTest

final class MLXManagedBufferTests: XCTestCase {
  func testPageAlignedManagedPointerIsObservedWithoutHostCopy() throws {
    let owner = try PageAlignedPointer(byteCount: 4096)
    let values = owner.pointer.bindMemory(to: Float.self, capacity: 4)
    values.initialize(from: [1, 2, 3, 4], count: 4)
    let input = MLXArray(
      rawPointer: owner.pointer,
      [4],
      dtype: .float32
    ) { [owner] in
      owner.finalizerCalled = true
    }

    values[0] = 7
    let output = input + 1
    eval(output)

    XCTAssertFalse(owner.finalizerCalled)
    XCTAssertEqual(output.asArray(Float.self), [8, 3, 4, 5])
  }
}

private final class PageAlignedPointer: @unchecked Sendable {
  let pointer: UnsafeMutableRawPointer
  var finalizerCalled = false

  init(byteCount: Int) throws {
    var result: UnsafeMutableRawPointer?
    let status = posix_memalign(&result, 4096, byteCount)
    guard status == 0, let result else {
      throw POSIXError(POSIXErrorCode(rawValue: status) ?? .ENOMEM)
    }
    pointer = result
  }

  deinit {
    free(pointer)
  }
}
