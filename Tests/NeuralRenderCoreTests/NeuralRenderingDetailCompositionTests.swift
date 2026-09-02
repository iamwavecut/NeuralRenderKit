import Foundation
import XCTest

@testable import NeuralRenderCore

final class NeuralRenderingDetailCompositionTests: XCTestCase {
  private func tensor(_ values: [Float], height: Int, width: Int) throws -> HostTensor {
    let descriptor = try TensorDescriptor(
      name: "color", shape: [1, height, width, 3], dataType: .float32, layout: .nhwc
    )
    return try HostTensor(
      descriptor: descriptor, bytes: values.withUnsafeBufferPointer { Data(buffer: $0) })
  }

  private func floats(_ tensor: HostTensor) -> [Float] {
    tensor.bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
  }

  private func gradient(height: Int, width: Int) -> [Float] {
    (0..<(height * width * 3)).map { index in
      let pixel = index / 3
      let x = Float(pixel % width) / Float(width - 1)
      let y = Float(pixel / width) / Float(height - 1)
      return [0.2 + 0.6 * x, 0.3 + 0.4 * y, 0.5][index % 3]
    }
  }

  func testResampleToSameSizeIsIdentity() throws {
    let source = try tensor(gradient(height: 12, width: 16), height: 12, width: 16)
    let same = try NeuralRenderingDetailComposition.resample(source, width: 16, height: 12)
    XCTAssertEqual(same, source)
  }

  func testResampleUpAndDownRecoversSmoothContent() throws {
    let values = gradient(height: 24, width: 32)
    let source = try tensor(values, height: 24, width: 32)
    let up = try NeuralRenderingDetailComposition.resample(source, width: 64, height: 48)
    XCTAssertEqual(up.descriptor.shape, [1, 48, 64, 3])
    let down = try NeuralRenderingDetailComposition.resample(up, width: 32, height: 24)
    let errors = zip(floats(down), values).map { abs($0 - $1) }
    XCTAssertLessThan(errors.reduce(0, +) / Float(errors.count), 0.01)
  }

  func testUnitStrengthsReturnTheOutputUnchanged() throws {
    let input = try tensor(gradient(height: 16, width: 16), height: 16, width: 16)
    let output = try tensor(gradient(height: 16, width: 16).map { min($0 + 0.05, 1) }, height: 16, width: 16)
    let composed = try NeuralRenderingDetailComposition.compose(
      input: input, output: output, detailStrength: 1, colourStrength: 1)
    XCTAssertEqual(composed, output)
  }

  func testZeroStrengthsReturnTheInput() throws {
    let input = try tensor(gradient(height: 16, width: 16), height: 16, width: 16)
    let output = try tensor(gradient(height: 16, width: 16).map { min($0 + 0.05, 1) }, height: 16, width: 16)
    let composed = try NeuralRenderingDetailComposition.compose(
      input: input, output: output, detailStrength: 0, colourStrength: 0)
    let errors = zip(floats(composed), floats(input)).map { abs($0 - $1) }
    XCTAssertLessThan(errors.max() ?? 1, 1e-6)
  }

  func testUniformChangeIsPureColourAndDetailStrengthDoesNotTouchIt() throws {
    let base = gradient(height: 20, width: 20)
    let input = try tensor(base, height: 20, width: 20)
    let output = try tensor(base.map { $0 + 0.1 }, height: 20, width: 20)
    let detailOnly = try NeuralRenderingDetailComposition.compose(
      input: input, output: output, detailStrength: 5, colourStrength: 1)
    let outputErrors = zip(floats(detailOnly), floats(output)).map { abs($0 - $1) }
    XCTAssertLessThan(outputErrors.max() ?? 1, 1e-4)
    let colourOff = try NeuralRenderingDetailComposition.compose(
      input: input, output: output, detailStrength: 1, colourStrength: 0)
    let inputErrors = zip(floats(colourOff), floats(input)).map { abs($0 - $1) }
    XCTAssertLessThan(inputErrors.max() ?? 1, 1e-4)
  }

  func testDetailStrengthScalesTheHighPassPart() throws {
    let base = gradient(height: 20, width: 20)
    var changed = base
    changed[(10 * 20 + 10) * 3] += 0.2  // a single-pixel detail
    let input = try tensor(base, height: 20, width: 20)
    let output = try tensor(changed, height: 20, width: 20)
    let doubled = try NeuralRenderingDetailComposition.compose(
      input: input, output: output, detailStrength: 2, colourStrength: 1)
    let index = (10 * 20 + 10) * 3
    let single = floats(output)[index] - floats(input)[index]
    let composed = floats(doubled)[index] - floats(input)[index]
    XCTAssertGreaterThan(composed, 1.5 * single)
    XCTAssertLessThan(composed, 2.0 * single + 1e-4)
  }

  func testGaussianKernelIsNormalizedAndSymmetric() {
    let kernel = NeuralRenderingDetailComposition.gaussianKernel(radius: 4)
    XCTAssertEqual(kernel.count, 25)
    XCTAssertEqual(kernel.reduce(0, +), 1, accuracy: 1e-5)
    XCTAssertEqual(kernel[0], kernel[24], accuracy: 1e-7)
  }
}
