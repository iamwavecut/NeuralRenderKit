import Foundation
import MLX
import DLSSCore
import XCTest

@testable import DLSSMLX

final class NeuralRenderingTransformerOperatorTests: XCTestCase {
  func testGraphContractAcceptsVendorThreeTwentyExtent() {
    XCTAssertEqual(NeuralRenderingGraphContract.minimumInputExtent, 128)
    XCTAssertEqual(NeuralRenderingGraphContract.inputExtentMultiple, 64)
    XCTAssertTrue(320.isMultiple(of: NeuralRenderingGraphContract.inputExtentMultiple))
  }

  func testRecoveredGraphCoversEveryBlockInExecutionOrder() {
    let blocks = NeuralRenderingGraphContract.blocks

    XCTAssertEqual(blocks.map(\.index), Array(0...70))
    XCTAssertEqual(blocks.filter { $0.family == .pre }.count, 1)
    XCTAssertEqual(blocks.filter { $0.family == .downsample }.count, 4)
    XCTAssertEqual(blocks.filter { $0.family == .splitWindow }.count, 15)
    XCTAssertEqual(blocks.filter { $0.family == .splitDownsample }.count, 1)
    XCTAssertEqual(blocks.filter { $0.family == .globalAttention }.count, 8)
    XCTAssertEqual(blocks.filter { $0.family == .bridge }.count, 1)
    XCTAssertEqual(blocks.filter { $0.family == .upsample }.count, 4)
    XCTAssertEqual(blocks.filter { $0.family == .post }.count, 1)
  }

  func testRecoveredEncoderAndDecoderWidthsAreSymmetric() {
    XCTAssertEqual(
      NeuralRenderingGraphContract.encoderStages.map(\.channels),
      [32, 64, 128, 256]
    )
    XCTAssertEqual(
      NeuralRenderingGraphContract.decoderStages.map(\.channels),
      [256, 128, 64, 32]
    )
    XCTAssertEqual(
      NeuralRenderingGraphContract.encoderStages.map(\.blockRange),
      [0...4, 5...8, 9...14, 15...22]
    )
    XCTAssertEqual(
      NeuralRenderingGraphContract.decoderStages.map(\.blockRange),
      [48...55, 56...61, 62...65, 66...69]
    )
  }

  func testRecoveredWindowOriginsFollowVendorQuadrantCycle() {
    let expected = [
      NeuralRenderingWindowOrigin.zero,
      NeuralRenderingWindowOrigin(y: -4, x: -4),
      NeuralRenderingWindowOrigin(y: 0, x: -4),
      NeuralRenderingWindowOrigin(y: -4, x: 0),
    ]

    XCTAssertEqual(
      (9...14).map(NeuralRenderingGraphContract.windowOrigin(for:)),
      expected + Array(expected.prefix(2))
    )
    XCTAssertEqual(
      NeuralRenderingGraphContract.windowOrigin(for: 56),
      NeuralRenderingWindowOrigin(y: 0, x: -4)
    )
    XCTAssertEqual(
      NeuralRenderingGraphContract.windowOrigin(for: 70),
      NeuralRenderingWindowOrigin(y: -4, x: -4)
    )
  }

  func testQuadraticGateActivationMatchesRecoveredSASSFormula() {
    let input = MLXArray([Float(-5), -4, -1, 0, 1, 4, 5])

    let output = NeuralRenderingTransformerOperations.quadraticGateActivation(input)
    eval(output)

    let expected: [Float] = [
      0,
      0,
      -0.502_929_687_5,
      0,
      1.286_132_812_5,
      7.156_25,
      8.945_312_5,
    ]
    let actual = output.asArray(Float.self)
    XCTAssertEqual(actual.count, expected.count)
    for (actualValue, expectedValue) in zip(actual, expected) {
      XCTAssertEqual(actualValue, expectedValue, accuracy: 0.000_001)
    }
  }

  func testFusedSimpleFeedForwardMatchesLiteralOperation() {
    let input = MLXArray(
      (0..<(8 * 32)).map { sin(Float($0) * 0.01) },
      [1, 2, 4, 32]
    )
    let expansion = MLXArray(
      (0..<(32 * 128)).map { cos(Float($0) * 0.003) * 0.05 },
      [32, 128]
    )
    let projection = MLXArray(
      (0..<(128 * 32)).map { sin(Float($0) * 0.005) * 0.05 },
      [128, 32]
    )
    let expected = matmul(
      NeuralRenderingTransformerOperations.e4m3RoundTrip(
        NeuralRenderingTransformerOperations.quadraticGateActivation(
          matmul(input, expansion)
        )
      ),
      projection
    )
    let actual = NeuralRenderingTransformerOperations.fusedSimpleFeedForward(
      input,
      expansionWeight: expansion,
      projectionWeight: projection
    )
    eval(expected, actual)
    let errors = zip(actual.asArray(Float.self), expected.asArray(Float.self))
      .map { abs($0.0 - $0.1) }

    XCTAssertEqual(actual.shape, expected.shape)
    XCTAssertLessThanOrEqual(errors.max() ?? .infinity, 0.000_1)
  }

  func testChunkedFusedSimpleFeedForwardMatchesLiteralOperation() {
    let input = MLXArray(
      (0..<(16 * 32)).map { sin(Float($0) * 0.01) },
      [1, 4, 4, 32]
    )
    let expansion = MLXArray(
      (0..<(32 * 128)).map { cos(Float($0) * 0.003) * 0.05 },
      [32, 128]
    )
    let projection = MLXArray(
      (0..<(128 * 32)).map { sin(Float($0) * 0.005) * 0.05 },
      [128, 32]
    )
    let expected = matmul(
      NeuralRenderingTransformerOperations.e4m3RoundTrip(
        NeuralRenderingTransformerOperations.quadraticGateActivation(
          matmul(input, expansion)
        )
      ),
      projection
    )
    let actual = NeuralRenderingTransformerOperations.fusedSimpleFeedForward(
      input,
      expansionWeight: expansion,
      projectionWeight: projection,
      maximumIntermediateBytes: 4_096
    )
    eval(expected, actual)
    let errors = zip(actual.asArray(Float.self), expected.asArray(Float.self))
      .map { abs($0.0 - $0.1) }

    XCTAssertEqual(actual.shape, expected.shape)
    XCTAssertLessThanOrEqual(errors.max() ?? .infinity, 0.000_1)
  }

  func testE4M3RoundTripMatchesVendorLandmarksAndTies() {
    let input = MLXArray(
      [
        Float(0),
        Float(1.0 / 512.0),
        Float(7.0 / 512.0),
        Float(1.0 / 64.0),
        1.0625,
        1.1875,
        448,
        500,
        -1,
      ]
    )

    let output = NeuralRenderingTransformerOperations.e4m3RoundTrip(input)
    eval(output)

    XCTAssertEqual(
      output.asArray(Float.self),
      [0, 1.0 / 512.0, 7.0 / 512.0, 1.0 / 64.0, 1, 1.25, 448, 448, -1]
    )
  }

  func testCosineResidualAddsScaledSkipWithoutChangingShape() {
    let skip = MLXArray([Float(1), 2, 3, 4], [1, 1, 1, 4])
    let branch = MLXArray([Float(10), 20, 30, 40], [1, 1, 1, 4])
    let cosine = MLXArray([Float(0.5), 0.25, 0, -0.5])

    let output = NeuralRenderingTransformerOperations.cosineResidual(
      skip: skip,
      branch: branch,
      cosine: cosine
    )
    eval(output)

    XCTAssertEqual(output.shape, [1, 1, 1, 4])
    let expected: [Float] = [
      10.5,
      20.5,
      30,
      38,
    ]
    for (actual, expected) in zip(output.asArray(Float.self), expected) {
      XCTAssertEqual(actual, expected, accuracy: 0.000_01)
    }
  }

  func testCosineAttentionNormalizesQueryAndKeyBeforeSoftmax() {
    let input = MLXArray([Float(1), 0, 0, 1], [1, 2, 2])
    let qkvWeight = MLXArray(
      [
        Float(1), 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1,
      ],
      [2, 6]
    )
    let scale = MLXArray([Float(1)])
    let bias = MLXArray.zeros([1, 2, 2])
    let projection = MLXArray([Float(1), 0, 0, 1], [2, 2])

    let output = NeuralRenderingTransformerOperations.cosineAttention(
      input,
      qkvWeight: qkvWeight,
      attentionScale: scale,
      attentionBias: bias,
      projectionWeight: projection,
      headCount: 1
    )
    eval(output)

    let expected: [Float] = [
      0.75, 0.281_25,
      0.281_25, 0.75,
    ]
    XCTAssertEqual(output.shape, [1, 2, 2])
    for (actual, expected) in zip(output.asArray(Float.self), expected) {
      XCTAssertEqual(actual, expected, accuracy: 0.000_001)
    }
  }

  func testVendorApproximateSoftmaxMatchesRecoveredHalfBitAffineLandmarks() {
    var logits = [Float](repeating: -10, count: 64)
    logits[0] = 0

    let output = NeuralRenderingTransformerOperations.vendorApproximateSoftmax(
      MLXArray(logits, [1, 64])
    )
    eval(output)

    XCTAssertEqual(output.asArray(Float.self)[0], 0.875)
    XCTAssertEqual(
      Array(output.asArray(Float.self).dropFirst()),
      [Float](repeating: 0.001_953_125, count: 63)
    )
  }

  func testVendorApproximateSoftmaxSupportsRecoveredNinetySixTokenLaunch() {
    var logits = [Float](repeating: -10, count: 96)
    logits[0] = 0
    let output = NeuralRenderingTransformerOperations.vendorApproximateSoftmax(
      MLXArray(logits, [1, 96])
    )
    eval(output)
    let values = output.asArray(Float.self)

    XCTAssertEqual(values.count, 96)
    XCTAssertTrue(values.allSatisfy(\.isFinite))
    XCTAssertGreaterThan(values[0], 0.75)
    XCTAssertEqual(values.reduce(0, +), 1, accuracy: 0.02)
  }

  func testVendorCosineNormalize32MatchesRecoveredHalfReductionTree() {
    let values = (0..<32).map { Float($0 % 11 - 5) / 8 }
    let output = NeuralRenderingTransformerOperations.vendorCosineNormalize(
      MLXArray(values, [1, 32])
    )
    eval(output)

    func half(_ value: Float) -> Float { Float(Float16(value)) }
    var partial = [[Float]](
      repeating: [Float](repeating: 0, count: 2),
      count: 4
    )
    for lane in 0..<4 {
      for parity in 0..<2 {
        let channel = lane * 2 + parity
        let first = half(
          half(values[channel] * values[channel])
            + values[channel + 8] * values[channel + 8]
        )
        let second = half(
          half(values[channel + 16] * values[channel + 16])
            + values[channel + 24] * values[channel + 24]
        )
        partial[lane][parity] = half(first + second)
      }
    }
    var xorTwo = partial
    for lane in 0..<4 {
      for parity in 0..<2 {
        xorTwo[lane][parity] = half(
          partial[lane][parity] + partial[lane ^ 2][parity]
        )
      }
    }
    var xorOne = xorTwo
    for lane in 0..<4 {
      for parity in 0..<2 {
        xorOne[lane][parity] = half(
          xorTwo[lane][parity] + xorTwo[lane ^ 1][parity]
        )
      }
    }
    let norm = max(
      half(xorOne[0][0] + xorOne[0][1]),
      0.000_061_988_830_566_406_25
    )
    let reciprocal = half(1 / norm.squareRoot())
    let expected = values.map { half(half($0) * reciprocal) }

    XCTAssertEqual(output.shape, [1, 32])
    for (actual, expected) in zip(output.asArray(Float.self), expected) {
      XCTAssertEqual(actual, expected, accuracy: 0.000_5)
    }
  }

  func testExternalVendorCosineNormalizeMatchesCUDAFragmentCandidate() throws {
    guard
      let path = ProcessInfo.processInfo.environment["MLXDLSS_BLOCK48_NORM_FIXTURE"]
    else {
      throw XCTSkip(
        "set MLXDLSS_BLOCK48_NORM_FIXTURE to run the private normalization probe"
      )
    }
    let root = URL(fileURLWithPath: path, isDirectory: true)
    for name in ["query", "key"] {
      let input = try Data(contentsOf: root.appendingPathComponent("\(name)-input.f16"))
      let expected = try Data(
        contentsOf: root.appendingPathComponent("\(name)-normalized.f16")
      )
      XCTAssertEqual(input.count, expected.count)
      XCTAssertTrue(input.count.isMultiple(of: 32 * MemoryLayout<Float16>.size))
      let vectorCount = input.count / (32 * MemoryLayout<Float16>.size)
      let output = NeuralRenderingTransformerOperations.vendorCosineNormalize(
        MLXArray(input, [vectorCount, 32], dtype: .float16)
      )
      eval(output)
      let actual = output.asArray(Float16.self)
      let reference = expected.withUnsafeBytes { bytes in
        Array(bytes.bindMemory(to: Float16.self))
      }
      let matches = zip(actual, reference).filter { $0.0 == $0.1 }.count
      let mean =
        zip(actual, reference).reduce(Float.zero) {
          $0 + abs(Float($1.0) - Float($1.1))
        } / Float(reference.count)
      let maximum = zip(actual, reference).reduce(Float.zero) {
        max($0, abs(Float($1.0) - Float($1.1)))
      }
      print(
        "vendor cosine \(name): byteMatch="
          + "\(Float(matches) / Float(reference.count)) "
          + "mean=\(mean) max=\(maximum)"
      )
      XCTAssertGreaterThanOrEqual(
        Float(matches) / Float(reference.count),
        0.99,
        name
      )
      XCTAssertLessThanOrEqual(maximum, 0.001, name)
    }

    func decodeE4M3(_ bits: UInt8) -> Float16 {
      let exponent = Int((bits >> 3) & 0x0f)
      let mantissa = Float(bits & 0x07)
      let magnitude =
        exponent == 0
        ? mantissa * Float(sign: .plus, exponent: -9, significand: 1)
        : (1 + mantissa / 8)
          * Float(sign: .plus, exponent: exponent - 7, significand: 1)
      return Float16((bits & 0x80) == 0 ? magnitude : -magnitude)
    }
    let scaleData = try Data(
      contentsOf: root.appendingPathComponent("query-scale.f16")
    )
    for name in ["query", "key"] {
      let inputData = try Data(
        contentsOf: root.appendingPathComponent("\(name)-input.f16")
      )
      let expectedBits = try Data(
        contentsOf: root.appendingPathComponent("\(name)-published.e4m3")
      )
      let input = MLXArray(inputData, [9, 64, 8, 32], dtype: .float16)
        .transposed(0, 2, 1, 3)
      let output = NeuralRenderingTransformerOperations.vendorCosinePublish(
        input,
        scale: name == "query"
          ? MLXArray(scaleData, [8], dtype: .float16)
          : nil
      )
      eval(output)
      let actual = output.asArray(Float16.self)
      let rawReference = expectedBits.map(decodeE4M3)
      var reference: [Float16] = []
      reference.reserveCapacity(rawReference.count)
      for window in 0..<9 {
        for head in 0..<8 {
          for token in 0..<64 {
            let base = ((window * 64 + token) * 8 + head) * 32
            reference.append(contentsOf: rawReference[base..<base + 32])
          }
        }
      }
      XCTAssertEqual(actual, reference, "published \(name)")
    }
  }

  func testWindowBlockRunsFeedForwardBeforeAttentionResidual() {
    let input = MLXArray((0..<32).map { Float($0 + 1) }, [1, 4, 4, 2])
    let output = NeuralRenderingTransformerOperations.windowBlock(
      input,
      expansionWeight: MLXArray.zeros([2, 4]),
      feedForwardProjectionWeight: MLXArray.zeros([4, 2]),
      feedForwardCosine: MLXArray([Float(0.5), 0.5]),
      qkvWeight: MLXArray.zeros([2, 6]),
      attentionScale: MLXArray([Float(1)]),
      attentionBias: MLXArray.zeros([1, 4, 4]),
      attentionProjectionWeight: MLXArray.zeros([2, 2]),
      attentionCosine: MLXArray([Float(0.25), 0.25]),
      headCount: 1,
      windowSize: 2
    )
    eval(output)

    XCTAssertEqual(output.shape, input.shape)
    let expected = (0..<32).map { Float($0 + 1) * 0.125 }
    for (actual, expected) in zip(output.asArray(Float.self), expected) {
      XCTAssertEqual(actual, expected, accuracy: 0.000_001)
    }
  }

  func testUpsampleWindowOwnsMergedLatentAndSkipResidual() throws {
    let prefix = "block66.layer0"
    var projection = [Float](repeating: 0, count: 64 * 32)
    projection[0] = 1
    let arrays: [String: MLXArray] = [
      "\(prefix).weight0": MLXArray(projection, [64, 32]),
      "\(prefix).weight1": .zeros([32, 128]),
      "\(prefix).weight2": .zeros([128, 32]),
      "\(prefix).ffn_cos_skip": .ones([32]),
      "\(prefix).sin": .zeros([32]),
      "\(prefix).qkv_weight": .zeros([32, 96]),
      "\(prefix).attn_scale": .ones([1]),
      "\(prefix).attn_bias": .zeros([1, 64, 64]),
      "\(prefix).projection_weight": .zeros([32, 32]),
      "\(prefix).attn_cos_skip": .ones([32]),
    ]
    let block = try NeuralRenderingUpsampleBlock(
      weights: ValidatedWeights(arrays: arrays),
      blockIndex: 66,
      channels: 32,
      hiddenChannels: 128,
      headCount: 1
    )
    var latent = [Float](repeating: 0, count: 64)
    latent[0] = 1

    let output = block(
      MLXArray(latent, [1, 1, 1, 64]),
      skip: .zeros([1, 2, 2, 32])
    )
    eval(output)

    var expected = [Float](repeating: 0, count: 4 * 32)
    for pixel in 0..<4 {
      expected[pixel * 32] = 1
    }
    XCTAssertEqual(output.asArray(Float.self), expected)
  }

  func testShiftedWindowBlockPadsAndCropsToInputShape() {
    let input = MLXArray((0..<32).map { Float($0 + 1) }, [1, 4, 4, 2])
    let output = NeuralRenderingTransformerOperations.windowBlock(
      input,
      expansionWeight: MLXArray.zeros([2, 4]),
      feedForwardProjectionWeight: MLXArray.zeros([4, 2]),
      feedForwardCosine: MLXArray([Float(0.5), 0.5]),
      qkvWeight: MLXArray.zeros([2, 6]),
      attentionScale: MLXArray([Float(1)]),
      attentionBias: MLXArray.zeros([1, 4, 4]),
      attentionProjectionWeight: MLXArray.zeros([2, 2]),
      attentionCosine: MLXArray([Float(0.25), 0.25]),
      headCount: 1,
      windowSize: 2,
      windowOrigin: NeuralRenderingWindowOrigin(y: -1, x: -1)
    )
    eval(output)

    XCTAssertEqual(output.shape, input.shape)
    let expected = (0..<32).map { Float($0 + 1) * 0.125 }
    for (actual, expected) in zip(output.asArray(Float.self), expected) {
      XCTAssertEqual(actual, expected, accuracy: 0.000_001)
    }
  }

  func testShiftedWindowAttentionMatchesRecoveredPaddedGeometry() {
    let input = MLXArray((1...16).map(Float.init), [1, 4, 4, 1])
    let output = NeuralRenderingTransformerOperations.windowBlock(
      input,
      expansionWeight: .zeros([1, 1]),
      feedForwardProjectionWeight: .zeros([1, 1]),
      feedForwardCosine: MLXArray([Float(1)]),
      qkvWeight: MLXArray([Float](repeating: 1, count: 3), [1, 3]),
      attentionScale: MLXArray([Float(1)]),
      attentionBias: .zeros([1, 4, 4]),
      attentionProjectionWeight: MLXArray([Float(1)], [1, 1]),
      attentionCosine: MLXArray([Float(0)]),
      headCount: 1,
      windowSize: 2,
      windowOrigin: NeuralRenderingWindowOrigin(y: -1, x: -1)
    )
    eval(output)

    let expected: [Float] = [
      0.468_75, 1.875, 1.875, 1.875,
      5, 8, 8, 7.5,
      5, 8, 8, 7.5,
      6, 11, 11, 7.5,
    ]
    XCTAssertEqual(output.asArray(Float.self), expected)
  }

  func testBranchedFeedForwardAccumulatesKBeforeQuadraticActivation() {
    let inputValues = (1...32).map(Float.init) + [Float](repeating: 0, count: 32)
    var expansionValues = [Float](repeating: 0, count: 2 * 4 * 2 * 32 * 32)
    var branchProjectionValues = [Float](repeating: 0, count: 2 * 4 * 32 * 32)
    var outputProjectionValues = [Float](repeating: 0, count: 64 * 64)
    for index in 0..<32 {
      expansionValues[index * 32 + index] = 1
      branchProjectionValues[index * 32 + index] = 1
    }
    for index in 0..<64 {
      outputProjectionValues[index * 64 + index] = 1
    }

    let output = NeuralRenderingTransformerOperations.branchedFeedForward(
      MLXArray(inputValues, [1, 1, 1, 64]),
      expansionWeight: MLXArray(expansionValues, [2, 4, 2, 32, 32]),
      branchProjectionWeight: MLXArray(
        branchProjectionValues, [2, 4, 32, 32]
      ),
      outputProjectionWeight: MLXArray(outputProjectionValues, [64, 64])
    )
    eval(output)

    // The gated expansion and the per-head branch sum are published as E4M3
    // before their projections, matching the vendor block-5 FFN captures.
    let expectedActivation = NeuralRenderingTransformerOperations.e4m3RoundTrip(
      NeuralRenderingTransformerOperations.quadraticGateActivation(
        MLXArray(Array(inputValues.prefix(32)))
      )
    )
    eval(expectedActivation)
    let expected =
      expectedActivation.asArray(Float.self)
      + [Float](repeating: 0, count: 32)
    XCTAssertEqual(output.shape, [1, 1, 1, 64])
    for (actual, expected) in zip(output.asArray(Float.self), expected) {
      XCTAssertEqual(actual, expected, accuracy: 0.000_01)
    }
  }

  func testFusedBranchedFeedForwardMatchesLiteralOperation() {
    let input = MLXArray(
      (0..<(8 * 64)).map { sin(Float($0) * 0.01) },
      [1, 2, 4, 64]
    )
    let expansion = MLXArray(
      (0..<(2 * 4 * 2 * 32 * 32)).map { cos(Float($0) * 0.003) * 0.05 },
      [2, 4, 2, 32, 32]
    )
    let branchProjection = MLXArray(
      (0..<(2 * 4 * 32 * 32)).map { sin(Float($0) * 0.005) * 0.05 },
      [2, 4, 32, 32]
    )
    let outputProjection = MLXArray.identity(64)

    let expected = NeuralRenderingTransformerOperations.branchedFeedForward(
      input,
      expansionWeight: expansion,
      branchProjectionWeight: branchProjection,
      outputProjectionWeight: outputProjection
    )
    let actual = NeuralRenderingTransformerOperations.fusedBranchedFeedForward(
      input,
      expansionWeight: expansion,
      branchProjectionWeight: branchProjection,
      outputProjectionWeight: outputProjection
    )
    eval(expected, actual)
    let errors = zip(actual.asArray(Float.self), expected.asArray(Float.self))
      .map { abs($0.0 - $0.1) }

    XCTAssertEqual(actual.shape, expected.shape)
    XCTAssertLessThanOrEqual(errors.max() ?? .infinity, 0.000_1)
  }

  func testDownsampleAveragesTwoByTwoThenProjectsChannels() {
    let input = MLXArray((0..<16).map(Float.init), [1, 4, 4, 1])
    let weight = MLXArray([Float(2), -1], [1, 2])

    let output = NeuralRenderingTransformerOperations.downsample(
      input,
      weight: weight
    )
    eval(output)

    XCTAssertEqual(output.shape, [1, 2, 2, 2])
    XCTAssertEqual(
      output.asArray(Float.self),
      [5, -2.5, 9, -4.5, 21, -10.5, 25, -12.5]
    )
  }

  func testSpatialEndPaddingMatchesSplitTransitionExtent() {
    let input = MLXArray.ones([1, 6, 6, 1])

    let output = NeuralRenderingTransformerOperations.padSpatialEnd(
      input,
      multiple: 8
    )
    eval(output)

    XCTAssertEqual(output.shape, [1, 8, 8, 1])
    XCTAssertEqual(sum(output).item(Float.self), 36, accuracy: 0.000_001)
  }

  func testSplitWindowBlockAppliesGroupMLPWithQuadraticGate() {
    // k / 8 for k < 16 is exactly representable in E4M3.
    let values = (0..<256).map { Float($0 % 16) / 8 }
    let input = MLXArray(values, [1, 2, 2, 64])
    var identity = [Float](repeating: 0, count: 64 * 64)
    var expand = [Float](repeating: 0, count: 64 * 256)
    var project = [Float](repeating: 0, count: 256 * 64)
    for index in 0..<64 {
      identity[index * 64 + index] = 1
      expand[index * 256 + index] = 1
      project[index * 64 + index] = 1
    }

    let output = NeuralRenderingTransformerOperations.splitWindowBlock(
      input,
      firstProjectionWeight: MLXArray(identity, [64, 64]),
      expandWeight: MLXArray(expand, [1, 64, 256]),
      projectWeight: MLXArray(project, [1, 256, 64]),
      feedForwardProjectionWeight: MLXArray(identity, [64, 64]),
      feedForwardCosine: .zeros([64]),
      qkvWeight: .zeros([64, 192]),
      attentionScale: MLXArray([Float(1)]),
      attentionBias: .zeros([1, 4, 4]),
      attentionProjectionWeight: .zeros([64, 64]),
      attentionCosine: MLXArray.ones([64]),
      headCount: 1,
      windowSize: 2
    )
    // The group MLP output is published as E4M3 before weight3 (vendor
    // ffwd_512_chained captures); an identity projection exposes it.
    let expected = NeuralRenderingTransformerOperations.e4m3RoundTrip(
      NeuralRenderingTransformerOperations.quadraticGateActivation(input)
    )
    eval(output, expected)

    XCTAssertEqual(output.shape, input.shape)
    for (actual, expected) in zip(output.asArray(Float.self), expected.asArray(Float.self)) {
      XCTAssertEqual(actual, expected, accuracy: 0.000_001)
    }
  }

  func testSplitGroupFeedForwardMatchesPerGroupLoop() {
    let input = MLXArray(
      (0..<(8 * 128)).map { sin(Float($0) * 0.01) },
      [1, 2, 4, 128]
    )
    let firstProjection = MLXArray(
      (0..<(128 * 128)).map { cos(Float($0) * 0.002) * 0.1 },
      [128, 128]
    )
    let expandWeight = MLXArray(
      (0..<(2 * 64 * 256)).map { cos(Float($0) * 0.003) * 0.05 },
      [2, 64, 256]
    )
    let projectWeight = MLXArray(
      (0..<(2 * 256 * 64)).map { sin(Float($0) * 0.005) * 0.05 },
      [2, 256, 64]
    )

    let actual = NeuralRenderingTransformerOperations.splitGroupFeedForward(
      input,
      firstProjectionWeight: firstProjection,
      expandWeight: expandWeight,
      projectWeight: projectWeight
    )
    let hidden = NeuralRenderingTransformerOperations.e4m3RoundTrip(
      matmul(input, firstProjection)
    )
    let groups = (0..<2).map { group in
      matmul(
        NeuralRenderingTransformerOperations.quadraticGateActivation(
          matmul(hidden[.ellipsis, group * 64..<(group + 1) * 64], expandWeight[group])
        ),
        projectWeight[group]
      )
    }
    let expected = NeuralRenderingTransformerOperations.e4m3RoundTrip(
      concatenated(groups, axis: -1)
    )
    eval(actual, expected)
    let errors = zip(actual.asArray(Float.self), expected.asArray(Float.self))
      .map { abs($0.0 - $0.1) }

    XCTAssertEqual(actual.shape, expected.shape)
    XCTAssertLessThanOrEqual(errors.max() ?? .infinity, 0.000_1)
  }

  func testGlobalBlockAttendsAcrossAllFlattenedTokens() {
    let input = MLXArray((0..<32).map { Float($0 + 1) }, [1, 4, 4, 2])

    let output = NeuralRenderingTransformerOperations.globalBlock(
      input,
      expansionWeight: .zeros([2, 4]),
      feedForwardProjectionWeight: .zeros([4, 2]),
      feedForwardCosine: MLXArray([Float(0.5), 0.5]),
      qkvWeight: .zeros([2, 6]),
      attentionScale: MLXArray([Float(1)]),
      attentionProjectionWeight: .zeros([2, 2]),
      attentionCosine: MLXArray([Float(0.25), 0.25]),
      headCount: 1
    )
    eval(output)

    let expected = (0..<32).map { Float($0 + 1) * 0.125 }
    XCTAssertEqual(output.shape, input.shape)
    XCTAssertEqual(output.asArray(Float.self), expected)
  }

  func testGlobalBlockScalesCosineLogitsBySquareRootHeadWidth() {
    let input = MLXArray([Float(1), 0, 0, 1], [1, 1, 2, 2])
    let qkvWeight = MLXArray(
      [
        Float(1), 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1,
      ],
      [2, 6]
    )

    let output = NeuralRenderingTransformerOperations.globalBlock(
      input,
      expansionWeight: .zeros([2, 2]),
      feedForwardProjectionWeight: .zeros([2, 2]),
      feedForwardCosine: .ones([2]),
      qkvWeight: qkvWeight,
      attentionScale: MLXArray([Float(1)]),
      attentionProjectionWeight: MLXArray([Float(1), 0, 0, 1], [2, 2]),
      attentionCosine: .zeros([2]),
      headCount: 1
    )
    eval(output)

    let expected: [Float] = [0.8125, 0.203_125, 0.203_125, 0.8125]
    for (actual, expected) in zip(output.asArray(Float.self), expected) {
      XCTAssertEqual(actual, expected, accuracy: 0.000_001)
    }
  }

  func testGlobalBlockClampsLogitsSymmetrically() {
    // Anti-parallel tokens: cross logit -scale·√2, self logit +scale·√2; the
    // vit_1d kernels clamp both sides to ±3.
    let input = MLXArray([Float(1), 0, -1, 0], [1, 1, 2, 2])
    let qkvWeight = MLXArray(
      [
        Float(1), 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1,
      ],
      [2, 6]
    )
    func run(scale: Float, logitCap: Float?) -> [Float] {
      let output = NeuralRenderingTransformerOperations.globalBlock(
        input,
        expansionWeight: .zeros([2, 2]),
        feedForwardProjectionWeight: .zeros([2, 2]),
        feedForwardCosine: .ones([2]),
        qkvWeight: qkvWeight,
        attentionScale: MLXArray([scale]),
        attentionProjectionWeight: MLXArray([Float(1), 0, 0, 1], [2, 2]),
        attentionCosine: .zeros([2]),
        headCount: 1,
        logitCap: logitCap
      )
      eval(output)
      return output.asArray(Float.self)
    }
    let capped = run(scale: 10, logitCap: 3)
    let reference = run(scale: 3 / Float(2).squareRoot(), logitCap: nil)
    XCTAssertEqual(capped, reference)
  }

  func testGlobalBlockAppliesExperimentalLogitCapOnlyWhenRequested() {
    let input = MLXArray([Float(1), 0, 0, 1], [1, 1, 2, 2])
    let qkvWeight = MLXArray(
      [
        Float(1), 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1,
      ],
      [2, 6]
    )
    func run(scale: Float, logitCap: Float?) -> [Float] {
      let output = NeuralRenderingTransformerOperations.globalBlock(
        input,
        expansionWeight: .zeros([2, 2]),
        feedForwardProjectionWeight: .zeros([2, 2]),
        feedForwardCosine: .ones([2]),
        qkvWeight: qkvWeight,
        attentionScale: MLXArray([scale]),
        attentionProjectionWeight: MLXArray([Float(1), 0, 0, 1], [2, 2]),
        attentionCosine: .zeros([2]),
        headCount: 1,
        logitCap: logitCap
      )
      eval(output)
      return output.asArray(Float.self)
    }

    // Scale 10 puts the self-logit at 10·√2 (published as 14); the opt-in cap
    // clamps it to 3, so the block matches an uncapped block whose logits are 3.
    let capped = run(
      scale: 10,
      logitCap: NeuralRenderingTransformerOperations.experimentalGlobalAttentionLogitCap
    )
    let uncapped = run(scale: 10, logitCap: nil)
    let reference = run(scale: 3 / Float(2).squareRoot(), logitCap: nil)

    XCTAssertEqual(
      NeuralRenderingTransformerOperations.experimentalGlobalAttentionLogitCap, 3
    )
    XCTAssertEqual(capped, reference)
    XCTAssertNotEqual(capped, uncapped)
  }

  func testLearnedUpsampleInterpolatesMidpointsAndClampsEdges() {
    let input = MLXArray([Float(0), 2, 4, 6], [1, 2, 2, 1])

    let output = NeuralRenderingTransformerOperations.learnedUpsample2(
      input,
      interpolation: MLXArray([Float(0.5)])
    )
    eval(output)

    XCTAssertEqual(output.shape, [1, 4, 4, 1])
    XCTAssertEqual(
      output.asArray(Float.self),
      [
        0, 1, 2, 2,
        2, 3, 4, 4,
        4, 5, 6, 6,
        4, 5, 6, 6,
      ])
  }

  func testDecoderInputDoublesCropsAndAddsSineScaledSplitSkip() {
    let input = MLXArray([Float(0), 1, 2, 3], [1, 2, 2, 1])
    let skip = MLXArray([Float](repeating: 10, count: 9), [1, 3, 3, 1])

    let output = NeuralRenderingTransformerOperations.decoderInputMerge(
      input,
      skip: skip,
      skipSine: MLXArray([Float(0.25)])
    )
    eval(output)

    XCTAssertEqual(output.shape, skip.shape)
    XCTAssertEqual(
      output.asArray(Float.self),
      [2.5, 2.5, 3.5, 2.5, 2.5, 3.5, 4.5, 4.5, 5.5]
    )
  }

  func testPostMergeAppliesPerChannelSineCosineRotation() {
    let decoder = MLXArray([Float(2), 4], [1, 1, 1, 2])
    let skip = MLXArray([Float(10), 20], [1, 1, 1, 2])

    let output = NeuralRenderingTransformerOperations.postMerge(
      decoder,
      skip: skip,
      sine: MLXArray([Float(0.6), 0.8]),
      cosine: MLXArray([Float(0.8), 0.6])
    )
    eval(output)

    XCTAssertEqual(output.asArray(Float.self), [9.2, 15.2])
  }

  func testExternalRecoveredPostBlockExecutesWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS to run the local recovered-weight probe")
    }
    let arrays = try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)
    let post = try NeuralRenderingPostBlock(
      weights: ValidatedWeights(arrays: arrays)
    )
    let decoder = MLXArray(
      (0..<(128 * 128 * 32)).map { sin(Float($0) * 0.001) },
      [1, 128, 128, 32]
    )
    let skip = MLXArray(
      (0..<(128 * 128 * 32)).map { cos(Float($0) * 0.001) },
      [1, 128, 128, 32]
    )

    let output = Device.withDefaultDevice(.gpu) { post(decoder, skip: skip) }
    eval(output)

    XCTAssertEqual(output.shape, [1, 128, 128, 4])
    XCTAssertTrue(output.asArray(Float.self).allSatisfy(\.isFinite))
  }

  func testExternalRecoveredDecoderInputExecutesWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS to run the local recovered-weight probe")
    }
    let arrays = try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)
    let bridge = try NeuralRenderingDecoderInput(
      weights: ValidatedWeights(arrays: arrays)
    )
    let input = MLXArray(
      (0..<(4 * 4 * 1024)).map { cos(Float($0) * 0.002) },
      [1, 4, 4, 1024]
    )

    let skip = MLXArray(
      (0..<(6 * 6 * 512)).map { sin(Float($0) * 0.003) },
      [1, 6, 6, 512]
    )

    let output = Device.withDefaultDevice(.gpu) { bridge(input, skip: skip) }
    eval(output)

    XCTAssertEqual(output.shape, skip.shape)
    XCTAssertTrue(output.asArray(Float.self).allSatisfy(\.isFinite))
  }

  func testExternalRecoveredFirstDecoderStageExecutesWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS to run the local recovered-weight probe")
    }
    let arrays = try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)
    let stage = try NeuralRenderingFirstDecoderStage(
      weights: ValidatedWeights(arrays: arrays)
    )
    let input = MLXArray(
      (0..<(8 * 8 * 512)).map { sin(Float($0) * 0.002) },
      [1, 8, 8, 512]
    )
    let skip = MLXArray(
      (0..<(16 * 16 * 256)).map { cos(Float($0) * 0.003) },
      [1, 16, 16, 256]
    )

    let output = Device.withDefaultDevice(.gpu) { stage(input, skip: skip) }
    eval(output)

    XCTAssertEqual(output.shape, skip.shape)
    XCTAssertTrue(output.asArray(Float.self).allSatisfy(\.isFinite))
  }

  func testExternalRecoveredDecoderExecutesThroughBlockSixtyNine() throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS to run the local recovered-weight probe")
    }
    let arrays = try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)
    let decoder = try NeuralRenderingDecoder(
      weights: ValidatedWeights(arrays: arrays)
    )
    let input = MLXArray(
      (0..<(8 * 8 * 512)).map { sin(Float($0) * 0.002) },
      [1, 8, 8, 512]
    )
    let skips = [
      MLXArray(
        (0..<(128 * 128 * 32)).map { cos(Float($0) * 0.001) },
        [1, 128, 128, 32]),
      MLXArray(
        (0..<(64 * 64 * 64)).map { sin(Float($0) * 0.002) },
        [1, 64, 64, 64]),
      MLXArray(
        (0..<(32 * 32 * 128)).map { cos(Float($0) * 0.003) },
        [1, 32, 32, 128]),
      MLXArray(
        (0..<(16 * 16 * 256)).map { sin(Float($0) * 0.004) },
        [1, 16, 16, 256]),
    ]

    let output = Device.withDefaultDevice(.gpu) { decoder(input, skips: skips) }
    eval(output)

    XCTAssertEqual(output.shape, [1, 128, 128, 32])
    XCTAssertTrue(output.asArray(Float.self).allSatisfy(\.isFinite))
  }

  func testExternalRecoveredSplitBlockExecutesWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS to run the local recovered-weight probe")
    }
    let arrays = try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)
    let block = try NeuralRenderingSplitWindowBlock(
      weights: ValidatedWeights(arrays: arrays),
      blockIndex: 23
    )
    let input = MLXArray(
      (0..<(8 * 8 * 512)).map { sin(Float($0) * 0.003) },
      [1, 8, 8, 512]
    )

    let output = Device.withDefaultDevice(.gpu) { block(input) }
    eval(output)

    XCTAssertEqual(output.shape, input.shape)
    XCTAssertTrue(output.asArray(Float.self).allSatisfy(\.isFinite))
  }

  func testExternalRecoveredSplitEncoderStageExecutesWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS to run the local recovered-weight probe")
    }
    let arrays = try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)
    let stage = try NeuralRenderingSplitEncoderStage(
      weights: ValidatedWeights(arrays: arrays)
    )
    let input = MLXArray(
      (0..<(8 * 8 * 512)).map { cos(Float($0) * 0.002) },
      [1, 8, 8, 512]
    )

    let output = Device.withDefaultDevice(.gpu) { stage(input) }
    eval(output.skip, output.latent)

    XCTAssertEqual(output.skip.shape, input.shape)
    XCTAssertEqual(output.latent.shape, [1, 4, 4, 1024])
    XCTAssertTrue(output.skip.asArray(Float.self).allSatisfy(\.isFinite))
    XCTAssertTrue(output.latent.asArray(Float.self).allSatisfy(\.isFinite))
  }

  func testExternalRecoveredGlobalStageExecutesWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS to run the local recovered-weight probe")
    }
    let arrays = try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)
    let stage = try NeuralRenderingGlobalStage(
      weights: ValidatedWeights(arrays: arrays)
    )
    let input = MLXArray(
      (0..<(12 * 12 * 1024)).map { sin(Float($0) * 0.002) },
      [1, 12, 12, 1024]
    )

    let output = Device.withDefaultDevice(.gpu) { stage(input) }
    eval(output)

    XCTAssertEqual(output.shape, input.shape)
    XCTAssertTrue(output.asArray(Float.self).allSatisfy(\.isFinite))
  }

  func testExternalFacePreStageDistributionWhenConfigured() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let weightsPath = environment["MLXDLSS_LOGICAL_WEIGHTS"],
      let colorPath = environment["MLXDLSS_FACE_COLOR"],
      let captureRoot = environment["MLXDLSS_FACE_PRE_CAPTURE_ROOT"]
    else {
      throw XCTSkip(
        "set MLXDLSS_LOGICAL_WEIGHTS, MLXDLSS_FACE_COLOR, and MLXDLSS_FACE_PRE_CAPTURE_ROOT"
      )
    }
    let arrays = try loadArrays(
      url: URL(fileURLWithPath: weightsPath),
      stream: .cpu
    )
    let pre = try NeuralRenderingPreBlock(
      weights: ValidatedWeights(arrays: arrays),
      compileBlocks: true
    )
    let colorData = try Data(contentsOf: URL(fileURLWithPath: colorPath))
    let color = try HostTensor(
      descriptor: TensorDescriptor(
        name: "color",
        shape: [1, 256, 256, 3],
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: colorData
    )
    let geometry = try NeuralRenderingNetworkGeometry.vendorAligned(
      outputWidth: 256,
      outputHeight: 256
    )

    func decodeE4M3(_ bits: UInt8) -> Float {
      let exponent = Int((bits >> 3) & 0x0f)
      let mantissa = Float(bits & 0x07)
      let magnitude =
        exponent == 0
        ? mantissa / 512
        : (1 + mantissa / 8) * pow(2, Float(exponent - 7))
      return (bits & 0x80) == 0 ? magnitude : -magnitude
    }

    var local: [String: [Float]] = [:]
    var vendor: [String: [Float]] = [:]
    for (name, tone, structure) in [
      ("standard", Float(1), Float(1)),
      ("neutral", Float(0), Float(0)),
    ] {
      let features = try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(
        from: color,
        geometry: geometry,
        localToneStrength: tone,
        localStructureStrength: structure
      )
      let output = Device.withDefaultDevice(.gpu) {
        let projected = pre.project(
          MLXArray(features.bytes, features.descriptor.shape, dtype: .float32)
        )
        return pre.transform(
          NeuralRenderingTransformerOperations.averagePool2(projected)
        )
      }
      eval(output)
      XCTAssertEqual(output.shape, [1, 160, 160, 32])
      local[name] = output.asArray(Float.self)

      let capture = try Data(
        contentsOf: URL(fileURLWithPath: captureRoot)
          .appendingPathComponent("pre-\(name).bin")
      )
      XCTAssertGreaterThanOrEqual(capture.count, 0x3E8000)
      vendor[name] = capture[0x320000..<0x3E8000].map(decodeE4M3)
      XCTAssertEqual(vendor[name]!.count, 160 * 160 * 32)

      let localSorted = local[name]!.sorted()
      let vendorSorted = vendor[name]!.sorted()
      let errors = zip(localSorted, vendorSorted).map { abs($0.0 - $0.1) }
      print(
        "face-pre-stage profile=\(name) sortedMAE="
          + "\(errors.reduce(0, +) / Float(errors.count)) "
          + "max=\(errors.max() ?? .infinity)"
      )
    }

    let localDelta = zip(local["standard"]!, local["neutral"]!)
      .map { abs($0.0 - $0.1) }
    let vendorDelta = zip(vendor["standard"]!, vendor["neutral"]!)
      .map { abs($0.0 - $0.1) }
    let localDeltaMean = localDelta.reduce(0, +) / Float(localDelta.count)
    let vendorDeltaMean = vendorDelta.reduce(0, +) / Float(vendorDelta.count)
    print(
      "face-pre-control-delta localMAE="
        + "\(localDeltaMean) vendorMAE=\(vendorDeltaMean)"
    )
    XCTAssertTrue(localDelta.allSatisfy(\.isFinite))
    XCTAssertTrue(vendorDelta.allSatisfy(\.isFinite))
    XCTAssertEqual(localDeltaMean, vendorDeltaMean, accuracy: 0.001)
  }

  func testExternalWindowStagesAgainstVendorInputsWhenConfigured() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let weightsPath = environment["MLXDLSS_LOGICAL_WEIGHTS"],
      let stageRoot = environment["MLXDLSS_WINDOW_STAGE_ROOT"]
    else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS and MLXDLSS_WINDOW_STAGE_ROOT")
    }
    let arrays = try loadArrays(
      url: URL(fileURLWithPath: weightsPath),
      stream: .cpu
    )
    let weights = ValidatedWeights(arrays: arrays)
    let root = URL(fileURLWithPath: stageRoot, isDirectory: true)

    func values(_ index: Int) throws -> [Float] {
      try Data(contentsOf: root.appendingPathComponent("block\(index).f32"))
        .withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    for index in 1...4 {
      let inputValues = try values(index - 1)
      let expected = try values(index)
      let block = try NeuralRenderingWindowBlock(
        weights: weights,
        blockIndex: index,
        channels: 32,
        hiddenChannels: 128,
        headCount: 1,
        compileBlock: true
      )
      let output = Device.withDefaultDevice(.gpu) {
        block(MLXArray(inputValues, [1, 160, 160, 32]).asType(.float16))
      }
      eval(output)
      let actual = output.asArray(Float.self)
      let errors = zip(actual, expected).map { abs($0.0 - $0.1) }
      let actualMean = actual.reduce(0, +) / Float(actual.count)
      let expectedMean = expected.reduce(0, +) / Float(expected.count)
      let covariance = zip(actual, expected).reduce(Float.zero) {
        $0 + ($1.0 - actualMean) * ($1.1 - expectedMean)
      }
      let actualVariance = actual.reduce(Float.zero) {
        $0 + ($1 - actualMean) * ($1 - actualMean)
      }
      let expectedVariance = expected.reduce(Float.zero) {
        $0 + ($1 - expectedMean) * ($1 - expectedMean)
      }
      let correlation = covariance / sqrt(actualVariance * expectedVariance)
      print(
        "swift-vendor-stage block=\(index) mae="
          + "\(errors.reduce(0, +) / Float(errors.count)) corr=\(correlation)"
      )
      XCTAssertTrue(actual.allSatisfy(\.isFinite))
      XCTAssertTrue(correlation.isFinite)
    }
  }

  func testExternalRecoveredTrunkExecutesBlocksZeroThroughThirtyEight() throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS to run the local recovered-weight probe")
    }
    let arrays = try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)
    let trunk = try NeuralRenderingTrunk(
      weights: ValidatedWeights(arrays: arrays)
    )
    let input = MLXArray(
      (0..<(128 * 128 * 16)).map { sin(Float($0) * 0.001) },
      [1, 128, 128, 16]
    )

    let output = Device.withDefaultDevice(.gpu) { trunk(input) }
    eval(output.latent)

    XCTAssertEqual(output.fullResolutionSkip.shape, [1, 128, 128, 32])
    XCTAssertEqual(
      output.skips.map(\.shape),
      [
        [1, 64, 64, 32],
        [1, 32, 32, 64],
        [1, 16, 16, 128],
        [1, 8, 8, 256],
      ])
    XCTAssertEqual(output.latent.shape, [1, 4, 4, 1024])
    XCTAssertTrue(output.latent.asArray(Float.self).allSatisfy(\.isFinite))
  }

  func testExternalRecoveredModelExecutesAllSeventyOneBlocks() throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS to run the local recovered-weight probe")
    }
    let arrays = try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)
    let model = try NeuralRenderingTransformerModel(
      weights: ValidatedWeights(arrays: arrays)
    )
    let input = MLXArray(
      (0..<(128 * 128 * 16)).map { sin(Float($0) * 0.001) },
      [1, 128, 128, 16]
    )

    let output = Device.withDefaultDevice(.gpu) { model(input) }
    eval(output)
    let values = output.asArray(Float.self)

    XCTAssertEqual(output.shape, [1, 128, 128, 4])
    XCTAssertTrue(values.allSatisfy(\.isFinite))
    XCTAssertGreaterThan(values.map(abs).reduce(0, +), 0)
  }

  func testExternalRecoveredPackageRunsThroughPublicBackend() async throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_NEURAL_RENDERING_PACKAGE"] else {
      throw XCTSkip("set MLXDLSS_NEURAL_RENDERING_PACKAGE to run the package integration probe")
    }
    let values = (0..<(128 * 128 * 16)).map { sin(Float($0) * 0.001) }
    let descriptor = try TensorDescriptor(
      name: "color",
      shape: [1, 128, 128, 16],
      dataType: .float32,
      layout: .nhwc
    )
    let tensor = try HostTensor(
      descriptor: descriptor,
      bytes: values.withUnsafeBytes { Data($0) }
    )
    let backend = try MLXNeuralRenderer(
      packageURL: URL(fileURLWithPath: path),
      executionMode: .eager,
      computePrecision: .float32
    )

    let result = try await backend.render(
      NeuralRenderRequest(sequenceID: 1, inputs: [tensor])
    )

    let output = try XCTUnwrap(result.output(named: "color"))
    XCTAssertEqual(output.descriptor.shape, [1, 128, 128, 4])
    let outputValues = output.bytes.withUnsafeBytes { bytes in
      Array(bytes.bindMemory(to: Float.self))
    }
    XCTAssertTrue(outputValues.allSatisfy(\.isFinite))
    XCTAssertGreaterThan(outputValues.map(abs).reduce(0, +), 0)
  }

  func testExternalRecoveredPackageRejectsUnsafeWholeGraphCompilation() throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_NEURAL_RENDERING_PACKAGE"] else {
      throw XCTSkip("set MLXDLSS_NEURAL_RENDERING_PACKAGE to run the package integration probe")
    }

    // The package may carry the current identifier or the pre-rename one; both load.
    let architecture = try ModelPackageLoader.load(url: URL(fileURLWithPath: path)).manifest.architecture
    XCTAssertTrue(MLXNeuralRenderer.transformerArchitectures.contains(architecture), architecture)
    XCTAssertThrowsError(
      try MLXNeuralRenderer(
        packageURL: URL(fileURLWithPath: path),
        executionMode: .compiled
      )
    ) { error in
      XCTAssertEqual(
        error as? MLXBackendError,
        .compiledExecutionUnsupported(architecture: architecture)
      )
    }
  }

  func testExternalRecoveredPackageRunsBlockCompiledHeadWithinTolerance() async throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_NEURAL_RENDERING_PACKAGE"] else {
      throw XCTSkip("set MLXDLSS_NEURAL_RENDERING_PACKAGE to run the package integration probe")
    }
    let values = (0..<(128 * 128 * 16)).map { sin(Float($0) * 0.001) }
    let tensor = try HostTensor(
      descriptor: TensorDescriptor(
        name: "color",
        shape: [1, 128, 128, 16],
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: values.withUnsafeBytes { Data($0) }
    )
    let packageURL = URL(fileURLWithPath: path)
    let eager = try MLXNeuralRenderer(packageURL: packageURL)
    let blockCompiled = try MLXNeuralRenderer(
      packageURL: packageURL,
      executionMode: .blockCompiled
    )
    let request = try NeuralRenderRequest(sequenceID: 1, inputs: [tensor])
    let eagerResult = try await eager.render(request)
    let compiledResult = try await blockCompiled.render(request)
    let eagerValues: [Float] = eagerResult.output(named: "color")!.bytes.withUnsafeBytes {
      Array($0.bindMemory(to: Float.self))
    }
    let compiledValues: [Float] = compiledResult.output(named: "color")!.bytes.withUnsafeBytes {
      Array($0.bindMemory(to: Float.self))
    }
    let errors = zip(compiledValues, eagerValues).map { abs($0.0 - $0.1) }

    let maximum = errors.max() ?? .infinity
    let mean = errors.reduce(0, +) / Float(errors.count)
    XCTAssertTrue(compiledValues.allSatisfy(\.isFinite))
    XCTAssertLessThanOrEqual(maximum, 0.000_1)
    XCTAssertLessThanOrEqual(mean, 0.000_02)
  }

  func testExternalRecoveredCompiledWindowSequenceMatchesEagerBlocks() throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS to run the local recovered-weight probe")
    }
    let arrays = try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)
    let weights = ValidatedWeights(arrays: arrays)
    let eagerBlocks = try (1...3).map {
      try NeuralRenderingWindowBlock(
        weights: weights,
        blockIndex: $0,
        channels: 32,
        hiddenChannels: 128,
        headCount: 1
      )
    }
    let compiledSequence = try NeuralRenderingWindowSequence(
      weights: weights,
      blockIndices: 1...3,
      channels: 32,
      hiddenChannels: 128,
      headCount: 1,
      compileSequence: true
    )
    let input = MLXArray(
      (0..<(128 * 128 * 32)).map { sin(Float($0) * 0.001) },
      [1, 128, 128, 32]
    )
    let reference = eagerBlocks.reduce(input) { value, block in block(value) }
    let sequence: MLXArray = compiledSequence(input)
    eval(reference, sequence)
    let sequenceValues: [Float] = sequence.asArray(Float.self)
    let referenceValues: [Float] = reference.asArray(Float.self)
    let maximumError = zip(
      sequenceValues,
      referenceValues
    ).reduce(Float.zero) {
      max($0, abs($1.0 - $1.1))
    }

    XCTAssertEqual(sequence.shape, reference.shape)
    XCTAssertLessThanOrEqual(maximumError, 0.000_01)
  }

  func testExternalRecoveredQuantizedGlobalFFNStaysWithinExperimentalTolerance() throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS to run the local recovered-weight probe")
    }
    let arrays = try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)
    let weights = ValidatedWeights(arrays: arrays)
    let referenceModel = try NeuralRenderingTransformerModel(
      weights: weights,
      compileBlocks: true
    )
    let quantizedModel = try NeuralRenderingTransformerModel(
      weights: weights,
      compileBlocks: true,
      quantizeGlobalFFN: true
    )
    let input = MLXArray(
      (0..<(128 * 128 * 16)).map { sin(Float($0) * 0.001) },
      [1, 128, 128, 16]
    )
    let reference = referenceModel(input)
    let candidate = quantizedModel(input)
    eval(reference, candidate)
    let candidateValues: [Float] = candidate.asArray(Float.self)
    let referenceValues: [Float] = reference.asArray(Float.self)
    let errors = zip(candidateValues, referenceValues).map { abs($0.0 - $0.1) }

    XCTAssertTrue(candidateValues.allSatisfy(\.isFinite))
    XCTAssertLessThanOrEqual(errors.max() ?? .infinity, 0.05)
    XCTAssertLessThanOrEqual(errors.reduce(0, +) / Float(errors.count), 0.005)
  }

  func testExternalRecoveredPackageRunsInt8FastHeadWithinTolerance() async throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_NEURAL_RENDERING_PACKAGE"] else {
      throw XCTSkip("set MLXDLSS_NEURAL_RENDERING_PACKAGE to run the package integration probe")
    }
    let inputValues = (0..<(128 * 128 * 16)).map { sin(Float($0) * 0.001) }
    let input = try HostTensor(
      descriptor: TensorDescriptor(
        name: "color",
        shape: [1, 128, 128, 16],
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: inputValues.withUnsafeBytes { Data($0) }
    )
    let packageURL = URL(fileURLWithPath: path)
    let reference = try MLXNeuralRenderer(
      packageURL: packageURL,
      executionMode: .blockCompiled
    )
    let int8 = try MLXNeuralRenderer(
      packageURL: packageURL,
      executionMode: .int8Fast
    )
    let request = try NeuralRenderRequest(sequenceID: 1, inputs: [input])
    let referenceResult = try await reference.render(request)
    let int8Result = try await int8.render(request)
    let referenceValues: [Float] = referenceResult.output(named: "color")!.bytes
      .withUnsafeBytes { bytes -> [Float] in
        Array(bytes.bindMemory(to: Float.self))
      }
    let candidateValues: [Float] = int8Result.output(named: "color")!.bytes
      .withUnsafeBytes { bytes -> [Float] in
        Array(bytes.bindMemory(to: Float.self))
      }
    let errors = zip(candidateValues, referenceValues).map { abs($0.0 - $0.1) }

    XCTAssertTrue(candidateValues.allSatisfy(\.isFinite))
    XCTAssertLessThanOrEqual(errors.max() ?? .infinity, 0.05)
    XCTAssertLessThanOrEqual(errors.reduce(0, +) / Float(errors.count), 0.005)
  }

  func testExternalRecoveredFP16StageBoundariesStayFinite() throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS to run the local recovered-weight probe")
    }
    let loaded = try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)
    let weights = ValidatedWeights(arrays: loaded).cast(to: .float16)
    let input = MLXArray(
      (0..<(128 * 128 * 16)).map { sin(Float($0) * 0.001) },
      [1, 128, 128, 16]
    ).asType(.float16)
    let pre = try NeuralRenderingPreBlock(weights: weights)
    var current = pre.project(input)
    guard checkFinite(current, stage: "input-adapter") else { return }
    current = NeuralRenderingTransformerOperations.averagePool2(current)
    guard checkFinite(current, stage: "block0-pool") else { return }
    current = pre.transform(current)
    guard checkFinite(current, stage: "block0") else { return }
    for index in 1...3 {
      current = try NeuralRenderingWindowBlock(
        weights: weights,
        blockIndex: index,
        channels: 32,
        hiddenChannels: 128,
        headCount: 1
      )(current)
      guard checkFinite(current, stage: "block\(index)") else { return }
    }
    current = try NeuralRenderingDownsampleBlock(
      weights: weights,
      blockIndex: 4,
      channels: 32,
      hiddenChannels: 128,
      headCount: 1
    )(current)
    guard checkFinite(current, stage: "block4") else { return }
    for index in 5...7 {
      current = try NeuralRenderingWindowBlock(
        weights: weights,
        blockIndex: index,
        channels: 64,
        hiddenChannels: 224,
        headCount: 2
      )(current)
      guard checkFinite(current, stage: "block\(index)") else { return }
    }
    current = try NeuralRenderingDownsampleBlock(
      weights: weights,
      blockIndex: 8,
      channels: 64,
      hiddenChannels: 224,
      headCount: 2
    )(current)
    guard checkFinite(current, stage: "block8") else { return }
    for index in 9...13 {
      current = try NeuralRenderingWindowBlock(
        weights: weights,
        blockIndex: index,
        channels: 128,
        hiddenChannels: 384,
        headCount: 4
      )(current)
      guard checkFinite(current, stage: "block\(index)") else { return }
    }
    current = try NeuralRenderingDownsampleBlock(
      weights: weights,
      blockIndex: 14,
      channels: 128,
      hiddenChannels: 384,
      headCount: 4
    )(current)
    guard checkFinite(current, stage: "block14") else { return }
    for index in 15...21 {
      current = try NeuralRenderingWindowBlock(
        weights: weights,
        blockIndex: index,
        channels: 256,
        hiddenChannels: 704,
        headCount: 8
      )(current)
      guard checkFinite(current, stage: "block\(index)") else { return }
    }
    current = try NeuralRenderingDownsampleBlock(
      weights: weights,
      blockIndex: 22,
      channels: 256,
      hiddenChannels: 704,
      headCount: 8
    )(current)
    guard checkFinite(current, stage: "block22") else { return }

    let encoder = try NeuralRenderingEncoder(weights: weights)
    let split = try NeuralRenderingSplitEncoderStage(weights: weights)
    let global = try NeuralRenderingGlobalStage(weights: weights)
    let bridge = try NeuralRenderingDecoderInput(weights: weights)
    let decoder = try NeuralRenderingDecoder(weights: weights)
    let post = try NeuralRenderingPostBlock(weights: weights)
    let encoded = encoder(input)
    assertFinite(encoded.latent, stage: "encoder")
    let splitOutput = split(encoded.latent)
    assertFinite(splitOutput.skip, stage: "split-skip")
    assertFinite(splitOutput.latent, stage: "split-latent")
    let globalOutput = global(splitOutput.latent)
    assertFinite(globalOutput, stage: "global")
    let bridgeOutput = bridge(globalOutput, skip: splitOutput.skip)
    assertFinite(bridgeOutput, stage: "bridge")
    let decoderOutput = decoder(bridgeOutput, skips: encoded.skips)
    assertFinite(decoderOutput, stage: "decoder")
    let output = post(decoderOutput, skip: encoded.skips[0])
    assertFinite(output, stage: "post")
  }

  private func assertFinite(_ value: MLXArray, stage: String) {
    _ = checkFinite(value, stage: stage)
  }

  private func checkFinite(_ value: MLXArray, stage: String) -> Bool {
    eval(value)
    let values = value.asType(.float32).asArray(Float.self)
    let finite = values.allSatisfy(\.isFinite)
    XCTAssertTrue(finite, "non-finite values after \(stage)")
    return finite
  }

  func testWindowBlockBindsRecoveredWeightNames() throws {
    let prefix = "block1.layer0"
    let arrays: [String: MLXArray] = [
      "\(prefix).weight1": .zeros([2, 4]),
      "\(prefix).weight2": .zeros([4, 2]),
      "\(prefix).ffn_cos_skip": MLXArray([Float(0.5), 0.5]),
      "\(prefix).qkv_weight": .zeros([2, 6]),
      "\(prefix).attn_scale": MLXArray([Float(1)]),
      "\(prefix).attn_bias": .zeros([1, 64, 64]),
      "\(prefix).projection_weight": .zeros([2, 2]),
      "\(prefix).attn_cos_skip": MLXArray([Float(0.25), 0.25]),
    ]
    let block = try NeuralRenderingWindowBlock(
      weights: ValidatedWeights(arrays: arrays),
      blockIndex: 1,
      channels: 2,
      hiddenChannels: 4,
      headCount: 1
    )
    let input = MLXArray((0..<128).map { Float($0 + 1) }, [1, 8, 8, 2])

    let output = block(input)
    eval(output)

    let expected = NeuralRenderingTransformerOperations.e4m3RoundTrip(
      MLXArray((0..<128).map { Float($0 + 1) * 0.125 })
    )
    eval(expected)
    XCTAssertEqual(output.asArray(Float.self), expected.asArray(Float.self))
  }

  func testExternalRecoveredBlockOneExecutesWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS to run the local recovered-weight probe")
    }
    let arrays = try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)
    let block = try NeuralRenderingWindowBlock(
      weights: ValidatedWeights(arrays: arrays),
      blockIndex: 1,
      channels: 32,
      hiddenChannels: 128,
      headCount: 1
    )
    let input = MLXArray(
      (0..<(8 * 8 * 32)).map { sin(Float($0) * 0.01) },
      [1, 8, 8, 32]
    )

    let output = Device.withDefaultDevice(.gpu) { block(input) }
    eval(output)
    let values = output.asArray(Float.self)

    XCTAssertEqual(output.shape, input.shape)
    XCTAssertTrue(values.allSatisfy(\.isFinite))
    XCTAssertGreaterThan(values.map(abs).reduce(0, +), 0)
  }

  func testExternalRecoveredBlockFiveExecutesWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS to run the local recovered-weight probe")
    }
    let arrays = try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)
    let block = try NeuralRenderingWindowBlock(
      weights: ValidatedWeights(arrays: arrays),
      blockIndex: 5,
      channels: 64,
      hiddenChannels: 224,
      headCount: 2
    )
    let input = MLXArray(
      (0..<(8 * 8 * 64)).map { sin(Float($0) * 0.01) },
      [1, 8, 8, 64]
    )

    let output = Device.withDefaultDevice(.gpu) { block(input) }
    eval(output)
    let values = output.asArray(Float.self)

    XCTAssertEqual(output.shape, input.shape)
    XCTAssertTrue(values.allSatisfy(\.isFinite))
    XCTAssertGreaterThan(values.map(abs).reduce(0, +), 0)
  }

  func testPreBlockProjectsSixteenFeaturesBeforeWindowBlock() throws {
    let prefix = "block0.layer0"
    let arrays: [String: MLXArray] = [
      "\(prefix).input_adapter_weight": MLXArray([Float(2), -1], [1, 2]),
      "\(prefix).weight1": .zeros([2, 4]),
      "\(prefix).weight2": .zeros([4, 2]),
      "\(prefix).ffn_cos_skip": MLXArray([Float(1), 1]),
      "\(prefix).qkv_weight": .zeros([2, 6]),
      "\(prefix).attn_scale": MLXArray([Float(1)]),
      "\(prefix).attn_bias": .zeros([1, 64, 64]),
      "\(prefix).projection_weight": .zeros([2, 2]),
      "\(prefix).attn_cos_skip": MLXArray([Float(1), 1]),
    ]
    let pre = try NeuralRenderingPreBlock(
      weights: ValidatedWeights(arrays: arrays),
      inputChannels: 1,
      channels: 2,
      hiddenChannels: 4,
      headCount: 1
    )
    let input = MLXArray([Float](repeating: 3, count: 64), [1, 8, 8, 1])

    let output = pre(input)
    eval(output)

    XCTAssertEqual(output.shape, [1, 8, 8, 2])
    XCTAssertEqual(
      output.asArray(Float.self),
      Array(repeating: [Float(6), -3], count: 64).flatMap { $0 }
    )
  }

  func testExternalRecoveredEncoderStemExecutesWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS to run the local recovered-weight probe")
    }
    let arrays = try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)
    let stem = try NeuralRenderingEncoderStem(
      weights: ValidatedWeights(arrays: arrays)
    )
    let input = MLXArray(
      (0..<(8 * 8 * 16)).map { cos(Float($0) * 0.017) },
      [1, 8, 8, 16]
    )

    let output = Device.withDefaultDevice(.gpu) { stem(input) }
    eval(output.fullResolutionSkip, output.skip)
    let values = output.skip.asArray(Float.self)

    XCTAssertEqual(output.fullResolutionSkip.shape, [1, 8, 8, 32])
    XCTAssertEqual(output.skip.shape, [1, 4, 4, 32])
    XCTAssertTrue(values.allSatisfy(\.isFinite))
    XCTAssertGreaterThan(values.map(abs).reduce(0, +), 0)
  }

  func testDownsampleBlockRunsTransformerThenSpatialTransition() throws {
    let prefix = "block4.layer0"
    let arrays: [String: MLXArray] = [
      "\(prefix).weight1": .zeros([1, 2]),
      "\(prefix).weight2": .zeros([2, 1]),
      "\(prefix).ffn_cos_skip": MLXArray([Float(1)]),
      "\(prefix).qkv_weight": .zeros([1, 3]),
      "\(prefix).attn_scale": MLXArray([Float(1)]),
      "\(prefix).attn_bias": .zeros([1, 64, 64]),
      "\(prefix).projection_weight": .zeros([1, 1]),
      "\(prefix).attn_cos_skip": MLXArray([Float(1)]),
      "\(prefix).weight0": MLXArray([Float(2), -1], [1, 2]),
    ]
    let block = try NeuralRenderingDownsampleBlock(
      weights: ValidatedWeights(arrays: arrays),
      blockIndex: 4,
      channels: 1,
      hiddenChannels: 2,
      headCount: 1
    )
    let input = MLXArray((0..<64).map(Float.init), [1, 8, 8, 1])

    let output = block(input)
    eval(output)

    XCTAssertEqual(output.shape, [1, 4, 4, 2])
    XCTAssertEqual(Array(output.asArray(Float.self).prefix(4)), [9, -4.5, 13, -6.5])
  }

  func testExternalRecoveredFirstEncoderStageExecutesWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS to run the local recovered-weight probe")
    }
    let arrays = try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)
    let stage = try NeuralRenderingFirstEncoderStage(
      weights: ValidatedWeights(arrays: arrays)
    )
    let input = MLXArray(
      (0..<(16 * 16 * 16)).map { sin(Float($0) * 0.013) },
      [1, 16, 16, 16]
    )

    let output = Device.withDefaultDevice(.gpu) { stage(input) }
    eval(output.fullResolutionSkip, output.skip, output.downsampled)

    XCTAssertEqual(output.fullResolutionSkip.shape, [1, 16, 16, 32])
    XCTAssertEqual(output.skip.shape, [1, 8, 8, 32])
    XCTAssertEqual(output.downsampled.shape, [1, 4, 4, 64])
    XCTAssertTrue(output.skip.asArray(Float.self).allSatisfy(\.isFinite))
    XCTAssertTrue(output.downsampled.asArray(Float.self).allSatisfy(\.isFinite))
  }

  func testExternalRecoveredEncoderExecutesWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS to run the local recovered-weight probe")
    }
    let arrays = try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)
    let encoder = try NeuralRenderingEncoder(
      weights: ValidatedWeights(arrays: arrays)
    )
    let input = MLXArray(
      (0..<(64 * 64 * 16)).map { cos(Float($0) * 0.007) },
      [1, 64, 64, 16]
    )

    let output = Device.withDefaultDevice(.gpu) { encoder(input) }
    eval(output.latent)

    XCTAssertEqual(output.fullResolutionSkip.shape, [1, 64, 64, 32])
    XCTAssertEqual(
      output.skips.map(\.shape),
      [
        [1, 32, 32, 32],
        [1, 16, 16, 64],
        [1, 8, 8, 128],
        [1, 4, 4, 256],
      ])
    XCTAssertEqual(output.latent.shape, [1, 4, 4, 512])
    XCTAssertTrue(output.latent.asArray(Float.self).allSatisfy(\.isFinite))
  }

  func testExternalRecoveredStageTimingProfile() throws {
    let environment = ProcessInfo.processInfo.environment
    guard environment["MLXDLSS_PROFILE_NEURAL_RENDERING_STAGES"] == "1",
      let path = environment["MLXDLSS_LOGICAL_WEIGHTS"]
    else {
      throw XCTSkip("set external weights and MLXDLSS_PROFILE_NEURAL_RENDERING_STAGES=1")
    }
    let arrays = try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)
    let weights = ValidatedWeights(arrays: arrays)
    let fuseWindowFFN = environment["MLXDLSS_PROFILE_NEURAL_RENDERING_FUSED"] == "1"
    let encoder = try NeuralRenderingEncoder(
      weights: weights,
      compileBlocks: fuseWindowFFN
    )
    let split = try NeuralRenderingSplitEncoderStage(weights: weights)
    let global = try NeuralRenderingGlobalStage(weights: weights)
    let bridge = try NeuralRenderingDecoderInput(weights: weights)
    let decoder = try NeuralRenderingDecoder(
      weights: weights,
      compileBlocks: fuseWindowFFN
    )
    let post = try NeuralRenderingPostBlock(weights: weights)
    let input = MLXArray(
      (0..<(128 * 128 * 16)).map { sin(Float($0) * 0.001) },
      [1, 128, 128, 16]
    )

    func elapsed(_ operation: () -> [MLXArray]) -> UInt64 {
      let start = DispatchTime.now().uptimeNanoseconds
      eval(operation())
      return DispatchTime.now().uptimeNanoseconds - start
    }

    for iteration in 0..<5 {
      var timing: [String: UInt64] = [:]
      var encoded: NeuralRenderingEncoderOutput!
      timing["encoder_0_22"] = elapsed {
        encoded = encoder(input)
        return [encoded.fullResolutionSkip] + encoded.skips + [encoded.latent]
      }
      var splitOutput: NeuralRenderingSplitEncoderOutput!
      timing["split_23_30"] = elapsed {
        splitOutput = split(encoded.latent)
        return [splitOutput.skip, splitOutput.latent]
      }
      var globalOutput: MLXArray!
      timing["global_31_38"] = elapsed {
        globalOutput = global(splitOutput.latent)
        return [globalOutput]
      }
      var bridgeOutput: MLXArray!
      timing["bridge_39"] = elapsed {
        bridgeOutput = bridge(globalOutput, skip: splitOutput.skip)
        return [bridgeOutput]
      }
      var decoderOutput: MLXArray!
      timing["decoder_40_69"] = elapsed {
        decoderOutput = decoder(bridgeOutput, skips: encoded.skips)
        return [decoderOutput]
      }
      timing["post_70"] = elapsed {
        [post(decoderOutput, skip: encoded.fullResolutionSkip)]
      }
      let values = timing.keys.sorted().map { key in
        "\(key)=\(Double(timing[key]!) / 1_000_000)"
      }
      print("mlxdlss-stage-profile iteration=\(iteration) " + values.joined(separator: " "))
    }
  }
}
