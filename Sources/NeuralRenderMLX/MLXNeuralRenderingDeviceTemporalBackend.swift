import Foundation
import MLX
import NeuralRenderCore

/// Device-resident temporal path for the recovered neural-rendering transformer.
///
/// Inputs and final RGB retain the portable `HostTensor` boundary. Base features
/// use the deterministic CPU reference; temporal reprojection, the 71-block head,
/// postprocessing, and retained display history stay as MLX arrays.
public actor MLXNeuralRenderingDeviceTemporalBackend: NeuralRenderBackend {
  public nonisolated let temporalCadence: NeuralRenderTemporalCadence = .consecutiveFrames

  private let head: MLXNeuralRenderer
  private let computePrecision: MLXComputePrecision
  private let depthInverted: Bool
  private let historyTransform: NeuralRenderingTextureTransform?
  private let motionTransform: NeuralRenderingTextureTransform?
  private let controlMaskIntensity: Float
  private let featureControls: NeuralRenderingFeatureControls
  private let geometryPolicy: NeuralRenderingNetworkGeometryPolicy
  private let temporalProcessor = MLXTemporalFeatureProcessor()
  private let baseFeatureProcessor = MLXFirstFrameFeatureProcessor()
  /// `NRK_NR_DEVICE_FEATURES=0` builds the base features with the CPU preprocessor and uploads them (diagnostics).
  nonisolated(unsafe) static var deviceFeaturesEnabled: Bool = ProcessInfo.processInfo.environment["NRK_NR_DEVICE_FEATURES"] != "0"
  private let postprocessor: MLXTemporalPostprocessor
  private var tracker = TemporalLifecycleTracker()
  private var history: MLXArray?
  private var noiseFrameIndex: UInt32 = 0
  private var extensionIndices: (geometry: NeuralRenderingNetworkGeometry, rows: MLXArray, columns: MLXArray)?

  /// - Parameter geometry: how logical frames map onto the network extent.
  ///   `vendorAligned` (default) pads every frame to the recovered minimum
  ///   `320` / multiple-of-`64` extent before the head and crops afterwards;
  ///   `matchOutput` feeds the logical frame to the head unchanged.
  public init(
    packageURL: URL,
    executionMode: MLXExecutionMode = .eager,
    computePrecision: MLXComputePrecision = .float32,
    depthInverted: Bool = false,
    historyTransform: NeuralRenderingTextureTransform? = nil,
    motionTransform: NeuralRenderingTextureTransform? = nil,
    controlMaskIntensity: Float = 1,
    blendScale: Float = 0.739_746_093_75,
    featureControls: NeuralRenderingFeatureControls = .init(),
    geometry: NeuralRenderingNetworkGeometryPolicy = .vendorAligned
  ) throws {
    let package = try ModelPackageLoader.load(url: packageURL)
    guard package.manifest.architecture == "nrk.neural-rendering-transformer.v1" else {
      throw MLXBackendError.unsupportedArchitecture(package.manifest.architecture)
    }
    self.head = try MLXNeuralRenderer(
      packageURL: packageURL,
      executionMode: executionMode,
      computePrecision: computePrecision
    )
    self.computePrecision = computePrecision
    self.depthInverted = depthInverted
    self.historyTransform = historyTransform
    self.motionTransform = motionTransform
    self.controlMaskIntensity = controlMaskIntensity
    self.featureControls = featureControls
    self.geometryPolicy = geometry
    self.postprocessor = MLXTemporalPostprocessor(blendScale: blendScale)
  }

  public func render(_ request: NeuralRenderRequest) async throws -> NeuralRenderResult {
    guard let context = request.temporalContext else {
      throw TemporalLifecycleError.missingFrameContext
    }
    let color = try requiredInput("color", in: request)
    let motion = try requiredInput("motion", in: request)
    let depth = try requiredInput("depth", in: request)
    let controlMask = request.input(named: "controlMask")
    try validate(color, name: "color", channels: 3, expectedSpatialShape: nil)
    let spatialShape = Array(color.descriptor.shape.prefix(3))
    try validateResourceShape(
      historyTransform,
      name: "history",
      actualWidth: color.descriptor.shape[2],
      actualHeight: color.descriptor.shape[1]
    )
    try validate(
      motion,
      name: "motion",
      channels: 2,
      expectedSpatialShape: motionTransform == nil ? spatialShape : nil
    )
    try validateResourceShape(
      motionTransform,
      name: "motion",
      actualWidth: motion.descriptor.shape[2],
      actualHeight: motion.descriptor.shape[1]
    )
    try validate(depth, name: "depth", channels: 1, expectedSpatialShape: spatialShape)
    if let controlMask {
      try validate(
        controlMask,
        name: "controlMask",
        channels: 3,
        expectedSpatialShape: spatialShape
      )
    }
    let descriptors = [color.descriptor, motion.descriptor, depth.descriptor]
    if let resetRequest = try tracker.prepare(
      cadence: temporalCadence,
      context: context,
      inputDescriptors: descriptors
    ) {
      clearReferenceState()
      await head.reset(resetRequest)
    }

    do {
      let colorArray = array(color)
      let controlMaskArray = controlMask.map(array)
      let started = ContinuousClock.now
      let logicalHeight = color.descriptor.shape[1]
      let logicalWidth = color.descriptor.shape[2]
      let geometry = try geometryPolicy.resolve(
        outputWidth: logicalWidth,
        outputHeight: logicalHeight
      )
      let features: MLXArray
      let networkFeatures: MLXArray
      if Self.deviceFeaturesEnabled, let history, geometry.isIdentity {
        // Base features and the reprojected history in one kernel; nothing but the
        // colour, motion and depth crosses the host boundary.
        features = temporalProcessor(
          color: colorArray,
          controlMask: controlMaskArray,
          noiseFrameIndex: noiseFrameIndex,
          history: history,
          historyTransform: historyTransform,
          motion: array(motion),
          motionTransform: motionTransform,
          depth: array(depth),
          depthInverted: depthInverted,
          depthGuideMode: .observedZeroDescriptor,
          featureControls: featureControls
        )
        networkFeatures = features
      } else {
        let networkBaseFeatures: MLXArray
        if Self.deviceFeaturesEnabled {
          // The base features at the network extent from the colour gathered onto it
          // (the noise is a function of the network pixel, as in the CPU preprocessor).
          let extendedColor = geometry.isIdentity ? colorArray : extendColor(colorArray, geometry: geometry)
          let extendedMask = controlMaskArray.map { geometry.isIdentity ? $0 : extendColor($0, geometry: geometry) }
          networkBaseFeatures = baseFeatureProcessor(
            color: extendedColor,
            controlMask: extendedMask,
            noiseFrameIndex: noiseFrameIndex,
            featureControls: featureControls
          )
        } else {
          let baseFeatureTensor = try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(
            from: color,
            noiseFrameIndex: noiseFrameIndex,
            geometry: geometry,
            normalizedStyle: featureControls.normalizedStyle,
            localToneStrength: featureControls.localToneStrength,
            localStructureStrength: featureControls.localStructureStrength,
            automaticMask: featureControls.automaticMask,
            controlMask: controlMask
          )
          networkBaseFeatures = array(baseFeatureTensor)
        }
        let logicalBaseFeatures =
          geometry.isIdentity
          ? networkBaseFeatures
          : networkBaseFeatures[0..., 0..<logicalHeight, 0..<logicalWidth, 0...]
        if let history {
          features = temporalProcessor(
            baseFeatures: logicalBaseFeatures,
            history: history,
            historyTransform: historyTransform,
            motion: array(motion),
            motionTransform: motionTransform,
            depth: array(depth),
            depthInverted: depthInverted,
            depthGuideMode: .observedZeroDescriptor,
            featureControls: featureControls
          )
          networkFeatures =
            geometry.isIdentity
            ? features
            : extendToNetworkExtent(
              features,
              networkBaseFeatures: networkBaseFeatures,
              geometry: geometry
            )
        } else {
          features = logicalBaseFeatures
          networkFeatures = networkBaseFeatures
        }
      }
      let preparedFeatures = MLXArrayTransfer(
        array: networkFeatures.asType(computePrecision.mlxDataType)
      )
      let networkHeadOutput = try await head.statelessOutputTransfer(preparedFeatures).array
      let headOutput =
        geometry.isIdentity
        ? networkHeadOutput
        : networkHeadOutput[0..., 0..<logicalHeight, 0..<logicalWidth, 0...]
      let output = postprocessor(
        head: headOutput,
        currentColor: colorArray,
        features: features,
        hasHistory: history != nil,
        controlMask: controlMaskArray,
        intensity: controlMaskIntensity
      )
      eval(output)
      let executionNanoseconds = nanoseconds(
        in: started.duration(to: ContinuousClock.now)
      )
      let outputTensor = try hostTensor(output, shape: color.descriptor.shape)
      history = stopGradient(output)
      noiseFrameIndex &+= 1
      tracker.commit(context: context, inputDescriptors: descriptors)
      return try NeuralRenderResult(
        outputs: [outputTensor],
        timing: NeuralRenderTiming(executionNanoseconds: executionNanoseconds)
      )
    } catch {
      tracker.markInferenceFailure(frameIndex: context.frameIndex)
      throw error
    }
  }

  public func reset(sequenceID: UInt64?) async {
    await reset(NeuralRenderResetRequest(streamID: sequenceID, reason: .explicit))
  }

  public func reset(_ request: NeuralRenderResetRequest) async {
    tracker.reset()
    clearReferenceState()
    await head.reset(request)
  }

  private func array(_ tensor: HostTensor) -> MLXArray {
    MLXArray(
      tensor.bytes,
      tensor.descriptor.shape,
      dtype: .float32
    )
  }

  private func hostTensor(_ array: MLXArray, shape: [Int]) throws -> HostTensor {
    return try HostTensor(
      descriptor: TensorDescriptor(
        name: "color",
        shape: shape,
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: contiguous(array.asType(.float32)).asData(access: .copy).data   // one host copy
    )
  }

  private func requiredInput(
    _ name: String,
    in request: NeuralRenderRequest
  ) throws -> HostTensor {
    guard let input = request.input(named: name) else {
      throw NeuralRenderingTemporalReferenceBackendError.missingInput(name)
    }
    return input
  }

  private func validate(
    _ tensor: HostTensor,
    name: String,
    channels: Int,
    expectedSpatialShape: [Int]?
  ) throws {
    let descriptor = tensor.descriptor
    guard descriptor.shape.count == 4,
      descriptor.shape[0] == 1,
      descriptor.shape[3] == channels,
      descriptor.dataType == .float32,
      descriptor.layout == .nhwc
    else {
      throw NeuralRenderingTemporalPreprocessorError.expectedFloat32NHWC(
        name: name,
        channels: channels,
        shape: descriptor.shape,
        dataType: descriptor.dataType,
        layout: descriptor.layout
      )
    }
    if let expectedSpatialShape {
      let actual = Array(descriptor.shape.prefix(3))
      guard actual == expectedSpatialShape else {
        throw NeuralRenderingTemporalPreprocessorError.spatialShapeMismatch(
          name: name,
          expected: expectedSpatialShape,
          actual: actual
        )
      }
    }
  }

  private func validateResourceShape(
    _ transform: NeuralRenderingTextureTransform?,
    name: String,
    actualWidth: Int,
    actualHeight: Int
  ) throws {
    guard let transform else { return }
    guard transform.resourceWidth == actualWidth,
      transform.resourceHeight == actualHeight
    else {
      throw NeuralRenderingTextureTransformError.resourceShapeMismatch(
        name: name,
        expectedWidth: transform.resourceWidth,
        expectedHeight: transform.resourceHeight,
        actualWidth: actualWidth,
        actualHeight: actualHeight
      )
    }
  }

  private func clearReferenceState() {
    history = nil
    noiseFrameIndex = 0
  }

  /// Gathers logical temporal features onto the network extent through the
  /// recovered reflect-then-clamp mapping and takes the padded pixels' noise
  /// channels from the network-extent base features.
  private func extendToNetworkExtent(
    _ features: MLXArray,
    networkBaseFeatures: MLXArray,
    geometry: NeuralRenderingNetworkGeometry
  ) -> MLXArray {
    let indices = extensionIndices(for: geometry)
    let gathered = features.take(indices.rows, axis: 1).take(indices.columns, axis: 2)
    return concatenated(
      [
        networkBaseFeatures[0..., 0..., 0..., 0..<3],
        gathered[0..., 0..., 0..., 3...],
      ],
      axis: -1
    )
  }

  /// A logical `[1, H, W, C]` array gathered onto the network extent (reflect-then-clamp rows and columns).
  private func extendColor(_ array: MLXArray, geometry: NeuralRenderingNetworkGeometry) -> MLXArray {
    let indices = extensionIndices(for: geometry)
    return array.take(indices.rows, axis: 1).take(indices.columns, axis: 2)
  }

  private func extensionIndices(
    for geometry: NeuralRenderingNetworkGeometry
  ) -> (rows: MLXArray, columns: MLXArray) {
    if let cached = extensionIndices, cached.geometry == geometry {
      return (cached.rows, cached.columns)
    }
    let rows = MLXArray(
      (0..<geometry.networkHeight).map { y in
        Int32(geometry.sourceCoordinate(x: 0, y: y).y)
      }
    )
    let columns = MLXArray(
      (0..<geometry.networkWidth).map { x in
        Int32(geometry.sourceCoordinate(x: x, y: 0).x)
      }
    )
    extensionIndices = (geometry, rows, columns)
    return (rows, columns)
  }

  private func nanoseconds(in duration: Duration) -> UInt64 {
    let components = duration.components
    return UInt64(max(0, components.seconds)) * 1_000_000_000
      + UInt64(max(0, components.attoseconds) / 1_000_000_000)
  }
}

final class MLXTemporalPostprocessor: @unchecked Sendable {
  private let alphaLookup: MLXArray
  private let kernel = MLXFast.metalKernel(
    name: "nrk_temporal_postprocess",
    inputNames: ["head", "currentColor", "features", "alphaLookup", "controlMask"],
    outputNames: ["output"],
    source: #"""
      uint pixel = thread_position_in_grid.x;
      uint pixelCount = uint(currentColor_shape[1] * currentColor_shape[2]);
      if (pixel >= pixelCount) {
        return;
      }
      uint colorOffset = pixel * 3;
      uint headOffset = pixel * 4;
      uint featureOffset = pixel * 16;
      float alpha = 0.0f;
      if (hasHistory) {
        ushort logitIndex = as_type<ushort>(half(head[headOffset + 3]));
        alpha = alphaLookup[logitIndex];
      }
      for (uint channel = 0; channel < 3; ++channel) {
        float residual = float(half(head[headOffset + channel])) * 0.25f;
        float predicted = clamp(
          currentColor[colorOffset + channel] + residual,
          0.0f,
          1.0f
        );
        float temporal = predicted;
        if (hasHistory) {
          float history = features[featureOffset + 7 + channel] * 8.0f + 0.5f;
          temporal = predicted + alpha * (history - predicted);
        }
        if (hasEffectBlend) {
          float intensity = as_type<float>(uint(intensityBits));
          float red = hasControlMask ? controlMask[colorOffset] : 1.0f;
          float blend = clamp(red * intensity, 0.0f, 1.0f);
          float current = currentColor[colorOffset + channel];
          output[colorOffset + channel] = clamp(
            current + blend * (temporal - current),
            0.0f,
            1.0f
          );
        } else {
          output[colorOffset + channel] = temporal;
        }
      }
      """#,
    header: #"""
      #pragma clang fp contract(off)

      """#
  )

  init(blendScale: Float = 0.739_746_093_75) {
    let recoveredBlendScale = Float(Float16(blendScale))
    let values = (UInt32.zero...UInt32(UInt16.max)).map { bits in
      let logit = Float(Float16(bitPattern: UInt16(bits)))
      return min(
        1,
        max(0, 1 / (1 + exp(-logit)) * recoveredBlendScale)
      )
    }
    alphaLookup = MLXArray(values)
  }

  func callAsFunction(
    head: MLXArray,
    currentColor: MLXArray,
    features: MLXArray,
    hasHistory: Bool,
    controlMask: MLXArray? = nil,
    intensity: Float = 1
  ) -> MLXArray {
    kernel(
      [head, currentColor, features, alphaLookup, controlMask ?? currentColor],
      template: [
        ("hasHistory", hasHistory),
        ("hasEffectBlend", controlMask != nil || intensity != 1),
        ("hasControlMask", controlMask != nil),
        ("intensityBits", Int(intensity.bitPattern)),
      ],
      grid: (currentColor.shape[1] * currentColor.shape[2], 1, 1),
      threadGroup: (256, 1, 1),
      outputShapes: [currentColor.shape],
      outputDTypes: [.float32]
    )[0]
  }
}

private func featureTemplate(
  _ controls: NeuralRenderingFeatureControls,
  hasControlMask: Bool
) -> [(String, any KernelTemplateArg)] {
  return [
    ("hasAutomaticMask", controls.automaticMask != nil),
    ("hasControlMask", hasControlMask),
  ]
}

private func featureControlArray(
  _ controls: NeuralRenderingFeatureControls
) -> MLXArray {
  MLXArray([
    controls.normalizedStyle,
    controls.localToneStrength,
    controls.localStructureStrength,
    controls.automaticMask?.skinStructureStrength ?? -1,
    controls.automaticMask?.automaticMaskStructureStrength ?? -1,
  ])
}

final class MLXFirstFrameFeatureProcessor: @unchecked Sendable {
  private let kernel = MLXFast.metalKernel(
    name: "nrk_first_frame_features",
    inputNames: ["color", "controlMask", "frameIndex", "featureControls"],
    outputNames: ["output"],
    source: source,
    header: header
  )

  func callAsFunction(
    color: MLXArray,
    controlMask: MLXArray? = nil,
    noiseFrameIndex: UInt32,
    featureControls: NeuralRenderingFeatureControls = .init()
  ) -> MLXArray {
    let height = color.shape[1]
    let width = color.shape[2]
    return kernel(
      [
        color,
        controlMask ?? color,
        MLXArray(noiseFrameIndex),
        featureControlArray(featureControls),
      ],
      template: featureTemplate(
        featureControls,
        hasControlMask: controlMask != nil
      ),
      grid: (width, height, 1),
      threadGroup: (8, 8, 1),
      outputShapes: [[1, height, width, 16]],
      outputDTypes: [.float32]
    )[0]
  }

  static let header = #"""
    using namespace metal;
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)

    METAL_FUNC uint nrkDynamicShiftMix(uint input) {
      // PCG RXS-M-XS output step; the shift is never zero.
      uint shift = (input >> 28) + 4u;
      return (input ^ (input >> shift)) * 0x108ef2d9u;
    }

    METAL_FUNC float nrkUniform24(uint input) {
      uint multiplied = nrkDynamicShiftMix(input);
      uint bits = (multiplied >> 30) ^ (multiplied >> 8);
      return float(bits + 1u) * 5.960464477539063e-8f;
    }

    METAL_FUNC float nrkHalf(float value) {
      return float(half(value));
    }

    // log, sin and cos from multiplications and additions only (Cephes), the same
    // operation sequence as NeuralRenderingNoiseMath on the CPU: bit-identical noise.
    METAL_FUNC float nrkLog(float x) {
      uint bits = as_type<uint>(x);
      float e = float(int((bits >> 23) & 0xffu) - 126);
      float m = as_type<float>((bits & 0x807fffffu) | 0x3f000000u);
      if (m < 0.707106781186547524f) { e -= 1.0f; m = m + m - 1.0f; } else { m = m - 1.0f; }
      float z = m * m;
      float y = 7.0376836292e-2f * m - 1.1514610310e-1f;
      y = y * m + 1.1676998740e-1f;
      y = y * m - 1.2420140846e-1f;
      y = y * m + 1.4249322787e-1f;
      y = y * m - 1.6668057665e-1f;
      y = y * m + 2.0000714765e-1f;
      y = y * m - 2.4999993993e-1f;
      y = y * m + 3.3333331174e-1f;
      y = y * m * z;
      y = y + -2.12194440e-4f * e;
      y = y + -0.5f * z;
      float result = m + y;
      result = result + 0.693359375f * e;
      return result;
    }

    METAL_FUNC float nrkSinPoly(float z, float x) {
      float y = -1.9515295891e-4f * z + 8.3321608736e-3f;
      y = y * z - 1.6666654611e-1f;
      y = y * z * x;
      return y + x;
    }

    METAL_FUNC float nrkCosPoly(float z) {
      float y = 2.443315711809948e-5f * z - 1.388731625493765e-3f;
      y = y * z + 4.166664568298827e-2f;
      y = y * z * z;
      y = y - 0.5f * z;
      return y + 1.0f;
    }

    METAL_FUNC float nrkSin(float input) {
      float sign = 1.0f;
      float x = input;
      if (x < 0.0f) { x = -x; sign = -1.0f; }
      int j = int(1.27323954473516f * x);
      float y = float(j);
      if (j & 1) { j += 1; y += 1.0f; }
      j &= 7;
      if (j > 3) { sign = -sign; j -= 4; }
      x = ((x - y * 0.78515625f) - y * 2.4187564849853515625e-4f) - y * 3.77489497744594108e-8f;
      float z = x * x;
      float result = (j == 1 || j == 2) ? nrkCosPoly(z) : nrkSinPoly(z, x);
      return sign < 0.0f ? -result : result;
    }

    METAL_FUNC float nrkCos(float input) {
      float x = input < 0.0f ? -input : input;
      int j = int(1.27323954473516f * x);
      float y = float(j);
      if (j & 1) { j += 1; y += 1.0f; }
      j &= 7;
      float sign = 1.0f;
      if (j > 3) { j -= 4; sign = -sign; }
      if (j > 1) { sign = -sign; }
      x = ((x - y * 0.78515625f) - y * 2.4187564849853515625e-4f) - y * 3.77489497744594108e-8f;
      float z = x * x;
      float result = (j == 1 || j == 2) ? nrkSinPoly(z, x) : nrkCosPoly(z);
      return sign < 0.0f ? -result : result;
    }

    METAL_FUNC float nrkScaledColor(float value) {
      float sampled = nrkHalf(value);
      float centered = nrkHalf(sampled - 0.5f);
      return nrkHalf(centered * 0.125f);
    }

    template <typename ColorPointer, typename ControlMaskPointer>
    METAL_FUNC void nrkWriteFirstFrameFeatures(
      device float* output,
      ColorPointer color,
      ControlMaskPointer controlMask,
      uint frameIndex,
      uint x,
      uint y,
      uint width,
      float normalizedStyle,
      float localToneStrength,
      float localStructureStrength,
      float skinStructureStrength,
      float automaticMaskStructureStrength,
      bool hasAutomaticMask,
      bool hasControlMask
    ) {
      uint pixel = y * width + x;
      uint colorOffset = pixel * 3;
      uint featureOffset = pixel * 16;
      uint seed = y * 0xd8163841u;
      seed ^= x * 0x8da6b343u;
      seed ^= frameIndex * 0x9e3779b9u;
      seed ^= 0x243f6a88u;
      uint multiplied = nrkDynamicShiftMix(seed);
      uint mixed = multiplied ^ (multiplied >> 22);

      float radiusAUniform = nrkUniform24(
        mixed * 0xcaa5b80du + 0x21dd796bu
      );
      float angleBUniform = nrkUniform24(
        mixed * 0x83232c31u + 0x3463e0acu
      );
      float radiusBUniform = nrkUniform24(
        mixed * 0x2c9277b5u + 0xac564b05u
      );
      float angleAUniform = nrkUniform24(
        mixed * 0xfa6dc5f9u + 0x4712a88eu
      );
      float radiusA = metal::precise::sqrt(-2.0f * nrkLog(radiusAUniform));
      float radiusB = metal::precise::sqrt(-2.0f * nrkLog(radiusBUniform));
      float tau = 6.2831854820251465f;
      float angleA = tau * angleAUniform;
      float angleB = tau * angleBUniform;
      output[featureOffset] = nrkHalf(radiusB * nrkCos(angleA));
      output[featureOffset + 1] = nrkHalf(radiusB * nrkSin(angleA));
      output[featureOffset + 2] = nrkHalf(radiusA * nrkCos(angleB));
      output[featureOffset + 3] = 1.0f;

      float red = nrkScaledColor(color[colorOffset]);
      float green = nrkScaledColor(color[colorOffset + 1]);
      float blue = nrkScaledColor(color[colorOffset + 2]);
      output[featureOffset + 4] = red;
      output[featureOffset + 5] = green;
      output[featureOffset + 6] = blue;
      output[featureOffset + 7] = red;
      output[featureOffset + 8] = green;
      output[featureOffset + 9] = blue;
      float structure;
      float skinStructure;
      float automaticMaskStructure;
      if (hasControlMask) {
        structure = 0.0f;
        skinStructure = 0.0f;
        automaticMaskStructure = 0.0f;
      } else if (hasAutomaticMask) {
        bool hasEnabledStrength = max(
          skinStructureStrength,
          automaticMaskStructureStrength
        ) >= 0.0f;
        structure = nrkHalf(
          hasEnabledStrength ? 1.0f : localStructureStrength
        );
        skinStructure = nrkHalf(
          hasEnabledStrength
            ? (skinStructureStrength >= 0.0f
              ? skinStructureStrength
              : localStructureStrength)
            : -1.0f
        );
        automaticMaskStructure = nrkHalf(
          hasEnabledStrength
            ? (automaticMaskStructureStrength >= 0.0f
              ? automaticMaskStructureStrength
              : localStructureStrength)
            : -1.0f
        );
      } else {
        structure = nrkHalf(localStructureStrength);
        skinStructure = -1.0f;
        automaticMaskStructure = -1.0f;
      }
      output[featureOffset + 10] = nrkHalf(normalizedStyle);
      output[featureOffset + 11] = hasControlMask
        ? nrkHalf(controlMask[colorOffset + 1] * localToneStrength)
        : nrkHalf(localToneStrength);
      output[featureOffset + 12] = hasControlMask
        ? nrkHalf(controlMask[colorOffset + 2] * localStructureStrength)
        : structure;
      output[featureOffset + 13] = skinStructure;
      output[featureOffset + 14] = automaticMaskStructure;
      output[featureOffset + 15] = 0.0f;
    }
    """#

  private static let source = #"""
    uint x = thread_position_in_grid.x;
    uint y = thread_position_in_grid.y;
    uint height = uint(color_shape[1]);
    uint width = uint(color_shape[2]);
    if (x >= width || y >= height) {
      return;
    }
    nrkWriteFirstFrameFeatures(
      output,
      color,
      controlMask,
      frameIndex,
      x,
      y,
      width,
      featureControls[0],
      featureControls[1],
      featureControls[2],
      featureControls[3],
      featureControls[4],
      hasAutomaticMask,
      hasControlMask
    );
    """#
}

final class MLXTemporalFeatureProcessor: @unchecked Sendable {
  private let kernel = MLXFast.metalKernel(
    name: "nrk_temporal_history_features",
    inputNames: [
      "baseFeatures", "color", "controlMask", "frameIndex",
      "featureControls", "history", "motion", "depth",
    ],
    outputNames: ["output"],
    source: source,
    header: MLXFirstFrameFeatureProcessor.header + temporalHeader
  )

  func callAsFunction(
    baseFeatures: MLXArray,
    history: MLXArray,
    historyTransform: NeuralRenderingTextureTransform? = nil,
    motion: MLXArray,
    motionTransform: NeuralRenderingTextureTransform? = nil,
    depth: MLXArray,
    depthInverted: Bool,
    depthGuideMode: NeuralRenderingDepthGuideMode = .observedZeroDescriptor,
    featureControls: NeuralRenderingFeatureControls = .init()
  ) -> MLXArray {
    process(
      baseFeatures: baseFeatures,
      color: baseFeatures,
      controlMask: nil,
      noiseFrameIndex: 0,
      generateBaseFeatures: false,
      history: history,
      historyTransform: historyTransform,
      motion: motion,
      motionTransform: motionTransform,
      depth: depth,
      depthInverted: depthInverted,
      depthGuideMode: depthGuideMode,
      featureControls: featureControls
    )
  }

  func callAsFunction(
    color: MLXArray,
    controlMask: MLXArray? = nil,
    noiseFrameIndex: UInt32,
    history: MLXArray,
    historyTransform: NeuralRenderingTextureTransform? = nil,
    motion: MLXArray,
    motionTransform: NeuralRenderingTextureTransform? = nil,
    depth: MLXArray,
    depthInverted: Bool,
    depthGuideMode: NeuralRenderingDepthGuideMode = .observedZeroDescriptor,
    featureControls: NeuralRenderingFeatureControls = .init()
  ) -> MLXArray {
    process(
      baseFeatures: color,
      color: color,
      controlMask: controlMask,
      noiseFrameIndex: noiseFrameIndex,
      generateBaseFeatures: true,
      history: history,
      historyTransform: historyTransform,
      motion: motion,
      motionTransform: motionTransform,
      depth: depth,
      depthInverted: depthInverted,
      depthGuideMode: depthGuideMode,
      featureControls: featureControls
    )
  }

  private func process(
    baseFeatures: MLXArray,
    color: MLXArray,
    controlMask: MLXArray?,
    noiseFrameIndex: UInt32,
    generateBaseFeatures: Bool,
    history: MLXArray,
    historyTransform: NeuralRenderingTextureTransform?,
    motion: MLXArray,
    motionTransform: NeuralRenderingTextureTransform?,
    depth: MLXArray,
    depthInverted: Bool,
    depthGuideMode: NeuralRenderingDepthGuideMode,
    featureControls: NeuralRenderingFeatureControls
  ) -> MLXArray {
    let logicalWidth = baseFeatures.shape[2]
    let logicalHeight = baseFeatures.shape[1]
    let resolvedHistoryTransform =
      historyTransform
      ?? identityTransform(
        width: history.shape[2],
        height: history.shape[1]
      )
    let resolvedMotionTransform =
      motionTransform
      ?? identityTransform(
        width: motion.shape[2],
        height: motion.shape[1]
      )
    precondition(
      resolvedHistoryTransform.resourceWidth == history.shape[2]
        && resolvedHistoryTransform.resourceHeight == history.shape[1]
    )
    precondition(
      resolvedMotionTransform.resourceWidth == motion.shape[2]
        && resolvedMotionTransform.resourceHeight == motion.shape[1]
    )
    var template = featureTemplate(
      featureControls,
      hasControlMask: controlMask != nil
    )
    template += [
      ("generateBaseFeatures", generateBaseFeatures),
      ("depthInverted", depthInverted),
      ("useClosestDepth", depthGuideMode == .closestDepth),
      ("logicalWidth", logicalWidth),
      ("logicalHeight", logicalHeight),
      ("historyBaseX", resolvedHistoryTransform.baseX),
      ("historyBaseY", resolvedHistoryTransform.baseY),
      ("historyExtentWidth", resolvedHistoryTransform.extentWidth),
      ("historyExtentHeight", resolvedHistoryTransform.extentHeight),
      ("historyResourceWidth", resolvedHistoryTransform.resourceWidth),
      ("historyResourceHeight", resolvedHistoryTransform.resourceHeight),
      ("motionBaseX", resolvedMotionTransform.baseX),
      ("motionBaseY", resolvedMotionTransform.baseY),
      ("motionExtentWidth", resolvedMotionTransform.extentWidth),
      ("motionExtentHeight", resolvedMotionTransform.extentHeight),
      ("motionResourceWidth", resolvedMotionTransform.resourceWidth),
      ("motionResourceHeight", resolvedMotionTransform.resourceHeight),
    ]
    return kernel(
      [
        baseFeatures,
        color,
        controlMask ?? color,
        MLXArray(noiseFrameIndex),
        featureControlArray(featureControls),
        history,
        motion,
        depth,
      ],
      template: template,
      grid: (baseFeatures.shape[2], baseFeatures.shape[1], 1),
      threadGroup: (8, 8, 1),
      outputShapes: [[1, logicalHeight, logicalWidth, 16]],
      outputDTypes: [.float32]
    )[0]
  }

  private func identityTransform(width: Int, height: Int) -> NeuralRenderingTextureTransform {
    try! NeuralRenderingTextureTransform(
      baseX: 0,
      baseY: 0,
      extentWidth: width,
      extentHeight: height,
      resourceWidth: width,
      resourceHeight: height
    )
  }

  private static let temporalHeader = #"""
    using namespace metal;
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)

    struct NRKCatmullCoordinates {
      float outer0;
      float middle;
      float outer3;
      float w0;
      float w3;
      float g;
    };

    METAL_FUNC NRKCatmullCoordinates nrkCatmullCoordinates(
      float normalized,
      int dimension
    ) {
      float pixel = normalized * float(dimension) - 0.5f;
      float baseIndex = floor(pixel);
      float t = clamp(pixel - baseIndex, 0.0f, 1.0f);
      float square = t * t;
      float cube = square * t;
      float w0 = -0.5f * t + square - 0.5f * cube;
      float w1 = 1.0f - 2.5f * square + 1.5f * cube;
      float w2 = 0.5f * t + 2.0f * square - 1.5f * cube;
      float w3 = -0.5f * square + 0.5f * cube;
      float g = w1 + w2;
      float base = baseIndex + 0.5f;
      float upper = float(dimension) - 0.5f;
      return {
        clamp(base - 1.0f, 0.5f, upper),
        clamp(base + w2 / g, 0.5f, upper),
        clamp(base + 2.0f, 0.5f, upper),
        w0,
        w3,
        g
      };
    }

    METAL_FUNC float3 nrkLoadRGB(
      const device float* image,
      int width,
      int x,
      int y
    ) {
      int offset = (y * width + x) * 3;
      return float3(image[offset], image[offset + 1], image[offset + 2]);
    }

    METAL_FUNC float3 nrkSampleLinear(
      const device float* image,
      int width,
      int height,
      float x,
      float y
    ) {
      float pixelX = x - 0.5f;
      float pixelY = y - 0.5f;
      int x0 = clamp(int(floor(pixelX)), 0, width - 1);
      int y0 = clamp(int(floor(pixelY)), 0, height - 1);
      int x1 = min(x0 + 1, width - 1);
      int y1 = min(y0 + 1, height - 1);
      float tx = clamp(pixelX - float(x0), 0.0f, 1.0f);
      float ty = clamp(pixelY - float(y0), 0.0f, 1.0f);
      float3 top = nrkLoadRGB(image, width, x0, y0) * (1.0f - tx)
        + nrkLoadRGB(image, width, x1, y0) * tx;
      float3 bottom = nrkLoadRGB(image, width, x0, y1) * (1.0f - tx)
        + nrkLoadRGB(image, width, x1, y1) * tx;
      return top * (1.0f - ty) + bottom * ty;
    }

    METAL_FUNC float nrkMapTexturePixel(
      float logicalPixel,
      int logicalDimension,
      int base,
      int extent
    ) {
      return float(base) + logicalPixel * float(extent) / float(logicalDimension);
    }
    """#

  private static let source = #"""
    uint xPosition = thread_position_in_grid.x;
    uint yPosition = thread_position_in_grid.y;
    int height = baseFeatures_shape[1];
    int width = baseFeatures_shape[2];
    if (xPosition >= uint(width) || yPosition >= uint(height)) {
      return;
    }
    int pixel = int(yPosition) * width + int(xPosition);
    int featureOffset = pixel * 16;
    if (generateBaseFeatures) {
      nrkWriteFirstFrameFeatures(
        output,
        color,
        controlMask,
        frameIndex,
        xPosition,
        yPosition,
        uint(width),
        featureControls[0],
        featureControls[1],
        featureControls[2],
        featureControls[3],
        featureControls[4],
        hasAutomaticMask,
        hasControlMask
      );
    } else {
      for (int channel = 0; channel < 16; ++channel) {
        output[featureOffset + channel] = baseFeatures[featureOffset + channel];
      }
    }

    int selectedX = int(xPosition);
    int selectedY = int(yPosition);
    if (useClosestDepth) {
      float selectedDepth = depth[pixel];
      const int2 diagonals[4] = {
        int2(-1, -1), int2(1, -1), int2(-1, 1), int2(1, 1)
      };
      for (int index = 0; index < 4; ++index) {
        int candidateX = clamp(int(xPosition) + diagonals[index].x, 0, width - 1);
        int candidateY = clamp(int(yPosition) + diagonals[index].y, 0, height - 1);
        float candidateDepth = depth[candidateY * width + candidateX];
        bool closer = depthInverted
          ? candidateDepth > selectedDepth
          : candidateDepth < selectedDepth;
        if (closer) {
          selectedX = candidateX;
          selectedY = candidateY;
          selectedDepth = candidateDepth;
        }
      }
    }

    int motionX = clamp(
      int(floor(nrkMapTexturePixel(
        float(selectedX) + 0.5f,
        logicalWidth,
        motionBaseX,
        motionExtentWidth
      ))),
      0,
      motionResourceWidth - 1
    );
    int motionY = clamp(
      int(floor(nrkMapTexturePixel(
        float(selectedY) + 0.5f,
        logicalHeight,
        motionBaseY,
        motionExtentHeight
      ))),
      0,
      motionResourceHeight - 1
    );
    int motionOffset = (motionY * motionResourceWidth + motionX) * 2;
    float u = (float(xPosition) + 0.5f) / float(width) + motion[motionOffset];
    float v = (float(yPosition) + 0.5f) / float(height) + motion[motionOffset + 1];
    NRKCatmullCoordinates x = nrkCatmullCoordinates(u, width);
    NRKCatmullCoordinates y = nrkCatmullCoordinates(v, height);
    float leftWeight = x.w0 * y.g;
    float topWeight = x.g * y.w0;
    float middleWeight = x.g * y.g;
    float bottomWeight = x.g * y.w3;
    float rightWeight = x.w3 * y.g;
    float outer0X = nrkMapTexturePixel(
      x.outer0, logicalWidth, historyBaseX, historyExtentWidth
    );
    float middleX = nrkMapTexturePixel(
      x.middle, logicalWidth, historyBaseX, historyExtentWidth
    );
    float outer3X = nrkMapTexturePixel(
      x.outer3, logicalWidth, historyBaseX, historyExtentWidth
    );
    float outer0Y = nrkMapTexturePixel(
      y.outer0, logicalHeight, historyBaseY, historyExtentHeight
    );
    float middleY = nrkMapTexturePixel(
      y.middle, logicalHeight, historyBaseY, historyExtentHeight
    );
    float outer3Y = nrkMapTexturePixel(
      y.outer3, logicalHeight, historyBaseY, historyExtentHeight
    );
    float3 sampled = (
      nrkSampleLinear(
        history, historyResourceWidth, historyResourceHeight, outer0X, middleY
      ) * leftWeight
      + nrkSampleLinear(
        history, historyResourceWidth, historyResourceHeight, middleX, outer0Y
      ) * topWeight
      + nrkSampleLinear(
        history, historyResourceWidth, historyResourceHeight, middleX, middleY
      ) * middleWeight
      + nrkSampleLinear(
        history, historyResourceWidth, historyResourceHeight, middleX, outer3Y
      ) * bottomWeight
      + nrkSampleLinear(
        history, historyResourceWidth, historyResourceHeight, outer3X, middleY
      ) * rightWeight
    ) / (leftWeight + topWeight + middleWeight + bottomWeight + rightWeight);
    output[featureOffset + 7] = nrkScaledColor(sampled.x);
    output[featureOffset + 8] = nrkScaledColor(sampled.y);
    output[featureOffset + 9] = nrkScaledColor(sampled.z);
    """#
}
