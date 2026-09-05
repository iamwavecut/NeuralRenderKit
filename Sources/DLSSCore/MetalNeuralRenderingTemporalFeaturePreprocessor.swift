import Foundation
@preconcurrency import Metal
import DLSSCore

public enum MetalTemporalPreprocessorError: Error, Equatable, Sendable {
  case deviceUnavailable
  case commandQueueUnavailable
  case functionUnavailable
  case bufferAllocation
  case commandEncoding
  case execution(String)
}

/// Metal implementation of the recovered history sampling path.
///
/// Deterministic noise and current-frame feature assembly stay on the checked CPU
/// oracle. Metal optionally executes closest-depth selection and always executes five-tap
/// Catmull-Rom hotspot, then returns the same portable `HostTensor` surface.
public final class MetalNeuralRenderingTemporalFeaturePreprocessor:
  NeuralRenderingTemporalFeaturePreprocessing, @unchecked Sendable
{
  private struct Constants {
    let width: UInt32
    let height: UInt32
    let depthInverted: UInt32
    let useClosestDepth: UInt32
    let historyBaseX: UInt32
    let historyBaseY: UInt32
    let historyExtentWidth: UInt32
    let historyExtentHeight: UInt32
    let historyResourceWidth: UInt32
    let historyResourceHeight: UInt32
    let motionBaseX: UInt32
    let motionBaseY: UInt32
    let motionExtentWidth: UInt32
    let motionExtentHeight: UInt32
    let motionResourceWidth: UInt32
    let motionResourceHeight: UInt32
  }

  private let device: any MTLDevice
  private let commandQueue: any MTLCommandQueue
  private let pipeline: any MTLComputePipelineState
  private let depthGuideMode: NeuralRenderingDepthGuideMode
  private let historyTransform: NeuralRenderingTextureTransform?
  private let motionTransform: NeuralRenderingTextureTransform?
  public let featureControls: NeuralRenderingFeatureControls

  public convenience init() throws {
    try self.init(
      depthGuideMode: .observedZeroDescriptor,
      historyTransform: nil,
      motionTransform: nil
    )
  }

  public convenience init(depthGuideMode: NeuralRenderingDepthGuideMode) throws {
    try self.init(
      depthGuideMode: depthGuideMode,
      historyTransform: nil,
      motionTransform: nil
    )
  }

  public init(
    depthGuideMode: NeuralRenderingDepthGuideMode,
    historyTransform: NeuralRenderingTextureTransform?,
    motionTransform: NeuralRenderingTextureTransform?,
    featureControls: NeuralRenderingFeatureControls = .init()
  ) throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw MetalTemporalPreprocessorError.deviceUnavailable
    }
    guard let commandQueue = device.makeCommandQueue() else {
      throw MetalTemporalPreprocessorError.commandQueueUnavailable
    }
    let options = MTLCompileOptions()
    options.fastMathEnabled = false
    let library = try device.makeLibrary(source: Self.source, options: options)
    guard let function = library.makeFunction(name: "temporalHistoryFeatures") else {
      throw MetalTemporalPreprocessorError.functionUnavailable
    }
    self.device = device
    self.commandQueue = commandQueue
    self.pipeline = try device.makeComputePipelineState(function: function)
    self.depthGuideMode = depthGuideMode
    self.historyTransform = historyTransform
    self.motionTransform = motionTransform
    self.featureControls = featureControls
  }

  public func makeFeatureTensor(
    currentColor: HostTensor,
    historyColor: HostTensor,
    controlMask: HostTensor? = nil,
    normalizedMotion: HostTensor,
    depth: HostTensor,
    depthInverted: Bool,
    noiseFrameIndex: UInt32
  ) throws -> HostTensor {
    let resolvedHistoryTransform = try resolveTransform(
      historyTransform,
      tensor: historyColor,
      name: "history",
      channels: 3,
      like: currentColor
    )
    let resolvedMotionTransform = try resolveTransform(
      motionTransform,
      tensor: normalizedMotion,
      name: "motion",
      channels: 2,
      like: currentColor
    )
    _ = try resolveTransform(
      nil,
      tensor: depth,
      name: "depth",
      channels: 1,
      like: currentColor
    )
    let features = try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(
      from: currentColor,
      noiseFrameIndex: noiseFrameIndex,
      normalizedStyle: featureControls.normalizedStyle,
      localToneStrength: featureControls.localToneStrength,
      localStructureStrength: featureControls.localStructureStrength,
      automaticMask: featureControls.automaticMask,
      controlMask: controlMask
    )
    let width = currentColor.descriptor.shape[2]
    let height = currentColor.descriptor.shape[1]
    guard let featureBuffer = makeBuffer(features.bytes),
      let historyBuffer = makeBuffer(historyColor.bytes),
      let motionBuffer = makeBuffer(normalizedMotion.bytes),
      let depthBuffer = makeBuffer(depth.bytes)
    else {
      throw MetalTemporalPreprocessorError.bufferAllocation
    }
    guard let commandBuffer = commandQueue.makeCommandBuffer(),
      let encoder = commandBuffer.makeComputeCommandEncoder()
    else {
      throw MetalTemporalPreprocessorError.commandEncoding
    }
    var constants = Constants(
      width: UInt32(width),
      height: UInt32(height),
      depthInverted: depthInverted ? 1 : 0,
      useClosestDepth: depthGuideMode == .closestDepth ? 1 : 0,
      historyBaseX: UInt32(resolvedHistoryTransform.baseX),
      historyBaseY: UInt32(resolvedHistoryTransform.baseY),
      historyExtentWidth: UInt32(resolvedHistoryTransform.extentWidth),
      historyExtentHeight: UInt32(resolvedHistoryTransform.extentHeight),
      historyResourceWidth: UInt32(resolvedHistoryTransform.resourceWidth),
      historyResourceHeight: UInt32(resolvedHistoryTransform.resourceHeight),
      motionBaseX: UInt32(resolvedMotionTransform.baseX),
      motionBaseY: UInt32(resolvedMotionTransform.baseY),
      motionExtentWidth: UInt32(resolvedMotionTransform.extentWidth),
      motionExtentHeight: UInt32(resolvedMotionTransform.extentHeight),
      motionResourceWidth: UInt32(resolvedMotionTransform.resourceWidth),
      motionResourceHeight: UInt32(resolvedMotionTransform.resourceHeight)
    )
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(featureBuffer, offset: 0, index: 0)
    encoder.setBuffer(historyBuffer, offset: 0, index: 1)
    encoder.setBuffer(motionBuffer, offset: 0, index: 2)
    encoder.setBuffer(depthBuffer, offset: 0, index: 3)
    encoder.setBytes(&constants, length: MemoryLayout<Constants>.stride, index: 4)
    encoder.dispatchThreads(
      MTLSize(width: width, height: height, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1)
    )
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    guard commandBuffer.status == .completed else {
      throw MetalTemporalPreprocessorError.execution(
        commandBuffer.error.map(String.init(reflecting:)) ?? "unknown Metal error"
      )
    }
    return try HostTensor(
      descriptor: features.descriptor,
      bytes: Data(bytes: featureBuffer.contents(), count: features.bytes.count)
    )
  }

  private func makeBuffer(_ data: Data) -> (any MTLBuffer)? {
    data.withUnsafeBytes { bytes in
      device.makeBuffer(
        bytes: bytes.baseAddress!,
        length: bytes.count,
        options: .storageModeShared
      )
    }
  }

  private func resolveTransform(
    _ transform: NeuralRenderingTextureTransform?,
    tensor: HostTensor,
    name: String,
    channels: Int,
    like color: HostTensor
  ) throws -> NeuralRenderingTextureTransform {
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
    let resourceWidth = descriptor.shape[2]
    let resourceHeight = descriptor.shape[1]
    if let transform {
      guard transform.resourceWidth == resourceWidth,
        transform.resourceHeight == resourceHeight
      else {
        throw NeuralRenderingTextureTransformError.resourceShapeMismatch(
          name: name,
          expectedWidth: transform.resourceWidth,
          expectedHeight: transform.resourceHeight,
          actualWidth: resourceWidth,
          actualHeight: resourceHeight
        )
      }
      return transform
    }
    let expected = Array(color.descriptor.shape.prefix(3))
    let actual = Array(descriptor.shape.prefix(3))
    guard actual == expected else {
      throw NeuralRenderingTemporalPreprocessorError.spatialShapeMismatch(
        name: name,
        expected: expected,
        actual: actual
      )
    }
    return try NeuralRenderingTextureTransform(
      baseX: 0,
      baseY: 0,
      extentWidth: resourceWidth,
      extentHeight: resourceHeight,
      resourceWidth: resourceWidth,
      resourceHeight: resourceHeight
    )
  }

  private static let source = #"""
    #include <metal_stdlib>
    using namespace metal;

    struct Constants {
      uint width;
      uint height;
      uint depthInverted;
      uint useClosestDepth;
      uint historyBaseX;
      uint historyBaseY;
      uint historyExtentWidth;
      uint historyExtentHeight;
      uint historyResourceWidth;
      uint historyResourceHeight;
      uint motionBaseX;
      uint motionBaseY;
      uint motionExtentWidth;
      uint motionExtentHeight;
      uint motionResourceWidth;
      uint motionResourceHeight;
    };

    struct CatmullCoordinates {
      float outer0;
      float middle;
      float outer3;
      float w0;
      float w3;
      float g;
    };

    inline CatmullCoordinates catmullCoordinates(float normalized, uint dimension) {
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

    inline float3 loadRGB(
      const device float* image,
      uint width,
      int x,
      int y
    ) {
      uint offset = (uint(y) * width + uint(x)) * 3;
      return float3(image[offset], image[offset + 1], image[offset + 2]);
    }

    inline float3 sampleLinear(
      const device float* image,
      uint width,
      uint height,
      float x,
      float y
    ) {
      float pixelX = x - 0.5f;
      float pixelY = y - 0.5f;
      int x0 = clamp(int(floor(pixelX)), 0, int(width) - 1);
      int y0 = clamp(int(floor(pixelY)), 0, int(height) - 1);
      int x1 = min(x0 + 1, int(width) - 1);
      int y1 = min(y0 + 1, int(height) - 1);
      float tx = clamp(pixelX - float(x0), 0.0f, 1.0f);
      float ty = clamp(pixelY - float(y0), 0.0f, 1.0f);
      float3 top = loadRGB(image, width, x0, y0) * (1.0f - tx)
        + loadRGB(image, width, x1, y0) * tx;
      float3 bottom = loadRGB(image, width, x0, y1) * (1.0f - tx)
        + loadRGB(image, width, x1, y1) * tx;
      return top * (1.0f - ty) + bottom * ty;
    }

    inline float scaledColor(float value) {
      float sampled = float(half(value));
      float centered = float(half(sampled - 0.5f));
      return float(half(centered * 0.125f));
    }

    inline float mapTexturePixel(
      float logicalPixel,
      uint logicalDimension,
      uint base,
      uint extent
    ) {
      return float(base) + logicalPixel * float(extent) / float(logicalDimension);
    }

    kernel void temporalHistoryFeatures(
      device float* features [[buffer(0)]],
      const device float* history [[buffer(1)]],
      const device float* motion [[buffer(2)]],
      const device float* depth [[buffer(3)]],
      constant Constants& constants [[buffer(4)]],
      uint2 position [[thread_position_in_grid]]
    ) {
      if (position.x >= constants.width || position.y >= constants.height) {
        return;
      }
      int selectedX = int(position.x);
      int selectedY = int(position.y);
      if (constants.useClosestDepth != 0) {
        float selectedDepth = depth[position.y * constants.width + position.x];
        const int2 diagonals[4] = {
          int2(-1, -1), int2(1, -1), int2(-1, 1), int2(1, 1)
        };
        for (uint index = 0; index < 4; ++index) {
          int candidateX = clamp(
            int(position.x) + diagonals[index].x,
            0,
            int(constants.width) - 1
          );
          int candidateY = clamp(
            int(position.y) + diagonals[index].y,
            0,
            int(constants.height) - 1
          );
          float candidateDepth = depth[
            uint(candidateY) * constants.width + uint(candidateX)
          ];
          bool closer = constants.depthInverted != 0
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
        int(floor(mapTexturePixel(
          float(selectedX) + 0.5f,
          constants.width,
          constants.motionBaseX,
          constants.motionExtentWidth
        ))),
        0,
        int(constants.motionResourceWidth) - 1
      );
      int motionY = clamp(
        int(floor(mapTexturePixel(
          float(selectedY) + 0.5f,
          constants.height,
          constants.motionBaseY,
          constants.motionExtentHeight
        ))),
        0,
        int(constants.motionResourceHeight) - 1
      );
      uint motionOffset = (
        uint(motionY) * constants.motionResourceWidth + uint(motionX)
      ) * 2;
      float u = (float(position.x) + 0.5f) / float(constants.width)
        + motion[motionOffset];
      float v = (float(position.y) + 0.5f) / float(constants.height)
        + motion[motionOffset + 1];
      CatmullCoordinates x = catmullCoordinates(u, constants.width);
      CatmullCoordinates y = catmullCoordinates(v, constants.height);
      float leftWeight = x.w0 * y.g;
      float topWeight = x.g * y.w0;
      float middleWeight = x.g * y.g;
      float bottomWeight = x.g * y.w3;
      float rightWeight = x.w3 * y.g;
      float outer0X = mapTexturePixel(
        x.outer0, constants.width, constants.historyBaseX,
        constants.historyExtentWidth
      );
      float middleX = mapTexturePixel(
        x.middle, constants.width, constants.historyBaseX,
        constants.historyExtentWidth
      );
      float outer3X = mapTexturePixel(
        x.outer3, constants.width, constants.historyBaseX,
        constants.historyExtentWidth
      );
      float outer0Y = mapTexturePixel(
        y.outer0, constants.height, constants.historyBaseY,
        constants.historyExtentHeight
      );
      float middleY = mapTexturePixel(
        y.middle, constants.height, constants.historyBaseY,
        constants.historyExtentHeight
      );
      float outer3Y = mapTexturePixel(
        y.outer3, constants.height, constants.historyBaseY,
        constants.historyExtentHeight
      );
      float3 sampled = (
        sampleLinear(
          history,
          constants.historyResourceWidth,
          constants.historyResourceHeight,
          outer0X,
          middleY
        )
          * leftWeight
        + sampleLinear(
          history,
          constants.historyResourceWidth,
          constants.historyResourceHeight,
          middleX,
          outer0Y
        )
          * topWeight
        + sampleLinear(
          history,
          constants.historyResourceWidth,
          constants.historyResourceHeight,
          middleX,
          middleY
        )
          * middleWeight
        + sampleLinear(
          history,
          constants.historyResourceWidth,
          constants.historyResourceHeight,
          middleX,
          outer3Y
        )
          * bottomWeight
        + sampleLinear(
          history,
          constants.historyResourceWidth,
          constants.historyResourceHeight,
          outer3X,
          middleY
        )
          * rightWeight
      ) / (
        leftWeight + topWeight + middleWeight + bottomWeight + rightWeight
      );
      uint featureOffset = (
        position.y * constants.width + position.x
      ) * 16 + 7;
      features[featureOffset] = scaledColor(sampled.x);
      features[featureOffset + 1] = scaledColor(sampled.y);
      features[featureOffset + 2] = scaledColor(sampled.z);
    }
    """#
}
