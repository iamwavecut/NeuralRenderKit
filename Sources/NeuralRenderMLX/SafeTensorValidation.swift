import MLX
import NeuralRenderCore

public enum MLXBackendError: Error, Equatable, Sendable {
    case unsupportedArchitecture(String)
    case missingWeight(String)
    case extraWeight(String)
    case weightShapeMismatch(name: String, expected: [Int], actual: [Int])
    case weightDataTypeMismatch(name: String, expected: TensorDataType, actual: String)
    case missingOutput(String)
    case unsupportedTensorDataType(TensorDataType)
    case unsupportedSpatialShape(
        architecture: String,
        height: Int,
        width: Int,
        minimum: Int,
        multiple: Int
    )
    case modelStateKindMismatch(
        architecture: String,
        expected: ModelStateKind,
        actual: ModelStateKind
    )
    case compiledExecutionUnsupported(architecture: String)
}

struct ValidatedWeights {
    private let arrays: [String: MLXArray]

    var names: [String] {
        arrays.keys.sorted()
    }

    subscript(name: String) -> MLXArray? {
        arrays[name]
    }

    func required(_ name: String) throws -> MLXArray {
        guard let array = arrays[name] else {
            throw MLXBackendError.missingWeight(name)
        }
        return array
    }

    func cast(to precision: MLXComputePrecision) -> ValidatedWeights {
        ValidatedWeights(
            arrays: arrays.mapValues { array in
                array.dtype == precision.mlxDataType
                    ? array
                    : array.asType(precision.mlxDataType)
            }
        )
    }

    init(arrays: [String: MLXArray]) {
        self.arrays = arrays
    }
}

enum SafeTensorValidation {
    static func loadAndValidate(package: LoadedModelPackage) throws -> ValidatedWeights {
        let arrays = try loadArrays(url: package.weightsURL, stream: .cpu)
        let specs = package.manifest.weights.tensors
        let actualNames = Set(arrays.keys)
        let expectedNames = Set(specs.map(\.name))

        for spec in specs where !actualNames.contains(spec.name) {
            throw MLXBackendError.missingWeight(spec.name)
        }
        if let extra = actualNames.subtracting(expectedNames).sorted().first {
            throw MLXBackendError.extraWeight(extra)
        }

        for spec in specs {
            guard let array = arrays[spec.name] else {
                throw MLXBackendError.missingWeight(spec.name)
            }
            guard array.shape == spec.shape else {
                throw MLXBackendError.weightShapeMismatch(
                    name: spec.name,
                    expected: spec.shape,
                    actual: array.shape
                )
            }
            guard matches(array.dtype, expected: spec.dataType) else {
                throw MLXBackendError.weightDataTypeMismatch(
                    name: spec.name,
                    expected: spec.dataType,
                    actual: dataTypeName(array.dtype)
                )
            }
        }

        return ValidatedWeights(arrays: arrays)
    }

    private static func matches(_ actual: DType, expected: TensorDataType) -> Bool {
        switch (actual, expected) {
        case (.float16, .float16), (.float32, .float32):
            true
        default:
            false
        }
    }

    private static func dataTypeName(_ dataType: DType) -> String {
        switch dataType {
        case .bool: "bool"
        case .uint8: "uint8"
        case .uint16: "uint16"
        case .uint32: "uint32"
        case .uint64: "uint64"
        case .int8: "int8"
        case .int16: "int16"
        case .int32: "int32"
        case .int64: "int64"
        case .float16: "float16"
        case .float32: "float32"
        case .bfloat16: "bfloat16"
        case .complex64: "complex64"
        case .float64: "float64"
        }
    }
}
