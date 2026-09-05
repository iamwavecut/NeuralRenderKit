import Foundation

public enum ShapeDimension: Equatable, Sendable, Codable {
    case fixed(Int)
    case symbol(String)
    case scaled(symbol: String, multiplier: Int)

    private struct ScaledDimension: Codable {
        let symbol: String
        let multiplier: Int
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Int.self) {
            self = .fixed(value)
        } else if let value = try? container.decode(String.self) {
            self = .symbol(value)
        } else if let value = try? container.decode(ScaledDimension.self) {
            self = .scaled(symbol: value.symbol, multiplier: value.multiplier)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "A shape dimension must be an integer or symbol"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case let .fixed(value):
            try container.encode(value)
        case let .symbol(value):
            try container.encode(value)
        case let .scaled(symbol, multiplier):
            try container.encode(
                ScaledDimension(symbol: symbol, multiplier: multiplier)
            )
        }
    }
}

public struct ModelTensorSpec: Codable, Equatable, Sendable {
    public let name: String
    public let dataType: TensorDataType
    public let layout: TensorLayout
    public let shape: [ShapeDimension]
}

public enum ModelStateKind: String, Codable, Equatable, Sendable {
    case stateless
    case recurrent
}

public enum ModelStateResetPolicy: String, Codable, Equatable, Sendable {
    case explicit
    case explicitAndSceneCut
}

public struct ModelStateSpec: Codable, Equatable, Sendable {
    public let kind: ModelStateKind
    public let cadence: NeuralRenderTemporalCadence
    public let resetPolicy: ModelStateResetPolicy
    public let tensors: [ModelTensorSpec]

    private enum CodingKeys: String, CodingKey {
        case kind
        case cadence
        case resetPolicy
        case tensors
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.kind = try container.decode(ModelStateKind.self, forKey: .kind)
        self.cadence = try container.decodeIfPresent(
            NeuralRenderTemporalCadence.self,
            forKey: .cadence
        ) ?? .frameIndependent
        self.resetPolicy = try container.decodeIfPresent(
            ModelStateResetPolicy.self,
            forKey: .resetPolicy
        ) ?? .explicit
        self.tensors = try container.decodeIfPresent(
            [ModelTensorSpec].self,
            forKey: .tensors
        ) ?? []
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(kind, forKey: .kind)
        if cadence != .frameIndependent {
            try container.encode(cadence, forKey: .cadence)
        }
        if resetPolicy != .explicit {
            try container.encode(resetPolicy, forKey: .resetPolicy)
        }
        if !tensors.isEmpty {
            try container.encode(tensors, forKey: .tensors)
        }
    }
}

public struct WeightTensorSpec: Codable, Equatable, Sendable {
    public let name: String
    public let dataType: TensorDataType
    public let shape: [Int]
}

public struct WeightFileSpec: Codable, Equatable, Sendable {
    public let file: String
    public let sha256: String
    public let tensors: [WeightTensorSpec]
}

public enum ManifestError: Error, Equatable, Sendable {
    case malformedManifest
    case unsupportedSchemaVersion(Int)
    case invalidShapeDimension(Int)
    case invalidShapeSymbol(String)
    case invalidShapeMultiplier(Int)
    case duplicateTensorName(String)
    case missingInput(String)
    case extraInput(String)
    case layoutMismatch(name: String, expected: TensorLayout, actual: TensorLayout)
    case dataTypeMismatch(name: String, expected: TensorDataType, actual: TensorDataType)
    case rankMismatch(name: String, expected: Int, actual: Int)
    case fixedDimensionMismatch(name: String, axis: Int, expected: Int, actual: Int)
    case symbolMismatch(symbol: String, expected: Int, actual: Int)
    case unboundShapeSymbol(String)
    case scaledDimensionNotDivisible(
        name: String,
        axis: Int,
        symbol: String,
        multiplier: Int,
        actual: Int
    )
    case scaledDimensionOverflow(symbol: String, multiplier: Int, value: Int)
    case statelessStateDeclaresTensors
    case statelessStateRequiresFrameIndependentCadence
    case statelessStateRequiresExplicitResetPolicy
    case recurrentStateRequiresTensors
    case recurrentStateRequiresTemporalCadence
}

public struct ModelPackageManifest: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let identifier: String
    public let architecture: String
    public let inputs: [ModelTensorSpec]
    public let outputs: [ModelTensorSpec]
    public let state: ModelStateSpec
    public let weights: WeightFileSpec

    public static func decode(data: Data) throws -> Self {
        let manifest: Self
        do {
            manifest = try JSONDecoder().decode(Self.self, from: data)
        } catch {
            throw ManifestError.malformedManifest
        }

        try manifest.validate()
        return manifest
    }

    public func resolveInputs(_ tensors: [HostTensor]) throws -> [String: Int] {
        var byName: [String: HostTensor] = [:]
        for tensor in tensors {
            if byName.updateValue(tensor, forKey: tensor.descriptor.name) != nil {
                throw ManifestError.duplicateTensorName(tensor.descriptor.name)
            }
        }

        for spec in inputs where byName[spec.name] == nil {
            throw ManifestError.missingInput(spec.name)
        }

        let declaredNames = Set(inputs.map(\.name))
        if let extra = byName.keys.filter({ !declaredNames.contains($0) }).sorted().first {
            throw ManifestError.extraInput(extra)
        }

        var bindings: [String: Int] = [:]
        for spec in inputs {
            guard let tensor = byName[spec.name] else {
                throw ManifestError.missingInput(spec.name)
            }
            let descriptor = tensor.descriptor

            guard descriptor.layout == spec.layout else {
                throw ManifestError.layoutMismatch(
                    name: spec.name,
                    expected: spec.layout,
                    actual: descriptor.layout
                )
            }
            guard descriptor.dataType == spec.dataType else {
                throw ManifestError.dataTypeMismatch(
                    name: spec.name,
                    expected: spec.dataType,
                    actual: descriptor.dataType
                )
            }
            guard descriptor.shape.count == spec.shape.count else {
                throw ManifestError.rankMismatch(
                    name: spec.name,
                    expected: spec.shape.count,
                    actual: descriptor.shape.count
                )
            }

            for (axis, pair) in zip(spec.shape, descriptor.shape).enumerated() {
                let (expected, actual) = pair
                switch expected {
                case let .fixed(value):
                    guard value == actual else {
                        throw ManifestError.fixedDimensionMismatch(
                            name: spec.name,
                            axis: axis,
                            expected: value,
                            actual: actual
                        )
                    }
                case let .symbol(symbol):
                    if let bound = bindings[symbol] {
                        guard bound == actual else {
                            throw ManifestError.symbolMismatch(
                                symbol: symbol,
                                expected: bound,
                                actual: actual
                            )
                        }
                    } else {
                        bindings[symbol] = actual
                    }
                case let .scaled(symbol, multiplier):
                    if let bound = bindings[symbol] {
                        let expected = bound.multipliedReportingOverflow(
                            by: multiplier
                        )
                        guard !expected.overflow else {
                            throw ManifestError.scaledDimensionOverflow(
                                symbol: symbol,
                                multiplier: multiplier,
                                value: bound
                            )
                        }
                        guard expected.partialValue == actual else {
                            throw ManifestError.symbolMismatch(
                                symbol: symbol,
                                expected: expected.partialValue,
                                actual: actual
                            )
                        }
                    } else {
                        guard actual.isMultiple(of: multiplier) else {
                            throw ManifestError.scaledDimensionNotDivisible(
                                name: spec.name,
                                axis: axis,
                                symbol: symbol,
                                multiplier: multiplier,
                                actual: actual
                            )
                        }
                        bindings[symbol] = actual / multiplier
                    }
                }
            }
        }

        return bindings
    }

    public func resolveOutputDescriptors(
        bindings: [String: Int]
    ) throws -> [TensorDescriptor] {
        try outputs.map { spec in
            let shape = try spec.shape.map { dimension in
                switch dimension {
                case let .fixed(value):
                    return value
                case let .symbol(symbol):
                    guard let value = bindings[symbol] else {
                        throw ManifestError.unboundShapeSymbol(symbol)
                    }
                    return value
                case let .scaled(symbol, multiplier):
                    guard let value = bindings[symbol] else {
                        throw ManifestError.unboundShapeSymbol(symbol)
                    }
                    let scaled = value.multipliedReportingOverflow(
                        by: multiplier
                    )
                    guard !scaled.overflow else {
                        throw ManifestError.scaledDimensionOverflow(
                            symbol: symbol,
                            multiplier: multiplier,
                            value: value
                        )
                    }
                    return scaled.partialValue
                }
            }
            return try TensorDescriptor(
                name: spec.name,
                shape: shape,
                dataType: spec.dataType,
                layout: spec.layout
            )
        }
    }

    private func validate() throws {
        guard schemaVersion == 1 else {
            throw ManifestError.unsupportedSchemaVersion(schemaVersion)
        }

        try validateUniqueNames(inputs.map(\.name))
        try validateUniqueNames(outputs.map(\.name))
        try validateUniqueNames(weights.tensors.map(\.name))
        try validateUniqueNames(state.tensors.map(\.name))

        for dimension in inputs.flatMap(\.shape)
            + outputs.flatMap(\.shape)
            + state.tensors.flatMap(\.shape) {
            try validate(dimension)
        }
        for dimension in weights.tensors.flatMap(\.shape) where dimension <= 0 {
            throw ManifestError.invalidShapeDimension(dimension)
        }

        switch state.kind {
        case .stateless:
            guard state.tensors.isEmpty else {
                throw ManifestError.statelessStateDeclaresTensors
            }
            guard state.cadence == .frameIndependent else {
                throw ManifestError.statelessStateRequiresFrameIndependentCadence
            }
            guard state.resetPolicy == .explicit else {
                throw ManifestError.statelessStateRequiresExplicitResetPolicy
            }
        case .recurrent:
            guard !state.tensors.isEmpty else {
                throw ManifestError.recurrentStateRequiresTensors
            }
            guard state.cadence != .frameIndependent else {
                throw ManifestError.recurrentStateRequiresTemporalCadence
            }
        }
    }

    private func validate(_ dimension: ShapeDimension) throws {
        switch dimension {
        case let .fixed(value):
            guard value > 0 else {
                throw ManifestError.invalidShapeDimension(value)
            }
        case let .symbol(value):
            guard Self.isValidSymbol(value) else {
                throw ManifestError.invalidShapeSymbol(value)
            }
        case let .scaled(symbol, multiplier):
            guard Self.isValidSymbol(symbol) else {
                throw ManifestError.invalidShapeSymbol(symbol)
            }
            guard multiplier > 0 else {
                throw ManifestError.invalidShapeMultiplier(multiplier)
            }
        }
    }

    private func validateUniqueNames(_ names: [String]) throws {
        var seen: Set<String> = []
        for name in names where !seen.insert(name).inserted {
            throw ManifestError.duplicateTensorName(name)
        }
    }

    private static func isValidSymbol(_ value: String) -> Bool {
        guard let first = value.utf8.first, isASCIILetter(first) else {
            return false
        }

        return value.utf8.dropFirst().allSatisfy { byte in
            isASCIILetter(byte) || (byte >= 48 && byte <= 57) || byte == 95
        }
    }

    private static func isASCIILetter(_ byte: UInt8) -> Bool {
        (byte >= 65 && byte <= 90) || (byte >= 97 && byte <= 122)
    }
}
