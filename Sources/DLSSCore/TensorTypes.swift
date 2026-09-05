import Foundation

public enum TensorDataType: String, Codable, CaseIterable, Sendable {
    case float16
    case float32

    public var byteWidth: Int {
        switch self {
        case .float16:
            2
        case .float32:
            4
        }
    }
}

public enum TensorLayout: String, Codable, CaseIterable, Sendable {
    case nhwc
    case nchw
    case nthwc
    case vector
}

public enum TensorError: Error, Equatable, Sendable {
    case emptyName
    case emptyShape
    case invalidDimension(Int)
    case elementCountOverflow
    case byteCountOverflow
    case byteCount(expected: Int, actual: Int)
}

public struct TensorDescriptor: Equatable, Sendable {
    public let name: String
    public let shape: [Int]
    public let dataType: TensorDataType
    public let layout: TensorLayout
    public let elementCount: Int
    public let byteCount: Int

    public init(
        name: String,
        shape: [Int],
        dataType: TensorDataType,
        layout: TensorLayout
    ) throws {
        guard !name.isEmpty else {
            throw TensorError.emptyName
        }
        guard !shape.isEmpty else {
            throw TensorError.emptyShape
        }

        var elementCount = 1
        for dimension in shape {
            guard dimension > 0 else {
                throw TensorError.invalidDimension(dimension)
            }
            let product = elementCount.multipliedReportingOverflow(by: dimension)
            guard !product.overflow else {
                throw TensorError.elementCountOverflow
            }
            elementCount = product.partialValue
        }

        let byteCount = elementCount.multipliedReportingOverflow(by: dataType.byteWidth)
        guard !byteCount.overflow else {
            throw TensorError.byteCountOverflow
        }

        self.name = name
        self.shape = shape
        self.dataType = dataType
        self.layout = layout
        self.elementCount = elementCount
        self.byteCount = byteCount.partialValue
    }
}

public struct HostTensor: Equatable, Sendable {
    public let descriptor: TensorDescriptor
    public let bytes: Data

    public init(descriptor: TensorDescriptor, bytes: Data) throws {
        guard bytes.count == descriptor.byteCount else {
            throw TensorError.byteCount(
                expected: descriptor.byteCount,
                actual: bytes.count
            )
        }

        self.descriptor = descriptor
        self.bytes = bytes
    }
}
