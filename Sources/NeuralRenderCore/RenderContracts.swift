public enum RenderContractError: Error, Equatable, Sendable {
    case emptyInputs
    case emptyOutputs
    case duplicateTensor(String)
}

public struct NeuralRenderRequest: Equatable, Sendable {
    public let sequenceID: UInt64
    public let temporalContext: NeuralRenderFrameContext?
    public let inputs: [HostTensor]
    private let inputsByName: [String: HostTensor]

    public init(
        sequenceID: UInt64,
        temporalContext: NeuralRenderFrameContext? = nil,
        inputs: [HostTensor]
    ) throws {
        guard !inputs.isEmpty else {
            throw RenderContractError.emptyInputs
        }

        var inputsByName: [String: HostTensor] = [:]
        for input in inputs {
            let name = input.descriptor.name
            guard inputsByName.updateValue(input, forKey: name) == nil else {
                throw RenderContractError.duplicateTensor(name)
            }
        }

        self.sequenceID = sequenceID
        self.temporalContext = temporalContext
        self.inputs = inputs.sorted { $0.descriptor.name < $1.descriptor.name }
        self.inputsByName = inputsByName
    }

    public func input(named name: String) -> HostTensor? {
        inputsByName[name]
    }
}

public struct NeuralRenderTiming: Codable, Equatable, Sendable {
    public let executionNanoseconds: UInt64

    public init(executionNanoseconds: UInt64) {
        self.executionNanoseconds = executionNanoseconds
    }
}

public struct NeuralRenderResult: Equatable, Sendable {
    public let outputs: [HostTensor]
    public let timing: NeuralRenderTiming
    private let outputsByName: [String: HostTensor]

    public init(outputs: [HostTensor], timing: NeuralRenderTiming) throws {
        guard !outputs.isEmpty else {
            throw RenderContractError.emptyOutputs
        }

        var outputsByName: [String: HostTensor] = [:]
        for output in outputs {
            let name = output.descriptor.name
            guard outputsByName.updateValue(output, forKey: name) == nil else {
                throw RenderContractError.duplicateTensor(name)
            }
        }

        self.outputs = outputs.sorted { $0.descriptor.name < $1.descriptor.name }
        self.timing = timing
        self.outputsByName = outputsByName
    }

    public func output(named name: String) -> HostTensor? {
        outputsByName[name]
    }
}

public protocol NeuralRenderBackend: Sendable {
    var temporalCadence: NeuralRenderTemporalCadence { get }

    func render(_ request: NeuralRenderRequest) async throws -> NeuralRenderResult
    func reset(sequenceID: UInt64?) async
    func reset(_ request: NeuralRenderResetRequest) async
}

extension NeuralRenderBackend {
    public var temporalCadence: NeuralRenderTemporalCadence {
        .frameIndependent
    }

    public func reset(_ request: NeuralRenderResetRequest) async {
        await reset(sequenceID: request.streamID)
    }
}
