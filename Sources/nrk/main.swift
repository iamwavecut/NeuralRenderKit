import Foundation
import NeuralRenderMLX

enum CLIError: Error, Sendable {
    case usage(String)
    case missingOutput(String)
    case destinationExists(URL)
}

enum CLIOutput {
    static func writeJSON(_ object: [String: Any]) throws {
        var data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys]
        )
        data.append(0x0a)
        FileHandle.standardOutput.write(data)
    }

    static func writeEncodable<T: Encodable>(_ value: T) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        var data = try encoder.encode(value)
        data.append(0x0a)
        FileHandle.standardOutput.write(data)
    }

    static func writeError(_ error: any Error) {
        let message: String
        switch error {
        case let CLIError.usage(detail):
            message = "usage error: \(detail)"
        case let CLIError.missingOutput(name):
            message = "missing output: \(name)"
        case let CLIError.destinationExists(url):
            message = "destination already exists: \(url.path)"
        case let DemoModelPackageError.destinationExists(url):
            message = "destination already exists: \(url.path)"
        case let DemoModelPackageError.parentIsNotDirectory(url):
            message = "parent is not a directory: \(url.path)"
        default:
            message = String(describing: error)
        }
        FileHandle.standardError.write(Data("error: \(message)\n".utf8))
    }
}

func runCommand(_ arguments: [String]) async throws {
    guard let command = arguments.first else {
        throw CLIError.usage(
            "expected inspect, run, run-sequence, render-image, or stream"
        )
    }
    let commandArguments = Array(arguments.dropFirst())
    switch command {
    case "render-image":
        try await RenderImageCommand.run(arguments: commandArguments)
    case "inspect":
        try InspectCommand.run(arguments: commandArguments)
    case "run":
        try await RunCommand.run(arguments: commandArguments)
    case "run-sequence":
        try await RunSequenceCommand.run(arguments: commandArguments)
    case "stream":
        try await StreamCommand.run(arguments: commandArguments)
    default:
        throw CLIError.usage("unknown command '\(command)'")
    }
}

do {
    try await runCommand(Array(CommandLine.arguments.dropFirst()))
} catch {
    CLIOutput.writeError(error)
    exit(2)
}
