import Foundation
import DLSSCore

enum InspectCommand {
    static func run(arguments: [String]) throws {
        guard arguments.count == 1 else {
            throw CLIError.usage("inspect requires one model-package path")
        }

        let package = try ModelPackageLoader.load(
            url: URL(fileURLWithPath: arguments[0])
        )
        try CLIOutput.writeJSON([
            "schemaVersion": package.manifest.schemaVersion,
            "identifier": package.manifest.identifier,
            "architecture": package.manifest.architecture,
            "state": package.manifest.state.kind.rawValue,
            "stateCadence": package.manifest.state.cadence.rawValue,
            "stateResetPolicy": package.manifest.state.resetPolicy.rawValue,
            "stateTensorCount": package.manifest.state.tensors.count,
            "weightCount": package.manifest.weights.tensors.count,
        ])
    }
}
