import Foundation
import XCTest

final class VerificationScriptTests: XCTestCase {
    func testVerificationScriptRunsPortableChecksInCopiedTree() throws {
        let source = repositoryRoot()
        let destination = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: destination) }
        try FileManager.default.createDirectory(
            at: destination,
            withIntermediateDirectories: false
        )
        try copyPublicTree(from: source, to: destination)
        let initialize = try runProcess(
            executable: URL(fileURLWithPath: "/usr/bin/git"),
            arguments: ["init", "--quiet"],
            currentDirectory: destination
        )
        XCTAssertEqual(initialize.status, 0, initialize.stderr)

        let result = try runProcess(
            executable: destination.appendingPathComponent("scripts/verify.sh"),
            arguments: [],
            currentDirectory: destination,
            environment: [
                "NRK_SKIP_NESTED_SWIFT_TEST": "1",
                "NRK_SKIP_BUILD_AND_INFERENCE": "1",
            ]
        )

        XCTAssertEqual(result.status, 0, result.stdout + result.stderr)
        XCTAssertTrue(result.stdout.contains("architecture: arm64"))
        XCTAssertTrue(result.stdout.contains("public tree audit: clean"))
        XCTAssertTrue(result.stdout.contains("verification: passed"))
    }

    private func copyPublicTree(from source: URL, to destination: URL) throws {
        let listing = try runProcess(
            executable: URL(fileURLWithPath: "/usr/bin/git"),
            arguments: ["ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            currentDirectory: source,
            decodeOutput: false
        )
        XCTAssertEqual(listing.status, 0, listing.stderr)
        for relativePath in listing.stdoutData.split(separator: 0) {
            let relativePath = String(decoding: relativePath, as: UTF8.self)
            let sourceURL = source.appendingPathComponent(relativePath)
            let destinationURL = destination.appendingPathComponent(relativePath)
            try FileManager.default.createDirectory(
                at: destinationURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
        }
    }

    private func repositoryRoot() -> URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<3 {
            url.deleteLastPathComponent()
        }
        return url
    }

    private func runProcess(
        executable: URL,
        arguments: [String],
        currentDirectory: URL,
        environment additions: [String: String] = [:],
        decodeOutput: Bool = true
    ) throws -> VerificationProcessResult {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.currentDirectoryURL = currentDirectory
        process.environment = ProcessInfo.processInfo.environment.merging(additions) { _, new in new }
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()
        let stdoutData = stdout.fileHandleForReading.readDataToEndOfFile()
        let stderrData = stderr.fileHandleForReading.readDataToEndOfFile()
        return VerificationProcessResult(
            status: process.terminationStatus,
            stdoutData: stdoutData,
            stdout: decodeOutput ? String(data: stdoutData, encoding: .utf8) ?? "" : "",
            stderr: String(data: stderrData, encoding: .utf8) ?? ""
        )
    }
}

private struct VerificationProcessResult {
    let status: Int32
    let stdoutData: Data
    let stdout: String
    let stderr: String
}
