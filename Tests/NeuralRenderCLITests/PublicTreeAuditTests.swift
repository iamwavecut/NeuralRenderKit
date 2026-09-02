import Foundation
import XCTest

final class PublicTreeAuditTests: XCTestCase {
    func testRejectsForbiddenBinaryMagicNamesSizesAndLocalPaths() throws {
        let repository = try makeRepository()
        defer { try? FileManager.default.removeItem(at: repository) }

        try Data([0x4d, 0x5a, 0x00]).write(to: repository.appendingPathComponent("windows.bin"))
        try Data([0x7f, 0x45, 0x4c, 0x46]).write(to: repository.appendingPathComponent("elf.bin"))
        try Data([0xcf, 0xfa, 0xed, 0xfe]).write(to: repository.appendingPathComponent("macho.bin"))
        try Data([0x50, 0xed, 0x55, 0xba]).write(to: repository.appendingPathComponent("cuda.bin"))
        try Data(count: 6 * 1_024 * 1_024).write(
            to: repository.appendingPathComponent("oversized.bin")
        )
        try Data("private".utf8).write(to: repository.appendingPathComponent("vendor.dll"))
        try Data("private".utf8).write(
            to: repository.appendingPathComponent("model-private-oracle-notes.txt")
        )
        let localPath = "/" + "Users/wavecut/private/model"
        try Data("let path = \"\(localPath)\"".utf8).write(
            to: repository.appendingPathComponent("LocalPath.swift")
        )
        try Data([0, 1, 2]).write(
            to: repository.appendingPathComponent("weights.safetensors")
        )

        let result = try runAudit(repository: repository)

        XCTAssertEqual(result.status, 1, result.stderr)
        for path in [
            "windows.bin", "elf.bin", "macho.bin", "cuda.bin", "oversized.bin",
            "vendor.dll", "model-private-oracle-notes.txt", "LocalPath.swift",
            "weights.safetensors",
        ] {
            XCTAssertTrue(result.stdout.contains(path), "missing \(path) in \(result.stdout)")
        }
        XCTAssertFalse(result.stdout.contains(localPath))
    }

    func testAcceptsPortableSourcesAndSmallAllowlistedFixtures() throws {
        let repository = try makeRepository()
        defer { try? FileManager.default.removeItem(at: repository) }
        let fixtureDirectory = repository.appendingPathComponent("Tests/Fixtures/Public")
        try FileManager.default.createDirectory(
            at: fixtureDirectory,
            withIntermediateDirectories: true
        )
        try Data("public struct Example {}\n".utf8).write(
            to: repository.appendingPathComponent("Example.swift")
        )
        try Data("{\"schemaVersion\":1}\n".utf8).write(
            to: repository.appendingPathComponent("fixture.json")
        )
        try Data([8, 0, 0, 0, 0, 0, 0, 0]).write(
            to: fixtureDirectory.appendingPathComponent("tiny.safetensors")
        )

        let result = try runAudit(repository: repository)

        XCTAssertEqual(result.status, 0, result.stdout + result.stderr)
        XCTAssertTrue(result.stdout.contains("public tree audit: clean"))
    }

    func testRejectsLocalPathEvenInsideDocumentationFence() throws {
        let repository = try makeRepository()
        defer { try? FileManager.default.removeItem(at: repository) }
        let localPath = "/" + "Users/example/project"
        let documentation = "```sh\ncd \(localPath)\n```\n"
        try Data(documentation.utf8).write(
            to: repository.appendingPathComponent("README.md")
        )

        let result = try runAudit(repository: repository)

        XCTAssertEqual(result.status, 1)
        XCTAssertTrue(result.stdout.contains("README.md"))
        XCTAssertFalse(result.stdout.contains(localPath))
    }

    private func makeRepository() throws -> URL {
        let repository = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(
            at: repository,
            withIntermediateDirectories: false
        )
        let result = try runProcess(
            executable: URL(fileURLWithPath: "/usr/bin/git"),
            arguments: ["init", "--quiet"],
            currentDirectory: repository
        )
        XCTAssertEqual(result.status, 0, result.stderr)
        return repository
    }

    private func runAudit(repository: URL) throws -> AuditProcessResult {
        try runProcess(
            executable: repositoryRoot().appendingPathComponent("scripts/audit-public-tree.sh"),
            arguments: [repository.path],
            currentDirectory: repository
        )
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
        currentDirectory: URL
    ) throws -> AuditProcessResult {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.currentDirectoryURL = currentDirectory
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()
        return AuditProcessResult(
            status: process.terminationStatus,
            stdout: String(
                data: stdout.fileHandleForReading.readDataToEndOfFile(),
                encoding: .utf8
            ) ?? "",
            stderr: String(
                data: stderr.fileHandleForReading.readDataToEndOfFile(),
                encoding: .utf8
            ) ?? ""
        )
    }
}

private struct AuditProcessResult {
    let status: Int32
    let stdout: String
    let stderr: String
}
