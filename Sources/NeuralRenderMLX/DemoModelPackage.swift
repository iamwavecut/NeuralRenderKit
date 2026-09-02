import CryptoKit
import Foundation
import MLX

public enum DemoModelPackageError: Error, Equatable, Sendable {
    case destinationExists(URL)
    case parentIsNotDirectory(URL)
}

public actor DemoModelPackageWriter {
    public init() {}

    public func write(to destinationURL: URL) throws {
        let fileManager = FileManager.default
        let destinationURL = destinationURL.standardizedFileURL
        guard !fileManager.fileExists(atPath: destinationURL.path) else {
            throw DemoModelPackageError.destinationExists(destinationURL)
        }

        let parentURL = destinationURL.deletingLastPathComponent()
        var parentIsDirectory = ObjCBool(false)
        guard fileManager.fileExists(
            atPath: parentURL.path,
            isDirectory: &parentIsDirectory
        ), parentIsDirectory.boolValue else {
            throw DemoModelPackageError.parentIsNotDirectory(parentURL)
        }

        let stagingURL = parentURL.appendingPathComponent(
            ".\(destinationURL.lastPathComponent).\(UUID().uuidString).tmp",
            isDirectory: true
        )
        try fileManager.createDirectory(at: stagingURL, withIntermediateDirectories: false)
        defer { try? fileManager.removeItem(at: stagingURL) }

        let weightsURL = stagingURL.appendingPathComponent("weights.safetensors")
        let scale = MLXArray([Float(2), 0.5, -1])
        let bias = MLXArray([Float(0.1), 0.2, 1])
        try save(
            arrays: ["scale": scale, "bias": bias],
            url: weightsURL,
            stream: .cpu
        )

        let digest = try sha256(url: weightsURL)
        let manifest = try manifestData(digest: digest)
        try manifest.write(
            to: stagingURL.appendingPathComponent("manifest.json"),
            options: .atomic
        )
        try fileManager.moveItem(at: stagingURL, to: destinationURL)
    }

    private func sha256(url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 1_048_576), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private func manifestData(digest: String) throws -> Data {
        let manifest: [String: Any] = [
            "schemaVersion": 1,
            "identifier": "org.neuralrenderkit.demo.pixel-affine",
            "architecture": "nrk.pixel-affine.v1",
            "inputs": [[
                "name": "color",
                "dataType": "float32",
                "layout": "nhwc",
                "shape": [1, "height", "width", 3],
            ]],
            "outputs": [[
                "name": "color",
                "dataType": "float32",
                "layout": "nhwc",
                "shape": [1, "height", "width", 3],
            ]],
            "state": ["kind": "stateless"],
            "weights": [
                "file": "weights.safetensors",
                "sha256": digest,
                "tensors": [
                    ["name": "scale", "dataType": "float32", "shape": [3]],
                    ["name": "bias", "dataType": "float32", "shape": [3]],
                ],
            ],
        ]

        var data = try JSONSerialization.data(
            withJSONObject: manifest,
            options: [.prettyPrinted, .sortedKeys]
        )
        data.append(0x0a)
        return data
    }
}
