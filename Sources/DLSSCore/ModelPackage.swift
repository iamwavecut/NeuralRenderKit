import CryptoKit
import Foundation

public struct LoadedModelPackage: Equatable, Sendable {
    public let url: URL
    public let manifest: ModelPackageManifest
    public let weightsURL: URL
}

public enum ModelPackageError: Error, Equatable, Sendable {
    case packageIsNotDirectory
    case missingManifest
    case manifestTooLarge(maxBytes: Int)
    case unsafeWeightFile(String)
    case missingWeightFile(String)
    case invalidDigest(String)
    case digestMismatch(expected: String, actual: String)
}

public enum ModelPackageLoader {
    public static let maximumManifestBytes = 1_048_576

    public static func load(url: URL) throws -> LoadedModelPackage {
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            throw ModelPackageError.packageIsNotDirectory
        }

        let manifestURL = url.appendingPathComponent("manifest.json", isDirectory: false)
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw ModelPackageError.missingManifest
        }
        let manifestData = try readManifest(url: manifestURL)
        let manifest = try ModelPackageManifest.decode(data: manifestData)

        let weightFile = manifest.weights.file
        guard isSafeFileName(weightFile) else {
            throw ModelPackageError.unsafeWeightFile(weightFile)
        }
        guard isValidDigest(manifest.weights.sha256) else {
            throw ModelPackageError.invalidDigest(manifest.weights.sha256)
        }

        let weightsURL = url.appendingPathComponent(weightFile, isDirectory: false)
        var weightsAreDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(
            atPath: weightsURL.path,
            isDirectory: &weightsAreDirectory
        ), !weightsAreDirectory.boolValue else {
            throw ModelPackageError.missingWeightFile(weightFile)
        }

        let actualDigest = try sha256(url: weightsURL)
        guard actualDigest == manifest.weights.sha256 else {
            throw ModelPackageError.digestMismatch(
                expected: manifest.weights.sha256,
                actual: actualDigest
            )
        }

        return LoadedModelPackage(
            url: url,
            manifest: manifest,
            weightsURL: weightsURL
        )
    }

    private static func readManifest(url: URL) throws -> Data {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        let data = try handle.read(upToCount: maximumManifestBytes + 1) ?? Data()
        guard data.count <= maximumManifestBytes else {
            throw ModelPackageError.manifestTooLarge(maxBytes: maximumManifestBytes)
        }
        return data
    }

    private static func isSafeFileName(_ value: String) -> Bool {
        guard !value.isEmpty,
              value != ".",
              value != "..",
              !value.contains("/"),
              !value.contains("\\"),
              !(value as NSString).isAbsolutePath
        else {
            return false
        }
        return true
    }

    private static func isValidDigest(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy { byte in
            (byte >= 48 && byte <= 57) || (byte >= 97 && byte <= 102)
        }
    }

    private static func sha256(url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 1_048_576), !chunk.isEmpty {
            hasher.update(data: chunk)
        }

        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

