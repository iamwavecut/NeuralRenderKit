import MLX

public struct MLXMemorySnapshot: Equatable, Sendable {
    public let activeBytes: UInt64
    public let cacheBytes: UInt64
    public let peakActiveBytes: UInt64

    init(_ snapshot: Memory.Snapshot) {
        self.activeBytes = UInt64(max(0, snapshot.activeMemory))
        self.cacheBytes = UInt64(max(0, snapshot.cacheMemory))
        self.peakActiveBytes = UInt64(max(0, snapshot.peakMemory))
    }
}

public enum MLXRuntimeDiagnosticsError: Error, Equatable, Sendable {
    case cacheLimitTooLarge(UInt64)
}

public enum MLXRuntimeDiagnostics {
    public static func memorySnapshot() -> MLXMemorySnapshot {
        MLXMemorySnapshot(Memory.snapshot())
    }

    public static func resetPeakMemory() {
        Memory.peakMemory = 0
    }

    public static var cacheLimitBytes: UInt64 {
        UInt64(max(0, Memory.cacheLimit))
    }

    public static func setCacheLimitBytes(_ bytes: UInt64) throws {
        guard bytes <= UInt64(Int.max) else {
            throw MLXRuntimeDiagnosticsError.cacheLimitTooLarge(bytes)
        }
        Memory.cacheLimit = Int(bytes)
    }
}
