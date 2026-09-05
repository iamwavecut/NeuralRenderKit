// swift-tools-version: 6.2
import PackageDescription

let package = Package(
  name: "MLXDLSS",
  platforms: [
    .macOS(.v14)
  ],
  products: [
    .library(name: "DLSSCore", targets: ["DLSSCore"]),
    .library(name: "DLSSCoreML", targets: ["DLSSCoreML"]),
    .library(name: "DLSSMLX", targets: ["DLSSMLX"]),
    .executable(name: "mlxdlss", targets: ["mlxdlss"]),
  ],
  dependencies: [
    .package(
      url: "https://github.com/ml-explore/mlx-swift",
      exact: "0.31.6"
    )
  ],
  targets: [
    .target(
      name: "DLSSCore",
      linkerSettings: [
        .linkedFramework("Accelerate"),
        .linkedFramework("Metal"),
      ]
    ),
    .target(
      name: "DLSSCoreML",
      dependencies: ["DLSSCore"],
      linkerSettings: [.linkedFramework("CoreML")]
    ),
    .target(
      name: "DLSSMLX",
      dependencies: [
        "DLSSCore",
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
      ]
    ),
    .executableTarget(
      name: "mlxdlss",
      dependencies: [
        "DLSSCore",
        "DLSSCoreML",
        "DLSSMLX",
      ],
      linkerSettings: [
        .linkedFramework("CoreGraphics"),
        .linkedFramework("ImageIO"),
      ]
    ),
    .testTarget(
      name: "DLSSCoreTests",
      dependencies: ["DLSSCore"]
    ),
    .testTarget(
      name: "DLSSCoreMLTests",
      dependencies: [
        "DLSSCore",
        "DLSSCoreML",
      ]
    ),
    .testTarget(
      name: "DLSSMLXTests",
      dependencies: [
        "DLSSCore",
        "DLSSMLX",
        .product(name: "MLX", package: "mlx-swift"),
      ]
    ),
    .testTarget(
      name: "MLXDLSSCLITests",
      dependencies: ["mlxdlss", "DLSSMLX"]
    ),
  ]
)
