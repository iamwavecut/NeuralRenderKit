// swift-tools-version: 6.2
import PackageDescription

let package = Package(
  name: "NeuralRenderKit",
  platforms: [
    .macOS(.v14)
  ],
  products: [
    .library(name: "NeuralRenderCore", targets: ["NeuralRenderCore"]),
    .library(name: "NeuralRenderCoreML", targets: ["NeuralRenderCoreML"]),
    .library(name: "NeuralRenderMLX", targets: ["NeuralRenderMLX"]),
    .executable(name: "nrk", targets: ["nrk"]),
  ],
  dependencies: [
    .package(
      url: "https://github.com/ml-explore/mlx-swift",
      exact: "0.31.6"
    )
  ],
  targets: [
    .target(
      name: "NeuralRenderCore",
      linkerSettings: [
        .linkedFramework("Accelerate"),
        .linkedFramework("Metal"),
      ]
    ),
    .target(
      name: "NeuralRenderCoreML",
      dependencies: ["NeuralRenderCore"],
      linkerSettings: [.linkedFramework("CoreML")]
    ),
    .target(
      name: "NeuralRenderMLX",
      dependencies: [
        "NeuralRenderCore",
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
      ]
    ),
    .executableTarget(
      name: "nrk",
      dependencies: [
        "NeuralRenderCore",
        "NeuralRenderCoreML",
        "NeuralRenderMLX",
      ],
      linkerSettings: [
        .linkedFramework("CoreGraphics"),
        .linkedFramework("ImageIO"),
      ]
    ),
    .testTarget(
      name: "NeuralRenderCoreTests",
      dependencies: ["NeuralRenderCore"]
    ),
    .testTarget(
      name: "NeuralRenderCoreMLTests",
      dependencies: [
        "NeuralRenderCore",
        "NeuralRenderCoreML",
      ]
    ),
    .testTarget(
      name: "NeuralRenderMLXTests",
      dependencies: [
        "NeuralRenderCore",
        "NeuralRenderMLX",
        .product(name: "MLX", package: "mlx-swift"),
      ]
    ),
    .testTarget(
      name: "NeuralRenderCLITests",
      dependencies: ["nrk", "NeuralRenderMLX"]
    ),
  ]
)
