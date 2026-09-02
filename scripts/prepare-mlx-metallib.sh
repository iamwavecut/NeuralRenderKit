#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLX_SOURCE="$PROJECT_ROOT/.build/checkouts/mlx-swift/Source/Cmlx/mlx"
METAL_BUILD="$PROJECT_ROOT/.build/nrk-mlx-metallib"
METALLIB="$METAL_BUILD/mlx/backend/metal/kernels/mlx.metallib"
MIN_MACOS_VERSION="${NRK_MIN_MACOS_VERSION:-14.0}"
BUILD_JOBS="${NRK_BUILD_JOBS:-2}"

for command_name in cmake ninja xcrun swift; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command is unavailable: $command_name" >&2
    exit 69
  fi
done

if [[ ! -f "$MLX_SOURCE/CMakeLists.txt" ]]; then
  swift package --package-path "$PROJECT_ROOT" resolve
fi
if [[ ! -f "$MLX_SOURCE/CMakeLists.txt" ]]; then
  echo "MLX sources were not resolved at $MLX_SOURCE" >&2
  exit 66
fi

swift build --package-path "$PROJECT_ROOT" --build-tests --jobs "$BUILD_JOBS"

cmake \
  -S "$MLX_SOURCE" \
  -B "$METAL_BUILD" \
  -G Ninja \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_BUILD_GGUF=OFF \
  -DMLX_BUILD_SAFETENSORS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$MIN_MACOS_VERSION"

cmake --build "$METAL_BUILD" --target mlx-metallib --parallel "$BUILD_JOBS"

if [[ ! -s "$METALLIB" ]]; then
  echo "MLX metallib was not produced at $METALLIB" >&2
  exit 70
fi

if [[ "$#" -gt 0 ]]; then
  destinations=("$@")
else
  bin_path="$(swift build --package-path "$PROJECT_ROOT" --show-bin-path)"
  destinations=("$bin_path")
  while IFS= read -r destination; do
    destinations+=("$destination")
  done < <(find "$bin_path" -type d -path '*.xctest/Contents/MacOS' -print)
fi

for destination in "${destinations[@]}"; do
  mkdir -p "$destination"
  cp "$METALLIB" "$destination/mlx.metallib"
done

echo "$METALLIB"
