import MLX

struct PixelAffineModel {
    let scale: MLXArray
    let bias: MLXArray

    func callAsFunction(_ color: MLXArray) -> MLXArray {
        clip(color * scale + bias, min: Float(0), max: Float(1))
    }
}

