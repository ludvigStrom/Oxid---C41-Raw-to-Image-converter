//! ACES color space utilities: IDT (Input Device Transform) and ACEScg ↔ ACES2065-1.
//!
//! ACEScg uses AP1 primaries (linear); ACES2065-1 uses AP0 primaries (linear).
//! Matrix constants from ACES documentation (ACEScg specification, TRA_2).

use ndarray::Array3;

/// 3×3 identity matrix for IDT (no color transform).
pub const IDT_IDENTITY: [[f32; 3]; 3] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

/// ACEScg (AP1 linear) → ACES2065-1 (AP0 linear). TRA_2 from ACES spec.
/// Source: ACEScg specification, "Converting ACEScg RGB values to ACES2065-1 RGB values".
const ACESCG_TO_ACES2065_1: [[f32; 3]; 3] = [
    [0.695_452_241_4, 0.140_678_696_5, 0.163_869_062_2],
    [0.044_794_563_4, 0.859_671_118_5, 0.095_534_318_2],
    [-0.005_525_882_6, 0.004_025_210_3, 1.001_500_672_3],
];

/// Apply a 3×3 linear transform to each RGB pixel in place.
/// `matrix` is row-major: row i is applied to produce channel i.
pub fn apply_idt(image: &mut Array3<f32>, matrix: &[[f32; 3]; 3]) {
    let (h, w, _) = image.dim();
    for y in 0..h {
        for x in 0..w {
            let r = image[[y, x, 0]];
            let g = image[[y, x, 1]];
            let b = image[[y, x, 2]];
            image[[y, x, 0]] = matrix[0][0] * r + matrix[0][1] * g + matrix[0][2] * b;
            image[[y, x, 1]] = matrix[1][0] * r + matrix[1][1] * g + matrix[1][2] * b;
            image[[y, x, 2]] = matrix[2][0] * r + matrix[2][1] * g + matrix[2][2] * b;
        }
    }
}

/// Convert linear ACEScg image to ACES2065-1 in place.
/// Input and output are (H, W, 3) linear RGB.
pub fn linear_acescg_to_aces2065_1(image: &mut Array3<f32>) {
    apply_idt(image, &ACESCG_TO_ACES2065_1);
}
