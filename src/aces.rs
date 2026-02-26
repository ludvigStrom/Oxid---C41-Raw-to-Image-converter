//! ACES color space utilities: IDT (Input Device Transform) and ACEScg ↔ ACES2065-1.
//!
//! ACEScg uses AP1 primaries (linear); ACES2065-1 uses AP0 primaries (linear).
//! Matrix constants from ACES documentation (ACEScg specification, TRA_2).

use ndarray::Array3;
use nalgebra::Matrix3;

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

/// ACEScg (AP1 linear) → Linear sRGB (Rec. 709 primaries).
/// Apply before the RA-4 curve so display output (TIFF/EXR/JPEG) is standard sRGB.
pub const ACESCG_TO_LINEAR_SRGB: [[f32; 3]; 3] = [
    [1.705_051, -0.621_815, -0.083_236],
    [-0.130_256, 1.140_801, -0.010_545],
    [-0.024_007, -0.128_984, 1.152_991],
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

/// Convert linear ACEScg image to linear sRGB in place.
pub fn linear_acescg_to_linear_srgb(image: &mut Array3<f32>) {
    apply_idt(image, &ACESCG_TO_LINEAR_SRGB);
}

/// Convert u16 image from linear ACEScg to linear sRGB in place.
/// Treats u16 values as linear in [0, 1] (divide by 65535), applies the primaries matrix,
/// clamps, and writes back as u16. Used for curve output when use_acescg so export is sRGB.
pub fn convert_u16_linear_acescg_to_linear_srgb(image: &mut Array3<u16>) {
    let (h, w, _) = image.dim();
    let m = &ACESCG_TO_LINEAR_SRGB;
    let scale = 1.0 / 65535.0f32;
    for y in 0..h {
        for x in 0..w {
            let r = image[[y, x, 0]] as f32 * scale;
            let g = image[[y, x, 1]] as f32 * scale;
            let b = image[[y, x, 2]] as f32 * scale;
            let r_out = (m[0][0] * r + m[0][1] * g + m[0][2] * b).clamp(0.0, 1.0);
            let g_out = (m[1][0] * r + m[1][1] * g + m[1][2] * b).clamp(0.0, 1.0);
            let b_out = (m[2][0] * r + m[2][1] * g + m[2][2] * b).clamp(0.0, 1.0);
            image[[y, x, 0]] = (r_out * 65535.0).round() as u16;
            image[[y, x, 1]] = (g_out * 65535.0).round() as u16;
            image[[y, x, 2]] = (b_out * 65535.0).round() as u16;
        }
    }
}

/// Convert a density-domain calibration matrix from camera space to ACEScg.
/// When the pipeline runs in ACEScg, the density matrix must act in ACEScg;
/// if the profile was solved in camera space, use this so the same visual result is achieved:
/// `M_aces = T · M_cam · T^(-1)` where T is the IDT (camera → ACEScg).
/// If the IDT is singular, returns `m_cam` unchanged.
pub fn convert_density_matrix_to_acescg(
    m_cam: [[f32; 3]; 3],
    idt: &[[f32; 3]; 3],
) -> [[f32; 3]; 3] {
    let t = Matrix3::from_row_slice(&[
        idt[0][0], idt[0][1], idt[0][2],
        idt[1][0], idt[1][1], idt[1][2],
        idt[2][0], idt[2][1], idt[2][2],
    ]);
    let Some(t_inv) = t.try_inverse() else {
        return m_cam;
    };
    let m = Matrix3::from_row_slice(&[
        m_cam[0][0], m_cam[0][1], m_cam[0][2],
        m_cam[1][0], m_cam[1][1], m_cam[1][2],
        m_cam[2][0], m_cam[2][1], m_cam[2][2],
    ]);
    let result = t * m * t_inv;
    [
        [result[(0, 0)], result[(0, 1)], result[(0, 2)]],
        [result[(1, 0)], result[(1, 1)], result[(1, 2)]],
        [result[(2, 0)], result[(2, 1)], result[(2, 2)]],
    ]
}
