//! ACES color space utilities: IDT (Input Device Transform) and ACEScg ↔ ACES2065-1.
//!
//! ACEScg uses AP1 primaries (linear); ACES2065-1 uses AP0 primaries (linear).
//! Matrix constants from ACES documentation (ACEScg specification, TRA_2).

use std::path::Path;

use nalgebra::Matrix3;
use ndarray::Array3;
use serde::{Deserialize, Serialize};

/// Camera IDT profile: name and 3×3 matrix (linear camera RGB → ACEScg).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdtProfile {
    /// Display name, e.g. "Sony A7R II".
    pub name: String,
    /// 3×3 row-major matrix (camera linear → ACEScg).
    pub matrix: [[f32; 3]; 3],
}

/// Load all `.json` IDT profiles from a directory (e.g. `camera_idt/`).
/// Returns `(path, profile)`; files that fail to parse are skipped.
pub fn load_idt_profiles_from_dir(
    dir: &Path,
) -> anyhow::Result<Vec<(std::path::PathBuf, IdtProfile)>> {
    let mut out = Vec::new();
    if !dir.exists() {
        return Ok(out);
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(_) => continue,
        };
        match serde_json::from_str::<IdtProfile>(&text) {
            Ok(profile) => out.push((path, profile)),
            Err(_) => continue,
        }
    }
    Ok(out)
}

/// 3×3 identity matrix for IDT (no color transform).
pub const IDT_IDENTITY: [[f32; 3]; 3] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

/// Returns true when a 3×3 matrix is (approximately) the identity.
/// Used to detect the default IDT so we can skip the ACEScg→sRGB primaries conversion
/// (camera RGB ≈ sRGB for most DSLRs; treating it as ACEScg is wrong).
pub fn is_identity(m: &[[f32; 3]; 3]) -> bool {
    const EPS: f32 = 1e-6;
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            if (m[i][j] - expected).abs() > EPS {
                return false;
            }
        }
    }
    true
}

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
/// clamps, and writes back as u16. Used for curve output so export is sRGB.
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

/// Convert linear Rec.709 / sRGB to linear ACEScg in place.
pub fn linear_srgb_to_acescg(image: &mut Array3<f32>) {
    let s = Matrix3::from_row_slice(&[
        ACESCG_TO_LINEAR_SRGB[0][0], ACESCG_TO_LINEAR_SRGB[0][1], ACESCG_TO_LINEAR_SRGB[0][2],
        ACESCG_TO_LINEAR_SRGB[1][0], ACESCG_TO_LINEAR_SRGB[1][1], ACESCG_TO_LINEAR_SRGB[1][2],
        ACESCG_TO_LINEAR_SRGB[2][0], ACESCG_TO_LINEAR_SRGB[2][1], ACESCG_TO_LINEAR_SRGB[2][2],
    ]);
    let Some(inv) = s.try_inverse() else {
        return;
    };
    let m = [
        [inv[(0, 0)], inv[(0, 1)], inv[(0, 2)]],
        [inv[(1, 0)], inv[(1, 1)], inv[(1, 2)]],
        [inv[(2, 0)], inv[(2, 1)], inv[(2, 2)]],
    ];
    apply_idt(image, &m);
}

/// Convert linear print RGB toward ACES2065-1.
///
/// * When `working_space_is_rec709` (a real IDT was applied), Rec.709 → ACEScg → AP0.
/// * When the buffer is still camera RGB, the ACEScg→AP0 matrix is **not** applied
///   (that would treat camera primaries as AP1 and shift color, often toward magenta).
///   The image is left as linear camera/print RGB.
pub fn linear_print_to_aces2065_1(image: &mut Array3<f32>, working_space_is_rec709: bool) {
    if !working_space_is_rec709 {
        return;
    }
    linear_srgb_to_acescg(image);
    linear_acescg_to_aces2065_1(image);
}

/// Convert a density-domain calibration matrix from camera space to linear sRGB.
/// Combined transform: camera → ACEScg (IDT) → linear sRGB, conjugated around the density matrix.
/// `M_srgb = (S·T) · M_cam · (S·T)^(-1)` where S = ACEScg→sRGB and T = IDT.
/// For identity IDT and identity density matrix, result is identity.
pub fn convert_density_matrix_to_srgb(
    m_cam: [[f32; 3]; 3],
    idt: &[[f32; 3]; 3],
) -> [[f32; 3]; 3] {
    let t = Matrix3::from_row_slice(&[
        idt[0][0], idt[0][1], idt[0][2],
        idt[1][0], idt[1][1], idt[1][2],
        idt[2][0], idt[2][1], idt[2][2],
    ]);
    let s = Matrix3::from_row_slice(&[
        ACESCG_TO_LINEAR_SRGB[0][0], ACESCG_TO_LINEAR_SRGB[0][1], ACESCG_TO_LINEAR_SRGB[0][2],
        ACESCG_TO_LINEAR_SRGB[1][0], ACESCG_TO_LINEAR_SRGB[1][1], ACESCG_TO_LINEAR_SRGB[1][2],
        ACESCG_TO_LINEAR_SRGB[2][0], ACESCG_TO_LINEAR_SRGB[2][1], ACESCG_TO_LINEAR_SRGB[2][2],
    ]);
    let combined = s * t;
    let Some(combined_inv) = combined.try_inverse() else {
        return m_cam;
    };
    let m = Matrix3::from_row_slice(&[
        m_cam[0][0], m_cam[0][1], m_cam[0][2],
        m_cam[1][0], m_cam[1][1], m_cam[1][2],
        m_cam[2][0], m_cam[2][1], m_cam[2][2],
    ]);
    let result = combined * m * combined_inv;
    [
        [result[(0, 0)], result[(0, 1)], result[(0, 2)]],
        [result[(1, 0)], result[(1, 1)], result[(1, 2)]],
        [result[(2, 0)], result[(2, 1)], result[(2, 2)]],
    ]
}
