//! Calibration utilities and reference data (e.g. ColorChecker Classic).
//!
//! NOTE: The reference values below are placeholders and should be replaced
//! with measured or published linear RGB values for the chart.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::curve;

/// Reference linear RGB values for the 24 patches of a ColorChecker Classic.
///
/// Order is row-major (4 rows × 6 columns), each element is `[R, G, B]` in
/// linear [0, 1] for the chosen working space / illuminant.
pub const COLORCHECKER_CLASSIC_LINEAR_RGB: [[f32; 3]; 24] = [
    [0.0, 0.0, 0.0]; 24
];

/// Serializable calibration profile: 3×3 density matrix + metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationProfile {
    /// Human-friendly name, e.g. "Kodak Gold 200".
    pub name: String,
    /// Free-form notes about the light source / setup.
    pub light_source: String,
    /// 3×3 density-domain matrix.
    pub matrix: [[f32; 3]; 3],
    /// Optional D-min medians (R,G,B) used during calibration.
    pub dmin_medians: Option<(f32, f32, f32)>,
}

/// Convert an array of 24 linear RGB patches to density using the same
/// `D = -log10(T)` definition as the main curve pipeline.
pub fn linear_to_density_24(patches_linear: [[f32; 3]; 24]) -> [[f32; 3]; 24] {
    let mut out = [[0.0_f32; 3]; 24];
    for i in 0..24 {
        for c in 0..3 {
            out[i][c] = curve::transmittance_to_density(patches_linear[i][c], 1e-6);
        }
    }
    out
}

/// Convert the hardcoded ColorChecker reference linear RGB values into
/// density space (24×3).
pub fn reference_density_24() -> [[f32; 3]; 24] {
    linear_to_density_24(COLORCHECKER_CLASSIC_LINEAR_RGB)
}

/// Solve for the 3×3 density-domain calibration matrix `M` that best maps
/// measured densities `X` to reference densities `Y` in a least-squares sense:
///
/// `M = (X^T X)^-1 X^T Y`
///
/// Returns `(M, mse)` where `mse` is the mean squared error of `X * M` vs `Y`,
/// or `None` if the system is singular.
pub fn solve_density_matrix_ols(
    measured_density: [[f32; 3]; 24],
    reference_density: [[f32; 3]; 24],
) -> Option<([[f32; 3]; 3], f32)> {
    // Build X and Y as 24×3 dynamic matrices.
    let mut x = DMatrix::<f32>::zeros(24, 3);
    let mut y = DMatrix::<f32>::zeros(24, 3);
    for i in 0..24 {
        for c in 0..3 {
            x[(i, c)] = measured_density[i][c];
            y[(i, c)] = reference_density[i][c];
        }
    }

    let xt = x.transpose();
    let xtx = &xt * &x; // 3×3
    let xty = &xt * &y; // 3×3

    let inv = match xtx.try_inverse() {
        Some(m) => m,
        None => return None,
    };

    let m_mat = inv * xty; // 3×3 dynamic matrix

    // Compute MSE of X*M vs Y.
    let pred = &x * &m_mat;
    let mut mse = 0.0_f32;
    for i in 0..24 {
        for c in 0..3 {
            let d = pred[(i, c)] - y[(i, c)];
            mse += d * d;
        }
    }
    mse /= (24 * 3) as f32;

    let mut m_array = [[0.0_f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            m_array[r][c] = m_mat[(r, c)];
        }
    }

    Some((m_array, mse))
}

/// Save a calibration profile as pretty-printed JSON to the given path.
pub fn save_profile_to_path(
    profile: &CalibrationProfile,
    path: &std::path::Path,
) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(profile)?;
    std::fs::write(path, json)?;
    Ok(())
}

