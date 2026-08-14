//! Calibration utilities and reference data (e.g. ColorChecker Classic).
//!
//! Reference values below use the manufacturer's sRGB patch colors for the
//! ColorChecker Classic, converted to linear RGB at runtime before going to
//! density.
//!
//! `.c41` profile format: a zip file containing `profile.json` (CalibrationProfile)
//! and `lut.cube` (3D LUT generated from the matrix) for use in Process mode.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use zip::write::SimpleFileOptions;
use zip::ZipArchive;
use zip::ZipWriter;

use crate::curve;
use crate::lut3d;

/// Manufacturer sRGB patch colors for ColorChecker Classic (24 patches).
///
/// Order is row-major (4 rows × 6 columns), each element is `[R, G, B]` in
/// 8-bit sRGB. These are converted to linear RGB on the fly.
pub const COLORCHECKER_CLASSIC_SRGB_8BIT: [[u8; 3]; 24] = [
    // Row 1: natural colors
    [0x73, 0x52, 0x44], // Dark skin
    [0xc2, 0x96, 0x82], // Light skin
    [0x62, 0x7a, 0x9d], // Blue sky
    [0x57, 0x6c, 0x43], // Foliage
    [0x85, 0x80, 0xb1], // Blue flower
    [0x67, 0xbd, 0xaa], // Bluish green
    // Row 2: miscellaneous colors
    [0xd6, 0x7e, 0x2c], // Orange
    [0x50, 0x5b, 0xa6], // Purplish blue
    [0xc1, 0x5a, 0x63], // Moderate red
    [0x5e, 0x3c, 0x6c], // Purple
    [0x9d, 0xbc, 0x40], // Yellow green
    [0xe0, 0xa3, 0x2e], // Orange yellow
    // Row 3: primary / secondary
    [0x38, 0x3d, 0x96], // Blue
    [0x46, 0x94, 0x49], // Green
    [0xaf, 0x36, 0x3c], // Red
    [0xe7, 0xc7, 0x1f], // Yellow
    [0xbb, 0x56, 0x95], // Magenta
    [0x08, 0x85, 0xa1], // Cyan
    // Row 4: grayscale
    [0xf3, 0xf3, 0xf3], // White
    [0xc8, 0xc8, 0xc8], // Neutral 8
    [0xa0, 0xa0, 0xa0], // Neutral 6.5
    [0x7a, 0x7a, 0x7a], // Neutral 5
    [0x55, 0x55, 0x55], // Neutral 3.5
    [0x34, 0x34, 0x34], // Black
];

#[inline]
fn srgb8_to_linear(c: u8) -> f32 {
    let v = c as f32 / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

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

/// Convert the hardcoded ColorChecker reference patch sRGB values into
/// density space (24×3), via sRGB → linear → density.
pub fn reference_density_24() -> [[f32; 3]; 24] {
    let mut out = [[0.0_f32; 3]; 24];
    for i in 0..24 {
        for c in 0..3 {
            let lin = srgb8_to_linear(COLORCHECKER_CLASSIC_SRGB_8BIT[i][c]);
            out[i][c] = curve::transmittance_to_density(lin, 1e-6);
        }
    }
    out
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

/// Save a calibration profile as a `.c41` zip (profile.json + lut.cube).
/// LUT is generated from the profile matrix (17³, d_max 4.0).
pub fn save_c41_profile(
    profile: &CalibrationProfile,
    path: &Path,
) -> anyhow::Result<()> {
    let lut = lut3d::Lut3d::generate_from_matrix(&profile.matrix, 17, 4.0);
    let cube_content = lut3d::cube_to_string(&lut);
    let json_content = serde_json::to_string_pretty(profile)?;

    let file = File::create(path)?;
    let mut zip = ZipWriter::new(file);
    let opts = SimpleFileOptions::default().unix_permissions(0o644);

    zip.start_file("profile.json", opts)?;
    zip.write_all(json_content.as_bytes())?;
    zip.start_file("lut.cube", opts)?;
    zip.write_all(cube_content.as_bytes())?;
    zip.finish()?;
    Ok(())
}

/// Load a `.c41` zip profile. Extracts to `parent_of_path/.cache/<stem>/` and returns
/// the profile plus the path to the extracted `lut.cube` for use as `lut3d_path`.
pub fn load_c41_profile(path: &Path) -> anyhow::Result<(CalibrationProfile, std::path::PathBuf)> {
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)?;

    let parent = path
        .parent()
        .unwrap_or_else(|| path.as_ref());
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("profile");
    let cache_dir = parent.join(".cache").join(stem);
    std::fs::create_dir_all(&cache_dir)?;

    archive.extract(&cache_dir)?;

    let profile_path = cache_dir.join("profile.json");
    let lut_path = cache_dir.join("lut.cube");
    let json_content = std::fs::read_to_string(&profile_path)
        .map_err(|e| anyhow::anyhow!("Missing or unreadable profile.json in .c41: {}", e))?;
    if !lut_path.exists() {
        anyhow::bail!("Missing lut.cube in .c41");
    }
    let profile: CalibrationProfile = serde_json::from_str(&json_content)
        .map_err(|e| anyhow::anyhow!("Invalid profile.json in .c41: {}", e))?;
    Ok((profile, lut_path))
}

/// Load all `.json` and `.c41` calibration profiles from a directory.
///
/// Returns a vector of `(path, profile, optional_lut_path)`.
/// For .json the third element is None; for .c41 it is the path to the extracted lut.cube.
pub fn load_profiles_from_dir(
    dir: &std::path::Path,
) -> anyhow::Result<Vec<(std::path::PathBuf, CalibrationProfile, Option<std::path::PathBuf>)>> {
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
        let ext = path.extension().and_then(|e| e.to_str());
        if ext == Some("json") {
            let text = match std::fs::read_to_string(&path) {
                Ok(t) => t,
                Err(_) => continue,
            };
            if let Ok(profile) = serde_json::from_str::<CalibrationProfile>(&text) {
                out.push((path, profile, None));
            }
        } else if ext == Some("c41") {
            if let Ok((profile, lut_path)) = load_c41_profile(&path) {
                out.push((path, profile, Some(lut_path)));
            }
        }
    }
    Ok(out)
}


