//! Negative → positive inversion for C-41.
//!
//! Transmittance T is inverted in **density space** so that the film's color character
//! (e.g. orange base, orange lettering) is preserved. Simple 1−T would invert to the
//! complement (orange → teal); density inversion inverts brightness only.

use ndarray::Array3;

const T_MIN: f32 = 1e-6;
/// Maximum density for normalization; matches typical print-curve range.
const D_MAX: f32 = 3.0;

/// Invert in-place via density: D = -log10(T), display = 1 - D/D_max.
///
/// Preserves color: orange base stays orange, only luminance is inverted (negative → positive).
/// Use this for the no-curve path so film border lettering and base look correct.
pub fn invert_density(image: &mut Array3<f32>) {
    image.mapv_inplace(|t| {
        let t = t.clamp(T_MIN, 1.0);
        let d = -t.log10();
        let d_norm = (d / D_MAX).min(1.0);
        1.0 - d_norm
    });
}

/// Legacy: per-channel 1.0 - value (inverts color to complement; orange → teal).
/// Prefer `invert_density` for film.
#[allow(dead_code)]
pub fn invert(image: &mut Array3<f32>) {
    image.mapv_inplace(|v| 1.0 - v);
}
