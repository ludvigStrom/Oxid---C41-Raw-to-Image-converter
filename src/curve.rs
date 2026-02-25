//! Universal tone curve: 1D LUT (65 536 entries) applied to linear f32 → u16.
//!
//! Mimics RA-4 / Cineon-style contrast: gentle toe (shadows), steep midtones,
//! smooth shoulder (highlights). LUT is built once and applied via lookup for speed.

use ndarray::{Array3, Zip};

/// Number of entries in the 16-bit LUT (one per possible u16 input index).
const LUT_LEN: usize = 65_536;

/// S-curve (film-style): linear f32 in [0, 1] → f32 in [0, 1].
///
/// Uses a normalized logistic (sigmoid) for a gentle toe, steep midtones (~gamma-like),
/// and smooth shoulder. Output is strictly clamped to [0.0, 1.0].
///
/// * `x` — linear input, typically 0.0..=1.0 (out-of-range is clamped).
/// * `steepness` — controls midtone slope; ~8–12 gives film-like contrast.
#[inline]
fn tone_curve_sigmoid(x: f32, steepness: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    let t = (x - 0.5) * steepness;
    let s = 1.0 / (1.0 + (-t).exp());
    // Normalize so s(0)→0, s(1)→1 (sigmoid is only 0..1 at infinity)
    let s0 = 1.0 / (1.0 + (0.5 * steepness).exp());
    let s1 = 1.0 / (1.0 + (-0.5 * steepness).exp());
    let out = (s - s0) / (s1 - s0);
    out.clamp(0.0, 1.0)
}

/// Build the 16-bit LUT: for each index 0..65535, curve(normalize(index)) → u16.
///
/// LUT is 65 536 elements; index `i` corresponds to linear value `i / 65535`.
pub fn generate_16bit_lut(steepness: f32) -> Vec<u16> {
    (0..LUT_LEN)
        .map(|i| {
            let x = i as f32 / 65535.0;
            let y = tone_curve_sigmoid(x, steepness);
            (y * 65535.0).round() as u16
        })
        .collect()
}

/// Default steepness for the tone curve (film-like midtones).
pub const DEFAULT_STEEPNESS: f32 = 10.0;

/// Apply the tone curve via LUT and quantize to u16 in one pass (parallel).
///
/// * `image` — linear RGB f32 (H, W, 3), typically 0.0..=1.0.
/// * `lut` — from `generate_16bit_lut`; length must be 65 536.
///
/// Values below 0 or above 1 are clamped to the first/last LUT index to avoid panics.
/// Returns an `Array3<u16>` of the same shape, ready for 16-bit TIFF.
pub fn apply_curve_and_quantize(image: &Array3<f32>, lut: &[u16]) -> Array3<u16> {
    assert_eq!(lut.len(), LUT_LEN, "LUT must have 65536 entries");

    let mut out = Array3::<u16>::zeros(image.dim());

    Zip::from(image).and(out.view_mut()).par_for_each(|v, o| {
        let x = (*v).clamp(0.0, 1.0);
        let idx = (x * 65535.0).round() as usize;
        let idx = idx.min(65535);
        *o = lut[idx];
    });

    out
}
