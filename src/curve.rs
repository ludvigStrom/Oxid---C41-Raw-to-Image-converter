//! Cineon-style Log-to-Lin Print Film Emulation.
//!
//! Physically models a darkroom enlarger: transmittance → optical density → paper
//! exposure → RA-4 paper S-curve (Michaelis-Menten). The LUT inherently inverts the
//! negative (density domain), so the separate `1.0 - input` step is NOT needed when
//! this curve is active.
//!
//! Parameters map to real darkroom controls:
//!   offset — enlarger bulb intensity (print exposure bias)
//!   gamma  — paper grade / contrast
//!   pivot  — mid-gray pivot for the S-curve

use ndarray::{Array3, Zip};

const LUT_LEN: usize = 65_536;
const THRESHOLD: f32 = 1e-6;

/// Physical print film curve for a single sample.
///
/// Input `t` is linear transmittance [0, 1] (after D-min neutralization, before any inversion).
/// Output is positive screen brightness [0, 1].
///
/// Steps:
///   1. Clamp transmittance to avoid log10(0).
///   2. Optical density:  D = -log10(T)
///      High D = lots of dye = bright original subject.
///   3. Print log-exposure: logE = D + offset
///      Density IS the print exposure (high density → bright print). This is the inversion.
///   4. Linear exposure: E = 10^logE
///   5. RA-4 paper S-curve (Michaelis-Menten):
///      out = E^gamma / (E^gamma + pivot^gamma)
#[inline]
fn print_film_curve(t: f32, offset: f32, gamma: f32, pivot: f32) -> f32 {
    let t = t.clamp(THRESHOLD, 1.0);

    let density = -t.log10();
    let log_exposure = density + offset;
    let linear_exposure = (10.0f32).powf(log_exposure);

    let eg = linear_exposure.powf(gamma);
    let pg = pivot.powf(gamma);
    let out = eg / (eg + pg);

    out.clamp(0.0, 1.0)
}

/// Build the 65 536-entry LUT for the physical print film curve.
///
/// * `offset` — print exposure bias (log-domain shift). Default 0.0. Higher = brighter print.
/// * `gamma`  — paper grade / contrast. Default 2.5. Higher = harder paper (more contrast).
/// * `pivot`  — half-saturation exposure for S-curve. Default 3.0 (maps mid-density ≈ T=0.33 to 50% gray).
///
/// Index `i` corresponds to linear transmittance `i / 65535`.
pub fn generate_16bit_lut(offset: f32, gamma: f32, pivot: f32) -> Vec<u16> {
    (0..LUT_LEN)
        .map(|i| {
            let t = i as f32 / 65535.0;
            let y = print_film_curve(t, offset, gamma, pivot);
            (y * 65535.0).round() as u16
        })
        .collect()
}

/// Apply the tone curve via LUT and quantize to u16 in one pass (parallel).
///
/// `white_point` is a normalized code value in [0, 1] that should map to display
/// white. If `print_histogram` is true, a 256-bin summary is printed to stdout.
///
/// Values below 0 or above 1 are clamped to the first/last LUT index.
pub fn apply_curve_and_quantize(
    image: &Array3<f32>,
    lut: &[u16],
    white_point: f32,
    print_histogram: bool,
) -> Array3<u16> {
    assert_eq!(lut.len(), LUT_LEN, "LUT must have 65536 entries");

    let mut out = Array3::<u16>::zeros(image.dim());

    Zip::from(image).and(out.view_mut()).par_for_each(|v, o| {
        let x = (*v).clamp(0.0, 1.0);
        let idx = (x * 65535.0).round() as usize;
        let idx = idx.min(65535);
        *o = lut[idx];
    });

    // Optional white-point scaling: map `white_point` (normalized) to display white.
    let wp = white_point.clamp(1e-6, 1.0);
    if (wp - 1.0).abs() > f32::EPSILON {
        let inv_wp = 1.0 / wp;

        // Scale in-place. Integer arithmetic is fine here; we stay in u16 domain.
        Zip::from(out.view_mut()).par_for_each(|o| {
            let normalized = *o as f32 / 65535.0;
            let scaled = (normalized * inv_wp).min(1.0);
            *o = (scaled * 65535.0).round() as u16;
        });
    }

    // Optional 256-bin histogram (8-bit view of the u16 output), printed for inspection.
    if print_histogram {
        let mut hist = [0u64; 256];
        for v in out.iter() {
            let bin = (*v as usize) >> 8;
            hist[bin] += 1;
        }
        let total: u64 = hist.iter().sum();
        if total > 0 {
        let mut min_bin = 0usize;
        while min_bin < 256 && hist[min_bin] == 0 {
            min_bin += 1;
        }
        let mut max_bin = 255usize;
        while max_bin > 0 && hist[max_bin] == 0 {
            max_bin -= 1;
        }

        let mut cum = 0u64;
        let mut p50 = 0usize;
        let mut p90 = 0usize;
        let mut p99 = 0usize;
        for (i, &count) in hist.iter().enumerate() {
            cum += count;
            let frac = cum as f64 / total as f64;
            if p50 == 0 && frac >= 0.50 {
                p50 = i;
            }
            if p90 == 0 && frac >= 0.90 {
                p90 = i;
            }
            if p99 == 0 && frac >= 0.99 {
                p99 = i;
                break;
            }
        }

        println!(
            "Histogram (8-bit bins of u16 output): min={} p50={} p90={} p99={} max={}",
            min_bin, p50, p90, p99, max_bin
        );
        }
    }

    out
}
