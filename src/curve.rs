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

use ndarray::{Array3, Axis, Zip};
use ndarray::parallel::prelude::*;

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

    // Optional white-point scaling: map `white_point` to display white.
    let wp = white_point.clamp(1e-6, 10.0);
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

/// Parameters for the RA-4 paper S-curve.
#[derive(Debug, Clone, Copy)]
pub struct PrintCurveParams {
    pub offset: f32,
    pub gamma: f32,
    pub pivot: f32,
}

/// 3×3 density-domain calibration matrix (row-major).
#[derive(Debug, Clone, Copy)]
pub struct DensityMatrix {
    pub m: [[f32; 3]; 3],
}

/// Precomputed data for the multi-stage curve pipeline.
#[derive(Debug)]
pub struct CurvePipeline {
    pub t_to_d_lut: Vec<f32>,   // optional (can be empty to use direct log10)
    pub d_to_u16_lut: Vec<u16>, // required
    pub d_max: f32,
    pub params: PrintCurveParams,
    pub matrix: DensityMatrix,
    /// When set, used instead of the matrix for the density-domain color correction step.
    pub lut3d: Option<crate::lut3d::Lut3d>,
}

impl CurvePipeline {
    /// Construct a new curve pipeline.
    ///
    /// * `d_max` is the maximum density represented by the `d_to_u16_lut` (values above clamp).
    /// * If `use_t_to_d_lut` is false, transmittance → density uses direct log10 at runtime.
    /// * If `lut3d` is `Some`, it is applied after T→D instead of the density matrix.
    pub fn new(
        params: PrintCurveParams,
        matrix: DensityMatrix,
        d_max: f32,
        use_t_to_d_lut: bool,
        lut3d: Option<crate::lut3d::Lut3d>,
    ) -> Self {
        let t_to_d_lut = if use_t_to_d_lut {
            build_t_to_density_lut(THRESHOLD)
        } else {
            Vec::new()
        };
        let d_to_u16_lut = build_density_to_ra4_lut(params, d_max);

        Self {
            t_to_d_lut,
            d_to_u16_lut,
            d_max,
            params,
            matrix,
            lut3d,
        }
    }
}

/// Step 1: Transmittance → Density (scalar).
#[inline]
pub fn transmittance_to_density(t: f32, t_threshold: f32) -> f32 {
    let t = t.clamp(t_threshold, 1.0);
    -t.log10()
}

/// Step 2: Apply 3×3 density-domain matrix to a single RGB triplet.
#[inline]
pub fn apply_density_matrix_pixel(d_in: [f32; 3], matrix: &DensityMatrix) -> [f32; 3] {
    let m = &matrix.m;
    let r = d_in[0];
    let g = d_in[1];
    let b = d_in[2];

    [
        m[0][0] * r + m[0][1] * g + m[0][2] * b,
        m[1][0] * r + m[1][1] * g + m[1][2] * b,
        m[2][0] * r + m[2][1] * g + m[2][2] * b,
    ]
}

/// Step 3: Density → RA-4 output for a single channel.
#[inline]
pub fn density_to_ra4(density: f32, params: &PrintCurveParams) -> f32 {
    let density = density.max(0.0);
    let log_exposure = density + params.offset;
    let linear_exposure = (10.0f32).powf(log_exposure);

    let eg = linear_exposure.powf(params.gamma);
    let pg = params.pivot.powf(params.gamma);
    let out = eg / (eg + pg);

    out.clamp(0.0, 1.0)
}

/// LUT for Step 1: transmittance → density.
pub fn build_t_to_density_lut(t_threshold: f32) -> Vec<f32> {
    (0..LUT_LEN)
        .map(|i| {
            let t = i as f32 / 65535.0;
            transmittance_to_density(t, t_threshold)
        })
        .collect()
}

/// LUT for Step 3: density → RA-4 quantized u16, over [0, d_max].
pub fn build_density_to_ra4_lut(params: PrintCurveParams, d_max: f32) -> Vec<u16> {
    (0..LUT_LEN)
        .map(|i| {
            let d = (i as f32 / 65535.0) * d_max;
            let mut y = density_to_ra4(d, &params);

            // Soft highlight shoulder: gently compress values near white before 16-bit
            // quantization to avoid harsh clipping in very bright regions (clouds, speculars).
            // Starts at 0.93 to preserve saturation in the upper midtones/highlights where
            // channel separation matters most for color punch.
            const SHOULDER_START: f32 = 0.93;
            if y > SHOULDER_START {
                let t = (y - SHOULDER_START) / (1.0 - SHOULDER_START);
                let t_shaped = 1.0 - (1.0 - t).powf(1.5);
                y = SHOULDER_START + t_shaped * (1.0 - SHOULDER_START);
            }

            (y * 65535.0).round() as u16
        })
        .collect()
}

#[inline]
fn sample_t_to_density(t: f32, pipeline: &CurvePipeline) -> f32 {
    if pipeline.t_to_d_lut.is_empty() {
        transmittance_to_density(t, THRESHOLD)
    } else {
        let x = t.clamp(0.0, 1.0);
        let idx = (x * 65535.0).round() as usize;
        let idx = idx.min(65535);
        pipeline.t_to_d_lut[idx]
    }
}

#[inline]
fn sample_density_to_u16(d: f32, pipeline: &CurvePipeline) -> u16 {
    let d = d.clamp(0.0, pipeline.d_max);
    let x = if pipeline.d_max > 0.0 {
        d / pipeline.d_max
    } else {
        0.0
    };
    let idx = (x * 65535.0).round() as usize;
    let idx = idx.min(65535);
    pipeline.d_to_u16_lut[idx]
}

/// Density → linear print reflectance (unquantized RA-4). Used for ACES/EXR export
/// so archival linear files are print-referred, not raw density treated as ACEScg.
pub fn apply_ra4_from_density_f32(
    density_image: &Array3<f32>,
    params: PrintCurveParams,
    d_max: f32,
) -> Array3<f32> {
    let (h, w, c) = density_image.dim();
    assert_eq!(c, 3, "Expected 3-channel RGB image");
    let mut out = Array3::<f32>::zeros((h, w, c));
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let d = density_image[[y, x, ch]].clamp(0.0, d_max);
                out[[y, x, ch]] = density_to_ra4(d, &params);
            }
        }
    }
    out
}

/// Apply RA-4 curve from density input (T→D and matrix/WB already applied upstream).
///
/// `density_image` is in density domain (H, W, 3): D=0 is film base, higher D = more dye.
/// Produces the final positive u16 output: high density → bright (correct inversion).
pub fn apply_ra4_from_density(
    density_image: &Array3<f32>,
    params: PrintCurveParams,
    d_max: f32,
    white_point: f32,
) -> Array3<u16> {
    let (h, w, c) = density_image.dim();
    assert_eq!(c, 3, "Expected 3-channel RGB image");

    let d_to_u16_lut = build_density_to_ra4_lut(params, d_max);
    let mut out = Array3::<u16>::zeros((h, w, c));

    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(density_image.axis_iter(Axis(0)))
        .for_each(|(mut out_row, in_row)| {
            let row_w = in_row.dim().0;
            for x in 0..row_w {
                for ch in 0..3 {
                    let d = in_row[(x, ch)].clamp(0.0, d_max);
                    let frac = if d_max > 0.0 { d / d_max } else { 0.0 };
                    let idx = (frac * 65535.0).round() as usize;
                    let idx = idx.min(65535);
                    out_row[(x, ch)] = d_to_u16_lut[idx];
                }
            }
        });

    let wp = white_point.clamp(1e-6, 10.0);
    if (wp - 1.0).abs() > f32::EPSILON {
        let inv_wp = 1.0 / wp;
        Zip::from(out.view_mut()).par_for_each(|o| {
            let normalized = *o as f32 / 65535.0;
            let scaled = (normalized * inv_wp).clamp(0.0, 1.0);
            *o = (scaled * 65535.0).round() as u16;
        });
    }

    out
}

/// Apply the full multi-stage curve pipeline (T → D → matrix → RA-4) and quantize to u16.
///
/// `image` is linear transmittance [0, 1] with shape (H, W, 3).
pub fn apply_curve_pipeline(
    image: &Array3<f32>,
    pipeline: &CurvePipeline,
    white_point: f32,
    print_histogram: bool,
) -> Array3<u16> {
    let (h, w, c) = image.dim();
    assert_eq!(c, 3, "Expected 3-channel RGB image");

    let mut out = Array3::<u16>::zeros((h, w, c));

    // Parallelize over rows; each row's pixels are independent.
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(image.axis_iter(Axis(0)))
        .for_each(|(mut out_row, in_row)| {
            let (row_w, row_c) = in_row.dim();
            debug_assert_eq!(row_c, 3);

            for x in 0..row_w {
                let t_r = in_row[(x, 0)].clamp(0.0, 1.0);
                let t_g = in_row[(x, 1)].clamp(0.0, 1.0);
                let t_b = in_row[(x, 2)].clamp(0.0, 1.0);

                // Step 1: T -> D
                let d_in = [
                    sample_t_to_density(t_r, pipeline),
                    sample_t_to_density(t_g, pipeline),
                    sample_t_to_density(t_b, pipeline),
                ];

                // Step 2: matrix or 3D LUT in density domain
                let d_out = if let Some(ref lut) = pipeline.lut3d {
                    lut.sample_density(d_in[0], d_in[1], d_in[2])
                } else {
                    apply_density_matrix_pixel(d_in, &pipeline.matrix)
                };

                // Step 3: D -> RA-4 via LUT
                let y_r = sample_density_to_u16(d_out[0], pipeline);
                let y_g = sample_density_to_u16(d_out[1], pipeline);
                let y_b = sample_density_to_u16(d_out[2], pipeline);

                out_row[(x, 0)] = y_r;
                out_row[(x, 1)] = y_g;
                out_row[(x, 2)] = y_b;
            }
        });

    // Optional white-point scaling: map `white_point` to display white.
    let wp = white_point.clamp(1e-6, 10.0);
    if (wp - 1.0).abs() > f32::EPSILON {
        let inv_wp = 1.0 / wp;

        Zip::from(out.view_mut()).par_for_each(|o| {
            let normalized = *o as f32 / 65535.0;
            let scaled = (normalized * inv_wp).clamp(0.0, 1.0);
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

// ─── Film Print stage ──────────────────────────────────────────────────────

/// Per-channel parameters for the Film Print curve.
#[derive(Debug, Clone, Copy)]
pub struct FilmPrintParams {
    /// Base offset, gamma, pivot (shared starting point).
    pub base: PrintCurveParams,
    /// Per-channel offset deltas (added to base.offset).
    pub offset_rgb: [f32; 3],
    /// Per-channel gamma multipliers (multiplied with base.gamma).
    pub gamma_rgb: [f32; 3],
    /// White point.
    pub white_point: f32,
    /// Inter-channel density bleed before the curve (0.0–0.5).
    pub color_bleed: f32,
    /// Post-curve luminance-aware vibrance (0.0–2.0).
    pub vibrance: f32,
}

/// Build per-channel density→u16 LUTs for Film Print.
fn build_film_print_luts(params: &FilmPrintParams, d_max: f32) -> [Vec<u16>; 3] {
    let mut luts: [Vec<u16>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for ch in 0..3 {
        let offset = params.base.offset + params.offset_rgb[ch];
        let gamma = (params.base.gamma * params.gamma_rgb[ch]).max(0.1);
        let pivot = params.base.pivot;

        let lut: Vec<u16> = (0..LUT_LEN)
            .map(|i| {
                let d = (i as f32 / 65535.0) * d_max;
                let d = d.max(0.0);
                let log_exposure = d + offset;
                let linear_exposure = (10.0f32).powf(log_exposure);
                let eg = linear_exposure.powf(gamma);
                let pg = pivot.powf(gamma);
                let mut y = (eg / (eg + pg)).clamp(0.0, 1.0);

                const SHOULDER_START: f32 = 0.93;
                if y > SHOULDER_START {
                    let t = (y - SHOULDER_START) / (1.0 - SHOULDER_START);
                    let t_shaped = 1.0 - (1.0 - t).powf(1.5);
                    y = SHOULDER_START + t_shaped * (1.0 - SHOULDER_START);
                }

                (y * 65535.0).round() as u16
            })
            .collect();
        luts[ch] = lut;
    }
    luts
}

/// Apply color bleed: mix a fraction of each channel's density with its
/// neighbours. Symmetric blend toward the mean, modulated by `bleed`.
#[inline]
fn apply_color_bleed(dr: f32, dg: f32, db: f32, bleed: f32) -> (f32, f32, f32) {
    if bleed <= 0.0 {
        return (dr, dg, db);
    }
    let keep = 1.0 - bleed;
    let half_bleed = bleed * 0.5;
    (
        dr * keep + dg * half_bleed + db * half_bleed,
        dg * keep + dr * half_bleed + db * half_bleed,
        db * keep + dr * half_bleed + dg * half_bleed,
    )
}

/// Post-curve vibrance: luminance-aware saturation boost that affects muted
/// colors more than already-saturated ones.
///
/// Works on linear [0, 1] RGB. `strength` of 0 is identity; 1.0 is strong.
#[inline]
fn apply_vibrance_pixel(r: f32, g: f32, b: f32, strength: f32) -> (f32, f32, f32) {
    if strength.abs() < 1e-6 {
        return (r, g, b);
    }
    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let chroma = r.max(g).max(b) - r.min(g).min(b);
    let boost = 1.0 + strength * (1.0 - chroma.clamp(0.0, 1.0));
    (
        (luma + (r - luma) * boost).clamp(0.0, 1.0),
        (luma + (g - luma) * boost).clamp(0.0, 1.0),
        (luma + (b - luma) * boost).clamp(0.0, 1.0),
    )
}

/// Apply the Film Print curve from density input.
///
/// Per-channel Michaelis-Menten curves with color bleed and vibrance.
pub fn apply_film_print_from_density(
    density_image: &Array3<f32>,
    params: &FilmPrintParams,
    d_max: f32,
) -> Array3<u16> {
    let (h, w, c) = density_image.dim();
    assert_eq!(c, 3, "Expected 3-channel RGB image");

    let luts = build_film_print_luts(params, d_max);
    let bleed = params.color_bleed;
    let vibrance = params.vibrance;
    let wp = params.white_point.clamp(1e-6, 10.0);
    let apply_wp = (wp - 1.0).abs() > f32::EPSILON;
    let inv_wp = if apply_wp { 1.0 / wp } else { 1.0 };

    let mut out = Array3::<u16>::zeros((h, w, c));

    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(density_image.axis_iter(Axis(0)))
        .for_each(|(mut out_row, in_row)| {
            let row_w = in_row.dim().0;
            for x in 0..row_w {
                let dr = in_row[(x, 0)];
                let dg = in_row[(x, 1)];
                let db = in_row[(x, 2)];

                let (dr, dg, db) = apply_color_bleed(dr, dg, db, bleed);

                let mut rgb_f = [0.0_f32; 3];
                for ch in 0..3 {
                    let d = [dr, dg, db][ch].clamp(0.0, d_max);
                    let frac = d / d_max;
                    let idx = (frac * 65535.0).round().min(65535.0) as usize;
                    rgb_f[ch] = luts[ch][idx] as f32 / 65535.0;
                }

                if apply_wp {
                    for v in rgb_f.iter_mut() {
                        *v = (*v * inv_wp).clamp(0.0, 1.0);
                    }
                }

                let (r, g, b) = apply_vibrance_pixel(rgb_f[0], rgb_f[1], rgb_f[2], vibrance);

                out_row[(x, 0)] = (r.clamp(0.0, 1.0) * 65535.0).round() as u16;
                out_row[(x, 1)] = (g.clamp(0.0, 1.0) * 65535.0).round() as u16;
                out_row[(x, 2)] = (b.clamp(0.0, 1.0) * 65535.0).round() as u16;
            }
        });

    out
}
