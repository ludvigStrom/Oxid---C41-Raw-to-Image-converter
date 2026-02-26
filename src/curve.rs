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
}

impl CurvePipeline {
    /// Construct a new curve pipeline.
    ///
    /// * `d_max` is the maximum density represented by the `d_to_u16_lut` (values above clamp).
    /// * If `use_t_to_d_lut` is false, transmittance → density uses direct log10 at runtime.
    pub fn new(
        params: PrintCurveParams,
        matrix: DensityMatrix,
        d_max: f32,
        use_t_to_d_lut: bool,
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
            // Below `shoulder_start` the curve is unchanged; above it we remap into [0,1]
            // and apply a smooth roll-off.
            const SHOULDER_START: f32 = 0.85;
            if y > SHOULDER_START {
                let t = (y - SHOULDER_START) / (1.0 - SHOULDER_START);
                // Exponent > 1.0 yields a flatter shoulder near 1.0.
                let t_shaped = 1.0 - (1.0 - t).powf(2.0);
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

                // Step 2: matrix in density domain
                let d_out = apply_density_matrix_pixel(d_in, &pipeline.matrix);

                // Step 3: D -> RA-4 via LUT
                let y_r = sample_density_to_u16(d_out[0], pipeline);
                let y_g = sample_density_to_u16(d_out[1], pipeline);
                let y_b = sample_density_to_u16(d_out[2], pipeline);

                out_row[(x, 0)] = y_r;
                out_row[(x, 1)] = y_g;
                out_row[(x, 2)] = y_b;
            }
        });

    // Optional white-point scaling: map `white_point` (normalized) to display white.
    let wp = white_point.clamp(1e-6, 1.0);
    if (wp - 1.0).abs() > f32::EPSILON {
        let inv_wp = 1.0 / wp;

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
