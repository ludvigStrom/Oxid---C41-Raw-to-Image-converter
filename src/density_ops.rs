//! Density-domain operations: saturation, shadow cast correction, highlight speckle
//! compression, and zone-based density adjustments (shadows/midtones/highlights).

use ndarray::Array3;

/// Density-domain saturation boost: scale per-channel deviation from the
/// neutral axis (equal-density gray line).
///
///   D_mean  = (D_r + D_g + D_b) / 3
///   D_ch' = D_mean + saturation * (D_ch - D_mean)
///
/// Detect and compress density-domain "speckle" pixels where one channel is
/// an extreme outlier while the other two are close together.  Normal
/// saturated colors have channels that are roughly evenly spaced (e.g.
/// 0.3, 0.5, 0.7) and are left untouched.  Speckles have a skewed
/// distribution (e.g. 1.0, 1.0, 2.5) and get pulled toward the mean.
pub(crate) fn limit_highlight_density_spread(image: &mut Array3<f32>) {
    let (h, w, _) = image.dim();
    for y in 0..h {
        for x in 0..w {
            let r = image[[y, x, 0]];
            let g = image[[y, x, 1]];
            let b = image[[y, x, 2]];

            let mut lo = r;
            let mut mid = g;
            let mut hi = b;
            if lo > mid {
                std::mem::swap(&mut lo, &mut mid);
            }
            if mid > hi {
                std::mem::swap(&mut mid, &mut hi);
            }
            if lo > mid {
                std::mem::swap(&mut lo, &mut mid);
            }

            let range = hi - lo;
            if range < 0.02 {
                continue;
            }

            let mid_pos = (mid - lo) / range;
            let outlier = (0.5 - mid_pos).abs() * 2.0;
            if outlier < 0.5 {
                continue;
            }

            let excess = (outlier - 0.5) / 0.5;
            let blend = excess * 0.85;
            let mean = (r + g + b) * (1.0 / 3.0);

            image[[y, x, 0]] = r + (mean - r) * blend;
            image[[y, x, 1]] = g + (mean - g) * blend;
            image[[y, x, 2]] = b + (mean - b) * blend;
        }
    }
}

/// Reinhard-style highlight roll-off in density space. Compresses high densities
/// to mask noise in dense negative areas (skies). Per-channel: d_out = lerp(d, d/(1 + d/d_mid), strength).
/// When strength is 0, no change. d_mid is the knee density (e.g. 1.5).
pub(crate) fn apply_reinhard_highlight_rolloff(
    image: &mut Array3<f32>,
    d_mid: f32,
    strength: f32,
) {
    if strength.abs() < 1e-6 {
        return;
    }
    let d_mid = d_mid.max(1e-6);
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let d = image[[y, x, ch]];
                let d_reinhard = d / (1.0 + d / d_mid);
                image[[y, x, ch]] = d + (d_reinhard - d) * strength;
            }
        }
    }
}

/// sat > 1 widens channel spread → more colorful output after the S-curve.
/// sat = 1 is identity. Values are clamped to ≥ 0 after boosting.
pub(crate) fn apply_density_saturation(image: &mut Array3<f32>, saturation: f32) {
    if (saturation - 1.0).abs() < 1e-6 {
        return;
    }
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    for y in 0..h {
        for x in 0..w {
            let dr = image[[y, x, 0]];
            let dg = image[[y, x, 1]];
            let db = image[[y, x, 2]];
            let d_mean = (dr + dg + db) * (1.0 / 3.0);
            image[[y, x, 0]] = (d_mean + saturation * (dr - d_mean)).max(0.0);
            image[[y, x, 1]] = (d_mean + saturation * (dg - d_mean)).max(0.0);
            image[[y, x, 2]] = (d_mean + saturation * (db - d_mean)).max(0.0);
        }
    }
}

/// Analyze shadow cast: measure per-channel color imbalance in the low-density
/// (shadow) zone. Returns a correction vector (dr, dg, db) that pulls the shadow
/// average toward neutral gray. All zeros if no shadow pixels found.
pub(crate) fn analyze_shadow_cast(image: &Array3<f32>, threshold: f32) -> (f32, f32, f32) {
    let (h, w, _) = image.dim();
    let mut sum_r = 0.0_f64;
    let mut sum_g = 0.0_f64;
    let mut sum_b = 0.0_f64;
    let mut count = 0u64;

    for y in 0..h {
        for x in 0..w {
            let dr = image[[y, x, 0]];
            let dg = image[[y, x, 1]];
            let db = image[[y, x, 2]];
            let d_mean = (dr + dg + db) * (1.0 / 3.0);
            if d_mean < threshold {
                sum_r += dr as f64;
                sum_g += dg as f64;
                sum_b += db as f64;
                count += 1;
            }
        }
    }

    if count == 0 {
        return (0.0, 0.0, 0.0);
    }

    let avg_r = (sum_r / count as f64) as f32;
    let avg_g = (sum_g / count as f64) as f32;
    let avg_b = (sum_b / count as f64) as f32;
    let target = (avg_r + avg_g + avg_b) * (1.0 / 3.0);

    (target - avg_r, target - avg_g, target - avg_b)
}

/// Apply shadow cast correction: adds the correction vector weighted by a smooth
/// ramp that is strongest near D=0 (deep shadows) and fades to zero by
/// `threshold`. The exponent (1.5) makes the falloff nonlinear so midtones
/// are barely affected.
pub(crate) fn apply_shadow_cast_correction(
    image: &mut Array3<f32>,
    correction: (f32, f32, f32),
    strength: f32,
    threshold: f32,
) {
    if strength.abs() < 1e-6 {
        return;
    }
    let (cr, cg, cb) = correction;
    if cr.abs() < 1e-6 && cg.abs() < 1e-6 && cb.abs() < 1e-6 {
        return;
    }
    let (h, w, _) = image.dim();
    let inv_thresh = 1.0 / threshold.max(1e-6);

    for y in 0..h {
        for x in 0..w {
            let dr = image[[y, x, 0]];
            let dg = image[[y, x, 1]];
            let db = image[[y, x, 2]];
            let d_mean = (dr + dg + db) * (1.0 / 3.0);

            let t = (1.0 - d_mean * inv_thresh).max(0.0);
            let weight = t * t.sqrt() * strength; // t^1.5

            image[[y, x, 0]] = (dr + cr * weight).max(0.0);
            image[[y, x, 1]] = (dg + cg * weight).max(0.0);
            image[[y, x, 2]] = (db + cb * weight).max(0.0);
        }
    }
}

/// Compute the 2nd- and 98th-percentile effective density from a density image.
/// Uses uniform sampling (~4096 pixels) so it stays fast even for large images.
/// Returns `(d_zone_min, d_zone_max)`.
pub(crate) fn zone_density_range(image: &Array3<f32>, curve_offset: f32) -> (f32, f32) {
    let (h, w, _) = image.dim();
    let n = h * w;
    if n == 0 {
        return (0.0, 1.0);
    }
    let step = (n / 4096).max(1);
    let mut d_effs: Vec<f32> = (0..n)
        .step_by(step)
        .map(|i| {
            let y = i / w;
            let x = i % w;
            let d_mean = (image[[y, x, 0]] + image[[y, x, 1]] + image[[y, x, 2]]) / 3.0;
            d_mean + curve_offset
        })
        .collect();
    d_effs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let ns = d_effs.len();
    let lo = d_effs[(ns * 2 / 100).clamp(0, ns - 1)];
    let hi = d_effs[(ns * 98 / 100).clamp(0, ns - 1)];
    (lo, hi.max(lo + 0.1))
}

/// Estimate zone density range from a transmittance image (step-4 input).
/// Approximates density as −log10(T) × inv_gamma, ignoring per-channel WB.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn zone_density_range_from_transmittance(
    image: &Array3<f32>,
    options: &crate::PipelineOptions,
) -> (f32, f32) {
    let (h, w, _) = image.dim();
    let n = h * w;
    if n == 0 {
        return (0.0, 1.0);
    }
    let inv_gamma = 1.0 / options.film_gamma.max(0.1);
    let step = (n / 4096).max(1);
    let mut d_effs: Vec<f32> = (0..n)
        .step_by(step)
        .map(|i| {
            let y = i / w;
            let x = i % w;
            let tr = image[[y, x, 0]].max(1e-10);
            let tg = image[[y, x, 1]].max(1e-10);
            let tb = image[[y, x, 2]].max(1e-10);
            let d_mean = ((-tr.log10()).max(0.0)
                + (-tg.log10()).max(0.0)
                + (-tb.log10()).max(0.0))
                / 3.0
                * inv_gamma;
            d_mean + options.curve_offset
        })
        .collect();
    d_effs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let ns = d_effs.len();
    let lo = d_effs[(ns * 2 / 100).clamp(0, ns - 1)];
    let hi = d_effs[(ns * 98 / 100).clamp(0, ns - 1)];
    (lo, hi.max(lo + 0.1))
}

/// Gaussian-masked zone density adjustments: per-zone gain (multiplicative) then
/// shadow/mid/highlight offsets (additive). Operates in density space.
///
/// Zone masks are computed in **normalized** effective-density space: the image's own
/// 2nd–98th percentile density range is mapped to 0–1, then shadow/mid/highlight
/// are placed at 0.15/0.50/0.85 in that normalized space. This ensures the zones
/// always target the actual dark/middle/bright tones of each image regardless of
/// its overall exposure level.
pub(crate) fn apply_zone_density_adjustments(
    image: &mut Array3<f32>,
    curve_offset: f32,
    shadows: f32,
    highlights: f32,
    color_s: [f32; 3],
    color_m: [f32; 3],
    color_h: [f32; 3],
    zone_shadow_gain: f32,
    zone_mid_gain: f32,
    zone_highlight_gain: f32,
    color_shadow_gain: [f32; 3],
    color_mid_gain: [f32; 3],
    color_highlight_gain: [f32; 3],
) {
    let gains_zero = zone_shadow_gain.abs() < 1e-6
        && zone_mid_gain.abs() < 1e-6
        && zone_highlight_gain.abs() < 1e-6
        && color_shadow_gain.iter().all(|v| v.abs() < 1e-6)
        && color_mid_gain.iter().all(|v| v.abs() < 1e-6)
        && color_highlight_gain.iter().all(|v| v.abs() < 1e-6);
    let offsets_zero = shadows.abs() < 1e-6
        && highlights.abs() < 1e-6
        && color_s.iter().chain(color_m.iter()).chain(color_h.iter()).all(|v| v.abs() < 1e-6);
    if gains_zero && offsets_zero {
        return;
    }
    let (h, w, _) = image.dim();

    // Normalize d_eff to 0-1 using the image's own tonal range.
    let (d_zone_min, d_zone_max) = zone_density_range(image, curve_offset);
    let d_range = (d_zone_max - d_zone_min).max(0.01);

    // Fixed zone positions in normalized [0,1] space.
    const S_NRM: f32 = 0.15;  // shadow center
    const M_NRM: f32 = 0.50;  // midtone center
    const H_NRM: f32 = 0.85;  // highlight center
    // sigma = 0.20 for shadow/highlight, 0.25 for midtones → inv_2sig² = 1/(2*sigma²)
    const S_INV_2S2: f32 = 1.0 / (2.0 * 0.20 * 0.20);  // ≈ 12.5
    const M_INV_2S2: f32 = 1.0 / (2.0 * 0.25 * 0.25);  // ≈ 8.0
    const H_INV_2S2: f32 = 1.0 / (2.0 * 0.20 * 0.20);  // ≈ 12.5

    const SCALE: f32 = 2.0;
    let s_global = shadows * SCALE;
    let h_global = highlights * SCALE;
    let color_scale = -SCALE;

    for y in 0..h {
        for x in 0..w {
            let dr = image[[y, x, 0]];
            let dg = image[[y, x, 1]];
            let db = image[[y, x, 2]];
            let d_mean = (dr + dg + db) * (1.0 / 3.0);
            let d_eff = d_mean + curve_offset;
            let d_norm = (d_eff - d_zone_min) / d_range;

            let s_diff = d_norm - S_NRM;
            let s_mask = (-s_diff * s_diff * S_INV_2S2).exp();
            let m_diff = d_norm - M_NRM;
            let m_mask = (-m_diff * m_diff * M_INV_2S2).exp();
            let h_diff = d_norm - H_NRM;
            let h_mask = (-h_diff * h_diff * H_INV_2S2).exp();

            let mult_r = (1.0 + zone_shadow_gain * s_mask)
                * (1.0 + zone_mid_gain * m_mask)
                * (1.0 + zone_highlight_gain * h_mask)
                * (1.0 + color_shadow_gain[0] * s_mask)
                * (1.0 + color_mid_gain[0] * m_mask)
                * (1.0 + color_highlight_gain[0] * h_mask);
            let mult_g = (1.0 + zone_shadow_gain * s_mask)
                * (1.0 + zone_mid_gain * m_mask)
                * (1.0 + zone_highlight_gain * h_mask)
                * (1.0 + color_shadow_gain[1] * s_mask)
                * (1.0 + color_mid_gain[1] * m_mask)
                * (1.0 + color_highlight_gain[1] * h_mask);
            let mult_b = (1.0 + zone_shadow_gain * s_mask)
                * (1.0 + zone_mid_gain * m_mask)
                * (1.0 + zone_highlight_gain * h_mask)
                * (1.0 + color_shadow_gain[2] * s_mask)
                * (1.0 + color_mid_gain[2] * m_mask)
                * (1.0 + color_highlight_gain[2] * h_mask);

            let dr_g = dr * mult_r;
            let dg_g = dg * mult_g;
            let db_g = db * mult_b;

            let global_offset = s_global * s_mask + h_global * h_mask;
            let offset_r = global_offset
                + (color_s[0] * s_mask + color_m[0] * m_mask + color_h[0] * h_mask) * color_scale;
            let offset_g = global_offset
                + (color_s[1] * s_mask + color_m[1] * m_mask + color_h[1] * h_mask) * color_scale;
            let offset_b = global_offset
                + (color_s[2] * s_mask + color_m[2] * m_mask + color_h[2] * h_mask) * color_scale;

            image[[y, x, 0]] = (dr_g + offset_r).max(0.0);
            image[[y, x, 1]] = (dg_g + offset_g).max(0.0);
            image[[y, x, 2]] = (db_g + offset_b).max(0.0);
        }
    }
}
