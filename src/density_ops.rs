//! Density-domain operations: saturation, shadow cast correction, highlight speckle
//! compression, and zone-based density adjustments (shadows/midtones/highlights).

use ndarray::Array3;

/// Shadow cast analysis threshold (mean density). Pixels with d_mean below this
/// are considered shadow. 1.2 extends into lower midtones so more real-world
/// shadow regions are captured (0.8 was too restrictive for many scans).
pub(crate) const SHADOW_CAST_THRESHOLD: f32 = 1.2;

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
pub(crate) fn limit_highlight_density_spread_pixel(r: f32, g: f32, b: f32) -> [f32; 3] {
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
        return [r, g, b];
    }

    let mid_pos = (mid - lo) / range;
    let outlier = (0.5 - mid_pos).abs() * 2.0;
    if outlier < 0.5 {
        return [r, g, b];
    }

    let excess = (outlier - 0.5) / 0.5;
    let blend = excess * 0.85;
    let mean = (r + g + b) * (1.0 / 3.0);
    [
        r + (mean - r) * blend,
        g + (mean - g) * blend,
        b + (mean - b) * blend,
    ]
}

pub(crate) fn limit_highlight_density_spread(image: &mut Array3<f32>) {
    let (h, w, _) = image.dim();
    for y in 0..h {
        for x in 0..w {
            let [r, g, b] = limit_highlight_density_spread_pixel(
                image[[y, x, 0]],
                image[[y, x, 1]],
                image[[y, x, 2]],
            );
            image[[y, x, 0]] = r;
            image[[y, x, 1]] = g;
            image[[y, x, 2]] = b;
        }
    }
}

/// Reinhard-style highlight roll-off in density space. Compresses high densities
/// to mask noise in dense negative areas (skies). Per-channel: d_out = lerp(d, d/(1 + d/d_mid), strength).
/// When strength is 0, no change. d_mid is the knee density (e.g. 1.5).
pub(crate) fn apply_reinhard_highlight_rolloff(image: &mut Array3<f32>, d_mid: f32, strength: f32) {
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
                image[[y, x, ch]] = (d + (d_reinhard - d) * strength).max(0.0);
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

/// Per-zone saturation: effective_sat = saturation * (s_mask*zone_shadow_sat + m_mask*zone_mid_sat + h_mask*zone_highlight_sat).
/// Uses the same zone masks as apply_zone_density_adjustments. When all zone sats are 1.0, falls back to global saturation.
pub(crate) fn apply_zone_density_saturation(
    image: &mut Array3<f32>,
    curve_offset: f32,
    saturation: f32,
    zone_shadow_saturation: f32,
    zone_mid_saturation: f32,
    zone_highlight_saturation: f32,
    pinned_zone: Option<ZonePercentiles>,
) {
    let zone_sat_identity = (zone_shadow_saturation - 1.0).abs() < 1e-6
        && (zone_mid_saturation - 1.0).abs() < 1e-6
        && (zone_highlight_saturation - 1.0).abs() < 1e-6;
    if zone_sat_identity {
        apply_density_saturation(image, saturation);
        return;
    }

    let (h, w, _) = image.dim();
    let zp = pinned_zone.unwrap_or_else(|| zone_density_range(image, curve_offset));

    let gap_low = (zp.d_p33 - zp.d_min).max(0.01);
    let gap_high = (zp.d_max - zp.d_p66).max(0.01);
    let gap_mid = (zp.d_p66 - zp.d_p33).max(0.01);
    let tw = (gap_mid * 0.3)
        .min(gap_low * 0.5)
        .min(gap_high * 0.5)
        .max(0.005);

    for y in 0..h {
        for x in 0..w {
            let dr = image[[y, x, 0]];
            let dg = image[[y, x, 1]];
            let db = image[[y, x, 2]];
            let d_mean = (dr + dg + db) * (1.0 / 3.0);
            let d_eff = d_mean + curve_offset;

            let s_mask = 1.0 - smoothstep(zp.d_p33 - tw, zp.d_p33 + tw, d_eff);
            let h_mask = smoothstep(zp.d_p66 - tw, zp.d_p66 + tw, d_eff);
            let m_mask = 1.0 - s_mask - h_mask;

            let effective_sat = saturation
                * (s_mask * zone_shadow_saturation
                    + m_mask * zone_mid_saturation
                    + h_mask * zone_highlight_saturation);

            if (effective_sat - 1.0).abs() < 1e-6 {
                continue;
            }
            image[[y, x, 0]] = (d_mean + effective_sat * (dr - d_mean)).max(0.0);
            image[[y, x, 1]] = (d_mean + effective_sat * (dg - d_mean)).max(0.0);
            image[[y, x, 2]] = (d_mean + effective_sat * (db - d_mean)).max(0.0);
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

/// Percentile result for zone density analysis.
/// All four values are in raw effective-density units (not normalized).
#[derive(Clone, Copy)]
pub(crate) struct ZonePercentiles {
    /// 2nd percentile (zone floor).
    pub d_min: f32,
    /// 33rd percentile (shadow/midtone crossover).
    pub d_p33: f32,
    /// 66th percentile (midtone/highlight crossover).
    pub d_p66: f32,
    /// 98th percentile (zone ceiling).
    pub d_max: f32,
}

/// Sample ~4096 pixels and return the 2nd, 33rd, 66th, 98th percentiles of
/// effective density. The 33rd/66th percentiles are the adaptive crossover
/// points so each zone contains roughly 1/3 of the image's pixels regardless
/// of the density distribution.
pub(crate) fn zone_density_range(image: &Array3<f32>, curve_offset: f32) -> ZonePercentiles {
    let (h, w, _) = image.dim();
    let n = h * w;
    if n == 0 {
        return ZonePercentiles {
            d_min: 0.0,
            d_p33: 0.33,
            d_p66: 0.66,
            d_max: 1.0,
        };
    }
    let step = (n / 4096).max(1);
    let d_effs: Vec<f32> = (0..n)
        .step_by(step)
        .map(|i| {
            let y = i / w;
            let x = i % w;
            let d_mean = (image[[y, x, 0]] + image[[y, x, 1]] + image[[y, x, 2]]) / 3.0;
            d_mean + curve_offset
        })
        .collect();
    zone_percentiles_from_samples(d_effs)
}

pub(crate) fn zone_percentiles_from_samples(mut d_effs: Vec<f32>) -> ZonePercentiles {
    if d_effs.is_empty() {
        return ZonePercentiles {
            d_min: 0.0,
            d_p33: 0.33,
            d_p66: 0.66,
            d_max: 1.0,
        };
    }
    d_effs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let ns = d_effs.len();
    let lo = d_effs[(ns * 2 / 100).clamp(0, ns - 1)];
    let p33 = d_effs[(ns * 33 / 100).clamp(0, ns - 1)];
    let p66 = d_effs[(ns * 66 / 100).clamp(0, ns - 1)];
    let hi = d_effs[(ns * 98 / 100).clamp(0, ns - 1)];
    ZonePercentiles {
        d_min: lo,
        d_p33: p33.max(lo + 0.02),
        d_p66: p66.max(p33 + 0.02),
        d_max: hi.max(p66 + 0.02),
    }
}

/// Zone range for the current buffer, or full-frame pins from preview options.
pub(crate) fn zone_range_for_options(
    image: &Array3<f32>,
    options: &crate::PipelineOptions,
) -> ZonePercentiles {
    if let Some((d_min, d_p33, d_p66, d_max)) = options.pinned_zone {
        let o = options.curve_offset;
        ZonePercentiles {
            d_min: d_min + o,
            d_p33: d_p33 + o,
            d_p66: d_p66 + o,
            d_max: d_max + o,
        }
    } else {
        zone_density_range(image, options.curve_offset)
    }
}

/// Smoothstep: 0 for x <= edge0, 1 for x >= edge1, smooth cubic in between.
#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Zone density adjustments with adaptive percentile-based 3-way separation.
///
/// Zone crossovers are placed at the image's 33rd and 66th density percentiles
/// so each zone always contains ~1/3 of the pixels, regardless of the density
/// distribution. This matches DaVinci Resolve's approach: shadows, midtones,
/// and highlights correspond to what the user actually sees on screen.
///
/// Masks sum to 1.0 everywhere (smoothstep transitions).
/// Gains are applied as a weighted blend (not multiplicative compounding).
pub(crate) fn apply_zone_density_adjustments(
    image: &mut Array3<f32>,
    curve_offset: f32,
    shadows: f32,
    highlights: f32,
    zone_shadow_gain: f32,
    zone_mid_gain: f32,
    zone_highlight_gain: f32,
    color_shadow_gain: [f32; 3],
    color_mid_gain: [f32; 3],
    color_highlight_gain: [f32; 3],
    pinned_zone: Option<ZonePercentiles>,
) {
    let gains_zero = zone_shadow_gain.abs() < 1e-6
        && zone_mid_gain.abs() < 1e-6
        && zone_highlight_gain.abs() < 1e-6
        && color_shadow_gain.iter().all(|v| v.abs() < 1e-6)
        && color_mid_gain.iter().all(|v| v.abs() < 1e-6)
        && color_highlight_gain.iter().all(|v| v.abs() < 1e-6);
    let offsets_zero = shadows.abs() < 1e-6 && highlights.abs() < 1e-6;
    if gains_zero && offsets_zero {
        return;
    }
    let (h, w, _) = image.dim();

    let zp = pinned_zone.unwrap_or_else(|| zone_density_range(image, curve_offset));

    // Transition half-width: 30% of the gap between crossover percentiles,
    // so the two smoothstep transitions never overlap.
    let gap_low = (zp.d_p33 - zp.d_min).max(0.01);
    let gap_high = (zp.d_max - zp.d_p66).max(0.01);
    let gap_mid = (zp.d_p66 - zp.d_p33).max(0.01);
    let tw = (gap_mid * 0.3)
        .min(gap_low * 0.5)
        .min(gap_high * 0.5)
        .max(0.005);

    const SCALE: f32 = 2.0;
    let s_global = shadows * SCALE;
    let h_global = highlights * SCALE;

    for y in 0..h {
        for x in 0..w {
            let dr = image[[y, x, 0]];
            let dg = image[[y, x, 1]];
            let db = image[[y, x, 2]];
            let d_mean = (dr + dg + db) * (1.0 / 3.0);
            let d_eff = d_mean + curve_offset;

            let s_mask = 1.0 - smoothstep(zp.d_p33 - tw, zp.d_p33 + tw, d_eff);
            let h_mask = smoothstep(zp.d_p66 - tw, zp.d_p66 + tw, d_eff);
            let m_mask = 1.0 - s_mask - h_mask;

            let gain_r = s_mask * (1.0 + zone_shadow_gain + color_shadow_gain[0])
                + m_mask * (1.0 + zone_mid_gain + color_mid_gain[0])
                + h_mask * (1.0 + zone_highlight_gain + color_highlight_gain[0]);
            let gain_g = s_mask * (1.0 + zone_shadow_gain + color_shadow_gain[1])
                + m_mask * (1.0 + zone_mid_gain + color_mid_gain[1])
                + h_mask * (1.0 + zone_highlight_gain + color_highlight_gain[1]);
            let gain_b = s_mask * (1.0 + zone_shadow_gain + color_shadow_gain[2])
                + m_mask * (1.0 + zone_mid_gain + color_mid_gain[2])
                + h_mask * (1.0 + zone_highlight_gain + color_highlight_gain[2]);

            let global_offset = s_global * s_mask + h_global * h_mask;

            image[[y, x, 0]] = (dr * gain_r + global_offset).max(0.0);
            image[[y, x, 1]] = (dg * gain_g + global_offset).max(0.0);
            image[[y, x, 2]] = (db * gain_b + global_offset).max(0.0);
        }
    }
}
