//! Post-curve display tweaks: toe/shoulder shaping, highlight warmth, soft knee,
//! film-print params builder, and display-space 3D LUT application.

use ndarray::Array3;

use crate::curve;
use crate::lut3d;
use crate::PipelineOptions;

/// Smooth hermite interpolation: returns 0 for x <= edge0, 1 for x >= edge1.
#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Toe/shoulder shaping applied in **output space** (after the RA-4/FilmPrint curve),
/// operating on u16 values normalized to [0, 1].
///
/// This is the effective version: because the RA-4 S-curve compresses all high-density
/// values to near-white before any density-domain offset is visible, shaping must happen
/// after the curve where every increment corresponds to a visible brightness change.
pub(crate) fn apply_toe_shoulder_u16(
    image: &mut Array3<u16>,
    toe_strength: f32,
    shoulder_strength: f32,
) {
    let toe = toe_strength.clamp(-1.0, 1.0);
    let shoulder = shoulder_strength.clamp(-1.0, 1.0);
    if toe.abs() < 1e-6 && shoulder.abs() < 1e-6 {
        return;
    }
    const TOE_SCALE: f32 = 0.60;
    const SHOULDER_SCALE: f32 = 0.90;
    const MID: f32 = 0.5;
    let (h, w, c) = image.dim();
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let v = image[[y, x, ch]] as f32 / 65535.0;
                let toe_mask = 1.0 - smoothstep(0.07, 0.60, v);
                let shoulder_mask = smoothstep(0.45, 0.95, v);
                let toe_offset = toe * toe_mask * (MID - v) * TOE_SCALE;
                let shoulder_offset = shoulder * shoulder_mask * (MID - v) * SHOULDER_SCALE;
                let v_new = (v + toe_offset + shoulder_offset).clamp(0.0, 1.0);
                image[[y, x, ch]] = (v_new * 65535.0).round() as u16;
            }
        }
    }
}

/// Build `FilmPrintParams` from `PipelineOptions`.
pub(crate) fn build_film_print_params(opts: &PipelineOptions) -> curve::FilmPrintParams {
    curve::FilmPrintParams {
        base: curve::PrintCurveParams {
            offset: opts.curve_offset,
            gamma: opts.curve_gamma,
            pivot: opts.curve_pivot,
        },
        offset_rgb: [opts.fp_offset_r, opts.fp_offset_g, opts.fp_offset_b],
        gamma_rgb: [opts.fp_gamma_r, opts.fp_gamma_g, opts.fp_gamma_b],
        white_point: opts.curve_white,
        color_bleed: opts.fp_color_bleed,
        vibrance: opts.fp_vibrance,
    }
}

/// Post-curve highlight warmth: adds a golden/warm tint to neutral highlights
/// while leaving already-saturated pixels untouched (chroma-aware).
///
/// Works on u16 RA-4 output (0–65535). Only applies when warmth != 0.
///
/// Noritsu/Frontier scanners shift neutral highlights toward golden/cream
/// but leave punchy saturated colors (blue sky, red tones) alone.
pub(crate) fn apply_highlight_warmth_u16(image: &mut Array3<u16>, warmth: f32) {
    if warmth.abs() < 1e-6 {
        return;
    }
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    let scale = 1.0 / 65535.0_f32;

    for y in 0..h {
        for x in 0..w {
            let mut r = image[[y, x, 0]] as f32 * scale;
            let mut g = image[[y, x, 1]] as f32 * scale;
            let mut b = image[[y, x, 2]] as f32 * scale;

            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let chroma = r.max(g).max(b) - r.min(g).min(b);

            // Ramp: full effect in bright highlights, fades toward midtones.
            let highlight_ramp = smoothstep(0.35, 0.85, luma);
            // Neutrality gate: full effect on neutral tones, zero on saturated.
            let neutrality = 1.0 - smoothstep(0.04, 0.18, chroma);

            let strength = highlight_ramp * neutrality * warmth;

            r = (r + strength * 0.035).clamp(0.0, 1.0);
            g = (g + strength * 0.015).clamp(0.0, 1.0);
            b = (b - strength * 0.055).clamp(0.0, 1.0);

            // Extra safety: in very bright highlights, limit extreme chroma
            // so clipped channels do not produce colored speckles (e.g. pure blue)
            // on otherwise neutral speculars.
            let luma2 = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let chroma2 = r.max(g).max(b) - r.min(g).min(b);
            if luma2 > 0.96 && chroma2 > 0.10 {
                let t = smoothstep(0.96, 1.0, luma2);
                let max_chroma = 0.10;
                let reduce = ((chroma2 - max_chroma) / chroma2).clamp(0.0, 1.0) * t;
                // Pull channels toward neutral luma by `reduce` fraction.
                r = r + (luma2 - r) * reduce;
                g = g + (luma2 - g) * reduce;
                b = b + (luma2 - b) * reduce;
            }

            image[[y, x, 0]] = (r * 65535.0).round() as u16;
            image[[y, x, 1]] = (g * 65535.0).round() as u16;
            image[[y, x, 2]] = (b * 65535.0).round() as u16;
        }
    }
}

/// Scalar soft-knee mapping in [0, 1], inspired by film-like highlight roll-off.
/// `s` is the knee start: below `s` the curve is identity, above it rolls toward 1.0.
#[inline]
fn soft_knee_scalar(x: f32, s: f32) -> f32 {
    let s = s.clamp(0.0, 0.9999);
    if x <= s {
        x
    } else {
        let one_minus_s = 1.0 - s;
        let t = -(x - s) / one_minus_s;
        s + (1.0 - t.exp()) * one_minus_s
    }
}

/// Apply a post-curve soft knee to u16 RA-4 output.
pub(crate) fn apply_soft_knee_u16(image: &mut Array3<u16>, soft_clip: f32) {
    if !(0.0..0.999).contains(&soft_clip) {
        return;
    }
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    let inv = 1.0 / 65535.0_f32;
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let v = image[[y, x, ch]] as f32 * inv;
                let v_knee = soft_knee_scalar(v, soft_clip);
                image[[y, x, ch]] = (v_knee.clamp(0.0, 1.0) * 65535.0).round() as u16;
            }
        }
    }
}

/// Same as `apply_highlight_warmth_u16` but operates on normalized f32 [0, 1] RGB.
pub(crate) fn apply_highlight_warmth_f32(image: &mut Array3<f32>, warmth: f32) {
    if warmth.abs() < 1e-6 {
        return;
    }
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);

    for y in 0..h {
        for x in 0..w {
            let mut r = image[[y, x, 0]].clamp(0.0, 1.0);
            let mut g = image[[y, x, 1]].clamp(0.0, 1.0);
            let mut b = image[[y, x, 2]].clamp(0.0, 1.0);

            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let chroma = r.max(g).max(b) - r.min(g).min(b);

            let highlight_ramp = smoothstep(0.35, 0.85, luma);
            let neutrality = 1.0 - smoothstep(0.04, 0.18, chroma);
            let strength = highlight_ramp * neutrality * warmth;

            r = (r + strength * 0.035).clamp(0.0, 1.0);
            g = (g + strength * 0.015).clamp(0.0, 1.0);
            b = (b - strength * 0.055).clamp(0.0, 1.0);

            let luma2 = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let chroma2 = r.max(g).max(b) - r.min(g).min(b);
            if luma2 > 0.96 && chroma2 > 0.10 {
                let t = smoothstep(0.96, 1.0, luma2);
                let max_chroma = 0.10;
                let reduce = ((chroma2 - max_chroma) / chroma2).clamp(0.0, 1.0) * t;
                r = r + (luma2 - r) * reduce;
                g = g + (luma2 - g) * reduce;
                b = b + (luma2 - b) * reduce;
            }

            image[[y, x, 0]] = r;
            image[[y, x, 1]] = g;
            image[[y, x, 2]] = b;
        }
    }
}

/// Apply a post-curve soft knee to display-space f32 RGB in [0, 1].
pub(crate) fn apply_soft_knee_f32(image: &mut Array3<f32>, soft_clip: f32) {
    if !(0.0..0.999).contains(&soft_clip) {
        return;
    }
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let v = image[[y, x, ch]].clamp(0.0, 1.0);
                image[[y, x, ch]] = soft_knee_scalar(v, soft_clip);
            }
        }
    }
}

/// Apply a display-space 3D LUT to an image already in [0, 1].
pub(crate) fn apply_output_cube_rgb(image: &mut Array3<f32>, lut: &lut3d::Lut3d) {
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    for y in 0..h {
        for x in 0..w {
            let r = image[[y, x, 0]].clamp(0.0, 1.0);
            let g = image[[y, x, 1]].clamp(0.0, 1.0);
            let b = image[[y, x, 2]].clamp(0.0, 1.0);
            let [or, og, ob] = lut.sample_normalized(r, g, b);
            image[[y, x, 0]] = or;
            image[[y, x, 1]] = og;
            image[[y, x, 2]] = ob;
        }
    }
}
