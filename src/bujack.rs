//! De-Bujack: non-local compensation for perceptual diminishing returns.
//!
//! Bujack et al. showed that perceived color difference is not a Riemannian
//! metric — large differences compress (diminishing returns). A pointwise grade
//! cannot undo that: any pointwise map of a Riemannian metric is still
//! Riemannian. This pass is therefore spatial.
//!
//! Placement: after step 6 (output transform + display-space looks), before
//! grain / sharpen / encode. Works in OkLab on linear Rec.709-like RGB.
//!
//! Each pixel's difference from an edge-aware local mean is pushed through the
//! inverse of a saturating response `f(d) = k·d/(k+d)`, stretching large
//! differences while leaving small ones alone. Coefficients are uncalibrated
//! by design — the paper proves the effect exists, not its numbers — so the
//! knobs are taste.
//!
//! Math matches `bujack_shader_lab.html` (De-Bujack lab).

use ndarray::Array3;
use rayon::prelude::*;

use crate::color_space;
use crate::options::PipelineOptions;

/// Inverse-response gain, bounded to ~6.7×. `y` is |Δ| in OkLab.
#[inline]
fn un_dim(y: f32, k: f32) -> f32 {
    k / (k - y).max(k * 0.15)
}

#[inline]
fn cbrt_nonneg(v: f32) -> f32 {
    v.max(0.0).cbrt()
}

/// Linear Rec.709/sRGB → OkLab (Ottosson; same coefficients as the shader lab).
#[inline]
fn lin_to_oklab(r: f32, g: f32, b: f32) -> [f32; 3] {
    let l = cbrt_nonneg(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
    let m = cbrt_nonneg(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
    let s = cbrt_nonneg(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
    [
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ]
}

/// OkLab → linear Rec.709/sRGB (Ottosson; same coefficients as the shader lab).
#[inline]
fn oklab_to_lin(lab: [f32; 3]) -> [f32; 3] {
    let l = lab[0] + 0.3963377774 * lab[1] + 0.2158037573 * lab[2];
    let m = lab[0] - 0.1055613458 * lab[1] - 0.0638541728 * lab[2];
    let s = lab[0] - 0.0894841775 * lab[1] - 1.2914855480 * lab[2];
    let l = l * l * l;
    let m = m * m * m;
    let s = s * s * s;
    [
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    ]
}

/// Soft gamut clip: pull out-of-gamut colors toward their own luminance.
fn gamut_clip(rgb: [f32; 3]) -> [f32; 3] {
    let lum = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]).clamp(0.0, 1.0);
    let mut t = 1.0_f32;
    for i in 0..3 {
        let c = rgb[i];
        if c > 1.0 {
            t = t.min((1.0 - lum) / (c - lum).max(1e-5));
        }
        if c < 0.0 {
            t = t.min(lum / (lum - c).max(1e-5));
        }
    }
    let t = t.clamp(0.0, 1.0);
    [
        lum + (rgb[0] - lum) * t,
        lum + (rgb[1] - lum) * t,
        lum + (rgb[2] - lum) * t,
    ]
}

fn sample_lab(lab: &[[f32; 3]], w: usize, h: usize, x: i32, y: i32) -> [f32; 3] {
    let xx = x.clamp(0, w as i32 - 1) as usize;
    let yy = y.clamp(0, h as i32 - 1) as usize;
    lab[yy * w + xx]
}

/// Separable bilateral blur in OkLab (horizontal, then vertical).
fn bilateral_base(
    lab: &[[f32; 3]],
    w: usize,
    h: usize,
    radius: i32,
    edge_sigma: f32,
) -> Vec<[f32; 3]> {
    let radius = radius.clamp(2, 48);
    let sigma = (radius as f32 * 0.5).max(0.6);
    let rs = 1.0 / (2.0 * sigma * sigma);
    let es = 1.0 / (2.0 * edge_sigma.max(1e-4) * edge_sigma.max(1e-4));

    let mut tmp = vec![[0.0_f32; 3]; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
        for x in 0..w {
            let center = lab[y * w + x];
            let mut acc = [0.0_f32; 3];
            let mut wsum = 0.0_f32;
            for i in -radius..=radius {
                let s = sample_lab(lab, w, h, x as i32 + i, y as i32);
                let d0 = s[0] - center[0];
                let d1 = s[1] - center[1];
                let d2 = s[2] - center[2];
                let spatial = (-(i * i) as f32 * rs).exp();
                let range = (-(d0 * d0 + d1 * d1 + d2 * d2) * es).exp();
                let wt = spatial * range;
                acc[0] += s[0] * wt;
                acc[1] += s[1] * wt;
                acc[2] += s[2] * wt;
                wsum += wt;
            }
            let inv = 1.0 / wsum.max(1e-6);
            row_out[x] = [acc[0] * inv, acc[1] * inv, acc[2] * inv];
        }
    });

    let mut out = vec![[0.0_f32; 3]; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
        for x in 0..w {
            let center = tmp[y * w + x];
            let mut acc = [0.0_f32; 3];
            let mut wsum = 0.0_f32;
            for i in -radius..=radius {
                let s = sample_lab(&tmp, w, h, x as i32, y as i32 + i);
                let d0 = s[0] - center[0];
                let d1 = s[1] - center[1];
                let d2 = s[2] - center[2];
                let spatial = (-(i * i) as f32 * rs).exp();
                let range = (-(d0 * d0 + d1 * d1 + d2 * d2) * es).exp();
                let wt = spatial * range;
                acc[0] += s[0] * wt;
                acc[1] += s[1] * wt;
                acc[2] += s[2] * wt;
                wsum += wt;
            }
            let inv = 1.0 / wsum.max(1e-6);
            row_out[x] = [acc[0] * inv, acc[1] * inv, acc[2] * inv];
        }
    });
    out
}

fn apply_gain(lab: [f32; 3], base: [f32; 3], k_l: f32, k_c: f32, strength: f32) -> [f32; 3] {
    let d = [lab[0] - base[0], lab[1] - base[1], lab[2] - base[2]];
    let d_l = d[0] * un_dim(d[0].abs(), k_l);
    let c = (d[1] * d[1] + d[2] * d[2]).sqrt();
    let g_c = un_dim(c, k_c);
    let ab = [d[1] * g_c, d[2] * g_c];
    let stretched = [d_l, ab[0], ab[1]];
    [
        base[0] + d[0] + (stretched[0] - d[0]) * strength,
        base[1] + d[1] + (stretched[1] - d[1]) * strength,
        base[2] + d[2] + (stretched[2] - d[2]) * strength,
    ]
}

/// Apply De-Bujack to a linear Rec.709 RGB buffer in [0, 1] (or sRGB-encoded if `encoded_srgb`).
pub fn apply_to_f32(image: &mut Array3<f32>, options: &PipelineOptions, encoded_srgb: bool) {
    if !options.bujack_enabled || options.bujack_strength <= 0.0 {
        return;
    }
    let (h, w, c) = image.dim();
    if c != 3 || w == 0 || h == 0 {
        return;
    }

    let mut lab = vec![[0.0_f32; 3]; w * h];
    lab.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut r = image[[y, x, 0]];
            let mut g = image[[y, x, 1]];
            let mut b = image[[y, x, 2]];
            if encoded_srgb {
                r = color_space::srgb_to_linear(r);
                g = color_space::srgb_to_linear(g);
                b = color_space::srgb_to_linear(b);
            }
            row[x] = lin_to_oklab(r, g, b);
        }
    });

    let radius = options.bujack_radius.round() as i32;
    let base = bilateral_base(&lab, w, h, radius, options.bujack_edge);
    let k_l = options.bujack_k_l;
    let k_c = options.bujack_k_c;
    let strength = options.bujack_strength;

    let mut out_rgb = vec![[0.0_f32; 3]; w * h];
    out_rgb.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = y * w + x;
            let out_lab = apply_gain(lab[i], base[i], k_l, k_c, strength);
            let rgb = gamut_clip(oklab_to_lin(out_lab));
            row[x] = if encoded_srgb {
                [
                    color_space::linear_to_srgb(rgb[0]),
                    color_space::linear_to_srgb(rgb[1]),
                    color_space::linear_to_srgb(rgb[2]),
                ]
            } else {
                [
                    rgb[0].clamp(0.0, 1.0),
                    rgb[1].clamp(0.0, 1.0),
                    rgb[2].clamp(0.0, 1.0),
                ]
            };
        }
    });
    for y in 0..h {
        for x in 0..w {
            let p = out_rgb[y * w + x];
            image[[y, x, 0]] = p[0];
            image[[y, x, 1]] = p[1];
            image[[y, x, 2]] = p[2];
        }
    }
}

/// Apply De-Bujack to linear print RGB stored as u16 (0 = 0.0, 65535 = 1.0).
pub fn apply_to_u16_linear(image: &mut Array3<u16>, options: &PipelineOptions) {
    if !options.bujack_enabled || options.bujack_strength <= 0.0 {
        return;
    }
    let (h, w, c) = image.dim();
    if c != 3 || w == 0 || h == 0 {
        return;
    }
    let scale = 1.0 / 65535.0_f32;
    let mut fimg = Array3::<f32>::zeros((h, w, 3));
    for y in 0..h {
        for x in 0..w {
            fimg[[y, x, 0]] = image[[y, x, 0]] as f32 * scale;
            fimg[[y, x, 1]] = image[[y, x, 1]] as f32 * scale;
            fimg[[y, x, 2]] = image[[y, x, 2]] as f32 * scale;
        }
    }
    apply_to_f32(&mut fimg, options, false);
    for y in 0..h {
        for x in 0..w {
            image[[y, x, 0]] = (fimg[[y, x, 0]].clamp(0.0, 1.0) * 65535.0).round() as u16;
            image[[y, x, 1]] = (fimg[[y, x, 1]].clamp(0.0, 1.0) * 65535.0).round() as u16;
            image[[y, x, 2]] = (fimg[[y, x, 2]].clamp(0.0, 1.0) * 65535.0).round() as u16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oklab_roundtrip_mid_gray() {
        let lab = lin_to_oklab(0.18, 0.18, 0.18);
        let rgb = oklab_to_lin(lab);
        for i in 0..3 {
            assert!((rgb[i] - 0.18).abs() < 2e-4, "ch{i} {} vs 0.18", rgb[i]);
        }
    }

    #[test]
    fn un_dim_is_identity_at_zero() {
        assert!((un_dim(0.0, 0.25) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn un_dim_grows_then_caps() {
        let k = 0.25;
        let g_small = un_dim(0.01, k);
        let g_mid = un_dim(0.12, k);
        let g_cap = un_dim(k, k);
        assert!(g_small > 1.0 && g_small < g_mid);
        assert!((g_cap - 1.0 / 0.15).abs() < 1e-5);
    }

    #[test]
    fn strength_zero_is_identity() {
        let mut img = Array3::<f32>::from_shape_fn((8, 8, 3), |(y, x, c)| {
            let t = (x + y) as f32 / 14.0;
            [t, t * 0.8, 0.2 + t * 0.5][c]
        });
        let orig = img.clone();
        let mut opts = PipelineOptions::default();
        opts.bujack_enabled = true;
        opts.bujack_strength = 0.0;
        opts.bujack_radius = 4.0;
        apply_to_f32(&mut img, &opts, false);
        for (a, b) in img.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn disabled_is_identity() {
        let mut img = Array3::<f32>::from_elem((4, 4, 3), 0.4);
        let orig = img.clone();
        let mut opts = PipelineOptions::default();
        opts.bujack_enabled = false;
        opts.bujack_strength = 1.0;
        apply_to_f32(&mut img, &opts, false);
        for (a, b) in img.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
