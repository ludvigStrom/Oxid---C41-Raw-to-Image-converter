//! Color and display helpers: sRGB/linear, XYZ/Lab, white-balance temperature,
//! density levels, and Rec.709 pre-LUT transform.

use ndarray::Array3;

/// Black-body temperature (K) to white-balance gains (R, G, B). 5500 K ≈ (1, 1, 1). Lower K = warm light → gains correct toward cool.
pub(crate) fn temp_k_to_wb_gains(temp_k: f32) -> (f32, f32, f32) {
    let t = (temp_k / 100.0).clamp(1.0, 400.0);
    let (r, g, b) = if t <= 66.0 {
        let g = 99.4708025861 * t.ln() - 161.1195681661;
        let b = if t > 19.0 {
            138.5177312231 * (t - 10.0).ln() - 305.0447927307
        } else {
            0.0
        };
        (255.0, g.clamp(0.0, 255.0), b.clamp(0.0, 255.0))
    } else {
        let r = 329.698727446 * (t - 60.0).powf(-0.1332047592);
        let g = 288.1221695283 * (t - 60.0).powf(-0.0755148492);
        (r.clamp(0.0, 255.0), g.clamp(0.0, 255.0), 255.0)
    };
    let r = r.max(1.0);
    let g = g.max(1.0);
    let b = b.max(1.0);
    let gain_r = 255.0 / r;
    let gain_g = 255.0 / g;
    let gain_b = 255.0 / b;
    let geom = (gain_r * gain_g * gain_b).cbrt();
    (gain_r / geom, gain_g / geom, gain_b / geom)
}

#[inline]
pub(crate) fn linear_to_srgb_u8(v: f32) -> u8 {
    let x = v.clamp(0.0, 1.0);
    let y = if x <= 0.003_130_8 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (y.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Normalize density to [0, 1] with levels remap: D/d_max → [0,1], then
/// stretch [black, white] → [0, 1], then midpoint gamma: v^(1/mid).
/// Identity when black=0, white=1, mid=1.
pub(crate) fn apply_density_levels(
    image: &mut Array3<f32>,
    d_max: f32,
    in_black: f32,
    in_white: f32,
    mid: f32,
) {
    let range = (in_white - in_black).max(1e-6);
    let inv_mid = 1.0 / mid.clamp(0.01, 10.0);
    let apply_gamma = (mid - 1.0).abs() > 1e-6;
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let mut v = (image[[y, x, ch]] / d_max).clamp(0.0, 1.0);
                v = ((v - in_black) / range).clamp(0.0, 1.0);
                if apply_gamma {
                    v = v.powf(inv_mid);
                }
                image[[y, x, ch]] = v;
            }
        }
    }
}

/// sRGB / Rec.709 OETF (linear → gamma-encoded).
#[inline]
fn linear_to_srgb(v: f32) -> f32 {
    let x = v.clamp(0.0, 1.0);
    if x <= 0.003_130_8 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

#[inline]
fn srgb_to_linear(v: f32) -> f32 {
    let x = v.clamp(0.0, 1.0);
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn rgb_linear_to_xyz(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // sRGB/Rec.709 primaries, D65 white.
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
    (x, y, z)
}

#[inline]
fn xyz_to_rgb_linear(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    // Inverse of rgb_linear_to_xyz for sRGB/Rec.709, D65.
    let r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    let b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    (r, g, b)
}

#[inline]
fn xyz_to_lab(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    // D65 reference white (sRGB).
    const XN: f32 = 0.95047;
    const YN: f32 = 1.0;
    const ZN: f32 = 1.08883;

    let xr = x / XN;
    let yr = y / YN;
    let zr = z / ZN;

    #[inline]
    fn f(t: f32) -> f32 {
        const EPS: f32 = 216.0 / 24389.0; // ~0.008856
        const KAPPA: f32 = 24389.0 / 27.0; // ~903.3
        if t > EPS {
            t.cbrt()
        } else {
            (KAPPA * t + 16.0) / 116.0
        }
    }

    let fx = f(xr);
    let fy = f(yr);
    let fz = f(zr);

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);
    (l, a, b)
}

#[inline]
fn lab_to_xyz(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    const XN: f32 = 0.95047;
    const YN: f32 = 1.0;
    const ZN: f32 = 1.08883;

    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;

    #[inline]
    fn f_inv(t: f32) -> f32 {
        const EPS: f32 = 216.0 / 24389.0;
        const KAPPA: f32 = 24389.0 / 27.0;
        let t3 = t * t * t;
        if t3 > EPS {
            t3
        } else {
            (116.0 * t - 16.0) / KAPPA
        }
    }

    let xr = f_inv(fx);
    let yr = f_inv(fy);
    let zr = f_inv(fz);

    (xr * XN, yr * YN, zr * ZN)
}

/// Apply Lab-space separation on an f32 RGB image in [0, 1]. Strength is
/// typically 0.0–1.0. Neutrals (low chroma) are largely preserved; mid-chroma
/// colors are pushed outward in the a/b plane to increase separation.
pub(crate) fn apply_lab_separation_f32(image: &mut Array3<f32>, strength: f32) {
    if strength.abs() < 1e-6 {
        return;
    }
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    let s = strength.clamp(-2.0, 2.0);

    for y in 0..h {
        for x in 0..w {
            let sr = image[[y, x, 0]].clamp(0.0, 1.0);
            let sg = image[[y, x, 1]].clamp(0.0, 1.0);
            let sb = image[[y, x, 2]].clamp(0.0, 1.0);

            let r_lin = srgb_to_linear(sr);
            let g_lin = srgb_to_linear(sg);
            let b_lin = srgb_to_linear(sb);

            let (xv, yv, zv) = rgb_linear_to_xyz(r_lin, g_lin, b_lin);
            let (l, a, b) = xyz_to_lab(xv, yv, zv);

            let c_ab = (a * a + b * b).sqrt();
            if c_ab < 1e-4 {
                // Near-neutral; keep as-is.
                continue;
            }
            let c_norm = (c_ab / 100.0).clamp(0.0, 1.0);
            // Bell-shaped mid-chroma emphasis over 0..1.
            let mid_boost = 1.0 + s * (c_norm * (1.0 - c_norm)) * 2.0;
            // Soften boost near very high chroma to avoid clipping.
            let edge_soften = 1.0 + 0.2 * s * (1.0 - c_norm);
            let gain = (mid_boost * edge_soften).max(0.0);

            let scale = gain;
            let a2 = a * scale;
            let b2 = b * scale;

            let (x2, y2, z2) = lab_to_xyz(l, a2, b2);
            let (r_lin2, g_lin2, b_lin2) = xyz_to_rgb_linear(x2, y2, z2);

            image[[y, x, 0]] = linear_to_srgb(r_lin2).clamp(0.0, 1.0);
            image[[y, x, 1]] = linear_to_srgb(g_lin2).clamp(0.0, 1.0);
            image[[y, x, 2]] = linear_to_srgb(b_lin2).clamp(0.0, 1.0);
        }
    }
}

/// Apply Lab separation to a u16 RGB image (0–65535) in-place by converting
/// to f32, running `apply_lab_separation_f32`, then quantizing back.
pub(crate) fn apply_lab_separation_u16(image: &mut Array3<u16>, strength: f32) {
    if strength.abs() < 1e-6 {
        return;
    }
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    let inv = 1.0 / 65535.0_f32;

    // Convert to f32 in [0,1]
    let mut fimg = Array3::<f32>::zeros((h, w, c));
    for y in 0..h {
        for x in 0..w {
            fimg[[y, x, 0]] = image[[y, x, 0]] as f32 * inv;
            fimg[[y, x, 1]] = image[[y, x, 1]] as f32 * inv;
            fimg[[y, x, 2]] = image[[y, x, 2]] as f32 * inv;
        }
    }

    apply_lab_separation_f32(&mut fimg, strength);

    // Quantize back.
    for y in 0..h {
        for x in 0..w {
            image[[y, x, 0]] = (fimg[[y, x, 0]].clamp(0.0, 1.0) * 65535.0).round() as u16;
            image[[y, x, 1]] = (fimg[[y, x, 1]].clamp(0.0, 1.0) * 65535.0).round() as u16;
            image[[y, x, 2]] = (fimg[[y, x, 2]].clamp(0.0, 1.0) * 65535.0).round() as u16;
        }
    }
}

/// Normalize density to [0, 1] with sRGB/Rec.709 gamma, then levels remap + midpoint.
/// The LUT handles the neg→pos inversion (print emulation), so we keep
/// density orientation: D / d_max → gamma-encode → levels → midpoint.
pub(crate) fn density_to_rec709_leveled(
    image: &mut Array3<f32>,
    in_black: f32,
    in_white: f32,
    mid: f32,
) {
    const D_MAX: f32 = 2.5;
    let range = (in_white - in_black).max(1e-6);
    let inv_mid = 1.0 / mid.clamp(0.01, 10.0);
    let apply_mid = (mid - 1.0).abs() > 1e-6;
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let norm = (image[[y, x, ch]] / D_MAX).clamp(0.0, 1.0);
                let gamma = linear_to_srgb(norm);
                let mut v = ((gamma - in_black) / range).clamp(0.0, 1.0);
                if apply_mid {
                    v = v.powf(inv_mid);
                }
                image[[y, x, ch]] = v;
            }
        }
    }
}
