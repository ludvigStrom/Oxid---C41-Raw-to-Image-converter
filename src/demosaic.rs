//! Demosaicing: Bayer (H, W, 1) → RGB (H, W, 3).
//!
//! This module provides:
//! - `demosaic_bilinear`: simple bilinear interpolation (reference / fallback).
//! - `demosaic_edge_aware`: edge-aware green (Hamilton–Adams) + bilinear R/B.
//! - `demosaic_quality`: best-in-class — edge-aware green plus **color-difference**
//!   (R−G, B−G) interpolation for R and B. Interpolating differences instead of
//!   raw R/B greatly reduces false color and zippering while preserving detail.
//!
//! Sony a7R II uses RGGB (red at top-left of 2×2 Bayer block).

use anyhow::{bail, Result};
use ndarray::{Array3, ArrayView2};
use rayon::prelude::*;

/// Bayer filter layout of the 2×2 block (first row, then second row).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(dead_code)] // Grbg, Gbrg, Bggr for other cameras
pub enum BayerPattern {
    /// Red at (0,0), Green at (0,1) and (1,0), Blue at (1,1). Used by Sony a7R II.
    #[default]
    Rggb,
    /// Green at (0,0), Red at (0,1), Blue at (1,0), Green at (1,1).
    Grbg,
    /// Green at (0,0), Blue at (0,1), Red at (1,0), Green at (1,1).
    Gbrg,
    /// Blue at (0,0), Green at (0,1) and (1,0), Red at (1,1).
    Bggr,
}

/// Top-level CFA descriptor passed through the pipeline.
///
/// Wraps either a standard 2×2 Bayer variant or a Fujifilm 6×6 X-Trans tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfaPattern {
    /// Standard 2×2 Bayer (RGGB / GRBG / GBRG / BGGR).
    Bayer(BayerPattern),
    /// Fujifilm X-Trans 6×6 CFA.
    ///
    /// `pattern[r][c]` gives channel (0=R, 1=G, 2=B) for any cropped-image pixel
    /// at row `y` and column `x`, where `r = y % 6` and `c = x % 6`.
    XTrans([[u8; 6]; 6]),
}

/// Demosaic a single-channel CFA image into linear RGB (H, W, 3) using
/// plain bilinear interpolation. Kept as a reference implementation and
/// fallback for non-RGGB patterns. Supports both standard Bayer and X-Trans.
///
/// Input must be shape (H, W, 1). Output is (H, W, 3) with channels in R, G, B order.
pub fn demosaic_bilinear(bayer: &Array3<f32>, pattern: CfaPattern) -> Result<Array3<f32>> {
    match pattern {
        CfaPattern::Bayer(p) => demosaic_bilinear_bayer(bayer, p),
        CfaPattern::XTrans(xt) => demosaic_xtrans(bayer, xt),
    }
}

fn demosaic_bilinear_bayer(bayer: &Array3<f32>, pattern: BayerPattern) -> Result<Array3<f32>> {
    let (height, width, c) = bayer.dim();
    if c != 1 {
        bail!("Demosaic expects (H, W, 1), got channels {}", c);
    }

    let bayer_2d = bayer.slice(ndarray::s![.., .., 0]); // (H, W)

    let mut rgb = Array3::<f32>::zeros((height, width, 3));
    let ptr_base = rgb.as_mut_ptr() as usize;

    // Parallel over rows: each thread writes to distinct rows (safe). Pass address as usize so the closure is Sync.
    (0..height).into_par_iter().for_each(|y| {
        let ptr = ptr_base as *mut f32;
        for x in 0..width {
            let r = sample_channel_bilinear(&bayer_2d, y, x, height, width, pattern, Channel::R);
            let g = sample_channel_bilinear(&bayer_2d, y, x, height, width, pattern, Channel::G);
            let b = sample_channel_bilinear(&bayer_2d, y, x, height, width, pattern, Channel::B);
            let base = y * width * 3 + x * 3;
            unsafe {
                *ptr.add(base) = r;
                *ptr.add(base + 1) = g;
                *ptr.add(base + 2) = b;
            }
        }
    });

    Ok(rgb)
}

/// Demosaic a CFA image using a simple edge-aware scheme:
///
/// - For RGGB Bayer, the green channel at R/B sites is interpolated directionally
///   based on local gradients (horizontal vs vertical), which greatly reduces
///   zippering along edges while keeping math linear.
/// - Red/blue at all sites still use bilinear interpolation.
/// - For non-RGGB Bayer patterns falls back to `demosaic_bilinear_bayer`.
/// - For X-Trans, delegates to `demosaic_xtrans`.
pub fn demosaic_edge_aware(bayer: &Array3<f32>, pattern: CfaPattern) -> Result<Array3<f32>> {
    match pattern {
        CfaPattern::Bayer(p) => demosaic_edge_aware_bayer(bayer, p),
        CfaPattern::XTrans(xt) => demosaic_xtrans(bayer, xt),
    }
}

fn demosaic_edge_aware_bayer(bayer: &Array3<f32>, pattern: BayerPattern) -> Result<Array3<f32>> {
    if !matches!(pattern, BayerPattern::Rggb) {
        return demosaic_bilinear_bayer(bayer, pattern);
    }

    let (height, width, c) = bayer.dim();
    if c != 1 {
        bail!("Demosaic expects (H, W, 1), got channels {}", c);
    }

    let bayer_2d = bayer.slice(ndarray::s![.., .., 0]); // (H, W)

    let mut rgb = Array3::<f32>::zeros((height, width, 3));
    let ptr_base = rgb.as_mut_ptr() as usize;

    (0..height).into_par_iter().for_each(|y| {
        let ptr = ptr_base as *mut f32;
        for x in 0..width {
            let r = sample_channel_bilinear(&bayer_2d, y, x, height, width, pattern, Channel::R);
            let g = sample_channel_edge_aware_rggb_g(&bayer_2d, y, x, height, width);
            let b = sample_channel_bilinear(&bayer_2d, y, x, height, width, pattern, Channel::B);
            let base = y * width * 3 + x * 3;
            unsafe {
                *ptr.add(base) = r;
                *ptr.add(base + 1) = g;
                *ptr.add(base + 2) = b;
            }
        }
    });

    Ok(rgb)
}

/// Best-in-class demosaic: edge-aware green plus **color-difference** (R−G, B−G)
/// interpolation for red and blue on RGGB Bayer; for other Bayer patterns falls
/// back to `demosaic_edge_aware_bayer`; for X-Trans delegates to `demosaic_xtrans`.
///
/// High-frequency content in R and B is highly correlated with G. Interpolating
/// (R−G) and (B−G) instead of R and B directly sharply reduces false color and
/// zippering while preserving detail and grain.
pub fn demosaic_quality(bayer: &Array3<f32>, pattern: CfaPattern) -> Result<Array3<f32>> {
    match pattern {
        CfaPattern::Bayer(p) => demosaic_quality_bayer(bayer, p),
        CfaPattern::XTrans(xt) => demosaic_xtrans(bayer, xt),
    }
}

fn demosaic_quality_bayer(bayer: &Array3<f32>, pattern: BayerPattern) -> Result<Array3<f32>> {
    if !matches!(pattern, BayerPattern::Rggb) {
        return demosaic_edge_aware_bayer(bayer, pattern);
    }

    let (height, width, c) = bayer.dim();
    if c != 1 {
        bail!("Demosaic expects (H, W, 1), got channels {}", c);
    }

    let bayer_2d = bayer.slice(ndarray::s![.., .., 0]);
    let h = height;
    let w = width;

    // Pass 1: interpolate green at every pixel (edge-aware), parallel over rows.
    let mut g_plane = ndarray::Array2::<f32>::zeros((height, width));
    let g_ptr = g_plane.as_mut_ptr() as usize;
    (0..height).into_par_iter().for_each(|y| {
        let ptr = g_ptr as *mut f32;
        for x in 0..width {
            let g = sample_channel_edge_aware_rggb_g(&bayer_2d, y, x, h, w);
            unsafe { *ptr.add(y * width + x) = g };
        }
    });

    // Pass 2: at each pixel, interpolate (R-G) and (B-G) from native sites, then R = (R-G)+G, B = (B-G)+G.
    let mut rgb = Array3::<f32>::zeros((height, width, 3));
    let ptr_base = rgb.as_mut_ptr() as usize;
    let g_ptr = g_plane.as_ptr() as usize;

    (0..height).into_par_iter().for_each(|y| {
        let ptr = ptr_base as *mut f32;
        let g_base = g_ptr as *const f32;
        for x in 0..width {
            let g = unsafe { *g_base.add(y * width + x) };
            let r_minus_g = interpolate_r_minus_g_rggb(&bayer_2d, &g_plane, y, x, h, w);
            let b_minus_g = interpolate_b_minus_g_rggb(&bayer_2d, &g_plane, y, x, h, w);
            let r = r_minus_g + g;
            let b = b_minus_g + g;
            let base = y * width * 3 + x * 3;
            unsafe {
                *ptr.add(base) = r;
                *ptr.add(base + 1) = g;
                *ptr.add(base + 2) = b;
            }
        }
    });

    Ok(rgb)
}

/// X-Trans 6×6 demosaic via inverse-distance-squared weighted interpolation.
///
/// For each output pixel the native channel is read directly. The two missing
/// channels are computed as a 1/d² weighted average of all same-colour pixels
/// within a ±3-pixel search window (7×7 = 49 candidates). Because the 6×6
/// X-Trans tile fits inside that window, every colour is guaranteed to appear
/// at least once for any interior pixel. Border pixels within 3 pixels of an
/// edge may have fewer samples but are typically cropped out of the final image.
///
/// This is equivalent in quality to bilinear for standard Bayer and is a solid
/// foundation; a Markesteijn-style green-first pass can be added later.
fn demosaic_xtrans(bayer: &Array3<f32>, xtrans: [[u8; 6]; 6]) -> Result<Array3<f32>> {
    let (height, width, c) = bayer.dim();
    if c != 1 {
        bail!("Demosaic expects (H, W, 1), got channels {}", c);
    }

    let bayer_2d = bayer.slice(ndarray::s![.., .., 0]);
    let h = height as i32;
    let w = width as i32;

    let mut rgb = Array3::<f32>::zeros((height, width, 3));
    let ptr_base = rgb.as_mut_ptr() as usize;

    (0..height).into_par_iter().for_each(|y| {
        let ptr = ptr_base as *mut f32;
        let y_i = y as i32;
        for x in 0..width {
            let x_i = x as i32;
            let native_ch = xtrans[y % 6][x % 6] as usize;

            let mut sum = [0f32; 3];
            let mut wt = [0f32; 3];

            // Native pixel — exact value, weight 1.
            sum[native_ch] += bayer_2d[[y, x]];
            wt[native_ch] += 1.0;

            // Search ±3 (7×7 window) for the two missing channels.
            for dy in -3i32..=3 {
                for dx in -3i32..=3 {
                    if dy == 0 && dx == 0 {
                        continue;
                    }
                    let sy = y_i + dy;
                    let sx = x_i + dx;
                    if sy < 0 || sy >= h || sx < 0 || sx >= w {
                        continue;
                    }
                    let sy = sy as usize;
                    let sx = sx as usize;
                    let ch = xtrans[sy % 6][sx % 6] as usize;
                    let inv_d2 = 1.0 / (dy * dy + dx * dx) as f32;
                    sum[ch] += bayer_2d[[sy, sx]] * inv_d2;
                    wt[ch] += inv_d2;
                }
            }

            let base = y * width * 3 + x * 3;
            unsafe {
                *ptr.add(base) = if wt[0] > 0.0 {
                    sum[0] / wt[0]
                } else {
                    bayer_2d[[y, x]]
                };
                *ptr.add(base + 1) = if wt[1] > 0.0 {
                    sum[1] / wt[1]
                } else {
                    bayer_2d[[y, x]]
                };
                *ptr.add(base + 2) = if wt[2] > 0.0 {
                    sum[2] / wt[2]
                } else {
                    bayer_2d[[y, x]]
                };
            }
        }
    });

    Ok(rgb)
}

/// Average of (R - G) at the four nearest R sites (RGGB: even, even).
#[inline]
fn interpolate_r_minus_g_rggb(
    bayer: &ArrayView2<f32>,
    g_plane: &ndarray::Array2<f32>,
    y: usize,
    x: usize,
    h: usize,
    w: usize,
) -> f32 {
    let yt = (y >> 1) << 1;
    let xt = (x >> 1) << 1;
    let mut sum = 0f32;
    let mut n = 0u32;
    for dy in [0, 2] {
        for dx in [0, 2] {
            let (yy, xx) = clamp(yt as i32 + dy, xt as i32 + dx, h, w);
            sum += bayer[[yy, xx]] - g_plane[[yy, xx]];
            n += 1;
        }
    }
    sum / (n as f32)
}

/// Average of (B - G) at the four nearest B sites (RGGB: odd, odd).
#[inline]
fn interpolate_b_minus_g_rggb(
    bayer: &ArrayView2<f32>,
    g_plane: &ndarray::Array2<f32>,
    y: usize,
    x: usize,
    h: usize,
    w: usize,
) -> f32 {
    let base_y = (y >> 1) << 1;
    let base_x = (x >> 1) << 1;
    let mut sum = 0f32;
    let mut n = 0u32;
    for dy in [1, 3] {
        for dx in [1, 3] {
            let (yy, xx) = clamp(base_y as i32 + dy, base_x as i32 + dx, h, w);
            sum += bayer[[yy, xx]] - g_plane[[yy, xx]];
            n += 1;
        }
    }
    sum / (n as f32)
}

#[derive(Clone, Copy)]
enum Channel {
    R,
    G,
    B,
}

/// Bilinear sample of one color channel at (y, x) from the Bayer plane.
fn sample_channel_bilinear(
    bayer: &ArrayView2<f32>,
    y: usize,
    x: usize,
    h: usize,
    w: usize,
    pattern: BayerPattern,
    channel: Channel,
) -> f32 {
    match pattern {
        BayerPattern::Rggb => sample_rggb(bayer, y, x, h, w, channel),
        BayerPattern::Grbg => sample_grbg(bayer, y, x, h, w, channel),
        BayerPattern::Gbrg => sample_gbrg(bayer, y, x, h, w, channel),
        BayerPattern::Bggr => sample_bggr(bayer, y, x, h, w, channel),
    }
}

/// Edge-aware interpolation of the green channel for RGGB Bayer at (y, x).
///
/// At R/B sites (even+even or odd+odd), choose between horizontal and vertical
/// interpolation based on local gradients (Hamilton–Adams style). At native G
/// sites, just return the observed value.
fn sample_channel_edge_aware_rggb_g(
    bayer: &ArrayView2<f32>,
    y: usize,
    x: usize,
    h: usize,
    w: usize,
) -> f32 {
    let y_i = y as i32;
    let x_i = x as i32;
    let h_u = h as usize;
    let w_u = w as usize;

    // In RGGB, G is at (even, odd) and (odd, even): parity (y + x) % 2 == 1.
    if (y + x) % 2 == 1 {
        return bayer[[y, x]];
    }

    // Clamp helpers for neighbors.
    let s = |yy: i32, xx: i32| {
        let (yyc, xyc) = clamp(yy, xx, h_u, w_u);
        bayer[[yyc, xyc]]
    };

    // First-order gradients (difference of neighbors).
    let gh = (s(y_i, x_i - 1) - s(y_i, x_i + 1)).abs();
    let gv = (s(y_i - 1, x_i) - s(y_i + 1, x_i)).abs();

    // Second-order term (Laplacian-like) to stabilize interpolation.
    let lh = (2.0 * s(y_i, x_i) - s(y_i, x_i - 2) - s(y_i, x_i + 2)).abs();
    let lv = (2.0 * s(y_i, x_i) - s(y_i - 2, x_i) - s(y_i + 2, x_i)).abs();

    let dh = gh + lh;
    let dv = gv + lv;

    if dh < dv {
        // Prefer horizontal interpolation.
        0.5 * (s(y_i, x_i - 1) + s(y_i, x_i + 1))
    } else if dv < dh {
        // Prefer vertical interpolation.
        0.5 * (s(y_i - 1, x_i) + s(y_i + 1, x_i))
    } else {
        // Isotropic case: average of both directions.
        0.25 * (s(y_i, x_i - 1) + s(y_i, x_i + 1) + s(y_i - 1, x_i) + s(y_i + 1, x_i))
    }
}

#[inline]
fn clamp(y: i32, x: i32, h: usize, w: usize) -> (usize, usize) {
    let y = y.clamp(0, (h as i32) - 1) as usize;
    let x = x.clamp(0, (w as i32) - 1) as usize;
    (y, x)
}

// RGGB: R(0,0), G(0,1), G(1,0), B(1,1) in 2×2 block
fn sample_rggb(
    bayer: &ArrayView2<f32>,
    y: usize,
    x: usize,
    h: usize,
    w: usize,
    channel: Channel,
) -> f32 {
    let y = y as i32;
    let x = x as i32;
    let h = h as i32;
    let w = w as i32;

    match channel {
        Channel::R => {
            // R at (even, even). 4 nearest R: (yt&!1, xt&!1), (yt&!1, xt+2), (yt+2, xt&!1), (yt+2, xt+2)
            let yt = (y >> 1) << 1;
            let xt = (x >> 1) << 1;
            let mut sum = 0f32;
            let mut n = 0;
            for dy in [0, 2] {
                for dx in [0, 2] {
                    let (yy, xx) = clamp(yt + dy, xt + dx, h as usize, w as usize);
                    sum += bayer[[yy, xx]];
                    n += 1;
                }
            }
            sum / (n as f32)
        }
        Channel::G => {
            // G at (even, odd) and (odd, even)
            if (y as usize + x as usize) % 2 == 1 {
                return bayer[[y as usize, x as usize]];
            }
            let mut sum = 0f32;
            let mut n = 0;
            for (dy, dx) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let (yy, xx) = clamp(y + dy, x + dx, h as usize, w as usize);
                sum += bayer[[yy, xx]];
                n += 1;
            }
            sum / (n as f32)
        }
        Channel::B => {
            // B at (odd, odd). 4 nearest B: (yt+1, xt+1), (yt+1, xt+3), (yt+3, xt+1), (yt+3, xt+3)
            let yt = (y >> 1) << 1;
            let xt = (x >> 1) << 1;
            let mut sum = 0f32;
            let mut n = 0;
            for dy in [1, 3] {
                for dx in [1, 3] {
                    let (yy, xx) = clamp(yt + dy, xt + dx, h as usize, w as usize);
                    sum += bayer[[yy, xx]];
                    n += 1;
                }
            }
            sum / (n as f32)
        }
    }
}

// GRBG: G(0,0), R(0,1), B(1,0), G(1,1)
fn sample_grbg(
    bayer: &ArrayView2<f32>,
    y: usize,
    x: usize,
    h: usize,
    w: usize,
    channel: Channel,
) -> f32 {
    let y = y as i32;
    let x = x as i32;
    let h = h as i32;
    let w = w as i32;

    match channel {
        Channel::R => {
            let yt = (y >> 1) << 1;
            let xt = (x >> 1) << 1;
            let mut sum = 0f32;
            let mut n = 0;
            for dy in [0, 2] {
                for dx in [1, 3] {
                    let (yy, xx) = clamp(yt + dy, xt + dx, h as usize, w as usize);
                    sum += bayer[[yy, xx]];
                    n += 1;
                }
            }
            sum / (n as f32)
        }
        Channel::G => {
            if (y as usize + x as usize) % 2 == 0 {
                return bayer[[y as usize, x as usize]];
            }
            let mut sum = 0f32;
            let mut n = 0;
            for (dy, dx) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let (yy, xx) = clamp(y + dy, x + dx, h as usize, w as usize);
                sum += bayer[[yy, xx]];
                n += 1;
            }
            sum / (n as f32)
        }
        Channel::B => {
            let yt = (y >> 1) << 1;
            let xt = (x >> 1) << 1;
            let mut sum = 0f32;
            let mut n = 0;
            for dy in [1, 3] {
                for dx in [0, 2] {
                    let (yy, xx) = clamp(yt + dy, xt + dx, h as usize, w as usize);
                    sum += bayer[[yy, xx]];
                    n += 1;
                }
            }
            sum / (n as f32)
        }
    }
}

// GBRG: G(0,0), B(0,1), R(1,0), G(1,1)
fn sample_gbrg(
    bayer: &ArrayView2<f32>,
    y: usize,
    x: usize,
    h: usize,
    w: usize,
    channel: Channel,
) -> f32 {
    let y = y as i32;
    let x = x as i32;
    let h = h as i32;
    let w = w as i32;

    match channel {
        Channel::R => {
            let yt = (y >> 1) << 1;
            let xt = (x >> 1) << 1;
            let mut sum = 0f32;
            let mut n = 0;
            for dy in [1, 3] {
                for dx in [0, 2] {
                    let (yy, xx) = clamp(yt + dy, xt + dx, h as usize, w as usize);
                    sum += bayer[[yy, xx]];
                    n += 1;
                }
            }
            sum / (n as f32)
        }
        Channel::G => {
            if (y as usize + x as usize) % 2 == 0 {
                return bayer[[y as usize, x as usize]];
            }
            let mut sum = 0f32;
            let mut n = 0;
            for (dy, dx) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let (yy, xx) = clamp(y + dy, x + dx, h as usize, w as usize);
                sum += bayer[[yy, xx]];
                n += 1;
            }
            sum / (n as f32)
        }
        Channel::B => {
            let yt = (y >> 1) << 1;
            let xt = (x >> 1) << 1;
            let mut sum = 0f32;
            let mut n = 0;
            for dy in [0, 2] {
                for dx in [1, 3] {
                    let (yy, xx) = clamp(yt + dy, xt + dx, h as usize, w as usize);
                    sum += bayer[[yy, xx]];
                    n += 1;
                }
            }
            sum / (n as f32)
        }
    }
}

// BGGR: B(0,0), G(0,1), G(1,0), R(1,1)
fn sample_bggr(
    bayer: &ArrayView2<f32>,
    y: usize,
    x: usize,
    h: usize,
    w: usize,
    channel: Channel,
) -> f32 {
    let y = y as i32;
    let x = x as i32;
    let h = h as i32;
    let w = w as i32;

    match channel {
        Channel::R => {
            let yt = (y >> 1) << 1;
            let xt = (x >> 1) << 1;
            let mut sum = 0f32;
            let mut n = 0;
            for dy in [1, 3] {
                for dx in [1, 3] {
                    let (yy, xx) = clamp(yt + dy, xt + dx, h as usize, w as usize);
                    sum += bayer[[yy, xx]];
                    n += 1;
                }
            }
            sum / (n as f32)
        }
        Channel::G => {
            if (y as usize + x as usize) % 2 == 1 {
                return bayer[[y as usize, x as usize]];
            }
            let mut sum = 0f32;
            let mut n = 0;
            for (dy, dx) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let (yy, xx) = clamp(y + dy, x + dx, h as usize, w as usize);
                sum += bayer[[yy, xx]];
                n += 1;
            }
            sum / (n as f32)
        }
        Channel::B => {
            let yt = (y >> 1) << 1;
            let xt = (x >> 1) << 1;
            let mut sum = 0f32;
            let mut n = 0;
            for dy in [0, 2] {
                for dx in [0, 2] {
                    let (yy, xx) = clamp(yt + dy, xt + dx, h as usize, w as usize);
                    sum += bayer[[yy, xx]];
                    n += 1;
                }
            }
            sum / (n as f32)
        }
    }
}
