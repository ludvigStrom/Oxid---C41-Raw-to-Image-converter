//! Demosaicing: Bayer (H, W, 1) → RGB (H, W, 3).
//!
//! This module provides:
//! - `demosaic_bilinear`: simple bilinear interpolation (reference / fallback).
//! - `demosaic_edge_aware`: a lightweight edge-aware variant that preserves grain
//!   and reduces zippering by adapting green-channel interpolation to local
//!   gradients (Hamilton–Adams style) while keeping everything strictly linear.
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

/// Demosaic a single-channel Bayer image into linear RGB (H, W, 3) using
/// plain bilinear interpolation. Kept as a reference implementation and
/// fallback for non-RGGB patterns.
///
/// Input must be shape (H, W, 1). Output is (H, W, 3) with channels in R, G, B order.
pub fn demosaic_bilinear(bayer: &Array3<f32>, pattern: BayerPattern) -> Result<Array3<f32>> {
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

/// Demosaic a Bayer image using a simple edge-aware scheme:
///
/// - For RGGB, the green channel at R/B sites is interpolated directionally
///   based on local gradients (horizontal vs vertical), which greatly reduces
///   zippering along edges while keeping math linear.
/// - Red/blue at all sites still use bilinear interpolation.
/// - For non-RGGB patterns we currently fall back to `demosaic_bilinear`.
pub fn demosaic_edge_aware(bayer: &Array3<f32>, pattern: BayerPattern) -> Result<Array3<f32>> {
    // For now, only RGGB gets the edge-aware green treatment.
    if !matches!(pattern, BayerPattern::Rggb) {
        return demosaic_bilinear(bayer, pattern);
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
            let r =
                sample_channel_bilinear(&bayer_2d, y, x, height, width, pattern, Channel::R);
            let g = sample_channel_edge_aware_rggb_g(&bayer_2d, y, x, height, width);
            let b =
                sample_channel_bilinear(&bayer_2d, y, x, height, width, pattern, Channel::B);
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
        0.25
            * (s(y_i, x_i - 1)
                + s(y_i, x_i + 1)
                + s(y_i - 1, x_i)
                + s(y_i + 1, x_i))
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
