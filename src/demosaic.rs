//! Linear bilinear demosaic: Bayer (H, W, 1) → RGB (H, W, 3).
//!
//! Purely linear interpolation; no edge-aware or iterative methods.
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

/// Demosaic a single-channel Bayer image into linear RGB (H, W, 3).
///
/// Input must be shape (H, W, 1). Output is (H, W, 3) with channels in R, G, B order.
/// Uses bilinear interpolation only; no gamma or non-linear filtering.
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
