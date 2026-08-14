//! Flat-field loading, blurring, and application for luminance calibration.
//!
//! RAW flat-field inputs are linearized and heavily blurred to remove grain/dust;
//! image inputs (e.g. 32f TIFF) are used as-is. Division against the map corrects
//! for light source and lens vignetting.

use std::path::Path;

use anyhow::Result;
use image::{imageops, imageops::FilterType, open, Rgb, Rgb32FImage};
use ndarray::Array3;

use crate::demosaic;
use crate::raw_reader;

/// Load a RAW flat-field frame through Step 1 (LibRaw) and Step 2 (Demosaic) only.
/// Returns linear RGB transmittance as `Array3<f32>` (H, W, 3). No D-min, no curve.
/// Use for luminance (flat-field) calibration reference.
pub fn load_flat_field_linear(path: &Path) -> Result<Array3<f32>> {
    let (bayer, pattern) = raw_reader::load_raw_as_ndarray(path)?;
    let mut img = demosaic::demosaic_quality(&bayer, pattern)?;
    img.mapv_inplace(|v| v.max(0.0));
    Ok(img)
}

/// Build a 1D Gaussian kernel (odd length, normalized). Sigma in pixels.
fn gaussian_1d_kernel(sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 {
        return vec![1.0];
    }
    let half_len = (3.0 * sigma).ceil().max(1.0) as usize;
    let len = 2 * half_len + 1;
    let mut k = Vec::with_capacity(len);
    let mut sum = 0.0_f32;
    for i in 0..len {
        let x = (i as f32) - (half_len as f32);
        let w = (-x * x / (2.0 * sigma * sigma)).exp();
        k.push(w);
        sum += w;
    }
    for w in k.iter_mut() {
        *w /= sum;
    }
    k
}

/// Separable Gaussian blur on (H, W, 3) f32 array. Uses full f32 precision to avoid banding.
/// Sigma in pixels; boundary uses edge clamping.
fn separable_gaussian_f32(image: &Array3<f32>, sigma: f32) -> Array3<f32> {
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    if sigma <= 0.0 {
        return image.to_owned();
    }
    let kernel = gaussian_1d_kernel(sigma);
    let half = kernel.len() / 2;

    // Horizontal pass: (y, x, ch) -> temp(y, x, ch)
    let mut temp = Array3::<f32>::zeros((h, w, 3));
    for y in 0..h {
        for ch in 0..3 {
            for x in 0..w {
                let mut acc = 0.0_f32;
                for (i, &k) in kernel.iter().enumerate() {
                    let xi = (x as i32 + i as i32 - half as i32).clamp(0, w as i32 - 1) as usize;
                    acc += image[(y, xi, ch)] * k;
                }
                temp[(y, x, ch)] = acc;
            }
        }
    }

    // Vertical pass: temp -> out
    let mut out = Array3::<f32>::zeros((h, w, 3));
    for x in 0..w {
        for ch in 0..3 {
            for y in 0..h {
                let mut acc = 0.0_f32;
                for (i, &k) in kernel.iter().enumerate() {
                    let yi = (y as i32 + i as i32 - half as i32).clamp(0, h as i32 - 1) as usize;
                    acc += temp[(yi, x, ch)] * k;
                }
                out[(y, x, ch)] = acc;
            }
        }
    }
    out
}

/// Apply a heavy Gaussian blur to a linear RGB flat-field image to remove film grain and dust,
/// leaving only low-frequency luminance falloff (light source + lens vignetting).
///
/// Input and output are `(height, width, 3)` arrays in linear [0, 1] space.
/// Uses a separable f32 Gaussian to avoid banding from external blur implementations.
pub fn blur_flat_field(input: &Array3<f32>, radius: f32) -> Array3<f32> {
    separable_gaussian_f32(input, radius)
}

/// Load a flat-field map from an image file (e.g. 32f TIFF saved from the GUI).
/// Interprets the data as linear RGB in [0, 1] (or higher).
fn load_flat_field_from_image(path: &Path) -> Result<Array3<f32>> {
    let img = open(path)?;
    let rgb = img.to_rgb32f();
    let (w, h) = rgb.dimensions();
    let mut out = Array3::<f32>::zeros((h as usize, w as usize, 3));
    for (x, y, pixel) in rgb.enumerate_pixels() {
        let [r, g, b] = pixel.0;
        let yi = y as usize;
        let xi = x as usize;
        out[(yi, xi, 0)] = r;
        out[(yi, xi, 1)] = g;
        out[(yi, xi, 2)] = b;
    }
    Ok(out)
}

/// Load or reconstruct a flat-field map for the pipeline.
/// - RAW inputs are linearized + heavily blurred
/// - Image inputs (TIFF/PNG/etc.) are treated as already-linear maps (no extra blur)
pub(crate) fn load_flat_field_map(path: &Path) -> Result<Array3<f32>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        // RAW formats -> linearize then blur to remove grain/dust.
        "arw" | "nef" | "nrw" | "cr2" | "cr3" | "crw" | "dng" | "raf" | "orf" | "rw2" => {
            let linear = load_flat_field_linear(path)?;
            Ok(blur_flat_field(&linear, 60.0))
        }
        // Everything else is treated as an already-prepared map (e.g. 32f TIFF).
        _ => load_flat_field_from_image(path),
    }
}

/// Resize a blurred flat-field to match the target image dimensions (height, width, 3).
pub(crate) fn resize_flat_field(flat: &Array3<f32>, height: usize, width: usize) -> Array3<f32> {
    let (fh, fw, fc) = flat.dim();
    assert_eq!(fc, 3, "resize_flat_field expects 3-channel input");
    if fh == height && fw == width {
        return flat.clone();
    }

    let mut img = Rgb32FImage::new(fw as u32, fh as u32);
    for y in 0..fh {
        for x in 0..fw {
            let r = flat[(y, x, 0)];
            let g = flat[(y, x, 1)];
            let b = flat[(y, x, 2)];
            img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    let resized = imageops::resize(&img, width as u32, height as u32, FilterType::Triangle);

    let mut out = Array3::<f32>::zeros((height, width, 3));
    for (x, y, pixel) in resized.enumerate_pixels() {
        let [r, g, b] = pixel.0;
        let yi = y as usize;
        let xi = x as usize;
        out[(yi, xi, 0)] = r;
        out[(yi, xi, 1)] = g;
        out[(yi, xi, 2)] = b;
    }

    out
}

/// Apply pixel-by-pixel flat-field division:
/// T_out(x, y) = T_in(x, y) / T_flat_blurred(x, y), with safe division.
pub(crate) fn apply_flat_field_division(image: &mut Array3<f32>, flat_blurred: &Array3<f32>) {
    let (h, w, c) = image.dim();
    assert_eq!(c, 3, "apply_flat_field_division expects 3-channel image");

    let flat_resampled = resize_flat_field(flat_blurred, h, w);
    let eps = 1.0e-6_f32;

    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let denom = flat_resampled[(y, x, ch)].max(eps);
                image[(y, x, ch)] /= denom;
            }
        }
    }
}
