//! Load PNG (or other raster) images into Array3<f32> for the development pipeline.
//!
//! Accepts any size; 8-bit RGB is assumed sRGB and linearized to [0, 1] linear light.
//! No demosaic (image is already RGB).

use std::path::Path;

use anyhow::{bail, Context, Result};
use ndarray::Array3;

/// sRGB (gamma-encoded) to linear. Input and output in [0, 1].
#[inline]
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Load a PNG (or image crate–supported format) into RGB Array3<f32> (linear light).
///
/// * Format: JPEG, PNG, etc. via the `image` crate. Converted to RGB; alpha dropped.
/// * Assumes 8-bit values are sRGB-encoded; they are linearized so the pipeline (D-min, curve) sees linear light.
/// * Shape: (height, width, 3), channel order R, G, B.
pub fn load_png_as_ndarray(path: &Path) -> Result<Array3<f32>> {
    let img = image::open(path)
        .with_context(|| format!("Failed to open image {}", path.display()))?;

    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let w = width as usize;
    let h = height as usize;

    let raw = rgb.as_raw();
    if raw.len() != w * h * 3 {
        bail!(
            "Unexpected RGB length: {}x{}x3 = {}, got {}",
            w,
            h,
            w * h * 3,
            raw.len()
        );
    }

    let mut data = Vec::with_capacity(w * h * 3);
    for &v in raw {
        let s = (v as f32) / 255.0;
        data.push(srgb_to_linear(s));
    }

    let arr = Array3::from_shape_vec((h, w, 3), data)
        .with_context(|| format!("Failed to reshape image {}x{}x3", h, w))?;

    Ok(arr)
}
