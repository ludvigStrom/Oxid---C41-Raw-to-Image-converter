//! Load PNG (or other raster) images into Array3<f32> for the development pipeline.
//!
//! Accepts any size; 8-bit RGB is normalized to [0, 1]. No demosaic (image is already RGB).

use std::path::Path;

use anyhow::{bail, Context, Result};
use ndarray::Array3;

/// Load a PNG (or image crate–supported format) into RGB Array3<f32>.
///
/// * Format: JPEG, PNG, etc. via the `image` crate. Converted to RGB; alpha dropped.
/// * Normalization: 8-bit channels → [0.0, 1.0] (divide by 255).
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
        data.push((v as f32) / 255.0);
    }

    let arr = Array3::from_shape_vec((h, w, 3), data)
        .with_context(|| format!("Failed to reshape image {}x{}x3", h, w))?;

    Ok(arr)
}
