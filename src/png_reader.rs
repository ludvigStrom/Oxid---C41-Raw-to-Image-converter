//! Load PNG, JPEG, TIFF (and other raster) images into Array3<f32> for the development pipeline.
//!
//! 8-bit RGB is assumed sRGB and linearized. 16-bit and float files are treated as
//! already-linear (typical for film-scan TIFF). No demosaic (image is already RGB).

use std::path::Path;

use anyhow::{bail, Context, Result};
use image::ColorType;
use ndarray::Array3;

use crate::color_space;

/// Load a PNG, JPEG, or TIFF (or image crate–supported format) into RGB Array3<f32> (linear light).
///
/// * Format: JPEG, PNG, TIFF, etc. via the `image` crate. Converted to RGB; alpha dropped.
/// * 8-bit values are sRGB-encoded and linearized.
/// * 16-bit / float values are treated as linear [0, 1] (no sRGB EOTF). Applying
///   sRGB decode to a linear 16-bit scan darkens midtones and shifts color.
/// * Shape: (height, width, 3), channel order R, G, B.
pub fn load_png_as_ndarray(path: &Path) -> Result<Array3<f32>> {
    let img =
        image::open(path).with_context(|| format!("Failed to open image {}", path.display()))?;

    let decode_srgb = matches!(
        img.color(),
        ColorType::L8 | ColorType::La8 | ColorType::Rgb8 | ColorType::Rgba8
    );

    let rgb = img.to_rgb32f();
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
    if decode_srgb {
        for &v in raw {
            data.push(color_space::srgb_to_linear(v));
        }
    } else {
        data.extend_from_slice(raw);
    }

    let arr = Array3::from_shape_vec((h, w, 3), data)
        .with_context(|| format!("Failed to reshape image {}x{}x3", h, w))?;

    Ok(arr)
}
