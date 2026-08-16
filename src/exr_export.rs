use std::path::Path;

use anyhow::{anyhow, Result};
use exr::prelude::*;
use ndarray::Array3;

/// Write an RGB EXR from an f32 image in [0, 1].
///
/// The array is expected to be (height, width, 3).
pub fn write_exr_f32(image: &Array3<f32>, path: &Path) -> Result<()> {
    let (height, width, channels) = image.dim();
    if channels != 3 {
        return Err(anyhow!(
            "write_exr_f32 expects 3 channels, got {}",
            channels
        ));
    }

    let data = image
        .as_slice()
        .ok_or_else(|| anyhow!("EXR export requires contiguous f32 image data"))?;

    write_rgb_file(path, width, height, |x, y| {
        let base = (y * width + x) * 3;
        (data[base], data[base + 1], data[base + 2])
    })
    .map_err(|e| anyhow!("Failed to write EXR file {}: {}", path.display(), e))?;

    Ok(())
}

/// Write linear ACES2065-1 (AP0) RGB to EXR. Same f32 write as [write_exr_f32];
/// use this for the ACES archival/VFX branch. Color space metadata (e.g. chromaticities)
/// can be added here if the `exr` crate exposes attribute APIs.
///
/// The array is expected to be (height, width, 3), linear ACES2065-1.
pub fn write_exr_aces2065_1(image: &Array3<f32>, path: &Path) -> Result<()> {
    write_exr_f32(image, path)
}

/// Write an RGB EXR from a u16 image by normalizing to [0, 1] f32.
///
/// The array is expected to be (height, width, 3).
pub fn write_exr_u16(image: &Array3<u16>, path: &Path) -> Result<()> {
    let (height, width, channels) = image.dim();
    if channels != 3 {
        return Err(anyhow!(
            "write_exr_u16 expects 3 channels, got {}",
            channels
        ));
    }

    let data = image
        .as_slice()
        .ok_or_else(|| anyhow!("EXR export requires contiguous u16 image data"))?;

    let max_u16 = std::u16::MAX as f32;

    write_rgb_file(path, width, height, |x, y| {
        let base = (y * width + x) * 3;
        let r = data[base] as f32 / max_u16;
        let g = data[base + 1] as f32 / max_u16;
        let b = data[base + 2] as f32 / max_u16;
        (r, g, b)
    })
    .map_err(|e| anyhow!("Failed to write EXR file {}: {}", path.display(), e))?;

    Ok(())
}
