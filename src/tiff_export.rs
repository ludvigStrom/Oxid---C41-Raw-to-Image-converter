//! Write 16-bit uncompressed RGB TIFF from linear f32 image (H, W, 3).

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyhow::{bail, Context, Result};
use ndarray::Array3;
use tiff::encoder::{TiffEncoder, colortype::RGB16};

/// Scale linear f32 [0, 1] to u16 [0, 65535]. Values outside [0, 1] are clamped.
#[inline]
fn f32_to_u16(v: f32) -> u16 {
    let clamped = v.clamp(0.0, 1.0);
    (clamped * 65535.0).round() as u16
}

/// Write image as uncompressed 16-bit RGB TIFF.
///
/// `image` must be shape (height, width, 3) with channels R, G, B in [0.0, 1.0] (values outside are clamped).
pub fn write_rgb16_tiff(image: &Array3<f32>, path: &Path) -> Result<()> {
    let (height, width, c) = image.dim();
    if c != 3 {
        bail!("TIFF export expects RGB (3 channels), got {}", c);
    }

    let mut buf: Vec<u16> = Vec::with_capacity(height * width * 3);
    for row in image.axis_iter(ndarray::Axis(0)) {
        for pixel in row.axis_iter(ndarray::Axis(0)) {
            buf.push(f32_to_u16(pixel[0]));
            buf.push(f32_to_u16(pixel[1]));
            buf.push(f32_to_u16(pixel[2]));
        }
    }

    let width = u32::try_from(width).context("Image width too large for TIFF")?;
    let height = u32::try_from(height).context("Image height too large for TIFF")?;

    let file = File::create(path).with_context(|| format!("Failed to create {}", path.display()))?;
    let writer = BufWriter::new(file);

    let mut tiff = TiffEncoder::new(writer).with_context(|| "Failed to create TIFF encoder")?;
    tiff.write_image::<RGB16>(width, height, &buf)
        .with_context(|| format!("Failed to write TIFF to {}", path.display()))?;

    Ok(())
}
