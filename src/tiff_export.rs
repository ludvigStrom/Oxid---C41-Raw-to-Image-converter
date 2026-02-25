//! TIFF export: 32-bit float (max data) or 16-bit integer (display/print).
//!
//! **32-bit float** — No clamping, no quantization. Preserves full dynamic range and
//! precision from the pipeline (e.g. values >1 after D-min). Best for archival and
//! further linear processing.
//!
//! **16-bit** — Clamps to [0, 1] and quantizes to 0–65535. Smaller files, universal
//! compatibility; use for viewing or printing when float is not needed.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyhow::{bail, Context, Result};
use ndarray::Array3;
use tiff::encoder::{colortype::RGB16, colortype::RGB32Float, TiffEncoder};

/// Output bit depth / format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TiffFormat {
    /// 32-bit float per channel — no clipping, no quantization. Preserves all data.
    #[default]
    Float32,
    /// 16-bit integer per channel — clamp to [0,1], scale to 0–65535. Smaller, widely compatible.
    U16,
}

/// Scale linear f32 [0, 1] to u16 [0, 65535]. Values outside [0, 1] are clamped.
#[inline]
fn f32_to_u16(v: f32) -> u16 {
    let clamped = v.clamp(0.0, 1.0);
    (clamped * 65535.0).round() as u16
}

/// Write image as uncompressed TIFF. Format chosen by `format`.
pub fn write_tiff(image: &Array3<f32>, path: &Path, format: TiffFormat) -> Result<()> {
    let (height, width, c) = image.dim();
    if c != 3 {
        bail!("TIFF export expects RGB (3 channels), got {}", c);
    }

    let width_u = u32::try_from(width).context("Image width too large for TIFF")?;
    let height_u = u32::try_from(height).context("Image height too large for TIFF")?;

    let file = File::create(path).with_context(|| format!("Failed to create {}", path.display()))?;
    let writer = BufWriter::new(file);
    let mut tiff = TiffEncoder::new(writer).with_context(|| "Failed to create TIFF encoder")?;

    match format {
        TiffFormat::Float32 => {
            let mut buf: Vec<f32> = Vec::with_capacity(height * width * 3);
            for row in image.axis_iter(ndarray::Axis(0)) {
                for pixel in row.axis_iter(ndarray::Axis(0)) {
                    buf.push(pixel[0]);
                    buf.push(pixel[1]);
                    buf.push(pixel[2]);
                }
            }
            tiff.write_image::<RGB32Float>(width_u, height_u, &buf)
                .with_context(|| format!("Failed to write 32-bit float TIFF to {}", path.display()))?;
        }
        TiffFormat::U16 => {
            let mut buf: Vec<u16> = Vec::with_capacity(height * width * 3);
            for row in image.axis_iter(ndarray::Axis(0)) {
                for pixel in row.axis_iter(ndarray::Axis(0)) {
                    buf.push(f32_to_u16(pixel[0]));
                    buf.push(f32_to_u16(pixel[1]));
                    buf.push(f32_to_u16(pixel[2]));
                }
            }
            tiff.write_image::<RGB16>(width_u, height_u, &buf)
                .with_context(|| format!("Failed to write 16-bit TIFF to {}", path.display()))?;
        }
    }

    Ok(())
}

