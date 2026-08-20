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
use serde::{Deserialize, Serialize};
use tiff::encoder::{colortype::RGB32Float, colortype::RGB16, TiffEncoder};
use tiff::tags::Tag;

/// Output bit depth / format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

    let file =
        File::create(path).with_context(|| format!("Failed to create {}", path.display()))?;
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
                .with_context(|| {
                    format!("Failed to write 32-bit float TIFF to {}", path.display())
                })?;
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
            write_rgb16_with_icc(&mut tiff, width_u, height_u, &buf, crate::color_space::SRGB_ICC)
                .with_context(|| format!("Failed to write 16-bit TIFF to {}", path.display()))?;
        }
    }

    Ok(())
}

fn write_rgb16_with_icc<W: std::io::Write + std::io::Seek>(
    tiff: &mut TiffEncoder<W>,
    width: u32,
    height: u32,
    buf: &[u16],
    icc: &[u8],
) -> Result<()> {
    let mut image = tiff
        .new_image::<RGB16>(width, height)
        .context("Failed to create 16-bit TIFF image")?;
    image
        .encoder()
        .write_tag(Tag::IccProfile, icc)
        .context("Failed to write ICC profile")?;
    image
        .write_data(buf)
        .context("Failed to write 16-bit TIFF image data")?;
    Ok(())
}

/// Write linear print-RGB u16 as an sRGB-encoded 16-bit TIFF.
///
/// RA-4 / FilmPrint output is linear. Viewers (Preview, Photoshop without a
/// linear profile) assume sRGB, so we apply the OETF here.
pub fn write_tiff_u16(image: &Array3<u16>, path: &Path) -> Result<()> {
    let (height, width, c) = image.dim();
    if c != 3 {
        bail!("TIFF export expects RGB (3 channels), got {}", c);
    }

    let width_u = u32::try_from(width).context("Image width too large for TIFF")?;
    let height_u = u32::try_from(height).context("Image height too large for TIFF")?;

    let mut buf: Vec<u16> = Vec::with_capacity(height * width * 3);
    let scale = 1.0 / 65535.0_f32;
    for row in image.axis_iter(ndarray::Axis(0)) {
        for pixel in row.axis_iter(ndarray::Axis(0)) {
            buf.push(crate::color_space::linear_to_srgb_u16(
                pixel[0] as f32 * scale,
            ));
            buf.push(crate::color_space::linear_to_srgb_u16(
                pixel[1] as f32 * scale,
            ));
            buf.push(crate::color_space::linear_to_srgb_u16(
                pixel[2] as f32 * scale,
            ));
        }
    }

    let file =
        File::create(path).with_context(|| format!("Failed to create {}", path.display()))?;
    let writer = BufWriter::new(file);
    let mut tiff = TiffEncoder::new(writer).with_context(|| "Failed to create TIFF encoder")?;
    write_rgb16_with_icc(&mut tiff, width_u, height_u, &buf, crate::color_space::SRGB_ICC)
        .with_context(|| format!("Failed to write 16-bit TIFF to {}", path.display()))?;

    Ok(())
}

/// Write already-encoded 16-bit RGB and embed `icc`.
pub fn write_tiff_rgb16_with_icc(
    buf: &[u16],
    width: u32,
    height: u32,
    path: &Path,
    icc: &[u8],
) -> Result<()> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(3))
        .context("TIFF dimensions overflow")?;
    if buf.len() != expected {
        bail!(
            "TIFF RGB buffer length {} does not match {}x{}x3",
            buf.len(),
            width,
            height
        );
    }
    let file =
        File::create(path).with_context(|| format!("Failed to create {}", path.display()))?;
    let writer = BufWriter::new(file);
    let mut tiff = TiffEncoder::new(writer).with_context(|| "Failed to create TIFF encoder")?;
    write_rgb16_with_icc(&mut tiff, width, height, buf, icc)
        .with_context(|| format!("Failed to write 16-bit TIFF to {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use tiff::decoder::Decoder;
    use tiff::tags::Tag;

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "oxid-icc-{}-{}-{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn tiff16_embeds_srgb_icc() {
        let img = Array3::<u16>::from_elem((2, 2, 3), 32768);
        let path = temp_path("u16.tiff");
        write_tiff_u16(&img, &path).expect("write");
        let file = File::open(&path).expect("open");
        let mut dec = Decoder::new(file).expect("decode");
        let icc = dec
            .find_tag(Tag::IccProfile)
            .expect("tag")
            .expect("icc present")
            .into_u8_vec()
            .expect("bytes");
        assert_eq!(icc, crate::color_space::SRGB_ICC);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn tiff16_from_f32_embeds_srgb_icc() {
        let img = Array3::<f32>::from_elem((2, 2, 3), 0.5);
        let path = temp_path("f32u16.tiff");
        write_tiff(&img, &path, TiffFormat::U16).expect("write");
        let file = File::open(&path).expect("open");
        let mut dec = Decoder::new(file).expect("decode");
        let icc = dec
            .find_tag(Tag::IccProfile)
            .expect("tag")
            .expect("icc present")
            .into_u8_vec()
            .expect("bytes");
        assert_eq!(icc, crate::color_space::SRGB_ICC);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn tiff16_embeds_custom_icc() {
        let icc = crate::cms::output_icc_bytes(crate::options::OutputIcc::DisplayP3, None).unwrap();
        let buf = [1000u16, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 1000, 2000, 3000];
        let path = temp_path("p3.tiff");
        write_tiff_rgb16_with_icc(&buf, 2, 2, &path, &icc).expect("write");
        let file = File::open(&path).expect("open");
        let mut dec = Decoder::new(file).expect("decode");
        let got = dec
            .find_tag(Tag::IccProfile)
            .expect("tag")
            .expect("icc present")
            .into_u8_vec()
            .expect("bytes");
        assert_eq!(got, icc);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn tiff32_has_no_icc() {
        let img = Array3::<f32>::from_elem((2, 2, 3), 0.5);
        let path = temp_path("f32.tiff");
        write_tiff(&img, &path, TiffFormat::Float32).expect("write");
        let file = File::open(&path).expect("open");
        let mut dec = Decoder::new(file).expect("decode");
        let icc = dec.find_tag(Tag::IccProfile).expect("tag");
        assert!(icc.is_none(), "linear float TIFF must stay untagged");
        let _ = std::fs::remove_file(&path);
    }
}
