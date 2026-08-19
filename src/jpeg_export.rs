//! 8-bit JPEG export with an embedded sRGB ICC profile.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyhow::{bail, Context, Result};
use image::codecs::jpeg::JpegEncoder;
use image::{ExtendedColorType, ImageEncoder};

/// Default quality matches `image` crate `RgbImage::save` (75).
const JPEG_QUALITY: u8 = 75;

/// Write sRGB-encoded 8-bit RGB as JPEG with an IEC 61966-2.1 ICC profile.
pub fn write_jpeg_srgb(path: &Path, width: u32, height: u32, rgb: &[u8]) -> Result<()> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(3))
        .context("JPEG dimensions overflow")?;
    if rgb.len() != expected {
        bail!(
            "JPEG RGB buffer length {} does not match {}x{}x3",
            rgb.len(),
            width,
            height
        );
    }

    let file =
        File::create(path).with_context(|| format!("Failed to create {}", path.display()))?;
    let mut encoder = JpegEncoder::new_with_quality(BufWriter::new(file), JPEG_QUALITY);
    encoder
        .set_icc_profile(crate::color_space::SRGB_ICC.to_vec())
        .map_err(|e| anyhow::anyhow!("JPEG encoder rejected sRGB ICC: {e}"))?;
    encoder
        .encode(rgb, width, height, ExtendedColorType::Rgb8)
        .with_context(|| format!("Failed to write JPEG to {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::codecs::jpeg::JpegDecoder;
    use image::ImageDecoder;
    use std::io::{BufReader, Cursor};

    #[test]
    fn jpeg_embeds_srgb_icc() {
        let rgb = [200u8, 40, 40, 40, 200, 40, 40, 40, 200, 180, 180, 40];
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "oxid-icc-jpeg-{}-{}.jpg",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_jpeg_srgb(&path, 2, 2, &rgb).expect("write");
        let bytes = std::fs::read(&path).expect("read");
        let mut dec = JpegDecoder::new(BufReader::new(Cursor::new(bytes))).expect("decode");
        let icc = dec.icc_profile().expect("icc read").expect("icc present");
        assert_eq!(icc, crate::color_space::SRGB_ICC);
        let _ = std::fs::remove_file(&path);
    }
}
