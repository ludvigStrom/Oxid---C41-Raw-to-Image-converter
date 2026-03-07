use std::path::Path;

use anyhow::{bail, Context, Result};
use ndarray::{Array2, Array3, Axis};
use rawloader::RawImageData;

/// Load a RAW file into a strictly linear `Array3<f32>`.
///
/// Uses **rawloader** (pure Rust) to decode the file. No gamma, white balance,
/// or base curve is applied — only the raw Bayer (or CFA) plane is extracted.
/// Data is normalized to [0.0, 1.0] for pipeline compatibility.
///
/// Shape of the returned array:
/// - `(height, width, 1)` — a single channel representing the raw Bayer values.
///   Later steps can demosaic and build true RGB while keeping everything linear.
///
/// Supported formats include Sony ARW, Canon CR2/CRW, Nikon NEF/NRW, Fuji RAF,
/// Panasonic RW2, Adobe DNG, and others supported by rawloader.
pub fn load_raw_as_ndarray(path: &Path) -> Result<Array3<f32>> {
    let raw_image = rawloader::decode_file(path)
        .with_context(|| format!("rawloader failed to decode {}", path.display()))?;

    let width = raw_image.width;
    let height = raw_image.height;

    // We expect a single-channel Bayer/CFA image (cpp = 1). X-Trans or other multi-component
    // data would need different handling.
    if raw_image.cpp != 1 {
        bail!(
            "Expected Bayer/CFA raw (cpp=1), got cpp={}. Unsupported format.",
            raw_image.cpp
        );
    }

    let expected_len = width * height;

    let float_data: Vec<f32> = match &raw_image.data {
        RawImageData::Integer(data) => {
            if data.len() != expected_len {
                bail!(
                    "Unexpected raw buffer length: expected {} pixels, got {}",
                    expected_len,
                    data.len()
                );
            }
            let max_value = u16::MAX as f32;
            data.iter().map(|&v| (v as f32) / max_value).collect()
        }
        RawImageData::Float(data) => {
            if data.len() != expected_len {
                bail!(
                    "Unexpected raw buffer length: expected {} pixels, got {}",
                    expected_len,
                    data.len()
                );
            }
            // Assume float data is already in a linear 0..1 range (or relative); use as-is.
            data.clone()
        }
    };

    let array2: Array2<f32> = Array2::from_shape_vec((height, width), float_data).with_context(|| {
        format!(
            "Failed to reshape RAW data into 2D array (height={}, width={})",
            height, width
        )
    })?;

    let array3: Array3<f32> = array2.insert_axis(Axis(2));
    Ok(array3)
}
