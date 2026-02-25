use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use ndarray::{Array2, Array3, Axis};
use libraw::{RawImage, Sizes};
use libraw as libraw_rs; // clarify that this comes from the libraw-rs crate

/// Load a Sony .ARW file into a strictly linear `Array3<f32>`.
///
/// - Uses LibRaw's **raw Bayer plane** (`RawImage`), not the dcraw-style processed image.
/// - This means:
///   - No gamma curve is applied (data is in sensor space).
///   - No camera or auto white balance is applied.
///   - No hidden base curve or tone mapping is applied.
/// - The resulting array is normalized to [0.0, 1.0] by dividing by `u16::MAX`.
///
/// Shape of the returned array:
/// - `(height, width, 1)` — a single channel representing the raw Bayer values.
///   Later steps can demosaic and build true RGB while keeping everything linear.
pub fn load_raw_as_ndarray(path: &Path) -> Result<Array3<f32>> {
    // Read the entire RAW file into memory; LibRaw will open it from a buffer.
    let buf = fs::read(path).with_context(|| format!("Failed to read RAW file {}", path.display()))?;

    // Decode to a raw Bayer plane (no gamma/WB/base curve).
    let processor = libraw_rs::Processor::default();
    let raw_image: RawImage = processor
        .decode(&buf)
        .with_context(|| format!("LibRaw failed to decode {}", path.display()))?;

    let sizes: Sizes = raw_image.sizes();
    let width = sizes.raw_width as usize;
    let height = sizes.raw_height as usize;
    let expected_len = width * height;

    // RawImage derefs to &[u16] containing the linear sensor values.
    let raw_slice: &[u16] = &*raw_image;

    if raw_slice.len() != expected_len {
        bail!(
            "Unexpected RAW buffer length: expected {} pixels, got {}",
            expected_len,
            raw_slice.len()
        );
    }

    // Convert to f32 in [0.0, 1.0] by dividing by the maximum representable u16.
    // Note: for sensors with e.g. 14-bit data, this preserves linearity; the dynamic
    // range will just occupy a subset of [0, 1], which is fine because D-min
    // neutralization and later math will all be relative.
    let max_value = u16::MAX as f32;

    let float_data: Vec<f32> = raw_slice
        .iter()
        .map(|&v| (v as f32) / max_value)
        .collect();

    // Create a 2D ndarray from the raw data, then add a singleton channel dimension.
    let array2: Array2<f32> =
        Array2::from_shape_vec((height, width), float_data).with_context(|| {
            format!(
                "Failed to reshape RAW data into 2D array (height={}, width={})",
                height, width
            )
        })?;

    // (H, W, 1) — channel dimension last, ready for future RGB steps.
    let array3: Array3<f32> = array2.insert_axis(Axis(2));

    Ok(array3)
}