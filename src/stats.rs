//! Per-channel statistics for debug logging and auto white balance.

use ndarray::Array3;

use crate::crop_array3;
use crate::scale_dmin_rect;
use crate::PipelineOptions;

/// Compute per-channel statistics (min, max, median) for a (H, W, 3) image.
pub(crate) fn channel_stats(image: &Array3<f32>) -> [(f32, f32, f32); 3] {
    let mut stats = [(0.0_f32, 0.0_f32, 0.0_f32); 3];
    for ch in 0..3 {
        let slice = image.slice(ndarray::s![.., .., ch]);
        let mut vals: Vec<f32> = slice.iter().copied().collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let min = vals.first().copied().unwrap_or(0.0);
        let max = vals.last().copied().unwrap_or(0.0);
        let median = if vals.is_empty() {
            0.0
        } else {
            vals[vals.len() / 2]
        };
        stats[ch] = (min, max, median);
    }
    stats
}

/// Channel stats source for Auto WB.
/// When crop is enabled, evaluate statistics inside the crop only.
pub(crate) fn wb_channel_stats(
    image: &Array3<f32>,
    options: &PipelineOptions,
) -> [(f32, f32, f32); 3] {
    if options.apply_crop {
        if let Some(rect) = options.crop_rect {
            let (h, w, _) = image.dim();
            let (x, y, rw, rh) =
                scale_dmin_rect(rect, options.crop_rect_reference_size, w as u32, h as u32);
            let cropped = crop_array3(image, x, y, rw, rh);
            return channel_stats(&cropped);
        }
    }
    channel_stats(image)
}

pub(crate) fn fmt_stats(label: &str, stats: &[(f32, f32, f32); 3]) -> String {
    format!(
        "{}\n  R: min={:.6} max={:.6} med={:.6}\n  G: min={:.6} max={:.6} med={:.6}\n  B: min={:.6} max={:.6} med={:.6}\n",
        label,
        stats[0].0,
        stats[0].1,
        stats[0].2,
        stats[1].0,
        stats[1].1,
        stats[1].2,
        stats[2].0,
        stats[2].1,
        stats[2].2,
    )
}
