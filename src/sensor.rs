//! Sensor loading and D-min computation from cached sensor data.
//!
//! Used by the GUI to load a full-resolution frame once, then compute D-min
//! medians from a user-defined rect without re-decoding the RAW.

use std::path::Path;

use anyhow::Result;
use ndarray::Array3;

use crate::apply_rotation;
use crate::demosaic;
use crate::png_reader;
use crate::raw_reader;
use crate::scale_dmin_rect;
use crate::Rect;

/// Raw sensor data cached for fast previews/exports.
#[derive(Debug, Clone)]
pub enum CachedSensor {
    /// Single-channel CFA mosaic (Bayer or X-Trans) plus its pattern descriptor.
    Bayer {
        data: Array3<f32>,
        pattern: demosaic::CfaPattern,
    },
    /// Linear RGB image (e.g. PNG or already-demosaiced source).
    Rgb(Array3<f32>),
}

/// Load raw sensor data (Bayer or RGB) from disk into a cached representation.
pub fn load_sensor_from_path(path: &Path) -> Result<CachedSensor> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "arw" | "nef" | "nrw" | "cr2" | "cr3" | "crw" | "dng" | "raf" | "orf" | "rw2" => {
            let (bayer, pattern) = raw_reader::load_raw_as_ndarray(path)?;
            Ok(CachedSensor::Bayer { data: bayer, pattern })
        }
        "png" => {
            let img = png_reader::load_png_as_ndarray(path)?;
            Ok(CachedSensor::Rgb(img))
        }
        _ => anyhow::bail!("Unsupported extension for sensor cache"),
    }
}

fn median_f32(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    if v.len() % 2 == 1 {
        v[mid]
    } else {
        (v[mid - 1] + v[mid]) / 2.0
    }
}

/// Compute D-min medians from cached full-resolution sensor data.
///
/// - Uses the same D-min rect + reference-size semantics as the main pipeline.
/// - For Bayer sources, demosaics the full frame, then samples the rect.
pub fn compute_dmin_from_sensor(
    sensor: &CachedSensor,
    rect: Rect,
    reference_size: Option<(u32, u32)>,
    rotation_degrees: i32,
    neutral_only: bool,
) -> Result<(f32, f32, f32)> {
    let mut rgb: Array3<f32> = match sensor {
        CachedSensor::Bayer { data, pattern } => {
            let mut img = demosaic::demosaic_quality(data, *pattern)?;
            img.mapv_inplace(|v| v.max(0.0));
            img
        }
        CachedSensor::Rgb(image) => image.clone(),
    };

    if rotation_degrees != 0 {
        rgb = apply_rotation(&rgb, rotation_degrees);
    }

    // Scale rect to current image size.
    let (h, w, c) = rgb.dim();
    if c != 3 {
        anyhow::bail!("compute_dmin_from_sensor expects RGB image with 3 channels");
    }
    let (x, y, rw, rh) = scale_dmin_rect(rect, reference_size, w as u32, h as u32);

    // Clamp rect and collect values in region, mirroring dmin::neutralize.
    let x_us = x as usize;
    let y_us = y as usize;
    let rw_us = rw as usize;
    let rh_us = rh as usize;

    let x_end = (x_us + rw_us).min(w);
    let y_end = (y_us + rh_us).min(h);
    let x_start = x_us.min(w.saturating_sub(1));
    let y_start = y_us.min(h.saturating_sub(1));

    if x_start >= x_end || y_start >= y_end {
        anyhow::bail!(
            "D-min rect [{}, {}] + {}x{} is outside or zero-size for image {}x{}",
            x_us,
            y_us,
            rw_us,
            rh_us,
            w,
            h
        );
    }

    let region = rgb.slice(ndarray::s![y_start..y_end, x_start..x_end, ..]);

    let mut r_vals = Vec::with_capacity((y_end - y_start) * (x_end - x_start));
    let mut g_vals = Vec::with_capacity((y_end - y_start) * (x_end - x_start));
    let mut b_vals = Vec::with_capacity((y_end - y_start) * (x_end - x_start));

    for row in region.axis_iter(ndarray::Axis(0)) {
        for pixel in row.axis_iter(ndarray::Axis(0)) {
            r_vals.push(pixel[0]);
            g_vals.push(pixel[1]);
            b_vals.push(pixel[2]);
        }
    }

    let med_r = median_f32(r_vals);
    let med_g = median_f32(g_vals);
    let med_b = median_f32(b_vals);

    if neutral_only {
        // Geometric mean, same as dmin::neutralize neutral_only path.
        let g = (med_r * med_g * med_b).max(0.0).cbrt();
        let k = if g > 0.0 { g } else { 1.0 };
        Ok((k, k, k))
    } else {
        Ok((med_r, med_g, med_b))
    }
}
