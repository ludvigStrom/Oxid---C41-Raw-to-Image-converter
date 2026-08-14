//! Sensor loading and D-min computation from cached sensor data.
//!
//! Used by the GUI to load a full-resolution frame once, then compute D-min
//! medians from a user-defined rect without re-decoding the RAW.

use std::path::Path;

use anyhow::Result;
use ndarray::Array3;

use crate::apply_rotation;
use crate::{flip_array3_horizontal, flip_array3_vertical};
use crate::color_space;
use crate::demosaic;
use crate::dmin;
use crate::pipeline;
use crate::png_reader;
use crate::raw_reader;
use crate::scale_dmin_rect;
use crate::stats;
use crate::DminMode;
use crate::PipelineOptions;
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
        "png" | "jpeg" | "jpg" | "tiff" | "tif" => {
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
    flip_horizontal: bool,
    flip_vertical: bool,
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
    if flip_horizontal {
        rgb = flip_array3_horizontal(&rgb);
    }
    if flip_vertical {
        rgb = flip_array3_vertical(&rgb);
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

/// Full-res D-min divisors and auto-WB scales, matching the export pipeline.
/// Cached by the GUI so slider tweaks that do not affect these stats stay cheap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreviewSceneStats {
    pub dmin: Option<(f32, f32, f32)>,
    pub auto_wb: Option<(f32, f32, f32)>,
}

fn hash_f32(h: &mut impl std::hash::Hasher, v: f32) {
    std::hash::Hash::hash(&v.to_bits(), h);
}

fn hash_rect(h: &mut impl std::hash::Hasher, r: &Rect) {
    use std::hash::Hash;
    r.x.hash(h);
    r.y.hash(h);
    r.width.hash(h);
    r.height.hash(h);
}

/// Hash of options that change full-res D-min / auto-WB. Manual WB and curve are excluded.
pub fn preview_scene_stats_key(opts: &PipelineOptions) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    opts.rotation_degrees.hash(&mut h);
    opts.flip_horizontal.hash(&mut h);
    opts.flip_vertical.hash(&mut h);
    opts.synthetic_negative_input.hash(&mut h);
    for row in &opts.idt_matrix {
        for &v in row {
            hash_f32(&mut h, v);
        }
    }
    opts.dmin_mode.hash(&mut h);
    hash_f32(&mut h, opts.auto_norm_buffer);
    opts.dmin_neutral_only.hash(&mut h);
    opts.dmin_rect_reference_size.hash(&mut h);
    if let Some(r) = opts.dmin_rect {
        hash_rect(&mut h, &r);
    }
    if let Some((r, g, b)) = opts.dmin_fixed {
        hash_f32(&mut h, r);
        hash_f32(&mut h, g);
        hash_f32(&mut h, b);
    }
    opts.auto_wb.hash(&mut h);
    opts.apply_crop.hash(&mut h);
    opts.crop_rect_reference_size.hash(&mut h);
    if let Some(r) = opts.crop_rect {
        hash_rect(&mut h, &r);
    }
    opts.flat_field_path
        .as_ref()
        .map(|p| p.display().to_string())
        .hash(&mut h);
    h.finish()
}

fn sensor_to_working_rgb(
    sensor: &CachedSensor,
    options: &PipelineOptions,
) -> Result<Array3<f32>> {
    let mut rgb: Array3<f32> = match sensor {
        CachedSensor::Bayer { data, pattern } => {
            let mut img = demosaic::demosaic_quality(data, *pattern)?;
            img.mapv_inplace(|v| v.max(0.0));
            img
        }
        CachedSensor::Rgb(image) => {
            let mut img = image.clone();
            if options.synthetic_negative_input {
                pipeline::apply_synthetic_negative_invert(&mut img);
            }
            img
        }
    };

    if options.rotation_degrees != 0 {
        rgb = apply_rotation(&rgb, options.rotation_degrees);
    }
    if options.flip_horizontal {
        rgb = flip_array3_horizontal(&rgb);
    }
    if options.flip_vertical {
        rgb = flip_array3_vertical(&rgb);
    }
    color_space::apply_input_idt_to_working_space(&mut rgb, &options.idt_matrix);
    Ok(rgb)
}

/// Compute export-matching D-min and auto-WB from the full-resolution cached sensor.
///
/// Preview runs on a downscaled buffer, so AutoPercentile / auto WB on that buffer
/// diverge from export. Pin those global stats to the full frame instead.
pub fn compute_preview_scene_stats(
    sensor: &CachedSensor,
    options: &PipelineOptions,
) -> Result<PreviewSceneStats> {
    if options.flat_field_path.is_some() {
        return Ok(PreviewSceneStats {
            dmin: None,
            auto_wb: None,
        });
    }

    let mut rgb = sensor_to_working_rgb(sensor, options)?;

    let dmin = match options.dmin_mode {
        DminMode::Off => None,
        DminMode::Fixed => options.dmin_fixed,
        DminMode::AutoPercentile => {
            Some(dmin::compute_auto_percentile_divisors(&rgb, options.auto_norm_buffer)?)
        }
        DminMode::SampleRegion => {
            if let Some(rect) = options.dmin_rect {
                let (h, w, _) = rgb.dim();
                let (x, y, rw, rh) =
                    scale_dmin_rect(rect, options.dmin_rect_reference_size, w as u32, h as u32);
                Some(dmin::compute_neutralize_divisors(
                    &rgb,
                    x,
                    y,
                    rw,
                    rh,
                    options.dmin_neutral_only,
                )?)
            } else {
                None
            }
        }
    };

    let auto_wb = if options.auto_wb && options.dmin_mode != DminMode::Off {
        if let Some((dr, dg, db)) = dmin {
            dmin::neutralize_with_medians(&mut rgb, dr, dg, db)?;
            rgb.mapv_inplace(|v| v.max(0.0));
            rgb.mapv_inplace(|t| (-(t.max(1e-10_f32)).log10()).max(0.0));
            let stats = stats::wb_channel_stats(&rgb, options);
            let med_r = stats[0].2.max(1e-4);
            let med_g = stats[1].2.max(1e-4);
            let med_b = stats[2].2.max(1e-4);
            let mean_d = (med_r + med_g + med_b) / 3.0;
            Some((mean_d / med_r, mean_d / med_g, mean_d / med_b))
        } else {
            None
        }
    } else {
        None
    };

    Ok(PreviewSceneStats { dmin, auto_wb })
}
