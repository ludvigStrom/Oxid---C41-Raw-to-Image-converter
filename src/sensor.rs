//! Sensor loading and D-min computation from cached sensor data.
//!
//! Used by the GUI to load a full-resolution frame once, then compute D-min
//! medians from a user-defined rect without re-decoding the RAW.

use std::path::Path;

use anyhow::Result;
use ndarray::Array3;

use crate::apply_rotation;
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
use crate::{flip_array3_horizontal, flip_array3_vertical};

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
            Ok(CachedSensor::Bayer {
                data: bayer,
                pattern,
            })
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
    /// Zone percentiles of post-step-4 density at curve_offset 0.
    pub zone: Option<(f32, f32, f32, f32)>,
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
    // Rotation/flip do not change full-frame channel stats (same pixels).
    // Including them here forced a full-res remosaic on the UI thread after
    // every rotate/flip click.
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
    hash_f32(&mut h, opts.film_gamma);
    opts.apply_color_profile.hash(&mut h);
    for row in &opts.density_matrix {
        for &v in row {
            hash_f32(&mut h, v);
        }
    }
    // Crop is display/histogram/export only — it must not remosaic the full sensor.
    opts.flat_field_path
        .as_ref()
        .map(|p| p.display().to_string())
        .hash(&mut h);
    h.finish()
}

fn sensor_to_working_rgb(sensor: &CachedSensor, options: &PipelineOptions) -> Result<Array3<f32>> {
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
            zone: None,
        });
    }

    let mut rgb = sensor_to_working_rgb(sensor, options)?;

    let dmin = match options.dmin_mode {
        DminMode::Off => None,
        DminMode::Fixed => options.dmin_fixed,
        DminMode::AutoPercentile => Some(dmin::compute_auto_percentile_divisors(
            &rgb,
            options.auto_norm_buffer,
        )?),
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

    // Match step 4 so zone masks are the same on every tile.
    if let Some((dr, dg, db)) = dmin {
        if auto_wb.is_none() {
            dmin::neutralize_with_medians(&mut rgb, dr, dg, db)?;
            rgb.mapv_inplace(|v| v.max(0.0));
        }
    }
    if auto_wb.is_none() {
        rgb.mapv_inplace(|t| (-(t.max(1e-10_f32)).log10()).max(0.0));
    }
    let (ar, ag, ab) = auto_wb.unwrap_or((1.0, 1.0, 1.0));
    let inv_gamma = 1.0 / options.film_gamma.max(0.1);
    let (mr, mg, mb) = if options.apply_white_balance {
        (options.wb_r, options.wb_g, options.wb_b)
    } else {
        (1.0, 1.0, 1.0)
    };
    let s_r = ar * mr * inv_gamma;
    let s_g = ag * mg * inv_gamma;
    let s_b = ab * mb * inv_gamma;
    rgb.slice_mut(ndarray::s![.., .., 0])
        .mapv_inplace(|v| v * s_r);
    rgb.slice_mut(ndarray::s![.., .., 1])
        .mapv_inplace(|v| v * s_g);
    rgb.slice_mut(ndarray::s![.., .., 2])
        .mapv_inplace(|v| v * s_b);
    if options.apply_color_profile {
        let m = options.density_matrix;
        let (h, w, _) = rgb.dim();
        for y in 0..h {
            for x in 0..w {
                let dr = rgb[[y, x, 0]];
                let dg = rgb[[y, x, 1]];
                let db = rgb[[y, x, 2]];
                rgb[[y, x, 0]] = m[0][0] * dr + m[0][1] * dg + m[0][2] * db;
                rgb[[y, x, 1]] = m[1][0] * dr + m[1][1] * dg + m[1][2] * db;
                rgb[[y, x, 2]] = m[2][0] * dr + m[2][1] * dg + m[2][2] * db;
            }
        }
        rgb.mapv_inplace(|v| v.max(0.0));
    }
    let zp = crate::density_ops::zone_density_range(&rgb, 0.0);
    let zone = Some((zp.d_min, zp.d_p33, zp.d_p66, zp.d_max));

    Ok(PreviewSceneStats {
        dmin,
        auto_wb,
        zone,
    })
}

impl CachedSensor {
    /// Unrotated sensor / file dimensions (width, height).
    pub fn dimensions(&self) -> (u32, u32) {
        match self {
            CachedSensor::Bayer { data, .. } => {
                let (h, w, _) = data.dim();
                (w as u32, h as u32)
            }
            CachedSensor::Rgb(img) => {
                let (h, w, _) = img.dim();
                (w as u32, h as u32)
            }
        }
    }
}

/// Oriented (after rotation) full-frame size from unrotated sensor dims.
pub fn oriented_sensor_size(sensor_w: u32, sensor_h: u32, rotation_degrees: i32) -> (u32, u32) {
    let r = ((rotation_degrees % 360 + 360) % 360) / 90;
    if r == 1 || r == 3 {
        (sensor_h, sensor_w)
    } else {
        (sensor_w, sensor_h)
    }
}

fn cfa_align(sensor: &CachedSensor) -> u32 {
    match sensor {
        CachedSensor::Bayer {
            pattern: crate::demosaic::CfaPattern::XTrans(_),
            ..
        } => 6,
        CachedSensor::Bayer { .. } => 2,
        CachedSensor::Rgb(_) => 1,
    }
}

fn align_down(v: i32, a: i32) -> i32 {
    if a <= 1 {
        return v.max(0);
    }
    let v = v.max(0);
    v / a * a
}

fn align_up(v: i32, a: i32) -> i32 {
    if a <= 1 {
        return v.max(0);
    }
    let v = v.max(0);
    ((v + a - 1) / a) * a
}

/// Map an oriented pixel to unrotated sensor coordinates.
pub(crate) fn oriented_to_sensor(
    ox: i32,
    oy: i32,
    sensor_w: i32,
    sensor_h: i32,
    rotation_degrees: i32,
    flip_h: bool,
    flip_v: bool,
) -> (i32, i32) {
    let (ow, oh) = oriented_sensor_size(sensor_w as u32, sensor_h as u32, rotation_degrees);
    let mut x = ox;
    let mut y = oy;
    if flip_h {
        x = ow as i32 - 1 - x;
    }
    if flip_v {
        y = oh as i32 - 1 - y;
    }
    let r = ((rotation_degrees % 360 + 360) % 360) / 90;
    match r {
        1 => (y, sensor_h - 1 - x), // inverse of 90 CW
        2 => (sensor_w - 1 - x, sensor_h - 1 - y),
        3 => (sensor_w - 1 - y, x), // inverse of 270 CW
        _ => (x, y),
    }
}

/// Map an unrotated sensor pixel to oriented coordinates.
pub(crate) fn sensor_to_oriented(
    sx: i32,
    sy: i32,
    sensor_w: i32,
    sensor_h: i32,
    rotation_degrees: i32,
    flip_h: bool,
    flip_v: bool,
) -> (i32, i32) {
    let r = ((rotation_degrees % 360 + 360) % 360) / 90;
    let (mut ox, mut oy) = match r {
        1 => (sensor_h - 1 - sy, sx), // 90 CW
        2 => (sensor_w - 1 - sx, sensor_h - 1 - sy),
        3 => (sy, sensor_w - 1 - sx), // 270 CW
        _ => (sx, sy),
    };
    let (ow, oh) = oriented_sensor_size(sensor_w as u32, sensor_h as u32, rotation_degrees);
    if flip_h {
        ox = ow as i32 - 1 - ox;
    }
    if flip_v {
        oy = oh as i32 - 1 - oy;
    }
    (ox, oy)
}

/// Crop of `CachedSensor` covering an oriented rectangle, plus the UV of the
/// processed tile in the oriented full frame (0–1).
pub struct SensorTileCrop {
    pub sensor: CachedSensor,
    pub uv_left: f32,
    pub uv_top: f32,
    pub uv_right: f32,
    pub uv_bottom: f32,
}

/// Extract a CFA-aligned sensor crop that covers `oriented` pixels plus `halo`.
///
/// `oriented_*` is in the rotated/flipped full-frame space (same as the GUI image).
pub fn crop_sensor_for_oriented_rect(
    sensor: &CachedSensor,
    oriented_x: u32,
    oriented_y: u32,
    oriented_w: u32,
    oriented_h: u32,
    rotation_degrees: i32,
    flip_h: bool,
    flip_v: bool,
    halo: u32,
) -> Result<SensorTileCrop> {
    let (sw, sh) = sensor.dimensions();
    let (ow, oh) = oriented_sensor_size(sw, sh, rotation_degrees);
    if ow == 0 || oh == 0 {
        anyhow::bail!("empty sensor");
    }

    let x0 = oriented_x as i32 - halo as i32;
    let y0 = oriented_y as i32 - halo as i32;
    let x1 = (oriented_x + oriented_w) as i32 + halo as i32;
    let y1 = (oriented_y + oriented_h) as i32 + halo as i32;

    let corners = [(x0, y0), (x1 - 1, y0), (x0, y1 - 1), (x1 - 1, y1 - 1)];
    let mut min_sx = sw as i32;
    let mut min_sy = sh as i32;
    let mut max_sx = 0;
    let mut max_sy = 0;
    for (ox, oy) in corners {
        let (sx, sy) = oriented_to_sensor(
            ox,
            oy,
            sw as i32,
            sh as i32,
            rotation_degrees,
            flip_h,
            flip_v,
        );
        min_sx = min_sx.min(sx);
        min_sy = min_sy.min(sy);
        max_sx = max_sx.max(sx);
        max_sy = max_sy.max(sy);
    }

    let align = cfa_align(sensor) as i32;
    min_sx = align_down(min_sx, align).clamp(0, sw as i32 - 1);
    min_sy = align_down(min_sy, align).clamp(0, sh as i32 - 1);
    max_sx = align_up(max_sx + 1, align).clamp(min_sx + align, sw as i32);
    max_sy = align_up(max_sy + 1, align).clamp(min_sy + align, sh as i32);

    let cw = (max_sx - min_sx) as usize;
    let ch = (max_sy - min_sy) as usize;
    if cw == 0 || ch == 0 {
        anyhow::bail!("empty sensor tile crop");
    }

    let cropped = match sensor {
        CachedSensor::Bayer { data, pattern } => {
            let slice = data.slice(ndarray::s![
                min_sy as usize..max_sy as usize,
                min_sx as usize..max_sx as usize,
                ..
            ]);
            CachedSensor::Bayer {
                data: slice.to_owned(),
                pattern: *pattern,
            }
        }
        CachedSensor::Rgb(img) => {
            let slice = img.slice(ndarray::s![
                min_sy as usize..max_sy as usize,
                min_sx as usize..max_sx as usize,
                ..
            ]);
            CachedSensor::Rgb(slice.to_owned())
        }
    };

    // Forward-map the actual crop corners to oriented UV.
    let fwd = [
        (min_sx, min_sy),
        (max_sx - 1, min_sy),
        (min_sx, max_sy - 1),
        (max_sx - 1, max_sy - 1),
    ];
    let mut min_ox = ow as i32;
    let mut min_oy = oh as i32;
    let mut max_ox = 0;
    let mut max_oy = 0;
    for (sx, sy) in fwd {
        let (ox, oy) = sensor_to_oriented(
            sx,
            sy,
            sw as i32,
            sh as i32,
            rotation_degrees,
            flip_h,
            flip_v,
        );
        min_ox = min_ox.min(ox);
        min_oy = min_oy.min(oy);
        max_ox = max_ox.max(ox);
        max_oy = max_oy.max(oy);
    }
    let owf = ow as f32;
    let ohf = oh as f32;
    Ok(SensorTileCrop {
        sensor: cropped,
        uv_left: (min_ox as f32 / owf).clamp(0.0, 1.0),
        uv_top: (min_oy as f32 / ohf).clamp(0.0, 1.0),
        uv_right: ((max_ox + 1) as f32 / owf).clamp(0.0, 1.0),
        uv_bottom: ((max_oy + 1) as f32 / ohf).clamp(0.0, 1.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensor_oriented_roundtrip() {
        let sw = 20;
        let sh = 12;
        for rot in [0, 90, 180, 270] {
            for flip_h in [false, true] {
                for flip_v in [false, true] {
                    for sy in 0..sh {
                        for sx in 0..sw {
                            let (ox, oy) = sensor_to_oriented(sx, sy, sw, sh, rot, flip_h, flip_v);
                            let (rx, ry) = oriented_to_sensor(ox, oy, sw, sh, rot, flip_h, flip_v);
                            assert_eq!(
                                (rx, ry),
                                (sx, sy),
                                "rot={rot} flip={flip_h},{flip_v} ({sx},{sy})"
                            );
                        }
                    }
                }
            }
        }
    }
}
