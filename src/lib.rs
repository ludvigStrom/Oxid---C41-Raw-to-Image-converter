//! C-41 RAW pipeline library. Used by both CLI and GUI.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::Context;
use image::{
    imageops::{self, FilterType},
    Rgb, Rgb32FImage, RgbImage,
};
use ndarray::{self, Array3};

pub mod aces;
pub mod calibration;
pub mod color;
pub mod curve;
pub mod demosaic;
pub mod density_ops;
pub mod dmin;
pub mod exr_export;
pub mod flat_field;
pub mod inversion;
pub mod lut3d;
pub mod options;
pub mod png_reader;
pub mod post_curve;
pub mod raw_reader;
pub mod sensor;
pub mod stats;
pub mod tiff_export;

pub use flat_field::{blur_flat_field, load_flat_field_linear};
pub use options::{DminMode, OutputLutEncoding, OutputStage, PipelineOptions, Rect};
pub use sensor::{compute_dmin_from_sensor, load_sensor_from_path, CachedSensor};
pub use tiff_export::TiffFormat;

use crate::demosaic::CfaPattern;

/// Scale D-min rect from reference size to current image size. If reference is None or matches current size, returns rect as-is.
pub(crate) fn scale_dmin_rect(
    rect: Rect,
    reference_size: Option<(u32, u32)>,
    current_w: u32,
    current_h: u32,
) -> (u32, u32, u32, u32) {
    let (x, y, rw, rh) = (rect.x, rect.y, rect.width, rect.height);
    match reference_size {
        None => (x, y, rw, rh),
        Some((ref_w, ref_h)) if ref_w == current_w && ref_h == current_h => (x, y, rw, rh),
        Some((ref_w, ref_h)) if ref_w > 0 && ref_h > 0 => {
            let sx = current_w as f32 / ref_w as f32;
            let sy = current_h as f32 / ref_h as f32;
            (
                (x as f32 * sx).round() as u32,
                (y as f32 * sy).round() as u32,
                (rw as f32 * sx).round().max(1.0) as u32,
                (rh as f32 * sy).round().max(1.0) as u32,
            )
        }
        _ => (x, y, rw, rh),
    }
}

/// Crop an image to `(x, y, w, h)` (clamped), returning a new `(H, W, 3)` array.
pub(crate) fn crop_array3(image: &Array3<f32>, x: u32, y: u32, width: u32, height: u32) -> Array3<f32> {
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    let x = x as usize;
    let y = y as usize;
    let rw = width as usize;
    let rh = height as usize;

    let x_start = x.min(w.saturating_sub(1));
    let y_start = y.min(h.saturating_sub(1));
    let x_end = (x + rw).min(w).max(x_start + 1);
    let y_end = (y + rh).min(h).max(y_start + 1);

    image
        .slice(ndarray::s![y_start..y_end, x_start..x_end, ..])
        .to_owned()
}

/// Rotate Array3<f32> (H, W, 3) by 90° clockwise. Returns new array with shape (W, H, 3).
fn rotate_array3_90_cw(image: &Array3<f32>) -> Array3<f32> {
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    let mut out = Array3::<f32>::zeros((w, h, 3));
    for y in 0..h {
        for x in 0..w {
            let (new_y, new_x) = (x, h - 1 - y);
            for ch in 0..3 {
                out[(new_y, new_x, ch)] = image[(y, x, ch)];
            }
        }
    }
    out
}

/// Downsample a single-channel X-Trans array for preview, preserving the 6×6
/// tile period so the CFA pattern survives the downscale intact.
fn downsample_xtrans_for_preview(bayer: &Array3<f32>, max_width: u32) -> Array3<f32> {
    let (h, w, c) = bayer.dim();
    assert_eq!(c, 1, "Expected single-channel CFA for preview");

    if w as u32 <= max_width {
        return bayer.clone();
    }

    let n_super_w = w / 6;
    let n_super_h = h / 6;
    let max_super_w = (max_width as usize / 6).max(1);
    let step = ((n_super_w as f32 / max_super_w as f32).ceil() as usize).max(1);

    let out_super_w = n_super_w / step;
    let out_super_h = n_super_h / step;
    let out_w = out_super_w * 6;
    let out_h = out_super_h * 6;

    let mut out = Array3::<f32>::zeros((out_h, out_w, 1));
    for sy in 0..out_super_h {
        for sx in 0..out_super_w {
            let src_sy = sy * step * 6;
            let src_sx = sx * step * 6;
            for dy in 0..6 {
                for dx in 0..6 {
                    out[(sy * 6 + dy, sx * 6 + dx, 0)] = bayer[(src_sy + dy, src_sx + dx, 0)];
                }
            }
        }
    }
    out
}

/// Dispatch to the correct CFA-aware preview downsampler.
fn downsample_raw_for_preview(
    bayer: &Array3<f32>,
    pattern: CfaPattern,
    max_width: u32,
) -> Array3<f32> {
    match pattern {
        CfaPattern::Bayer(_) => downsample_bayer_for_preview(bayer, max_width),
        CfaPattern::XTrans(_) => downsample_xtrans_for_preview(bayer, max_width),
    }
}

/// Apply rotation (0, 90, 180, 270) to an image. Returns a new Array3.
pub(crate) fn apply_rotation(image: &Array3<f32>, rotation_degrees: i32) -> Array3<f32> {
    let r = ((rotation_degrees % 360 + 360) % 360) / 90;
    match r {
        0 => image.clone(),
        1 => rotate_array3_90_cw(image),
        2 => rotate_array3_90_cw(&rotate_array3_90_cw(image)),
        3 => rotate_array3_90_cw(&rotate_array3_90_cw(&rotate_array3_90_cw(image))),
        _ => image.clone(),
    }
}

/// Downsample a single-channel Bayer array for preview, preserving the 2×2
/// RGGB pattern so demosaic can produce real color.
///
/// Strides through 2×2 super-pixels and copies each block intact.
/// Old code sampled every Nth pixel with N even, which always landed on the
/// same Bayer position (e.g. all R) → grayscale after demosaic.
fn downsample_bayer_for_preview(bayer: &Array3<f32>, max_width: u32) -> Array3<f32> {
    let (h, w, c) = bayer.dim();
    assert_eq!(c, 1, "Expected single-channel Bayer for preview");

    let w_u32 = w as u32;
    if w_u32 <= max_width {
        return bayer.clone();
    }

    let n_super_w = w / 2;
    let n_super_h = h / 2;
    let max_super_w = (max_width as usize / 2).max(1);

    let step = (n_super_w as f32 / max_super_w as f32).ceil().max(1.0) as usize;

    let out_super_w = n_super_w / step;
    let out_super_h = n_super_h / step;
    let out_w = out_super_w * 2;
    let out_h = out_super_h * 2;

    let mut out = Array3::<f32>::zeros((out_h, out_w, 1));

    for sy in 0..out_super_h {
        for sx in 0..out_super_w {
            let src_sy = sy * step * 2;
            let src_sx = sx * step * 2;
            for dy in 0..2 {
                for dx in 0..2 {
                    out[(sy * 2 + dy, sx * 2 + dx, 0)] =
                        bayer[(src_sy + dy, src_sx + dx, 0)];
                }
            }
        }
    }

    out
}

/// Downsample an RGB image for preview to fit within `max_width`×`max_height`,
/// preserving aspect ratio. Used for non-RAW (PNG) previews so the full C-41
/// pipeline only runs on a smaller working resolution.
fn downsample_rgb_for_preview(
    image: &Array3<f32>,
    max_width: u32,
    max_height: u32,
) -> Array3<f32> {
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    let w_u32 = w as u32;
    let h_u32 = h as u32;
    if w_u32 <= max_width && h_u32 <= max_height {
        return image.clone();
    }

    let scale_w = max_width as f32 / w_u32 as f32;
    let scale_h = max_height as f32 / h_u32 as f32;
    let scale = scale_w.min(scale_h).min(1.0);
    let new_w = (w_u32 as f32 * scale).round().max(1.0) as u32;
    let new_h = (h_u32 as f32 * scale).round().max(1.0) as u32;

    let mut img = Rgb32FImage::new(w_u32, h_u32);
    for y in 0..h {
        for x in 0..w {
            let r = image[(y, x, 0)];
            let g = image[(y, x, 1)];
            let b = image[(y, x, 2)];
            img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    let resized = imageops::resize(&img, new_w, new_h, FilterType::CatmullRom);

    let mut out = Array3::<f32>::zeros((new_h as usize, new_w as usize, 3));
    for (x, y, pixel) in resized.enumerate_pixels() {
        let [r, g, b] = pixel.0;
        let yi = y as usize;
        let xi = x as usize;
        out[(yi, xi, 0)] = r;
        out[(yi, xi, 1)] = g;
        out[(yi, xi, 2)] = b;
    }

    out
}

/// **Pipeline order (do not reorder without updating this comment).**
///
/// 1. **Load** RAW (linear Bayer) or PNG → demosaic → **linear RGB**.
/// 3. **D-min / flat-field** (optional).
/// 4. **White balance** (optional).
/// 5. **Optional ACES2065-1 export**: clone image, convert to AP0, write EXR.
/// 6. **Display path**: If curve: T→D → density matrix → RA-4. If no curve: direct density map.
pub fn process_files(
    paths: &[PathBuf],
    output_dir: &Path,
    options: &PipelineOptions,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output directory {}", output_dir.display()))?;

    let lut3d = options
        .lut3d_path
        .as_ref()
        .and_then(|p| lut3d::read_cube(p).ok());

    let output_lut_cube = options
        .output_lut_cube
        .as_ref()
        .and_then(|p| lut3d::read_cube(p).ok());

    // RA-4 curve parameters (used at step 6 if !no_curve).
    let ra4_params = curve::PrintCurveParams {
        offset: options.curve_offset,
        gamma: options.curve_gamma,
        pivot: options.curve_pivot,
    };

    let flat_field_map: Option<Array3<f32>> =
        if let Some(ref flat_path) = options.flat_field_path {
            let ff = flat_field::load_flat_field_map(flat_path)?;
            Some(ff)
        } else {
            None
        };

    for path in paths {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();

        let mut image = match ext.as_str() {
            "arw" | "nef" | "nrw" | "cr2" | "cr3" | "crw" | "dng" | "raf" | "orf" | "rw2" => {
                let (bayer, pattern) = raw_reader::load_raw_as_ndarray(path)?;
                let mut img = demosaic::demosaic_quality(&bayer, pattern)?;
                img.mapv_inplace(|v| v.max(0.0));
                img
            }
            "png" => png_reader::load_png_as_ndarray(path)?,
            _ => continue,
        };

        if options.rotation_degrees != 0 {
            image = apply_rotation(&image, options.rotation_degrees);
        }

        // Step 3: D-min / flat-field.
        if options.debug_pipeline_step >= 3 && options.dmin_mode != DminMode::Off {
            if let Some(ref flat) = flat_field_map {
                flat_field::apply_flat_field_division(&mut image, flat);
            } else {
                match options.dmin_mode {
                    DminMode::Fixed => {
                        if let Some((r, g, b)) = options.dmin_fixed {
                            dmin::neutralize_with_medians(&mut image, r, g, b)?;
                        }
                    }
                    DminMode::SampleRegion => {
                        if let Some(rect) = options.dmin_rect {
                            let (h, w, _) = image.dim();
                            let (x, y, rw, rh) = scale_dmin_rect(
                                rect,
                                options.dmin_rect_reference_size,
                                w as u32,
                                h as u32,
                            );
                            dmin::neutralize(
                                &mut image, x, y, rw, rh, options.dmin_neutral_only,
                            )?;
                        }
                    }
                    DminMode::AutoPercentile => {
                        dmin::auto_percentile_normalize(&mut image, options.auto_norm_buffer)?;
                    }
                    DminMode::Off => {}
                }
            }
            image.mapv_inplace(|v| v.max(0.0));
        }

        // Step 4: T → D → WB (multiplicative) → Film γ
        if options.debug_pipeline_step >= 4 {
            image.mapv_inplace(|t| (-(t.max(1e-10_f32)).log10()).max(0.0));

            // Auto WB: multiplicative density scaling (per-channel γ correction).
            let (auto_s_r, auto_s_g, auto_s_b) = if options.auto_wb && options.dmin_mode != DminMode::Off {
                let stats = stats::wb_channel_stats(&image, options);
                let med_r = stats[0].2.max(1e-4);
                let med_g = stats[1].2.max(1e-4);
                let med_b = stats[2].2.max(1e-4);
                let mean_d = (med_r + med_g + med_b) / 3.0;
                (mean_d / med_r, mean_d / med_g, mean_d / med_b)
            } else {
                (1.0, 1.0, 1.0)
            };

            // Manual WB: density scale factors (default 1.0).
            let (man_s_r, man_s_g, man_s_b) = if options.apply_white_balance {
                (options.wb_r, options.wb_g, options.wb_b)
            } else {
                (1.0, 1.0, 1.0)
            };

            // Film gamma decompression.
            let inv_gamma = 1.0 / options.film_gamma.max(0.1);

            // Combined per-channel scale (single pass).
            let s_r = auto_s_r * man_s_r * inv_gamma;
            let s_g = auto_s_g * man_s_g * inv_gamma;
            let s_b = auto_s_b * man_s_b * inv_gamma;

            image.slice_mut(ndarray::s![.., .., 0]).mapv_inplace(|v| v * s_r);
            image.slice_mut(ndarray::s![.., .., 1]).mapv_inplace(|v| v * s_g);
            image.slice_mut(ndarray::s![.., .., 2]).mapv_inplace(|v| v * s_b);

            // Color temperature: additive density offset (small correction).
            if let Some(k) = options.temp_k {
                let (tr, tg, tb) = color::temp_k_to_wb_gains(k);
                let off_r = -(tr.max(1e-6) as f64).log10() as f32;
                let off_g = -(tg.max(1e-6) as f64).log10() as f32;
                let off_b = -(tb.max(1e-6) as f64).log10() as f32;
                image.slice_mut(ndarray::s![.., .., 0]).mapv_inplace(|v| v + off_r);
                image.slice_mut(ndarray::s![.., .., 1]).mapv_inplace(|v| v + off_g);
                image.slice_mut(ndarray::s![.., .., 2]).mapv_inplace(|v| v + off_b);
            }

            // Step 4.5: Shadow cast correction (auto-neutralize shadow color cast).
            if options.shadow_cast_strength > 0.0 {
                let cast = density_ops::analyze_shadow_cast(&image, 0.8);
                density_ops::apply_shadow_cast_correction(&mut image, cast, options.shadow_cast_strength, 0.8);
            }
        }

        // Step 5: Density matrix / 3D LUT.
        if options.debug_pipeline_step >= 5 {
            let m = if options.apply_color_profile {
                options.density_matrix
            } else {
                [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
            };
            if let Some(ref lut) = lut3d {
                let (h, w, _) = image.dim();
                for y in 0..h {
                    for x in 0..w {
                        let dr = image[[y, x, 0]];
                        let dg = image[[y, x, 1]];
                        let db = image[[y, x, 2]];
                        let [or, og, ob] = lut.sample_density(dr, dg, db);
                        image[[y, x, 0]] = or;
                        image[[y, x, 1]] = og;
                        image[[y, x, 2]] = ob;
                    }
                }
            } else {
                let is_identity = (m[0][0] - 1.0).abs() < 1e-6
                    && m[0][1].abs() < 1e-6 && m[0][2].abs() < 1e-6
                    && m[1][0].abs() < 1e-6 && (m[1][1] - 1.0).abs() < 1e-6 && m[1][2].abs() < 1e-6
                    && m[2][0].abs() < 1e-6 && m[2][1].abs() < 1e-6 && (m[2][2] - 1.0).abs() < 1e-6;
                if !is_identity {
                    let (h, w, _) = image.dim();
                    for y in 0..h {
                        for x in 0..w {
                            let dr = image[[y, x, 0]];
                            let dg = image[[y, x, 1]];
                            let db = image[[y, x, 2]];
                            image[[y, x, 0]] = m[0][0]*dr + m[0][1]*dg + m[0][2]*db;
                            image[[y, x, 1]] = m[1][0]*dr + m[1][1]*dg + m[1][2]*db;
                            image[[y, x, 2]] = m[2][0]*dr + m[2][1]*dg + m[2][2]*db;
                        }
                    }
                }
            }
            image.mapv_inplace(|v| v.max(0.0));
            density_ops::limit_highlight_density_spread(&mut image);
        }

        // Step 5.5: Density-domain saturation boost (before RA-4 curve).
        if options.debug_pipeline_step >= 5 {
            density_ops::apply_density_saturation(&mut image, options.saturation);
            density_ops::apply_zone_density_adjustments(
                &mut image,
                options.zone_shadows,
                options.zone_highlights,
                [options.color_shadows_r, options.color_shadows_g, options.color_shadows_b],
                [options.color_mids_r, options.color_mids_g, options.color_mids_b],
                [options.color_highlights_r, options.color_highlights_g, options.color_highlights_b],
            );
        }

        // Optional crop (export path only): keep only selected region.
        if options.apply_crop {
            if let Some(rect) = options.crop_rect {
                let (h, w, _) = image.dim();
                let (x, y, rw, rh) = scale_dmin_rect(
                    rect,
                    options.crop_rect_reference_size,
                    w as u32,
                    h as u32,
                );
                image = crop_array3(&image, x, y, rw, rh);
            }
        }

        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("image");
        let out_path = output_dir.join(format!("{}.tiff", stem));
        let jpg_path = output_dir.join(format!("{}.jpg", stem));
        let exr_path = output_dir.join(format!("{}.exr", stem));
        let aces_exr_path = output_dir.join(format!("{}_aces2065-1.exr", stem));

        // ACES2065-1 only: image is already in ACEScg (after Step 2).
        if options.write_aces2065_only {
            let mut aces2065 = image.clone();
            aces::linear_acescg_to_aces2065_1(&mut aces2065);
            exr_export::write_exr_aces2065_1(&aces2065, &aces_exr_path)?;
            continue;
        }

        if options.export_aces_exr {
            let mut aces2065 = image.clone();
            aces::linear_acescg_to_aces2065_1(&mut aces2065);
            exr_export::write_exr_aces2065_1(&aces2065, &aces_exr_path)?;
        }

        let write_jpeg_this = options.write_jpeg || options.write_jpeg_only;

        // Step 6: render/output stage.
        if options.debug_pipeline_step >= 6 {
            match options.output_stage {
                OutputStage::Ra4 => {
                    let mut leveled = image.clone();
                    let levels_active = options.lut_in_black != 0.0
                        || options.lut_in_white != 1.0
                        || (options.lut_in_mid - 1.0).abs() > 1e-6;
                    if levels_active {
                        color::apply_density_levels(
                            &mut leveled,
                            4.0,
                            options.lut_in_black,
                            options.lut_in_white,
                            options.lut_in_mid,
                        );
                        leveled.mapv_inplace(|v| v * 4.0);
                    }
                    let mut image_u16 =
                        curve::apply_ra4_from_density(&leveled, ra4_params, 4.0, options.curve_white);
                    post_curve::apply_toe_shoulder_u16(&mut image_u16, options.toe_strength, options.shoulder_strength);
                    post_curve::apply_soft_knee_u16(&mut image_u16, options.soft_clip);
                    if options.apply_lab {
                        color::apply_lab_separation_u16(&mut image_u16, options.lab_separation);
                    }
                    post_curve::apply_highlight_warmth_u16(&mut image_u16, options.highlight_warmth);
                    if !options.write_jpeg_only {
                        tiff_export::write_tiff_u16(&image_u16, &out_path)?;
                    }
                    if options.write_exr {
                        exr_export::write_exr_u16(&image_u16, &exr_path)?;
                    }
                    if write_jpeg_this {
                        let (height, width, _) = image_u16.dim();
                        let mut buf = Vec::with_capacity(height * width * 3);
                        for chunk in image_u16.iter() {
                            buf.push((*chunk >> 8) as u8);
                        }
                        let img = RgbImage::from_raw(width as u32, height as u32, buf)
                            .ok_or_else(|| anyhow::anyhow!("Invalid JPEG dimensions"))?;
                        img.save(&jpg_path)?;
                    }
                }
                OutputStage::FilmPrint => {
                    let fp_params = post_curve::build_film_print_params(options);
                    let mut leveled = image.clone();
                    let levels_active = options.lut_in_black != 0.0
                        || options.lut_in_white != 1.0
                        || (options.lut_in_mid - 1.0).abs() > 1e-6;
                    if levels_active {
                        color::apply_density_levels(
                            &mut leveled,
                            4.0,
                            options.lut_in_black,
                            options.lut_in_white,
                            options.lut_in_mid,
                        );
                        leveled.mapv_inplace(|v| v * 4.0);
                    }
                    let mut image_u16 =
                        curve::apply_film_print_from_density(&leveled, &fp_params, 4.0);
                    post_curve::apply_toe_shoulder_u16(&mut image_u16, options.toe_strength, options.shoulder_strength);
                    post_curve::apply_soft_knee_u16(&mut image_u16, options.soft_clip);
                    if options.apply_lab {
                        color::apply_lab_separation_u16(&mut image_u16, options.lab_separation);
                    }
                    post_curve::apply_highlight_warmth_u16(&mut image_u16, options.highlight_warmth);
                    if !options.write_jpeg_only {
                        tiff_export::write_tiff_u16(&image_u16, &out_path)?;
                    }
                    if options.write_exr {
                        exr_export::write_exr_u16(&image_u16, &exr_path)?;
                    }
                    if write_jpeg_this {
                        let (height, width, _) = image_u16.dim();
                        let mut buf = Vec::with_capacity(height * width * 3);
                        for chunk in image_u16.iter() {
                            buf.push((*chunk >> 8) as u8);
                        }
                        let img = RgbImage::from_raw(width as u32, height as u32, buf)
                            .ok_or_else(|| anyhow::anyhow!("Invalid JPEG dimensions"))?;
                        img.save(&jpg_path)?;
                    }
                }
                OutputStage::None => {
                    // No print curve: direct density → display mapping (existing no-curve path).
                    let mut display = image.clone();
                    if !options.no_invert {
                        const D_DISP_MAX: f32 = 2.5;
                        display.mapv_inplace(|v| (v / D_DISP_MAX).clamp(0.0, 1.0));
                    }
                    if !options.write_jpeg_only {
                        tiff_export::write_tiff(&display, &out_path, options.format)?;
                    }
                    if options.write_exr {
                        exr_export::write_exr_f32(&display, &exr_path)?;
                    }
                    if write_jpeg_this {
                        let (height, width, _) = display.dim();
                        let mut buf = Vec::with_capacity(height * width * 3);
                        for v in display.iter() {
                            buf.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
                        }
                        let img = RgbImage::from_raw(width as u32, height as u32, buf)
                            .ok_or_else(|| anyhow::anyhow!("Invalid JPEG dimensions"))?;
                        img.save(&jpg_path)?;
                    }
                }
                OutputStage::Lut2383 => {
                    let mut display = image.clone();
                    match options.output_lut_encoding {
                        OutputLutEncoding::Rec709 => {
                            color::density_to_rec709_leveled(
                                &mut display,
                                options.lut_in_black,
                                options.lut_in_white,
                                options.lut_in_mid,
                            );
                        }
                        enc => {
                            let d_max = match enc {
                                OutputLutEncoding::CineonLog => 2.046_f32,
                                OutputLutEncoding::LinearDensity => 2.5_f32,
                                OutputLutEncoding::Rec709 => unreachable!(),
                            };
                            color::apply_density_levels(
                                &mut display,
                                d_max,
                                options.lut_in_black,
                                options.lut_in_white,
                                options.lut_in_mid,
                            );
                        }
                    }
                    if let Some(ref lut) = output_lut_cube {
                        post_curve::apply_output_cube_rgb(&mut display, lut);
                    }
                    if options.apply_lab {
                        color::apply_lab_separation_f32(&mut display, options.lab_separation);
                    }
                    post_curve::apply_soft_knee_f32(&mut display, options.soft_clip);
                    post_curve::apply_highlight_warmth_f32(&mut display, options.highlight_warmth);

                    if !options.write_jpeg_only {
                        // Quantize to 16-bit and write TIFF.
                        tiff_export::write_tiff(&display, &out_path, TiffFormat::U16)?;
                    }
                    if options.write_exr {
                        // Write linear display-space EXR.
                        exr_export::write_exr_f32(&display, &exr_path)?;
                    }
                    if write_jpeg_this {
                        let (height, width, _) = display.dim();
                        let mut buf = Vec::with_capacity(height * width * 3);
                        for v in display.iter() {
                            buf.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
                        }
                        let img = RgbImage::from_raw(width as u32, height as u32, buf)
                            .ok_or_else(|| anyhow::anyhow!("Invalid JPEG dimensions"))?;
                        img.save(&jpg_path)?;
                    }
                }
            }
        } else {
            // Steps 1–5: output density (or transmittance if step < 4).
            if !options.write_jpeg_only {
                tiff_export::write_tiff(&image, &out_path, options.format)?;
            }
            if options.write_exr {
                exr_export::write_exr_f32(&image, &exr_path)?;
            }
        }
    }

    Ok(())
}

/// Process a single image for GUI preview. Pipeline order matches `process_files`: load → demosaic →
/// D-min/flat-field → WB → curve or no-curve.
///
/// Returns `(input_w, input_h, preview_w, preview_h, rgb_u8, debug_log)`.
pub fn process_one_to_preview(
    path: &Path,
    options: &PipelineOptions,
    max_width: u32,
    max_height: u32,
) -> anyhow::Result<(u32, u32, u32, u32, Vec<u8>, String)> {
    let mut dbg = String::new();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    // True source dimensions captured before any downsampling.
    let (mut true_src_w, mut true_src_h);

    let mut image = match ext.as_str() {
        "arw" | "nef" | "nrw" | "cr2" | "cr3" | "crw" | "dng" | "raf" | "orf" | "rw2" => {
            let (bayer, pattern) = raw_reader::load_raw_as_ndarray(path)?;
            let (bh, bw, _) = bayer.dim();
            true_src_w = bw as u32;
            true_src_h = bh as u32;
            let small_bayer = downsample_raw_for_preview(&bayer, pattern, max_width);
            let mut img = if options.debug_preview_simple_debayer {
                demosaic::demosaic_bilinear(&small_bayer, pattern)?
            } else {
                demosaic::demosaic_quality(&small_bayer, pattern)?
            };
            img.mapv_inplace(|v| v.max(0.0));
            img
        }
        "png" => {
            let img = png_reader::load_png_as_ndarray(path)?;
            let (ph, pw, _) = img.dim();
            true_src_w = pw as u32;
            true_src_h = ph as u32;
            downsample_rgb_for_preview(&img, max_width, max_height)
        }
        _ => anyhow::bail!("Unsupported extension for preview"),
    };

    // Swap to match output orientation after rotation.
    if options.rotation_degrees == 90 || options.rotation_degrees == 270 {
        std::mem::swap(&mut true_src_w, &mut true_src_h);
    }

    let (dim_h, dim_w, _) = image.dim();
    let _ = writeln!(dbg, "=== Pipeline Debug ===");
    let _ = writeln!(dbg, "image: {}x{} (preview downsampled)", dim_w, dim_h);
    let _ = writeln!(dbg, "rotation: {}°", options.rotation_degrees);
    let _ = writeln!(dbg, "pipeline step: {}", options.debug_pipeline_step);
    let _ = writeln!(dbg);

    if options.rotation_degrees != 0 {
        image = apply_rotation(&image, options.rotation_degrees);
    }

    // Step 1: load + demosaic + rotate
    if options.verbose_debug {
        let _ = write!(dbg, "{}", stats::fmt_stats("Step 1 (load+demosaic+rot):", &stats::channel_stats(&image)));
        let _ = writeln!(dbg);
    }

    // Debug preview mode: show simple demosaic only.
    if options.debug_preview_simple_debayer && ext != "png" {
        let (orig_h, orig_w, _) = image.dim();
        let orig_w = orig_w as u32;
        let orig_h = orig_h as u32;
        let max_v = image
            .iter()
            .copied()
            .fold(0.0_f32, f32::max)
            .max(1.0e-6);
        let inv_max = 1.0 / max_v;
        let rgb_u8: Vec<u8> = image
            .iter()
            .map(|v| color::linear_to_srgb_u8(v * inv_max))
            .collect();
        let img = RgbImage::from_raw(orig_w, orig_h, rgb_u8)
            .ok_or_else(|| anyhow::anyhow!("Invalid image dimensions"))?;
        let scale = (max_width as f32 / orig_w as f32)
            .min(max_height as f32 / orig_h as f32)
            .min(1.0);
        let new_w = (orig_w as f32 * scale).round().max(1.0) as u32;
        let new_h = (orig_h as f32 * scale).round().max(1.0) as u32;
        let resized = imageops::resize(&img, new_w, new_h, FilterType::CatmullRom);
        let out = resized.into_raw();
        return Ok((true_src_w, true_src_h, new_w, new_h, out, dbg));
    }

    // Step 3: D-min / flat-field.
    if options.debug_pipeline_step >= 3 && options.dmin_mode != DminMode::Off {
        if let Some(ref flat_path) = options.flat_field_path {
            let flat_map = flat_field::load_flat_field_map(flat_path)?;
            flat_field::apply_flat_field_division(&mut image, &flat_map);
            let _ = writeln!(dbg, "D-min mode: flat-field ({})", flat_path.display());
        } else {
            match options.dmin_mode {
                DminMode::Fixed => {
                    if let Some((r, g, b)) = options.dmin_fixed {
                        let _ = writeln!(dbg, "D-min mode: fixed ({:.6}, {:.6}, {:.6})", r, g, b);
                        dmin::neutralize_with_medians(&mut image, r, g, b)?;
                    }
                }
                DminMode::SampleRegion => {
                    if let Some(rect) = options.dmin_rect {
                        let (h, w, _) = image.dim();
                        let (x, y, rw, rh) = scale_dmin_rect(
                            rect,
                            options.dmin_rect_reference_size,
                            w as u32,
                            h as u32,
                        );
                        let _ = writeln!(
                            dbg,
                            "D-min mode: rect x={} y={} w={} h={} neutral_only={}",
                            x, y, rw, rh, options.dmin_neutral_only
                        );
                        if options.verbose_debug {
                            let x0 = (x as usize).min(w.saturating_sub(1));
                            let y0 = (y as usize).min(h.saturating_sub(1));
                            let x1 = ((x + rw) as usize).min(w).max(x0 + 1);
                            let y1 = ((y + rh) as usize).min(h).max(y0 + 1);
                            let region = image.slice(ndarray::s![y0..y1, x0..x1, ..]).to_owned();
                            let _ = write!(dbg, "{}", stats::fmt_stats("  D-min sample region (before):", &stats::channel_stats(&region)));
                        }
                        dmin::neutralize(&mut image, x, y, rw, rh, options.dmin_neutral_only)?;
                    }
                }
                DminMode::AutoPercentile => {
                    let _ = writeln!(dbg, "D-min mode: auto-percentile (buffer={:.2})", options.auto_norm_buffer);
                    dmin::auto_percentile_normalize(&mut image, options.auto_norm_buffer)?;
                }
                DminMode::Off => {}
            }
        }
        image.mapv_inplace(|v| v.max(0.0));
        if options.verbose_debug {
            let _ = write!(dbg, "{}", stats::fmt_stats("Step 3 (after D-min, clamped [0,1]):", &stats::channel_stats(&image)));
        }
    } else if options.debug_pipeline_step >= 3 {
        let _ = writeln!(dbg, "Step 3: D-min SKIPPED (dmin_mode=Off)");
    } else {
        let _ = writeln!(dbg, "Step 3: SKIPPED (pipeline_step < 3)");
    }
    let _ = writeln!(dbg);

    // ──────────────────────────────────────────────────────────────────────
    // Step 4: Transmittance → Optical Density → WB (multiplicative) → Film γ
    //
    //   4a  D = -log₁₀(T)
    //   4b  Auto WB:  D *= mean_D / ch_median_D  (per-channel γ correction)
    //   4c  Manual WB: D *= slider              (density scale, default 1.0)
    //   4d  Film γ:   D *= 1/γ                  (density → scene log-exposure)
    //
    //   Multiplicative WB preserves D=0 → 0 for all channels (no black-point shift).
    //   Film γ decompresses the density range by the film's characteristic curve slope.
    // ──────────────────────────────────────────────────────────────────────
    if options.debug_pipeline_step >= 4 {
        // 4a: T → D  (clamp D >= 0: T > 1 after D-min is noise, not signal)
        image.mapv_inplace(|t| (-(t.max(1e-10_f32)).log10()).max(0.0));
        if options.verbose_debug {
            let _ = write!(dbg, "{}", stats::fmt_stats("Step 4a (T→D, density):", &stats::channel_stats(&image)));
        }

        // 4b: Auto WB — multiplicative equalization of per-channel density medians.
        //     D *= mean_D / ch_median_D  (equivalent to per-channel gamma correction).
        let (auto_s_r, auto_s_g, auto_s_b) = if options.auto_wb && options.dmin_mode != DminMode::Off {
            let stats = stats::wb_channel_stats(&image, options);
            let med_r = stats[0].2.max(1e-4);
            let med_g = stats[1].2.max(1e-4);
            let med_b = stats[2].2.max(1e-4);
            let mean_d = (med_r + med_g + med_b) / 3.0;
            (mean_d / med_r, mean_d / med_g, mean_d / med_b)
        } else {
            (1.0, 1.0, 1.0)
        };

        // 4c: Manual WB — density scale factors (slider default 1.0).
        //     >1 = more density = brighter in positive = more of that color.
        let (man_s_r, man_s_g, man_s_b) = if options.apply_white_balance {
            (options.wb_r, options.wb_g, options.wb_b)
        } else {
            (1.0, 1.0, 1.0)
        };

        // 4d: Film gamma — D_scene = D_film / γ.
        let inv_gamma = 1.0 / options.film_gamma.max(0.1);

        // Combined per-channel scale (single pass over the data).
        let s_r = auto_s_r * man_s_r * inv_gamma;
        let s_g = auto_s_g * man_s_g * inv_gamma;
        let s_b = auto_s_b * man_s_b * inv_gamma;

        let _ = writeln!(dbg, "Auto WB (×): R={:.4} G={:.4} B={:.4} (enabled={})",
            auto_s_r, auto_s_g, auto_s_b, options.auto_wb && options.dmin_mode != DminMode::Off);
        let _ = writeln!(dbg, "Manual WB (×): R={:.4} G={:.4} B={:.4}", man_s_r, man_s_g, man_s_b);
        let _ = writeln!(dbg, "Film gamma: {:.3} → 1/γ = {:.4}", options.film_gamma, inv_gamma);
        let _ = writeln!(dbg, "Combined density scale: R={:.4} G={:.4} B={:.4}", s_r, s_g, s_b);

        image.slice_mut(ndarray::s![.., .., 0]).mapv_inplace(|v| v * s_r);
        image.slice_mut(ndarray::s![.., .., 1]).mapv_inplace(|v| v * s_g);
        image.slice_mut(ndarray::s![.., .., 2]).mapv_inplace(|v| v * s_b);

        // Color temperature: additive density offset (small correction, OK to shift black slightly).
        if let Some(k) = options.temp_k {
            let (tr, tg, tb) = color::temp_k_to_wb_gains(k);
            let off_r = -(tr.max(1e-6) as f64).log10() as f32;
            let off_g = -(tg.max(1e-6) as f64).log10() as f32;
            let off_b = -(tb.max(1e-6) as f64).log10() as f32;
            let _ = writeln!(dbg, "Temp {} K → density offset: R={:+.4} G={:+.4} B={:+.4}", k, off_r, off_g, off_b);
            image.slice_mut(ndarray::s![.., .., 0]).mapv_inplace(|v| v + off_r);
            image.slice_mut(ndarray::s![.., .., 1]).mapv_inplace(|v| v + off_g);
            image.slice_mut(ndarray::s![.., .., 2]).mapv_inplace(|v| v + off_b);
        }

        // Step 4.5: Shadow cast correction (auto-neutralize shadow color cast).
        if options.shadow_cast_strength > 0.0 {
            let cast = density_ops::analyze_shadow_cast(&image, 0.8);
            density_ops::apply_shadow_cast_correction(&mut image, cast, options.shadow_cast_strength, 0.8);
            let _ = writeln!(dbg, "Shadow cast correction: vec=({:+.4}, {:+.4}, {:+.4}) strength={:.2}",
                cast.0, cast.1, cast.2, options.shadow_cast_strength);
        }

        if options.verbose_debug {
            let _ = write!(dbg, "{}", stats::fmt_stats("Step 4 (after WB + film γ + shadow cast):", &stats::channel_stats(&image)));
        }
    } else {
        let _ = writeln!(dbg, "Step 4: SKIPPED (pipeline_step < 4)");
    }
    let _ = writeln!(dbg);

    // ──────────────────────────────────────────────────────────────────────
    // Step 5: Density-domain color calibration (3×3 matrix or 3D LUT).
    // ──────────────────────────────────────────────────────────────────────
    if options.debug_pipeline_step >= 5 {
        let m = if options.apply_color_profile {
            options.density_matrix
        } else {
            [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
        };
        let lut3d = options
            .lut3d_path
            .as_ref()
            .and_then(|p| lut3d::read_cube(p).ok());

        let _ = writeln!(
            dbg,
            "Step 5: density matrix [{:.4},{:.4},{:.4}] [{:.4},{:.4},{:.4}] [{:.4},{:.4},{:.4}], lut3d: {}",
            m[0][0], m[0][1], m[0][2],
            m[1][0], m[1][1], m[1][2],
            m[2][0], m[2][1], m[2][2],
            lut3d.is_some(),
        );

        if let Some(ref lut) = lut3d {
            let (h, w, _) = image.dim();
            for y in 0..h {
                for x in 0..w {
                    let dr = image[[y, x, 0]];
                    let dg = image[[y, x, 1]];
                    let db = image[[y, x, 2]];
                    let [or, og, ob] = lut.sample_density(dr, dg, db);
                    image[[y, x, 0]] = or;
                    image[[y, x, 1]] = og;
                    image[[y, x, 2]] = ob;
                }
            }
        } else {
            let is_identity = (m[0][0] - 1.0).abs() < 1e-6
                && m[0][1].abs() < 1e-6 && m[0][2].abs() < 1e-6
                && m[1][0].abs() < 1e-6 && (m[1][1] - 1.0).abs() < 1e-6 && m[1][2].abs() < 1e-6
                && m[2][0].abs() < 1e-6 && m[2][1].abs() < 1e-6 && (m[2][2] - 1.0).abs() < 1e-6;
            if !is_identity {
                let (h, w, _) = image.dim();
                for y in 0..h {
                    for x in 0..w {
                        let dr = image[[y, x, 0]];
                        let dg = image[[y, x, 1]];
                        let db = image[[y, x, 2]];
                        image[[y, x, 0]] = m[0][0]*dr + m[0][1]*dg + m[0][2]*db;
                        image[[y, x, 1]] = m[1][0]*dr + m[1][1]*dg + m[1][2]*db;
                        image[[y, x, 2]] = m[2][0]*dr + m[2][1]*dg + m[2][2]*db;
                    }
                }
            }
        }
        image.mapv_inplace(|v| v.max(0.0));
        density_ops::limit_highlight_density_spread(&mut image);
        if options.verbose_debug {
            let _ = write!(dbg, "{}", stats::fmt_stats("Step 5 (after density matrix):", &stats::channel_stats(&image)));
        }
    } else {
        let _ = writeln!(dbg, "Step 5: SKIPPED (pipeline_step < 5)");
    }
    let _ = writeln!(dbg);

    // Step 5.5: Density-domain saturation boost (before RA-4 curve).
    if options.debug_pipeline_step >= 5 {
        density_ops::apply_density_saturation(&mut image, options.saturation);
        density_ops::apply_zone_density_adjustments(
            &mut image,
            options.zone_shadows,
            options.zone_highlights,
            [options.color_shadows_r, options.color_shadows_g, options.color_shadows_b],
            [options.color_mids_r, options.color_mids_g, options.color_mids_b],
            [options.color_highlights_r, options.color_highlights_g, options.color_highlights_b],
        );
        if options.verbose_debug {
            let _ = writeln!(dbg, "Step 5.5: saturation={:.2}  zone_shadows={:.3}  zone_highlights={:.3}  color_s=[{:.3},{:.3},{:.3}]  color_m=[{:.3},{:.3},{:.3}]  color_h=[{:.3},{:.3},{:.3}]",
                options.saturation, options.zone_shadows, options.zone_highlights,
                options.color_shadows_r, options.color_shadows_g, options.color_shadows_b,
                options.color_mids_r, options.color_mids_g, options.color_mids_b,
                options.color_highlights_r, options.color_highlights_g, options.color_highlights_b);
            let _ = write!(dbg, "{}", stats::fmt_stats("  after saturation+zones:", &stats::channel_stats(&image)));
        }
    }
    let _ = writeln!(dbg);

    let (orig_h, orig_w, _) = image.dim();
    let orig_w = orig_w as u32;
    let orig_h = orig_h as u32;

    // ──────────────────────────────────────────────────────────────────────
    // Step 6: render/output stage.
    // ──────────────────────────────────────────────────────────────────────
    let rgb_u8: Vec<u8> = if options.debug_pipeline_step >= 6 {
        match options.output_stage {
            OutputStage::Ra4 => {
                let params = curve::PrintCurveParams {
                    offset: options.curve_offset,
                    gamma: options.curve_gamma,
                    pivot: options.curve_pivot,
                };
                let _ = writeln!(
                    dbg,
                    "Step 6: RA-4 curve (offset={:.3} gamma={:.3} pivot={:.3} white={:.4} levels=[{:.3}, {:.2}, {:.3}])",
                    params.offset, params.gamma, params.pivot, options.curve_white,
                    options.lut_in_black, options.lut_in_mid, options.lut_in_white,
                );
                let mut leveled = image.clone();
                let levels_active = options.lut_in_black != 0.0
                    || options.lut_in_white != 1.0
                    || (options.lut_in_mid - 1.0).abs() > 1e-6;
                if levels_active {
                    color::apply_density_levels(
                        &mut leveled,
                        4.0,
                        options.lut_in_black,
                        options.lut_in_white,
                        options.lut_in_mid,
                    );
                    leveled.mapv_inplace(|v| v * 4.0);
                }
                if options.verbose_debug {
                    let _ =
                        write!(dbg, "{}", stats::fmt_stats("  pre-curve density:", &stats::channel_stats(&leveled)));
                }
                let mut u16_img =
                    curve::apply_ra4_from_density(&leveled, params, 4.0, options.curve_white);
                post_curve::apply_toe_shoulder_u16(&mut u16_img, options.toe_strength, options.shoulder_strength);
                if options.apply_lab {
                    color::apply_lab_separation_u16(&mut u16_img, options.lab_separation);
                }
                post_curve::apply_highlight_warmth_u16(&mut u16_img, options.highlight_warmth);
                if options.verbose_debug {
                    let u16_stats: [(u16, u16, u16); 3] = {
                        let mut s = [(0u16, 0u16, 0u16); 3];
                        for ch in 0..3 {
                            let slice = u16_img.slice(ndarray::s![.., .., ch]);
                            let mut vals: Vec<u16> = slice.iter().copied().collect();
                            vals.sort_unstable();
                            s[ch] = (
                                vals.first().copied().unwrap_or(0),
                                vals.last().copied().unwrap_or(0),
                                if vals.is_empty() { 0 } else { vals[vals.len() / 2] },
                            );
                        }
                        s
                    };
                    let _ = writeln!(dbg, "  u16 output:");
                    let _ = writeln!(
                        dbg,
                        "    R: min={} max={} med={}",
                        u16_stats[0].0, u16_stats[0].1, u16_stats[0].2
                    );
                    let _ = writeln!(
                        dbg,
                        "    G: min={} max={} med={}",
                        u16_stats[1].0, u16_stats[1].1, u16_stats[1].2
                    );
                    let _ = writeln!(
                        dbg,
                        "    B: min={} max={} med={}",
                        u16_stats[2].0, u16_stats[2].1, u16_stats[2].2
                    );
                }
                u16_img
                    .iter()
                    .map(|v| ((*v as u32) >> 8).min(255) as u8)
                    .collect()
            }
            OutputStage::FilmPrint => {
                let fp_params = post_curve::build_film_print_params(options);
                let _ = writeln!(
                    dbg,
                    "Step 6: Film Print (offset={:.3} gamma={:.3} pivot={:.3} bleed={:.3} vibrance={:.2})",
                    fp_params.base.offset, fp_params.base.gamma, fp_params.base.pivot,
                    fp_params.color_bleed, fp_params.vibrance,
                );
                let _ = writeln!(
                    dbg,
                    "  per-ch offset: [{:+.3}, {:+.3}, {:+.3}]  gamma: [{:.3}, {:.3}, {:.3}]",
                    fp_params.offset_rgb[0], fp_params.offset_rgb[1], fp_params.offset_rgb[2],
                    fp_params.gamma_rgb[0], fp_params.gamma_rgb[1], fp_params.gamma_rgb[2],
                );
                let mut leveled = image.clone();
                let levels_active = options.lut_in_black != 0.0
                    || options.lut_in_white != 1.0
                    || (options.lut_in_mid - 1.0).abs() > 1e-6;
                if levels_active {
                    color::apply_density_levels(
                        &mut leveled,
                        4.0,
                        options.lut_in_black,
                        options.lut_in_white,
                        options.lut_in_mid,
                    );
                    leveled.mapv_inplace(|v| v * 4.0);
                }
                let mut u16_img =
                    curve::apply_film_print_from_density(&leveled, &fp_params, 4.0);
                post_curve::apply_toe_shoulder_u16(&mut u16_img, options.toe_strength, options.shoulder_strength);
                if options.apply_lab {
                    color::apply_lab_separation_u16(&mut u16_img, options.lab_separation);
                }
                post_curve::apply_highlight_warmth_u16(&mut u16_img, options.highlight_warmth);
                u16_img
                    .iter()
                    .map(|v| ((*v as u32) >> 8).min(255) as u8)
                    .collect()
            }
            OutputStage::None => {
                // No-curve positive: density → linear brightness.
                let _ = writeln!(dbg, "Step 6: linear density inversion (no curve)");
                if options.verbose_debug {
                    let _ = write!(dbg, "{}", stats::fmt_stats("  density:", &stats::channel_stats(&image)));
                }
                const D_DISP_MAX: f32 = 2.5;
                image
                    .iter()
                    .map(|v| ((*v / D_DISP_MAX).clamp(0.0, 1.0) * 255.0).round() as u8)
                    .collect()
            }
            OutputStage::Lut2383 => {
                let enc_label = match options.output_lut_encoding {
                    OutputLutEncoding::CineonLog => "Cineon log (D/2.046)",
                    OutputLutEncoding::Rec709 => "Rec.709 (D→linear→sRGB OETF)",
                    OutputLutEncoding::LinearDensity => "Linear (D/2.5)",
                };
                let _ = writeln!(
                    dbg,
                    "Step 6: output LUT (encoding={}, levels=[{:.3}, {:.3}], cube={})",
                    enc_label,
                    options.lut_in_black,
                    options.lut_in_white,
                    options
                        .output_lut_cube
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "none".into()),
                );
                let mut display = image.clone();
                match options.output_lut_encoding {
                    OutputLutEncoding::Rec709 => {
                        color::density_to_rec709_leveled(
                            &mut display,
                            options.lut_in_black,
                            options.lut_in_white,
                            options.lut_in_mid,
                        );
                    }
                    enc => {
                        let d_max = match enc {
                            OutputLutEncoding::CineonLog => 2.046_f32,
                            OutputLutEncoding::LinearDensity => 2.5_f32,
                            OutputLutEncoding::Rec709 => unreachable!(),
                        };
                        color::apply_density_levels(
                            &mut display,
                            d_max,
                            options.lut_in_black,
                            options.lut_in_white,
                            options.lut_in_mid,
                        );
                    }
                }
                if let Some(output_lut) = options
                    .output_lut_cube
                    .as_ref()
                    .and_then(|p| lut3d::read_cube(p).ok())
                {
                    post_curve::apply_output_cube_rgb(&mut display, &output_lut);
                }
                if options.apply_lab {
                    color::apply_lab_separation_f32(&mut display, options.lab_separation);
                }
                post_curve::apply_soft_knee_f32(&mut display, options.soft_clip);
                post_curve::apply_highlight_warmth_f32(&mut display, options.highlight_warmth);

                display
                    .iter()
                    .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
                    .collect()
            }
        }
    } else {
        // Pipeline stopped early: density still in image, map for display.
        let _ = writeln!(dbg, "Steps 1-5 only: density → display");
        const D_DISP_MAX: f32 = 2.5;
        image
            .iter()
            .map(|v| ((*v / D_DISP_MAX).clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect()
    };

    let _ = writeln!(dbg);
    let _ = writeln!(dbg, "=== end pipeline debug ===");

    let img = RgbImage::from_raw(orig_w, orig_h, rgb_u8)
        .ok_or_else(|| anyhow::anyhow!("Invalid image dimensions"))?;

    // Keep full preview resolution (already limited by max_width/max_height at RAW load time).
    // GUI handles zoom/crop/fit-to-window from this buffer.
    let out = img.into_raw();
    Ok((true_src_w, true_src_h, orig_w, orig_h, out, dbg))
}
