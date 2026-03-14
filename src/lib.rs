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
pub mod pipeline_cache;
pub mod png_reader;
pub mod pipeline;
pub mod post_curve;

#[cfg(feature = "gpu")]
pub mod gpu;
pub mod raw_reader;
pub mod sensor;
pub mod stats;
pub mod tiff_export;

pub use flat_field::{blur_flat_field, load_flat_field_linear};
pub use options::{DminMode, OutputLutEncoding, OutputStage, PipelineOptions, Rect};
pub use pipeline_cache::{PreviewStepCache, hash_after_load, hash_after_step3, hash_after_step4, hash_after_step5};
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

        pipeline::step_3_dmin(&mut image, options, flat_field_map.as_ref())?;
        pipeline::step_4_t_to_d_wb(&mut image, options);
        pipeline::step_5_calibration(&mut image, options, lut3d.as_ref());

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

        let display = pipeline::step_6_render(
            &image,
            options,
            &ra4_params,
            output_lut_cube.as_ref(),
        );

        match &display {
            pipeline::Step6Display::PassthroughDensity(img) => {
                if !options.write_jpeg_only {
                    tiff_export::write_tiff(img, &out_path, options.format)?;
                }
                if options.write_exr {
                    exr_export::write_exr_f32(img, &exr_path)?;
                }
            }
            pipeline::Step6Display::U16(img) => {
                if !options.write_jpeg_only {
                    tiff_export::write_tiff_u16(img, &out_path)?;
                }
                if options.write_exr {
                    exr_export::write_exr_u16(img, &exr_path)?;
                }
                if write_jpeg_this {
                    let (height, width, _) = img.dim();
                    let buf: Vec<u8> = img.iter().map(|v| (*v >> 8) as u8).collect();
                    let rgb = RgbImage::from_raw(width as u32, height as u32, buf)
                        .ok_or_else(|| anyhow::anyhow!("Invalid JPEG dimensions"))?;
                    rgb.save(&jpg_path)?;
                }
            }
            pipeline::Step6Display::F32(img) => {
                if !options.write_jpeg_only {
                    tiff_export::write_tiff(
                        img,
                        &out_path,
                        if options.output_stage == OutputStage::None {
                            options.format
                        } else {
                            TiffFormat::U16
                        },
                    )?;
                }
                if options.write_exr {
                    exr_export::write_exr_f32(img, &exr_path)?;
                }
                if write_jpeg_this {
                    let (height, width, _) = img.dim();
                    let buf: Vec<u8> = img
                        .iter()
                        .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
                        .collect();
                    let rgb = RgbImage::from_raw(width as u32, height as u32, buf)
                        .ok_or_else(|| anyhow::anyhow!("Invalid JPEG dimensions"))?;
                    rgb.save(&jpg_path)?;
                }
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

    // Step 3: D-min / flat-field (shared pipeline step; debug logging only).
    let flat_map_preview = options
        .flat_field_path
        .as_ref()
        .and_then(|p| flat_field::load_flat_field_map(p).ok());
    if options.debug_pipeline_step >= 3 && options.dmin_mode != DminMode::Off {
        if flat_map_preview.is_some() {
            let _ = writeln!(dbg, "D-min mode: flat-field ({})", options.flat_field_path.as_ref().unwrap().display());
        } else {
            match options.dmin_mode {
                DminMode::Fixed => {
                    if let Some((r, g, b)) = options.dmin_fixed {
                        let _ = writeln!(dbg, "D-min mode: fixed ({:.6}, {:.6}, {:.6})", r, g, b);
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
                    }
                }
                DminMode::AutoPercentile => {
                    let _ = writeln!(dbg, "D-min mode: auto-percentile (buffer={:.2})", options.auto_norm_buffer);
                }
                DminMode::Off => {}
            }
        }
    }
    pipeline::step_3_dmin(&mut image, options, flat_map_preview.as_ref())?;
    if options.debug_pipeline_step >= 3 && options.dmin_mode != DminMode::Off && options.verbose_debug {
        let _ = write!(dbg, "{}", stats::fmt_stats("Step 3 (after D-min, clamped [0,1]):", &stats::channel_stats(&image)));
    } else if options.debug_pipeline_step >= 3 && options.dmin_mode == DminMode::Off {
        let _ = writeln!(dbg, "Step 3: D-min SKIPPED (dmin_mode=Off)");
    } else if options.debug_pipeline_step < 3 {
        let _ = writeln!(dbg, "Step 3: SKIPPED (pipeline_step < 3)");
    }
    let _ = writeln!(dbg);

    // Step 4: T→D, WB, film γ (shared pipeline step).
    if options.debug_pipeline_step >= 4 && options.verbose_debug {
        let _ = writeln!(dbg, "Step 4: T→D, WB, film γ (see pipeline)");
    }
    pipeline::step_4_t_to_d_wb(&mut image, options);
    if options.debug_pipeline_step >= 4 && options.verbose_debug {
        let _ = write!(dbg, "{}", stats::fmt_stats("Step 4 (after WB + film γ + shadow cast):", &stats::channel_stats(&image)));
    } else if options.debug_pipeline_step < 4 {
        let _ = writeln!(dbg, "Step 4: SKIPPED (pipeline_step < 4)");
    }
    let _ = writeln!(dbg);

    // Step 5: density matrix / LUT, saturation, zones (shared pipeline step).
    let lut3d_preview = options.lut3d_path.as_ref().and_then(|p| lut3d::read_cube(p).ok());
    if options.debug_pipeline_step >= 5 {
        let _ = writeln!(dbg, "Step 5: density matrix [...], lut3d: {}", lut3d_preview.is_some());
    }
    pipeline::step_5_calibration(&mut image, options, lut3d_preview.as_ref());
    if options.debug_pipeline_step >= 5 && options.verbose_debug {
        let _ = write!(dbg, "{}", stats::fmt_stats("Step 5 (after density matrix):", &stats::channel_stats(&image)));
        let _ = writeln!(dbg, "Step 5.5: saturation={:.2} zones ...", options.saturation);
        let _ = write!(dbg, "{}", stats::fmt_stats("  after saturation+zones:", &stats::channel_stats(&image)));
    } else if options.debug_pipeline_step < 5 {
        let _ = writeln!(dbg, "Step 5: SKIPPED (pipeline_step < 5)");
    }
    let _ = writeln!(dbg);

    let (orig_h, orig_w, _) = image.dim();
    let orig_w = orig_w as u32;
    let orig_h = orig_h as u32;

    let ra4_params = curve::PrintCurveParams {
        offset: options.curve_offset,
        gamma: options.curve_gamma,
        pivot: options.curve_pivot,
    };
    let output_lut_preview = options.output_lut_cube.as_ref().and_then(|p| lut3d::read_cube(p).ok());
    let display = pipeline::step_6_render(&image, options, &ra4_params, output_lut_preview.as_ref());

    if options.debug_pipeline_step >= 6 {
        let _ = writeln!(dbg, "Step 6: {:?} (shared pipeline)", options.output_stage);
        if options.verbose_debug {
            if let pipeline::Step6Display::U16(ref u16_img) = display {
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
                let _ = writeln!(dbg, "  u16 output: R min={} max={} med={}  G min={} max={} med={}  B min={} max={} med={}",
                    s[0].0, s[0].1, s[0].2, s[1].0, s[1].1, s[1].2, s[2].0, s[2].1, s[2].2);
            }
        }
    } else {
        let _ = writeln!(dbg, "Steps 1-5 only: density → display");
    }

    let rgb_u8 = pipeline::step6_display_to_u8(&display);

    let _ = writeln!(dbg);
    let _ = writeln!(dbg, "=== end pipeline debug ===");

    let img = RgbImage::from_raw(orig_w, orig_h, rgb_u8)
        .ok_or_else(|| anyhow::anyhow!("Invalid image dimensions"))?;

    // Keep full preview resolution (already limited by max_width/max_height at RAW load time).
    // GUI handles zoom/crop/fit-to-window from this buffer.
    let out = img.into_raw();
    Ok((true_src_w, true_src_h, orig_w, orig_h, out, dbg))
}

/// Preview with step cache: reuses cached buffers when only options for later steps changed.
/// Returns `(input_w, input_h, preview_w, preview_h, rgb_u8, debug_log, new_cache)`.
/// Pass `cache` from the previous run (e.g. per-image); use the returned cache for the next call.
pub fn process_one_to_preview_with_cache(
    path: &Path,
    options: &PipelineOptions,
    max_width: u32,
    max_height: u32,
    cache: Option<&PreviewStepCache>,
    capture_debug: bool,
) -> anyhow::Result<(u32, u32, u32, u32, Vec<u8>, String, PreviewStepCache)> {
    use pipeline_cache::{hash_after_load, hash_after_step3, hash_after_step4, hash_after_step5};

    // Simple-debayer path: no cache; delegate to full pipeline.
    if options.debug_preview_simple_debayer {
        let mut opts = options.clone();
        opts.verbose_debug = capture_debug;
        let (a, b, c, d, rgb, dbg) = process_one_to_preview(path, &opts, max_width, max_height)?;
        return Ok((a, b, c, d, rgb, dbg, PreviewStepCache::default()));
    }

    let h1 = hash_after_load(path, options, max_width, max_height);
    let h3 = hash_after_step3(path, options, max_width, max_height);
    let h4 = hash_after_step4(path, options, max_width, max_height);
    let h5 = hash_after_step5(path, options, max_width, max_height);

    let mut start_step = 1u8;
    let mut image: Option<Array3<f32>> = None;
    let mut true_src_w: u32 = 0;
    let mut true_src_h: u32 = 0;
    let mut dbg = String::new();

    if let Some(c) = cache {
        if let Some((hash, ref buf, tw, th)) = c.after_load.as_ref() {
            if *hash == h1 {
                image = Some(buf.clone());
                true_src_w = *tw;
                true_src_h = *th;
                start_step = 3;
            }
        }
        if start_step <= 3 {
            if let Some((hash, ref buf)) = c.after_step3.as_ref() {
                if *hash == h3 {
                    image = Some(buf.clone());
                    start_step = 4;
                }
            }
        }
        if start_step <= 4 {
            if let Some((hash, ref buf)) = c.after_step4.as_ref() {
                if *hash == h4 {
                    image = Some(buf.clone());
                    start_step = 5;
                }
            }
        }
        if start_step <= 5 {
            if let Some((hash, ref buf)) = c.after_step5.as_ref() {
                if *hash == h5 {
                    image = Some(buf.clone());
                    start_step = 6;
                }
            }
        }
    }

    let mut new_cache = PreviewStepCache::default();
    // Preserve cache slots we didn't recompute (so e.g. step-6-only change keeps step 3–5 cache).
    if let Some(c) = cache {
        if start_step > 1 {
            new_cache.after_load = c.after_load.clone();
        }
        if start_step > 3 {
            new_cache.after_step3 = c.after_step3.clone();
        }
        if start_step > 4 {
            new_cache.after_step4 = c.after_step4.clone();
        }
        if start_step > 5 {
            new_cache.after_step5 = c.after_step5.clone();
        }
    }

    if start_step == 1 {
        let (mut img, tw, th) = load_and_demosaic_preview(path, options, max_width, max_height)?;
        true_src_w = tw;
        true_src_h = th;
        let _ = writeln!(dbg, "=== Pipeline Debug (with cache) ===");
        let _ = writeln!(dbg, "image: {}x{} (preview)", img.dim().1, img.dim().0);
        let _ = writeln!(dbg, "rotation: {}°", options.rotation_degrees);
        if options.rotation_degrees == 90 || options.rotation_degrees == 270 {
            std::mem::swap(&mut true_src_w, &mut true_src_h);
        }
        if options.rotation_degrees != 0 {
            img = apply_rotation(&img, options.rotation_degrees);
        }
        new_cache.after_load = Some((h1, img.clone(), true_src_w, true_src_h));
        image = Some(img);
    } else {
        let _ = writeln!(dbg, "=== Pipeline Debug (cached from step {}) ===", start_step);
    }

    let mut image = image.expect("image set by load or cache");

    let flat_map_preview = options
        .flat_field_path
        .as_ref()
        .and_then(|p| flat_field::load_flat_field_map(p).ok());
    let lut3d_preview = options.lut3d_path.as_ref().and_then(|p| lut3d::read_cube(p).ok());
    let ra4_params = curve::PrintCurveParams {
        offset: options.curve_offset,
        gamma: options.curve_gamma,
        pivot: options.curve_pivot,
    };
    let output_lut_preview = options.output_lut_cube.as_ref().and_then(|p| lut3d::read_cube(p).ok());

    if start_step <= 3 {
        pipeline::step_3_dmin(&mut image, options, flat_map_preview.as_ref())?;
        new_cache.after_step3 = Some((h3, image.clone()));
    }
    if start_step <= 4 {
        pipeline::step_4_t_to_d_wb(&mut image, options);
        new_cache.after_step4 = Some((h4, image.clone()));
    }
    if start_step <= 5 {
        pipeline::step_5_calibration(&mut image, options, lut3d_preview.as_ref());
        new_cache.after_step5 = Some((h5, image.clone()));
    }

    let (orig_h, orig_w, _) = image.dim();
    let orig_w = orig_w as u32;
    let orig_h = orig_h as u32;
    let display = pipeline::step_6_render(&image, options, &ra4_params, output_lut_preview.as_ref());
    let rgb_u8 = pipeline::step6_display_to_u8(&display);

    let _ = writeln!(dbg, "=== end pipeline debug ===");
    let img = RgbImage::from_raw(orig_w, orig_h, rgb_u8)
        .ok_or_else(|| anyhow::anyhow!("Invalid image dimensions"))?;
    let out = img.into_raw();
    Ok((true_src_w, true_src_h, orig_w, orig_h, out, dbg, new_cache))
}

/// Load and demosaic for preview only (no rotation). Returns (image, true_src_w, true_src_h).
fn load_and_demosaic_preview(
    path: &Path,
    _options: &PipelineOptions,
    max_width: u32,
    max_height: u32,
) -> anyhow::Result<(Array3<f32>, u32, u32)> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();
    let (true_src_w, true_src_h);
    let image = match ext.as_str() {
        "arw" | "nef" | "nrw" | "cr2" | "cr3" | "crw" | "dng" | "raf" | "orf" | "rw2" => {
            let (bayer, pattern) = raw_reader::load_raw_as_ndarray(path)?;
            let (bh, bw, _) = bayer.dim();
            true_src_w = bw as u32;
            true_src_h = bh as u32;
            let small_bayer = downsample_raw_for_preview(&bayer, pattern, max_width);
            let mut img = demosaic::demosaic_quality(&small_bayer, pattern)?;
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
    Ok((image, true_src_w, true_src_h))
}
