//! C-41 RAW pipeline library. Used by both CLI and GUI.

use std::path::{Path, PathBuf};

use anyhow::Context;
use image::{imageops::FilterType, imageops::resize, RgbImage};
use ndarray::{self, Array3};

pub mod curve;
pub mod demosaic;
pub mod dmin;
pub mod exr_export;
pub mod inversion;
pub mod png_reader;
pub mod raw_reader;
pub mod tiff_export;

pub use tiff_export::TiffFormat;

/// Rectangle for D-min sampling (pixel coordinates).
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// All pipeline options (CLI flags / GUI state).
#[derive(Debug, Clone)]
pub struct PipelineOptions {
    pub dmin_rect: Option<Rect>,
    pub dmin_fixed: Option<(f32, f32, f32)>,
    pub format: TiffFormat,
    pub write_exr: bool,
    pub write_jpeg: bool,
    pub no_invert: bool,
    pub no_curve: bool,
    pub wb_r: f32,
    pub wb_g: f32,
    pub wb_b: f32,
    pub curve_offset: f32,
    pub curve_gamma: f32,
    pub curve_pivot: f32,
    pub curve_white: f32,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            dmin_rect: None,
            dmin_fixed: None,
            format: TiffFormat::Float32,
            write_exr: false,
            write_jpeg: false,
            no_invert: false,
            no_curve: false,
            wb_r: 1.0,
            wb_g: 1.0,
            wb_b: 1.0,
            curve_offset: 0.0,
            curve_gamma: 2.5,
            curve_pivot: 3.0,
            curve_white: 1.0,
        }
    }
}

fn downsample_bayer_for_preview(bayer: &Array3<f32>, max_width: u32) -> Array3<f32> {
    let (h, w, c) = bayer.dim();
    assert_eq!(c, 1, "Expected single-channel Bayer for preview");

    let w_u32 = w as u32;
    if w_u32 <= max_width {
        return bayer.clone();
    }

    let mut factor = (w_u32 as f32 / max_width as f32).ceil() as usize;
    if factor < 1 {
        factor = 1;
    }
    if factor % 2 != 0 {
        factor += 1;
    }

    let new_w = w / factor;
    let new_h = h / factor;
    let mut out = Array3::<f32>::zeros((new_h, new_w, 1));

    for y in 0..new_h {
        for x in 0..new_w {
            let src_y = y * factor;
            let src_x = x * factor;
            out[(y, x, 0)] = bayer[(src_y, src_x, 0)];
        }
    }

    out
}

/// Process a list of input files and write TIFF (and optionally EXR) to `output_dir`.
pub fn process_files(
    paths: &[PathBuf],
    output_dir: &Path,
    options: &PipelineOptions,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output directory {}", output_dir.display()))?;

    let lut = (!options.no_curve)
        .then(|| curve::generate_16bit_lut(options.curve_offset, options.curve_gamma, options.curve_pivot));

    for path in paths {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();

        let mut image = match ext.as_str() {
            // RAW formats handled by LibRaw; we assume a Bayer sensor and RGGB for now.
            "arw" | "nef" | "nrw" | "cr2" | "cr3" | "crw" | "dng" | "raf" | "orf" | "rw2" => {
                let bayer = raw_reader::load_raw_as_ndarray(path)?;
                demosaic::demosaic_bilinear(&bayer, demosaic::BayerPattern::Rggb)?
            }
            "png" => png_reader::load_png_as_ndarray(path)?,
            _ => continue,
        };

        if let Some((r, g, b)) = options.dmin_fixed {
            dmin::neutralize_with_medians(&mut image, r, g, b)?;
        } else if let Some(rect) = options.dmin_rect {
            dmin::neutralize(&mut image, rect.x, rect.y, rect.width, rect.height)?;
        }

        if options.wb_r != 1.0 || options.wb_g != 1.0 || options.wb_b != 1.0 {
            image.slice_mut(ndarray::s![.., .., 0]).mapv_inplace(|v| v * options.wb_r);
            image.slice_mut(ndarray::s![.., .., 1]).mapv_inplace(|v| v * options.wb_g);
            image.slice_mut(ndarray::s![.., .., 2]).mapv_inplace(|v| v * options.wb_b);
        }

        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("image");
        let out_path = output_dir.join(format!("{}.tiff", stem));
        let exr_path = output_dir.join(format!("{}.exr", stem));

        if let Some(ref lut) = lut {
            let image_u16 = curve::apply_curve_and_quantize(&image, lut, options.curve_white, true);
            tiff_export::write_tiff_u16(&image_u16, &out_path)?;
            if options.write_exr {
                exr_export::write_exr_u16(&image_u16, &exr_path)?;
            }
            if options.write_jpeg {
                // JPEG is 8-bit; down-quantize from u16.
                let jpg_path = output_dir.join(format!("{}.jpg", stem));
                let (height, width, _) = image_u16.dim();
                let mut buf = Vec::with_capacity(height * width * 3);
                for chunk in image_u16.iter() {
                    buf.push((*chunk >> 8) as u8);
                }
                let img = RgbImage::from_raw(width as u32, height as u32, buf)
                    .ok_or_else(|| anyhow::anyhow!("Invalid JPEG dimensions"))?;
                img.save(&jpg_path)?;
            }
        } else {
            if !options.no_invert {
                inversion::invert(&mut image);
            }
            tiff_export::write_tiff(&image, &out_path, options.format)?;
            if options.write_exr {
                exr_export::write_exr_f32(&image, &exr_path)?;
            }
            if options.write_jpeg {
                let jpg_path = output_dir.join(format!("{}.jpg", stem));
                let (height, width, _) = image.dim();
                let mut buf = Vec::with_capacity(height * width * 3);
                for v in image.iter() {
                    let x = v.clamp(0.0, 1.0);
                    buf.push((x * 255.0).round() as u8);
                }
                let img = RgbImage::from_raw(width as u32, height as u32, buf)
                    .ok_or_else(|| anyhow::anyhow!("Invalid JPEG dimensions"))?;
                img.save(&jpg_path)?;
            }
        }
    }

    Ok(())
}

/// Process a single image in memory and return RGB u8 pixels scaled to fit within
/// `max_width` x `max_height` (aspect ratio preserved). For GUI preview.
pub fn process_one_to_preview(
    path: &Path,
    options: &PipelineOptions,
    max_width: u32,
    max_height: u32,
) -> anyhow::Result<(u32, u32, Vec<u8>)> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    let mut image = match ext.as_str() {
        "arw" | "nef" | "nrw" | "cr2" | "cr3" | "crw" | "dng" | "raf" | "orf" | "rw2" => {
            let bayer = raw_reader::load_raw_as_ndarray(path)?;
            let small_bayer = downsample_bayer_for_preview(&bayer, max_width);
            demosaic::demosaic_bilinear(&small_bayer, demosaic::BayerPattern::Rggb)?
        }
        "png" => png_reader::load_png_as_ndarray(path)?,
        _ => anyhow::bail!("Unsupported extension for preview"),
    };

    if let Some((r, g, b)) = options.dmin_fixed {
        dmin::neutralize_with_medians(&mut image, r, g, b)?;
    } else if let Some(rect) = options.dmin_rect {
        dmin::neutralize(&mut image, rect.x, rect.y, rect.width, rect.height)?;
    }

    if options.wb_r != 1.0 || options.wb_g != 1.0 || options.wb_b != 1.0 {
        image.slice_mut(ndarray::s![.., .., 0]).mapv_inplace(|v| v * options.wb_r);
        image.slice_mut(ndarray::s![.., .., 1]).mapv_inplace(|v| v * options.wb_g);
        image.slice_mut(ndarray::s![.., .., 2]).mapv_inplace(|v| v * options.wb_b);
    }

    let (h, w, _) = image.dim();
    let w = w as u32;
    let h = h as u32;

    let rgb_u8: Vec<u8> = if !options.no_curve {
        let lut = curve::generate_16bit_lut(
            options.curve_offset,
            options.curve_gamma,
            options.curve_pivot,
        );
        let u16_img = curve::apply_curve_and_quantize(&image, &lut, options.curve_white, false);
        u16_img
            .iter()
            .map(|v| ((*v as u32) >> 8).min(255) as u8)
            .collect()
    } else {
        if !options.no_invert {
            inversion::invert(&mut image);
        }
        image
            .iter()
            .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect()
    };

    let img = RgbImage::from_raw(w, h, rgb_u8).ok_or_else(|| anyhow::anyhow!("Invalid image dimensions"))?;

    let scale = (max_width as f32 / w as f32).min(max_height as f32 / h as f32).min(1.0);
    let new_w = (w as f32 * scale).round().max(1.0) as u32;
    let new_h = (h as f32 * scale).round().max(1.0) as u32;

    let resized = resize(&img, new_w, new_h, FilterType::Triangle);
    let out = resized.into_raw();
    Ok((new_w, new_h, out))
}
