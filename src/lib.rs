//! C-41 RAW pipeline library. Used by both CLI and GUI.

use std::path::{Path, PathBuf};

use anyhow::Context;
use ndarray;

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
#[derive(Debug, Clone, Copy)]
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
            "arw" => {
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
            let image_u16 = curve::apply_curve_and_quantize(&image, lut, options.curve_white);
            tiff_export::write_tiff_u16(&image_u16, &out_path)?;
            if options.write_exr {
                exr_export::write_exr_u16(&image_u16, &exr_path)?;
            }
        } else {
            if !options.no_invert {
                inversion::invert(&mut image);
            }
            tiff_export::write_tiff(&image, &out_path, options.format)?;
            if options.write_exr {
                exr_export::write_exr_f32(&image, &exr_path)?;
            }
        }
    }

    Ok(())
}
