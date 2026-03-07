//! C-41 RAW pipeline library. Used by both CLI and GUI.

use std::path::{Path, PathBuf};

use anyhow::Context;
use image::{
    imageops::{self, FilterType},
    Rgb, Rgb32FImage, RgbImage,
};
use ndarray::{self, Array3};

pub mod aces;
pub mod calibration;
pub mod curve;
pub mod demosaic;
pub mod dmin;
pub mod exr_export;
pub mod inversion;
pub mod lut3d;
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
    /// When false, D-min and flat-field correction are skipped.
    pub apply_dmin: bool,
    /// When false, white balance gains are not applied.
    pub apply_white_balance: bool,
    pub dmin_rect: Option<Rect>,
    pub dmin_fixed: Option<(f32, f32, f32)>,
    pub format: TiffFormat,
    pub write_exr: bool,
    pub write_jpeg: bool,
    /// When true, output is JPEG only (no TIFF). Implies JPEG write; "Also export JPG" is irrelevant.
    pub write_jpeg_only: bool,
    pub no_invert: bool,
    pub no_curve: bool,
    pub wb_r: f32,
    pub wb_g: f32,
    pub wb_b: f32,
    pub curve_offset: f32,
    pub curve_gamma: f32,
    pub curve_pivot: f32,
    pub curve_white: f32,
    /// When false, the 3×3 density matrix is ignored (identity is used instead).
    pub apply_color_profile: bool,
    pub density_matrix: [[f32; 3]; 3],
    /// Path to a RAW flat-field (unexposed) frame for luminance calibration. Optional.
    pub flat_field_path: Option<PathBuf>,
    /// 3×3 IDT matrix (camera linear RGB → ACEScg). Default identity; optional profiles in camera_idt/.
    pub idt_matrix: [[f32; 3]; 3],
    /// When true, also write a linear ACES2065-1 EXR alongside display output.
    pub export_aces_exr: bool,
    /// When true, output is only ACES2065-1 EXR (32-bit float); no TIFF/JPEG.
    pub write_aces2065_only: bool,
    /// Optional 3D LUT (density domain) used instead of the density matrix when set.
    /// If present, applied after T→D, before D→RA-4. Generated via "Generate 3D LUT" from current matrix.
    pub lut3d_path: Option<PathBuf>,
    /// Output rotation in degrees: 0, 90, 180, or 270 (applied after load/demosaic).
    pub rotation_degrees: i32,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            apply_dmin: true,
            apply_white_balance: true,
            dmin_rect: None,
            dmin_fixed: None,
            format: TiffFormat::Float32,
            write_exr: false,
            write_jpeg: false,
            write_jpeg_only: false,
            no_invert: false,
            no_curve: false,
            wb_r: 1.0,
            wb_g: 1.0,
            wb_b: 1.0,
            curve_offset: 0.0,
            curve_gamma: 2.5,
            curve_pivot: 3.0,
            curve_white: 1.0,
            apply_color_profile: true,
            density_matrix: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            flat_field_path: None,
            idt_matrix: aces::IDT_IDENTITY,
            export_aces_exr: false,
            write_aces2065_only: false,
            lut3d_path: None,
            rotation_degrees: 0,
        }
    }
}

/// Load a RAW flat-field frame through Step 1 (LibRaw) and Step 2 (Demosaic) only.
/// Returns linear RGB transmittance as `Array3<f32>` (H, W, 3). No D-min, no curve.
/// Use for luminance (flat-field) calibration reference.
pub fn load_flat_field_linear(path: &Path) -> anyhow::Result<Array3<f32>> {
    let bayer = raw_reader::load_raw_as_ndarray(path)?;
    demosaic::demosaic_quality(&bayer, demosaic::BayerPattern::Rggb)
}

/// Build a 1D Gaussian kernel (odd length, normalized). Sigma in pixels.
fn gaussian_1d_kernel(sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 {
        return vec![1.0];
    }
    let half_len = (3.0 * sigma).ceil().max(1.0) as usize;
    let len = 2 * half_len + 1;
    let mut k = Vec::with_capacity(len);
    let mut sum = 0.0_f32;
    for i in 0..len {
        let x = (i as f32) - (half_len as f32);
        let w = (-x * x / (2.0 * sigma * sigma)).exp();
        k.push(w);
        sum += w;
    }
    for w in k.iter_mut() {
        *w /= sum;
    }
    k
}

/// Separable Gaussian blur on (H, W, 3) f32 array. Uses full f32 precision to avoid banding.
/// Sigma in pixels; boundary uses edge clamping.
fn separable_gaussian_f32(image: &Array3<f32>, sigma: f32) -> Array3<f32> {
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    if sigma <= 0.0 {
        return image.to_owned();
    }
    let kernel = gaussian_1d_kernel(sigma);
    let half = kernel.len() / 2;

    // Horizontal pass: (y, x, ch) -> temp(y, x, ch)
    let mut temp = Array3::<f32>::zeros((h, w, 3));
    for y in 0..h {
        for ch in 0..3 {
            for x in 0..w {
                let mut acc = 0.0_f32;
                for (i, &k) in kernel.iter().enumerate() {
                    let xi = (x as i32 + i as i32 - half as i32).clamp(0, w as i32 - 1) as usize;
                    acc += image[(y, xi, ch)] * k;
                }
                temp[(y, x, ch)] = acc;
            }
        }
    }

    // Vertical pass: temp -> out
    let mut out = Array3::<f32>::zeros((h, w, 3));
    for x in 0..w {
        for ch in 0..3 {
            for y in 0..h {
                let mut acc = 0.0_f32;
                for (i, &k) in kernel.iter().enumerate() {
                    let yi = (y as i32 + i as i32 - half as i32).clamp(0, h as i32 - 1) as usize;
                    acc += temp[(yi, x, ch)] * k;
                }
                out[(y, x, ch)] = acc;
            }
        }
    }
    out
}

/// Apply a heavy Gaussian blur to a linear RGB flat-field image to remove film grain and dust,
/// leaving only low-frequency luminance falloff (light source + lens vignetting).
///
/// Input and output are `(height, width, 3)` arrays in linear [0, 1] space.
/// Uses a separable f32 Gaussian to avoid banding from external blur implementations.
pub fn blur_flat_field(input: &Array3<f32>, radius: f32) -> Array3<f32> {
    separable_gaussian_f32(input, radius)
}

/// Load a flat-field map from an image file (e.g. 32f TIFF saved from the GUI).
/// Interprets the data as linear RGB in [0, 1] (or higher).
fn load_flat_field_from_image(path: &Path) -> anyhow::Result<Array3<f32>> {
    let img = image::open(path)?;
    let rgb = img.to_rgb32f();
    let (w, h) = rgb.dimensions();
    let mut out = Array3::<f32>::zeros((h as usize, w as usize, 3));
    for (x, y, pixel) in rgb.enumerate_pixels() {
        let [r, g, b] = pixel.0;
        let yi = y as usize;
        let xi = x as usize;
        out[(yi, xi, 0)] = r;
        out[(yi, xi, 1)] = g;
        out[(yi, xi, 2)] = b;
    }
    Ok(out)
}

/// Load or reconstruct a flat-field map for the pipeline.
/// - RAW inputs are linearized + heavily blurred
/// - Image inputs (TIFF/PNG/etc.) are treated as already-linear maps (no extra blur)
fn load_flat_field_map(path: &Path) -> anyhow::Result<Array3<f32>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        // RAW formats -> linearize then blur to remove grain/dust.
        "arw" | "nef" | "nrw" | "cr2" | "cr3" | "crw" | "dng" | "raf" | "orf" | "rw2" => {
            let linear = load_flat_field_linear(path)?;
            Ok(blur_flat_field(&linear, 60.0))
        }
        // Everything else is treated as an already-prepared map (e.g. 32f TIFF).
        _ => load_flat_field_from_image(path),
    }
}

/// Resize a blurred flat-field to match the target image dimensions (height, width, 3).
fn resize_flat_field(flat: &Array3<f32>, height: usize, width: usize) -> Array3<f32> {
    let (fh, fw, fc) = flat.dim();
    assert_eq!(fc, 3, "resize_flat_field expects 3-channel input");
    if fh == height && fw == width {
        return flat.clone();
    }

    let mut img = Rgb32FImage::new(fw as u32, fh as u32);
    for y in 0..fh {
        for x in 0..fw {
            let r = flat[(y, x, 0)];
            let g = flat[(y, x, 1)];
            let b = flat[(y, x, 2)];
            img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    let resized = imageops::resize(&img, width as u32, height as u32, FilterType::Triangle);

    let mut out = Array3::<f32>::zeros((height, width, 3));
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

/// Apply pixel-by-pixel flat-field division:
/// T_out(x, y) = T_in(x, y) / T_flat_blurred(x, y), with safe division.
fn apply_flat_field_division(image: &mut Array3<f32>, flat_blurred: &Array3<f32>) {
    let (h, w, c) = image.dim();
    assert_eq!(c, 3, "apply_flat_field_division expects 3-channel image");

    let flat_resampled = resize_flat_field(flat_blurred, h, w);
    let eps = 1.0e-6_f32;

    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let denom = flat_resampled[(y, x, ch)].max(eps);
                image[(y, x, ch)] /= denom;
            }
        }
    }
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

/// Apply rotation (0, 90, 180, 270) to an image. Returns a new Array3.
fn apply_rotation(image: &Array3<f32>, rotation_degrees: i32) -> Array3<f32> {
    let r = ((rotation_degrees % 360 + 360) % 360) / 90;
    match r {
        0 => image.clone(),
        1 => rotate_array3_90_cw(image),
        2 => rotate_array3_90_cw(&rotate_array3_90_cw(image)),
        3 => rotate_array3_90_cw(&rotate_array3_90_cw(&rotate_array3_90_cw(image))),
        _ => image.clone(),
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

/// **Pipeline order (do not reorder without updating this comment).**
///
/// Internal colorspace is always ACEScg.
///
/// 1. **Load** RAW (linear Bayer) or PNG → demosaic → **linear camera RGB**.
/// 2. **IDT**: linear camera RGB → **ACEScg** (identity or camera profile from camera_idt/).
///    Flat-field map is converted with the same IDT.
/// 3. **D-min / flat-field** (optional).
/// 4. **White balance** (optional).
/// 5. **Optional ACES2065-1 export**: clone ACEScg image, convert to AP0, write EXR.
/// 6. **Display path**: If curve: T→D → density matrix (in ACEScg) → RA-4 → quantize; convert to
///    **linear sRGB** for TIFF/EXR/JPEG. If no curve: ACEScg → linear sRGB, optional invert, export.
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

    let curve_pipeline = (!options.no_curve).then(|| {
        let params = curve::PrintCurveParams {
            offset: options.curve_offset,
            gamma: options.curve_gamma,
            pivot: options.curve_pivot,
        };
        let base = if options.apply_color_profile {
            options.density_matrix
        } else {
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        };
        let m = aces::convert_density_matrix_to_acescg(base, &options.idt_matrix);
        let matrix = curve::DensityMatrix { m };
        curve::CurvePipeline::new(params, matrix, 4.0, true, lut3d.clone())
    });

    // Optional flat-field: load map once, convert with IDT so division is in ACEScg.
    let mut flat_field_map: Option<Array3<f32>> =
        if let Some(ref flat_path) = options.flat_field_path {
            Some(load_flat_field_map(flat_path)?)
        } else {
            None
        };
    if let Some(ref mut flat) = flat_field_map {
        aces::apply_idt(flat, &options.idt_matrix);
    }

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
                demosaic::demosaic_quality(&bayer, demosaic::BayerPattern::Rggb)?
            }
            "png" => png_reader::load_png_as_ndarray(path)?,
            _ => continue,
        };

        if options.rotation_degrees != 0 {
            image = apply_rotation(&image, options.rotation_degrees);
        }

        // IDT: linear camera RGB → ACEScg (identity or camera profile).
        aces::apply_idt(&mut image, &options.idt_matrix);

        // Step 3: D-min / flat-field (skipped if apply_dmin is false).
        if options.apply_dmin {
            if let Some(ref flat) = flat_field_map {
                apply_flat_field_division(&mut image, flat);
            } else if let Some((r, g, b)) = options.dmin_fixed {
                dmin::neutralize_with_medians(&mut image, r, g, b)?;
            } else if let Some(rect) = options.dmin_rect {
                dmin::neutralize(&mut image, rect.x, rect.y, rect.width, rect.height)?;
            }
        }

        // White balance (skipped if apply_white_balance is false).
        if options.apply_white_balance
            && (options.wb_r != 1.0 || options.wb_g != 1.0 || options.wb_b != 1.0)
        {
            image.slice_mut(ndarray::s![.., .., 0]).mapv_inplace(|v| v * options.wb_r);
            image.slice_mut(ndarray::s![.., .., 1]).mapv_inplace(|v| v * options.wb_g);
            image.slice_mut(ndarray::s![.., .., 2]).mapv_inplace(|v| v * options.wb_b);
        }

        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("image");
        let out_path = output_dir.join(format!("{}.tiff", stem));
        let jpg_path = output_dir.join(format!("{}.jpg", stem));
        let exr_path = output_dir.join(format!("{}.exr", stem));
        let aces_exr_path = output_dir.join(format!("{}_aces2065-1.exr", stem));

        // ACES2065-1 only: write just the EXR (32-bit float) and skip display output.
        if options.write_aces2065_only {
            let mut aces2065 = image.clone();
            aces::linear_acescg_to_aces2065_1(&mut aces2065);
            exr_export::write_exr_aces2065_1(&aces2065, &aces_exr_path)?;
            continue;
        }

        // Optional ACES2065-1 alongside display output.
        if options.export_aces_exr {
            let mut aces2065 = image.clone();
            aces::linear_acescg_to_aces2065_1(&mut aces2065);
            exr_export::write_exr_aces2065_1(&aces2065, &aces_exr_path)?;
        }

        let write_jpeg_this = options.write_jpeg || options.write_jpeg_only;

        if let Some(ref pipeline) = curve_pipeline {
            let mut image_u16 =
                curve::apply_curve_pipeline(&image, pipeline, options.curve_white, true);
            aces::convert_u16_linear_acescg_to_linear_srgb(&mut image_u16);
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
        } else {
            aces::linear_acescg_to_linear_srgb(&mut image);
            if !options.no_invert {
                inversion::invert(&mut image);
            }
            if !options.write_jpeg_only {
                tiff_export::write_tiff(&image, &out_path, options.format)?;
            }
            if options.write_exr {
                exr_export::write_exr_f32(&image, &exr_path)?;
            }
            if write_jpeg_this {
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

/// Process a single image for GUI preview. Pipeline order matches `process_files`: load → demosaic →
/// IDT (camera → ACEScg) → D-min/flat-field → WB → curve (density matrix in ACEScg) or no-curve;
/// display output is converted to sRGB.
pub fn process_one_to_preview(
    path: &Path,
    options: &PipelineOptions,
    max_width: u32,
    max_height: u32,
) -> anyhow::Result<(u32, u32, u32, u32, Vec<u8>)> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    let mut image = match ext.as_str() {
        "arw" | "nef" | "nrw" | "cr2" | "cr3" | "crw" | "dng" | "raf" | "orf" | "rw2" => {
            let bayer = raw_reader::load_raw_as_ndarray(path)?;
            let small_bayer = downsample_bayer_for_preview(&bayer, max_width);
            demosaic::demosaic_quality(&small_bayer, demosaic::BayerPattern::Rggb)?
        }
        "png" => png_reader::load_png_as_ndarray(path)?,
        _ => anyhow::bail!("Unsupported extension for preview"),
    };

    if options.rotation_degrees != 0 {
        image = apply_rotation(&image, options.rotation_degrees);
    }

    aces::apply_idt(&mut image, &options.idt_matrix);

    // Step 3 for preview: D-min / flat-field (skipped if apply_dmin is false).
    if options.apply_dmin {
        if let Some(ref flat_path) = options.flat_field_path {
            let mut flat_map = load_flat_field_map(flat_path)?;
            aces::apply_idt(&mut flat_map, &options.idt_matrix);
            apply_flat_field_division(&mut image, &flat_map);
        } else if let Some((r, g, b)) = options.dmin_fixed {
            dmin::neutralize_with_medians(&mut image, r, g, b)?;
        } else if let Some(rect) = options.dmin_rect {
            dmin::neutralize(&mut image, rect.x, rect.y, rect.width, rect.height)?;
        }
    }

    if options.apply_white_balance
        && (options.wb_r != 1.0 || options.wb_g != 1.0 || options.wb_b != 1.0)
    {
        image.slice_mut(ndarray::s![.., .., 0]).mapv_inplace(|v| v * options.wb_r);
        image.slice_mut(ndarray::s![.., .., 1]).mapv_inplace(|v| v * options.wb_g);
        image.slice_mut(ndarray::s![.., .., 2]).mapv_inplace(|v| v * options.wb_b);
    }

    let (orig_h, orig_w, _) = image.dim();
    let orig_w = orig_w as u32;
    let orig_h = orig_h as u32;

    let rgb_u8: Vec<u8> = if !options.no_curve {
        let params = curve::PrintCurveParams {
            offset: options.curve_offset,
            gamma: options.curve_gamma,
            pivot: options.curve_pivot,
        };
        let base = if options.apply_color_profile {
            options.density_matrix
        } else {
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        };
        let m = aces::convert_density_matrix_to_acescg(base, &options.idt_matrix);
        let matrix = curve::DensityMatrix { m };
        let lut3d = options
            .lut3d_path
            .as_ref()
            .and_then(|p| lut3d::read_cube(p).ok());
        let pipeline = curve::CurvePipeline::new(params, matrix, 4.0, true, lut3d);
        let mut u16_img = curve::apply_curve_pipeline(&image, &pipeline, options.curve_white, false);
        aces::convert_u16_linear_acescg_to_linear_srgb(&mut u16_img);
        u16_img
            .iter()
            .map(|v| ((*v as u32) >> 8).min(255) as u8)
            .collect()
    } else {
        aces::linear_acescg_to_linear_srgb(&mut image);
        if !options.no_invert {
            inversion::invert(&mut image);
        }
        image
            .iter()
            .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect()
    };

    let img = RgbImage::from_raw(orig_w, orig_h, rgb_u8)
        .ok_or_else(|| anyhow::anyhow!("Invalid image dimensions"))?;

    let scale = (max_width as f32 / orig_w as f32)
        .min(max_height as f32 / orig_h as f32)
        .min(1.0);
    let new_w = (orig_w as f32 * scale).round().max(1.0) as u32;
    let new_h = (orig_h as f32 * scale).round().max(1.0) as u32;

    let resized = imageops::resize(&img, new_w, new_h, FilterType::Triangle);
    let out = resized.into_raw();
    Ok((orig_w, orig_h, new_w, new_h, out))
}
