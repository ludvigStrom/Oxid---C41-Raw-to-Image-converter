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
    /// When true, automatically equalize per-channel density medians after D-min
    /// using multiplicative correction (per-channel gamma). Preserves D=0 black point.
    pub auto_wb: bool,
    /// C-41 film negative gamma. Scene log-exposure = density / film_gamma.
    /// Typical values: 0.55–0.75 for C-41, ~0.65 default. Applied as D *= 1/gamma
    /// to decompress density into scene-relative log-exposure before the paper curve.
    pub film_gamma: f32,
    pub dmin_rect: Option<Rect>,
    /// When set, the rect is in pixels for this (width, height). Used to scale rect when exporting at full size.
    pub dmin_rect_reference_size: Option<(u32, u32)>,
    /// When false, crop is skipped and full frame is exported.
    pub apply_crop: bool,
    /// Optional export crop rectangle in pixel coordinates.
    pub crop_rect: Option<Rect>,
    /// Reference size for `crop_rect` coordinates (width, height), used to scale to current image size.
    pub crop_rect_reference_size: Option<(u32, u32)>,
    pub dmin_fixed: Option<(f32, f32, f32)>,
    /// When true and using rect, divide all channels by the same value (geometric mean of medians) to remove density without shifting color.
    pub dmin_neutral_only: bool,
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
    /// When set, multiply WB by gains derived from this color temperature (K). e.g. 5500 = daylight, 3000 = tungsten.
    pub temp_k: Option<f32>,
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
    /// Debug: only run pipeline up to this step (1..=6). Preview and export use this. See TODO_DEBUG.md.
    pub debug_pipeline_step: u32,
    /// Debug preview mode: for RAW files, show only a simple bilinear demosaic
    /// (plus optional rotation) and skip the rest of the pipeline.
    pub debug_preview_simple_debayer: bool,
    /// When true, compute per-step channel statistics (min/max/median) in the
    /// debug log. Expensive (sorts entire image per channel per step). Only
    /// enable when the Debug tab is active.
    pub verbose_debug: bool,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            apply_dmin: true,
            apply_white_balance: true,
            auto_wb: true,
            film_gamma: 0.65,
            dmin_rect: None,
            dmin_rect_reference_size: None,
            apply_crop: false,
            crop_rect: None,
            crop_rect_reference_size: None,
            dmin_fixed: None,
            dmin_neutral_only: false,
            format: TiffFormat::Float32,
            write_exr: false,
            write_jpeg: false,
            write_jpeg_only: false,
            no_invert: false,
            no_curve: false,
            wb_r: 1.0,
            wb_g: 1.0,
            wb_b: 1.0,
            temp_k: None,
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
            debug_pipeline_step: 6,
            debug_preview_simple_debayer: false,
            verbose_debug: false,
        }
    }
}

/// Load a RAW flat-field frame through Step 1 (LibRaw) and Step 2 (Demosaic) only.
/// Returns linear RGB transmittance as `Array3<f32>` (H, W, 3). No D-min, no curve.
/// Use for luminance (flat-field) calibration reference.
pub fn load_flat_field_linear(path: &Path) -> anyhow::Result<Array3<f32>> {
    let (bayer, pattern) = raw_reader::load_raw_as_ndarray(path)?;
    let mut img = demosaic::demosaic_quality(&bayer, pattern)?;
    img.mapv_inplace(|v| v.max(0.0));
    Ok(img)
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

/// Black-body temperature (K) to white-balance gains (R, G, B). 5500 K ≈ (1, 1, 1). Lower K = warm light → gains correct toward cool.
fn temp_k_to_wb_gains(temp_k: f32) -> (f32, f32, f32) {
    let t = (temp_k / 100.0).clamp(1.0, 400.0);
    let (r, g, b) = if t <= 66.0 {
        let g = 99.4708025861 * t.ln() - 161.1195681661;
        let b = if t > 19.0 {
            138.5177312231 * (t - 10.0).ln() - 305.0447927307
        } else {
            0.0
        };
        (255.0, g.clamp(0.0, 255.0), b.clamp(0.0, 255.0))
    } else {
        let r = 329.698727446 * (t - 60.0).powf(-0.1332047592);
        let g = 288.1221695283 * (t - 60.0).powf(-0.0755148492);
        (r.clamp(0.0, 255.0), g.clamp(0.0, 255.0), 255.0)
    };
    let r = r.max(1.0);
    let g = g.max(1.0);
    let b = b.max(1.0);
    let gain_r = 255.0 / r;
    let gain_g = 255.0 / g;
    let gain_b = 255.0 / b;
    let geom = (gain_r * gain_g * gain_b).cbrt();
    (gain_r / geom, gain_g / geom, gain_b / geom)
}

/// Scale D-min rect from reference size to current image size. If reference is None or matches current size, returns rect as-is.
fn scale_dmin_rect(
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
fn crop_array3(image: &Array3<f32>, x: u32, y: u32, width: u32, height: u32) -> Array3<f32> {
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

#[inline]
fn linear_to_srgb_u8(v: f32) -> u8 {
    let x = v.clamp(0.0, 1.0);
    let y = if x <= 0.003_130_8 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (y.clamp(0.0, 1.0) * 255.0).round() as u8
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
/// 6. **Display path**: If curve: T→D → density matrix (in ACEScg) → RA-4. If no curve: direct density map.
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

    // RA-4 curve parameters (used at step 6 if !no_curve).
    let ra4_params = curve::PrintCurveParams {
        offset: options.curve_offset,
        gamma: options.curve_gamma,
        pivot: options.curve_pivot,
    };

    // Optional flat-field map (Step 3 input). If IDT is active, transform the map to ACEScg once
    // so flat-field division stays in the same space as the main image.
    let flat_field_map: Option<Array3<f32>> =
        if let Some(ref flat_path) = options.flat_field_path {
            let mut ff = load_flat_field_map(flat_path)?;
            if options.debug_pipeline_step >= 2 && !aces::is_identity(&options.idt_matrix) {
                aces::apply_idt(&mut ff, &options.idt_matrix);
                ff.mapv_inplace(|v| v.max(0.0));
            }
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

        // Step 2: Camera RGB -> ACEScg via IDT.
        if options.debug_pipeline_step >= 2 {
            if !aces::is_identity(&options.idt_matrix) {
                aces::apply_idt(&mut image, &options.idt_matrix);
                // Density pipeline expects non-negative transmittance values.
                image.mapv_inplace(|v| v.max(0.0));
            }
        }

        // Step 3: D-min / flat-field (skipped if apply_dmin is false).
        if options.debug_pipeline_step >= 3 && options.apply_dmin {
            if let Some(ref flat) = flat_field_map {
                apply_flat_field_division(&mut image, flat);
            } else if let Some((r, g, b)) = options.dmin_fixed {
                dmin::neutralize_with_medians(&mut image, r, g, b)?;
            } else if let Some(rect) = options.dmin_rect {
                let (h, w, _) = image.dim();
                let (x, y, rw, rh) = scale_dmin_rect(
                    rect,
                    options.dmin_rect_reference_size,
                    w as u32,
                    h as u32,
                );
                dmin::neutralize(
                    &mut image,
                    x,
                    y,
                    rw,
                    rh,
                    options.dmin_neutral_only,
                )?;
            }
            image.mapv_inplace(|v| v.clamp(0.0, 1.0));
        }

        // Step 4: T → D → WB (multiplicative) → Film γ
        if options.debug_pipeline_step >= 4 {
            image.mapv_inplace(|t| -(t.max(1e-10_f32)).log10());

            // Auto WB: multiplicative density scaling (per-channel γ correction).
            let (auto_s_r, auto_s_g, auto_s_b) = if options.auto_wb && options.apply_dmin {
                let stats = channel_stats(&image);
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
                let (tr, tg, tb) = temp_k_to_wb_gains(k);
                let off_r = -(tr.max(1e-6) as f64).log10() as f32;
                let off_g = -(tg.max(1e-6) as f64).log10() as f32;
                let off_b = -(tb.max(1e-6) as f64).log10() as f32;
                image.slice_mut(ndarray::s![.., .., 0]).mapv_inplace(|v| v + off_r);
                image.slice_mut(ndarray::s![.., .., 1]).mapv_inplace(|v| v + off_g);
                image.slice_mut(ndarray::s![.., .., 2]).mapv_inplace(|v| v + off_b);
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

        // Step 6: RA-4 curve (from density) or direct density output.
        if options.debug_pipeline_step >= 6 && !options.no_curve {
            let image_u16 = curve::apply_ra4_from_density(&image, ra4_params, 4.0, options.curve_white);
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
        } else if options.debug_pipeline_step >= 6 {
            // No-curve: output density values (positive: D/D_max → [0,1]).
            if !options.no_invert {
                const D_DISP_MAX: f32 = 2.5;
                image.mapv_inplace(|v| (v / D_DISP_MAX).clamp(0.0, 1.0));
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
                    buf.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
                }
                let img = RgbImage::from_raw(width as u32, height as u32, buf)
                    .ok_or_else(|| anyhow::anyhow!("Invalid JPEG dimensions"))?;
                img.save(&jpg_path)?;
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

/// Compute per-channel statistics (min, max, median) for a (H, W, 3) image.
fn channel_stats(image: &Array3<f32>) -> [(f32, f32, f32); 3] {
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

fn fmt_stats(label: &str, stats: &[(f32, f32, f32); 3]) -> String {
    format!(
        "{}\n  R: min={:.6} max={:.6} med={:.6}\n  G: min={:.6} max={:.6} med={:.6}\n  B: min={:.6} max={:.6} med={:.6}\n",
        label,
        stats[0].0, stats[0].1, stats[0].2,
        stats[1].0, stats[1].1, stats[1].2,
        stats[2].0, stats[2].1, stats[2].2,
    )
}

/// Process a single image for GUI preview. Pipeline order matches `process_files`: load → demosaic →
/// IDT → D-min/flat-field → WB → curve or no-curve.
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

    let mut image = match ext.as_str() {
        "arw" | "nef" | "nrw" | "cr2" | "cr3" | "crw" | "dng" | "raf" | "orf" | "rw2" => {
            let (bayer, pattern) = raw_reader::load_raw_as_ndarray(path)?;
            let small_bayer = downsample_bayer_for_preview(&bayer, max_width);
            let mut img = if options.debug_preview_simple_debayer {
                demosaic::demosaic_bilinear(&small_bayer, pattern)?
            } else {
                demosaic::demosaic_quality(&small_bayer, pattern)?
            };
            img.mapv_inplace(|v| v.max(0.0));
            img
        }
        "png" => png_reader::load_png_as_ndarray(path)?,
        _ => anyhow::bail!("Unsupported extension for preview"),
    };

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
        let _ = write!(dbg, "{}", fmt_stats("Step 1 (load+demosaic+rot):", &channel_stats(&image)));
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
            .map(|v| linear_to_srgb_u8(v * inv_max))
            .collect();
        let img = RgbImage::from_raw(orig_w, orig_h, rgb_u8)
            .ok_or_else(|| anyhow::anyhow!("Invalid image dimensions"))?;
        let scale = (max_width as f32 / orig_w as f32)
            .min(max_height as f32 / orig_h as f32)
            .min(1.0);
        let new_w = (orig_w as f32 * scale).round().max(1.0) as u32;
        let new_h = (orig_h as f32 * scale).round().max(1.0) as u32;
        let resized = imageops::resize(&img, new_w, new_h, FilterType::Triangle);
        let out = resized.into_raw();
        return Ok((orig_w, orig_h, new_w, new_h, out, dbg));
    }

    // Step 2: Camera RGB -> ACEScg via IDT.
    if options.debug_pipeline_step >= 2 {
        let idt_is_identity = aces::is_identity(&options.idt_matrix);
        let _ = writeln!(dbg, "Step 2: IDT (identity={})", idt_is_identity);
        if !idt_is_identity {
            aces::apply_idt(&mut image, &options.idt_matrix);
            image.mapv_inplace(|v| v.max(0.0));
        }
        if options.verbose_debug {
            let _ = write!(dbg, "{}", fmt_stats("Step 2 (after IDT):", &channel_stats(&image)));
        }
    } else {
        let _ = writeln!(dbg, "Step 2: SKIPPED (pipeline_step < 2)");
    }
    let _ = writeln!(dbg);

    // Step 3: D-min / flat-field.
    if options.debug_pipeline_step >= 3 && options.apply_dmin {
        if let Some(ref flat_path) = options.flat_field_path {
            let mut flat_map = load_flat_field_map(flat_path)?;
            if options.debug_pipeline_step >= 2 && !aces::is_identity(&options.idt_matrix) {
                aces::apply_idt(&mut flat_map, &options.idt_matrix);
                flat_map.mapv_inplace(|v| v.max(0.0));
            }
            apply_flat_field_division(&mut image, &flat_map);
            let _ = writeln!(dbg, "D-min mode: flat-field ({})", flat_path.display());
        } else if let Some((r, g, b)) = options.dmin_fixed {
            let _ = writeln!(dbg, "D-min mode: fixed ({:.6}, {:.6}, {:.6})", r, g, b);
            dmin::neutralize_with_medians(&mut image, r, g, b)?;
        } else if let Some(rect) = options.dmin_rect {
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
                let _ = write!(dbg, "{}", fmt_stats("  D-min sample region (before):", &channel_stats(&region)));
            }
            dmin::neutralize(&mut image, x, y, rw, rh, options.dmin_neutral_only)?;
        }
        image.mapv_inplace(|v| v.clamp(0.0, 1.0));
        if options.verbose_debug {
            let _ = write!(dbg, "{}", fmt_stats("Step 3 (after D-min, clamped [0,1]):", &channel_stats(&image)));
        }
    } else if options.debug_pipeline_step >= 3 {
        let _ = writeln!(dbg, "Step 3: D-min SKIPPED (apply_dmin=false)");
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
        // 4a: T → D
        image.mapv_inplace(|t| -(t.max(1e-10_f32)).log10());
        if options.verbose_debug {
            let _ = write!(dbg, "{}", fmt_stats("Step 4a (T→D, density):", &channel_stats(&image)));
        }

        // 4b: Auto WB — multiplicative equalization of per-channel density medians.
        //     D *= mean_D / ch_median_D  (equivalent to per-channel gamma correction).
        let (auto_s_r, auto_s_g, auto_s_b) = if options.auto_wb && options.apply_dmin {
            let stats = channel_stats(&image);
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
            auto_s_r, auto_s_g, auto_s_b, options.auto_wb && options.apply_dmin);
        let _ = writeln!(dbg, "Manual WB (×): R={:.4} G={:.4} B={:.4}", man_s_r, man_s_g, man_s_b);
        let _ = writeln!(dbg, "Film gamma: {:.3} → 1/γ = {:.4}", options.film_gamma, inv_gamma);
        let _ = writeln!(dbg, "Combined density scale: R={:.4} G={:.4} B={:.4}", s_r, s_g, s_b);

        image.slice_mut(ndarray::s![.., .., 0]).mapv_inplace(|v| v * s_r);
        image.slice_mut(ndarray::s![.., .., 1]).mapv_inplace(|v| v * s_g);
        image.slice_mut(ndarray::s![.., .., 2]).mapv_inplace(|v| v * s_b);

        // Color temperature: additive density offset (small correction, OK to shift black slightly).
        if let Some(k) = options.temp_k {
            let (tr, tg, tb) = temp_k_to_wb_gains(k);
            let off_r = -(tr.max(1e-6) as f64).log10() as f32;
            let off_g = -(tg.max(1e-6) as f64).log10() as f32;
            let off_b = -(tb.max(1e-6) as f64).log10() as f32;
            let _ = writeln!(dbg, "Temp {} K → density offset: R={:+.4} G={:+.4} B={:+.4}", k, off_r, off_g, off_b);
            image.slice_mut(ndarray::s![.., .., 0]).mapv_inplace(|v| v + off_r);
            image.slice_mut(ndarray::s![.., .., 1]).mapv_inplace(|v| v + off_g);
            image.slice_mut(ndarray::s![.., .., 2]).mapv_inplace(|v| v + off_b);
        }

        if options.verbose_debug {
            let _ = write!(dbg, "{}", fmt_stats("Step 4 (after WB + film γ):", &channel_stats(&image)));
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
        if options.verbose_debug {
            let _ = write!(dbg, "{}", fmt_stats("Step 5 (after density matrix):", &channel_stats(&image)));
        }
    } else {
        let _ = writeln!(dbg, "Step 5: SKIPPED (pipeline_step < 5)");
    }
    let _ = writeln!(dbg);

    let (orig_h, orig_w, _) = image.dim();
    let orig_w = orig_w as u32;
    let orig_h = orig_h as u32;

    // ──────────────────────────────────────────────────────────────────────
    // Step 6: RA-4 print curve (density → positive) or linear density map.
    //         Image is already in density domain; the curve applies the
    //         virtual enlarger exposure + Michaelis-Menten S-curve.
    // ──────────────────────────────────────────────────────────────────────
    let rgb_u8: Vec<u8> = if options.debug_pipeline_step >= 6 && !options.no_curve {
        let params = curve::PrintCurveParams {
            offset: options.curve_offset,
            gamma: options.curve_gamma,
            pivot: options.curve_pivot,
        };
        let _ = writeln!(
            dbg,
            "Step 6: RA-4 curve (offset={:.3} gamma={:.3} pivot={:.3} white={:.4})",
            params.offset, params.gamma, params.pivot, options.curve_white
        );
        if options.verbose_debug {
            let _ = write!(dbg, "{}", fmt_stats("  pre-curve density:", &channel_stats(&image)));
        }
        let u16_img = curve::apply_ra4_from_density(&image, params, 4.0, options.curve_white);
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
            let _ = writeln!(dbg, "    R: min={} max={} med={}", u16_stats[0].0, u16_stats[0].1, u16_stats[0].2);
            let _ = writeln!(dbg, "    G: min={} max={} med={}", u16_stats[1].0, u16_stats[1].1, u16_stats[1].2);
            let _ = writeln!(dbg, "    B: min={} max={} med={}", u16_stats[2].0, u16_stats[2].1, u16_stats[2].2);
        }
        u16_img
            .iter()
            .map(|v| ((*v as u32) >> 8).min(255) as u8)
            .collect()
    } else if options.debug_pipeline_step >= 6 && options.no_invert {
        // Debug: output raw density as grayscale (D/4 → [0,1]).
        let _ = writeln!(dbg, "Step 6: raw density output (no curve, no invert)");
        const D_DISP_MAX: f32 = 4.0;
        image
            .iter()
            .map(|v| ((*v / D_DISP_MAX).clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect()
    } else if options.debug_pipeline_step >= 6 {
        // No-curve positive: density → linear brightness.
        // Higher density = more dye = brighter subject = brighter output.
        let _ = writeln!(dbg, "Step 6: linear density inversion (no curve)");
        if options.verbose_debug {
            let _ = write!(dbg, "{}", fmt_stats("  density:", &channel_stats(&image)));
        }
        const D_DISP_MAX: f32 = 2.5;
        image
            .iter()
            .map(|v| ((*v / D_DISP_MAX).clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect()
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

    let scale = (max_width as f32 / orig_w as f32)
        .min(max_height as f32 / orig_h as f32)
        .min(1.0);
    let new_w = (orig_w as f32 * scale).round().max(1.0) as u32;
    let new_h = (orig_h as f32 * scale).round().max(1.0) as u32;

    let resized = imageops::resize(&img, new_w, new_h, FilterType::Triangle);
    let out = resized.into_raw();
    Ok((orig_w, orig_h, new_w, new_h, out, dbg))
}
