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
pub mod options;
pub mod png_reader;
pub mod raw_reader;
pub mod tiff_export;

pub use options::{DminMode, OutputLutEncoding, OutputStage, PipelineOptions, Rect};
pub use tiff_export::TiffFormat;

use crate::demosaic::CfaPattern;

/// Raw sensor data cached for fast previews/exports.
#[derive(Debug, Clone)]
pub enum CachedSensor {
    /// Single-channel CFA mosaic (Bayer or X-Trans) plus its pattern descriptor.
    Bayer {
        data: Array3<f32>,
        pattern: CfaPattern,
    },
    /// Linear RGB image (e.g. PNG or already-demosaiced source).
    Rgb(Array3<f32>),
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

/// Load raw sensor data (Bayer or RGB) from disk into a cached representation.
pub fn load_sensor_from_path(path: &Path) -> anyhow::Result<CachedSensor> {
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
) -> anyhow::Result<(f32, f32, f32)> {
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

    // 4) Clamp rect and collect values in region, mirroring dmin::neutralize.
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
    let n = (y_end - y_start) * (x_end - x_start);

    let mut r_vals = Vec::with_capacity(n);
    let mut g_vals = Vec::with_capacity(n);
    let mut b_vals = Vec::with_capacity(n);

    for row in region.axis_iter(ndarray::Axis(0)) {
        for pixel in row.axis_iter(ndarray::Axis(0)) {
            r_vals.push(pixel[0]);
            g_vals.push(pixel[1]);
            b_vals.push(pixel[2]);
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

/// Normalize density to [0, 1] with levels remap: D/d_max → [0,1], then
/// stretch [black, white] → [0, 1], then midpoint gamma: v^(1/mid).
/// Identity when black=0, white=1, mid=1.
fn apply_density_levels(image: &mut Array3<f32>, d_max: f32, in_black: f32, in_white: f32, mid: f32) {
    let range = (in_white - in_black).max(1e-6);
    let inv_mid = 1.0 / mid.clamp(0.01, 10.0);
    let apply_gamma = (mid - 1.0).abs() > 1e-6;
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let mut v = (image[[y, x, ch]] / d_max).clamp(0.0, 1.0);
                v = ((v - in_black) / range).clamp(0.0, 1.0);
                if apply_gamma {
                    v = v.powf(inv_mid);
                }
                image[[y, x, ch]] = v;
            }
        }
    }
}

/// sRGB / Rec.709 OETF (linear → gamma-encoded).
#[inline]
fn linear_to_srgb(v: f32) -> f32 {
    let x = v.clamp(0.0, 1.0);
    if x <= 0.003_130_8 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

#[inline]
fn srgb_to_linear(v: f32) -> f32 {
    let x = v.clamp(0.0, 1.0);
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

// ───────────────────────────── Lab helpers ─────────────────────────────

#[inline]
fn rgb_linear_to_xyz(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // sRGB/Rec.709 primaries, D65 white.
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
    (x, y, z)
}

#[inline]
fn xyz_to_rgb_linear(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    // Inverse of rgb_linear_to_xyz for sRGB/Rec.709, D65.
    let r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    let b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    (r, g, b)
}

#[inline]
fn xyz_to_lab(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    // D65 reference white (sRGB).
    const XN: f32 = 0.95047;
    const YN: f32 = 1.0;
    const ZN: f32 = 1.08883;

    let xr = x / XN;
    let yr = y / YN;
    let zr = z / ZN;

    #[inline]
    fn f(t: f32) -> f32 {
        const EPS: f32 = 216.0 / 24389.0; // ~0.008856
        const KAPPA: f32 = 24389.0 / 27.0; // ~903.3
        if t > EPS {
            t.cbrt()
        } else {
            (KAPPA * t + 16.0) / 116.0
        }
    }

    let fx = f(xr);
    let fy = f(yr);
    let fz = f(zr);

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);
    (l, a, b)
}

#[inline]
fn lab_to_xyz(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    const XN: f32 = 0.95047;
    const YN: f32 = 1.0;
    const ZN: f32 = 1.08883;

    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;

    #[inline]
    fn f_inv(t: f32) -> f32 {
        const EPS: f32 = 216.0 / 24389.0;
        const KAPPA: f32 = 24389.0 / 27.0;
        let t3 = t * t * t;
        if t3 > EPS {
            t3
        } else {
            (116.0 * t - 16.0) / KAPPA
        }
    }

    let xr = f_inv(fx);
    let yr = f_inv(fy);
    let zr = f_inv(fz);

    (xr * XN, yr * YN, zr * ZN)
}

/// Apply Lab-space separation on an f32 RGB image in [0, 1]. Strength is
/// typically 0.0–1.0. Neutrals (low chroma) are largely preserved; mid-chroma
/// colors are pushed outward in the a/b plane to increase separation.
fn apply_lab_separation_f32(image: &mut Array3<f32>, strength: f32) {
    if strength.abs() < 1e-6 {
        return;
    }
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    let s = strength.clamp(-2.0, 2.0);

    for y in 0..h {
        for x in 0..w {
            let sr = image[[y, x, 0]].clamp(0.0, 1.0);
            let sg = image[[y, x, 1]].clamp(0.0, 1.0);
            let sb = image[[y, x, 2]].clamp(0.0, 1.0);

            let r_lin = srgb_to_linear(sr);
            let g_lin = srgb_to_linear(sg);
            let b_lin = srgb_to_linear(sb);

            let (xv, yv, zv) = rgb_linear_to_xyz(r_lin, g_lin, b_lin);
            let (l, a, b) = xyz_to_lab(xv, yv, zv);

            let c_ab = (a * a + b * b).sqrt();
            if c_ab < 1e-4 {
                // Near-neutral; keep as-is.
                continue;
            }
            let c_norm = (c_ab / 100.0).clamp(0.0, 1.0);
            // Bell-shaped mid-chroma emphasis over 0..1.
            let mid_boost = 1.0 + s * (c_norm * (1.0 - c_norm)) * 2.0;
            // Soften boost near very high chroma to avoid clipping.
            let edge_soften = 1.0 + 0.2 * s * (1.0 - c_norm);
            let gain = (mid_boost * edge_soften).max(0.0);

            let scale = gain;
            let a2 = a * scale;
            let b2 = b * scale;

            let (x2, y2, z2) = lab_to_xyz(l, a2, b2);
            let (r_lin2, g_lin2, b_lin2) = xyz_to_rgb_linear(x2, y2, z2);

            image[[y, x, 0]] = linear_to_srgb(r_lin2).clamp(0.0, 1.0);
            image[[y, x, 1]] = linear_to_srgb(g_lin2).clamp(0.0, 1.0);
            image[[y, x, 2]] = linear_to_srgb(b_lin2).clamp(0.0, 1.0);
        }
    }
}

/// Apply Lab separation to a u16 RGB image (0–65535) in-place by converting
/// to f32, running `apply_lab_separation_f32`, then quantizing back.
fn apply_lab_separation_u16(image: &mut Array3<u16>, strength: f32) {
    if strength.abs() < 1e-6 {
        return;
    }
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    let inv = 1.0 / 65535.0_f32;

    // Convert to f32 in [0,1]
    let mut fimg = Array3::<f32>::zeros((h, w, c));
    for y in 0..h {
        for x in 0..w {
            fimg[[y, x, 0]] = image[[y, x, 0]] as f32 * inv;
            fimg[[y, x, 1]] = image[[y, x, 1]] as f32 * inv;
            fimg[[y, x, 2]] = image[[y, x, 2]] as f32 * inv;
        }
    }

    apply_lab_separation_f32(&mut fimg, strength);

    // Quantize back.
    for y in 0..h {
        for x in 0..w {
            image[[y, x, 0]] = (fimg[[y, x, 0]].clamp(0.0, 1.0) * 65535.0).round() as u16;
            image[[y, x, 1]] = (fimg[[y, x, 1]].clamp(0.0, 1.0) * 65535.0).round() as u16;
            image[[y, x, 2]] = (fimg[[y, x, 2]].clamp(0.0, 1.0) * 65535.0).round() as u16;
        }
    }
}

/// Normalize density to [0, 1] with sRGB/Rec.709 gamma, then levels remap + midpoint.
/// The LUT handles the neg→pos inversion (print emulation), so we keep
/// density orientation: D / d_max → gamma-encode → levels → midpoint.
fn density_to_rec709_leveled(
    image: &mut Array3<f32>,
    in_black: f32,
    in_white: f32,
    mid: f32,
) {
    const D_MAX: f32 = 2.5;
    let range = (in_white - in_black).max(1e-6);
    let inv_mid = 1.0 / mid.clamp(0.01, 10.0);
    let apply_mid = (mid - 1.0).abs() > 1e-6;
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let norm = (image[[y, x, ch]] / D_MAX).clamp(0.0, 1.0);
                let gamma = linear_to_srgb(norm);
                let mut v = ((gamma - in_black) / range).clamp(0.0, 1.0);
                if apply_mid {
                    v = v.powf(inv_mid);
                }
                image[[y, x, ch]] = v;
            }
        }
    }
}

/// Density-domain saturation boost: scale per-channel deviation from the
/// neutral axis (equal-density gray line).
///
///   D_mean  = (D_r + D_g + D_b) / 3
///   D_ch' = D_mean + saturation * (D_ch - D_mean)
///
/// Detect and compress density-domain "speckle" pixels where one channel is
/// an extreme outlier while the other two are close together.  Normal
/// saturated colors have channels that are roughly evenly spaced (e.g.
/// 0.3, 0.5, 0.7) and are left untouched.  Speckles have a skewed
/// distribution (e.g. 1.0, 1.0, 2.5) and get pulled toward the mean.
fn limit_highlight_density_spread(image: &mut Array3<f32>) {
    let (h, w, _) = image.dim();
    for y in 0..h {
        for x in 0..w {
            let r = image[[y, x, 0]];
            let g = image[[y, x, 1]];
            let b = image[[y, x, 2]];

            let mut lo = r;
            let mut mid = g;
            let mut hi = b;
            if lo > mid { std::mem::swap(&mut lo, &mut mid); }
            if mid > hi { std::mem::swap(&mut mid, &mut hi); }
            if lo > mid { std::mem::swap(&mut lo, &mut mid); }

            let range = hi - lo;
            if range < 0.02 { continue; }

            let mid_pos = (mid - lo) / range;
            let outlier = (0.5 - mid_pos).abs() * 2.0;
            if outlier < 0.5 { continue; }

            let excess = (outlier - 0.5) / 0.5;
            let blend = excess * 0.85;
            let mean = (r + g + b) * (1.0 / 3.0);

            image[[y, x, 0]] = r + (mean - r) * blend;
            image[[y, x, 1]] = g + (mean - g) * blend;
            image[[y, x, 2]] = b + (mean - b) * blend;
        }
    }
}

/// sat > 1 widens channel spread → more colorful output after the S-curve.
/// sat = 1 is identity. Values are clamped to ≥ 0 after boosting.
fn apply_density_saturation(image: &mut Array3<f32>, saturation: f32) {
    if (saturation - 1.0).abs() < 1e-6 {
        return;
    }
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    for y in 0..h {
        for x in 0..w {
            let dr = image[[y, x, 0]];
            let dg = image[[y, x, 1]];
            let db = image[[y, x, 2]];
            let d_mean = (dr + dg + db) * (1.0 / 3.0);
            image[[y, x, 0]] = (d_mean + saturation * (dr - d_mean)).max(0.0);
            image[[y, x, 1]] = (d_mean + saturation * (dg - d_mean)).max(0.0);
            image[[y, x, 2]] = (d_mean + saturation * (db - d_mean)).max(0.0);
        }
    }
}

/// Analyze shadow cast: measure per-channel color imbalance in the low-density
/// (shadow) zone. Returns a correction vector (dr, dg, db) that pulls the shadow
/// average toward neutral gray. All zeros if no shadow pixels found.
fn analyze_shadow_cast(image: &Array3<f32>, threshold: f32) -> (f32, f32, f32) {
    let (h, w, _) = image.dim();
    let mut sum_r = 0.0_f64;
    let mut sum_g = 0.0_f64;
    let mut sum_b = 0.0_f64;
    let mut count = 0u64;

    for y in 0..h {
        for x in 0..w {
            let dr = image[[y, x, 0]];
            let dg = image[[y, x, 1]];
            let db = image[[y, x, 2]];
            let d_mean = (dr + dg + db) * (1.0 / 3.0);
            if d_mean < threshold {
                sum_r += dr as f64;
                sum_g += dg as f64;
                sum_b += db as f64;
                count += 1;
            }
        }
    }

    if count == 0 {
        return (0.0, 0.0, 0.0);
    }

    let avg_r = (sum_r / count as f64) as f32;
    let avg_g = (sum_g / count as f64) as f32;
    let avg_b = (sum_b / count as f64) as f32;
    let target = (avg_r + avg_g + avg_b) * (1.0 / 3.0);

    (target - avg_r, target - avg_g, target - avg_b)
}

/// Apply shadow cast correction: adds the correction vector weighted by a smooth
/// ramp that is strongest near D=0 (deep shadows) and fades to zero by
/// `threshold`. The exponent (1.5) makes the falloff nonlinear so midtones
/// are barely affected.
fn apply_shadow_cast_correction(
    image: &mut Array3<f32>,
    correction: (f32, f32, f32),
    strength: f32,
    threshold: f32,
) {
    if strength.abs() < 1e-6 {
        return;
    }
    let (cr, cg, cb) = correction;
    if cr.abs() < 1e-6 && cg.abs() < 1e-6 && cb.abs() < 1e-6 {
        return;
    }
    let (h, w, _) = image.dim();
    let inv_thresh = 1.0 / threshold.max(1e-6);

    for y in 0..h {
        for x in 0..w {
            let dr = image[[y, x, 0]];
            let dg = image[[y, x, 1]];
            let db = image[[y, x, 2]];
            let d_mean = (dr + dg + db) * (1.0 / 3.0);

            let t = (1.0 - d_mean * inv_thresh).max(0.0);
            let weight = t * t.sqrt() * strength; // t^1.5

            image[[y, x, 0]] = (dr + cr * weight).max(0.0);
            image[[y, x, 1]] = (dg + cg * weight).max(0.0);
            image[[y, x, 2]] = (db + cb * weight).max(0.0);
        }
    }
}


/// Toe/shoulder shaping applied in **output space** (after the RA-4/FilmPrint curve),
/// operating on u16 values normalized to [0, 1].
///
/// This is the effective version: because the RA-4 S-curve compresses all high-density
/// values to near-white before any density-domain offset is visible, shaping must happen
/// after the curve where every increment corresponds to a visible brightness change.
fn apply_toe_shoulder_u16(image: &mut Array3<u16>, toe_strength: f32, shoulder_strength: f32) {
    let toe = toe_strength.clamp(-1.0, 1.0);
    let shoulder = shoulder_strength.clamp(-1.0, 1.0);
    if toe.abs() < 1e-6 && shoulder.abs() < 1e-6 {
        return;
    }
    const TOE_SCALE: f32 = 0.60;
    const SHOULDER_SCALE: f32 = 0.90;
    const MID: f32 = 0.5;
    let (h, w, c) = image.dim();
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let v = image[[y, x, ch]] as f32 / 65535.0;
                let toe_mask = 1.0 - smoothstep(0.07, 0.60, v);
                let shoulder_mask = smoothstep(0.45, 0.95, v);
                let toe_offset = toe * toe_mask * (MID - v) * TOE_SCALE;
                let shoulder_offset = shoulder * shoulder_mask * (MID - v) * SHOULDER_SCALE;
                let v_new = (v + toe_offset + shoulder_offset).clamp(0.0, 1.0);
                image[[y, x, ch]] = (v_new * 65535.0).round() as u16;
            }
        }
    }
}

/// Gaussian-masked zone density adjustments: apply shadow and highlight offsets
/// that only affect their respective tonal zone, leaving midtones untouched.
///
/// Operates in density space (higher D = brighter output through RA-4).
/// `shadows` > 0 adds density to the shadow zone (brightens shadows).
/// `highlights` < 0 subtracts density from the highlight zone (darkens highlights).
///
/// Each pixel's mean density determines its zone membership via a Gaussian mask.
/// The shadow mask is centered on low-density values, the highlight mask on high.
fn apply_zone_density_adjustments(
    image: &mut Array3<f32>,
    shadows: f32,
    highlights: f32,
    color_s: [f32; 3],
    color_m: [f32; 3],
    color_h: [f32; 3],
) {
    let all_zero = shadows.abs() < 1e-6
        && highlights.abs() < 1e-6
        && color_s.iter().chain(color_m.iter()).chain(color_h.iter()).all(|v| v.abs() < 1e-6);
    if all_zero {
        return;
    }
    let (h, w, _) = image.dim();

    // Shadow zone: Gaussian centered at D=0.4, σ²=0.25
    const S_CENTER: f32 = 0.4;
    const S_INV_2S2: f32 = 1.0 / 0.25;
    // Midtone zone: Gaussian centered at D=1.3, σ²=0.20
    const M_CENTER: f32 = 1.3;
    const M_INV_2S2: f32 = 1.0 / 0.20;
    // Highlight zone: Gaussian centered at D=2.2, σ²=0.50
    const H_CENTER: f32 = 2.2;
    const H_INV_2S2: f32 = 1.0 / 0.50;
    const SCALE: f32 = 2.0;

    let s_global = shadows * SCALE;
    let h_global = highlights * SCALE;

    for y in 0..h {
        for x in 0..w {
            let dr = image[[y, x, 0]];
            let dg = image[[y, x, 1]];
            let db = image[[y, x, 2]];
            let d_mean = (dr + dg + db) * (1.0 / 3.0);

            let s_diff = d_mean - S_CENTER;
            let s_mask = (-s_diff * s_diff * S_INV_2S2).exp();

            let m_diff = d_mean - M_CENTER;
            let m_mask = (-m_diff * m_diff * M_INV_2S2).exp();

            let h_diff = d_mean - H_CENTER;
            let h_mask = (-h_diff * h_diff * H_INV_2S2).exp();

            // Global (luminance) offset from existing shadows/highlights sliders.
            let global_offset = s_global * s_mask + h_global * h_mask;

            // Per-channel color balance offsets.
            // Negated because the pipeline is negative-film density: adding to a channel's
            // density increases that channel's RA-4 output, but the dye layers are
            // complementary — the perceived hue shift is opposite to the density direction.
            // Negating makes +R visually warmer (more red in print), +B visually cooler
            // (more blue), etc., matching the R–C / G–M / B–Y label convention.
            let color_scale = -SCALE;
            let offset_r = global_offset
                + (color_s[0] * s_mask + color_m[0] * m_mask + color_h[0] * h_mask) * color_scale;
            let offset_g = global_offset
                + (color_s[1] * s_mask + color_m[1] * m_mask + color_h[1] * h_mask) * color_scale;
            let offset_b = global_offset
                + (color_s[2] * s_mask + color_m[2] * m_mask + color_h[2] * h_mask) * color_scale;

            image[[y, x, 0]] = (dr + offset_r).max(0.0);
            image[[y, x, 1]] = (dg + offset_g).max(0.0);
            image[[y, x, 2]] = (db + offset_b).max(0.0);
        }
    }
}

/// Build `FilmPrintParams` from `PipelineOptions`.
fn build_film_print_params(opts: &PipelineOptions) -> curve::FilmPrintParams {
    curve::FilmPrintParams {
        base: curve::PrintCurveParams {
            offset: opts.curve_offset,
            gamma: opts.curve_gamma,
            pivot: opts.curve_pivot,
        },
        offset_rgb: [opts.fp_offset_r, opts.fp_offset_g, opts.fp_offset_b],
        gamma_rgb: [opts.fp_gamma_r, opts.fp_gamma_g, opts.fp_gamma_b],
        white_point: opts.curve_white,
        color_bleed: opts.fp_color_bleed,
        vibrance: opts.fp_vibrance,
    }
}

/// Smooth hermite interpolation: returns 0 for x <= edge0, 1 for x >= edge1.
#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Post-curve highlight warmth: adds a golden/warm tint to neutral highlights
/// while leaving already-saturated pixels untouched (chroma-aware).
///
/// Works on u16 RA-4 output (0–65535). Only applies when warmth != 0.
///
/// Noritsu/Frontier scanners shift neutral highlights toward golden/cream
/// but leave punchy saturated colors (blue sky, red tones) alone.
fn apply_highlight_warmth_u16(image: &mut Array3<u16>, warmth: f32) {
    if warmth.abs() < 1e-6 {
        return;
    }
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    let scale = 1.0 / 65535.0_f32;

    for y in 0..h {
        for x in 0..w {
            let mut r = image[[y, x, 0]] as f32 * scale;
            let mut g = image[[y, x, 1]] as f32 * scale;
            let mut b = image[[y, x, 2]] as f32 * scale;

            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let chroma = r.max(g).max(b) - r.min(g).min(b);

            // Ramp: full effect in bright highlights, fades toward midtones.
            let highlight_ramp = smoothstep(0.35, 0.85, luma);
            // Neutrality gate: full effect on neutral tones, zero on saturated.
            let neutrality = 1.0 - smoothstep(0.04, 0.18, chroma);

            let strength = highlight_ramp * neutrality * warmth;

            r = (r + strength * 0.035).clamp(0.0, 1.0);
            g = (g + strength * 0.015).clamp(0.0, 1.0);
            b = (b - strength * 0.055).clamp(0.0, 1.0);

            // Extra safety: in very bright highlights, limit extreme chroma
            // so clipped channels do not produce colored speckles (e.g. pure blue)
            // on otherwise neutral speculars.
            let luma2 = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let chroma2 = r.max(g).max(b) - r.min(g).min(b);
            if luma2 > 0.96 && chroma2 > 0.10 {
                let t = smoothstep(0.96, 1.0, luma2);
                let max_chroma = 0.10;
                let reduce = ((chroma2 - max_chroma) / chroma2).clamp(0.0, 1.0) * t;
                // Pull channels toward neutral luma by `reduce` fraction.
                r = r + (luma2 - r) * reduce;
                g = g + (luma2 - g) * reduce;
                b = b + (luma2 - b) * reduce;
            }

            image[[y, x, 0]] = (r * 65535.0).round() as u16;
            image[[y, x, 1]] = (g * 65535.0).round() as u16;
            image[[y, x, 2]] = (b * 65535.0).round() as u16;
        }
    }
}

/// Scalar soft-knee mapping in [0, 1], inspired by film-like highlight roll-off.
/// `s` is the knee start: below `s` the curve is identity, above it rolls toward 1.0.
#[inline]
fn soft_knee_scalar(x: f32, s: f32) -> f32 {
    let s = s.clamp(0.0, 0.9999);
    if x <= s {
        x
    } else {
        let one_minus_s = 1.0 - s;
        let t = -(x - s) / one_minus_s;
        s + (1.0 - t.exp()) * one_minus_s
    }
}

/// Apply a post-curve soft knee to u16 RA-4 output.
fn apply_soft_knee_u16(image: &mut Array3<u16>, soft_clip: f32) {
    if !(0.0..0.999).contains(&soft_clip) {
        return;
    }
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    let inv = 1.0 / 65535.0_f32;
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let v = image[[y, x, ch]] as f32 * inv;
                let v_knee = soft_knee_scalar(v, soft_clip);
                image[[y, x, ch]] = (v_knee.clamp(0.0, 1.0) * 65535.0).round() as u16;
            }
        }
    }
}

/// Same as `apply_highlight_warmth_u16` but operates on normalized f32 [0, 1] RGB.
fn apply_highlight_warmth_f32(image: &mut Array3<f32>, warmth: f32) {
    if warmth.abs() < 1e-6 {
        return;
    }
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);

    for y in 0..h {
        for x in 0..w {
            let mut r = image[[y, x, 0]].clamp(0.0, 1.0);
            let mut g = image[[y, x, 1]].clamp(0.0, 1.0);
            let mut b = image[[y, x, 2]].clamp(0.0, 1.0);

            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let chroma = r.max(g).max(b) - r.min(g).min(b);

            let highlight_ramp = smoothstep(0.35, 0.85, luma);
            let neutrality = 1.0 - smoothstep(0.04, 0.18, chroma);
            let strength = highlight_ramp * neutrality * warmth;

            r = (r + strength * 0.035).clamp(0.0, 1.0);
            g = (g + strength * 0.015).clamp(0.0, 1.0);
            b = (b - strength * 0.055).clamp(0.0, 1.0);

            let luma2 = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let chroma2 = r.max(g).max(b) - r.min(g).min(b);
            if luma2 > 0.96 && chroma2 > 0.10 {
                let t = smoothstep(0.96, 1.0, luma2);
                let max_chroma = 0.10;
                let reduce = ((chroma2 - max_chroma) / chroma2).clamp(0.0, 1.0) * t;
                r = r + (luma2 - r) * reduce;
                g = g + (luma2 - g) * reduce;
                b = b + (luma2 - b) * reduce;
            }

            image[[y, x, 0]] = r;
            image[[y, x, 1]] = g;
            image[[y, x, 2]] = b;
        }
    }
}

/// Apply a post-curve soft knee to display-space f32 RGB in [0, 1].
fn apply_soft_knee_f32(image: &mut Array3<f32>, soft_clip: f32) {
    if !(0.0..0.999).contains(&soft_clip) {
        return;
    }
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let v = image[[y, x, ch]].clamp(0.0, 1.0);
                image[[y, x, ch]] = soft_knee_scalar(v, soft_clip);
            }
        }
    }
}

/// Apply a display-space 3D LUT to an image already in [0, 1].
fn apply_output_cube_rgb(image: &mut Array3<f32>, lut: &lut3d::Lut3d) {
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    for y in 0..h {
        for x in 0..w {
            let r = image[[y, x, 0]].clamp(0.0, 1.0);
            let g = image[[y, x, 1]].clamp(0.0, 1.0);
            let b = image[[y, x, 2]].clamp(0.0, 1.0);
            let [or, og, ob] = lut.sample_normalized(r, g, b);
            image[[y, x, 0]] = or;
            image[[y, x, 1]] = og;
            image[[y, x, 2]] = ob;
        }
    }
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
            let ff = load_flat_field_map(flat_path)?;
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
                apply_flat_field_division(&mut image, flat);
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
                let stats = wb_channel_stats(&image, options);
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

            // Step 4.5: Shadow cast correction (auto-neutralize shadow color cast).
            if options.shadow_cast_strength > 0.0 {
                let cast = analyze_shadow_cast(&image, 0.8);
                apply_shadow_cast_correction(&mut image, cast, options.shadow_cast_strength, 0.8);
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
            limit_highlight_density_spread(&mut image);
        }

        // Step 5.5: Density-domain saturation boost (before RA-4 curve).
        if options.debug_pipeline_step >= 5 {
            apply_density_saturation(&mut image, options.saturation);
            apply_zone_density_adjustments(
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
                        apply_density_levels(
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
                    apply_toe_shoulder_u16(&mut image_u16, options.toe_strength, options.shoulder_strength);
                    apply_soft_knee_u16(&mut image_u16, options.soft_clip);
                    if options.apply_lab {
                        apply_lab_separation_u16(&mut image_u16, options.lab_separation);
                    }
                    apply_highlight_warmth_u16(&mut image_u16, options.highlight_warmth);
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
                    let fp_params = build_film_print_params(options);
                    let mut leveled = image.clone();
                    let levels_active = options.lut_in_black != 0.0
                        || options.lut_in_white != 1.0
                        || (options.lut_in_mid - 1.0).abs() > 1e-6;
                    if levels_active {
                        apply_density_levels(
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
                    apply_toe_shoulder_u16(&mut image_u16, options.toe_strength, options.shoulder_strength);
                    apply_soft_knee_u16(&mut image_u16, options.soft_clip);
                    if options.apply_lab {
                        apply_lab_separation_u16(&mut image_u16, options.lab_separation);
                    }
                    apply_highlight_warmth_u16(&mut image_u16, options.highlight_warmth);
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
                            density_to_rec709_leveled(
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
                            apply_density_levels(
                                &mut display,
                                d_max,
                                options.lut_in_black,
                                options.lut_in_white,
                                options.lut_in_mid,
                            );
                        }
                    }
                    if let Some(ref lut) = output_lut_cube {
                        apply_output_cube_rgb(&mut display, lut);
                    }
                    if options.apply_lab {
                        apply_lab_separation_f32(&mut display, options.lab_separation);
                    }
                    apply_soft_knee_f32(&mut display, options.soft_clip);
                    apply_highlight_warmth_f32(&mut display, options.highlight_warmth);

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

/// Channel stats source for Auto WB.
/// When crop is enabled, evaluate statistics inside the crop only.
fn wb_channel_stats(image: &Array3<f32>, options: &PipelineOptions) -> [(f32, f32, f32); 3] {
    if options.apply_crop {
        if let Some(rect) = options.crop_rect {
            let (h, w, _) = image.dim();
            let (x, y, rw, rh) = scale_dmin_rect(
                rect,
                options.crop_rect_reference_size,
                w as u32,
                h as u32,
            );
            let cropped = crop_array3(image, x, y, rw, rh);
            return channel_stats(&cropped);
        }
    }
    channel_stats(image)
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
        let resized = imageops::resize(&img, new_w, new_h, FilterType::CatmullRom);
        let out = resized.into_raw();
        return Ok((true_src_w, true_src_h, new_w, new_h, out, dbg));
    }

    // Step 3: D-min / flat-field.
    if options.debug_pipeline_step >= 3 && options.dmin_mode != DminMode::Off {
        if let Some(ref flat_path) = options.flat_field_path {
            let flat_map = load_flat_field_map(flat_path)?;
            apply_flat_field_division(&mut image, &flat_map);
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
                            let _ = write!(dbg, "{}", fmt_stats("  D-min sample region (before):", &channel_stats(&region)));
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
            let _ = write!(dbg, "{}", fmt_stats("Step 3 (after D-min, clamped [0,1]):", &channel_stats(&image)));
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
            let _ = write!(dbg, "{}", fmt_stats("Step 4a (T→D, density):", &channel_stats(&image)));
        }

        // 4b: Auto WB — multiplicative equalization of per-channel density medians.
        //     D *= mean_D / ch_median_D  (equivalent to per-channel gamma correction).
        let (auto_s_r, auto_s_g, auto_s_b) = if options.auto_wb && options.dmin_mode != DminMode::Off {
            let stats = wb_channel_stats(&image, options);
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
            let (tr, tg, tb) = temp_k_to_wb_gains(k);
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
            let cast = analyze_shadow_cast(&image, 0.8);
            apply_shadow_cast_correction(&mut image, cast, options.shadow_cast_strength, 0.8);
            let _ = writeln!(dbg, "Shadow cast correction: vec=({:+.4}, {:+.4}, {:+.4}) strength={:.2}",
                cast.0, cast.1, cast.2, options.shadow_cast_strength);
        }

        if options.verbose_debug {
            let _ = write!(dbg, "{}", fmt_stats("Step 4 (after WB + film γ + shadow cast):", &channel_stats(&image)));
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
        limit_highlight_density_spread(&mut image);
        if options.verbose_debug {
            let _ = write!(dbg, "{}", fmt_stats("Step 5 (after density matrix):", &channel_stats(&image)));
        }
    } else {
        let _ = writeln!(dbg, "Step 5: SKIPPED (pipeline_step < 5)");
    }
    let _ = writeln!(dbg);

    // Step 5.5: Density-domain saturation boost (before RA-4 curve).
    if options.debug_pipeline_step >= 5 {
        apply_density_saturation(&mut image, options.saturation);
        apply_zone_density_adjustments(
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
            let _ = write!(dbg, "{}", fmt_stats("  after saturation+zones:", &channel_stats(&image)));
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
                    apply_density_levels(
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
                        write!(dbg, "{}", fmt_stats("  pre-curve density:", &channel_stats(&leveled)));
                }
                let mut u16_img =
                    curve::apply_ra4_from_density(&leveled, params, 4.0, options.curve_white);
                apply_toe_shoulder_u16(&mut u16_img, options.toe_strength, options.shoulder_strength);
                if options.apply_lab {
                    apply_lab_separation_u16(&mut u16_img, options.lab_separation);
                }
                apply_highlight_warmth_u16(&mut u16_img, options.highlight_warmth);
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
                let fp_params = build_film_print_params(options);
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
                    apply_density_levels(
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
                apply_toe_shoulder_u16(&mut u16_img, options.toe_strength, options.shoulder_strength);
                if options.apply_lab {
                    apply_lab_separation_u16(&mut u16_img, options.lab_separation);
                }
                apply_highlight_warmth_u16(&mut u16_img, options.highlight_warmth);
                u16_img
                    .iter()
                    .map(|v| ((*v as u32) >> 8).min(255) as u8)
                    .collect()
            }
            OutputStage::None => {
                // No-curve positive: density → linear brightness.
                let _ = writeln!(dbg, "Step 6: linear density inversion (no curve)");
                if options.verbose_debug {
                    let _ = write!(dbg, "{}", fmt_stats("  density:", &channel_stats(&image)));
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
                        density_to_rec709_leveled(
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
                        apply_density_levels(
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
                    apply_output_cube_rgb(&mut display, &output_lut);
                }
                if options.apply_lab {
                    apply_lab_separation_f32(&mut display, options.lab_separation);
                }
                apply_soft_knee_f32(&mut display, options.soft_clip);
                apply_highlight_warmth_f32(&mut display, options.highlight_warmth);

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
