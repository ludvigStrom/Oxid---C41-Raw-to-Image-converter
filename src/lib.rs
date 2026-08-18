//! C-41 RAW pipeline library. Used by both CLI and GUI.

use std::collections::{HashMap, VecDeque};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::Context;
use image::{
    imageops::{self, FilterType},
    Rgb, Rgb32FImage, RgbImage,
};
use ndarray::{self, Array3};

pub mod aces;
pub mod auto;
pub mod auto_crop;
pub mod bujack;
pub mod calibration;
pub mod color;
pub mod color_space;
pub mod curve;
pub mod demosaic;
pub mod density_ops;
pub mod dmin;
pub mod dust;
pub(crate) mod dust_grain;
pub(crate) mod dust_pm;
pub(crate) mod dust_wfc;
pub mod exr_export;
pub mod flat_field;
pub mod inversion;
pub mod lut3d;
pub mod options;
pub mod pipeline;
pub mod pipeline_cache;
pub mod png_reader;
pub mod post_curve;
pub mod preset;
pub mod project;

#[cfg(feature = "gpu")]
pub mod gpu;
pub mod raw_reader;
pub mod sensor;
pub mod stats;
pub mod tiff_export;
pub mod undo;

pub use auto::{auto_tune, AutoTuneResult, AUTO_PROXY_MAX_SIDE};
pub use auto_crop::{
    detect_crop, AutoCropResult, CropConfidence, FilmFormat, SurroundClass, CROP_PROXY_MAX_SIDE,
};
pub use dust::{
    apply_dust_removal, apply_dust_removal_with, crop_mask_uv, hash_dust, hash_strokes,
    mask_at_image_size, rasterize_strokes, rasterize_strokes_uv, stamp_disc, DustHealParams,
    DustInfill, DustMask, DustStroke, DustTool,
    ProjectDust,
};
pub use flat_field::{blur_flat_field, load_flat_field_linear};
pub use options::{
    reset_wb_for_picker, sync_wb_flags_from_mode, DminMode, OutputLutEncoding, OutputStage,
    PipelineOptions, Rect, WbMode,
};
pub use pipeline_cache::{
    cached_start_step, hash_after_load, hash_after_step3, hash_after_step4, hash_after_step5,
    PreviewStepCache,
};
pub use preset::{load_develop_preset, save_develop_preset, DevelopPreset, PRESET_VERSION};
pub use project::{
    load_project, save_project, LoadedProject, ProjectExportFormat, ProjectFile, ProjectImage,
    PROJECT_EXTENSION, PROJECT_EXTENSION_LEGACY, PROJECT_VERSION,
};
pub use sensor::{
    compute_dmin_from_sensor, compute_preview_scene_stats, crop_sensor_for_oriented_rect,
    load_sensor_from_path, oriented_sensor_size, preview_scene_stats_key, CachedSensor,
    PreviewSceneStats, SensorTileCrop,
};
pub use tiff_export::TiffFormat;
pub use undo::{UndoManager, UNDO_LIMIT};

use crate::demosaic::{BayerPattern, CfaPattern};

fn apply_optional_dust(image: &mut Array3<f32>, options: &PipelineOptions) {
    if options.dust_mask_hash == 0 {
        return;
    }
    let (h, w, _) = image.dim();
    if !options.dust_strokes.is_empty() {
        let reference = options
            .dust_reference_size
            .unwrap_or((w as u32, h as u32));
        let mask = dust::mask_at_image_size(
            &options.dust_strokes,
            reference,
            options.dust_uv,
            w as u32,
            h as u32,
        );
        dust::apply_dust_removal_with(image, &mask, options.dust_heal);
        return;
    }
    if let Some(mask) = options.dust_mask.as_ref() {
        dust::apply_dust_removal_with(image, mask, options.dust_heal);
    }
}

#[cfg(feature = "gpu")]
fn apply_optional_dust_gpu(
    image: &mut Array3<f32>,
    options: &PipelineOptions,
    gpu: &crate::gpu::unified::GpuPipeline,
) {
    if options.dust_mask_hash == 0 {
        return;
    }
    let (h, w, _) = image.dim();
    let owned = if !options.dust_strokes.is_empty() {
        let reference = options
            .dust_reference_size
            .unwrap_or((w as u32, h as u32));
        Some(dust::mask_at_image_size(
            &options.dust_strokes,
            reference,
            options.dust_uv,
            w as u32,
            h as u32,
        ))
    } else {
        None
    };
    let Some(mask) = owned.as_ref().or(options.dust_mask.as_deref()) else {
        return;
    };
    if options.dust_heal.infill == dust::DustInfill::WaveFunction
        && gpu.dust_wfc.run(image, mask, options.dust_heal).is_ok()
    {
        return;
    }
    dust::apply_dust_removal_with(image, mask, options.dust_heal);
}

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
pub(crate) fn crop_array3(
    image: &Array3<f32>,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Array3<f32> {
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
    downsample_cfa_box(bayer, max_width, 6)
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

/// Flip image horizontally (mirror left–right). Returns a new Array3.
pub(crate) fn flip_array3_horizontal(image: &Array3<f32>) -> Array3<f32> {
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    let mut out = Array3::<f32>::zeros((h, w, 3));
    for y in 0..h {
        for x in 0..w {
            let new_x = w - 1 - x;
            for ch in 0..3 {
                out[(y, new_x, ch)] = image[(y, x, ch)];
            }
        }
    }
    out
}

/// Flip image vertically (mirror top–bottom). Returns a new Array3.
pub(crate) fn flip_array3_vertical(image: &Array3<f32>) -> Array3<f32> {
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    let mut out = Array3::<f32>::zeros((h, w, 3));
    for y in 0..h {
        for x in 0..w {
            let new_y = h - 1 - y;
            for ch in 0..3 {
                out[(new_y, x, ch)] = image[(y, x, ch)];
            }
        }
    }
    out
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

/// Box-filter a CFA mosaic down to `max_width`, keeping the `period`×`period`
/// pattern intact (2 for Bayer, 6 for X-Trans).
///
/// Same-phase photosites in each `step`×`step` block of tiles are averaged so
/// the reduced preview is not a nearest-neighbor skip (which aliases badly
/// next to 1:1 tiles).
fn downsample_cfa_box(bayer: &Array3<f32>, max_width: u32, period: usize) -> Array3<f32> {
    let (h, w, c) = bayer.dim();
    assert_eq!(c, 1, "Expected single-channel CFA for preview");
    assert!(period > 0);

    if w as u32 <= max_width {
        return bayer.clone();
    }

    let n_super_w = w / period;
    let n_super_h = h / period;
    let max_super_w = (max_width as usize / period).max(1);
    let step = ((n_super_w as f32 / max_super_w as f32).ceil() as usize).max(1);
    let out_super_w = n_super_w / step;
    let out_super_h = n_super_h / step;
    let out_w = out_super_w * period;
    let out_h = out_super_h * period;
    let mut out = Array3::<f32>::zeros((out_h, out_w, 1));

    for sy in 0..out_super_h {
        for sx in 0..out_super_w {
            for dy in 0..period {
                for dx in 0..period {
                    let mut acc = 0.0f32;
                    let mut n = 0.0f32;
                    for iy in 0..step {
                        for ix in 0..step {
                            let y = sy * step * period + iy * period + dy;
                            let x = sx * step * period + ix * period + dx;
                            if y < h && x < w {
                                acc += bayer[(y, x, 0)];
                                n += 1.0;
                            }
                        }
                    }
                    out[(sy * period + dy, sx * period + dx, 0)] =
                        if n > 0.0 { acc / n } else { 0.0 };
                }
            }
        }
    }
    out
}

/// Downsample a single-channel Bayer array for preview, preserving the 2×2
/// RGGB pattern so demosaic can produce real color.
fn downsample_bayer_for_preview(bayer: &Array3<f32>, max_width: u32) -> Array3<f32> {
    downsample_cfa_box(bayer, max_width, 2)
}

/// Pack each 2×2 Bayer tile into one RGB pixel (native R, mean G, native B).
///
/// Used for the GUI backdrop instead of CFA-bin-then-demosaic, which invents
/// chroma that C-41 inversion turns into a blue/cyan cast.
fn pack_bayer_2x2_to_rgb(bayer: &Array3<f32>, pattern: BayerPattern) -> Array3<f32> {
    let (h, w, c) = bayer.dim();
    assert_eq!(c, 1, "Expected single-channel CFA");
    let out_h = h / 2;
    let out_w = w / 2;
    let mut out = Array3::<f32>::zeros((out_h, out_w, 3));
    for y in 0..out_h {
        for x in 0..out_w {
            let y0 = y * 2;
            let x0 = x * 2;
            let a = bayer[(y0, x0, 0)];
            let b = bayer[(y0, x0 + 1, 0)];
            let c = bayer[(y0 + 1, x0, 0)];
            let d = bayer[(y0 + 1, x0 + 1, 0)];
            let (r, g, bl) = match pattern {
                BayerPattern::Rggb => (a, 0.5 * (b + c), d),
                BayerPattern::Grbg => (b, 0.5 * (a + d), c),
                BayerPattern::Gbrg => (c, 0.5 * (a + d), b),
                BayerPattern::Bggr => (d, 0.5 * (b + c), a),
            };
            out[(y, x, 0)] = r;
            out[(y, x, 1)] = g;
            out[(y, x, 2)] = bl;
        }
    }
    out
}

/// RAW → working RGB for a preview job. Tiles / full-res demosaic natively.
/// Bayer backdrops pack 2×2 → RGB then downsample (no interpolated chroma).
fn preview_rgb_from_mosaic(
    bayer: &Array3<f32>,
    pattern: CfaPattern,
    max_width: u32,
    max_height: u32,
    simple_debayer: bool,
) -> anyhow::Result<Array3<f32>> {
    let full_res = max_width == u32::MAX && max_height == u32::MAX;
    if !full_res {
        if let CfaPattern::Bayer(p) = pattern {
            let packed = pack_bayer_2x2_to_rgb(bayer, p);
            return Ok(downsample_rgb_for_preview(&packed, max_width, max_height));
        }
    }
    let working = if full_res {
        bayer.clone()
    } else {
        downsample_raw_for_preview(bayer, pattern, preview_mosaic_working_width(max_width))
    };
    let mut img = if simple_debayer {
        demosaic::demosaic_bilinear(&working, pattern)?
    } else {
        demosaic::demosaic_quality(&working, pattern)?
    };
    img.mapv_inplace(|v| v.max(0.0));
    Ok(finish_preview_rgb(img, max_width, max_height))
}

/// Mosaic working width. Full-res / 1:1 tiles pass `u32::MAX`.
fn preview_mosaic_working_width(max_width: u32) -> u32 {
    max_width
}

fn finish_preview_rgb(img: Array3<f32>, max_width: u32, max_height: u32) -> Array3<f32> {
    if max_width == u32::MAX && max_height == u32::MAX {
        img
    } else {
        downsample_rgb_for_preview(&img, max_width, max_height)
    }
}

/// Downsample an RGB image for preview to fit within `max_width`×`max_height`,
/// preserving aspect ratio. Used for non-RAW (PNG) previews so the full C-41
/// pipeline only runs on a smaller working resolution.
fn downsample_rgb_for_preview(image: &Array3<f32>, max_width: u32, max_height: u32) -> Array3<f32> {
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

    let resized = imageops::resize(&img, new_w, new_h, FilterType::Triangle);

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

/// Snapshot of an in-flight export for the GUI progress dialog.
#[derive(Clone, Debug)]
pub struct ExportProgress {
    /// 0-based index of the file most recently started.
    pub current: usize,
    /// Files that have finished (success).
    pub completed: usize,
    pub total: usize,
    pub file_name: String,
    pub stage: String,
    /// Overall 0–1 progress across the batch.
    pub fraction: f32,
}

/// Cooperative cancel + progress for [`process_files_with_control`].
pub struct ExportControl {
    cancel: AtomicBool,
    completed: AtomicUsize,
    progress: Mutex<ExportProgress>,
}

impl ExportControl {
    pub fn new(total: usize) -> Self {
        Self {
            cancel: AtomicBool::new(false),
            completed: AtomicUsize::new(0),
            progress: Mutex::new(ExportProgress {
                current: 0,
                completed: 0,
                total,
                file_name: String::new(),
                stage: "Starting".to_string(),
                fraction: 0.0,
            }),
        }
    }

    pub fn request_cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    pub fn completed(&self) -> usize {
        self.completed.load(Ordering::Relaxed)
    }

    pub fn snapshot(&self) -> ExportProgress {
        self.progress
            .lock()
            .map(|g| g.clone())
            .unwrap_or_else(|e| e.into_inner().clone())
    }

    /// Mark which file in a batch is being processed. Call before each
    /// [`process_files_with_control`] when options differ per image.
    pub fn begin_file(&self, index: usize, total: usize, file_name: impl Into<String>) {
        let done = self.completed.load(Ordering::Relaxed);
        if let Ok(mut p) = self.progress.lock() {
            p.current = index;
            p.completed = done;
            p.total = total;
            p.file_name = file_name.into();
            p.stage = "Starting".to_string();
            p.fraction = done as f32 / total.max(1) as f32;
        }
    }

    pub fn mark_completed(&self) {
        let done = self.completed.fetch_add(1, Ordering::Relaxed) + 1;
        if let Ok(mut p) = self.progress.lock() {
            p.completed = done;
            let n = p.total.max(1) as f32;
            p.fraction = done as f32 / n;
        }
    }

    fn set_stage(&self, stage: &str, within_file: f32) {
        if let Ok(mut p) = self.progress.lock() {
            p.stage = stage.to_string();
            let done = self.completed.load(Ordering::Relaxed);
            p.completed = done;
            let n = p.total.max(1) as f32;
            p.fraction = (done as f32 + within_file.clamp(0.0, 1.0)) / n;
        }
    }
}

/// Returned from [`process_files_with_control`] when the user cancels.
#[derive(Debug)]
pub struct ExportCancelled;

impl std::fmt::Display for ExportCancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Export cancelled")
    }
}

impl std::error::Error for ExportCancelled {}

fn export_tick(
    control: Option<&ExportControl>,
    stage: &str,
    within_file: f32,
) -> anyhow::Result<()> {
    if let Some(c) = control {
        if c.is_cancelled() {
            return Err(anyhow::Error::new(ExportCancelled));
        }
        c.set_stage(stage, within_file);
    }
    Ok(())
}

/// One file in a batch export. Options may differ per image (GUI); the CLI
/// uses the same options for every path.
#[derive(Clone)]
pub struct ExportJobSpec {
    pub path: PathBuf,
    pub options: PipelineOptions,
    /// Native width × height when known (GUI preview). Used for the RAM cap.
    pub source_size: Option<(u32, u32)>,
}

/// Shared LUTs / flat-field, loaded once per unique path for a batch.
struct ExportAssets {
    lut3d: HashMap<PathBuf, Arc<lut3d::Lut3d>>,
    output_lut: HashMap<PathBuf, Arc<lut3d::Lut3d>>,
    flat_field: HashMap<PathBuf, Arc<Array3<f32>>>,
}

impl ExportAssets {
    fn preload(jobs: &[ExportJobSpec]) -> anyhow::Result<Self> {
        let mut lut3d_maps = HashMap::new();
        let mut output_lut_maps = HashMap::new();
        let mut flat_field_maps = HashMap::new();
        for job in jobs {
            if let Some(p) = &job.options.lut3d_path {
                if !lut3d_maps.contains_key(p) {
                    if let Ok(lut) = lut3d::read_cube(p) {
                        lut3d_maps.insert(p.clone(), Arc::new(lut));
                    }
                }
            }
            if let Some(p) = &job.options.output_lut_cube {
                if !output_lut_maps.contains_key(p) {
                    if let Ok(lut) = lut3d::read_cube(p) {
                        output_lut_maps.insert(p.clone(), Arc::new(lut));
                    }
                }
            }
            if let Some(p) = &job.options.flat_field_path {
                if !flat_field_maps.contains_key(p) {
                    let ff = crate::flat_field::load_flat_field_map(p)?;
                    flat_field_maps.insert(p.clone(), Arc::new(ff));
                }
            }
        }
        Ok(Self {
            lut3d: lut3d_maps,
            output_lut: output_lut_maps,
            flat_field: flat_field_maps,
        })
    }

    fn lut3d_for(&self, options: &PipelineOptions) -> Option<&lut3d::Lut3d> {
        options
            .lut3d_path
            .as_ref()
            .and_then(|p| self.lut3d.get(p))
            .map(|a| a.as_ref())
    }

    fn output_lut_for(&self, options: &PipelineOptions) -> Option<&lut3d::Lut3d> {
        options
            .output_lut_cube
            .as_ref()
            .and_then(|p| self.output_lut.get(p))
            .map(|a| a.as_ref())
    }

    fn flat_field_for(&self, options: &PipelineOptions) -> Option<&Array3<f32>> {
        options
            .flat_field_path
            .as_ref()
            .and_then(|p| self.flat_field.get(p))
            .map(|a| a.as_ref())
    }
}

/// Fallback when image size is unknown (Sony a7R II class).
const DEFAULT_EXPORT_PIXELS: u64 = 42_000_000;
const FALLBACK_EXPORT_BUDGET_BYTES: u64 = 4 * 1024 * 1024 * 1024;

fn system_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let mut size: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let name = b"hw.memsize\0";
        let ret = unsafe {
            sysctlbyname(
                name.as_ptr().cast(),
                (&mut size as *mut u64).cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        return (ret == 0).then_some(size);
    }
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb.saturating_mul(1024));
            }
        }
        return None;
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

#[cfg(target_os = "macos")]
extern "C" {
    fn sysctlbyname(
        name: *const i8,
        oldp: *mut std::ffi::c_void,
        oldlenp: *mut usize,
        newp: *mut std::ffi::c_void,
        newlen: usize,
    ) -> i32;
}

fn export_working_set_budget_bytes() -> u64 {
    system_memory_bytes()
        .map(|total| total / 2)
        .unwrap_or(FALLBACK_EXPORT_BUDGET_BYTES)
}

fn job_pixel_count(job: &ExportJobSpec) -> u64 {
    job.source_size
        .map(|(w, h)| u64::from(w).saturating_mul(u64::from(h)))
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_EXPORT_PIXELS)
}

fn estimate_export_peak_bytes(pixels: u64, options: &PipelineOptions) -> u64 {
    // Working RGB f32 (12) + step-6 u16 (6) + TIFF flatten (6). Levels run in place.
    let mut bpp: u64 = 24;
    if options.export_aces_exr || options.write_aces2065_only {
        bpp += 12;
    }
    if options.bujack_enabled && options.bujack_strength > 0.0 {
        bpp += 12;
    }
    pixels.saturating_mul(bpp)
}

fn export_concurrency(budget_bytes: u64, peak_bytes: u64) -> usize {
    if peak_bytes == 0 {
        return 1;
    }
    let fit = (budget_bytes / peak_bytes).max(1).min(2);
    fit as usize
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
    process_files_with_control(paths, output_dir, options, None)
}

/// Same as [`process_files`], with optional cancel / progress for the GUI.
pub fn process_files_with_control(
    paths: &[PathBuf],
    output_dir: &Path,
    options: &PipelineOptions,
    control: Option<&ExportControl>,
) -> anyhow::Result<()> {
    let jobs: Vec<ExportJobSpec> = paths
        .iter()
        .map(|path| ExportJobSpec {
            path: path.clone(),
            options: options.clone(),
            source_size: None,
        })
        .collect();
    process_export_jobs(&jobs, output_dir, control)
}

/// Batch export with a hard cap of two images in flight (one if the RAM
/// budget cannot fit two peaks). Intra-image rayon stays on the global pool.
pub fn process_export_jobs(
    jobs: &[ExportJobSpec],
    output_dir: &Path,
    control: Option<&ExportControl>,
) -> anyhow::Result<()> {
    export_tick(control, "Preparing", 0.0)?;
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output directory {}", output_dir.display()))?;
    if jobs.is_empty() {
        return Ok(());
    }

    let assets = Arc::new(ExportAssets::preload(jobs)?);
    let budget = export_working_set_budget_bytes();
    let max_peak = jobs
        .iter()
        .map(|j| estimate_export_peak_bytes(job_pixel_count(j), &j.options))
        .max()
        .unwrap_or(1);
    let workers = export_concurrency(budget, max_peak);
    let total = jobs.len();
    let output_dir = output_dir.to_path_buf();

    if workers <= 1 {
        for (i, job) in jobs.iter().enumerate() {
            export_tick(control, "Starting", 0.0)?;
            if let Some(c) = control {
                let name = job
                    .path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("image");
                c.begin_file(i, total, name);
            }
            process_one_export(&job.path, &output_dir, &job.options, &assets, control)?;
            if let Some(c) = control {
                c.mark_completed();
            }
        }
        return Ok(());
    }

    let queue = Mutex::new((0..total).collect::<VecDeque<usize>>());
    let first_err: Mutex<Option<anyhow::Error>> = Mutex::new(None);

    thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                if control.map(|c| c.is_cancelled()).unwrap_or(false) {
                    return;
                }
                if first_err.lock().map(|g| g.is_some()).unwrap_or(false) {
                    return;
                }
                let Some(i) = queue.lock().ok().and_then(|mut q| q.pop_front()) else {
                    return;
                };
                let job = &jobs[i];
                if let Some(c) = control {
                    let name = job
                        .path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("image");
                    c.begin_file(i, total, name);
                }
                match process_one_export(&job.path, &output_dir, &job.options, &assets, control) {
                    Ok(()) => {
                        if let Some(c) = control {
                            c.mark_completed();
                        }
                    }
                    Err(e) if e.downcast_ref::<ExportCancelled>().is_some() => return,
                    Err(e) => {
                        if let Ok(mut slot) = first_err.lock() {
                            if slot.is_none() {
                                *slot = Some(e);
                            }
                        }
                        return;
                    }
                }
            });
        }
    });

    if control.map(|c| c.is_cancelled()).unwrap_or(false) {
        return Err(anyhow::Error::new(ExportCancelled));
    }
    if let Some(e) = first_err.lock().ok().and_then(|mut g| g.take()) {
        return Err(e);
    }
    Ok(())
}

fn process_one_export(
    path: &Path,
    output_dir: &Path,
    options: &PipelineOptions,
    assets: &ExportAssets,
    control: Option<&ExportControl>,
) -> anyhow::Result<()> {
    export_tick(control, "Loading", 0.05)?;
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    let mut image = match ext.as_str() {
        "arw" | "nef" | "nrw" | "cr2" | "cr3" | "crw" | "dng" | "raf" | "orf" | "rw2" => {
            let (bayer, pattern) = raw_reader::load_raw_as_ndarray(path)?;
            export_tick(control, "Demosaic", 0.15)?;
            let mut img = demosaic::demosaic_quality(&bayer, pattern)?;
            img.mapv_inplace(|v| v.max(0.0));
            img
        }
        "png" | "jpeg" | "jpg" | "tiff" | "tif" => {
            let mut img = png_reader::load_png_as_ndarray(path)?;
            if options.synthetic_negative_input {
                pipeline::apply_synthetic_negative_invert(&mut img);
            }
            img
        }
        _ => return Ok(()),
    };

    export_tick(control, "Geometry", 0.30)?;
    if options.rotation_degrees != 0 {
        image = apply_rotation(&image, options.rotation_degrees);
    }
    if options.flip_horizontal {
        image = flip_array3_horizontal(&image);
    }
    if options.flip_vertical {
        image = flip_array3_vertical(&image);
    }

    color_space::apply_input_idt_to_working_space(&mut image, &options.idt_matrix);

    apply_optional_dust(&mut image, options);

    export_tick(control, "D-min", 0.40)?;
    pipeline::step_3_dmin(&mut image, options, assets.flat_field_for(options))?;
    export_tick(control, "White balance", 0.52)?;
    pipeline::step_4_t_to_d_wb(&mut image, options);
    export_tick(control, "Color", 0.65)?;
    pipeline::step_5_calibration(&mut image, options, assets.lut3d_for(options));

    if options.apply_crop {
        if let Some(rect) = options.crop_rect {
            let (h, w, _) = image.dim();
            let (x, y, rw, rh) =
                scale_dmin_rect(rect, options.crop_rect_reference_size, w as u32, h as u32);
            image = crop_array3(&image, x, y, rw, rh);
        }
    }

    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    let out_path = output_dir.join(format!("{}.tiff", stem));
    let jpg_path = output_dir.join(format!("{}.jpg", stem));
    let exr_path = output_dir.join(format!("{}.exr", stem));
    let aces_exr_path = output_dir.join(format!("{}_aces2065-1.exr", stem));

    let ra4_params = curve::PrintCurveParams {
        offset: options.curve_offset,
        gamma: options.curve_gamma,
        pivot: options.curve_pivot,
    };

    if options.write_aces2065_only || options.export_aces_exr {
        export_tick(control, "ACES", 0.78)?;
        let mut aces2065 = curve::apply_ra4_from_density_f32(&image, ra4_params, 4.0);
        let rec709 = !aces::is_identity(&options.idt_matrix);
        aces::linear_print_to_aces2065_1(&mut aces2065, rec709);
        exr_export::write_exr_aces2065_1(&aces2065, &aces_exr_path)?;
        drop(aces2065);
        if options.write_aces2065_only {
            return Ok(());
        }
    }

    let write_jpeg_this = options.write_jpeg || options.write_jpeg_only;

    export_tick(control, "Rendering", 0.82)?;
    let mut display =
        pipeline::step_6_render_owned(image, options, &ra4_params, assets.output_lut_for(options));
    pipeline::apply_bujack(&mut display, options);

    export_tick(control, "Writing", 0.92)?;
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
                let buf: Vec<u8> = img
                    .iter()
                    .map(|v| color_space::linear_to_srgb_u8(*v as f32 / 65535.0))
                    .collect();
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

    Ok(())
}

#[cfg(test)]
mod export_mem_tests {
    use super::*;

    #[test]
    fn concurrency_caps_at_two_and_falls_back_to_one() {
        let opts = PipelineOptions::default();
        let peak = estimate_export_peak_bytes(24_000_000, &opts);
        assert!(peak > 500_000_000);
        assert_eq!(export_concurrency(4 * 1024 * 1024 * 1024, peak), 2);
        assert_eq!(export_concurrency(peak, peak.saturating_mul(3)), 1);
        assert_eq!(export_concurrency(100, peak), 1);
    }
}

#[cfg(test)]
mod live_preview_tests {
    use super::*;
    use ndarray::Array3;

    fn synthetic_cache(path: &Path, opts: &PipelineOptions, w: u32, h: u32) -> PreviewStepCache {
        let mut img = Array3::<f32>::from_elem((h as usize, w as usize, 3), 0.35);
        img[[0, 0, 0]] = 0.55;
        img[[0, 0, 1]] = 0.40;
        img[[0, 0, 2]] = 0.30;
        let h1 = hash_after_load(path, opts, w, h);
        let h3 = hash_after_step3(path, opts, w, h);
        let h4 = hash_after_step4(path, opts, w, h);
        let h5 = hash_after_step5(path, opts, w, h);
        let mut after4 = img.clone();
        pipeline::step_4_t_to_d_wb(&mut after4, opts);
        let mut after5 = after4.clone();
        pipeline::step_5_calibration(&mut after5, opts, None);
        PreviewStepCache {
            after_load: Some((h1, img.clone(), w, h)),
            after_step3: Some((h3, img)),
            after_step4: Some((h4, after4)),
            after_step5: Some((h5, after5)),
        }
    }

    #[test]
    fn live_apply_starts_at_step6_for_curve_only() {
        let path = Path::new("/live.png");
        let opts = PipelineOptions::default();
        let cache = synthetic_cache(path, &opts, 8, 8);
        let h5 = hash_after_step5(path, &opts, 8, 8);
        let mut live = opts.clone();
        live.curve_offset = 0.22;
        let (iw, ih, w, h, rgb, new_cache) =
            apply_preview_from_cache(path, &live, 8, 8, &cache).expect("live apply");
        assert_eq!((iw, ih, w, h), (8, 8, 8, 8));
        assert_eq!(rgb.len(), 8 * 8 * 3);
        assert_eq!(
            new_cache.after_step5.as_ref().map(|(hash, _)| *hash),
            Some(h5)
        );
        assert_eq!(cached_start_step(path, &live, 8, 8, &cache), 6);
    }

    #[test]
    fn live_apply_returns_none_without_step3_cache() {
        let path = Path::new("/live.png");
        let opts = PipelineOptions::default();
        let empty = PreviewStepCache::default();
        assert!(apply_preview_from_cache(path, &opts, 8, 8, &empty).is_none());
    }
}

#[cfg(test)]
mod pack_bayer_tests {
    use super::*;
    use ndarray::Array3;

    fn mosaic_2x2(a: f32, b: f32, c: f32, d: f32) -> Array3<f32> {
        let mut m = Array3::<f32>::zeros((2, 2, 1));
        m[(0, 0, 0)] = a;
        m[(0, 1, 0)] = b;
        m[(1, 0, 0)] = c;
        m[(1, 1, 0)] = d;
        m
    }

    #[test]
    fn rggb_pack_uses_native_r_mean_g_native_b() {
        let rgb = pack_bayer_2x2_to_rgb(&mosaic_2x2(1.0, 0.4, 0.6, 0.2), BayerPattern::Rggb);
        assert_eq!(rgb.dim(), (1, 1, 3));
        assert!((rgb[(0, 0, 0)] - 1.0).abs() < 1e-6);
        assert!((rgb[(0, 0, 1)] - 0.5).abs() < 1e-6);
        assert!((rgb[(0, 0, 2)] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn bggr_pack_swaps_r_and_b() {
        let rgb = pack_bayer_2x2_to_rgb(&mosaic_2x2(0.2, 0.4, 0.6, 1.0), BayerPattern::Bggr);
        assert!((rgb[(0, 0, 0)] - 1.0).abs() < 1e-6);
        assert!((rgb[(0, 0, 1)] - 0.5).abs() < 1e-6);
        assert!((rgb[(0, 0, 2)] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn preview_mosaic_packs_bayer_instead_of_demosaic() {
        let mut bayer = Array3::<f32>::zeros((4, 4, 1));
        for y in 0..4 {
            for x in 0..4 {
                let rggb = match (y % 2, x % 2) {
                    (0, 0) => 1.0,
                    (1, 1) => 0.25,
                    _ => 0.5,
                };
                bayer[(y, x, 0)] = rggb;
            }
        }
        let rgb =
            preview_rgb_from_mosaic(&bayer, CfaPattern::Bayer(BayerPattern::Rggb), 2, 2, false)
                .expect("pack");
        assert_eq!(rgb.dim(), (2, 2, 3));
        assert!((rgb[(0, 0, 0)] - 1.0).abs() < 1e-5);
        assert!((rgb[(0, 0, 1)] - 0.5).abs() < 1e-5);
        assert!((rgb[(0, 0, 2)] - 0.25).abs() < 1e-5);
    }
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
            preview_rgb_from_mosaic(
                &bayer,
                pattern,
                max_width,
                max_height,
                options.debug_preview_simple_debayer,
            )?
        }
        "png" | "jpeg" | "jpg" | "tiff" | "tif" => {
            let mut img = png_reader::load_png_as_ndarray(path)?;
            if options.synthetic_negative_input {
                pipeline::apply_synthetic_negative_invert(&mut img);
            }
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
    if options.flip_horizontal {
        image = flip_array3_horizontal(&image);
    }
    if options.flip_vertical {
        image = flip_array3_vertical(&image);
    }

    color_space::apply_input_idt_to_working_space(&mut image, &options.idt_matrix);

    apply_optional_dust(&mut image, options);

    // Step 1: load + demosaic + rotate
    if options.verbose_debug {
        let _ = write!(
            dbg,
            "{}",
            stats::fmt_stats("Step 1 (load+demosaic+rot):", &stats::channel_stats(&image))
        );
        let _ = writeln!(dbg);
    }

    // Debug preview mode: show simple demosaic only.
    if options.debug_preview_simple_debayer
        && matches!(
            ext.as_str(),
            "arw" | "nef" | "nrw" | "cr2" | "cr3" | "crw" | "dng" | "raf" | "orf" | "rw2"
        )
    {
        let (orig_h, orig_w, _) = image.dim();
        let orig_w = orig_w as u32;
        let orig_h = orig_h as u32;
        let max_v = image.iter().copied().fold(0.0_f32, f32::max).max(1.0e-6);
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
            let _ = writeln!(
                dbg,
                "D-min mode: flat-field ({})",
                options.flat_field_path.as_ref().unwrap().display()
            );
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
                    let _ = writeln!(
                        dbg,
                        "D-min mode: auto-percentile (buffer={:.2})",
                        options.auto_norm_buffer
                    );
                }
                DminMode::Off => {}
            }
        }
    }
    pipeline::step_3_dmin(&mut image, options, flat_map_preview.as_ref())?;
    if options.debug_pipeline_step >= 3
        && options.dmin_mode != DminMode::Off
        && options.verbose_debug
    {
        let _ = write!(
            dbg,
            "{}",
            stats::fmt_stats(
                "Step 3 (after D-min, clamped [0,1]):",
                &stats::channel_stats(&image)
            )
        );
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
        let _ = write!(
            dbg,
            "{}",
            stats::fmt_stats(
                "Step 4 (after WB + film γ + shadow cast):",
                &stats::channel_stats(&image)
            )
        );
    } else if options.debug_pipeline_step < 4 {
        let _ = writeln!(dbg, "Step 4: SKIPPED (pipeline_step < 4)");
    }
    let _ = writeln!(dbg);

    // Step 5: density matrix / LUT, saturation, zones (shared pipeline step).
    let lut3d_preview = options
        .lut3d_path
        .as_ref()
        .and_then(|p| lut3d::read_cube(p).ok());
    if options.debug_pipeline_step >= 5 {
        let _ = writeln!(
            dbg,
            "Step 5: density matrix [...], lut3d: {}",
            lut3d_preview.is_some()
        );
    }
    pipeline::step_5_calibration(&mut image, options, lut3d_preview.as_ref());
    if options.debug_pipeline_step >= 5 && options.verbose_debug {
        let _ = write!(
            dbg,
            "{}",
            stats::fmt_stats(
                "Step 5 (after density matrix):",
                &stats::channel_stats(&image)
            )
        );
        let _ = writeln!(
            dbg,
            "Step 5.5: saturation={:.2} zones ...",
            options.saturation
        );
        let _ = write!(
            dbg,
            "{}",
            stats::fmt_stats("  after saturation+zones:", &stats::channel_stats(&image))
        );
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
    let output_lut_preview = options
        .output_lut_cube
        .as_ref()
        .and_then(|p| lut3d::read_cube(p).ok());
    let mut display =
        pipeline::step_6_render(&image, options, &ra4_params, output_lut_preview.as_ref());
    pipeline::apply_bujack(&mut display, options);

    if options.debug_pipeline_step >= 6 {
        let _ = writeln!(dbg, "Step 6: {:?} (shared pipeline)", options.output_stage);
        if options.bujack_enabled {
            let _ = writeln!(
                dbg,
                "De-Bujack: kL={:.3} kC={:.3} strength={:.2} radius={:.0} edge={:.2}",
                options.bujack_k_l,
                options.bujack_k_c,
                options.bujack_strength,
                options.bujack_radius,
                options.bujack_edge
            );
        }
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
                        if vals.is_empty() {
                            0
                        } else {
                            vals[vals.len() / 2]
                        },
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
    sensor: Option<&CachedSensor>,
) -> anyhow::Result<(u32, u32, u32, u32, Vec<u8>, String, PreviewStepCache)> {
    process_one_to_preview_with_cache_on_progress(
        path,
        options,
        max_width,
        max_height,
        cache,
        capture_debug,
        sensor,
        &mut |_, _| {},
    )
}

/// Same as [`process_one_to_preview_with_cache`], with 97–100% stage callbacks.
pub fn process_one_to_preview_with_cache_on_progress(
    path: &Path,
    options: &PipelineOptions,
    max_width: u32,
    max_height: u32,
    cache: Option<&PreviewStepCache>,
    capture_debug: bool,
    sensor: Option<&CachedSensor>,
    on_progress: &mut dyn FnMut(&str, f32),
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
        on_progress("Demosaicing…", 0.97);
        let (mut img, tw, th) =
            load_and_demosaic_preview(path, options, max_width, max_height, CpuDemosaic, sensor)?;
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
        if options.flip_horizontal {
            img = flip_array3_horizontal(&img);
        }
        if options.flip_vertical {
            img = flip_array3_vertical(&img);
        }
        color_space::apply_input_idt_to_working_space(&mut img, &options.idt_matrix);
        new_cache.after_load = Some((h1, img.clone(), true_src_w, true_src_h));
        image = Some(img);
    } else {
        let _ = writeln!(
            dbg,
            "=== Pipeline Debug (cached from step {}) ===",
            start_step
        );
    }

    let mut image = image.expect("image set by load or cache");

    let flat_map_preview = options
        .flat_field_path
        .as_ref()
        .and_then(|p| flat_field::load_flat_field_map(p).ok());
    let lut3d_preview = options
        .lut3d_path
        .as_ref()
        .and_then(|p| lut3d::read_cube(p).ok());
    let ra4_params = curve::PrintCurveParams {
        offset: options.curve_offset,
        gamma: options.curve_gamma,
        pivot: options.curve_pivot,
    };
    let output_lut_preview = options
        .output_lut_cube
        .as_ref()
        .and_then(|p| lut3d::read_cube(p).ok());

    if start_step <= 3 {
        apply_optional_dust(&mut image, options);
        on_progress("D-min…", 0.98);
        pipeline::step_3_dmin(&mut image, options, flat_map_preview.as_ref())?;
        new_cache.after_step3 = Some((h3, image.clone()));
    }
    if start_step <= 4 {
        on_progress("Tone and white balance…", 0.98);
        pipeline::step_4_t_to_d_wb(&mut image, options);
        new_cache.after_step4 = Some((h4, image.clone()));
    }
    if start_step <= 5 {
        on_progress("Color…", 0.99);
        pipeline::step_5_calibration(&mut image, options, lut3d_preview.as_ref());
        new_cache.after_step5 = Some((h5, image.clone()));
    }

    on_progress("Print curve…", 0.99);
    let (orig_h, orig_w, _) = image.dim();
    let orig_w = orig_w as u32;
    let orig_h = orig_h as u32;
    let mut display =
        pipeline::step_6_render(&image, options, &ra4_params, output_lut_preview.as_ref());
    pipeline::apply_bujack(&mut display, options);
    let rgb_u8 = pipeline::step6_display_to_u8(&display);
    on_progress("Preview ready", 1.0);

    let _ = writeln!(dbg, "=== end pipeline debug ===");
    let img = RgbImage::from_raw(orig_w, orig_h, rgb_u8)
        .ok_or_else(|| anyhow::anyhow!("Invalid image dimensions"))?;
    let out = img.into_raw();
    Ok((true_src_w, true_src_h, orig_w, orig_h, out, dbg, new_cache))
}

/// Live preview from an existing step cache. Returns `None` if the change
/// requires steps 1–3 (caller should use the full preview job).
///
/// Runs `start_step..=6` and skips De-Bujack. Earlier cache slots are preserved;
/// slots for steps that actually ran are updated.
pub fn apply_preview_from_cache(
    path: &Path,
    options: &PipelineOptions,
    max_width: u32,
    max_height: u32,
    cache: &PreviewStepCache,
) -> Option<(u32, u32, u32, u32, Vec<u8>, PreviewStepCache)> {
    apply_preview_from_cache_on_progress(
        path,
        options,
        max_width,
        max_height,
        cache,
        &mut |_, _| {},
    )
}

/// Live preview from cache, reporting 97–100% as steps 4–6 finish.
pub fn apply_preview_from_cache_on_progress(
    path: &Path,
    options: &PipelineOptions,
    max_width: u32,
    max_height: u32,
    cache: &PreviewStepCache,
    on_progress: &mut dyn FnMut(&str, f32),
) -> Option<(u32, u32, u32, u32, Vec<u8>, PreviewStepCache)> {
    apply_preview_from_cache_cpu(path, options, max_width, max_height, cache, on_progress)
}

fn apply_preview_from_cache_cpu(
    path: &Path,
    options: &PipelineOptions,
    max_width: u32,
    max_height: u32,
    cache: &PreviewStepCache,
    on_progress: &mut dyn FnMut(&str, f32),
) -> Option<(u32, u32, u32, u32, Vec<u8>, PreviewStepCache)> {
    use pipeline_cache::{hash_after_step4, hash_after_step5};

    let start_step = cached_start_step(path, options, max_width, max_height, cache);
    if start_step < 4 {
        return None;
    }

    let (mut image, true_src_w, true_src_h) =
        live_cache_input_buffer(cache, start_step, max_width, max_height)?;

    let mut new_cache = preserve_cache_before(cache, start_step);
    let h4 = hash_after_step4(path, options, max_width, max_height);
    let h5 = hash_after_step5(path, options, max_width, max_height);
    let lut3d = options
        .lut3d_path
        .as_ref()
        .and_then(|p| lut3d::read_cube(p).ok());
    let output_lut = options
        .output_lut_cube
        .as_ref()
        .and_then(|p| lut3d::read_cube(p).ok());
    let ra4_params = curve::PrintCurveParams {
        offset: options.curve_offset,
        gamma: options.curve_gamma,
        pivot: options.curve_pivot,
    };

    on_progress("Applying settings…", 0.97);
    if start_step <= 4 {
        on_progress("Tone and white balance…", 0.98);
        pipeline::step_4_t_to_d_wb(&mut image, options);
        new_cache.after_step4 = Some((h4, image.clone()));
    }
    if start_step <= 5 {
        on_progress("Color…", 0.99);
        pipeline::step_5_calibration(&mut image, options, lut3d.as_ref());
        new_cache.after_step5 = Some((h5, image.clone()));
    }

    on_progress("Print curve…", 0.99);
    let (orig_h, orig_w, _) = image.dim();
    let display = pipeline::step_6_render(&image, options, &ra4_params, output_lut.as_ref());
    let rgb_u8 = pipeline::step6_display_to_u8(&display);
    on_progress("Preview ready", 1.0);
    Some((
        true_src_w,
        true_src_h,
        orig_w as u32,
        orig_h as u32,
        rgb_u8,
        new_cache,
    ))
}

/// GPU live preview from cache. Falls back to CPU if GPU is unavailable or fails.
#[cfg(feature = "gpu")]
pub fn apply_preview_from_cache_gpu(
    path: &Path,
    options: &PipelineOptions,
    max_width: u32,
    max_height: u32,
    cache: &PreviewStepCache,
    gpu: Option<&gpu::unified::GpuPipeline>,
) -> Option<(u32, u32, u32, u32, Vec<u8>, PreviewStepCache)> {
    use pipeline_cache::{hash_after_step4, hash_after_step5};

    let use_gpu = gpu.is_some() && options.use_gpu;
    if !use_gpu {
        return apply_preview_from_cache_cpu(
            path,
            options,
            max_width,
            max_height,
            cache,
            &mut |_, _| {},
        );
    }
    let gpu = gpu.unwrap();

    let start_step = cached_start_step(path, options, max_width, max_height, cache);
    if start_step < 4 {
        return None;
    }

    let (image, true_src_w, true_src_h) =
        live_cache_input_buffer(cache, start_step, max_width, max_height)?;

    let mut new_cache = preserve_cache_before(cache, start_step);
    let h4 = hash_after_step4(path, options, max_width, max_height);
    let h5 = hash_after_step5(path, options, max_width, max_height);
    let lut3d = options
        .lut3d_path
        .as_ref()
        .and_then(|p| lut3d::read_cube(p).ok());
    let output_lut = options
        .output_lut_cube
        .as_ref()
        .and_then(|p| lut3d::read_cube(p).ok());
    let ra4_params = curve::PrintCurveParams {
        offset: options.curve_offset,
        gamma: options.curve_gamma,
        pivot: options.curve_pivot,
    };

    fill_step45_cache(
        &image,
        start_step as u32,
        options,
        lut3d.as_ref(),
        h4,
        h5,
        &mut new_cache,
    );

    let display = gpu
        .run_from_step(
            &image,
            start_step as u32,
            options,
            lut3d.as_ref(),
            &ra4_params,
            output_lut.as_ref(),
        )
        .ok()
        .or_else(|| {
            let mut cpu_img = image;
            if start_step <= 4 {
                pipeline::step_4_t_to_d_wb(&mut cpu_img, options);
            }
            if start_step <= 5 {
                pipeline::step_5_calibration(&mut cpu_img, options, lut3d.as_ref());
            }
            Some(pipeline::step_6_render(
                &cpu_img,
                options,
                &ra4_params,
                output_lut.as_ref(),
            ))
        })?;

    let (orig_h, orig_w, _) = match &display {
        pipeline::Step6Display::PassthroughDensity(img) | pipeline::Step6Display::F32(img) => {
            img.dim()
        }
        pipeline::Step6Display::U16(img) => img.dim(),
    };
    let rgb_u8 = pipeline::step6_display_to_u8(&display);
    Some((
        true_src_w,
        true_src_h,
        orig_w as u32,
        orig_h as u32,
        rgb_u8,
        new_cache,
    ))
}

fn live_cache_input_buffer(
    cache: &PreviewStepCache,
    start_step: u8,
    _max_width: u32,
    _max_height: u32,
) -> Option<(Array3<f32>, u32, u32)> {
    let (true_src_w, true_src_h) = cache
        .after_load
        .as_ref()
        .map(|(_, _, tw, th)| (*tw, *th))
        .unwrap_or((0, 0));
    let buf = if start_step >= 6 {
        cache.after_step5.as_ref().map(|(_, b)| b.clone())
    } else if start_step >= 5 {
        cache.after_step4.as_ref().map(|(_, b)| b.clone())
    } else {
        cache.after_step3.as_ref().map(|(_, b)| b.clone())
    }?;
    let (h, w, _) = buf.dim();
    let tw = if true_src_w == 0 {
        w as u32
    } else {
        true_src_w
    };
    let th = if true_src_h == 0 {
        h as u32
    } else {
        true_src_h
    };
    Some((buf, tw, th))
}

fn preserve_cache_before(cache: &PreviewStepCache, start_step: u8) -> PreviewStepCache {
    let mut new_cache = PreviewStepCache::default();
    if start_step > 1 {
        new_cache.after_load = cache.after_load.clone();
    }
    if start_step > 3 {
        new_cache.after_step3 = cache.after_step3.clone();
    }
    if start_step > 4 {
        new_cache.after_step4 = cache.after_step4.clone();
    }
    if start_step > 5 {
        new_cache.after_step5 = cache.after_step5.clone();
    }
    new_cache
}

#[cfg(feature = "gpu")]
fn fill_step45_cache(
    image: &Array3<f32>,
    start_step: u32,
    options: &PipelineOptions,
    lut3d: Option<&lut3d::Lut3d>,
    h4: u64,
    h5: u64,
    new_cache: &mut PreviewStepCache,
) {
    if start_step <= 4 {
        let mut buf = image.clone();
        pipeline::step_4_t_to_d_wb(&mut buf, options);
        new_cache.after_step4 = Some((h4, buf.clone()));
        pipeline::step_5_calibration(&mut buf, options, lut3d);
        new_cache.after_step5 = Some((h5, buf));
    } else if start_step <= 5 {
        let mut buf = image.clone();
        pipeline::step_5_calibration(&mut buf, options, lut3d);
        new_cache.after_step5 = Some((h5, buf));
    }
}

fn bake_scene_stats_into_options(options: &mut PipelineOptions, stats: &PreviewSceneStats) {
    if let Some(dmin) = stats.dmin {
        options.dmin_mode = DminMode::Fixed;
        options.dmin_fixed = Some(dmin);
    }
    if let Some((ar, ag, ab)) = stats.auto_wb {
        options.auto_wb = false;
        options.apply_white_balance = true;
        options.wb_r *= ar;
        options.wb_g *= ag;
        options.wb_b *= ab;
    }
    options.pinned_zone = stats.zone;
}

/// Load one file, pin full-res D-min / auto-WB, build a 384 px after-step-3
/// buffer, then run [`auto_tune`]. The sensor is dropped before the search.
pub fn run_auto_for_path(
    path: &Path,
    options: &PipelineOptions,
    on_progress: &mut auto::AutoProgressCb<'_>,
) -> anyhow::Result<AutoTuneResult> {
    on_progress("Loading…", 0.0, Some("Loading…"));
    let sensor = load_sensor_from_path(path)?;
    let stats = compute_preview_scene_stats(&sensor, options)?;
    let mut baked = options.clone();
    bake_scene_stats_into_options(&mut baked, &stats);

    on_progress("Preparing analysis…", 0.02, Some("Preparing analysis…"));
    let side = AUTO_PROXY_MAX_SIDE as u32;
    let (_, _, _, _, _, _, cache) =
        process_one_to_preview_with_cache(path, &baked, side, side, None, false, Some(&sensor))?;
    drop(sensor);

    let after_step3 = cache
        .after_step3
        .map(|(_, img)| img)
        .ok_or_else(|| anyhow::anyhow!("Auto: no D-min buffer to analyse."))?;
    auto_tune(&after_step3, &baked, on_progress)
}

/// Load one file, pin full-res D-min / auto-WB, build a proxy after-step-3
/// buffer, then run [`detect_crop`].
pub fn run_auto_crop_for_path(
    path: &Path,
    options: &PipelineOptions,
    on_progress: &mut auto::AutoProgressCb<'_>,
) -> anyhow::Result<AutoCropResult> {
    on_progress("Loading…", 0.0, Some("Loading…"));
    let sensor = load_sensor_from_path(path)?;
    let stats = compute_preview_scene_stats(&sensor, options)?;
    let mut baked = options.clone();
    bake_scene_stats_into_options(&mut baked, &stats);

    on_progress("Detecting frame…", 0.35, Some("Detecting frame…"));
    let side = CROP_PROXY_MAX_SIDE as u32;
    let (_, _, _, _, _, _, cache) =
        process_one_to_preview_with_cache(path, &baked, side, side, None, false, Some(&sensor))?;
    drop(sensor);

    let after_step3 = cache
        .after_step3
        .map(|(_, img)| img)
        .ok_or_else(|| anyhow::anyhow!("Auto crop: no D-min buffer to analyse."))?;
    on_progress("Detecting frame…", 0.85, Some("Detecting frame…"));
    detect_crop(
        &after_step3,
        baked.dmin_rect,
        baked.dmin_rect_reference_size,
    )
    .ok_or_else(|| anyhow::anyhow!("Auto crop: no clear frame boundary found."))
}

/// Load one file to after-step-3 and return crop-detector probe text.
pub fn probe_auto_crop_for_path(path: &Path, options: &PipelineOptions) -> anyhow::Result<String> {
    let sensor = load_sensor_from_path(path)?;
    let stats = compute_preview_scene_stats(&sensor, options)?;
    let mut baked = options.clone();
    bake_scene_stats_into_options(&mut baked, &stats);
    let side = CROP_PROXY_MAX_SIDE as u32;
    let (_, _, _, _, _, _, cache) =
        process_one_to_preview_with_cache(path, &baked, side, side, None, false, Some(&sensor))?;
    let after_step3 = cache
        .after_step3
        .map(|(_, img)| img)
        .ok_or_else(|| anyhow::anyhow!("Auto crop probe: no D-min buffer."))?;
    Ok(auto_crop::crop_probe(&after_step3))
}

/// GPU-accelerated version of `process_one_to_preview_with_cache`.
///
/// When `gpu` is `Some` and `options.use_gpu` is true, runs steps 4→5→6 on
/// the GPU in a single upload/readback via the unified pipeline. Steps 1–3
/// (load, demosaic, D-min) still run on CPU.
///
/// Display pixels still come from the unified GPU pass. Step 4/5 cache slots
/// are filled on the worker with the CPU reference so live sliders can start
/// mid-pipeline without a remosaic.
#[cfg(feature = "gpu")]
pub fn process_one_to_preview_with_cache_gpu(
    path: &Path,
    options: &PipelineOptions,
    max_width: u32,
    max_height: u32,
    cache: Option<&PreviewStepCache>,
    capture_debug: bool,
    gpu: Option<&gpu::unified::GpuPipeline>,
    sensor: Option<&CachedSensor>,
) -> anyhow::Result<(u32, u32, u32, u32, Vec<u8>, String, PreviewStepCache)> {
    let use_gpu = gpu.is_some() && options.use_gpu;
    if !use_gpu {
        return process_one_to_preview_with_cache(
            path,
            options,
            max_width,
            max_height,
            cache,
            capture_debug,
            sensor,
        );
    }
    let gpu = gpu.unwrap();

    use pipeline_cache::{hash_after_load, hash_after_step3, hash_after_step4, hash_after_step5};

    if options.debug_preview_simple_debayer {
        return process_one_to_preview_with_cache(
            path,
            options,
            max_width,
            max_height,
            cache,
            capture_debug,
            sensor,
        );
    }

    let h1 = hash_after_load(path, options, max_width, max_height);
    let h3 = hash_after_step3(path, options, max_width, max_height);
    let h4 = hash_after_step4(path, options, max_width, max_height);
    let h5 = hash_after_step5(path, options, max_width, max_height);

    let mut image: Option<Array3<f32>> = None;
    let mut start_step = 1u32;
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
        // GPU path also checks step 4/5 cache (populated by previous CPU runs)
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
        let (mut img, tw, th) =
            load_and_demosaic_preview(path, options, max_width, max_height, &gpu.demosaic, sensor)?;
        true_src_w = tw;
        true_src_h = th;
        let _ = writeln!(dbg, "=== Pipeline Debug (GPU, with cache) ===");
        let _ = writeln!(dbg, "image: {}x{} (preview)", img.dim().1, img.dim().0);
        let _ = writeln!(dbg, "rotation: {}°", options.rotation_degrees);
        if options.rotation_degrees == 90 || options.rotation_degrees == 270 {
            std::mem::swap(&mut true_src_w, &mut true_src_h);
        }
        if options.rotation_degrees != 0 {
            img = apply_rotation(&img, options.rotation_degrees);
        }
        if options.flip_horizontal {
            img = flip_array3_horizontal(&img);
        }
        if options.flip_vertical {
            img = flip_array3_vertical(&img);
        }
        color_space::apply_input_idt_to_working_space(&mut img, &options.idt_matrix);
        new_cache.after_load = Some((h1, img.clone(), true_src_w, true_src_h));
        image = Some(img);
    } else {
        let _ = writeln!(
            dbg,
            "=== Pipeline Debug (GPU, cached from step {}) ===",
            start_step
        );
    }

    let mut image = image.expect("image set by load or cache");

    // Step 3: flat-field divide and D-min divide on GPU; rect/percentile on CPU
    if start_step <= 3 {
        apply_optional_dust_gpu(&mut image, options, gpu);
        let flat_map_preview = options
            .flat_field_path
            .as_ref()
            .and_then(|p| flat_field::load_flat_field_map(p).ok());
        pipeline::step_3_dmin_gpu(
            &mut image,
            options,
            flat_map_preview.as_ref(),
            Some(&gpu.step3),
        )?;
        new_cache.after_step3 = Some((h3, image.clone()));
    }

    // Steps 4→5→6 on GPU (unified: single upload/readback)
    let gpu_start = start_step.max(4);
    let lut3d_preview = options
        .lut3d_path
        .as_ref()
        .and_then(|p| lut3d::read_cube(p).ok());
    let ra4_params = curve::PrintCurveParams {
        offset: options.curve_offset,
        gamma: options.curve_gamma,
        pivot: options.curve_pivot,
    };
    let output_lut_preview = options
        .output_lut_cube
        .as_ref()
        .and_then(|p| lut3d::read_cube(p).ok());

    // CPU reference buffers for live slider apply (unified GPU skips intermediates).
    fill_step45_cache(
        &image,
        start_step,
        options,
        lut3d_preview.as_ref(),
        h4,
        h5,
        &mut new_cache,
    );

    let mut display = gpu.run_from_step(
        &image,
        gpu_start,
        options,
        lut3d_preview.as_ref(),
        &ra4_params,
        output_lut_preview.as_ref(),
    )?;
    pipeline::apply_bujack(&mut display, options);

    let (orig_h, orig_w, _) = image.dim();
    let orig_w = orig_w as u32;
    let orig_h = orig_h as u32;
    let rgb_u8 = pipeline::step6_display_to_u8(&display);

    let _ = writeln!(dbg, "GPU steps {}-6 complete", gpu_start);
    let _ = writeln!(dbg, "=== end pipeline debug ===");
    let img = RgbImage::from_raw(orig_w, orig_h, rgb_u8)
        .ok_or_else(|| anyhow::anyhow!("Invalid image dimensions"))?;
    let out = img.into_raw();
    Ok((true_src_w, true_src_h, orig_w, orig_h, out, dbg, new_cache))
}

/// Demosaic backend: CPU-only or GPU when available.
trait DemosaicBackend {
    fn demosaic(&self, bayer: &Array3<f32>, pattern: CfaPattern) -> anyhow::Result<Array3<f32>>;
}

struct CpuDemosaic;

impl DemosaicBackend for CpuDemosaic {
    fn demosaic(&self, bayer: &Array3<f32>, pattern: CfaPattern) -> anyhow::Result<Array3<f32>> {
        let mut rgb = demosaic::demosaic_quality(bayer, pattern)?;
        rgb.mapv_inplace(|v| v.max(0.0));
        Ok(rgb)
    }
}

#[cfg(feature = "gpu")]
impl DemosaicBackend for &gpu::demosaic::DemosaicPipeline {
    fn demosaic(&self, bayer: &Array3<f32>, pattern: CfaPattern) -> anyhow::Result<Array3<f32>> {
        gpu::demosaic::demosaic_gpu_or_cpu(bayer, pattern, Some(self))
    }
}

/// Load and demosaic from an already-decoded sensor cache (no disk I/O).
fn load_preview_from_sensor<D: DemosaicBackend>(
    sensor: &CachedSensor,
    options: &PipelineOptions,
    max_width: u32,
    max_height: u32,
    demosaic_backend: D,
) -> anyhow::Result<(Array3<f32>, u32, u32)> {
    match sensor {
        CachedSensor::Bayer { data, pattern } => {
            let (bh, bw, _) = data.dim();
            let full_res = max_width == u32::MAX && max_height == u32::MAX;
            let img = if !full_res {
                if let CfaPattern::Bayer(p) = *pattern {
                    downsample_rgb_for_preview(
                        &pack_bayer_2x2_to_rgb(data, p),
                        max_width,
                        max_height,
                    )
                } else {
                    let small = downsample_raw_for_preview(
                        data,
                        *pattern,
                        preview_mosaic_working_width(max_width),
                    );
                    finish_preview_rgb(
                        demosaic_backend.demosaic(&small, *pattern)?,
                        max_width,
                        max_height,
                    )
                }
            } else {
                demosaic_backend.demosaic(data, *pattern)?
            };
            Ok((img, bw as u32, bh as u32))
        }
        CachedSensor::Rgb(img) => {
            let mut img = img.clone();
            if options.synthetic_negative_input {
                pipeline::apply_synthetic_negative_invert(&mut img);
            }
            let (ph, pw, _) = img.dim();
            let out = downsample_rgb_for_preview(&img, max_width, max_height);
            Ok((out, pw as u32, ph as u32))
        }
    }
}

/// Load and demosaic for preview only (no rotation). Returns (image, true_src_w, true_src_h).
/// When `sensor` is `Some`, skips disk decode and downsamples the cached buffer.
fn load_and_demosaic_preview<D: DemosaicBackend>(
    path: &Path,
    options: &PipelineOptions,
    max_width: u32,
    max_height: u32,
    demosaic_backend: D,
    sensor: Option<&CachedSensor>,
) -> anyhow::Result<(Array3<f32>, u32, u32)> {
    if let Some(sensor) = sensor {
        return load_preview_from_sensor(sensor, options, max_width, max_height, demosaic_backend);
    }
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
            preview_rgb_from_mosaic(
                &bayer,
                pattern,
                max_width,
                max_height,
                options.debug_preview_simple_debayer,
            )?
        }
        "png" | "jpeg" | "jpg" | "tiff" | "tif" => {
            let mut img = png_reader::load_png_as_ndarray(path)?;
            if options.synthetic_negative_input {
                pipeline::apply_synthetic_negative_invert(&mut img);
            }
            let (ph, pw, _) = img.dim();
            true_src_w = pw as u32;
            true_src_h = ph as u32;
            downsample_rgb_for_preview(&img, max_width, max_height)
        }
        _ => anyhow::bail!("Unsupported extension for preview"),
    };
    Ok((image, true_src_w, true_src_h))
}
