//! C-41 RAW Tool GUI: three-panel layout — center preview, right per-image settings, bottom image strip + global output/convert.

// On Windows, use GUI subsystem so closing the window exits with code 0 instead of 0xC000013A (STATUS_CONTROL_C_EXIT).
#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

use std::collections::{hash_map::DefaultHasher, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

use c41_raw_tool::{
    blur_flat_field,
    calibration,
    color,
    demosaic,
    dmin,
    load_flat_field_linear,
    png_reader,
    process_files,
    process_one_to_preview,
    raw_reader,
    tiff_export,
    PipelineOptions,
    PreviewStepCache,
    Rect,
    TiffFormat,
    OutputStage,
    OutputLutEncoding,
    DminMode,
    WbMode,
    CachedSensor,
    load_sensor_from_path,
    compute_dmin_from_sensor,
};
#[cfg(feature = "gpu")]
use c41_raw_tool::process_one_to_preview_with_cache_gpu;
use eframe::egui;

const PREVIEW_MAX_WIDTH: u32 = 1920;
const PREVIEW_MAX_HEIGHT: u32 = 1200;
const THUMB_MAX_SIZE: u32 = 64;
const PREVIEW_DEBOUNCE_MS: u64 = 180;
const BOTTOM_PANEL_HEIGHT: f32 = 150.0;
const RIGHT_PANEL_WIDTH: f32 = 330.0;
const ICON_CLOSE_PATH: &str = "X.png";
const ICON_ROTATE_RIGHT_PATH: &str = "rotate_right.png";
const ICON_ROTATE_LEFT_PATH: &str = "rotate_left.png";
const ICON_LOGO_PATH: &str = "logo.png";

fn app_icon_data() -> Option<egui::IconData> {
    let bytes = include_bytes!("../img/logo.png");
    let rgba = image::load_from_memory(bytes).ok()?.to_rgba8();
    let (width, height) = rgba.dimensions();
    Some(egui::IconData {
        rgba: rgba.into_raw(),
        width,
        height,
    })
}

#[derive(Default)]
struct UiIcons {
    close: Option<egui::TextureHandle>,
    rotate_left: Option<egui::TextureHandle>,
    rotate_right: Option<egui::TextureHandle>,
    logo: Option<egui::TextureHandle>,
}

fn main() -> eframe::Result<()> {
    let mut native_options = if cfg!(target_os = "macos") {
        let mut o = eframe::NativeOptions::default();
        o.viewport = o
            .viewport
            .clone()
            .with_fullsize_content_view(true)
            .with_titlebar_shown(false)
            .with_title_shown(false); // hide OS title so only our white title in the dark bar shows
        o
    } else if cfg!(target_os = "windows") {
        eframe::NativeOptions::default()
    } else {
        eframe::NativeOptions::default()
    };
    if let Some(icon) = app_icon_data() {
        native_options.viewport = native_options.viewport.clone().with_icon(Arc::new(icon));
    }
    eframe::run_native(
        "C-41 RAW Tool",
        native_options,
        Box::new(|cc| {
            let mut visuals = egui::Visuals::dark();
            visuals.window_fill = egui::Color32::from_gray(35);
            visuals.panel_fill = egui::Color32::from_gray(30);
            visuals.override_text_color = Some(egui::Color32::from_gray(240));
            cc.egui_ctx.set_visuals(visuals);
            Ok(Box::new(C41Gui::default()))
        }),
    )
}

struct ImageEntry {
    path: PathBuf,
    options: PipelineOptions,
    /// Current preview texture for the visible ROI (re-built when zoom/pan/size changes).
    preview_texture: Option<egui::TextureHandle>,
    /// Hash of `PipelineOptions` at the time of last full preview processing.
    preview_hash: u64,
    /// Full processed preview buffer (post-curve) at `preview_full_size`.
    /// Used for zoom/pan ROI rendering without re-running the pipeline.
    preview_full_rgb: Option<(u32, u32, Vec<u8>)>,
    /// Dimensions of the preview working image (downscaled). Used as the coordinate reference
    /// space for crop/dmin rect overlays.
    preview_input_size: Option<[u32; 2]>,
    /// True source resolution of the imported file (full sensor/image dimensions).
    /// Used only for display in the info bar.
    raw_source_size: Option<[u32; 2]>,
    /// Zoom factor for preview: 1.0 = full-frame fit, >1.0 = zoom in.
    preview_zoom: f32,
    /// Preview pan in normalized image space [0, 1] (center of the view).
    preview_pan: egui::Vec2,
    /// Small thumbnail for the image strip (generated when loading).
    thumbnail_texture: Option<egui::TextureHandle>,
    // Per-channel histograms (R, G, B) over 0–255
    histogram: Option<([u32; 256], [u32; 256], [u32; 256])>,
    export_format: ExportFormat,
    /// Rawloader debug report for this file (Debug tab).
    raw_debug_report: Option<String>,
    /// Pipeline debug log from the most recent preview render.
    pipeline_debug_log: Option<String>,
    /// Cached full-resolution sensor data (Bayer or RGB) for fast previews/exports.
    cached_sensor: Option<Arc<CachedSensor>>,
    /// Step cache for preview: reuse pipeline stages when only later options change.
    preview_step_cache: Option<PreviewStepCache>,
    /// Process tab (Input/Develop/Export) — persists per image when switching.
    process_tab: ProcessTab,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExportFormat {
    Tiff16,
    Tiff32,
    Exr,
    Jpeg,
    /// EXR ACES2065-1 only (32-bit float).
    ExrAces2065,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UIMode {
    Process,
    Calibrate,
    LuminanceCalibrate,
    Debug,
}

/// Target for the white balance eyedropper: which neutral to set from the sampled point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WbPickerTarget {
    WhitePoint,
    GrayPoint,
    BlackPoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProcessTab {
    Input,
    Develop,
    Export,
}

/// Calibration overlay state: 4 anchor points in normalized image space.
///
/// Corner order: [top-left, top-right, bottom-left, bottom-right], each in
/// normalized coordinates (0..1) relative to the underlying image / preview.
#[derive(Clone, Copy, Debug)]
struct CalibrationOverlay {
    corners: [egui::Pos2; 4],
    /// Half-size of patch bounding boxes as a fraction of the preview height.
    bbox_half_height_frac: f32,
}

impl Default for CalibrationOverlay {
    fn default() -> Self {
        Self {
            corners: [
                egui::pos2(0.20, 0.20), // top-left
                egui::pos2(0.80, 0.20), // top-right
                egui::pos2(0.20, 0.80), // bottom-left
                egui::pos2(0.80, 0.80), // bottom-right
            ],
            // Roughly 10 px on a ~400px tall preview; scaled with preview height.
            bbox_half_height_frac: 10.0 / 400.0,
        }
    }
}

struct C41Gui {
    images: Vec<ImageEntry>,
    selected_index: Option<usize>,
    output_dir: Option<PathBuf>,
    status: String,
    preview_receiver: Option<mpsc::Receiver<anyhow::Result<(usize, u32, u32, u32, u32, Vec<u8>, String, bool, PreviewStepCache)>>>,
    preview_started_at: Option<Instant>,
    /// One-shot flag: capture detailed pipeline debug log on the next preview render.
    capture_pipeline_debug_next: bool,
    /// Thumbnails for the image strip: (path, Ok((w, h, rgb)) or Err).
    thumbnail_receiver: Option<mpsc::Receiver<(PathBuf, anyhow::Result<(u32, u32, Vec<u8>)>)>>,
    thumbnail_pending: HashSet<PathBuf>,
    mode: UIMode,
    calibration_overlay: CalibrationOverlay,
    calibration_result: Option<([[f32; 3]; 3], f32)>, // (matrix, mse)
    calibration_profile_name: String,
    calibration_light_source: String,
    /// (path, profile, LUT path for .c41 or None for .json)
    calibration_profiles: Vec<(PathBuf, calibration::CalibrationProfile, Option<PathBuf>)>,
    selected_profile_idx: Option<usize>,
    /// Luminance calibration: path and linearized flat-field image (RAW → demosaic only).
    flat_field_path: Option<PathBuf>,
    flat_field_image: Option<ndarray::Array3<f32>>,
    ui_icons: UiIcons,
    /// Suppresses preview reprocessing while the user is dragging a rect handle (crop / d-min).
    rect_dragging: bool,
    /// Debounce state for preview refreshes: (image index, options hash) currently waiting to settle.
    pending_preview_key: Option<(usize, u64)>,
    pending_preview_since: Option<Instant>,
    /// Deferred file dialog flag: open the output LUT browser outside the egui render loop
    /// to avoid macOS NSOpenPanel re-entrance crashes.
    pending_output_lut_browse: bool,
    /// When true, preview uses full resolution (export pipeline). Deactivates on option change or image switch.
    full_res_preview_active: bool,
    /// One-shot: set by the full-res button so we don't deactivate on the preview request it triggers.
    full_res_preview_button_clicked: bool,
    /// When set, the next click on the preview will sample density and set WB gains (white/gray/black point).
    wb_picker_pending: Option<WbPickerTarget>,
    /// Canvas size (w, h) in points from last layout — used to request preview at screen resolution.
    preview_canvas_size: Option<(f32, f32)>,
    #[cfg(feature = "gpu")]
    gpu_pipeline: Option<std::sync::Arc<c41_raw_tool::gpu::unified::GpuPipeline>>,
}

impl Default for C41Gui {
    fn default() -> Self {
        Self {
            images: Vec::new(),
            selected_index: None,
            output_dir: None,
            status: String::new(),
            preview_receiver: None,
            preview_started_at: None,
            capture_pipeline_debug_next: false,
            thumbnail_receiver: None,
            thumbnail_pending: HashSet::new(),
            mode: UIMode::Process,
            calibration_overlay: CalibrationOverlay::default(),
            calibration_result: None,
            calibration_profile_name: String::new(),
            calibration_light_source: String::new(),
            calibration_profiles: Vec::new(),
            selected_profile_idx: None,
            flat_field_path: None,
            flat_field_image: None,
            ui_icons: UiIcons::default(),
            rect_dragging: false,
            pending_preview_key: None,
            pending_preview_since: None,
            pending_output_lut_browse: false,
            full_res_preview_active: false,
            full_res_preview_button_clicked: false,
            wb_picker_pending: None,
            preview_canvas_size: None,
            #[cfg(feature = "gpu")]
            gpu_pipeline: c41_raw_tool::gpu::unified::GpuPipeline::try_new()
                .map(std::sync::Arc::new),
        }
    }
}

fn load_linear_transmittance_for_calibration(
    path: &Path,
    opts: &PipelineOptions,
) -> anyhow::Result<ndarray::Array3<f32>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    let mut image = match ext.as_str() {
        "arw" | "nef" | "nrw" | "cr2" | "cr3" | "crw" | "dng" | "raf" | "orf" | "rw2" => {
            let (bayer, pattern) = raw_reader::load_raw_as_ndarray(path)?;
            demosaic::demosaic_quality(&bayer, pattern)?
        }
        "png" | "jpeg" | "jpg" | "tiff" | "tif" => png_reader::load_png_as_ndarray(path)?,
        _ => anyhow::bail!("Unsupported extension for calibration"),
    };

    if let Some((r, g, b)) = opts.dmin_fixed {
        dmin::neutralize_with_medians(&mut image, r, g, b)?;
    } else if let Some(rect) = opts.dmin_rect {
        dmin::neutralize(
            &mut image,
            rect.x,
            rect.y,
            rect.width,
            rect.height,
            opts.dmin_neutral_only,
        )?;
    }

    Ok(image)
}

fn compute_patch_centers_normalized(
    corners: [egui::Pos2; 4],
) -> [[f32; 2]; 24] {
    let mut centers = [[0.0_f32; 2]; 24];
    let rows = 4usize;
    let cols = 6usize;

    for row in 0..rows {
        let v = if rows > 1 {
            row as f32 / (rows as f32 - 1.0)
        } else {
            0.0
        };
        let left = corners[0].lerp(corners[2], v);
        let right = corners[1].lerp(corners[3], v);

        for col in 0..cols {
            let u = if cols > 1 {
                col as f32 / (cols as f32 - 1.0)
            } else {
                0.0
            };
            let center = left.lerp(right, u);
            let idx = row * cols + col;
            centers[idx][0] = center.x;
            centers[idx][1] = center.y;
        }
    }

    centers
}

fn sample_patch_medians(
    image: &ndarray::Array3<f32>,
    centers_norm: &[[f32; 2]; 24],
    bbox_half_size_px: f32,
) -> [[f32; 3]; 24] {
    use std::cmp::{max, min};

    let (h, w, _) = image.dim();
    let mut out = [[0.0_f32; 3]; 24];

    for (i, center) in centers_norm.iter().enumerate() {
        let cx = center[0].clamp(0.0, 1.0) * (w as f32 - 1.0);
        let cy = center[1].clamp(0.0, 1.0) * (h as f32 - 1.0);

        let half = bbox_half_size_px;
        let x_min = max(0, (cx - half).floor() as isize) as usize;
        let y_min = max(0, (cy - half).floor() as isize) as usize;
        let x_max = min(w.saturating_sub(1), (cx + half).ceil().max(0.0) as usize);
        let y_max = min(h.saturating_sub(1), (cy + half).ceil().max(0.0) as usize);

        let mut r_vals = Vec::new();
        let mut g_vals = Vec::new();
        let mut b_vals = Vec::new();

        for y in y_min..=y_max {
            for x in x_min..=x_max {
                let r = image[(y, x, 0)];
                let g = image[(y, x, 1)];
                let b = image[(y, x, 2)];
                r_vals.push(r);
                g_vals.push(g);
                b_vals.push(b);
            }
        }

        if !r_vals.is_empty() {
            r_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            g_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            b_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = r_vals.len() / 2;
            out[i][0] = r_vals[mid];
            out[i][1] = g_vals[mid];
            out[i][2] = b_vals[mid];
        }
    }

    out
}

fn default_options() -> PipelineOptions {
    PipelineOptions {
        dmin_mode: DminMode::AutoPercentile,
        auto_norm_buffer: 0.2,
        apply_white_balance: true,
        auto_wb: true,
        wb_mode: WbMode::Auto,
        film_gamma: 0.65,
        dmin_rect: None,
        dmin_rect_reference_size: None,
        apply_crop: false,
        crop_rect: None,
        crop_rect_reference_size: None,
        dmin_fixed: Some((0.222537, 0.108183, 0.054116)),
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
        apply_color_profile: false,
        density_matrix: [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        flat_field_path: None,
        export_aces_exr: false,
        write_aces2065_only: false,
        lut3d_path: None,
        output_stage: OutputStage::Ra4,
        output_lut_cube: None,
        output_lut_encoding: OutputLutEncoding::CineonLog,
        lut_in_black: 0.0,
        lut_in_white: 1.0,
        lut_in_mid: 1.0,
        fp_offset_r: 0.0,
        fp_offset_g: 0.0,
        fp_offset_b: 0.0,
        fp_gamma_r: 1.0,
        fp_gamma_g: 1.0,
        fp_gamma_b: 1.0,
        fp_color_bleed: 0.08,
        fp_vibrance: 0.3,
        saturation: 1.0,
        toe_strength: 0.0,
        shoulder_strength: 0.0,
        shadow_cast_strength: 0.0,
        zone_shadows: 0.0,
        zone_highlights: 0.0,
        zone_shadow_gain: 0.0,
        zone_mid_gain: 0.0,
        zone_highlight_gain: 0.0,
        color_shadow_gain_r: 0.0,
        color_shadow_gain_g: 0.0,
        color_shadow_gain_b: 0.0,
        color_mid_gain_r: 0.0,
        color_mid_gain_g: 0.0,
        color_mid_gain_b: 0.0,
        color_highlight_gain_r: 0.0,
        color_highlight_gain_g: 0.0,
        color_highlight_gain_b: 0.0,
        zone_shadow_saturation: 1.0,
        zone_mid_saturation: 1.0,
        zone_highlight_saturation: 1.0,
        highlight_rolloff: 0.0,
        highlight_rolloff_d_mid: 1.5,
        highlight_warmth: 0.0,
        soft_clip: 0.93,
        apply_lab: false,
        lab_separation: 0.0,
        skin_magenta_shift: 0.0,
        rotation_degrees: 0,
        flip_horizontal: false,
        flip_vertical: false,
        debug_pipeline_step: 6,
        debug_preview_simple_debayer: false,
        verbose_debug: false,
        use_gpu: cfg!(feature = "gpu"),
    }
}

fn options_hash_for(path: &PathBuf, opts: &PipelineOptions) -> u64 {
    let mut h = DefaultHasher::new();
    path.display().to_string().hash(&mut h);
    opts.dmin_mode.hash(&mut h);
    opts.auto_norm_buffer.to_bits().hash(&mut h);
    opts.apply_white_balance.hash(&mut h);
    opts.wb_mode.hash(&mut h);
    opts.auto_wb.hash(&mut h);
    opts.film_gamma.to_bits().hash(&mut h);
    opts.saturation.to_bits().hash(&mut h);
    opts.highlight_warmth.to_bits().hash(&mut h);
    opts.apply_lab.hash(&mut h);
    opts.lab_separation.to_bits().hash(&mut h);
    opts.skin_magenta_shift.to_bits().hash(&mut h);
    opts.apply_color_profile.hash(&mut h);
    opts.dmin_rect.hash(&mut h);
    opts.apply_crop.hash(&mut h);
    opts.crop_rect.hash(&mut h);
    opts.dmin_neutral_only.hash(&mut h);
    if let Some((r, g, b)) = opts.dmin_fixed {
        r.to_bits().hash(&mut h);
        g.to_bits().hash(&mut h);
        b.to_bits().hash(&mut h);
    }
    (opts.wb_r.to_bits(), opts.wb_g.to_bits(), opts.wb_b.to_bits()).hash(&mut h);
    opts.temp_k.map(|k| k.to_bits()).hash(&mut h);
    opts.no_curve.hash(&mut h);
    opts.no_invert.hash(&mut h);
    opts.curve_offset.to_bits().hash(&mut h);
    opts.curve_gamma.to_bits().hash(&mut h);
    opts.curve_pivot.to_bits().hash(&mut h);
    opts.curve_white.to_bits().hash(&mut h);
    (opts.format as u8).hash(&mut h);
    opts.write_exr.hash(&mut h);
    opts.write_jpeg.hash(&mut h);
    opts.write_jpeg_only.hash(&mut h);
    for row in &opts.density_matrix {
        for v in row {
            v.to_bits().hash(&mut h);
        }
    }
    opts.flat_field_path.as_ref().map(|p| p.display().to_string()).hash(&mut h);
    opts.export_aces_exr.hash(&mut h);
    opts.write_aces2065_only.hash(&mut h);
    opts.lut3d_path.as_ref().map(|p| p.display().to_string()).hash(&mut h);
    opts.output_stage.hash(&mut h);
    opts.output_lut_cube
        .as_ref()
        .map(|p| p.display().to_string())
        .hash(&mut h);
    opts.output_lut_encoding.hash(&mut h);
    opts.lut_in_black.to_bits().hash(&mut h);
    opts.lut_in_white.to_bits().hash(&mut h);
    opts.lut_in_mid.to_bits().hash(&mut h);
    opts.soft_clip.to_bits().hash(&mut h);
    opts.toe_strength.to_bits().hash(&mut h);
    opts.shoulder_strength.to_bits().hash(&mut h);
    opts.fp_offset_r.to_bits().hash(&mut h);
    opts.fp_offset_g.to_bits().hash(&mut h);
    opts.fp_offset_b.to_bits().hash(&mut h);
    opts.fp_gamma_r.to_bits().hash(&mut h);
    opts.fp_gamma_g.to_bits().hash(&mut h);
    opts.fp_gamma_b.to_bits().hash(&mut h);
    opts.fp_color_bleed.to_bits().hash(&mut h);
    opts.fp_vibrance.to_bits().hash(&mut h);
    opts.shadow_cast_strength.to_bits().hash(&mut h);
    opts.zone_shadows.to_bits().hash(&mut h);
    opts.zone_highlights.to_bits().hash(&mut h);
    opts.zone_shadow_gain.to_bits().hash(&mut h);
    opts.zone_mid_gain.to_bits().hash(&mut h);
    opts.zone_highlight_gain.to_bits().hash(&mut h);
    opts.color_shadow_gain_r.to_bits().hash(&mut h);
    opts.color_shadow_gain_g.to_bits().hash(&mut h);
    opts.color_shadow_gain_b.to_bits().hash(&mut h);
    opts.color_mid_gain_r.to_bits().hash(&mut h);
    opts.color_mid_gain_g.to_bits().hash(&mut h);
    opts.color_mid_gain_b.to_bits().hash(&mut h);
    opts.color_highlight_gain_r.to_bits().hash(&mut h);
    opts.color_highlight_gain_g.to_bits().hash(&mut h);
    opts.color_highlight_gain_b.to_bits().hash(&mut h);
    opts.zone_shadow_saturation.to_bits().hash(&mut h);
    opts.zone_mid_saturation.to_bits().hash(&mut h);
    opts.zone_highlight_saturation.to_bits().hash(&mut h);
    opts.highlight_rolloff.to_bits().hash(&mut h);
    opts.highlight_rolloff_d_mid.to_bits().hash(&mut h);
    opts.rotation_degrees.hash(&mut h);
    opts.flip_horizontal.hash(&mut h);
    opts.flip_vertical.hash(&mut h);
    opts.debug_pipeline_step.hash(&mut h);
    opts.debug_preview_simple_debayer.hash(&mut h);
    opts.verbose_debug.hash(&mut h);
    opts.use_gpu.hash(&mut h);
    h.finish()
}

/// Flip a rect horizontally (mirror left–right) within an image of `img_w` × `img_h`.
fn flip_rect_horizontal(rect: Rect, img_w: u32, _img_h: u32) -> Rect {
    let new_x = img_w.saturating_sub(rect.x).saturating_sub(rect.width).max(0);
    Rect {
        x: new_x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
    }
}

/// Flip a rect vertically (mirror top–bottom) within an image of `img_w` × `img_h`.
fn flip_rect_vertical(rect: Rect, _img_w: u32, img_h: u32) -> Rect {
    let new_y = img_h.saturating_sub(rect.y).saturating_sub(rect.height).max(0);
    Rect {
        x: rect.x,
        y: new_y,
        width: rect.width,
        height: rect.height,
    }
}

/// Rotate a pixel-space rect 90 degrees within an image of `img_w` × `img_h`.
/// Coordinates are treated as half-open bounds: [x, x+width) × [y, y+height).
fn rotate_dmin_rect_90(rect: Rect, img_w: u32, img_h: u32, clockwise: bool) -> Rect {
    let img_w = img_w as i64;
    let img_h = img_h as i64;

    // Clamp incoming bounds to source image size before rotating.
    let left = (rect.x as i64).clamp(0, img_w);
    let top = (rect.y as i64).clamp(0, img_h);
    let right = ((rect.x as i64) + (rect.width as i64)).clamp(0, img_w);
    let bottom = ((rect.y as i64) + (rect.height as i64)).clamp(0, img_h);

    let (new_left, new_top, new_right, new_bottom) = if clockwise {
        // 90 CW: (x, y) -> (h - y, x) in half-open space.
        (img_h - bottom, left, img_h - top, right)
    } else {
        // 90 CCW: (x, y) -> (y, w - x) in half-open space.
        (top, img_w - right, bottom, img_w - left)
    };

    Rect {
        x: new_left.max(0) as u32,
        y: new_top.max(0) as u32,
        width: (new_right - new_left).max(1) as u32,
        height: (new_bottom - new_top).max(1) as u32,
    }
}

fn scale_rect_to_size(
    rect: Rect,
    reference_size: Option<(u32, u32)>,
    current_w: u32,
    current_h: u32,
) -> Rect {
    let (x, y, rw, rh) = (rect.x, rect.y, rect.width, rect.height);
    match reference_size {
        None => Rect {
            x,
            y,
            width: rw.max(1),
            height: rh.max(1),
        },
        Some((ref_w, ref_h)) if ref_w == current_w && ref_h == current_h => Rect {
            x,
            y,
            width: rw.max(1),
            height: rh.max(1),
        },
        Some((ref_w, ref_h)) if ref_w > 0 && ref_h > 0 => Rect {
            x: (x as f32 * current_w as f32 / ref_w as f32).round() as u32,
            y: (y as f32 * current_h as f32 / ref_h as f32).round() as u32,
            width: (rw as f32 * current_w as f32 / ref_w as f32).round().max(1.0) as u32,
            height: (rh as f32 * current_h as f32 / ref_h as f32).round().max(1.0) as u32,
        },
        _ => Rect {
            x,
            y,
            width: rw.max(1),
            height: rh.max(1),
        },
    }
}

/// Analyses the processed preview buffer and returns the tightest crop rect
/// that encloses the actual film frame while excluding the uniform border
/// (film base / rebate).  Works regardless of whether the border is bright
/// or dark – it samples the outer perimeter to establish a border colour and
/// then scans inward row-by-row and column-by-column until it finds content
/// that differs meaningfully from that border colour.
///
/// Returns `None` if the image is too small or no clear frame boundary is
/// found (e.g. the image is already tightly cropped).
/// Samples the border (film-base) colour from the *processed* preview output
/// using only reliable D-min data sources.
///
/// Two sources, tried in order:
///
/// 1. **D-min rect** (`dmin_rect` already scaled to preview pixel space) –
///    the user explicitly placed this rectangle over the film base, so these
///    pixels are guaranteed to be rebate/border.
///
/// 2. **Auto-percentile outer strip** – mirrors the `AutoPercentile` D-min
///    logic: sample the outermost `auto_norm_buffer` fraction of each side
///    (the same region the pipeline excludes from its percentile calculation),
///    then take the top-10 % brightest pixels by luminance from that strip as
///    the film-base reference.  This strip is where the pipeline itself
///    expects to find film base; it is not guaranteed to be free of image
///    content on multi-frame strips, but it is the best available estimate
///    when no explicit rect exists.
///
/// Returns `None` when `dmin_rect` is absent *and* `auto_norm_buffer` is too
/// small to produce a meaningful sample (≤ 0.02), so the caller can
/// inform the user to set a D-min region.
///
/// Otherwise returns `Some((r, g, b, tolerance))` in 0–255 float scale where
/// `tolerance` is 2.5 × max per-channel std-dev, clamped to [12, 55].
fn sample_border_color(
    width: u32,
    height: u32,
    pixels: &[u8],
    dmin_rect: Option<Rect>,
    auto_norm_buffer: f32,
) -> Option<(f32, f32, f32, f32)> {
    let accumulate = |x: u32, y: u32,
                      sums:   &mut [f64; 3],
                      sq_sum: &mut [f64; 3],
                      count:  &mut u64| {
        if x >= width || y >= height { return; }
        let i = ((y * width + x) * 3) as usize;
        for c in 0..3 {
            let v = pixels[i + c] as f64;
            sums[c]   += v;
            sq_sum[c] += v * v;
        }
        *count += 1;
    };

    let mut sums   = [0.0f64; 3];
    let mut sq_sum = [0.0f64; 3];
    let mut count  = 0u64;

    // ── Source 1: explicit D-min rect ────────────────────────────────────────
    if let Some(r) = dmin_rect {
        let x1 = (r.x + r.width).min(width);
        let y1 = (r.y + r.height).min(height);
        for y in r.y..y1 {
            for x in r.x..x1 {
                accumulate(x, y, &mut sums, &mut sq_sum, &mut count);
            }
        }
    }

    // ── Source 2: auto-percentile outer strip ────────────────────────────────
    if count == 0 {
        if auto_norm_buffer <= 0.02 {
            return None; // Buffer too small; no reliable border source.
        }

        let bw = ((width  as f32 * auto_norm_buffer) as u32).max(2).min(width  / 3);
        let bh = ((height as f32 * auto_norm_buffer) as u32).max(2).min(height / 3);

        // Collect all pixels in the outer strip.
        let mut strip: Vec<(u8, u8, u8)> = Vec::new();
        for y in 0..bh {                        // top
            for x in 0..width  { let i = ((y*width+x)*3) as usize; strip.push((pixels[i],pixels[i+1],pixels[i+2])); }
        }
        for y in (height-bh)..height {          // bottom
            for x in 0..width  { let i = ((y*width+x)*3) as usize; strip.push((pixels[i],pixels[i+1],pixels[i+2])); }
        }
        for x in 0..bw {                        // left (excluding already-counted rows)
            for y in bh..(height-bh) { let i = ((y*width+x)*3) as usize; strip.push((pixels[i],pixels[i+1],pixels[i+2])); }
        }
        for x in (width-bw)..width {            // right
            for y in bh..(height-bh) { let i = ((y*width+x)*3) as usize; strip.push((pixels[i],pixels[i+1],pixels[i+2])); }
        }

        if strip.is_empty() { return None; }

        // Sort by luminance ascending; keep the darkest 10 % – after
        // inversion the film base is the darkest region of the output.
        strip.sort_unstable_by(|a, b| {
            let la = a.0 as u32 + a.1 as u32 + a.2 as u32;
            let lb = b.0 as u32 + b.1 as u32 + b.2 as u32;
            la.cmp(&lb)
        });
        let keep = ((strip.len() as f32 * 0.10) as usize).max(1);
        for &(r, g, b) in &strip[..keep] {
            sums[0]   += r as f64; sq_sum[0] += (r as f64) * (r as f64);
            sums[1]   += g as f64; sq_sum[1] += (g as f64) * (g as f64);
            sums[2]   += b as f64; sq_sum[2] += (b as f64) * (b as f64);
            count     += 1;
        }
    }

    if count == 0 { return None; }

    let n = count as f64;
    let mut means   = [0.0f32; 3];
    let mut max_std = 0.0f32;
    for c in 0..3 {
        let mean = sums[c] / n;
        let var  = (sq_sum[c] / n - mean * mean).max(0.0);
        means[c] = mean as f32;
        max_std   = max_std.max(var.sqrt() as f32);
    }

    // Sanity check: after inversion the film base is always very dark (near
    // black).  If the sampled colour is too bright the outer strip contained
    // no real border pixels – the frame is larger than auto_norm_buffer, or
    // the image is too tightly framed.  In that case the colour is useless
    // and would cause bright image content to be mis-classified as border.
    let lum = (means[0] + means[1] + means[2]) / 3.0;
    if lum > 80.0 {
        return None;
    }

    let tol = (max_std * 2.5).clamp(12.0, 55.0);
    Some((means[0], means[1], means[2], tol))
}

/// Otsu threshold for a 1D distribution: quantize into bins, choose the boundary
/// that maximizes between-class variance. Returns (threshold, separation_ratio)
/// where separation_ratio = between_var / total_var (high = clear bimodal split).
fn otsu_threshold_1d(values: &[f32], num_bins: usize) -> Option<(f32, f64)> {
    if values.is_empty() || num_bins < 2 {
        return None;
    }
    let min_v = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max_v = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max_v - min_v;
    if range <= 0.0 {
        return None;
    }
    let mut hist = vec![0u32; num_bins];
    let bin_w = range / num_bins as f32;
    for &v in values {
        let b = ((v - min_v) / range * num_bins as f32)
            .clamp(0.0, (num_bins - 1) as f32) as usize;
        hist[b] = hist[b].saturating_add(1);
    }
    let n: f64 = hist.iter().map(|&c| c as f64).sum();
    if n <= 0.0 {
        return None;
    }
    let total_mean: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, &c)| (i as f64 + 0.5) * bin_w as f64 * c as f64)
        .sum::<f64>()
        / n;
    let total_var: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let center = min_v as f64 + (i as f64 + 0.5) * bin_w as f64;
            (center - total_mean).powi(2) * c as f64
        })
        .sum::<f64>()
        / n;
    if total_var <= 0.0 {
        return None;
    }
    let mut best_t = 0usize;
    let mut best_var = 0.0f64;
    for t in 1..num_bins {
        let n_low: f64 = hist[..t].iter().map(|&c| c as f64).sum();
        let n_high = n - n_low;
        if n_low <= 0.0 || n_high <= 0.0 {
            continue;
        }
        let mean_low: f64 = hist[..t]
            .iter()
            .enumerate()
            .map(|(i, &c)| (i as f64 + 0.5) * bin_w as f64 * c as f64)
            .sum::<f64>()
            / n_low;
        let mean_high: f64 = hist[t..]
            .iter()
            .enumerate()
            .map(|(i, &c)| (t as f64 + i as f64 + 0.5) * bin_w as f64 * c as f64)
            .sum::<f64>()
            / n_high;
        let between = n_low * n_high * (mean_low - mean_high).powi(2);
        if between > best_var {
            best_var = between;
            best_t = t;
        }
    }
    let threshold_value = min_v + (best_t as f32 + 0.5) * bin_w;
    let separation = best_var / total_var;
    Some((threshold_value, separation))
}

/// Percentile of a 1D sample (0.0 = min, 1.0 = max). Uses linear interpolation.
fn percentile_1d(values: &[f32], p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let p = p.clamp(0.0, 1.0);
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = p * (n - 1) as f32;
    let i = idx.floor() as usize;
    let frac = idx - i as f32;
    if i >= n - 1 {
        sorted[n - 1]
    } else {
        sorted[i] * (1.0 - frac) + sorted[i + 1] * frac
    }
}

/// Detects the film-frame boundary via mean row/column luminance.
///
/// 1. Compute mean luminance per row and per column.
/// 2. Data-driven threshold (Otsu or percentile) plus a hard cap from the
///    sampled film-base colour: a row/column only counts as "border" if its
///    mean is both below the threshold and <= border_lum + tolerance.  That
///    avoids mistaking in-image edges (sky, landscape) for film border.
/// 3. Scan from centre outward; require minimum crop size so we never return
///    a tiny or zero-size rect.
fn auto_detect_crop(
    width: u32,
    height: u32,
    pixels: &[u8],
    border_color: (f32, f32, f32),
    tolerance: f32,
) -> Option<Rect> {
    if width < 16 || height < 16 || pixels.len() < (width * height * 3) as usize {
        return None;
    }

    let (br, bg, bb) = border_color;
    let border_lum = 0.2126 * br + 0.7152 * bg + 0.0722 * bb;
    let border_cap = (border_lum + tolerance * 2.0).min(100.0);

    // Rec. 709 weighted luminance, 0–255 scale.
    let lum_px = |x: u32, y: u32| -> f32 {
        let i = ((y * width + x) * 3) as usize;
        0.2126 * pixels[i] as f32
            + 0.7152 * pixels[i + 1] as f32
            + 0.0722 * pixels[i + 2] as f32
    };

    // Mean luminance for every row and every column.
    let row_mean: Vec<f32> = (0..height)
        .map(|y| (0..width).map(|x| lum_px(x, y)).sum::<f32>() / width as f32)
        .collect();
    let col_mean: Vec<f32> = (0..width)
        .map(|x| (0..height).map(|y| lum_px(x, y)).sum::<f32>() / height as f32)
        .collect();

    const BINS: usize = 64;
    const BIMODAL_MIN: f64 = 0.12;

    let row_threshold = {
        let (otsu_t, separation) = match otsu_threshold_1d(&row_mean, BINS) {
            Some(x) => x,
            None => return None,
        };
        let use_bimodal = separation >= BIMODAL_MIN;
        let t = if use_bimodal {
            otsu_t
        } else {
            let p15 = percentile_1d(&row_mean, 0.15);
            let p50 = percentile_1d(&row_mean, 0.5);
            p15 + 0.2 * (p50 - p15).max(0.0)
        };
        t.min(border_cap)
    };
    let col_threshold = {
        let (otsu_t, separation) = match otsu_threshold_1d(&col_mean, BINS) {
            Some(x) => x,
            None => return None,
        };
        let use_bimodal = separation >= BIMODAL_MIN;
        let t = if use_bimodal {
            otsu_t
        } else {
            let p15 = percentile_1d(&col_mean, 0.15);
            let p50 = percentile_1d(&col_mean, 0.5);
            p15 + 0.2 * (p50 - p15).max(0.0)
        };
        t.min(border_cap)
    };

    // Fraction of pixels in each row/col that are dark (below threshold and border_cap).
    // So thin borders (only a few dark pixels) still register.
    let row_dark_frac: Vec<f32> = (0..height)
        .map(|y| {
            let count = (0..width)
                .filter(|&x| {
                    let l = lum_px(x, y);
                    l < row_threshold && l <= border_cap
                })
                .count();
            count as f32 / width as f32
        })
        .collect();
    let col_dark_frac: Vec<f32> = (0..width)
        .map(|x| {
            let count = (0..height)
                .filter(|&y| {
                    let l = lum_px(x, y);
                    l < col_threshold && l <= border_cap
                })
                .count();
            count as f32 / height as f32
        })
        .collect();

    const DARK_FRAC_MIN: f32 = 0.04;
    /// Only use dark_frac as "border" when mean is below this (avoid content rows with a few dark pixels).
    const DARK_FRAC_MEAN_CEILING: f32 = 65.0;
    /// Only use dark_frac for border when row/col is in this fraction of the image edge (avoids center column/row with dark content).
    const EDGE_BAND: f32 = 0.05;
    let row_edge_band = (height as f32 * EDGE_BAND).ceil() as u32;
    let col_edge_band = (width as f32 * EDGE_BAND).ceil() as u32;

    let cx = width / 2;
    let cy = height / 2;
    // Border row/col: mean below threshold, or (in edge band + mean not bright + enough dark pixels for thin edge at margin).
    let br_row = |y: u32| {
        let m = row_mean[y as usize];
        let in_edge = y < row_edge_band || y >= height.saturating_sub(row_edge_band);
        (m < row_threshold && m <= border_cap)
            || (in_edge && m < DARK_FRAC_MEAN_CEILING && row_dark_frac[y as usize] >= DARK_FRAC_MIN)
    };
    let bc_col = |x: u32| {
        let m = col_mean[x as usize];
        let in_edge = x < col_edge_band || x >= width.saturating_sub(col_edge_band);
        (m < col_threshold && m <= border_cap)
            || (in_edge && m < DARK_FRAC_MEAN_CEILING && col_dark_frac[x as usize] >= DARK_FRAC_MIN)
    };
    // Mean-only border (for edge pull: only pull to image edge when edge is clearly border by luminance, not dark_frac).
    let br_row_mean_only = |y: u32| {
        let m = row_mean[y as usize];
        m < row_threshold && m <= border_cap
    };
    let bc_col_mean_only = |x: u32| {
        let m = col_mean[x as usize];
        m < col_threshold && m <= border_cap
    };

    // Scan from centre outward. Prefer 2-run (two consecutive border rows/cols); fall back to 1-run, then image edge.
    let top_2run = (1..cy).rev().find(|&y| br_row(y) && br_row(y - 1)).map(|y| y + 1);
    let top_1run = (1..cy).rev().find(|&y| br_row(y)).map(|y| y + 1);
    let top = top_2run.or(top_1run).unwrap_or(0);

    let bottom_2run = (cy + 1..height - 1).find(|&y| br_row(y) && br_row(y + 1));
    let bottom_1run = (cy + 1..height).find(|&y| br_row(y));
    let bottom = bottom_2run.or(bottom_1run).unwrap_or(height);

    let left_2run = (1..cx).rev().find(|&x| bc_col(x) && bc_col(x - 1)).map(|x| x + 1);
    let left_1run = (1..cx).rev().find(|&x| bc_col(x)).map(|x| x + 1);
    let left = left_2run.or(left_1run).unwrap_or(0);

    let right_2run = (cx + 1..width - 1).find(|&x| bc_col(x) && bc_col(x + 1));
    let right_1run = (cx + 1..width).find(|&x| bc_col(x));
    let right = right_2run.or(right_1run).unwrap_or(width);

    // Edge pull only when edge is border by mean (not just dark_frac), so we don't pull on mixed/ambiguous edges.
    let top = if top > 0 && br_row_mean_only(0) { 0 } else { top };
    let bottom = if bottom < height && br_row_mean_only(height - 1) { height } else { bottom };
    let left = if left > 0 && bc_col_mean_only(0) { 0 } else { left };
    let right = if right < width && bc_col_mean_only(width - 1) { width } else { right };

    // Inward trim to avoid including sprocket holes (detected border can sit just inside the frame).
    const SPROCKET_TRIM: u32 = 4;
    let left = (left + SPROCKET_TRIM).min(cx);
    let right = right.saturating_sub(SPROCKET_TRIM).max(cx + 1);
    let top = (top + SPROCKET_TRIM).min(cy);
    let bottom = bottom.saturating_sub(SPROCKET_TRIM).max(cy + 1);

    if right <= left || bottom <= top {
        return None;
    }

    if left >= cx || right <= cx || top >= cy || bottom <= cy {
        return None;
    }

    let w = right - left;
    let h = bottom - top;
    let min_side = (width.min(height) / 20).max(16);
    if w < min_side || h < min_side {
        return None;
    }

    let rect_cx = (left + right) / 2;
    let rect_cy = (top + bottom) / 2;
    let max_shift_x = (width / 5).max(1);
    let max_shift_y = (height / 5).max(1);
    let shift_x = rect_cx.abs_diff(cx);
    let shift_y = rect_cy.abs_diff(cy);
    if shift_x > max_shift_x || shift_y > max_shift_y {
        return None;
    }

    Some(Rect {
        x: left,
        y: top,
        width: w,
        height: h,
    })
}

fn compute_histogram_from_rgb(
    rgb: &[u8],
    w: u32,
    h: u32,
    opts: &PipelineOptions,
    input_w: u32,
    input_h: u32,
) -> ([u32; 256], [u32; 256], [u32; 256]) {
    let mut r_hist = [0u32; 256];
    let mut g_hist = [0u32; 256];
    let mut b_hist = [0u32; 256];
    let width = w as usize;

    let crop_in_preview = if opts.apply_crop {
        opts.crop_rect.map(|crop_rect| {
            let scaled = scale_rect_to_size(
                crop_rect,
                opts.crop_rect_reference_size,
                input_w,
                input_h,
            );
            scale_rect_to_size(scaled, Some((input_w, input_h)), w, h)
        })
    } else {
        None
    };

    for (i, c) in rgb.chunks_exact(3).enumerate() {
        if let Some(rect) = crop_in_preview {
            let x = (i % width) as u32;
            let y = (i / width) as u32;
            let x0 = rect.x.min(w.saturating_sub(1));
            let y0 = rect.y.min(h.saturating_sub(1));
            let x1 = (rect.x + rect.width).min(w).max(x0 + 1);
            let y1 = (rect.y + rect.height).min(h).max(y0 + 1);
            if x < x0 || x >= x1 || y < y0 || y >= y1 {
                continue;
            }
        }
        r_hist[c[0] as usize] += 1;
        g_hist[c[1] as usize] += 1;
        b_hist[c[2] as usize] += 1;
    }

    (r_hist, g_hist, b_hist)
}

fn parse_decimal_f64(input: &str) -> Option<f64> {
    let normalized = input.trim().replace(',', ".");
    normalized.parse::<f64>().ok()
}

fn dmin_values_to_clipboard_text(opts: &PipelineOptions) -> Option<String> {
    if let Some((r, g, b)) = opts.dmin_fixed {
        return Some(format!(
            "dmin:fixed:{:.6},{:.6},{:.6};neutral_only={}",
            r,
            g,
            b,
            if opts.dmin_neutral_only { 1 } else { 0 }
        ));
    }
    if let Some(rect) = opts.dmin_rect {
        return Some(format!(
            "dmin:rect:{},{},{},{};neutral_only={}",
            rect.x,
            rect.y,
            rect.width,
            rect.height,
            if opts.dmin_neutral_only { 1 } else { 0 }
        ));
    }
    None
}

fn parse_dmin_clipboard_text(text: &str) -> Option<(Option<(f32, f32, f32)>, Option<Rect>, Option<bool>)> {
    let t = text.trim();
    if t.is_empty() {
        return None;
    }

    // Parse optional `neutral_only` suffix.
    let mut neutral_only: Option<bool> = None;
    if let Some(idx) = t.find(';') {
        let suffix = t[idx + 1..].trim();
        if let Some(v) = suffix.strip_prefix("neutral_only=") {
            neutral_only = match v.trim() {
                "1" | "true" | "True" | "TRUE" => Some(true),
                "0" | "false" | "False" | "FALSE" => Some(false),
                _ => None,
            };
        }
    }

    // Preferred tagged formats copied by this app:
    // dmin:fixed:r,g,b[;neutral_only=0|1]
    // dmin:rect:x,y,w,h[;neutral_only=0|1]
    let main = t.split(';').next().unwrap_or(t).trim();
    if let Some(values) = main.strip_prefix("dmin:fixed:") {
        let nums: Vec<&str> = values.split(',').map(|s| s.trim()).collect();
        if nums.len() == 3 {
            let r = nums[0].parse::<f32>().ok()?;
            let g = nums[1].parse::<f32>().ok()?;
            let b = nums[2].parse::<f32>().ok()?;
            if r.is_finite()
                && g.is_finite()
                && b.is_finite()
                && (0.0..=1.0).contains(&r)
                && (0.0..=1.0).contains(&g)
                && (0.0..=1.0).contains(&b)
            {
                return Some((Some((r, g, b)), None, neutral_only));
            }
        }
        return None;
    }
    if let Some(values) = main.strip_prefix("dmin:rect:") {
        let nums: Vec<&str> = values.split(',').map(|s| s.trim()).collect();
        if nums.len() == 4 {
            let x = nums[0].parse::<u32>().ok()?;
            let y = nums[1].parse::<u32>().ok()?;
            let w = nums[2].parse::<u32>().ok()?;
            let h = nums[3].parse::<u32>().ok()?;
            if w > 0 && h > 0 {
                return Some((
                    None,
                    Some(Rect {
                        x,
                        y,
                        width: w,
                        height: h,
                    }),
                    neutral_only,
                ));
            }
        }
        return None;
    }

    // Back-compat convenience: plain "r,g,b"
    let csv: Vec<&str> = t.split(',').map(|s| s.trim()).collect();
    if csv.len() == 3 {
        let r = csv[0].parse::<f32>().ok()?;
        let g = csv[1].parse::<f32>().ok()?;
        let b = csv[2].parse::<f32>().ok()?;
        if r.is_finite()
            && g.is_finite()
            && b.is_finite()
            && (0.0..=1.0).contains(&r)
            && (0.0..=1.0).contains(&g)
            && (0.0..=1.0).contains(&b)
        {
            return Some((Some((r, g, b)), None, neutral_only));
        }
    }

    None
}

fn drag_decimal_f32<'a>(value: &'a mut f32) -> egui::DragValue<'a> {
    egui::DragValue::new(value).custom_parser(|s| parse_decimal_f64(s))
}

/// High-level exposure wrapper, inspired by negPy. This re-parameterizes
/// existing curve/levels controls into more photographic terms.
#[derive(Debug, Clone, Copy)]
struct ExposureParams {
    /// Overall print exposure. 1.0 = neutral (curve_offset = 0).
    pub density: f32,
    /// Paper grade / contrast in normalized units. 1.0 ≈ default curve_gamma.
    pub grade: f32,
    /// Zone-masked shadow density offset. Positive = brighten shadows.
    pub shadows: f32,
    /// Zone-masked highlight density offset. Positive = brighten highlights.
    pub highlights: f32,
    /// Midtone hardness (maps to lut_in_mid / effective contrast around mid-gray).
    pub hardness: f32,
}

fn exposure_from_opts(opts: &PipelineOptions) -> ExposureParams {
    // Inverse of `apply_exposure_to_opts` mapping so the sliders reflect
    // whatever the user last set via either Exposure or raw Levels/curve.
    let density = 1.0 + opts.curve_offset;
    let grade = if opts.curve_gamma > 0.0 {
        opts.curve_gamma / 2.5
    } else {
        1.0
    };

    ExposureParams {
        density,
        grade,
        shadows: opts.zone_shadows,
        highlights: opts.zone_highlights,
        // Expose hardness as a delta around neutral (lut_in_mid = 1.0).
        hardness: opts.lut_in_mid - 1.0,
    }
}

fn apply_exposure_to_opts(exp: &ExposureParams, opts: &mut PipelineOptions) {
    // Clamp user-facing slider domain to a sane photographic range.
    let density = exp.density.clamp(0.0, 2.0);
    let grade = exp.grade.clamp(0.2, 3.0);
    let shadows = exp.shadows.clamp(-0.3, 0.3);
    let highlights = exp.highlights.clamp(-0.5, 0.5);
    let hardness = exp.hardness.clamp(-0.5, 0.5);

    opts.curve_offset = density - 1.0;
    opts.curve_gamma = 2.5 * grade;
    // Map hardness delta back to lut_in_mid around neutral = 1.0.
    opts.lut_in_mid = 1.0 + hardness;

    opts.zone_shadows = shadows;
    opts.zone_highlights = highlights;
}

/// CMY-style print balance wrapper for FilmPrint per-channel offsets.
#[derive(Debug, Clone, Copy)]
struct PrintBalance {
    pub cyan: f32,
    pub magenta: f32,
    pub yellow: f32,
}

fn print_balance_from_opts(opts: &PipelineOptions) -> PrintBalance {
    const SCALE: f32 = 0.3;
    PrintBalance {
        // Cyan is the complement of red: positive C reduces red channel offset.
        cyan: (-opts.fp_offset_r) / SCALE,
        // Magenta is the complement of green.
        magenta: (-opts.fp_offset_g) / SCALE,
        // Yellow is the complement of blue.
        yellow: (-opts.fp_offset_b) / SCALE,
    }
}

fn apply_print_balance_to_opts(pb: &PrintBalance, opts: &mut PipelineOptions) {
    const SCALE: f32 = 0.3;
    let c = pb.cyan.clamp(-1.0, 1.0);
    let m = pb.magenta.clamp(-1.0, 1.0);
    let y = pb.yellow.clamp(-1.0, 1.0);

    opts.fp_offset_r = -c * SCALE;
    opts.fp_offset_g = -m * SCALE;
    opts.fp_offset_b = -y * SCALE;
}

fn icon_candidate_paths(path_hint: &str) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let hint_path = PathBuf::from(path_hint);
    let file_name = hint_path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| path_hint.to_string());

    if hint_path.is_absolute() {
        candidates.push(hint_path.clone());
    }

    // Relative to current process working directory.
    candidates.push(PathBuf::from(path_hint));
    candidates.push(PathBuf::from("assets").join(&file_name));
    candidates.push(PathBuf::from("src").join("img").join(&file_name));

    // Relative to crate root (works in both dev + packaged runs).
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    candidates.push(crate_root.join(path_hint));
    candidates.push(crate_root.join("assets").join(&file_name));
    candidates.push(crate_root.join("src").join("img").join(&file_name));

    // Relative to executable directory.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            candidates.push(exe_dir.join(path_hint));
            candidates.push(exe_dir.join("assets").join(&file_name));
            candidates.push(exe_dir.join("src").join("img").join(&file_name));
        }
    }

    candidates
}

fn load_icon_texture(ctx: &egui::Context, texture_name: &str, path_hint: &str) -> Option<egui::TextureHandle> {
    let image = icon_candidate_paths(path_hint)
        .into_iter()
        .find_map(|candidate| image::open(candidate).ok())?
        .to_rgba8();
    let size = [image.width() as usize, image.height() as usize];
    let pixels = image.into_vec();
    let color_image = egui::ColorImage::from_rgba_unmultiplied(size, &pixels);
    Some(ctx.load_texture(texture_name.to_string(), color_image, egui::TextureOptions::default()))
}

fn make_thumbnail_from_rgb(rgb: &[u8], src_w: u32, src_h: u32, max_side: u32) -> Option<egui::ColorImage> {
    if src_w == 0 || src_h == 0 || max_side == 0 {
        return None;
    }
    if rgb.len() != (src_w as usize) * (src_h as usize) * 3 {
        return None;
    }

    let scale = (max_side as f32 / src_w as f32)
        .min(max_side as f32 / src_h as f32)
        .min(1.0);
    let dst_w = ((src_w as f32 * scale).round().max(1.0)) as usize;
    let dst_h = ((src_h as f32 * scale).round().max(1.0)) as usize;

    let mut pixels = Vec::with_capacity(dst_w * dst_h);
    for y in 0..dst_h {
        let sy = ((y as f32 + 0.5) * src_h as f32 / dst_h as f32)
            .floor()
            .clamp(0.0, src_h.saturating_sub(1) as f32) as usize;
        for x in 0..dst_w {
            let sx = ((x as f32 + 0.5) * src_w as f32 / dst_w as f32)
                .floor()
                .clamp(0.0, src_w.saturating_sub(1) as f32) as usize;
            let i = (sy * src_w as usize + sx) * 3;
            pixels.push(egui::Color32::from_rgb(rgb[i], rgb[i + 1], rgb[i + 2]));
        }
    }

    Some(egui::ColorImage {
        size: [dst_w, dst_h],
        pixels,
    })
}

impl C41Gui {
    fn request_preview_for(&mut self, index: usize, ctx: &egui::Context) {
        if index >= self.images.len() {
            return;
        }
        let path = self.images[index].path.clone();
        let entry = &mut self.images[index];

        // Ensure we have cached sensor data for this image.
        if entry.cached_sensor.is_none() {
            match load_sensor_from_path(&path) {
                Ok(sensor) => {
                    entry.cached_sensor = Some(Arc::new(sensor));
                }
                Err(e) => {
                    self.status = format!("Failed to load sensor data: {}", e);
                    return;
                }
            }
        }

        let mut options = entry.options.clone();
        options.flat_field_path = self.flat_field_path.clone();

        // If D-min uses a rectangle, recompute fixed medians from full-res sensor.
        if options.dmin_mode == DminMode::SampleRegion {
            if let (Some(rect), Some(sensor)) = (options.dmin_rect, entry.cached_sensor.as_ref()) {
                if let Ok((r, g, b)) = compute_dmin_from_sensor(
                    sensor.as_ref(),
                    rect,
                    options.dmin_rect_reference_size,
                    options.rotation_degrees,
                    options.flip_horizontal,
                    options.flip_vertical,
                    options.dmin_neutral_only,
                ) {
                    options.dmin_fixed = Some((r, g, b));
                }
            }
        }

        let capture_debug = self.capture_pipeline_debug_next;
        self.capture_pipeline_debug_next = false;
        options.verbose_debug = capture_debug;

        // Full-res preview: use export pipeline at full resolution (no downsampling).
        let (max_width, max_height) = if self.full_res_preview_active {
            (u32::MAX, u32::MAX)
        } else {
            // Adaptive preview resolution based on zoom: base = screen resolution (canvas × DPI),
            // so initial load is sharp; higher zoom → higher working resolution up to 4× cap.
            let zoom = entry.preview_zoom.max(1.0);
            let scale = zoom.min(4.0); // up to 4× base preview resolution
            let ppp = ctx.pixels_per_point();
            let (base_w, base_h) = self
                .preview_canvas_size
                .map(|(w, h)| {
                    // Canvas size in physical pixels for crisp display at 1:1
                    let pw = (w * ppp).round().max(PREVIEW_MAX_WIDTH as f32);
                    let ph = (h * ppp).round().max(PREVIEW_MAX_HEIGHT as f32);
                    (pw, ph)
                })
                .unwrap_or((PREVIEW_MAX_WIDTH as f32, PREVIEW_MAX_HEIGHT as f32));
            let max_w = (base_w * scale).round() as u32;
            let max_h = (base_h * scale).round() as u32;
            (max_w, max_h)
        };

        let cache = entry.preview_step_cache.clone();
        #[cfg(feature = "gpu")]
        let gpu_arc = self.gpu_pipeline.clone();
        let (tx, rx) = mpsc::channel();
        self.preview_receiver = Some(rx);
        self.preview_started_at = Some(Instant::now());
        thread::spawn(move || {
            #[cfg(feature = "gpu")]
            let res = process_one_to_preview_with_cache_gpu(
                &path,
                &options,
                max_width,
                max_height,
                cache.as_ref(),
                capture_debug,
                gpu_arc.as_deref(),
            );
            #[cfg(not(feature = "gpu"))]
            let res = c41_raw_tool::process_one_to_preview_with_cache(
                &path,
                &options,
                max_width,
                max_height,
                cache.as_ref(),
                capture_debug,
            );
            let res = res
            .map(|(input_w, input_h, w, h, rgb, dbg_log, new_cache)| {
                (index, input_w, input_h, w, h, rgb, dbg_log, capture_debug, new_cache)
            });
            let _ = tx.send(res);
        });
        ctx.request_repaint();
    }
}

impl eframe::App for C41Gui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Apply dark theme every frame so it sticks (some backends reset after creation)
        let mut style = (*ctx.style()).clone();
        style.visuals = egui::Visuals::dark();
        style.visuals.window_fill = egui::Color32::from_gray(35);
        style.visuals.panel_fill = egui::Color32::from_gray(30);
        style.visuals.override_text_color = Some(egui::Color32::from_gray(240));
        style.visuals.selection.bg_fill = egui::Color32::from_gray(70); // selected tabs/items: gray instead of blue
        style.spacing.button_padding = egui::vec2(10.0, 4.0); // extra left/right and top/bottom around button text
        ctx.set_style(style);

        // Global shortcut: Ctrl+Shift+D toggles Debug mode visibility.
        let debug_shortcut =
            ctx.input(|i| i.modifiers.ctrl && i.modifiers.shift && i.key_pressed(egui::Key::D));
        if debug_shortcut {
            self.mode = if self.mode == UIMode::Debug {
                UIMode::Process
            } else {
                UIMode::Debug
            };
        }

        self.rect_dragging = false;

        if self.ui_icons.close.is_none() {
            self.ui_icons.close = load_icon_texture(ctx, "ui_icon_close", ICON_CLOSE_PATH);
        }
        if self.ui_icons.rotate_left.is_none() {
            self.ui_icons.rotate_left = load_icon_texture(ctx, "ui_icon_rotate_left", ICON_ROTATE_LEFT_PATH);
        }
        if self.ui_icons.rotate_right.is_none() {
            self.ui_icons.rotate_right = load_icon_texture(ctx, "ui_icon_rotate_right", ICON_ROTATE_RIGHT_PATH);
        }
        if self.ui_icons.logo.is_none() {
            self.ui_icons.logo = load_icon_texture(ctx, "ui_icon_logo", ICON_LOGO_PATH);
        }

        // Deferred output-LUT file dialog (runs outside the egui panel render loop
        // to avoid macOS NSOpenPanel re-entrance crashes on repeated opens).
        if self.pending_output_lut_browse {
            self.pending_output_lut_browse = false;
            let picked = rfd::FileDialog::new()
                .add_filter("3D LUT (.cube)", &["cube", "CUBE"])
                .add_filter("All files", &["*"])
                .pick_file();
            if let Some(idx) = self.selected_index {
                if idx < self.images.len() {
                    let opts = &mut self.images[idx].options;
                    match picked {
                        Some(path) => match c41_raw_tool::lut3d::read_cube(&path) {
                            Ok(lut) => {
                                self.status = format!(
                                    "Loaded output LUT: {} ({}³ grid)",
                                    path.display(),
                                    lut.size,
                                );
                                opts.output_lut_cube = Some(path);
                            }
                            Err(e) => {
                                self.status = format!(
                                    "Failed to parse output LUT {}: {}",
                                    path.display(),
                                    e
                                );
                            }
                        },
                        None => {
                            self.status = "Output LUT: file dialog cancelled.".into();
                        }
                    }
                }
            }
        }

        // Poll preview worker
        if let Some(rx) = self.preview_receiver.as_ref() {
            match rx.try_recv() {
                Ok(Ok((idx, input_w, input_h, w, h, rgb, dbg_log, captured_debug, new_cache))) => {
                    self.preview_receiver = None;
                    self.preview_started_at = None;
                    if idx < self.images.len() {
                        let size = [w as usize, h as usize];
                        let mut pixels: Vec<egui::Color32> = Vec::with_capacity(size[0] * size[1]);
                        for c in rgb.chunks_exact(3) {
                            pixels.push(egui::Color32::from_rgb(c[0], c[1], c[2]));
                        }
                        let _image = egui::ColorImage { size, pixels };
                        // Histogram is computed in the background worker using a fixed-resolution
                        // preview and the current crop; here we only update the preview buffers.
                        // Preview hash also incorporates zoom so changing zoom triggers re-render.
                        let base_hash =
                            options_hash_for(&self.images[idx].path, &self.images[idx].options);
                        let mut hh = DefaultHasher::new();
                        base_hash.hash(&mut hh);
                        self.images[idx].preview_zoom.to_bits().hash(&mut hh);
                        let hash = hh.finish();
                        self.images[idx].preview_texture = None;
                        self.images[idx].preview_hash = hash;
                        self.images[idx].preview_full_rgb = Some((w, h, rgb.clone()));
                        self.images[idx].preview_step_cache = Some(new_cache);
                        // preview_input_size = preview working dims (crop/dmin coord reference).
                        self.images[idx].preview_input_size = Some([w, h]);
                        // raw_source_size = true sensor/file dims (info bar display only).
                        self.images[idx].raw_source_size = Some([input_w, input_h]);
                        if let Some(thumb_image) = make_thumbnail_from_rgb(&rgb, w, h, THUMB_MAX_SIZE) {
                            let thumb_tex = ctx.load_texture(
                                format!(
                                    "thumb_{}",
                                    self.images[idx]
                                        .path
                                        .display()
                                        .to_string()
                                        .replace('\\', "_")
                                        .replace('/', "_")
                                ),
                                thumb_image,
                                egui::TextureOptions::default(),
                            );
                            self.images[idx].thumbnail_texture = Some(thumb_tex);
                        }
                        // Do not overwrite dmin/crop reference sizes here; they stay in the
                        // coordinate space where the user last edited them.

                        // Compute histogram inline from preview RGB (< 1ms).
                        {
                            let opts = &self.images[idx].options;
                            let hist = compute_histogram_from_rgb(
                                &rgb, w, h, opts, input_w, input_h,
                            );
                            self.images[idx].histogram = Some(hist);
                        }
                        if captured_debug {
                            self.images[idx].pipeline_debug_log = Some(dbg_log);
                        }
                    }
                }
                Ok(Err(e)) => {
                    self.preview_receiver = None;
                    self.preview_started_at = None;
                    self.status = format!("Preview error: {}", e);
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.preview_receiver = None;
                    self.preview_started_at = None;
                }
            }
        }


        // Request thumbnail for one image at a time (strip icons).
        if self.thumbnail_receiver.is_none() {
            if let Some(entry) = self
                .images
                .iter()
                .find(|e| e.thumbnail_texture.is_none() && !self.thumbnail_pending.contains(&e.path))
            {
                let path = entry.path.clone();
                let mut options = entry.options.clone();
                options.flat_field_path = self.flat_field_path.clone();
                let (tx, rx) = mpsc::channel();
                self.thumbnail_receiver = Some(rx);
                self.thumbnail_pending.insert(path.clone());
                thread::spawn(move || {
                    let result = process_one_to_preview(
                        &path,
                        &options,
                        THUMB_MAX_SIZE,
                        THUMB_MAX_SIZE,
                    )
                    .map(|(_orig_w, _orig_h, new_w, new_h, rgb, _dbg)| (new_w, new_h, rgb));
                    let _ = tx.send((path, result));
                });
            }
        }
        if let Some(rx) = self.thumbnail_receiver.as_ref() {
            match rx.try_recv() {
                Ok((path, Ok((w, h, rgb)))) => {
                    self.thumbnail_receiver = None;
                    self.thumbnail_pending.remove(&path);
                    if let Some(entry) = self.images.iter_mut().find(|e| e.path == path) {
                        let pixels: Vec<egui::Color32> = rgb
                            .chunks_exact(3)
                            .map(|c| egui::Color32::from_rgb(c[0], c[1], c[2]))
                            .collect();
                        let image = egui::ColorImage {
                            size: [w as usize, h as usize],
                            pixels,
                        };
                        let tex = ctx.load_texture(
                            format!("thumb_{}", path.display().to_string().replace('\\', "_").replace('/', "_")),
                            image,
                            egui::TextureOptions::default(),
                        );
                        entry.thumbnail_texture = Some(tex);
                    }
                }
                Ok((path, Err(_))) => {
                    self.thumbnail_receiver = None;
                    self.thumbnail_pending.remove(&path);
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.thumbnail_receiver = None;
                }
            }
        }

        // ---- Bottom panel: image strip + global output / convert ----
        egui::TopBottomPanel::bottom("bottom_panel")
            .min_height(BOTTOM_PANEL_HEIGHT)
            .show(ctx, |ui| {
                ui.vertical(|ui| {
                    ui.add_space(14.0);
                    ui.horizontal(|ui| {
                        if ui.button("Add image…").clicked() {
                            if let Some(paths) = rfd::FileDialog::new()
                                .add_filter(
                                    "RAW & images",
                                    &[
                                        "arw", "nef", "nrw", "cr2", "cr3", "crw", "dng", "raf", "orf", "rw2",
                                        "png", "jpeg", "jpg", "tiff", "tif",
                                    ],
                                )
                                .pick_files()
                            {
                                for p in paths {
                                    if !self.images.iter().any(|e| e.path == p) {
                                        self.images.push(ImageEntry {
                                            path: p.clone(),
                                            options: default_options(),
                                            preview_texture: None,
                                            preview_hash: 0,
                                            preview_full_rgb: None,
                                            preview_input_size: None,
                                            raw_source_size: None,
                                            preview_zoom: 1.0,
                                            preview_pan: egui::vec2(0.5, 0.5),
                                            thumbnail_texture: None,
                                            histogram: None,
                                            export_format: ExportFormat::Tiff16,
                                            raw_debug_report: None,
                                            pipeline_debug_log: None,
                                            cached_sensor: None,
                                            preview_step_cache: None,
                                            process_tab: ProcessTab::Input,
                                        });
                                        if self.selected_index.is_none() {
                                            self.selected_index = Some(self.images.len() - 1);
                                            self.full_res_preview_active = false;
                                        }
                                    }
                                }
                                self.status = format!("{} file(s)", self.images.len());
                            }
                        }
                    });

                    ui.add_space(10.0);

                    let mut to_remove = Vec::new();
                    const CARD_WIDTH: f32 = 88.0;
                    const CARD_HEIGHT: f32 = 96.0; // more space in bottom panel
                    const THUMB_SIZE: f32 = 44.0;
                    const NAME_MAX_CHARS: usize = 10;
                    const X_BUTTON_SIZE: f32 = 22.0;
                    const CARD_PADDING: f32 = 4.0;
                    const CARD_GAP: f32 = 8.0;

                    egui::ScrollArea::horizontal().show(ui, |ui| {
                        ui.horizontal(|ui| {
                            for (i, entry) in self.images.iter().enumerate() {
                                let name = entry
                                    .path
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or("?");
                                let display_name = if name.chars().count() > NAME_MAX_CHARS {
                                    format!(
                                        "{}...",
                                        name.chars().take(NAME_MAX_CHARS).collect::<String>()
                                    )
                                } else {
                                    name.to_string()
                                };
                                let selected = self.selected_index == Some(i);

                                let card_response = ui.allocate_ui(
                                    egui::vec2(CARD_WIDTH, CARD_HEIGHT),
                                    |ui| {
                                        let card_rect = ui.available_rect_before_wrap();
                                        let id = ui.make_persistent_id(("strip_card", i));
                                        let interact_resp =
                                            ui.interact(card_rect, id, egui::Sense::click());

                                        // Card background and border (drawn first so content is on top)
                                        let stroke = if selected {
                                            egui::Stroke::new(2.0, egui::Color32::from_rgb(100, 150, 255))
                                        } else if interact_resp.hovered() {
                                            egui::Stroke::new(1.0, egui::Color32::from_gray(120))
                                        } else {
                                            egui::Stroke::new(1.0, egui::Color32::from_gray(70))
                                        };
                                        ui.painter().rect_filled(
                                            card_rect,
                                            4.0,
                                            egui::Color32::from_gray(45),
                                        );
                                        ui.painter().rect_stroke(card_rect, 4.0, stroke);

                                        // Close button fixed in top-right corner of card
                                        let x_rect = egui::Rect::from_min_size(
                                            egui::pos2(
                                                card_rect.right() - X_BUTTON_SIZE - CARD_PADDING,
                                                card_rect.top() + CARD_PADDING,
                                            ),
                                            egui::vec2(X_BUTTON_SIZE, X_BUTTON_SIZE),
                                        );
                                        let x_clicked = ui
                                            .allocate_new_ui(egui::UiBuilder::new().max_rect(x_rect), |ui| {
                                                if let Some(icon) = &self.ui_icons.close {
                                                    ui.add(
                                                        egui::ImageButton::new((
                                                            icon.id(),
                                                            egui::vec2(X_BUTTON_SIZE - 6.0, X_BUTTON_SIZE - 6.0),
                                                        ))
                                                        .frame(false),
                                                    )
                                                    .clicked()
                                                } else {
                                                    ui.small_button("X").clicked()
                                                }
                                            })
                                            .inner;

                                        // Content area: thumbnail + name, clipped to card (below X row)
                                        let content_top = card_rect.top() + CARD_PADDING + X_BUTTON_SIZE + 2.0;
                                        let content_rect = egui::Rect::from_min_max(
                                            egui::pos2(card_rect.left() + CARD_PADDING, content_top),
                                            egui::pos2(card_rect.right() - CARD_PADDING, card_rect.bottom() - CARD_PADDING),
                                        );
                                        ui.allocate_new_ui(egui::UiBuilder::new().max_rect(content_rect), |ui| {
                                            ui.set_clip_rect(card_rect);
                                            ui.vertical_centered(|ui| {
                                                if let Some(ref thumb) = entry.thumbnail_texture {
                                                    let size = thumb.size();
                                                    let (w, h) = (size[0] as f32, size[1] as f32);
                                                    let scale =
                                                        (THUMB_SIZE / w).min(THUMB_SIZE / h).min(1.0);
                                                    ui.image((thumb.id(), egui::vec2(w * scale, h * scale)));
                                                } else {
                                                    ui.allocate_space(egui::vec2(THUMB_SIZE, THUMB_SIZE));
                                                }
                                                ui.add_space(2.0);
                                                ui.label(
                                                    egui::RichText::new(&display_name).small(),
                                                )
                                                .on_hover_text(entry.path.display().to_string());
                                            });
                                        });

                                        (interact_resp, x_clicked)
                                    },
                                );

                                let (interact_resp, x_clicked) = card_response.inner;

                                if x_clicked {
                                    to_remove.push(i);
                                } else if interact_resp.clicked() {
                                    self.selected_index = Some(i);
                                    self.full_res_preview_active = false;
                                }

                                if i + 1 < self.images.len() {
                                    ui.add_space(CARD_GAP);
                                }
                            }
                        });
                    });
                    ui.add_space(10.0);
                    let had_removals = !to_remove.is_empty();
                    if had_removals {
                        self.preview_receiver = None;
                        self.full_res_preview_active = false;
                        for &i in &to_remove {
                            if let Some(e) = self.images.get(i) {
                                self.thumbnail_pending.remove(&e.path);
                            }
                        }
                    }
                    for i in to_remove.into_iter().rev() {
                        self.images.remove(i);
                        if self.selected_index == Some(i) {
                            self.selected_index = None;
                        } else if self.selected_index.map(|s| s > i).unwrap_or(false) {
                            self.selected_index = self.selected_index.map(|s| s - 1);
                        }
                    }
                    if had_removals {
                        self.status = format!("{} file(s)", self.images.len());
                    }
                });
            });

        // ---- Right panel: mode toggle + per-image settings / calibration ----
        let mut auto_crop_requested = false;
        egui::SidePanel::right("settings_panel")
            .resizable(false)
            .exact_width(RIGHT_PANEL_WIDTH)
            .show(ctx, |ui| {
                ui.add_space(16.0);
                ui.horizontal(|ui| {
                    ui.add_space(10.0);
                    ui.selectable_value(&mut self.mode, UIMode::Process, "Process");
                    ui.selectable_value(&mut self.mode, UIMode::Calibrate, "Color calibration");
                    ui.selectable_value(
                        &mut self.mode,
                        UIMode::LuminanceCalibrate,
                        "Capture flat field",
                    );
                    ui.selectable_value(&mut self.mode, UIMode::Debug, "Debug");
                    ui.add_space(10.0);
                });
                ui.add_space(10.0);
                // Full-width divider: draw line across entire panel (no side margin)
                let sep_y = ui.cursor().top() + 1.0;
                ui.painter().hline(
                    ui.clip_rect().x_range(),
                    sep_y,
                    egui::Stroke::new(1.0, ui.visuals().window_stroke.color),
                );
                ui.allocate_space(egui::vec2(ui.available_width(), 4.0));
                ui.add_space(8.0);

                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.add_space(14.0);
                        ui.vertical(|ui| {
                match self.mode {
                    UIMode::Process => {
                        ui.heading("Image Settings");
                    }
                    UIMode::Calibrate => {
                        ui.heading("Color calibration");
                    }
                    UIMode::LuminanceCalibrate => {
                        ui.heading("Capture flat field");
                    }
                    UIMode::Debug => {
                        ui.heading("Debug");
                    }
                }
                ui.add_space(10.0);

                let Some(idx) = self.selected_index else {
                    ui.label("No image selected.");
                    if !self.status.is_empty() {
                        ui.add_space(4.0);
                        ui.label(egui::RichText::new(&self.status).small());
                    }
                    return;
                };

                if idx >= self.images.len() {
                    ui.label("No image selected.");
                    return;
                }

                let entry = &mut self.images[idx];
                // Snapshot of options for calibration tap (avoids borrow issues).
                let calibration_opts_snapshot = entry.options.clone();
                let opts = &mut entry.options;

                ui.label(
                    egui::RichText::new(
                        entry
                            .path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("image"),
                    )
                    .strong(),
                );
                ui.add_space(8.0);

                if self.mode == UIMode::Process {
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut entry.process_tab, ProcessTab::Input, "Input");
                        ui.selectable_value(&mut entry.process_tab, ProcessTab::Develop, "Develop");
                        ui.selectable_value(&mut entry.process_tab, ProcessTab::Export, "Export");
                    });
                    ui.add_space(6.0);
                    ui.separator();
                    ui.add_space(4.0);
                }

                if self.mode == UIMode::Debug {
                    ui.add(egui::Slider::new(&mut opts.debug_pipeline_step, 1..=6).text("Step"))
                        .on_hover_text("Pipeline step (1–6). Preview shows output up to this step.");
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if ui
                            .button(if opts.debug_preview_simple_debayer {
                                "Use full pipeline preview"
                            } else {
                                "Use simple debayer preview"
                            })
                            .clicked()
                        {
                            opts.debug_preview_simple_debayer = !opts.debug_preview_simple_debayer;
                        }
                        if opts.debug_preview_simple_debayer {
                            ui.label(egui::RichText::new("(simple debayer)").small().weak())
                                .on_hover_text("simple RAW bilinear debayer mode ON");
                        }
                    });
                    #[cfg(feature = "gpu")]
                    {
                        ui.add_space(6.0);
                        let gpu_available = self.gpu_pipeline.is_some();
                        ui.horizontal(|ui| {
                            let mut use_gpu = opts.use_gpu;
                            ui.add_enabled(gpu_available, egui::Checkbox::new(&mut use_gpu, "GPU acceleration"));
                            opts.use_gpu = use_gpu;
                            if gpu_available {
                                ui.label(egui::RichText::new("(available)").small().weak())
                                    .on_hover_text("GPU acceleration is available");
                            } else {
                                ui.label(egui::RichText::new("(no GPU adapter found)").small().weak())
                                    .on_hover_text("No GPU adapter found. Pipeline runs on CPU.");
                            }
                        });
                    }
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new("Rawloader report").strong())
                        .on_hover_text("1: load+demosaic+rot · 3: +D-min · 4: +WB · 6: full (curve/invert). Step 5 applies density matrix / LUT calibration.");
                    ui.separator();
                    ui.label(egui::RichText::new("Rawloader report").strong());
                    let ext = entry
                        .path
                        .extension()
                        .and_then(|e| e.to_str())
                        .map(|s| s.to_ascii_lowercase())
                        .unwrap_or_default();
                    let is_raw = matches!(
                        ext.as_str(),
                        "arw" | "nef" | "nrw" | "cr2" | "cr3" | "crw" | "dng" | "raf" | "orf" | "rw2"
                    );
                    if is_raw {
                        ui.horizontal(|ui| {
                            if ui.button("Run rawloader debug for selected file").clicked() {
                                match raw_reader::debug_raw_report(&entry.path) {
                                    Ok(report) => {
                                        entry.raw_debug_report = Some(report);
                                    }
                                    Err(e) => {
                                        entry.raw_debug_report = Some(format!(
                                            "Failed to decode raw file:\n{}",
                                            e
                                        ));
                                    }
                                }
                            }
                            if ui.button("Copy report").clicked() {
                                if let Some(report) = entry.raw_debug_report.as_ref() {
                                    ui.ctx().copy_text(report.clone());
                                }
                            }
                        });
                    } else {
                        ui.label("—")
                            .on_hover_text("Selected file is not a RAW format.");
                    }
                    ui.add_space(6.0);
                    if let Some(report) = entry.raw_debug_report.as_ref() {
                        egui::ScrollArea::vertical().max_height(520.0).show(ui, |ui| {
                            let mut report_text = report.clone();
                            ui.add(
                                egui::TextEdit::multiline(&mut report_text)
                                    .desired_width(f32::INFINITY)
                                    .font(egui::TextStyle::Monospace)
                                    .interactive(false),
                            );
                        });
                    } else {
                        ui.label("—")
                            .on_hover_text("No raw report yet. Click the button above.");
                    }

                    ui.add_space(8.0);
                    ui.separator();
                    ui.label(egui::RichText::new("Pipeline debug log").strong())
                        .on_hover_text("Captured on demand. Click the button to snapshot current settings.");
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        if ui
                            .button("Capture pipeline log for current settings")
                            .on_hover_text("Snapshot current settings into the pipeline debug log")
                            .clicked()
                        {
                            self.capture_pipeline_debug_next = true;
                            entry.preview_texture = None; // force one fresh render with debug capture
                        }
                        if let Some(ref log) = entry.pipeline_debug_log {
                            if ui.button("Copy pipeline log").clicked() {
                                ui.ctx().copy_text(log.clone());
                            }
                        }
                    });
                    ui.add_space(4.0);
                    if let Some(ref log) = entry.pipeline_debug_log {
                        egui::ScrollArea::vertical()
                            .id_salt("pipeline_debug_scroll")
                            .max_height(520.0)
                            .show(ui, |ui| {
                                let mut log_text = log.clone();
                                ui.add(
                                    egui::TextEdit::multiline(&mut log_text)
                                        .desired_width(f32::INFINITY)
                                        .font(egui::TextStyle::Monospace)
                                        .interactive(false),
                                );
                            });
                    } else {
                        ui.label("—")
                            .on_hover_text("No pipeline log yet. Preview must render first.");
                    }
                } else if self.mode != UIMode::LuminanceCalibrate {
                    let in_process = self.mode == UIMode::Process;

                  if !in_process || entry.process_tab == ProcessTab::Develop {
                    // ════════════════════════════════════════════════════════
                    // GROUP 1 — Exposure  (primary editing controls)
                    // ════════════════════════════════════════════════════════
                    ui.label(egui::RichText::new("Exposure").strong());
                    ui.add_space(4.0);
                    {
                        let mut exp = exposure_from_opts(opts);
                        let mut changed = false;
                        egui::Grid::new("exposure")
                            .num_columns(2)
                            .spacing([4.0, 2.0])
                            .show(ui, |ui| {
                                ui.label("Density");
                                changed |= ui
                                    .add(egui::Slider::new(&mut exp.density, 0.5..=1.5).fixed_decimals(2))
                                    .changed();
                                ui.end_row();
                                ui.label("Grade");
                                changed |= ui
                                    .add(egui::Slider::new(&mut exp.grade, 0.5..=2.0).fixed_decimals(2))
                                    .changed();
                                ui.end_row();
                                ui.label("Shadows");
                                changed |= ui
                                    .add(egui::Slider::new(&mut exp.shadows, -0.3..=0.3).fixed_decimals(3))
                                    .changed();
                                ui.end_row();
                                ui.label("Highlights");
                                changed |= ui
                                    .add(egui::Slider::new(&mut exp.highlights, -0.5..=0.5).fixed_decimals(3))
                                    .changed();
                                ui.end_row();
                                ui.label("Hardness");
                                changed |= ui
                                    .add(egui::Slider::new(&mut exp.hardness, -0.5..=0.5).fixed_decimals(3))
                                    .changed();
                                ui.end_row();
                            });
                        if changed {
                            apply_exposure_to_opts(&exp, opts);
                        }

                        if matches!(opts.output_stage, OutputStage::FilmPrint) {
                            ui.add_space(4.0);
                            ui.label("Print balance (CMY)")
                                .on_hover_text("Per-channel cyan, magenta, yellow adjustments for print balance");
                            let mut pb = print_balance_from_opts(opts);
                            let mut pb_changed = false;
                            ui.horizontal(|ui| {
                                ui.label("C");
                                pb_changed |= ui
                                    .add(
                                        egui::Slider::new(&mut pb.cyan, -0.5..=0.5)
                                            .fixed_decimals(2),
                                    )
                                    .changed();
                            });
                            ui.horizontal(|ui| {
                                ui.label("M");
                                pb_changed |= ui
                                    .add(
                                        egui::Slider::new(&mut pb.magenta, -0.5..=0.5)
                                            .fixed_decimals(2),
                                    )
                                    .changed();
                            });
                            ui.horizontal(|ui| {
                                ui.label("Y");
                                pb_changed |= ui
                                    .add(
                                        egui::Slider::new(&mut pb.yellow, -0.5..=0.5)
                                            .fixed_decimals(2),
                                    )
                                    .changed();
                            });
                            if pb_changed {
                                apply_print_balance_to_opts(&pb, opts);
                            }
                        }
                    }
                    ui.add_space(6.0);
                    ui.separator();

                    // ════════════════════════════════════════════════════════
                    // Highlight roll-off (Reinhard) — order matches workflow
                    // ════════════════════════════════════════════════════════
                    let cr_rolloff = ui.collapsing("Highlight roll-off", |ui| {
                        ui.add_space(4.0);
                        egui::Grid::new("highlight_rolloff")
                            .num_columns(2)
                            .spacing([4.0, 2.0])
                            .show(ui, |ui| {
                                ui.label("Strength");
                                ui.add(egui::Slider::new(&mut opts.highlight_rolloff, 0.0..=3.0).fixed_decimals(2));
                                ui.end_row();
                                ui.label("Knee");
                                ui.add(egui::Slider::new(&mut opts.highlight_rolloff_d_mid, 0.5..=3.0).fixed_decimals(2));
                                ui.end_row();
                            });
                        if ui.small_button("Reset").clicked() {
                            opts.highlight_rolloff = 0.0;
                            opts.highlight_rolloff_d_mid = 1.5;
                        }
                    });
                    cr_rolloff.header_response.on_hover_text("Reinhard-style compression in density space to mask noise in skies and dense negative areas.");

                    // ════════════════════════════════════════════════════════
                    // Tone shaping (advanced shadow/highlight)
                    // ════════════════════════════════════════════════════════
                    let cr_tone = ui.collapsing("Tone shaping", |ui| {
                        ui.add_space(4.0);
                        egui::Grid::new("tone_shaping")
                            .num_columns(2)
                            .spacing([4.0, 2.0])
                            .show(ui, |ui| {
                                ui.label("Toe");
                                ui.add(
                                    egui::Slider::new(&mut opts.toe_strength, -0.5..=0.5)
                                        .fixed_decimals(2),
                                );
                                ui.end_row();
                                ui.label("Shoulder");
                                ui.add(
                                    egui::Slider::new(&mut opts.shoulder_strength, -0.5..=0.5)
                                        .fixed_decimals(2),
                                );
                                ui.end_row();
                                ui.label("Shadow cast");
                                ui.add(
                                    egui::Slider::new(&mut opts.shadow_cast_strength, 0.0..=1.0)
                                        .fixed_decimals(2),
                                );
                                ui.end_row();
                            });
                    });
                    cr_tone.header_response.on_hover_text("Toe/shoulder: softer shadows and highlights. Shadow cast: auto-neutralize color cast in shadows.");

                    // ════════════════════════════════════════════════════════
                    // White balance & color neutrality
                    // ════════════════════════════════════════════════════════
                    ui.collapsing("White balance", |ui| {
                        let mut wb_mode = opts.wb_mode;
                        egui::ComboBox::from_id_salt(ui.id().with("wb_mode"))
                            .selected_text(match wb_mode {
                                WbMode::Auto => "Auto",
                                WbMode::Picker => "Picker",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut wb_mode, WbMode::Auto, "Auto");
                                ui.selectable_value(&mut wb_mode, WbMode::Picker, "Picker");
                            });
                        if wb_mode != opts.wb_mode {
                            opts.wb_mode = wb_mode;
                            if wb_mode == WbMode::Picker {
                                opts.auto_wb = false;
                                opts.apply_white_balance = true;
                            }
                        }

                        match opts.wb_mode {
                            WbMode::Auto => {
                                ui.checkbox(&mut opts.auto_wb, "Auto white balance")
                                    .on_hover_text(if opts.auto_wb {
                                        "Per-channel gamma: D × (mean_D / ch_median). Preserves black point."
                                    } else {
                                        "Auto WB disabled."
                                    });
                                ui.add_space(4.0);
                                ui.checkbox(&mut opts.apply_white_balance, "Manual white balance");
                            }
                            WbMode::Picker => {
                                ui.horizontal(|ui| {
                                    if ui.button("Pick white point").clicked() {
                                        self.wb_picker_pending = Some(WbPickerTarget::WhitePoint);
                                    }
                                    if ui.button("Pick gray point").clicked() {
                                        self.wb_picker_pending = Some(WbPickerTarget::GrayPoint);
                                    }
                                    if ui.button("Pick black point").clicked() {
                                        self.wb_picker_pending = Some(WbPickerTarget::BlackPoint);
                                    }
                                })
                                .response
                                .on_hover_text("Click a point on the preview to set it as neutral.");
                                if self.wb_picker_pending.is_some() {
                                    ui.label(egui::RichText::new("Click on the preview to sample.").small().color(egui::Color32::from_rgb(180, 220, 120)))
                                        .on_hover_text("Click on the preview image to sample.");
                                }
                            }
                        }

                        ui.add_space(4.0);
                        if opts.apply_white_balance {
                            let mut use_temp = opts.temp_k.is_some();
                            ui.checkbox(&mut use_temp, "Color temperature (K)");
                            if use_temp {
                                let mut k = opts.temp_k.unwrap_or(5500.0);
                                ui.add(egui::Slider::new(&mut k, 2500.0..=9000.0).suffix(" K"));
                                opts.temp_k = Some(k);
                            } else {
                                opts.temp_k = None;
                            }
                            egui::Grid::new("wb_rgb")
                                .num_columns(2)
                                .spacing([4.0, 2.0])
                                .show(ui, |ui| {
                                    ui.label("R")
                                        .on_hover_text("Density scale (1.0 = neutral, >1 = more color). Applies to R, G, B.");
                                    ui.add(egui::Slider::new(&mut opts.wb_r, 0.7..=1.5));
                                    ui.end_row();
                                    ui.label("G");
                                    ui.add(egui::Slider::new(&mut opts.wb_g, 0.7..=1.5));
                                    ui.end_row();
                                    ui.label("B");
                                    ui.add(egui::Slider::new(&mut opts.wb_b, 0.7..=1.5));
                                    ui.end_row();
                                });
                        }
                    });

                    // ════════════════════════════════════════════════════════
                    // Color character & separation
                    // ════════════════════════════════════════════════════════
                    let cr_color = ui.collapsing("Color", |ui| {
                        ui.add_space(4.0);
                        egui::Grid::new("color_main")
                            .num_columns(2)
                            .spacing([4.0, 2.0])
                            .show(ui, |ui| {
                                ui.label("Saturation");
                                ui.add(
                                    egui::Slider::new(&mut opts.saturation, 0.7..=1.6)
                                        .fixed_decimals(2),
                                );
                                ui.end_row();
                                ui.label("Warmth");
                                ui.add(
                                    egui::Slider::new(&mut opts.highlight_warmth, 0.0..=0.6)
                                        .fixed_decimals(2),
                                );
                                ui.end_row();
                            });
                        ui.checkbox(&mut opts.apply_lab, "Lab separation");
                        ui.add_enabled_ui(opts.apply_lab, |ui| {
                            egui::Grid::new("color_lab")
                                .num_columns(2)
                                .spacing([4.0, 2.0])
                                .show(ui, |ui| {
                                    ui.label("Separation");
                                    ui.add(
                                        egui::Slider::new(&mut opts.lab_separation, -1.5..=1.5)
                                            .fixed_decimals(2),
                                    );
                                    ui.end_row();
                                });
                        });
                        egui::Grid::new("color_skin_magenta")
                            .num_columns(2)
                            .spacing([4.0, 2.0])
                            .show(ui, |ui| {
                                ui.label("Skin magenta");
                                ui.add(
                                    egui::Slider::new(&mut opts.skin_magenta_shift, 0.0..=1.0)
                                        .fixed_decimals(2),
                                );
                                ui.end_row();
                            });
                    });
                    cr_color.header_response.on_hover_text("Saturation: density chroma. Warmth: golden highlights. Lab: mid-chroma separation in a/b. Skin magenta: rotates lips/eye magenta toward orange.");

                    // ════════════════════════════════════════════════════════
                    // Color zones (per-channel shadow/mid/highlight)
                    // ════════════════════════════════════════════════════════
                    let cr_zones = ui.collapsing("Color zones", |ui| {
                        ui.add_space(4.0);

                        ui.label(egui::RichText::new("Shadows").strong());
                        egui::Grid::new("color_zones_shadows")
                            .num_columns(2)
                            .spacing([4.0, 2.0])
                            .show(ui, |ui| {
                                ui.label("Gain");
                                ui.add(egui::Slider::new(&mut opts.zone_shadow_gain, -0.5..=0.5).fixed_decimals(3));
                                ui.end_row();
                                ui.label("Saturation");
                                ui.add(egui::Slider::new(&mut opts.zone_shadow_saturation, 0.5..=1.6).fixed_decimals(2));
                                ui.end_row();
                                ui.label("Gain R");
                                ui.add(egui::Slider::new(&mut opts.color_shadow_gain_r, -0.3..=0.3).fixed_decimals(3));
                                ui.end_row();
                                ui.label("Gain G");
                                ui.add(egui::Slider::new(&mut opts.color_shadow_gain_g, -0.3..=0.3).fixed_decimals(3));
                                ui.end_row();
                                ui.label("Gain B");
                                ui.add(egui::Slider::new(&mut opts.color_shadow_gain_b, -0.3..=0.3).fixed_decimals(3));
                                ui.end_row();
                            });

                        ui.add_space(4.0);
                        ui.label(egui::RichText::new("Midtones").strong());
                        egui::Grid::new("color_zones_mids")
                            .num_columns(2)
                            .spacing([4.0, 2.0])
                            .show(ui, |ui| {
                                ui.label("Gain");
                                ui.add(egui::Slider::new(&mut opts.zone_mid_gain, -0.5..=0.5).fixed_decimals(3));
                                ui.end_row();
                                ui.label("Saturation");
                                ui.add(egui::Slider::new(&mut opts.zone_mid_saturation, 0.5..=1.6).fixed_decimals(2));
                                ui.end_row();
                                ui.label("Gain R");
                                ui.add(egui::Slider::new(&mut opts.color_mid_gain_r, -0.3..=0.3).fixed_decimals(3));
                                ui.end_row();
                                ui.label("Gain G");
                                ui.add(egui::Slider::new(&mut opts.color_mid_gain_g, -0.3..=0.3).fixed_decimals(3));
                                ui.end_row();
                                ui.label("Gain B");
                                ui.add(egui::Slider::new(&mut opts.color_mid_gain_b, -0.3..=0.3).fixed_decimals(3));
                                ui.end_row();
                            });

                        ui.add_space(4.0);
                        ui.label(egui::RichText::new("Highlights").strong());
                        egui::Grid::new("color_zones_highlights")
                            .num_columns(2)
                            .spacing([4.0, 2.0])
                            .show(ui, |ui| {
                                ui.label("Gain");
                                ui.add(egui::Slider::new(&mut opts.zone_highlight_gain, -0.5..=0.5).fixed_decimals(3));
                                ui.end_row();
                                ui.label("Saturation");
                                ui.add(egui::Slider::new(&mut opts.zone_highlight_saturation, 0.5..=1.6).fixed_decimals(2));
                                ui.end_row();
                                ui.label("Gain R");
                                ui.add(egui::Slider::new(&mut opts.color_highlight_gain_r, -0.3..=0.3).fixed_decimals(3));
                                ui.end_row();
                                ui.label("Gain G");
                                ui.add(egui::Slider::new(&mut opts.color_highlight_gain_g, -0.3..=0.3).fixed_decimals(3));
                                ui.end_row();
                                ui.label("Gain B");
                                ui.add(egui::Slider::new(&mut opts.color_highlight_gain_b, -0.3..=0.3).fixed_decimals(3));
                                ui.end_row();
                            });

                        ui.add_space(6.0);
                        if ui.small_button("Reset all").clicked() {
                            opts.zone_shadow_gain = 0.0;
                            opts.zone_mid_gain = 0.0;
                            opts.zone_highlight_gain = 0.0;
                            opts.zone_shadow_saturation = 1.0;
                            opts.zone_mid_saturation = 1.0;
                            opts.zone_highlight_saturation = 1.0;
                            opts.color_shadow_gain_r = 0.0;
                            opts.color_shadow_gain_g = 0.0;
                            opts.color_shadow_gain_b = 0.0;
                            opts.color_mid_gain_r = 0.0;
                            opts.color_mid_gain_g = 0.0;
                            opts.color_mid_gain_b = 0.0;
                            opts.color_highlight_gain_r = 0.0;
                            opts.color_highlight_gain_g = 0.0;
                            opts.color_highlight_gain_b = 0.0;
                        }
                    });
                    cr_zones.header_response.on_hover_text("Gain = multiplicative. Saturation = 1.0 no change, <1 desaturate, >1 boost.");
                  } // end Develop tab guard (groups 1-4)

                  if !in_process || entry.process_tab == ProcessTab::Input {
                    // ════════════════════════════════════════════════════════
                    // Input — Crop
                    // ════════════════════════════════════════════════════════
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut opts.apply_crop, "Crop");
                            if opts.apply_crop {
                                if ui.button("Auto")
                                    .on_hover_text("Detect film frame boundaries and maximise crop")
                                    .clicked()
                                {
                                    auto_crop_requested = true;
                                }
                            }
                        });
                        if opts.apply_crop {
                            if opts.crop_rect.is_none() {
                                opts.crop_rect = Some(Rect {
                                    x: 40,
                                    y: 40,
                                    width: 240,
                                    height: 240,
                                });
                                auto_crop_requested = true;
                            }
                            if let Some(rect) = opts.crop_rect.as_mut() {
                                ui.horizontal(|ui| {
                                    ui.label("x,y,w,h")
                                        .on_hover_text("Preview darkens outside crop. Histogram + export use inside only.");
                                    ui.add(egui::DragValue::new(&mut rect.x).speed(1));
                                    ui.add(egui::DragValue::new(&mut rect.y).speed(1));
                                    ui.add(egui::DragValue::new(&mut rect.width).speed(1));
                                    ui.add(egui::DragValue::new(&mut rect.height).speed(1));
                                });
                            }
                        }
                        ui.add_space(4.0);
                        ui.separator();
                        ui.add_space(4.0);
                  }

                  if !in_process || entry.process_tab == ProcessTab::Input {
                    // ════════════════════════════════════════════════════════
                    // Input — Film γ
                    // ════════════════════════════════════════════════════════
                        ui.horizontal(|ui| {
                            ui.label("Film γ")
                                .on_hover_text("C-41 γ ≈ 0.55–0.75. Converts density → scene log-exposure.");
                            ui.add(egui::Slider::new(&mut opts.film_gamma, 0.4..=0.9));
                        });
                        ui.add_space(4.0);
                        ui.separator();
                        ui.add_space(4.0);
                  }

                  if !in_process || entry.process_tab == ProcessTab::Input {
                    // ════════════════════════════════════════════════════════
                    // Input — D-min, flat-field
                    // ════════════════════════════════════════════════════════
                        ui.label(egui::RichText::new("D-min").strong());
                        ui.add_space(2.0);

                        let dmin_label = match opts.dmin_mode {
                            DminMode::Off => "Off",
                            DminMode::Fixed => "Fixed values",
                            DminMode::SampleRegion => "Sample region",
                            DminMode::AutoPercentile => "Auto (percentile)",
                        };
                        egui::ComboBox::from_label("Mode")
                            .selected_text(dmin_label)
                            .show_ui(ui, |ui| {
                                if ui.selectable_label(opts.dmin_mode == DminMode::AutoPercentile, "Auto (percentile)").clicked() {
                                    opts.dmin_mode = DminMode::AutoPercentile;
                                }
                                if ui.selectable_label(opts.dmin_mode == DminMode::SampleRegion, "Sample region").clicked() {
                                    opts.dmin_mode = DminMode::SampleRegion;
                                    if opts.dmin_rect.is_none() {
                                        opts.dmin_rect = Some(Rect { x: 35, y: 15, width: 20, height: 20 });
                                    }
                                }
                                if ui.selectable_label(opts.dmin_mode == DminMode::Fixed, "Fixed values").clicked() {
                                    opts.dmin_mode = DminMode::Fixed;
                                    if opts.dmin_fixed.is_none() {
                                        opts.dmin_fixed = Some((0.222537, 0.108183, 0.054116));
                                    }
                                }
                                if ui.selectable_label(opts.dmin_mode == DminMode::Off, "Off").clicked() {
                                    opts.dmin_mode = DminMode::Off;
                                }
                            });
                        ui.add_space(4.0);

                        match opts.dmin_mode {
                            DminMode::Off => {
                                ui.label("—")
                                    .on_hover_text("D-min correction disabled.");
                            }
                            DminMode::Fixed => {
                                ui.horizontal(|ui| {
                                    if ui.button("Copy").clicked() {
                                        if let Some(text) = dmin_values_to_clipboard_text(opts) {
                                            ui.ctx().copy_text(text.clone());
                                            self.status = format!("D-min copied: {}", text);
                                        }
                                    }
                                    if ui
                                        .button("Paste")
                                        .on_hover_text("Format: dmin:fixed:r,g,b or dmin:rect:x,y,w,h")
                                        .clicked()
                                    {
                                        match arboard::Clipboard::new()
                                            .and_then(|mut cb| cb.get_text())
                                        {
                                            Ok(text) => {
                                                if let Some((fixed, rect, neutral_only)) =
                                                    parse_dmin_clipboard_text(&text)
                                                {
                                                    if let Some((r, g, b)) = fixed {
                                                        opts.dmin_mode = DminMode::Fixed;
                                                        opts.dmin_fixed = Some((r, g, b));
                                                        if let Some(v) = neutral_only {
                                                            opts.dmin_neutral_only = v;
                                                        }
                                                        self.status = format!(
                                                            "Applied D-min fixed: {:.6}, {:.6}, {:.6}",
                                                            r, g, b
                                                        );
                                                    } else if let Some(rect) = rect {
                                                        opts.dmin_mode = DminMode::SampleRegion;
                                                        opts.dmin_rect = Some(rect);
                                                        opts.dmin_rect_reference_size = None;
                                                        if let Some(v) = neutral_only {
                                                            opts.dmin_neutral_only = v;
                                                        }
                                                        self.status = format!(
                                                            "Applied D-min rect: {},{},{},{}",
                                                            rect.x, rect.y, rect.width, rect.height
                                                        );
                                                    } else {
                                                        self.status = "Clipboard has no valid D-min payload.".to_string();
                                                    }
                                                } else {
                                                    self.status = "Clipboard text is not a valid D-min value.".to_string();
                                                }
                                            }
                                            Err(e) => {
                                                self.status = format!("Could not read clipboard: {}", e);
                                            }
                                        }
                                    }
                                });
                                ui.add_space(4.0);
                                if opts.dmin_fixed.is_none() {
                                    opts.dmin_fixed = Some((0.222537, 0.108183, 0.054116));
                                }
                                let (mut r, mut g, mut b) = opts.dmin_fixed.unwrap();
                                ui.horizontal(|ui| {
                                    ui.label("R");
                                    ui.add(
                                        drag_decimal_f32(&mut r)
                                            .range(0.0..=1.0)
                                            .speed(0.01),
                                    );
                                    ui.label("G");
                                    ui.add(
                                        drag_decimal_f32(&mut g)
                                            .range(0.0..=1.0)
                                            .speed(0.01),
                                    );
                                    ui.label("B");
                                    ui.add(
                                        drag_decimal_f32(&mut b)
                                            .range(0.0..=1.0)
                                            .speed(0.01),
                                    );
                                });
                                opts.dmin_fixed = Some((r, g, b));
                            }
                            DminMode::SampleRegion => {
                                if opts.dmin_rect.is_none() {
                                    opts.dmin_rect = Some(Rect {
                                        x: 35,
                                        y: 15,
                                        width: 20,
                                        height: 20,
                                    });
                                }
                                if let Some(rect) = opts.dmin_rect.as_mut() {
                                    ui.horizontal(|ui| {
                                        ui.label("x,y,w,h")
                                            .on_hover_text("Drag the blue rectangle on the preview to select the film base region.");
                                        ui.add(egui::DragValue::new(&mut rect.x).speed(1));
                                        ui.add(egui::DragValue::new(&mut rect.y).speed(1));
                                        ui.add(egui::DragValue::new(&mut rect.width).speed(1));
                                        ui.add(egui::DragValue::new(&mut rect.height).speed(1));
                                    });
                                }
                                ui.checkbox(&mut opts.dmin_neutral_only, "Neutral only (geometric mean)");
                            }
                            DminMode::AutoPercentile => {
                                ui.horizontal(|ui| {
                                    ui.label("Border buffer")
                                        .on_hover_text("Automatic per-channel percentile normalization. Finds film base density (p0.5) and normalizes. Border buffer excludes edges from analysis.");
                                    ui.add(
                                        egui::Slider::new(&mut opts.auto_norm_buffer, 0.1..=0.3)
                                            .fixed_decimals(2),
                                    );
                                });
                            }
                        }

                        if opts.dmin_mode != DminMode::Off {
                            ui.add_space(4.0);
                            ui.separator();
                            ui.label("Flat-field override (luminance calibration)");
                            ui.horizontal(|ui| {
                                if ui.button("Load flat-field map…").clicked() {
                                    if let Some(path) = rfd::FileDialog::new()
                                        .add_filter(
                                            "Flat field",
                                            &[
                                                "tif", "tiff",
                                                "arw", "nef", "nrw", "cr2", "cr3", "crw", "dng", "raf",
                                                "orf", "rw2",
                                                "png", "jpeg", "jpg",
                                            ],
                                        )
                                        .pick_file()
                                    {
                                        self.flat_field_path = Some(path.clone());
                                        self.status = format!(
                                            "Using flat-field map from {} (overrides D-min).",
                                            path.display()
                                        );
                                    }
                                }
                                if self.flat_field_path.is_some()
                                    && ui.button("Clear flat-field override").clicked()
                                {
                                    self.flat_field_path = None;
                                    self.status =
                                        "Flat-field override cleared; D-min settings are active again."
                                            .to_string();
                                }
                            });
                            if let Some(ref p) = self.flat_field_path {
                                ui.label(egui::RichText::new(p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string()).small())
                                    .on_hover_text(format!("Flat-field: {}", p.display()));
                            } else {
                                ui.label("—")
                                    .on_hover_text("No flat-field override set.");
                            }
                        }

                        ui.add_space(4.0);
                        ui.separator();
                        ui.add_space(4.0);
                  } // end D-min Input tab guard

                  if !in_process || entry.process_tab == ProcessTab::Develop {
                    // ════════════════════════════════════════════════════════
                    // GROUP 6 — Curve / LUT & Output
                    // ════════════════════════════════════════════════════════
                    ui.collapsing("Output", |ui| {
                        let mut apply_curve = !opts.no_curve;
                        if ui.checkbox(&mut apply_curve, "Output curve").changed() {
                            if apply_curve {
                                if matches!(opts.output_stage, OutputStage::None) {
                                    opts.output_stage = OutputStage::Ra4;
                                }
                                opts.no_curve = false;
                            } else {
                                opts.no_curve = true;
                                opts.output_stage = OutputStage::None;
                            }
                        }

                        if apply_curve {
                            let current_label = match opts.output_stage {
                                OutputStage::Ra4 => "RA-4 print emulation",
                                OutputStage::FilmPrint => "Film Print",
                                OutputStage::Lut2383 => "3D LUT (display-space)",
                                OutputStage::None => "No curve",
                            };

                            egui::ComboBox::from_label("Output stage")
                                .selected_text(current_label)
                                .show_ui(ui, |ui| {
                                    if ui
                                        .selectable_label(
                                            matches!(opts.output_stage, OutputStage::Ra4),
                                            "RA-4 print emulation",
                                        )
                                        .clicked()
                                    {
                                        opts.output_stage = OutputStage::Ra4;
                                    }
                                    if ui
                                        .selectable_label(
                                            matches!(opts.output_stage, OutputStage::FilmPrint),
                                            "Film Print",
                                        )
                                        .clicked()
                                    {
                                        opts.output_stage = OutputStage::FilmPrint;
                                    }
                                    if ui
                                        .selectable_label(
                                            matches!(opts.output_stage, OutputStage::Lut2383),
                                            "3D LUT (display-space)",
                                        )
                                        .clicked()
                                    {
                                        opts.output_stage = OutputStage::Lut2383;
                                    }
                                    if ui
                                        .selectable_label(
                                            matches!(opts.output_stage, OutputStage::None),
                                            "No curve",
                                        )
                                        .clicked()
                                    {
                                        opts.output_stage = OutputStage::None;
                                        opts.no_curve = true;
                                    }
                                });

                            if matches!(opts.output_stage, OutputStage::Ra4) {
                                ui.add_space(4.0);
                                ui.horizontal(|ui| {
                                    ui.label("Pivot");
                                    ui.add(
                                        drag_decimal_f32(&mut opts.curve_pivot)
                                            .range(0.1..=10.0)
                                            .speed(0.1),
                                    );
                                });
                            }

                            if matches!(opts.output_stage, OutputStage::FilmPrint) {
                                ui.add_space(4.0);
                                ui.horizontal(|ui| {
                                    ui.label("R")
                                        .on_hover_text("Per-channel offsets (exposure shift)");
                                    ui.add(
                                        egui::Slider::new(&mut opts.fp_offset_r, -0.3..=0.3)
                                            .fixed_decimals(3),
                                    );
                                });
                                ui.horizontal(|ui| {
                                    ui.label("G");
                                    ui.add(
                                        egui::Slider::new(&mut opts.fp_offset_g, -0.3..=0.3)
                                            .fixed_decimals(3),
                                    );
                                });
                                ui.horizontal(|ui| {
                                    ui.label("B");
                                    ui.add(
                                        egui::Slider::new(&mut opts.fp_offset_b, -0.3..=0.3)
                                            .fixed_decimals(3),
                                    );
                                });

                                ui.add_space(4.0);
                                ui.horizontal(|ui| {
                                    ui.label("R")
                                        .on_hover_text("Per-channel gamma (contrast)");
                                    ui.add(
                                        egui::Slider::new(&mut opts.fp_gamma_r, 0.7..=1.5)
                                            .fixed_decimals(2),
                                    );
                                });
                                ui.horizontal(|ui| {
                                    ui.label("G");
                                    ui.add(
                                        egui::Slider::new(&mut opts.fp_gamma_g, 0.7..=1.5)
                                            .fixed_decimals(2),
                                    );
                                });
                                ui.horizontal(|ui| {
                                    ui.label("B");
                                    ui.add(
                                        egui::Slider::new(&mut opts.fp_gamma_b, 0.7..=1.5)
                                            .fixed_decimals(2),
                                    );
                                });

                                ui.add_space(4.0);
                                ui.horizontal(|ui| {
                                    ui.label("Color bleed");
                                    ui.add(
                                        egui::Slider::new(&mut opts.fp_color_bleed, 0.0..=0.3)
                                            .fixed_decimals(2),
                                    );
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Vibrance");
                                    ui.add(
                                        egui::Slider::new(&mut opts.fp_vibrance, 0.0..=1.0)
                                            .fixed_decimals(2),
                                    );
                                });
                            }

                            if matches!(opts.output_stage, OutputStage::Lut2383) {
                                ui.add_space(4.0);
                                let enc_label = match opts.output_lut_encoding {
                                    OutputLutEncoding::CineonLog => "Cineon log (D ÷ 2.046)",
                                    OutputLutEncoding::Rec709 => "Rec.709 (sRGB gamma)",
                                    OutputLutEncoding::LinearDensity => "Linear (D ÷ 2.5)",
                                };
                                let lut_combo = egui::ComboBox::from_label("LUT input encoding")
                                    .selected_text(enc_label)
                                    .show_ui(ui, |ui| {
                                        if ui
                                            .selectable_label(
                                                matches!(
                                                    opts.output_lut_encoding,
                                                    OutputLutEncoding::Rec709
                                                ),
                                                "Rec.709 (sRGB gamma)",
                                            )
                                            .clicked()
                                        {
                                            opts.output_lut_encoding = OutputLutEncoding::Rec709;
                                        }
                                        if ui
                                            .selectable_label(
                                                matches!(
                                                    opts.output_lut_encoding,
                                                    OutputLutEncoding::CineonLog
                                                ),
                                                "Cineon log (D ÷ 2.046)",
                                            )
                                            .clicked()
                                        {
                                            opts.output_lut_encoding = OutputLutEncoding::CineonLog;
                                        }
                                        if ui
                                            .selectable_label(
                                                matches!(
                                                    opts.output_lut_encoding,
                                                    OutputLutEncoding::LinearDensity
                                                ),
                                                "Linear (D ÷ 2.5)",
                                            )
                                            .clicked()
                                        {
                                            opts.output_lut_encoding = OutputLutEncoding::LinearDensity;
                                        }
                                    });
                                lut_combo.response.on_hover_text("Resolve-style Kodak 2383 cubes expect Cineon log input.");

                                ui.add_space(4.0);
                                if ui.button("Browse output LUT…").clicked() {
                                    self.pending_output_lut_browse = true;
                                }
                                if let Some(ref p) = opts.output_lut_cube {
                                    ui.label(egui::RichText::new(p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string()).small())
                                        .on_hover_text(p.display().to_string());
                                } else {
                                    ui.label("—")
                                        .on_hover_text("No output LUT loaded");
                                }
                            }
                        }

                    });
                  } // end Develop tab guard (group 6)
                }

                if self.mode == UIMode::Calibrate {
                    ui.add_space(12.0);
                    ui.separator();
                    ui.add_space(8.0);

                    if self.calibration_profile_name.is_empty() {
                        if let Some(stem) = entry
                            .path
                            .file_stem()
                            .and_then(|s| s.to_str())
                        {
                            self.calibration_profile_name = stem.to_string();
                        }
                    }
                    ui.label("Profile name / film stock")
                        .on_hover_text("Set the profile name / film stock and notes, then create the color profile in one step (matrix + 3D LUT saved as .c41).");
                    ui.text_edit_singleline(&mut self.calibration_profile_name);
                    ui.label("Notes (e.g. light source)");
                    ui.text_edit_singleline(&mut self.calibration_light_source);
                    ui.add_space(8.0);

                    if ui.button("Create color profile").clicked() {
                        let path = entry.path.clone();
                        let opts_clone = calibration_opts_snapshot.clone();
                        match load_linear_transmittance_for_calibration(&path, &opts_clone) {
                            Ok(image_lin) => {
                                let centers_norm =
                                    compute_patch_centers_normalized(self.calibration_overlay.corners);
                                let patches_linear =
                                    sample_patch_medians(&image_lin, &centers_norm, 5.0);
                                let measured_density =
                                    calibration::linear_to_density_24(patches_linear);
                                let reference_density =
                                    calibration::reference_density_24();

                                match calibration::solve_density_matrix_ols(
                                    measured_density,
                                    reference_density,
                                ) {
                                    Some((m, mse)) => {
                                        self.calibration_result = Some((m, mse));
                                        opts.density_matrix = m;
                                        let name = if self.calibration_profile_name.trim().is_empty() {
                                            "profile".to_string()
                                        } else {
                                            self.calibration_profile_name.trim().to_string()
                                        };
                                        let profile = calibration::CalibrationProfile {
                                            name: name.clone(),
                                            light_source: self.calibration_light_source.clone(),
                                            matrix: m,
                                            dmin_medians: calibration_opts_snapshot.dmin_fixed,
                                        };
                                        let base_dir = std::env::current_dir()
                                            .unwrap_or_else(|_| PathBuf::from("."))
                                            .join("profiles");
                                        let _ = std::fs::create_dir_all(&base_dir);
                                        if let Some(save_path) = rfd::FileDialog::new()
                                            .set_directory(&base_dir)
                                            .add_filter("C-41 profile", &["c41"])
                                            .set_file_name(&(name.clone() + ".c41"))
                                            .save_file()
                                        {
                                            match calibration::save_c41_profile(&profile, &save_path) {
                                                Ok(()) => {
                                                    self.status = format!(
                                                        "Created .c41 profile (MSE {:.6}): {}",
                                                        mse,
                                                        save_path.display()
                                                    );
                                                }
                                                Err(e) => {
                                                    self.status = format!("Failed to save .c41 profile: {}", e);
                                                }
                                            }
                                        } else {
                                            self.status = format!(
                                                "Solved matrix (MSE {:.6}); save cancelled.",
                                                mse
                                            );
                                        }
                                    }
                                    None => {
                                        self.status = "Color calibration failed: singular system".to_string();
                                    }
                                }
                            }
                            Err(e) => {
                                self.status = format!("Color calibration error: {}", e);
                            }
                        }
                    }

                    if let Some((m, mse)) = self.calibration_result {
                        ui.add_space(8.0);
                        ui.label(format!("Last result — MSE: {:.6}", mse));
                        ui.monospace(format!(
                            "[{:.4}, {:.4}, {:.4}]  [{:.4}, {:.4}, {:.4}]  [{:.4}, {:.4}, {:.4}]",
                            m[0][0], m[0][1], m[0][2],
                            m[1][0], m[1][1], m[1][2],
                            m[2][0], m[2][1], m[2][2],
                        ));
                    }
                }

                if self.mode == UIMode::Process && entry.process_tab == ProcessTab::Input {
                    ui.add_space(4.0);
                    ui.separator();
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new("Color Calibration").strong());
                    ui.add_space(2.0);
                    let apply_color_prev = opts.apply_color_profile;
                    ui.checkbox(&mut opts.apply_color_profile, "Color calibration profile");
                    if apply_color_prev && !opts.apply_color_profile {
                        opts.density_matrix = [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ];
                        self.selected_profile_idx = None;
                        opts.lut3d_path = None;
                    }
                    if opts.apply_color_profile {
                    ui.collapsing("Color calibration profile settings", |ui| {
                        ui.label("Profile (open dropdown to scan profiles/ folder)");
                        let mut current_idx = self.selected_profile_idx.unwrap_or(usize::MAX);
                        let selected_label = if let Some(i) = self.selected_profile_idx {
                            if let Some((_, p, _)) = self.calibration_profiles.get(i) {
                                p.name.as_str()
                            } else {
                                "None"
                            }
                        } else {
                            "None"
                        };

                        egui::ComboBox::from_label("Profile")
                            .selected_text(selected_label)
                            .show_ui(ui, |ui| {
                                // Refresh list when dropdown is open
                                let base_dir = std::env::current_dir()
                                    .unwrap_or_else(|_| PathBuf::from("."))
                                    .join("profiles");
                                if let Ok(list) = calibration::load_profiles_from_dir(&base_dir) {
                                    self.calibration_profiles = list;
                                }
                                if ui
                                    .selectable_label(
                                        self.selected_profile_idx.is_none(),
                                        "None",
                                    )
                                    .clicked()
                                {
                                    current_idx = usize::MAX;
                                }
                                for (i, (_, profile, _)) in
                                    self.calibration_profiles.iter().enumerate()
                                {
                                    let is_selected = self.selected_profile_idx == Some(i);
                                    if ui
                                        .selectable_label(is_selected, &profile.name)
                                        .clicked()
                                    {
                                        current_idx = i;
                                    }
                                }
                                if self.calibration_profiles.is_empty() {
                                    ui.label("No .json or .c41 profiles in profiles/");
                                }
                            });

                        // Apply selection to current image options.
                        if current_idx == usize::MAX {
                            self.selected_profile_idx = None;
                            opts.lut3d_path = None;
                        } else if let Some((_, profile, lut_path)) =
                            self.calibration_profiles.get(current_idx).cloned()
                        {
                            self.selected_profile_idx = Some(current_idx);
                            opts.density_matrix = profile.matrix;
                            if let Some(dmin) = profile.dmin_medians {
                                opts.dmin_fixed = Some(dmin);
                                opts.dmin_rect = None;
                                opts.dmin_rect_reference_size = None;
                            }
                            opts.lut3d_path = lut_path;
                            self.status = format!(
                                "Applied color calibration profile '{}' to current image.",
                                profile.name
                            );
                        }

                        ui.separator();
                        ui.label("Or use 3D LUT (generated in Color calibration tab):");
                        ui.horizontal(|ui| {
                            let path_str = opts
                                .lut3d_path
                                .as_ref()
                                .map(|p| p.display().to_string())
                                .unwrap_or_else(|| "None".to_string());
                            ui.label(path_str.as_str());
                            if ui.button("Browse…").clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("CUBE LUT", &["cube"])
                                    .pick_file()
                                {
                                    opts.lut3d_path = Some(path.clone());
                                    self.status = format!("Using 3D LUT: {}", path.display());
                                }
                            }
                            if opts.lut3d_path.is_some() && ui.button("Clear").clicked() {
                                opts.lut3d_path = None;
                                self.status = "Cleared 3D LUT; using profile matrix.".to_string();
                            }
                        });
                    });
                    }
                }

                if self.mode == UIMode::LuminanceCalibrate {
                    ui.collapsing("Flat field (luminance calibration)", |ui| {
                        ui.label("Reference frame: unexposed, developed RAW from the same roll.");
                        if ui.button("Load Reference Frame…").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter(
                                    "RAW",
                                    &[
                                        "arw", "nef", "nrw", "cr2", "cr3", "crw", "dng", "raf", "orf", "rw2",
                                    ],
                                )
                                .pick_file()
                            {
                                match load_flat_field_linear(&path) {
                                    Ok(arr) => {
                                        // Heavy blur to remove grain/dust, keep only luminance falloff.
                                        let radius = 60.0_f32;
                                        let blurred = blur_flat_field(&arr, radius);
                                        let (h, w, c) = blurred.dim();
                                        self.flat_field_path = Some(path.clone());
                                        self.flat_field_image = Some(blurred);
                                        self.status = format!(
                                            "Loaded and blurred flat-field {}×{} ({} ch), radius {:.1} from {}",
                                            h,
                                            w,
                                            c,
                                            radius,
                                            path.file_name().and_then(|n| n.to_str()).unwrap_or(""),
                                        );
                                    }
                                    Err(e) => {
                                        self.flat_field_path = None;
                                        self.flat_field_image = None;
                                        self.status = format!("Failed to load flat-field: {}", e);
                                    }
                                }
                            }
                        }
                        if let Some(ref p) = self.flat_field_path {
                            ui.label(egui::RichText::new(p.display().to_string()).small());
                            if let Some(ref arr) = self.flat_field_image {
                                let (h, w, _) = arr.dim();
                                ui.label(format!("Linearized: {}×{} RGB", h, w));
                                if ui.button("Save blurred flat-field as 32f TIFF…").clicked() {
                                    let default_name = p
                                        .file_stem()
                                        .and_then(|s| s.to_str())
                                        .map(|s| format!("{}_flat_field.tiff", s))
                                        .unwrap_or_else(|| "flat_field.tiff".to_string());
                                    if let Some(path) = rfd::FileDialog::new()
                                        .set_file_name(default_name)
                                        .save_file()
                                    {
                                        match tiff_export::write_tiff(
                                            arr,
                                            &path,
                                            TiffFormat::Float32,
                                        ) {
                                            Ok(()) => {
                                                self.status = format!(
                                                    "Saved blurred flat-field to {}",
                                                    path.display()
                                                );
                                            }
                                            Err(e) => {
                                                self.status = format!(
                                                    "Failed to save flat-field TIFF: {}",
                                                    e
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            ui.label("No reference frame loaded.");
                        }
                    });
                }

                if self.mode == UIMode::Process && entry.process_tab == ProcessTab::Export {
                ui.add_space(12.0);
                ui.separator();
                ui.add_space(8.0);
                ui.heading("Export");
                ui.add_space(8.0);

                // Per-image export options
                let label = match entry.export_format {
                    ExportFormat::Tiff16 => "TIFF 16-bit",
                    ExportFormat::Tiff32 => "TIFF 32-bit float",
                    ExportFormat::Exr => "EXR (32-bit float)",
                    ExportFormat::Jpeg => "JPEG",
                    ExportFormat::ExrAces2065 => "EXR ACES2065-1 (32-bit float)",
                };
                egui::ComboBox::from_label("Output format")
                    .selected_text(label)
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_label(matches!(entry.export_format, ExportFormat::Tiff16), "TIFF 16-bit")
                            .clicked()
                        {
                            entry.export_format = ExportFormat::Tiff16;
                        }
                        if ui
                            .selectable_label(matches!(entry.export_format, ExportFormat::Tiff32), "TIFF 32-bit float")
                            .clicked()
                        {
                            entry.export_format = ExportFormat::Tiff32;
                        }
                        if ui
                            .selectable_label(matches!(entry.export_format, ExportFormat::Exr), "EXR (32-bit float)")
                            .clicked()
                        {
                            entry.export_format = ExportFormat::Exr;
                        }
                        let aces_selected = matches!(entry.export_format, ExportFormat::ExrAces2065);
                        if ui
                            .selectable_label(aces_selected, "EXR ACES2065-1 (32-bit float)")
                            .clicked()
                        {
                            entry.export_format = ExportFormat::ExrAces2065;
                        }
                        if ui
                            .selectable_label(matches!(entry.export_format, ExportFormat::Jpeg), "JPEG")
                            .clicked()
                        {
                            entry.export_format = ExportFormat::Jpeg;
                        }
                    });

                // Keep PipelineOptions in sync with export dropdown
                match entry.export_format {
                    ExportFormat::Tiff16 => {
                        opts.format = TiffFormat::U16;
                        opts.write_exr = false;
                        opts.write_jpeg_only = false;
                        opts.export_aces_exr = false;
                        opts.write_aces2065_only = false;
                    }
                    ExportFormat::Tiff32 => {
                        opts.format = TiffFormat::Float32;
                        opts.write_exr = false;
                        opts.write_jpeg_only = false;
                        opts.export_aces_exr = false;
                        opts.write_aces2065_only = false;
                    }
                    ExportFormat::Exr => {
                        opts.format = TiffFormat::Float32;
                        opts.write_exr = true;
                        opts.write_jpeg_only = false;
                        opts.export_aces_exr = false;
                        opts.write_aces2065_only = false;
                    }
                    ExportFormat::Jpeg => {
                        opts.format = TiffFormat::U16;
                        opts.write_exr = false;
                        opts.write_jpeg_only = true;
                        opts.write_jpeg = false;
                        opts.export_aces_exr = false;
                        opts.write_aces2065_only = false;
                    }
                    ExportFormat::ExrAces2065 => {
                        opts.format = TiffFormat::Float32;
                        opts.write_exr = false;
                        opts.write_jpeg_only = false;
                        opts.export_aces_exr = false;
                        opts.write_aces2065_only = true;
                    }
                }

                ui.add_enabled(
                    entry.export_format != ExportFormat::Jpeg,
                    egui::Checkbox::new(&mut opts.write_jpeg, "Also export JPG"),
                );

                ui.add_space(8.0);

                // Global export: output folder + convert all
                ui.add_space(12.0);
                ui.separator();
                ui.add_space(8.0);
                ui.label(egui::RichText::new("Batch export").strong());

                let out_label = self
                    .output_dir
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "No output folder".to_string());
                if ui.button("Output folder…").clicked() {
                    if let Some(path) = rfd::FileDialog::new().pick_folder() {
                        self.output_dir = Some(path);
                    }
                }
                ui.label(egui::RichText::new(out_label).small());

                let ready = !self.images.is_empty() && self.output_dir.is_some();
                let selected_ready = self.selected_index.is_some()
                    && self.selected_index.unwrap() < self.images.len()
                    && self.output_dir.is_some();
                ui.horizontal(|ui| {
                    if ui.add_enabled(ready, egui::Button::new("Convert all")).clicked() {
                        let output_dir = self.output_dir.clone().unwrap();
                        let mut err: Option<anyhow::Error> = None;
                        // Use selected image's export format for all (dropdown only updates the selected image)
                        let export_template = self
                            .selected_index
                            .filter(|&i| i < self.images.len())
                            .map(|i| &self.images[i].options)
                            .unwrap_or_else(|| &self.images[0].options);
                        for img in &self.images {
                            let mut opts = img.options.clone();
                            opts.flat_field_path = self.flat_field_path.clone();
                            // Apply batch export format from the selected/first image
                            opts.format = export_template.format;
                            opts.write_exr = export_template.write_exr;
                            opts.write_jpeg = export_template.write_jpeg;
                            opts.write_jpeg_only = export_template.write_jpeg_only;
                            opts.export_aces_exr = export_template.export_aces_exr;
                            opts.write_aces2065_only = export_template.write_aces2065_only;
                            if let Err(e) = process_files(&[img.path.clone()], &output_dir, &opts) {
                                err = Some(e);
                                break;
                            }
                        }
                        self.status = if let Some(e) = err {
                            format!("Error: {}", e)
                        } else {
                            "Done.".to_string()
                        };
                    }
                    if ui.add_enabled(selected_ready, egui::Button::new("Export selected")).clicked() {
                        if let Some(idx) = self.selected_index {
                            if idx < self.images.len() {
                                let img = &self.images[idx];
                                let output_dir = self.output_dir.clone().unwrap();
                                let mut opts = img.options.clone();
                                opts.flat_field_path = self.flat_field_path.clone();
                                match process_files(&[img.path.clone()], &output_dir, &opts) {
                                    Ok(()) => self.status = "Done.".to_string(),
                                    Err(e) => self.status = format!("Error: {}", e),
                                }
                            }
                        }
                    }
                });

                if !self.status.is_empty() {
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new(&self.status).small());
                }
                } else if !self.status.is_empty() {
                    ui.add_space(12.0);
                    ui.separator();
                    ui.add_space(8.0);
                    ui.label(egui::RichText::new(&self.status).small());
                }
                    });
                    ui.add_space(16.0);
                });
                });
            });

        // ---- Auto-crop: run frame detection after sidebar borrow is released ----
        if auto_crop_requested {
            if let Some(idx) = self.selected_index {
                if idx < self.images.len() {
                    // Clone the pixel buffer so we can mutably write options
                    // on the same entry without a borrow conflict.
                    let preview_clone = self.images[idx].preview_full_rgb.clone();
                    if let Some((w, h, pixels)) = preview_clone {
                        // Scale the D-min rect to preview pixel space if one is set.
                        let dmin_rect_preview = self.images[idx].options.dmin_rect.map(|r| {
                            scale_rect_to_size(
                                r,
                                self.images[idx].options.dmin_rect_reference_size,
                                w, h,
                            )
                        });
                        let auto_norm_buffer = self.images[idx].options.auto_norm_buffer;
                        match sample_border_color(w, h, &pixels, dmin_rect_preview, auto_norm_buffer) {
                            None => {
                                self.status = "Auto crop: set a D-min region first so the film-base colour is known.".to_string();
                            }
                            Some((br, bg, bb, tol)) => {
                                match auto_detect_crop(w, h, &pixels, (br, bg, bb), tol) {
                                    Some(rect) => {
                                        self.images[idx].options.crop_rect = Some(rect);
                                        self.images[idx].options.crop_rect_reference_size = Some((w, h));
                                        self.images[idx].options.apply_crop = true;
                                    }
                                    None => {
                                        self.status = "Auto crop: no clear frame boundary found.".to_string();
                                    }
                                }
                            }
                        }
                    } else {
                        self.status = "Auto crop: waiting for preview to finish processing.".to_string();
                    }
                }
            }
        }

        // ---- Central panel: preview + histogram ----
        egui::CentralPanel::default().show(ctx, |ui| {
            let has_inflight = self.preview_receiver.is_some();
            let show_loader = has_inflight
                && self
                    .preview_started_at
                    .map(|t| t.elapsed() >= Duration::from_millis(2500))
                    .unwrap_or(true);

            if let Some(idx) = self.selected_index {
                if idx < self.images.len() {
                    {
                        let available = ui.available_rect_before_wrap();
                        const CONTROL_ROW_HEIGHT: f32 = 28.0;
                        const HISTOGRAM_HEIGHT: f32 = 72.0;
                        const BOTTOM_PADDING: f32 = 8.0;
                        const IMAGE_PREVIEW_BOTTOM_PADDING: f32 = 16.0;
                        const TOP_PADDING: f32 = 17.0;
                        const INFO_ROW_HEIGHT: f32 = 18.0;

                        let reserved_bottom = IMAGE_PREVIEW_BOTTOM_PADDING
                            + INFO_ROW_HEIGHT
                            + CONTROL_ROW_HEIGHT
                            + BOTTOM_PADDING
                            + HISTOGRAM_HEIGHT
                            + BOTTOM_PADDING;
                        let canvas_h = (available.height() - reserved_bottom).max(60.0);
                        let canvas_w = available.width();
                        self.preview_canvas_size = Some((canvas_w, canvas_h));

                        // Extract image dims with fallback so the layout is stable before
                        // the first preview arrives (no jump when data loads in).
                        let full_rgb_opt = self.images[idx].preview_full_rgb.clone();
                        let (full_w, full_h) = if let Some((w, h, _)) = &full_rgb_opt {
                            (*w, *h)
                        } else {
                            self.images[idx].preview_input_size
                                .map(|s| (s[0], s[1]))
                                .unwrap_or((canvas_w as u32, (canvas_w * 2.0 / 3.0) as u32))
                        };
                        let full_w_f = (full_w as f32).max(1.0);
                        let full_h_f = (full_h as f32).max(1.0);

                        // Upload texture only when RGB data is available.
                        let tex_opt: Option<egui::TextureHandle> =
                            if let Some((fw, fh, full_rgb)) = &full_rgb_opt {
                                let entry = &self.images[idx];
                                let size = [*fw as usize, *fh as usize];
                                let pixels: Vec<egui::Color32> = full_rgb
                                    .chunks_exact(3)
                                    .map(|c| egui::Color32::from_rgb(c[0], c[1], c[2]))
                                    .collect();
                                let image = egui::ColorImage { size, pixels };
                                let tex_opts = if entry.preview_zoom > 1.0 {
                                    egui::TextureOptions::NEAREST
                                } else {
                                    egui::TextureOptions::LINEAR
                                };
                                Some(ui.ctx().load_texture(
                                    format!("preview_full_{}", idx),
                                    image,
                                    tex_opts,
                                ))
                            } else {
                                None
                            };
                        if let Some(tex) = &tex_opt {
                            self.images[idx].preview_texture = Some(tex.clone());
                        }

                        // Allocate the full canvas area — always, so the layout never jumps.
                        ui.add_space(TOP_PADDING);
                        let (canvas_rect, canvas_resp) = ui.allocate_exact_size(
                            egui::vec2(canvas_w, canvas_h),
                            egui::Sense::click_and_drag(),
                        );
                        let canvas_painter = ui.painter_at(canvas_rect);
                        canvas_painter.rect_filled(canvas_rect, 0.0, egui::Color32::from_gray(30));

                        // Base scale: image size at zoom=1.0 to fit within canvas.
                        let base_scale = (canvas_w / full_w_f).min(canvas_h / full_h_f);

                        let entry = &mut self.images[idx];
                        let zoom = entry.preview_zoom.max(1.0);
                        let img_w = full_w_f * base_scale * zoom;
                        let img_h = full_h_f * base_scale * zoom;

                        // Pan: which image-normalized point sits at canvas center.
                        let pan_x = entry.preview_pan.x.clamp(0.0, 1.0);
                        let pan_y = entry.preview_pan.y.clamp(0.0, 1.0);

                        // Virtual image rect: where the full image lives in screen coords.
                        let vir_left = canvas_rect.center().x - pan_x * img_w;
                        let vir_top  = canvas_rect.center().y - pan_y * img_h;
                        let vir_rect = egui::Rect::from_min_size(
                            egui::pos2(vir_left, vir_top),
                            egui::vec2(img_w, img_h),
                        );

                        // Draw image only when texture is ready.
                        if let Some(tex) = &tex_opt {
                            let vis_rect = vir_rect.intersect(canvas_rect);
                            if vis_rect.width() > 0.0 && vis_rect.height() > 0.0 {
                                let uv_l = (vis_rect.left()   - vir_rect.left()) / img_w;
                                let uv_t = (vis_rect.top()    - vir_rect.top())  / img_h;
                                let uv_r = (vis_rect.right()  - vir_rect.left()) / img_w;
                                let uv_b = (vis_rect.bottom() - vir_rect.top())  / img_h;
                                let uv = egui::Rect::from_min_max(
                                    egui::pos2(uv_l, uv_t),
                                    egui::pos2(uv_r, uv_b),
                                );
                                canvas_painter.image(tex.id(), vis_rect, uv, egui::Color32::WHITE);
                            }
                        }

                        // image_rect = canvas_rect so overlays can paint across the full canvas.
                        let image_rect = canvas_rect;

                        // Camera helpers using the virtual image rect.
                        let vir_left_c = vir_rect.left();
                        let vir_top_c  = vir_rect.top();
                        let img_w_c = img_w;
                        let img_h_c = img_h;
                        let image_to_screen = |px: f32, py: f32| -> egui::Pos2 {
                            egui::pos2(
                                vir_left_c + (px / full_w_f) * img_w_c,
                                vir_top_c  + (py / full_h_f) * img_h_c,
                            )
                        };
                        let screen_to_image = |sx: f32, sy: f32| -> (f32, f32) {
                            (
                                ((sx - vir_left_c) / img_w_c) * full_w_f,
                                ((sy - vir_top_c)  / img_h_c) * full_h_f,
                            )
                        };
                        ui.add_space(IMAGE_PREVIEW_BOTTOM_PADDING);

                        // Loading spinner overlay — drawn entirely via canvas_painter so it
                        // never touches the UI layout cursor (which would shift the histogram).
                        // Also show when no preview data is available yet (first load).
                        if show_loader || full_rgb_opt.is_none() {
                            canvas_painter.rect_filled(
                                canvas_rect,
                                0.0,
                                egui::Color32::from_rgba_premultiplied(0, 0, 0, 90),
                            );
                            let center = canvas_rect.center();
                            let radius = 11.0;
                            let t = ctx.input(|i| i.time) as f32;
                            let start = t * std::f32::consts::TAU * 0.8;
                            let arc = std::f32::consts::PI * 1.5;
                            let steps = 28usize;
                            let pts: Vec<egui::Pos2> = (0..=steps)
                                .map(|i| {
                                    let a = start + arc * i as f32 / steps as f32;
                                    egui::pos2(
                                        center.x + a.cos() * radius,
                                        center.y + a.sin() * radius,
                                    )
                                })
                                .collect();
                            canvas_painter.add(egui::Shape::line(
                                pts,
                                egui::Stroke::new(2.5, egui::Color32::from_gray(210)),
                            ));
                            ctx.request_repaint();
                        }

                        // Zoom with scroll wheel — always works.
                        if canvas_resp.hovered()
                            && ui.input(|i| i.raw_scroll_delta.y != 0.0 && !i.modifiers.shift)
                        {
                            let scroll = ui.input(|i| i.raw_scroll_delta.y);
                            let factor = if scroll > 0.0 { 1.1 } else { 1.0 / 1.1 };
                            let entry = &mut self.images[idx];
                            let new_zoom = (entry.preview_zoom.max(1.0) * factor).clamp(1.0, 16.0);
                            let new_img_w = full_w_f * base_scale * new_zoom;
                            let new_img_h = full_h_f * base_scale * new_zoom;

                            if let Some(mp) = ui.input(|i| i.pointer.hover_pos()) {
                                let (img_px, img_py) = screen_to_image(mp.x, mp.y);
                                let u = (img_px / full_w_f).clamp(0.0, 1.0);
                                let v = (img_py / full_h_f).clamp(0.0, 1.0);
                                // Keep point under cursor fixed after zoom.
                                entry.preview_pan.x =
                                    (u - (mp.x - canvas_rect.center().x) / new_img_w).clamp(0.0, 1.0);
                                entry.preview_pan.y =
                                    (v - (mp.y - canvas_rect.center().y) / new_img_h).clamp(0.0, 1.0);
                            }

                            entry.preview_zoom = new_zoom;
                        }

                        // Pan with left drag (when no rect handle hit) or middle drag.
                        let middle_drag = ui.input(|i| i.pointer.middle_down()) && canvas_resp.dragged();
                        let left_drag = canvas_resp.dragged() && !self.rect_dragging;
                        if middle_drag || left_drag {
                            let delta = canvas_resp.drag_delta();
                            let entry = &mut self.images[idx];
                            entry.preview_pan.x =
                                (entry.preview_pan.x - delta.x / img_w).clamp(0.0, 1.0);
                            entry.preview_pan.y =
                                (entry.preview_pan.y - delta.y / img_h).clamp(0.0, 1.0);
                        }

                        // White balance picker: on click, sample density from cache and set WB gains.
                        if self.wb_picker_pending.is_some() && canvas_resp.clicked() {
                            if let Some(pos) = canvas_resp.interact_pointer_pos() {
                                let (px_f, py_f) = screen_to_image(pos.x, pos.y);
                                let px = (px_f as u32).min(full_w.saturating_sub(1));
                                let py = (py_f as u32).min(full_h.saturating_sub(1));
                                if let Some(ref cache) = self.images[idx].preview_step_cache {
                                    if let Some((_, ref buf)) = cache.after_step3 {
                                        let (h, w, _) = buf.dim();
                                        let y = (py as usize).min(h.saturating_sub(1));
                                        let x = (px as usize).min(w.saturating_sub(1));
                                        let tr = buf[[y, x, 0]].max(1e-10);
                                        let tg = buf[[y, x, 1]].max(1e-10);
                                        let tb = buf[[y, x, 2]].max(1e-10);
                                        let dr = -(tr as f64).log10() as f32;
                                        let dg = -(tg as f64).log10() as f32;
                                        let db = -(tb as f64).log10() as f32;
                                        let (wb_r, wb_g, wb_b) = color::density_to_wb_gains(dr, dg, db);
                                        let opts = &mut self.images[idx].options;
                                        opts.wb_r = wb_r;
                                        opts.wb_g = wb_g;
                                        opts.wb_b = wb_b;
                                        opts.apply_white_balance = true;
                                        opts.auto_wb = false;
                                        self.status = format!("WB set from {} point (R={:.3} G={:.3} B={:.3})",
                                            match self.wb_picker_pending.unwrap() {
                                                WbPickerTarget::WhitePoint => "white",
                                                WbPickerTarget::GrayPoint => "gray",
                                                WbPickerTarget::BlackPoint => "black",
                                            },
                                            wb_r, wb_g, wb_b);
                                    } else {
                                        self.status = "WB picker: no cache (re-run preview first).".to_string();
                                    }
                                } else {
                                    self.status = "WB picker: no cache (re-run preview first).".to_string();
                                }
                                self.wb_picker_pending = None;
                            }
                        }

                        // In Calibrate mode, draw and allow interaction with the
                        // 4-point overlay and the interpolated 24 patch boxes.
                        if self.mode == UIMode::Calibrate {
                            let mut corners = self.calibration_overlay.corners;
                            let handle_radius = 6.0;
                            let handle_size = egui::vec2(handle_radius * 2.0, handle_radius * 2.0);
                            let painter = ui.painter_at(image_rect);

                            // Helper to map normalized (0..1) coords to screen space inside image_rect.
                            let to_screen = |p: egui::Pos2| -> egui::Pos2 {
                                egui::pos2(
                                    image_rect.left() + p.x * image_rect.width(),
                                    image_rect.top() + p.y * image_rect.height(),
                                )
                            };

                            // Draw and update draggable corner handles.
                            for i in 0..4 {
                                let mut screen_pos = to_screen(corners[i]);
                                let handle_rect =
                                    egui::Rect::from_center_size(screen_pos, handle_size);
                                let id = ui.make_persistent_id(("calib_corner", i));
                                let resp =
                                    ui.interact(handle_rect, id, egui::Sense::click_and_drag());
                                if resp.dragged() {
                                    let delta = resp.drag_delta();
                                    screen_pos.x += delta.x;
                                    screen_pos.y += delta.y;
                                    // Clamp to image rectangle.
                                    screen_pos.x =
                                        screen_pos.x.clamp(image_rect.left(), image_rect.right());
                                    screen_pos.y =
                                        screen_pos.y.clamp(image_rect.top(), image_rect.bottom());
                                    // Convert back to normalized coordinates.
                                    let nx =
                                        (screen_pos.x - image_rect.left()) / image_rect.width();
                                    let ny =
                                        (screen_pos.y - image_rect.top()) / image_rect.height();
                                    corners[i] = egui::pos2(
                                        nx.clamp(0.0, 1.0),
                                        ny.clamp(0.0, 1.0),
                                    );
                                }

                                painter.circle_filled(
                                    screen_pos,
                                    handle_radius,
                                    egui::Color32::YELLOW,
                                );
                            }

                            // Persist any updated corner positions.
                            self.calibration_overlay.corners = corners;

                            // Interpolate a 6×4 grid of patch centers between the 4 corners.
                            // Corner layout: 0=TL, 1=TR, 2=BL, 3=BR.
                            let tl = to_screen(corners[0]);
                            let tr = to_screen(corners[1]);
                            let bl = to_screen(corners[2]);
                            let br = to_screen(corners[3]);

                            let rows = 4usize;
                            let cols = 6usize;
                            let bbox_half_h =
                                self.calibration_overlay.bbox_half_height_frac * image_rect.height();
                            let bbox_half_w = bbox_half_h; // keep boxes square

                            for row in 0..rows {
                                let v = if rows > 1 {
                                    row as f32 / (rows as f32 - 1.0)
                                } else {
                                    0.0
                                };
                                let left = tl.lerp(bl, v);
                                let right = tr.lerp(br, v);

                                for col in 0..cols {
                                    let u = if cols > 1 {
                                        col as f32 / (cols as f32 - 1.0)
                                    } else {
                                        0.0
                                    };
                                    let center = left.lerp(right, u);
                                    let rect = egui::Rect::from_center_size(
                                        center,
                                        egui::vec2(bbox_half_w * 2.0, bbox_half_h * 2.0),
                                    );
                                    painter.rect_stroke(
                                        rect,
                                        0.0,
                                        egui::Stroke::new(1.0, egui::Color32::LIGHT_GREEN),
                                    );
                                }
                            }
                        }

                        // D-min overlay: project image-space rect through camera.
                        if self.mode == UIMode::Process {
                            if let Some(entry) = self.images.get_mut(idx) {
                                let opts = &mut entry.options;
                                if opts.dmin_mode == DminMode::SampleRegion && self.flat_field_path.is_none() {
                                    if let (Some(rect), Some([input_w, input_h])) =
                                        (opts.dmin_rect, entry.preview_input_size)
                                    {
                                        if input_w > 0 && input_h > 0 {
                                            let painter = ui.painter_at(image_rect);

                                            // Scale from rect's reference space (where it was last edited)
                                            // into the current preview working resolution.
                                            let scaled = scale_rect_to_size(
                                                rect,
                                                opts.dmin_rect_reference_size,
                                                input_w,
                                                input_h,
                                            );

                                            let scr_tl = image_to_screen(scaled.x as f32, scaled.y as f32);
                                            let scr_br = image_to_screen(
                                                (scaled.x + scaled.width) as f32,
                                                (scaled.y + scaled.height) as f32,
                                            );
                                            let mut left = scr_tl.x;
                                            let mut top = scr_tl.y;
                                            let mut right = scr_br.x;
                                            let mut bottom = scr_br.y;

                                            let handle_radius = 5.0;
                                            let mut rect_changed = false;

                                            let corners = [
                                                egui::pos2(left, top),
                                                egui::pos2(right, top),
                                                egui::pos2(left, bottom),
                                                egui::pos2(right, bottom),
                                            ];
                                            for (ci, cp) in corners.iter().enumerate() {
                                                let hr = egui::Rect::from_center_size(
                                                    *cp,
                                                    egui::vec2(handle_radius * 2.0, handle_radius * 2.0),
                                                );
                                                let id = ui.make_persistent_id(("dmin_handle", idx, ci));
                                                let resp = ui.interact(hr, id, egui::Sense::click_and_drag());
                                                if resp.dragged() {
                                                    let d = resp.drag_delta();
                                                    rect_changed = true;
                                                    self.rect_dragging = true;
                                                    match ci {
                                                        0 => { left += d.x; top += d.y; }
                                                        1 => { right += d.x; top += d.y; }
                                                        2 => { left += d.x; bottom += d.y; }
                                                        3 => { right += d.x; bottom += d.y; }
                                                        _ => {}
                                                    }
                                                }
                                            }

                                            let screen_rect = egui::Rect::from_min_max(
                                                egui::pos2(left, top),
                                                egui::pos2(right, bottom),
                                            );
                                            painter.rect_stroke(
                                                screen_rect, 0.0,
                                                egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 200, 0)),
                                            );
                                            for p in [
                                                egui::pos2(left, top), egui::pos2(right, top),
                                                egui::pos2(left, bottom), egui::pos2(right, bottom),
                                            ] {
                                                painter.circle_filled(p, handle_radius, egui::Color32::from_rgb(255, 200, 0));
                                            }

                                            if rect_changed {
                                                let (ix0, iy0) = screen_to_image(left, top);
                                                let (ix1, iy1) = screen_to_image(right, bottom);
                                                let iw = input_w as f32;
                                                let ih = input_h as f32;
                                                let nx = ix0.round().clamp(0.0, iw - 1.0) as u32;
                                                let ny = iy0.round().clamp(0.0, ih - 1.0) as u32;
                                                let nx1 = ix1.round().clamp(0.0, iw) as u32;
                                                let ny1 = iy1.round().clamp(0.0, ih) as u32;
                                                let mut wp = nx1.saturating_sub(nx).max(1);
                                                let mut hp = ny1.saturating_sub(ny).max(1);
                                                wp = wp.min(input_w.saturating_sub(nx).max(1));
                                                hp = hp.min(input_h.saturating_sub(ny).max(1));
                                                opts.dmin_rect = Some(Rect { x: nx, y: ny, width: wp, height: hp });
                                                opts.dmin_rect_reference_size = Some((input_w, input_h));
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Crop overlay: project image-space crop rect through camera.
                        // Always draw the mask; only draw interactive handles in the Input tab.
                        if self.mode == UIMode::Process {
                            if let Some(entry) = self.images.get_mut(idx) {
                                let opts = &mut entry.options;
                                if opts.apply_crop {
                                    if let (Some(crop), Some([input_w, input_h])) =
                                        (opts.crop_rect, entry.preview_input_size)
                                    {
                                        if input_w > 0 && input_h > 0 {
                                            let painter = ui.painter_at(image_rect);
                                            let crop = scale_rect_to_size(
                                                crop,
                                                opts.crop_rect_reference_size,
                                                input_w,
                                                input_h,
                                            );

                                            let scr_tl = image_to_screen(crop.x as f32, crop.y as f32);
                                            let scr_br = image_to_screen(
                                                (crop.x + crop.width) as f32,
                                                (crop.y + crop.height) as f32,
                                            );
                                            let mut left = scr_tl.x;
                                            let mut top = scr_tl.y;
                                            let mut right = scr_br.x;
                                            let mut bottom = scr_br.y;

                                            // Mask outside the crop. In Develop/Export tabs we use a fully
                                            // opaque panel-fill color so the area behaves like a hard crop.
                                            let overlay = if entry.process_tab == ProcessTab::Develop
                                                || entry.process_tab == ProcessTab::Export
                                            {
                                                ui.visuals().panel_fill
                                            } else {
                                                egui::Color32::from_black_alpha(128)
                                            };
                                            painter.rect_filled(
                                                egui::Rect::from_min_max(
                                                    image_rect.min,
                                                    egui::pos2(image_rect.max.x, top),
                                                ),
                                                0.0,
                                                overlay,
                                            );
                                            painter.rect_filled(
                                                egui::Rect::from_min_max(
                                                    egui::pos2(image_rect.min.x, bottom),
                                                    image_rect.max,
                                                ),
                                                0.0,
                                                overlay,
                                            );
                                            painter.rect_filled(
                                                egui::Rect::from_min_max(
                                                    egui::pos2(image_rect.min.x, top),
                                                    egui::pos2(left, bottom),
                                                ),
                                                0.0,
                                                overlay,
                                            );
                                            painter.rect_filled(
                                                egui::Rect::from_min_max(
                                                    egui::pos2(right, top),
                                                    egui::pos2(image_rect.max.x, bottom),
                                                ),
                                                0.0,
                                                overlay,
                                            );

                                            // When in the Input tab, show interactive handles and D-min rect
                                            // *above* the overlay so the handles remain visible.
                                            if entry.process_tab == ProcessTab::Input {
                                                let handle_radius = 5.0;
                                                let mut rect_changed = false;

                                                let corners = [
                                                    egui::pos2(left, top),
                                                    egui::pos2(right, top),
                                                    egui::pos2(left, bottom),
                                                    egui::pos2(right, bottom),
                                                ];
                                                for (ci, cp) in corners.iter().enumerate() {
                                                    let hr = egui::Rect::from_center_size(
                                                        *cp,
                                                        egui::vec2(handle_radius * 2.0, handle_radius * 2.0),
                                                    );
                                                    let id = ui.make_persistent_id(("crop_handle", idx, ci));
                                                    let resp =
                                                        ui.interact(hr, id, egui::Sense::click_and_drag());
                                                    if resp.dragged() {
                                                        let d = resp.drag_delta();
                                                        rect_changed = true;
                                                        self.rect_dragging = true;
                                                        match ci {
                                                            0 => {
                                                                left += d.x;
                                                                top += d.y;
                                                            }
                                                            1 => {
                                                                right += d.x;
                                                                top += d.y;
                                                            }
                                                            2 => {
                                                                left += d.x;
                                                                bottom += d.y;
                                                            }
                                                            3 => {
                                                                right += d.x;
                                                                bottom += d.y;
                                                            }
                                                            _ => {}
                                                        }
                                                    }
                                                }

                                                let crop_screen = egui::Rect::from_min_max(
                                                    egui::pos2(left, top),
                                                    egui::pos2(right, bottom),
                                                );
                                                painter.rect_stroke(
                                                    crop_screen,
                                                    0.0,
                                                    egui::Stroke::new(
                                                        2.0,
                                                        egui::Color32::from_rgb(120, 230, 120),
                                                    ),
                                                );
                                                for p in [
                                                    egui::pos2(left, top),
                                                    egui::pos2(right, top),
                                                    egui::pos2(left, bottom),
                                                    egui::pos2(right, bottom),
                                                ] {
                                                    painter.circle_filled(
                                                        p,
                                                        handle_radius,
                                                        egui::Color32::from_rgb(120, 230, 120),
                                                    );
                                                }

                                                // Redraw D-min overlay above the darkened crop region.
                                                if opts.dmin_mode == DminMode::SampleRegion
                                                    && self.flat_field_path.is_none()
                                                {
                                                    if let Some(dmin_rect) = opts.dmin_rect {
                                                        let dmin_rect = scale_rect_to_size(
                                                            dmin_rect,
                                                            opts.dmin_rect_reference_size,
                                                            input_w,
                                                            input_h,
                                                        );
                                                        let dtl = image_to_screen(
                                                            dmin_rect.x as f32,
                                                            dmin_rect.y as f32,
                                                        );
                                                        let dbr = image_to_screen(
                                                            (dmin_rect.x + dmin_rect.width) as f32,
                                                            (dmin_rect.y + dmin_rect.height) as f32,
                                                        );
                                                        let dsr = egui::Rect::from_min_max(dtl, dbr);
                                                        painter.rect_stroke(
                                                            dsr,
                                                            0.0,
                                                            egui::Stroke::new(
                                                                1.5,
                                                                egui::Color32::from_rgb(255, 200, 0),
                                                            ),
                                                        );
                                                        for p in [
                                                            egui::pos2(dtl.x, dtl.y),
                                                            egui::pos2(dbr.x, dtl.y),
                                                            egui::pos2(dtl.x, dbr.y),
                                                            egui::pos2(dbr.x, dbr.y),
                                                        ] {
                                                            painter.circle_filled(
                                                                p,
                                                                5.0,
                                                                egui::Color32::from_rgb(255, 200, 0),
                                                            );
                                                        }
                                                    }
                                                }

                                                if rect_changed {
                                                    let (ix0, iy0) = screen_to_image(left, top);
                                                    let (ix1, iy1) = screen_to_image(right, bottom);
                                                    let iw = input_w as f32;
                                                    let ih = input_h as f32;
                                                    let x =
                                                        ix0.round().clamp(0.0, iw - 1.0) as u32;
                                                    let y =
                                                        iy0.round().clamp(0.0, ih - 1.0) as u32;
                                                    let x1 = ix1.round().clamp(0.0, iw) as u32;
                                                    let y1 = iy1.round().clamp(0.0, ih) as u32;
                                                    let mut wp = x1.saturating_sub(x).max(1);
                                                    let mut hp = y1.saturating_sub(y).max(1);
                                                    wp =
                                                        wp.min(input_w.saturating_sub(x).max(1));
                                                    hp =
                                                        hp.min(input_h.saturating_sub(y).max(1));
                                                    opts.crop_rect = Some(Rect {
                                                        x,
                                                        y,
                                                        width: wp,
                                                        height: hp,
                                                    });
                                                    opts.crop_rect_reference_size =
                                                        Some((input_w, input_h));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Info row: total resolution · crop resolution (if active) · zoom %
                        {
                            let entry = &self.images[idx];
                            let opts = &entry.options;

                            // True source resolution (full sensor dims, not preview working size).
                            let (src_w, src_h) = entry
                                .raw_source_size
                                .map(|s| (s[0], s[1]))
                                .unwrap_or((full_w, full_h));

                            // Crop resolution in source-pixel space (only when crop is active).
                            let crop_dims = if opts.apply_crop {
                                opts.crop_rect.map(|r| {
                                    let scaled = scale_rect_to_size(
                                        r,
                                        opts.crop_rect_reference_size,
                                        src_w,
                                        src_h,
                                    );
                                    (scaled.width.max(1), scaled.height.max(1))
                                })
                            } else {
                                None
                            };

                            // Photoshop-style zoom: 100 % = 1 image pixel : 1 screen pixel.
                            let zoom_pct = base_scale * entry.preview_zoom * 100.0;

                            let info_text = if let Some((cw, ch)) = crop_dims {
                                format!(
                                    "{} × {}  →  {} × {}  ·  {:.0}%",
                                    src_w, src_h, cw, ch, zoom_pct
                                )
                            } else {
                                format!("{} × {}  ·  {:.0}%", src_w, src_h, zoom_pct)
                            };

                            ui.allocate_ui(
                                egui::vec2(ui.available_width(), INFO_ROW_HEIGHT),
                                |ui| {
                                    ui.with_layout(
                                        egui::Layout::left_to_right(egui::Align::Center),
                                        |ui| {
                                            ui.label(
                                                egui::RichText::new(info_text)
                                                    .small()
                                                    .color(egui::Color32::from_gray(130)),
                                            );
                                        },
                                    );
                                },
                            );
                        }

                        // Row under the image: full filename (left) + full-res preview button + mirror/rotate buttons (right)
                        ui.horizontal(|ui| {
                            let full_name = self.images[idx].path.display().to_string();
                            let max_filename_w = (ui.available_width() - 310.0).max(80.0); // leave room for full-res button + mirror + rotate icons
                            ui.allocate_ui(egui::vec2(max_filename_w, CONTROL_ROW_HEIGHT), |ui| {
                                ui.label(
                                    egui::RichText::new(full_name).small().color(egui::Color32::from_gray(200)),
                                )
                                .on_hover_text(self.images[idx].path.display().to_string());
                            });
                            // Full resolution/bit depth preview: generates image with export pipeline.
                            // Deactivates on option change or image switch.
                            let full_res_clicked = ui
                                .selectable_label(
                                    self.full_res_preview_active,
                                    "Full resolution preview",
                                )
                                .on_hover_text("Use full resolution export pipeline for preview. Deactivates when adjusting settings or switching images.")
                                .clicked();
                            if full_res_clicked {
                                self.full_res_preview_active = !self.full_res_preview_active;
                                if self.full_res_preview_active {
                                    self.full_res_preview_button_clicked = true;
                                    self.images[idx].preview_texture = None; // Force re-render
                                    self.images[idx].preview_hash = 0; // Invalidate so need_new triggers
                                }
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                // Order: rotate right, rotate left, mirror right, mirror left (right to left)
                                let rotate_right_clicked = if let Some(icon) = &self.ui_icons.rotate_right {
                                    ui.add(egui::ImageButton::new((icon.id(), egui::vec2(20.0, 20.0))).frame(false))
                                        .on_hover_text("Rotate right")
                                        .clicked()
                                } else {
                                    ui.button("Rotate right").clicked()
                                };
                                if rotate_right_clicked {
                                    let entry = &mut self.images[idx];
                                    let preview_size =
                                        entry.preview_input_size.map(|[w, h]| (w, h));
                                    if let Some(rect) = entry.options.dmin_rect {
                                        let source_size = entry
                                            .options
                                            .dmin_rect_reference_size
                                            .or(preview_size);
                                        if let Some((w, h)) = source_size {
                                            entry.options.dmin_rect =
                                                Some(rotate_dmin_rect_90(rect, w, h, true));
                                            entry.options.dmin_rect_reference_size = Some((h, w));
                                        }
                                    }
                                    if let Some(rect) = entry.options.crop_rect {
                                        let source_size = entry
                                            .options
                                            .crop_rect_reference_size
                                            .or(preview_size);
                                        if let Some((w, h)) = source_size {
                                            entry.options.crop_rect =
                                                Some(rotate_dmin_rect_90(rect, w, h, true));
                                            entry.options.crop_rect_reference_size = Some((h, w));
                                        }
                                    }
                                    entry.options.rotation_degrees =
                                        (entry.options.rotation_degrees + 90).rem_euclid(360);
                                    self.preview_receiver = None;
                                }
                                let rotate_left_clicked = if let Some(icon) = &self.ui_icons.rotate_left {
                                    ui.add(egui::ImageButton::new((icon.id(), egui::vec2(20.0, 20.0))).frame(false))
                                        .on_hover_text("Rotate left")
                                        .clicked()
                                } else {
                                    ui.button("Rotate left").clicked()
                                };
                                if rotate_left_clicked {
                                    let entry = &mut self.images[idx];
                                    let preview_size =
                                        entry.preview_input_size.map(|[w, h]| (w, h));
                                    if let Some(rect) = entry.options.dmin_rect {
                                        let source_size = entry
                                            .options
                                            .dmin_rect_reference_size
                                            .or(preview_size);
                                        if let Some((w, h)) = source_size {
                                            entry.options.dmin_rect =
                                                Some(rotate_dmin_rect_90(rect, w, h, false));
                                            entry.options.dmin_rect_reference_size = Some((h, w));
                                        }
                                    }
                                    if let Some(rect) = entry.options.crop_rect {
                                        let source_size = entry
                                            .options
                                            .crop_rect_reference_size
                                            .or(preview_size);
                                        if let Some((w, h)) = source_size {
                                            entry.options.crop_rect =
                                                Some(rotate_dmin_rect_90(rect, w, h, false));
                                            entry.options.crop_rect_reference_size = Some((h, w));
                                        }
                                    }
                                    entry.options.rotation_degrees =
                                        (entry.options.rotation_degrees - 90).rem_euclid(360);
                                    self.preview_receiver = None;
                                }
                                let mirror_right_clicked = ui
                                    .small_button("↕")
                                    .on_hover_text("Mirror right (flip vertical)")
                                    .clicked();
                                if mirror_right_clicked {
                                    let entry = &mut self.images[idx];
                                    let preview_size =
                                        entry.preview_input_size.map(|[w, h]| (w, h));
                                    if let Some(rect) = entry.options.dmin_rect {
                                        let source_size = entry
                                            .options
                                            .dmin_rect_reference_size
                                            .or(preview_size);
                                        if let Some((w, h)) = source_size {
                                            entry.options.dmin_rect =
                                                Some(flip_rect_vertical(rect, w, h));
                                        }
                                    }
                                    if let Some(rect) = entry.options.crop_rect {
                                        let source_size = entry
                                            .options
                                            .crop_rect_reference_size
                                            .or(preview_size);
                                        if let Some((w, h)) = source_size {
                                            entry.options.crop_rect =
                                                Some(flip_rect_vertical(rect, w, h));
                                        }
                                    }
                                    entry.options.flip_vertical = !entry.options.flip_vertical;
                                    self.preview_receiver = None;
                                }
                                let mirror_left_clicked = ui
                                    .small_button("↔")
                                    .on_hover_text("Mirror left (flip horizontal)")
                                    .clicked();
                                if mirror_left_clicked {
                                    let entry = &mut self.images[idx];
                                    let preview_size =
                                        entry.preview_input_size.map(|[w, h]| (w, h));
                                    if let Some(rect) = entry.options.dmin_rect {
                                        let source_size = entry
                                            .options
                                            .dmin_rect_reference_size
                                            .or(preview_size);
                                        if let Some((w, h)) = source_size {
                                            entry.options.dmin_rect =
                                                Some(flip_rect_horizontal(rect, w, h));
                                        }
                                    }
                                    if let Some(rect) = entry.options.crop_rect {
                                        let source_size = entry
                                            .options
                                            .crop_rect_reference_size
                                            .or(preview_size);
                                        if let Some((w, h)) = source_size {
                                            entry.options.crop_rect =
                                                Some(flip_rect_horizontal(rect, w, h));
                                        }
                                    }
                                    entry.options.flip_horizontal = !entry.options.flip_horizontal;
                                    self.preview_receiver = None;
                                }
                            });
                        });
                        ui.add_space(BOTTOM_PADDING);
                        if let Some((r_hist, g_hist, b_hist)) = &self.images[idx].histogram {
                            const H_HIST: f32 = 72.0;
                            let (hist_rect, _) = ui.allocate_exact_size(
                                egui::vec2(ui.available_width(), H_HIST),
                                egui::Sense::hover(),
                            );
                            let painter = ui.painter_at(hist_rect);
                            let rect = hist_rect;
                            let draw_rect = rect.shrink(1.0);

                            // Fixed Y-scale: full height = this fraction of total pixels; above that we clip.
                            // 1% = full height so the curve typically uses most of the vertical space.
                            const FULL_HEIGHT_FRACTION: f32 = 0.01;
                            let total_pixels = r_hist.iter().sum::<u32>().max(1) as f32;
                            let scale_at_full = (total_pixels * FULL_HEIGHT_FRACTION).max(1.0);

                            // Axes: X (bottom) and Y (left). Draw first so bin 0 is not hidden by Y-axis.
                            let axis_color = egui::Color32::from_gray(100);
                            let stroke = egui::Stroke::new(1.0, axis_color);
                            painter.line_segment(
                                [egui::pos2(rect.left(), rect.bottom()), egui::pos2(rect.right(), rect.bottom())],
                                stroke,
                            );
                            painter.line_segment(
                                [egui::pos2(rect.left(), rect.bottom()), egui::pos2(rect.left(), rect.top())],
                                stroke,
                            );

                            // Photoshop/Resolve-style histogram rendering:
                            // per-channel curve (2px) + semi-transparent fill under each curve.
                            // Y-scale: full height = FULL_HEIGHT_FRACTION of pixels; clip above that.
                            let draw_channel =
                                |hist: &[u32; 256],
                                 line_color: egui::Color32,
                                 fill_color: egui::Color32,
                                 painter: &egui::Painter| {
                                    let mut curve_points = Vec::with_capacity(256);
                                    let w = draw_rect.width().max(1.0);
                                    for i in 0..256 {
                                        let x = draw_rect.left()
                                            + (i as f32 / 255.0) * w;
                                        let h_norm = (hist[i] as f32 / scale_at_full).min(1.0);
                                        let y = (draw_rect.bottom()
                                            - draw_rect.height() * h_norm)
                                            .clamp(draw_rect.top(), draw_rect.bottom());
                                        curve_points.push(egui::pos2(x, y));
                                    }

                                    // Per-bin vertical quads: straight down from curve to baseline.
                                    let y_base = draw_rect.bottom();
                                    for i in 0..255 {
                                        let p0 = curve_points[i];
                                        let p1 = curve_points[i + 1];
                                        let quad = vec![
                                            egui::pos2(p0.x, y_base),
                                            p0,
                                            p1,
                                            egui::pos2(p1.x, y_base),
                                        ];
                                        painter.add(egui::Shape::convex_polygon(
                                            quad,
                                            fill_color,
                                            egui::Stroke::NONE,
                                        ));
                                    }

                                    painter.add(egui::Shape::line(
                                        curve_points,
                                        egui::Stroke::new(1.0, line_color),
                                    ));
                                };

                            draw_channel(
                                r_hist,
                                egui::Color32::from_rgba_unmultiplied(220, 70, 70, 140),
                                egui::Color32::from_rgba_unmultiplied(200, 0, 0, 18),
                                &painter,
                            );
                            draw_channel(
                                g_hist,
                                egui::Color32::from_rgba_unmultiplied(80, 200, 80, 140),
                                egui::Color32::from_rgba_unmultiplied(0, 200, 0, 18),
                                &painter,
                            );
                            draw_channel(
                                b_hist,
                                egui::Color32::from_rgba_unmultiplied(80, 130, 240, 140),
                                egui::Color32::from_rgba_unmultiplied(0, 80, 220, 18),
                                &painter,
                            );
                        }
                    }
                }
            } else {
                ui.vertical_centered(|ui| {
                    ui.add_space(ui.available_height() / 2.0 - 20.0);
                    ui.label("Select an image in the strip below to see a preview.");
                });
            }
        });

        // If selected image has no preview or options changed, request a new one.
        // Runs after UI interactions so drag/release state is available for debounce.
        if let Some(idx) = self.selected_index {
            if idx < self.images.len() {
                // Recompute hash including zoom and full-res mode so changes trigger a new preview render.
                let base_hash = options_hash_for(&self.images[idx].path, &self.images[idx].options);
                let mut hh = DefaultHasher::new();
                base_hash.hash(&mut hh);
                self.images[idx].preview_zoom.to_bits().hash(&mut hh);
                self.full_res_preview_active.hash(&mut hh);
                let hash_now = hh.finish();
                let need_new = self.images[idx].preview_texture.is_none()
                    || self.images[idx].preview_hash != hash_now;
                if need_new {
                    let now = Instant::now();
                    let key = (idx, hash_now);
                    if self.pending_preview_key != Some(key) {
                        self.pending_preview_key = Some(key);
                        self.pending_preview_since = Some(now);
                    }

                    let gpu_active = {
                        #[cfg(feature = "gpu")]
                        { self.gpu_pipeline.is_some() && self.images[idx].options.use_gpu }
                        #[cfg(not(feature = "gpu"))]
                        { false }
                    };
                    let waiting_for_release = if gpu_active {
                        self.rect_dragging
                    } else {
                        self.rect_dragging || ctx.input(|i| i.pointer.any_down())
                    };
                    let debounce_ms = if gpu_active { 0 } else { PREVIEW_DEBOUNCE_MS };
                    let settled = self
                        .pending_preview_since
                        .map(|t| now.saturating_duration_since(t) >= Duration::from_millis(debounce_ms))
                        .unwrap_or(false);

                    if self.preview_receiver.is_none() && !waiting_for_release && settled {
                        // Deactivate full-res when preview is triggered by option/zoom change (not by the button).
                        if self.full_res_preview_active && !self.full_res_preview_button_clicked {
                            self.full_res_preview_active = false;
                        }
                        self.request_preview_for(idx, ctx);
                        self.full_res_preview_button_clicked = false;
                        self.pending_preview_since = None;
                    } else {
                        // Keep ticking while debounce/release conditions are pending.
                        ctx.request_repaint_after(Duration::from_millis(16));
                    }
                } else if self.pending_preview_key.map(|(i, _)| i) == Some(idx) {
                    self.pending_preview_key = None;
                    self.pending_preview_since = None;
                }
            }
        } else {
            self.pending_preview_key = None;
            self.pending_preview_since = None;
        }
    }
}

