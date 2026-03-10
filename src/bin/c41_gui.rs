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
    demosaic,
    dmin,
    load_flat_field_linear,
    png_reader,
    process_files,
    process_one_to_preview,
    raw_reader,
    tiff_export,
    PipelineOptions,
    Rect,
    TiffFormat,
    OutputStage,
    OutputLutEncoding,
};
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
    preview_texture: Option<egui::TextureHandle>,
    preview_hash: u64,
    /// Dimensions of the image at the stage where D-min/flat-field are applied (before preview downscale).
    preview_input_size: Option<[u32; 2]>,
    /// Small thumbnail for the image strip (generated when loading).
    thumbnail_texture: Option<egui::TextureHandle>,
    // Per-channel histograms (R, G, B) over 0–255
    histogram: Option<([u32; 256], [u32; 256], [u32; 256])>,
    export_format: ExportFormat,
    /// Rawloader debug report for this file (Debug tab).
    raw_debug_report: Option<String>,
    /// Pipeline debug log from the most recent preview render.
    pipeline_debug_log: Option<String>,
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
    preview_receiver: Option<mpsc::Receiver<anyhow::Result<(usize, u32, u32, u32, u32, Vec<u8>, String, bool)>>>,
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
    /// Camera IDT profiles loaded from camera_idt/ (path, profile).
    idt_profiles: Vec<(PathBuf, c41_raw_tool::aces::IdtProfile)>,
    ui_icons: UiIcons,
    /// Suppresses preview reprocessing while the user is dragging a rect handle (crop / d-min).
    rect_dragging: bool,
    /// Debounce state for preview refreshes: (image index, options hash) currently waiting to settle.
    pending_preview_key: Option<(usize, u64)>,
    pending_preview_since: Option<Instant>,
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
            idt_profiles: Vec::new(),
            ui_icons: UiIcons::default(),
            rect_dragging: false,
            pending_preview_key: None,
            pending_preview_since: None,
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
        "png" => png_reader::load_png_as_ndarray(path)?,
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
        apply_dmin: true,
        apply_white_balance: true,
        auto_wb: true,
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
        apply_color_profile: true,
        density_matrix: [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        flat_field_path: None,
        idt_matrix: c41_raw_tool::aces::IDT_IDENTITY,
        export_aces_exr: false,
        write_aces2065_only: false,
        lut3d_path: None,
        output_stage: OutputStage::Ra4,
        output_lut_cube: None,
        output_lut_encoding: OutputLutEncoding::CineonLog,
        lut_in_black: 0.0,
        lut_in_white: 1.0,
        rotation_degrees: 0,
        debug_pipeline_step: 6,
        debug_preview_simple_debayer: false,
        verbose_debug: false,
    }
}

fn options_hash_for(path: &PathBuf, opts: &PipelineOptions) -> u64 {
    let mut h = DefaultHasher::new();
    path.display().to_string().hash(&mut h);
    opts.apply_dmin.hash(&mut h);
    opts.apply_white_balance.hash(&mut h);
    opts.auto_wb.hash(&mut h);
    opts.film_gamma.to_bits().hash(&mut h);
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
    for row in &opts.idt_matrix {
        for v in row {
            v.to_bits().hash(&mut h);
        }
    }
    opts.lut3d_path.as_ref().map(|p| p.display().to_string()).hash(&mut h);
    opts.output_stage.hash(&mut h);
    opts.output_lut_cube
        .as_ref()
        .map(|p| p.display().to_string())
        .hash(&mut h);
    opts.output_lut_encoding.hash(&mut h);
    opts.lut_in_black.to_bits().hash(&mut h);
    opts.lut_in_white.to_bits().hash(&mut h);
    opts.rotation_degrees.hash(&mut h);
    opts.debug_pipeline_step.hash(&mut h);
    opts.debug_preview_simple_debayer.hash(&mut h);
    opts.verbose_debug.hash(&mut h);
    h.finish()
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
        let mut options = self.images[index].options.clone();
        options.flat_field_path = self.flat_field_path.clone();
        let capture_debug = self.capture_pipeline_debug_next;
        self.capture_pipeline_debug_next = false;
        options.verbose_debug = capture_debug;
        let (tx, rx) = mpsc::channel();
        self.preview_receiver = Some(rx);
        self.preview_started_at = Some(Instant::now());
        thread::spawn(move || {
            let res = process_one_to_preview(
                &path,
                &options,
                PREVIEW_MAX_WIDTH,
                PREVIEW_MAX_HEIGHT,
            )
            .map(|(input_w, input_h, w, h, rgb, dbg_log)| {
                (index, input_w, input_h, w, h, rgb, dbg_log, capture_debug)
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

        // Poll preview worker
        if let Some(rx) = self.preview_receiver.as_ref() {
            match rx.try_recv() {
                Ok(Ok((idx, input_w, input_h, w, h, rgb, dbg_log, captured_debug))) => {
                    self.preview_receiver = None;
                    self.preview_started_at = None;
                    if idx < self.images.len() {
                        let size = [w as usize, h as usize];
                        let mut r_hist = [0u32; 256];
                        let mut g_hist = [0u32; 256];
                        let mut b_hist = [0u32; 256];
                        let crop_in_preview = {
                            let opts = &self.images[idx].options;
                            if opts.apply_crop {
                                if let Some(crop_rect) = opts.crop_rect {
                                    let scaled = scale_rect_to_size(
                                        crop_rect,
                                        opts.crop_rect_reference_size,
                                        input_w,
                                        input_h,
                                    );
                                    Some(scale_rect_to_size(
                                        scaled,
                                        Some((input_w, input_h)),
                                        w,
                                        h,
                                    ))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        };

                        let mut pixels: Vec<egui::Color32> = Vec::with_capacity(size[0] * size[1]);
                        for (i, c) in rgb.chunks_exact(3).enumerate() {
                            let x = (i % size[0]) as u32;
                            let y = (i / size[0]) as u32;

                            let in_hist_crop = if let Some(rect) = crop_in_preview {
                                let x0 = rect.x.min(w.saturating_sub(1));
                                let y0 = rect.y.min(h.saturating_sub(1));
                                let x1 = (rect.x + rect.width).min(w).max(x0 + 1);
                                let y1 = (rect.y + rect.height).min(h).max(y0 + 1);
                                x >= x0 && x < x1 && y >= y0 && y < y1
                            } else {
                                true
                            };

                            if in_hist_crop {
                                let r = c[0] as usize;
                                let g = c[1] as usize;
                                let b = c[2] as usize;
                                r_hist[r] += 1;
                                g_hist[g] += 1;
                                b_hist[b] += 1;
                            }
                            pixels.push(egui::Color32::from_rgb(c[0], c[1], c[2]));
                        }
                        let image = egui::ColorImage { size, pixels };
                        let tex = ctx.load_texture(
                            format!("preview_{}", idx),
                            image,
                            egui::TextureOptions::default(),
                        );
                        let hash = options_hash_for(&self.images[idx].path, &self.images[idx].options);
                        self.images[idx].preview_texture = Some(tex);
                        self.images[idx].preview_hash = hash;
                        self.images[idx].preview_input_size = Some([input_w, input_h]);
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
                        if self.images[idx].options.dmin_rect.is_some() {
                            self.images[idx].options.dmin_rect_reference_size =
                                Some((input_w, input_h));
                        }
                        if self.images[idx].options.crop_rect.is_some() {
                            self.images[idx].options.crop_rect_reference_size =
                                Some((input_w, input_h));
                        }
                        self.images[idx].histogram = Some((r_hist, g_hist, b_hist));
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
                                    "RAW & PNG",
                                    &[
                                        "arw", "nef", "nrw", "cr2", "cr3", "crw", "dng", "raf", "orf", "rw2", "png",
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
                                            preview_input_size: None,
                                            thumbnail_texture: None,
                                            histogram: None,
                                            export_format: ExportFormat::Tiff16,
                                            raw_debug_report: None,
                                            pipeline_debug_log: None,
                                        });
                                        if self.selected_index.is_none() {
                                            self.selected_index = Some(self.images.len() - 1);
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
                                }

                                if i + 1 < self.images.len() {
                                    ui.add_space(CARD_GAP);
                                }
                            }
                        });
                    });
                    let had_removals = !to_remove.is_empty();
                    if had_removals {
                        self.preview_receiver = None;
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

                if self.mode == UIMode::Debug {
                    ui.label("Pipeline step (1–6). Preview shows output up to this step.");
                    ui.add(egui::Slider::new(&mut opts.debug_pipeline_step, 1..=6).text("Step"));
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
                            ui.label(
                                egui::RichText::new("simple RAW bilinear debayer mode ON")
                                    .small()
                                    .weak(),
                            );
                        }
                    });
                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new("1: load+demosaic+rot · 2: +IDT · 3: +D-min · 4: +WB · 6: full (curve/invert)")
                            .small(),
                    );
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new("Step 5 applies density matrix / LUT calibration.")
                            .small(),
                    );
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
                        ui.label(
                            egui::RichText::new("Selected file is not a RAW format.")
                                .small()
                                .weak(),
                        );
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
                        ui.label(
                            egui::RichText::new("No raw report yet. Click the button above.")
                                .small()
                                .weak(),
                        );
                    }

                    ui.add_space(8.0);
                    ui.separator();
                    ui.label(egui::RichText::new("Pipeline debug log").strong());
                    ui.label(
                        egui::RichText::new("Captured on demand. Click the button to snapshot current settings.")
                            .small()
                            .weak(),
                    );
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        if ui.button("Capture pipeline log for current settings").clicked() {
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
                        ui.label(
                            egui::RichText::new("No pipeline log yet. Preview must render first.")
                                .small()
                                .weak(),
                        );
                    }
                } else if self.mode != UIMode::LuminanceCalibrate {
                    // D-min, White balance, and Print curve apply only to normal processing.
                    ui.label("Camera IDT profile");
                    let current_label = if opts.idt_matrix == c41_raw_tool::aces::IDT_IDENTITY {
                        "Identity"
                    } else if let Some((_, p)) = self.idt_profiles.iter().find(|(_, p)| {
                        p.matrix.iter().zip(opts.idt_matrix.iter()).all(|(a, b)| {
                            a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-5)
                        })
                    }) {
                        p.name.as_str()
                    } else {
                        "Custom"
                    };
                    egui::ComboBox::from_label("IDT profile")
                        .selected_text(current_label)
                        .show_ui(ui, |ui| {
                            let base_dir = std::env::current_dir()
                                .unwrap_or_else(|_| PathBuf::from("."))
                                .join("camera_idt");
                            if let Ok(list) = c41_raw_tool::aces::load_idt_profiles_from_dir(&base_dir) {
                                self.idt_profiles = list;
                            }
                            if ui.selectable_label(
                                opts.idt_matrix == c41_raw_tool::aces::IDT_IDENTITY,
                                "Identity",
                            ).clicked()
                            {
                                opts.idt_matrix = c41_raw_tool::aces::IDT_IDENTITY;
                            }
                            for (_, profile) in &self.idt_profiles {
                                let selected = opts.idt_matrix
                                    .iter()
                                    .zip(profile.matrix.iter())
                                    .all(|(a, b)| a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-5));
                                if ui.selectable_label(selected, &profile.name).clicked() {
                                    opts.idt_matrix = profile.matrix;
                                }
                            }
                        });
                    ui.collapsing("IDT matrix (custom edit)", |ui| {
                        let m = &mut opts.idt_matrix;
                        ui.horizontal(|ui| {
                            for row in 0..3 {
                                for col in 0..3 {
                                    ui.add(drag_decimal_f32(&mut m[row][col]).speed(0.05));
                                }
                            }
                        });
                    });
                    ui.separator();

                    ui.checkbox(&mut opts.apply_dmin, "D-min");
                    if opts.apply_dmin {
                    ui.collapsing("D-min settings", |ui| {
                        ui.horizontal(|ui| {
                            if ui.button("Copy D-min").clicked() {
                                if let Some(text) = dmin_values_to_clipboard_text(opts) {
                                    ui.ctx().copy_text(text.clone());
                                    self.status = format!("D-min copied: {}", text);
                                } else {
                                    self.status =
                                        "No D-min values to copy (enable fixed or rectangle first).".to_string();
                                }
                            }
                            if ui.button("Paste D-min").clicked() {
                                match arboard::Clipboard::new()
                                    .and_then(|mut cb| cb.get_text())
                                {
                                    Ok(text) => {
                                        if let Some((fixed, rect, neutral_only)) =
                                            parse_dmin_clipboard_text(&text)
                                        {
                                            if let Some((r, g, b)) = fixed {
                                                opts.apply_dmin = true;
                                                opts.dmin_fixed = Some((r, g, b));
                                                opts.dmin_rect = None;
                                                opts.dmin_rect_reference_size = None;
                                                if let Some(v) = neutral_only {
                                                    opts.dmin_neutral_only = v;
                                                }
                                                self.status = format!(
                                                    "Applied pasted D-min fixed values: {:.6}, {:.6}, {:.6}",
                                                    r, g, b
                                                );
                                            } else if let Some(rect) = rect {
                                                opts.apply_dmin = true;
                                                opts.dmin_fixed = None;
                                                opts.dmin_rect = Some(rect);
                                                opts.dmin_rect_reference_size = None;
                                                if let Some(v) = neutral_only {
                                                    opts.dmin_neutral_only = v;
                                                }
                                                self.status = format!(
                                                    "Applied pasted D-min rectangle: {},{},{},{}",
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
                        ui.label(
                            egui::RichText::new("Format: dmin:fixed:r,g,b or dmin:rect:x,y,w,h")
                                .small()
                                .weak(),
                        );
                        ui.add_space(4.0);

                        // Option 1: classic D-min (fixed or crop) when no flat-field override is set.
                        let mut use_fixed = opts.dmin_fixed.is_some();
                        ui.checkbox(&mut use_fixed, "Use fixed D-min (R,G,B)");
                        if use_fixed {
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
                            opts.dmin_rect = None;
                            opts.dmin_rect_reference_size = None;
                        } else {
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
                                    ui.label("x,y,w,h");
                                    ui.add(egui::DragValue::new(&mut rect.x).speed(1));
                                    ui.add(egui::DragValue::new(&mut rect.y).speed(1));
                                    ui.add(egui::DragValue::new(&mut rect.width).speed(1));
                                    ui.add(egui::DragValue::new(&mut rect.height).speed(1));
                                });
                            }
                            opts.dmin_fixed = None;
                        }

                        ui.separator();
                        ui.label("Flat-field override (luminance calibration)");
                        ui.horizontal(|ui| {
                            if ui.button("Load flat-field map…").clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter(
                                        "Flat field",
                                        &[
                                            "tif", "tiff", // 32f TIFF profiles
                                            "arw", "nef", "nrw", "cr2", "cr3", "crw", "dng", "raf",
                                            "orf", "rw2", // RAW empty-frame
                                            "png",
                                        ],
                                    )
                                    .pick_file()
                                {
                                    self.flat_field_path = Some(path.clone());
                                    // When flat-field is active, disable per-image D-min.
                                    opts.dmin_fixed = None;
                                    opts.dmin_rect = None;
                                    opts.dmin_rect_reference_size = None;
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
                                opts.dmin_rect_reference_size = None;
                                self.status =
                                    "Flat-field override cleared; D-min settings are active again."
                                        .to_string();
                            }
                        });
                        if let Some(ref p) = self.flat_field_path {
                            ui.label(
                                egui::RichText::new(format!("Flat-field: {}", p.display())).small(),
                            );
                        } else {
                            ui.label(egui::RichText::new("No flat-field override set.").small());
                        }
                    });
                    }

                    ui.checkbox(&mut opts.apply_crop, "Crop");
                    if opts.apply_crop {
                        if opts.crop_rect.is_none() {
                            opts.crop_rect = Some(Rect {
                                x: 40,
                                y: 40,
                                width: 240,
                                height: 240,
                            });
                        }
                        ui.collapsing("Crop settings", |ui| {
                            if let Some(rect) = opts.crop_rect.as_mut() {
                                ui.horizontal(|ui| {
                                    ui.label("x,y,w,h");
                                    ui.add(egui::DragValue::new(&mut rect.x).speed(1));
                                    ui.add(egui::DragValue::new(&mut rect.y).speed(1));
                                    ui.add(egui::DragValue::new(&mut rect.width).speed(1));
                                    ui.add(egui::DragValue::new(&mut rect.height).speed(1));
                                });
                            }
                            ui.label(
                                egui::RichText::new(
                                    "Preview darkens outside crop. Histogram + export use inside only.",
                                )
                                .small()
                                .weak(),
                            );
                        });
                    }

                    ui.checkbox(&mut opts.auto_wb, "Auto white balance");
                    ui.label(
                        egui::RichText::new(
                            if opts.auto_wb {
                                "Per-channel gamma correction: D x (mean_D / ch_median). Preserves black point."
                            } else {
                                "Auto WB disabled."
                            },
                        )
                        .small()
                        .weak(),
                    );

                    ui.checkbox(&mut opts.apply_white_balance, "Manual white balance");
                    if opts.apply_white_balance {
                    ui.collapsing("White balance settings", |ui| {
                        let mut use_temp = opts.temp_k.is_some();
                        ui.checkbox(&mut use_temp, "Use color temperature (K)");
                        if use_temp {
                            let mut k = opts.temp_k.unwrap_or(5500.0);
                            ui.add(egui::Slider::new(&mut k, 2000.0..=12000.0).suffix(" K"));
                            opts.temp_k = Some(k);
                        } else {
                            opts.temp_k = None;
                        }
                        ui.label(egui::RichText::new("Density scale (1.0 = neutral, >1 = more color)").small().weak());
                        ui.horizontal(|ui| {
                            ui.label("R");
                            ui.add(egui::Slider::new(&mut opts.wb_r, 0.5..=2.0));
                        });
                        ui.horizontal(|ui| {
                            ui.label("G");
                            ui.add(egui::Slider::new(&mut opts.wb_g, 0.5..=2.0));
                        });
                        ui.horizontal(|ui| {
                            ui.label("B");
                            ui.add(egui::Slider::new(&mut opts.wb_b, 0.5..=2.0));
                        });
                    });
                    }

                    ui.collapsing("Film gamma", |ui| {
                        ui.label(
                            egui::RichText::new(
                                "C-41 γ ≈ 0.55–0.75. Converts density → scene log-exposure: D/γ"
                            )
                            .small()
                            .weak(),
                        );
                        ui.add(egui::Slider::new(&mut opts.film_gamma, 0.3..=1.2).text("γ"));
                    });

                    ui.collapsing("Levels (black / white point)", |ui| {
                        ui.label(
                            egui::RichText::new(
                                "Remap density before the output stage: [black, white] → [0, 1].\n\
                                 Applied after density calibration (step 5), before the output curve.\n\
                                 Default 0–1 = full range. Narrow the range to set exposure/contrast.",
                            )
                            .small()
                            .weak(),
                        );
                        ui.horizontal(|ui| {
                            ui.label("Black");
                            ui.add(
                                egui::Slider::new(&mut opts.lut_in_black, 0.0..=1.0)
                                    .fixed_decimals(3),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("White");
                            ui.add(
                                egui::Slider::new(&mut opts.lut_in_white, 0.0..=1.0)
                                    .fixed_decimals(3),
                            );
                        });
                        if opts.lut_in_black >= opts.lut_in_white {
                            ui.label(
                                egui::RichText::new("Warning: Black must be less than White")
                                    .small()
                                    .color(egui::Color32::from_rgb(255, 160, 0)),
                            );
                        }
                    });

                    // Output stage / curve selection.
                    let mut apply_curve = !opts.no_curve;
                    if ui.checkbox(&mut apply_curve, "Output curve").changed() {
                        if apply_curve {
                            // Re-enable output stage: if we were in "None", default back to RA-4.
                            if matches!(opts.output_stage, OutputStage::None) {
                                opts.output_stage = OutputStage::Ra4;
                            }
                            opts.no_curve = false;
                        } else {
                            // Disable output stage completely.
                            opts.no_curve = true;
                            opts.output_stage = OutputStage::None;
                        }
                    }

                    if apply_curve {
                    let current_label = match opts.output_stage {
                        OutputStage::Ra4 => "RA-4 print emulation",
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
                        ui.collapsing("RA-4 curve settings", |ui| {
                            ui.horizontal(|ui| {
                                ui.label("Offset");
                                ui.add(
                                    drag_decimal_f32(&mut opts.curve_offset)
                                        .range(-2.0..=2.0)
                                        .speed(0.05),
                                );
                            });
                            ui.horizontal(|ui| {
                                ui.label("Gamma");
                                ui.add(egui::Slider::new(&mut opts.curve_gamma, 0.5..=5.0));
                            });
                            ui.horizontal(|ui| {
                                ui.label("Pivot");
                                ui.add(
                                    drag_decimal_f32(&mut opts.curve_pivot)
                                        .range(0.1..=10.0)
                                        .speed(0.1),
                                );
                            });
                            ui.horizontal(|ui| {
                                ui.label("White");
                                ui.add(egui::Slider::new(&mut opts.curve_white, -1.0..=2.0));
                            });
                        });
                    }

                    if matches!(opts.output_stage, OutputStage::Lut2383) {
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new(
                                "Resolve-style Kodak 2383 cubes expect Cineon log input.",
                            )
                            .small()
                            .weak(),
                        );

                        let enc_label = match opts.output_lut_encoding {
                            OutputLutEncoding::CineonLog => "Cineon log (D ÷ 2.046)",
                            OutputLutEncoding::Rec709 => "Rec.709 (sRGB gamma)",
                            OutputLutEncoding::LinearDensity => "Linear (D ÷ 2.5)",
                        };
                        egui::ComboBox::from_label("LUT input encoding")
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

                        ui.add_space(4.0);
                        if ui.button("Browse output LUT…").clicked() {
                            let picked = rfd::FileDialog::new()
                                .add_filter("3D LUT (.cube)", &["cube", "CUBE"])
                                .add_filter("All files", &["*"])
                                .pick_file();
                            match picked {
                                Some(path) => {
                                    match c41_raw_tool::lut3d::read_cube(&path) {
                                        Ok(lut) => {
                                            let msg = format!(
                                                "Loaded output LUT: {} ({}³ grid)",
                                                path.display(),
                                                lut.size,
                                            );
                                            opts.output_lut_cube = Some(path);
                                            self.status = msg;
                                        }
                                        Err(e) => {
                                            self.status = format!(
                                                "Failed to parse output LUT {}: {}",
                                                path.display(),
                                                e
                                            );
                                        }
                                    }
                                }
                                None => {
                                    self.status = "Output LUT: file dialog cancelled.".into();
                                }
                            }
                        }
                        if let Some(ref p) = opts.output_lut_cube {
                            ui.label(egui::RichText::new(p.display().to_string()).small());
                        } else {
                            ui.label(egui::RichText::new("No output LUT loaded").small().weak());
                        }
                    }
                    }

                    // Pipeline: inversion (1-x). Only applies when no curve stage is used.
                    ui.add_enabled(
                        opts.no_curve,
                        egui::Checkbox::new(&mut opts.no_invert, "Skip color inversion"),
                    );
                    if !opts.no_curve {
                        ui.label(egui::RichText::new("(Applies when Output curve is off)").small());
                    }
                }

                if self.mode == UIMode::Calibrate {
                    ui.add_space(12.0);
                    ui.separator();
                    ui.add_space(8.0);
                    ui.label("Set the profile name / film stock and notes, then create the color profile in one step (matrix + 3D LUT saved as .c41).");
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
                    ui.label("Profile name / film stock");
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

                if self.mode == UIMode::Process {
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

                if self.mode == UIMode::Process {
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
                        for img in &self.images {
                            let mut opts = img.options.clone();
                            opts.flat_field_path = self.flat_field_path.clone();
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
                    if let Some(ref tex) = self.images[idx].preview_texture {
                        let size = tex.size();
                        let (w, h) = (size[0] as f32, size[1] as f32);
                        let available = ui.available_rect_before_wrap();
                        const CONTROL_ROW_HEIGHT: f32 = 28.0;
                        const HISTOGRAM_HEIGHT: f32 = 72.0;
                        const BOTTOM_PADDING: f32 = 8.0;
                        const IMAGE_PREVIEW_BOTTOM_PADDING: f32 = 16.0; // padding below the image
                        let reserved_bottom = IMAGE_PREVIEW_BOTTOM_PADDING + CONTROL_ROW_HEIGHT + BOTTOM_PADDING + HISTOGRAM_HEIGHT + BOTTOM_PADDING;
                        let area_for_image = (available.height() - reserved_bottom).max(60.0);
                        let scale = (available.width() / w).min(area_for_image / h).min(1.0);
                        let display_size = egui::vec2(w * scale, h * scale);
                        let margin_x = (available.width() - display_size.x) / 2.0;
                        let margin_y = (area_for_image - display_size.y) / 2.0;
                        ui.add_space(margin_y);
                        let image_resp = ui.horizontal(|ui| {
                            ui.add_space(margin_x);
                            ui.image((tex.id(), display_size))
                        }).inner;
                        let image_rect = image_resp.rect;
                        ui.add_space(IMAGE_PREVIEW_BOTTOM_PADDING);

                        // Keep the previous preview visible while a new render is in-flight.
                        // Only show loading overlay for unusually slow renders.
                        if show_loader {
                            let overlay_painter = ui.painter_at(image_rect);
                            overlay_painter.rect_filled(
                                image_rect,
                                0.0,
                                egui::Color32::from_rgba_premultiplied(0, 0, 0, 80),
                            );
                            let spinner_rect =
                                egui::Rect::from_center_size(image_rect.center(), egui::vec2(22.0, 22.0));
                            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(spinner_rect), |ui| {
                                ui.centered_and_justified(|ui| {
                                    ui.spinner();
                                });
                            });
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

                        // In Process mode, when D-min is active and using a rectangle (no flat-field /
                        // fixed D-min), draw the D-min sampling rectangle over the preview.
                        if self.mode == UIMode::Process {
                            if let Some(entry) = self.images.get_mut(idx) {
                                let opts = &mut entry.options;
                                if opts.apply_dmin
                                    && opts.dmin_fixed.is_none()
                                    && self.flat_field_path.is_none()
                                {
                                    if let (Some(rect), Some([input_w, input_h])) =
                                        (opts.dmin_rect, entry.preview_input_size)
                                    {
                                        if input_w > 0 && input_h > 0 {
                                            let painter = ui.painter_at(image_rect);
                                            let norm_x = rect.x as f32 / input_w as f32;
                                            let norm_y = rect.y as f32 / input_h as f32;
                                            let norm_w = rect.width as f32 / input_w as f32;
                                            let norm_h = rect.height as f32 / input_h as f32;

                                            let mut left =
                                                image_rect.left() + norm_x * image_rect.width();
                                            let mut top =
                                                image_rect.top() + norm_y * image_rect.height();
                                            let mut right = left + norm_w * image_rect.width();
                                            let mut bottom = top + norm_h * image_rect.height();

                                            // Draggable corner handles for direct D-min crop editing.
                                            let handle_radius = 5.0;
                                            let min_screen_size = 8.0;
                                            let mut rect_changed = false;

                                            let corners = [
                                                egui::pos2(left, top),
                                                egui::pos2(right, top),
                                                egui::pos2(left, bottom),
                                                egui::pos2(right, bottom),
                                            ];

                                            for (corner_idx, corner_pos) in corners.iter().enumerate() {
                                                let handle_rect = egui::Rect::from_center_size(
                                                    *corner_pos,
                                                    egui::vec2(handle_radius * 2.0, handle_radius * 2.0),
                                                );
                                                let id =
                                                    ui.make_persistent_id(("dmin_rect_handle", idx, corner_idx));
                                                let resp = ui.interact(
                                                    handle_rect,
                                                    id,
                                                    egui::Sense::click_and_drag(),
                                                );
                                                if resp.dragged() {
                                                    let delta = resp.drag_delta();
                                                    rect_changed = true;
                                                    self.rect_dragging = true;
                                                    match corner_idx {
                                                        0 => {
                                                            left += delta.x;
                                                            top += delta.y;
                                                        }
                                                        1 => {
                                                            right += delta.x;
                                                            top += delta.y;
                                                        }
                                                        2 => {
                                                            left += delta.x;
                                                            bottom += delta.y;
                                                        }
                                                        3 => {
                                                            right += delta.x;
                                                            bottom += delta.y;
                                                        }
                                                        _ => {}
                                                    }
                                                }
                                            }

                                            if rect_changed {
                                                left = left.clamp(
                                                    image_rect.left(),
                                                    image_rect.right() - min_screen_size,
                                                );
                                                top = top.clamp(
                                                    image_rect.top(),
                                                    image_rect.bottom() - min_screen_size,
                                                );
                                                right = right
                                                    .clamp(left + min_screen_size, image_rect.right());
                                                bottom = bottom
                                                    .clamp(top + min_screen_size, image_rect.bottom());
                                            } else {
                                                let screen_rect =
                                                    egui::Rect::from_min_max(
                                                        egui::pos2(left, top),
                                                        egui::pos2(right, bottom),
                                                    );
                                                let move_id =
                                                    ui.make_persistent_id(("dmin_rect_move", idx));
                                                let move_resp = ui.interact(
                                                    screen_rect,
                                                    move_id,
                                                    egui::Sense::click_and_drag(),
                                                );
                                                if move_resp.dragged() {
                                                    let delta = move_resp.drag_delta();
                                                    let rect_w = right - left;
                                                    let rect_h = bottom - top;

                                                    let mut dx = delta.x;
                                                    let mut dy = delta.y;
                                                    dx = dx
                                                        .max(image_rect.left() - left)
                                                        .min(image_rect.right() - right);
                                                    dy = dy
                                                        .max(image_rect.top() - top)
                                                        .min(image_rect.bottom() - bottom);

                                                    left += dx;
                                                    right = left + rect_w;
                                                    top += dy;
                                                    bottom = top + rect_h;
                                                    rect_changed = true;
                                                    self.rect_dragging = true;
                                                }
                                            }

                                            let screen_rect = egui::Rect::from_min_max(
                                                egui::pos2(left, top),
                                                egui::pos2(right, bottom),
                                            );
                                            painter.rect_stroke(
                                                screen_rect,
                                                0.0,
                                                egui::Stroke::new(
                                                    1.5,
                                                    egui::Color32::from_rgb(255, 200, 0),
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
                                                    egui::Color32::from_rgb(255, 200, 0),
                                                );
                                            }

                                            if rect_changed {
                                                let img_w = image_rect.width().max(1.0);
                                                let img_h = image_rect.height().max(1.0);

                                                let norm_left = (left - image_rect.left()) / img_w;
                                                let norm_top = (top - image_rect.top()) / img_h;
                                                let norm_width = (right - left) / img_w;
                                                let norm_height = (bottom - top) / img_h;

                                                let x = (norm_left * input_w as f32)
                                                    .round()
                                                    .clamp(0.0, input_w.saturating_sub(1) as f32)
                                                    as u32;
                                                let y = (norm_top * input_h as f32)
                                                    .round()
                                                    .clamp(0.0, input_h.saturating_sub(1) as f32)
                                                    as u32;
                                                let mut w_px =
                                                    (norm_width * input_w as f32).round().max(1.0) as u32;
                                                let mut h_px =
                                                    (norm_height * input_h as f32).round().max(1.0) as u32;
                                                w_px = w_px.min(input_w.saturating_sub(x).max(1));
                                                h_px = h_px.min(input_h.saturating_sub(y).max(1));

                                                opts.dmin_rect = Some(Rect {
                                                    x,
                                                    y,
                                                    width: w_px,
                                                    height: h_px,
                                                });
                                                opts.dmin_rect_reference_size = Some((input_w, input_h));
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // In Process mode, draw/edit crop overlay (optional):
                        // darken outside region and allow drag handles.
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
                                            let norm_x = crop.x as f32 / input_w as f32;
                                            let norm_y = crop.y as f32 / input_h as f32;
                                            let norm_w = crop.width as f32 / input_w as f32;
                                            let norm_h = crop.height as f32 / input_h as f32;

                                            let mut left =
                                                image_rect.left() + norm_x * image_rect.width();
                                            let mut top =
                                                image_rect.top() + norm_y * image_rect.height();
                                            let mut right = left + norm_w * image_rect.width();
                                            let mut bottom = top + norm_h * image_rect.height();

                                            let handle_radius = 5.0;
                                            let min_screen_size = 8.0;
                                            let mut rect_changed = false;

                                            let corners = [
                                                egui::pos2(left, top),
                                                egui::pos2(right, top),
                                                egui::pos2(left, bottom),
                                                egui::pos2(right, bottom),
                                            ];
                                            for (corner_idx, corner_pos) in corners.iter().enumerate() {
                                                let handle_rect = egui::Rect::from_center_size(
                                                    *corner_pos,
                                                    egui::vec2(handle_radius * 2.0, handle_radius * 2.0),
                                                );
                                                let id =
                                                    ui.make_persistent_id(("crop_rect_handle", idx, corner_idx));
                                                let resp = ui.interact(
                                                    handle_rect,
                                                    id,
                                                    egui::Sense::click_and_drag(),
                                                );
                                                if resp.dragged() {
                                                    let delta = resp.drag_delta();
                                                    rect_changed = true;
                                                    self.rect_dragging = true;
                                                    match corner_idx {
                                                        0 => {
                                                            left += delta.x;
                                                            top += delta.y;
                                                        }
                                                        1 => {
                                                            right += delta.x;
                                                            top += delta.y;
                                                        }
                                                        2 => {
                                                            left += delta.x;
                                                            bottom += delta.y;
                                                        }
                                                        3 => {
                                                            right += delta.x;
                                                            bottom += delta.y;
                                                        }
                                                        _ => {}
                                                    }
                                                }
                                            }

                                            if rect_changed {
                                                left = left.clamp(
                                                    image_rect.left(),
                                                    image_rect.right() - min_screen_size,
                                                );
                                                top = top.clamp(
                                                    image_rect.top(),
                                                    image_rect.bottom() - min_screen_size,
                                                );
                                                right = right
                                                    .clamp(left + min_screen_size, image_rect.right());
                                                bottom = bottom
                                                    .clamp(top + min_screen_size, image_rect.bottom());
                                            } else {
                                                let screen_rect =
                                                    egui::Rect::from_min_max(
                                                        egui::pos2(left, top),
                                                        egui::pos2(right, bottom),
                                                    );
                                                let move_id =
                                                    ui.make_persistent_id(("crop_rect_move", idx));
                                                let move_resp = ui.interact(
                                                    screen_rect,
                                                    move_id,
                                                    egui::Sense::click_and_drag(),
                                                );
                                                if move_resp.dragged() {
                                                    let delta = move_resp.drag_delta();
                                                    let rect_w = right - left;
                                                    let rect_h = bottom - top;
                                                    let mut dx = delta.x;
                                                    let mut dy = delta.y;
                                                    dx = dx
                                                        .max(image_rect.left() - left)
                                                        .min(image_rect.right() - right);
                                                    dy = dy
                                                        .max(image_rect.top() - top)
                                                        .min(image_rect.bottom() - bottom);
                                                    left += dx;
                                                    right = left + rect_w;
                                                    top += dy;
                                                    bottom = top + rect_h;
                                                    rect_changed = true;
                                                    self.rect_dragging = true;
                                                }
                                            }

                                            let screen_rect = egui::Rect::from_min_max(
                                                egui::pos2(left, top),
                                                egui::pos2(right, bottom),
                                            );
                                            let overlay = egui::Color32::from_black_alpha(128);
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

                                            painter.rect_stroke(
                                                screen_rect,
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

                                            // Keep D-min visuals above crop dark overlay.
                                            if opts.apply_dmin
                                                && opts.dmin_fixed.is_none()
                                                && self.flat_field_path.is_none()
                                            {
                                                if let Some(dmin_rect) = opts.dmin_rect {
                                                    let dmin_rect = scale_rect_to_size(
                                                        dmin_rect,
                                                        opts.dmin_rect_reference_size,
                                                        input_w,
                                                        input_h,
                                                    );
                                                    let dmin_norm_x =
                                                        dmin_rect.x as f32 / input_w as f32;
                                                    let dmin_norm_y =
                                                        dmin_rect.y as f32 / input_h as f32;
                                                    let dmin_norm_w =
                                                        dmin_rect.width as f32 / input_w as f32;
                                                    let dmin_norm_h =
                                                        dmin_rect.height as f32 / input_h as f32;
                                                    let dmin_left = image_rect.left()
                                                        + dmin_norm_x * image_rect.width();
                                                    let dmin_top = image_rect.top()
                                                        + dmin_norm_y * image_rect.height();
                                                    let dmin_right =
                                                        dmin_left + dmin_norm_w * image_rect.width();
                                                    let dmin_bottom =
                                                        dmin_top + dmin_norm_h * image_rect.height();
                                                    let dmin_screen_rect = egui::Rect::from_min_max(
                                                        egui::pos2(dmin_left, dmin_top),
                                                        egui::pos2(dmin_right, dmin_bottom),
                                                    );
                                                    painter.rect_stroke(
                                                        dmin_screen_rect,
                                                        0.0,
                                                        egui::Stroke::new(
                                                            1.5,
                                                            egui::Color32::from_rgb(255, 200, 0),
                                                        ),
                                                    );
                                                    for p in [
                                                        egui::pos2(dmin_left, dmin_top),
                                                        egui::pos2(dmin_right, dmin_top),
                                                        egui::pos2(dmin_left, dmin_bottom),
                                                        egui::pos2(dmin_right, dmin_bottom),
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
                                                let img_w = image_rect.width().max(1.0);
                                                let img_h = image_rect.height().max(1.0);
                                                let norm_left = (left - image_rect.left()) / img_w;
                                                let norm_top = (top - image_rect.top()) / img_h;
                                                let norm_width = (right - left) / img_w;
                                                let norm_height = (bottom - top) / img_h;

                                                let x = (norm_left * input_w as f32)
                                                    .round()
                                                    .clamp(0.0, input_w.saturating_sub(1) as f32)
                                                    as u32;
                                                let y = (norm_top * input_h as f32)
                                                    .round()
                                                    .clamp(0.0, input_h.saturating_sub(1) as f32)
                                                    as u32;
                                                let mut w_px =
                                                    (norm_width * input_w as f32).round().max(1.0) as u32;
                                                let mut h_px =
                                                    (norm_height * input_h as f32).round().max(1.0) as u32;
                                                w_px = w_px.min(input_w.saturating_sub(x).max(1));
                                                h_px = h_px.min(input_h.saturating_sub(y).max(1));

                                                opts.crop_rect = Some(Rect {
                                                    x,
                                                    y,
                                                    width: w_px,
                                                    height: h_px,
                                                });
                                                opts.crop_rect_reference_size = Some((input_w, input_h));
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Row under the image: full filename (left, truncated) + Rotate icon buttons (right)
                        ui.add_space(BOTTOM_PADDING);
                        ui.horizontal(|ui| {
                            let full_name = self.images[idx].path.display().to_string();
                            let max_filename_w = (ui.available_width() - 150.0).max(80.0); // leave room for rotate icons
                            ui.allocate_ui(egui::vec2(max_filename_w, CONTROL_ROW_HEIGHT), |ui| {
                                ui.label(
                                    egui::RichText::new(full_name).small().color(egui::Color32::from_gray(200)),
                                )
                                .on_hover_text(self.images[idx].path.display().to_string());
                            });
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
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

                            let max_r = r_hist.iter().copied().max().unwrap_or(1) as f32;
                            let max_g = g_hist.iter().copied().max().unwrap_or(1) as f32;
                            let max_b = b_hist.iter().copied().max().unwrap_or(1) as f32;
                            let max_all = max_r.max(max_g).max(max_b).max(1.0);

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
                                        let h_norm =
                                            (hist[i] as f32 / max_all).clamp(0.0, 1.0);
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
                                        egui::Stroke::new(2.0, line_color),
                                    ));
                                };

                            draw_channel(
                                r_hist,
                                egui::Color32::from_rgb(255, 80, 80),
                                egui::Color32::from_rgba_premultiplied(255, 0, 0, 5),
                                &painter,
                            );
                            draw_channel(
                                g_hist,
                                egui::Color32::from_rgb(110, 255, 110),
                                egui::Color32::from_rgba_premultiplied(0, 255, 0, 5),
                                &painter,
                            );
                            draw_channel(
                                b_hist,
                                egui::Color32::from_rgb(110, 160, 255),
                                egui::Color32::from_rgba_premultiplied(0, 90, 255, 5),
                                &painter,
                            );
                        }
                        return;
                    }
                }
                ui.vertical_centered(|ui| {
                    ui.add_space(ui.available_height() / 2.0 - 20.0);
                    if show_loader {
                        ui.spinner();
                        ui.add_space(8.0);
                    }
                    ui.label("Preview not ready yet.");
                });
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
                let hash_now = options_hash_for(&self.images[idx].path, &self.images[idx].options);
                let need_new = self.images[idx].preview_texture.is_none()
                    || self.images[idx].preview_hash != hash_now;
                if need_new {
                    let now = Instant::now();
                    let key = (idx, hash_now);
                    if self.pending_preview_key != Some(key) {
                        self.pending_preview_key = Some(key);
                        self.pending_preview_since = Some(now);
                    }

                    let pointer_down = ctx.input(|i| i.pointer.any_down());
                    let waiting_for_release = self.rect_dragging || pointer_down;
                    let settled = self
                        .pending_preview_since
                        .map(|t| now.saturating_duration_since(t) >= Duration::from_millis(PREVIEW_DEBOUNCE_MS))
                        .unwrap_or(false);

                    if self.preview_receiver.is_none() && !waiting_for_release && settled {
                        self.request_preview_for(idx, ctx);
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

