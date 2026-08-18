//! Oxid GUI: three-panel layout — center preview, right per-image settings, bottom image strip + global output/convert.

// On Windows, use GUI subsystem so closing the window exits with code 0 instead of 0xC000013A (STATUS_CONTROL_C_EXIT).
#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

use std::collections::{hash_map::DefaultHasher, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(not(feature = "gpu"))]
use c41_raw_tool::apply_preview_from_cache;
#[cfg(feature = "gpu")]
use c41_raw_tool::{apply_preview_from_cache_gpu, process_one_to_preview_with_cache_gpu};
use c41_raw_tool::{
    apply_preview_from_cache_on_progress, auto_tune, blur_flat_field, cached_start_step,
    calibration, color, compute_preview_scene_stats, crop_sensor_for_oriented_rect,
    demosaic, detect_crop, dmin, hash_dust, load_develop_preset, load_flat_field_linear,
    load_project, load_sensor_from_path, oriented_sensor_size, png_reader, preview_scene_stats_key,
    process_export_jobs,     process_one_to_preview, process_one_to_preview_with_cache,
    process_one_to_preview_with_cache_on_progress, rasterize_strokes,
    raw_reader,
    reset_wb_for_picker, run_auto_crop_for_path, run_auto_for_path, save_develop_preset,
    save_project, stamp_disc, sync_wb_flags_from_mode, tiff_export, AutoCropResult, AutoTuneResult,
    CachedSensor, CropConfidence, DminMode, DustHealParams, DustInfill, DustMask, DustStroke,
    DustTool,
    ExportCancelled, ExportControl, ExportJobSpec, LoadedProject, OutputLutEncoding, OutputStage,
    PipelineOptions, PreviewSceneStats, PreviewStepCache, ProjectDust, ProjectExportFormat,
    ProjectImage, Rect, TiffFormat, UndoManager, WbMode, PROJECT_EXTENSION,
    PROJECT_EXTENSION_LEGACY, UNDO_LIMIT,
};
use eframe::egui;

mod theme;

const PREVIEW_MAX_WIDTH: u32 = 1920;
const PREVIEW_MAX_HEIGHT: u32 = 1200;
/// Floor so a tiny window still produces a usable working image.
const PREVIEW_MIN_SIDE: u32 = 640;
/// Lightroom-style draft proxy: process this first, then refine to screen-res.
const PREVIEW_DRAFT_MAX: u32 = 1920;
const PREVIEW_TILE_SIZE: u32 = 512;
const PREVIEW_TILE_LRU: usize = 16;
const PREVIEW_TILE_MAX: usize = 192;
const PREVIEW_TILE_HALO: u32 = 32;
const THUMB_MAX_SIZE: u32 = 64;
const PREVIEW_DEBOUNCE_MS: u64 = 180;
/// Coalesce slider ticks so the draft proxy can update while dragging.
const PREVIEW_LIVE_DEBOUNCE_MS: u64 = 50;
/// Wait this long after pan/zoom before fetching 1:1 tiles.
const PREVIEW_VIEW_SETTLE_MS: u64 = 100;
/// Coalesce rapid rotate/flip clicks into one pipeline run.
const GEOMETRY_COALESCE_MS: u64 = 300;
/// Show the export progress dialog only if the job is still running after this.
const EXPORT_PROGRESS_DELAY_MS: u64 = 400;

const BOTTOM_PANEL_HEIGHT: f32 = 150.0;
const RIGHT_PANEL_WIDTH: f32 = 330.0;
const RIGHT_PANEL_MIN_WIDTH: f32 = 240.0;
const RIGHT_PANEL_MAX_WIDTH: f32 = 560.0;
const HISTOGRAM_HEIGHT: f32 = 72.0;
const HISTOGRAM_MIN_HEIGHT: f32 = 40.0;
const HISTOGRAM_MAX_HEIGHT: f32 = 320.0;
/// Left inset so the Archive menu clears macOS traffic lights (hidden title bar).
const MENU_BAR_MACOS_INSET: f32 = 78.0;
const PROJECT_SAVE_SHORTCUT: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::S);
const PROJECT_SAVE_AS_SHORTCUT: egui::KeyboardShortcut = egui::KeyboardShortcut::new(
    egui::Modifiers::COMMAND.plus(egui::Modifiers::SHIFT),
    egui::Key::S,
);
const PROJECT_LOAD_SHORTCUT: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::O);
const RECENT_PROJECTS_MAX: usize = 5;
const UNDO_SHORTCUT: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Z);
const REDO_SHORTCUT: egui::KeyboardShortcut = egui::KeyboardShortcut::new(
    egui::Modifiers::COMMAND.plus(egui::Modifiers::SHIFT),
    egui::Key::Z,
);
const DUST_PROCESS_SHORTCUT: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::P);
const DUST_EDIT_SHORTCUT: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::E);
const DUST_DISABLE_SHORTCUT: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::D);
const DUST_BRUSH_RADIUS_MAX: f32 = 150.0;
const APP_NAME: &str = "Oxid";
/// First launch fills most of the display; width scales with the screen.
const START_WINDOW_SCREEN_FRACTION: f32 = 0.80;
const START_WINDOW_MIN: [f32; 2] = [960.0, 640.0];
const ICON_LOGO_PATH: &str = "logo.png";
/// Extensions accepted by Add image… and drag-and-drop.
const IMPORT_EXTENSIONS: &[&str] = &[
    "arw", "nef", "nrw", "cr2", "cr3", "crw", "dng", "raf", "orf", "rw2", "png", "jpeg", "jpg",
    "tiff", "tif",
];

fn recent_projects_file() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        let home = std::env::var_os("HOME")?;
        Some(PathBuf::from(home).join("Library/Application Support/Oxid/recent_projects.json"))
    }
    #[cfg(target_os = "windows")]
    {
        let appdata = std::env::var_os("APPDATA")?;
        Some(PathBuf::from(appdata).join("Oxid").join("recent_projects.json"))
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let home = std::env::var_os("HOME")?;
        Some(PathBuf::from(home).join(".config/oxid/recent_projects.json"))
    }
}

fn load_recent_projects() -> Vec<PathBuf> {
    let Some(path) = recent_projects_file() else {
        return Vec::new();
    };
    let Ok(data) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let paths: Vec<String> = serde_json::from_str(&data).unwrap_or_default();
    paths
        .into_iter()
        .map(PathBuf::from)
        .take(RECENT_PROJECTS_MAX)
        .collect()
}

fn save_recent_projects(paths: &[PathBuf]) {
    let Some(path) = recent_projects_file() else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let strings: Vec<String> = paths
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    if let Ok(json) = serde_json::to_string_pretty(&strings) {
        let _ = std::fs::write(path, json);
    }
}

fn recent_project_label(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| path.display().to_string())
}

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

fn primary_screen_size() -> Option<(f32, f32)> {
    #[cfg(target_os = "macos")]
    {
        #[repr(C)]
        struct CgPoint {
            x: f64,
            y: f64,
        }
        #[repr(C)]
        struct CgSize {
            width: f64,
            height: f64,
        }
        #[repr(C)]
        struct CgRect {
            origin: CgPoint,
            size: CgSize,
        }
        #[link(name = "CoreGraphics", kind = "framework")]
        extern "C" {
            fn CGMainDisplayID() -> u32;
            fn CGDisplayBounds(display: u32) -> CgRect;
        }
        let bounds = unsafe { CGDisplayBounds(CGMainDisplayID()) };
        let w = bounds.size.width as f32;
        let h = bounds.size.height as f32;
        return (w > 1.0 && h > 1.0).then_some((w, h));
    }
    #[cfg(target_os = "windows")]
    {
        #[link(name = "user32")]
        extern "C" {
            fn GetSystemMetrics(index: i32) -> i32;
        }
        const SM_CXSCREEN: i32 = 0;
        const SM_CYSCREEN: i32 = 1;
        let w = unsafe { GetSystemMetrics(SM_CXSCREEN) };
        let h = unsafe { GetSystemMetrics(SM_CYSCREEN) };
        return (w > 1 && h > 1).then_some((w as f32, h as f32));
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

fn startup_window_size() -> [f32; 2] {
    let (screen_w, screen_h) = primary_screen_size().unwrap_or((1440.0, 900.0));
    let width = (screen_w * START_WINDOW_SCREEN_FRACTION).max(START_WINDOW_MIN[0]);
    let height = (screen_h * START_WINDOW_SCREEN_FRACTION).max(START_WINDOW_MIN[1]);
    [width.min(screen_w), height.min(screen_h)]
}

#[derive(Default)]
struct UiIcons {
    logo: Option<egui::TextureHandle>,
}

fn main() -> eframe::Result<()> {
    let start_size = startup_window_size();
    let mut native_options = if cfg!(target_os = "macos") {
        let mut o = eframe::NativeOptions::default();
        o.viewport = o
            .viewport
            .clone()
            .with_fullsize_content_view(true)
            .with_titlebar_shown(false)
            .with_title_shown(false) // hide OS title so only our white title in the dark bar shows
            .with_drag_and_drop(true);
        o
    } else if cfg!(target_os = "windows") {
        let mut o = eframe::NativeOptions::default();
        o.viewport = o.viewport.clone().with_drag_and_drop(true);
        o
    } else {
        let mut o = eframe::NativeOptions::default();
        o.viewport = o.viewport.clone().with_drag_and_drop(true);
        o
    };
    native_options.centered = true;
    // Always use the large default; a persisted tiny window would otherwise win.
    native_options.persist_window = false;
    native_options.viewport = native_options
        .viewport
        .clone()
        .with_title(APP_NAME)
        .with_app_id(APP_NAME)
        .with_inner_size(start_size)
        .with_min_inner_size(START_WINDOW_MIN);
    if let Some(icon) = app_icon_data() {
        native_options.viewport = native_options.viewport.clone().with_icon(Arc::new(icon));
    }
    eframe::run_native(
        APP_NAME,
        native_options,
        Box::new(|cc| {
            theme::install_fonts(&cc.egui_ctx);
            theme::apply(&cc.egui_ctx);
            Ok(Box::new(C41Gui::default()))
        }),
    )
}

struct ImageEntry {
    /// Stable identity for the session undo stack (not persisted).
    id: u64,
    path: PathBuf,
    options: PipelineOptions,
    /// Preview texture uploaded when RGB arrives; filter updated only when pixel scale crosses 1×.
    preview_texture: Option<egui::TextureHandle>,
    /// Whether `preview_texture` was uploaded with NEAREST (magnified) vs LINEAR (fit / minified).
    preview_texture_nearest: bool,
    /// `rotation_degrees` baked into `preview_texture` / `preview_full_rgb`.
    preview_texture_rotation: i32,
    /// `flip_horizontal` baked into `preview_texture` / `preview_full_rgb`.
    preview_texture_flip_h: bool,
    /// `flip_vertical` baked into `preview_texture` / `preview_full_rgb`.
    preview_texture_flip_v: bool,
    /// Hash of options + working size + full-res flag at last completed preview.
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
    /// Crop used for the last histogram so we can refresh without re-running the pipeline.
    histogram_crop_hash: u64,
    export_format: ExportFormat,
    /// Rawloader debug report for this file (Debug tab).
    raw_debug_report: Option<String>,
    /// Pipeline debug log from the most recent preview render.
    pipeline_debug_log: Option<String>,
    /// Cached full-resolution sensor data (Bayer or RGB) for fast previews/exports.
    cached_sensor: Option<Arc<CachedSensor>>,
    /// Full-res D-min / auto-WB pinned to export, keyed by `preview_scene_stats_key`.
    scene_stats: Option<(u64, PreviewSceneStats)>,
    /// Step cache of the currently displayed LOD (WB picker / overlays).
    preview_step_cache: Option<PreviewStepCache>,
    draft_step_cache: Option<PreviewStepCache>,
    screen_step_cache: Option<PreviewStepCache>,
    /// Which proxy is on screen (draft is soft; screen/full-res is sharp).
    preview_lod: PreviewLod,
    /// Options + full-res flag that produced the current texture (not working size).
    preview_options_hash: u64,
    /// Screen-res working size last applied.
    preview_screen_wh: (u32, u32),
    /// Last *requested* screen limits (avoids a refine loop when CFA downsample comes in smaller).
    preview_screen_requested_wh: (u32, u32),
    /// 1:1 tiles composited over the proxy when zoomed in. Front = most recently used.
    tile_cache: Vec<PreviewTile>,
    /// Process tab (Input/Develop/Export) — persists per image when switching.
    process_tab: ProcessTab,
    dust_strokes: Vec<DustStroke>,
    dust_reference_size: Option<(u32, u32)>,
    dust_mask: Vec<u8>,
    dust_mask_size: (u32, u32),
    dust_view: DustView,
    dust_tool: Option<DustTool>,
    dust_brush_radius: f32,
    dust_detect: f32,
    dust_feather: f32,
    dust_grain: f32,
    dust_grain_size: f32,
    dust_infill: DustInfill,
    dust_tile: u8,
    dust_match: f32,
    dust_overlay_texture: Option<egui::TextureHandle>,
    dust_overlay_dirty: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreviewLod {
    Draft,
    Screen,
    FullRes,
}

struct PreviewTile {
    ix: i32,
    iy: i32,
    options_hash: u64,
    texture: egui::TextureHandle,
    /// Placement in the oriented full frame (0–1), halo excluded.
    uv: egui::Rect,
    /// Corresponding region of `texture` (0–1), halo excluded.
    tex_uv: egui::Rect,
}

/// Visible 1:1 tile grid for the current canvas / pan / zoom.
/// `ix0..=ix1` / `iy0..=iy1` are the **unpadded** on-screen (core) tiles.
struct VisibleTileGrid {
    ix0: i32,
    iy0: i32,
    ix1: i32,
    iy1: i32,
    opt_hash: u64,
    /// True when 1:1 tiles should run (always: the proxy is CFA-downsampled).
    proxy_soft: bool,
    /// True when the core view fits in `PREVIEW_TILE_MAX` (we can cover it).
    tiles_fit: bool,
    core_n: usize,
}

/// Tiles that intersect sensor-pixel range `[px0, px1] × [py0, py1]`.
/// `px1`/`py1` are exclusive-ish (egui UV max). `ceil-1` still includes a
/// tile that is only a sliver on-screen; `floor(end - eps)` used to miss it.
fn tile_range_intersecting(
    px0: f32,
    py0: f32,
    px1: f32,
    py1: f32,
    tile: f32,
    tiles_x: i32,
    tiles_y: i32,
) -> (i32, i32, i32, i32) {
    let last_x = tiles_x.saturating_sub(1);
    let last_y = tiles_y.saturating_sub(1);
    if tile <= 0.0 || last_x < 0 || last_y < 0 {
        return (0, 0, 0, 0);
    }
    let x0 = px0.min(px1);
    let x1 = px0.max(px1);
    let y0 = py0.min(py1);
    let y1 = py0.max(py1);
    let ix0 = (x0 / tile).floor() as i32;
    let iy0 = (y0 / tile).floor() as i32;
    let ix1 = (x1 / tile).ceil() as i32 - 1;
    let iy1 = (y1 / tile).ceil() as i32 - 1;
    let ix0 = ix0.clamp(0, last_x);
    let iy0 = iy0.clamp(0, last_y);
    let ix1 = ix1.clamp(ix0, last_x);
    let iy1 = iy1.clamp(iy0, last_y);
    (ix0, iy0, ix1, iy1)
}

/// If the visible UV includes an image edge, that sensor tile row/column
/// must be in the grid. Preview-UV × sensor-size can start at iy=1.
fn include_image_edge_tiles(
    uv_l: f32,
    uv_t: f32,
    uv_r: f32,
    uv_b: f32,
    tiles_x: i32,
    tiles_y: i32,
    ix0: i32,
    iy0: i32,
    ix1: i32,
    iy1: i32,
) -> (i32, i32, i32, i32) {
    let last_x = tiles_x.saturating_sub(1);
    let last_y = tiles_y.saturating_sub(1);
    let ix0 = if uv_l <= 0.05 { 0 } else { ix0 }.clamp(0, last_x);
    let iy0 = if uv_t <= 0.05 { 0 } else { iy0 }.clamp(0, last_y);
    let ix1 = if uv_r >= 0.95 { last_x } else { ix1 }.clamp(ix0, last_x);
    let iy1 = if uv_b >= 0.95 { last_y } else { iy1 }.clamp(iy0, last_y);
    (ix0, iy0, ix1, iy1)
}

/// Active crop in `current_w × current_h` pixels, or `None` when crop is off.
fn scaled_crop_px(
    entry: &ImageEntry,
    current_w: u32,
    current_h: u32,
) -> Option<(u32, u32, u32, u32)> {
    if !entry.options.apply_crop || current_w == 0 || current_h == 0 {
        return None;
    }
    let rect = entry.options.crop_rect?;
    let s = scale_rect_to_size(rect, entry.options.crop_rect_reference_size, current_w, current_h);
    Some((s.x, s.y, s.width.max(1), s.height.max(1)))
}

/// Tile grid origin and size in oriented-sensor pixels.
/// When crop is on, (0,0) is the crop origin so the first row is the frame,
/// not the rebate above it.
fn tile_space(entry: &ImageEntry, ow: u32, oh: u32) -> (u32, u32, u32, u32) {
    scaled_crop_px(entry, ow, oh).unwrap_or((0, 0, ow, oh))
}

/// Preview layout size: crop size when crop is on, else the full buffer.
fn preview_display_wh(entry: &ImageEntry, tex_w: u32, tex_h: u32) -> (u32, u32) {
    let view_rot = preview_view_rotation(entry);
    let (fw, fh) = if view_rot == 90 || view_rot == 270 {
        (tex_h, tex_w)
    } else {
        (tex_w, tex_h)
    };
    if let Some((_, _, cw, ch)) = scaled_crop_px(entry, fw, fh) {
        (cw.max(1), ch.max(1))
    } else {
        (fw, fh)
    }
}

/// Crop in 0–1 of the preview texture, or `None` when crop is off.
fn preview_crop_uv(entry: &ImageEntry, tex_w: u32, tex_h: u32) -> Option<egui::Rect> {
    let (x, y, w, h) = scaled_crop_px(entry, tex_w, tex_h)?;
    let tw = (tex_w as f32).max(1.0);
    let th = (tex_h as f32).max(1.0);
    Some(egui::Rect::from_min_max(
        egui::pos2(x as f32 / tw, y as f32 / th),
        egui::pos2((x + w) as f32 / tw, (y + h) as f32 / th),
    ))
}

impl VisibleTileGrid {
    fn contains(&self, ix: i32, iy: i32) -> bool {
        ix >= self.ix0 && ix <= self.ix1 && iy >= self.iy0 && iy <= self.iy1
    }

    /// Tiles we fetch and keep when the core exceeds `PREVIEW_TILE_MAX`:
    /// the cap closest to the view center (the full core when it fits).
    fn is_priority(&self, ix: i32, iy: i32) -> bool {
        if !self.contains(ix, iy) {
            return false;
        }
        if self.tiles_fit {
            return true;
        }
        // Fit views that exceed the cap still keep the visible silhouette.
        // Those tiles are farthest from center and were dropped first, so
        // the CFA proxy showed through as a more-saturated L-shaped band.
        if iy == self.iy0 || iy == self.iy1 || ix == self.ix0 || ix == self.ix1 {
            return true;
        }
        let cx = (self.ix0 + self.ix1) as f32 * 0.5;
        let cy = (self.iy0 + self.iy1) as f32 * 0.5;
        let d = (ix as f32 - cx).hypot(iy as f32 - cy);
        let mut closer = 0usize;
        for y in self.iy0..=self.iy1 {
            for x in self.ix0..=self.ix1 {
                let dd = (x as f32 - cx).hypot(y as f32 - cy);
                let before = if dd < d - 1e-6 {
                    true
                } else if (dd - d).abs() <= 1e-6 {
                    y < iy || (y == iy && x < ix)
                } else {
                    false
                };
                if before {
                    closer += 1;
                    if closer >= PREVIEW_TILE_MAX {
                        return false;
                    }
                }
            }
        }
        true
    }
}

struct PreviewJobResult {
    gen: u64,
    lod: PreviewLod,
    index: usize,
    options_hash: u64,
    input_w: u32,
    input_h: u32,
    w: u32,
    h: u32,
    rgb: Vec<u8>,
    dbg_log: String,
    captured_debug: bool,
    new_cache: PreviewStepCache,
}

struct TileJobResult {
    gen: u64,
    index: usize,
    ix: i32,
    iy: i32,
    options_hash: u64,
    w: u32,
    h: u32,
    rgb: Vec<u8>,
    uv: egui::Rect,
    tex_uv: egui::Rect,
}

/// User-facing per-image edit state captured by the undo stack.
#[derive(Clone, PartialEq)]
struct EditSnapshot {
    options: PipelineOptions,
    export_format: ExportFormat,
    dust_strokes: Vec<DustStroke>,
    dust_reference_size: Option<(u32, u32)>,
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

impl ExportFormat {
    fn to_project(self) -> ProjectExportFormat {
        match self {
            Self::Tiff16 => ProjectExportFormat::Tiff16,
            Self::Tiff32 => ProjectExportFormat::Tiff32,
            Self::Exr => ProjectExportFormat::Exr,
            Self::Jpeg => ProjectExportFormat::Jpeg,
            Self::ExrAces2065 => ProjectExportFormat::ExrAces2065,
        }
    }

    fn from_project(fmt: ProjectExportFormat) -> Self {
        match fmt {
            ProjectExportFormat::Tiff16 => Self::Tiff16,
            ProjectExportFormat::Tiff32 => Self::Tiff32,
            ProjectExportFormat::Exr => Self::Exr,
            ProjectExportFormat::Jpeg => Self::Jpeg,
            ProjectExportFormat::ExrAces2065 => Self::ExrAces2065,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Tiff16 => "TIFF 16-bit",
            Self::Tiff32 => "TIFF 32-bit float",
            Self::Exr => "EXR (32-bit float)",
            Self::Jpeg => "JPEG",
            Self::ExrAces2065 => "EXR ACES2065-1 (32-bit float)",
        }
    }
}

fn apply_export_format_to_options(opts: &mut PipelineOptions, format: ExportFormat) {
    match format {
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
}

fn export_format_combo(ui: &mut egui::Ui, format: &mut ExportFormat) {
    egui::ComboBox::from_label("Output format")
        .selected_text(format.label())
        .show_ui(ui, |ui| {
            for (value, text) in [
                (ExportFormat::Tiff16, "TIFF 16-bit"),
                (ExportFormat::Tiff32, "TIFF 32-bit float"),
                (ExportFormat::Exr, "EXR (32-bit float)"),
                (ExportFormat::ExrAces2065, "EXR ACES2065-1 (32-bit float)"),
                (ExportFormat::Jpeg, "JPEG"),
            ] {
                if ui.selectable_label(*format == value, text).clicked() {
                    *format = value;
                }
            }
        });
}

struct BatchExportDialog {
    format: ExportFormat,
    write_jpeg: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UIMode {
    Process,
    Calibrate,
    LuminanceCalibrate,
    Debug,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProcessTab {
    Input,
    Develop,
    Dust,
    Export,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DustView {
    Disable,
    Edit,
    Process,
}

enum ExportJobOutcome {
    Done { count: usize },
    Cancelled { completed: usize },
    Error(String),
}

struct ExportJob {
    receiver: mpsc::Receiver<ExportJobOutcome>,
    started_at: Instant,
    control: Arc<ExportControl>,
}

#[derive(Clone, Debug)]
struct AutoProgressState {
    fraction: f32,
    message: String,
    log: Vec<String>,
    file_name: String,
    completed: usize,
    total: usize,
}

enum AutoJobOutcome {
    Done {
        index: usize,
        result: AutoTuneResult,
    },
    FileDone {
        path: PathBuf,
        result: AutoTuneResult,
    },
    CropFileDone {
        path: PathBuf,
        result: AutoCropResult,
    },
    BatchDone {
        completed: usize,
        errors: usize,
    },
    Cancelled {
        completed: usize,
    },
    Error {
        message: String,
    },
}

struct AutoJob {
    receiver: mpsc::Receiver<AutoJobOutcome>,
    progress: Arc<Mutex<AutoProgressState>>,
    cancel: Option<Arc<AtomicBool>>,
    batch: bool,
    title: &'static str,
    /// Single-image Auto: keep the dialog up until this preview is current.
    applying_preview: Option<usize>,
    ticker_stop: Arc<AtomicBool>,
}

impl Drop for AutoJob {
    fn drop(&mut self) {
        self.ticker_stop.store(true, Ordering::Relaxed);
    }
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

enum PendingLeaveAction {
    LoadDialog,
    LoadRecent(PathBuf),
    Quit,
}

struct C41Gui {
    images: Vec<ImageEntry>,
    selected_index: Option<usize>,
    output_dir: Option<PathBuf>,
    status: String,
    preview_receiver: Option<mpsc::Receiver<anyhow::Result<PreviewJobResult>>>,
    preview_started_at: Option<Instant>,
    /// Bumped when the desired options hash changes; in-flight jobs with an older gen are ignored.
    preview_gen: u64,
    tile_receiver: Option<mpsc::Receiver<anyhow::Result<TileJobResult>>>,
    tile_gen: u64,
    /// In-flight tile so we do not request the same one twice.
    tile_inflight: Option<(usize, i32, i32)>,
    /// Tiles that failed this generation — do not retry until options change.
    tile_failed: Vec<(u64, i32, i32)>,
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
    /// (path, profile, LUT path for .oxid or None for .json)
    #[allow(dead_code)]
    calibration_profiles: Vec<(PathBuf, calibration::CalibrationProfile, Option<PathBuf>)>,
    #[allow(dead_code)]
    selected_profile_idx: Option<usize>,
    /// Luminance calibration: path and linearized flat-field image (RAW → demosaic only).
    flat_field_path: Option<PathBuf>,
    flat_field_image: Option<ndarray::Array3<f32>>,
    ui_icons: UiIcons,
    /// Suppresses preview reprocessing while the user is dragging a rect handle (crop / d-min).
    rect_dragging: bool,
    /// True while painting or erasing on the Dust tab.
    dust_painting: bool,
    /// Ctrl+drag brush resize: (radius at press, locked pointer, accumulated delta).
    dust_brush_resize: Option<(f32, egui::Pos2, f32)>,
    /// True while the preview is being panned (left/middle drag).
    preview_view_dragging: bool,
    /// True while the pointer is down on the preview canvas (not a slider).
    preview_canvas_pointer: bool,
    /// Last pan/zoom; tiles wait until this has settled.
    preview_view_changed_at: Option<Instant>,
    /// Debounce state for preview refreshes: (image index, options hash) currently waiting to settle.
    pending_preview_key: Option<(usize, u64)>,
    pending_preview_since: Option<Instant>,
    /// Path of the last saved/loaded project (Save writes here; Save As always prompts).
    project_path: Option<PathBuf>,
    /// Persistable project state after the last successful load or save.
    clean_project: Vec<ProjectImage>,
    /// Deferred file dialog flag: open the output LUT browser outside the egui render loop
    /// to avoid macOS NSOpenPanel re-entrance crashes.
    pending_output_lut_browse: bool,
    pending_project_save_as: bool,
    pending_project_load: bool,
    /// Deferred load from Archive → Recent (outside the menu render loop).
    pending_recent_load: Option<PathBuf>,
    /// Confirm save before Load, Recent, or Quit when the project has unsaved changes.
    save_before_leave: Option<PendingLeaveAction>,
    /// After Yes with no project path: proceed only if deferred Save As succeeds.
    pending_save_then_leave: Option<PendingLeaveAction>,
    /// True after Yes/No on Quit so the follow-up close is not intercepted again.
    close_confirmed: bool,
    /// Last successfully loaded or saved projects, newest first.
    recent_projects: Vec<PathBuf>,
    /// When true, preview uses full resolution (export pipeline). Deactivates on option change or image switch.
    full_res_preview_active: bool,
    /// One-shot: set by the full-res button so we don't deactivate on the preview request it triggers.
    full_res_preview_button_clicked: bool,
    /// Canvas size (w, h) in points from last layout — used to request preview at screen resolution.
    preview_canvas_size: Option<(f32, f32)>,
    /// Invalidation hash of the in-flight preview job (options + working size, not zoom).
    preview_job_hash: Option<u64>,
    /// True when the in-flight preview is a live cache apply (not a full remosaic).
    preview_job_live: bool,
    /// Background export (Convert all / Export selected).
    export_job: Option<ExportJob>,
    /// One-shot Auto grade job (Develop tab).
    auto_job: Option<AutoJob>,
    /// Archive → Batch → Export settings dialog.
    batch_export_dialog: Option<BatchExportDialog>,
    /// True while the WB eyedropper is active (loupe + click-to-sample).
    wb_picker_armed: bool,
    /// While set and in the future, preview debounce uses [`GEOMETRY_COALESCE_MS`].
    geometry_coalesce_until: Option<Instant>,
    /// Session-wide chronological undo/redo of per-image edits.
    history: UndoManager<u64, EditSnapshot>,
    next_image_id: u64,
    /// Preview histogram height in points; drag the top edge to resize.
    histogram_height: f32,
    /// Right settings panel width in points; drag the inner edge to resize.
    right_panel_width: f32,
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
            preview_gen: 0,
            tile_receiver: None,
            tile_gen: 0,
            tile_inflight: None,
            tile_failed: Vec::new(),
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
            dust_painting: false,
            dust_brush_resize: None,
            preview_view_dragging: false,
            preview_canvas_pointer: false,
            preview_view_changed_at: None,
            pending_preview_key: None,
            pending_preview_since: None,
            project_path: None,
            clean_project: Vec::new(),
            pending_output_lut_browse: false,
            pending_project_save_as: false,
            pending_project_load: false,
            pending_recent_load: None,
            save_before_leave: None,
            pending_save_then_leave: None,
            close_confirmed: false,
            recent_projects: load_recent_projects(),
            full_res_preview_active: false,
            full_res_preview_button_clicked: false,
            preview_canvas_size: None,
            preview_job_hash: None,
            preview_job_live: false,
            export_job: None,
            auto_job: None,
            batch_export_dialog: None,
            wb_picker_armed: false,
            geometry_coalesce_until: None,
            history: UndoManager::new(UNDO_LIMIT),
            next_image_id: 1,
            histogram_height: HISTOGRAM_HEIGHT,
            right_panel_width: RIGHT_PANEL_WIDTH,
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

fn compute_patch_centers_normalized(corners: [egui::Pos2; 4]) -> [[f32; 2]; 24] {
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
        density_matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        idt_matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
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
        apply_lab: true,
        lab_separation: 1.0,
        skin_magenta_shift: 0.0,
        rotation_degrees: 0,
        flip_horizontal: false,
        flip_vertical: false,
        synthetic_negative_input: false,
        debug_pipeline_step: 6,
        debug_preview_simple_debayer: false,
        verbose_debug: false,
        use_gpu: cfg!(feature = "gpu"),
        bujack_enabled: false,
        bujack_k_l: 0.25,
        bujack_k_c: 0.30,
        bujack_strength: 0.2,
        bujack_radius: 16.0,
        bujack_edge: 0.25,
        pinned_zone: None,
        dust_mask_hash: 0,
        dust_mask: None,
        dust_strokes: Vec::new(),
        dust_reference_size: None,
        dust_uv: None,
        dust_heal: DustHealParams::default(),
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
    // Crop is an overlay + histogram window. Preview pixels do not change.
    opts.dmin_neutral_only.hash(&mut h);
    if let Some((r, g, b)) = opts.dmin_fixed {
        r.to_bits().hash(&mut h);
        g.to_bits().hash(&mut h);
        b.to_bits().hash(&mut h);
    }
    (
        opts.wb_r.to_bits(),
        opts.wb_g.to_bits(),
        opts.wb_b.to_bits(),
    )
        .hash(&mut h);
    opts.temp_k.map(|k| k.to_bits()).hash(&mut h);
    opts.no_curve.hash(&mut h);
    opts.no_invert.hash(&mut h);
    opts.synthetic_negative_input.hash(&mut h);
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
    for row in &opts.idt_matrix {
        for v in row {
            v.to_bits().hash(&mut h);
        }
    }
    opts.flat_field_path
        .as_ref()
        .map(|p| p.display().to_string())
        .hash(&mut h);
    opts.export_aces_exr.hash(&mut h);
    opts.write_aces2065_only.hash(&mut h);
    opts.lut3d_path
        .as_ref()
        .map(|p| p.display().to_string())
        .hash(&mut h);
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
    opts.bujack_enabled.hash(&mut h);
    opts.bujack_k_l.to_bits().hash(&mut h);
    opts.bujack_k_c.to_bits().hash(&mut h);
    opts.bujack_strength.to_bits().hash(&mut h);
    opts.bujack_radius.to_bits().hash(&mut h);
    opts.bujack_edge.to_bits().hash(&mut h);
    opts.dust_mask_hash.hash(&mut h);
    opts.dust_heal.detect.to_bits().hash(&mut h);
    opts.dust_heal.feather.to_bits().hash(&mut h);
    opts.dust_heal.grain.to_bits().hash(&mut h);
    opts.dust_heal.grain_sigma.to_bits().hash(&mut h);
    opts.dust_heal.infill.hash(&mut h);
    opts.dust_heal.tile.hash(&mut h);
    opts.dust_heal.match_loosen.to_bits().hash(&mut h);
    h.finish()
}

/// Flip a rect horizontally (mirror left–right) within an image of `img_w` × `img_h`.
fn flip_rect_horizontal(rect: Rect, img_w: u32, _img_h: u32) -> Rect {
    let new_x = img_w
        .saturating_sub(rect.x)
        .saturating_sub(rect.width)
        .max(0);
    Rect {
        x: new_x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
    }
}

/// Flip a rect vertically (mirror top–bottom) within an image of `img_w` × `img_h`.
fn flip_rect_vertical(rect: Rect, _img_w: u32, img_h: u32) -> Rect {
    let new_y = img_h
        .saturating_sub(rect.y)
        .saturating_sub(rect.height)
        .max(0);
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
            width: (rw as f32 * current_w as f32 / ref_w as f32)
                .round()
                .max(1.0) as u32,
            height: (rh as f32 * current_h as f32 / ref_h as f32)
                .round()
                .max(1.0) as u32,
        },
        _ => Rect {
            x,
            y,
            width: rw.max(1),
            height: rh.max(1),
        },
    }
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
            let scaled =
                scale_rect_to_size(crop_rect, opts.crop_rect_reference_size, input_w, input_h);
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

/// Shared linear Y-scale for the RGB overlay histogram.
///
/// Full height is the 98th percentile of non-zero bin counts across R/G/B.
/// Outlier spikes (film base, rebate, hard clip, a single giant mode) sit
/// above that and clip. One scale for all three channels so relative channel
/// height still reads as colour balance. Falls back to the absolute max if
/// the histogram is empty or degenerate.
fn histogram_y_scale(r: &[u32; 256], g: &[u32; 256], b: &[u32; 256]) -> f32 {
    let mut heights = Vec::with_capacity(256 * 3);
    for hist in [r, g, b] {
        for &count in hist {
            if count > 0 {
                heights.push(count);
            }
        }
    }
    if heights.is_empty() {
        return 1.0;
    }
    heights.sort_unstable();
    let idx = ((heights.len() as f32 - 1.0) * 0.98).round() as usize;
    let peak = heights[idx.min(heights.len() - 1)];
    peak.max(1) as f32
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

fn parse_dmin_clipboard_text(
    text: &str,
) -> Option<(Option<(f32, f32, f32)>, Option<Rect>, Option<bool>)> {
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

fn load_icon_texture(
    ctx: &egui::Context,
    texture_name: &str,
    path_hint: &str,
) -> Option<egui::TextureHandle> {
    let image = icon_candidate_paths(path_hint)
        .into_iter()
        .find_map(|candidate| image::open(candidate).ok())?
        .to_rgba8();
    let size = [image.width() as usize, image.height() as usize];
    let pixels = image.into_vec();
    let color_image = egui::ColorImage::from_rgba_unmultiplied(size, &pixels);
    Some(ctx.load_texture(
        texture_name.to_string(),
        color_image,
        egui::TextureOptions::default(),
    ))
}

fn make_thumbnail_from_rgb(
    rgb: &[u8],
    src_w: u32,
    src_h: u32,
    max_side: u32,
) -> Option<egui::ColorImage> {
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

const WB_PICKER_SAMPLE: usize = 4;

fn sample_rgb_u8_4x4(rgb: &[u8], w: u32, h: u32, cx: u32, cy: u32) -> [u8; 3] {
    let w = w as usize;
    let h = h as usize;
    if w == 0 || h == 0 {
        return [128, 128, 128];
    }
    let cx = (cx as usize).min(w - 1);
    let cy = (cy as usize).min(h - 1);
    let x0 = cx.saturating_sub(1);
    let y0 = cy.saturating_sub(1);
    let x1 = (x0 + WB_PICKER_SAMPLE).min(w);
    let y1 = (y0 + WB_PICKER_SAMPLE).min(h);
    let mut sr = 0u32;
    let mut sg = 0u32;
    let mut sb = 0u32;
    let mut n = 0u32;
    for y in y0..y1 {
        for x in x0..x1 {
            let i = (y * w + x) * 3;
            if i + 2 < rgb.len() {
                sr += rgb[i] as u32;
                sg += rgb[i + 1] as u32;
                sb += rgb[i + 2] as u32;
                n += 1;
            }
        }
    }
    if n == 0 {
        [128, 128, 128]
    } else {
        [(sr / n) as u8, (sg / n) as u8, (sb / n) as u8]
    }
}

fn sample_array3_4x4(buf: &ndarray::Array3<f32>, cx: usize, cy: usize) -> (f32, f32, f32) {
    let (h, w, _) = buf.dim();
    if w == 0 || h == 0 {
        return (1.0, 1.0, 1.0);
    }
    let cx = cx.min(w - 1);
    let cy = cy.min(h - 1);
    let x0 = cx.saturating_sub(1);
    let y0 = cy.saturating_sub(1);
    let x1 = (x0 + WB_PICKER_SAMPLE).min(w);
    let y1 = (y0 + WB_PICKER_SAMPLE).min(h);
    let mut sr = 0.0f32;
    let mut sg = 0.0f32;
    let mut sb = 0.0f32;
    let mut n = 0.0f32;
    for y in y0..y1 {
        for x in x0..x1 {
            sr += buf[[y, x, 0]];
            sg += buf[[y, x, 1]];
            sb += buf[[y, x, 2]];
            n += 1.0;
        }
    }
    if n <= 0.0 {
        (1.0, 1.0, 1.0)
    } else {
        (sr / n, sg / n, sb / n)
    }
}

fn rgb_u8_to_color_image(w: u32, h: u32, rgb: &[u8]) -> egui::ColorImage {
    let size = [w as usize, h as usize];
    let pixels: Vec<egui::Color32> = rgb
        .chunks_exact(3)
        .map(|c| egui::Color32::from_rgb(c[0], c[1], c[2]))
        .collect();
    egui::ColorImage { size, pixels }
}

/// Crop RGB to `uv` (0–1 of the buffer) so mipmaps do not average in the halo.
/// Halo pixels sit at the crop edge and C-41-invert into a saturated strip.
fn crop_rgb_u8_to_uv(w: u32, h: u32, rgb: &[u8], uv: egui::Rect) -> (u32, u32, Vec<u8>, egui::Rect) {
    let full = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
    if w == 0 || h == 0 || rgb.len() < w as usize * h as usize * 3 {
        return (w, h, rgb.to_vec(), uv);
    }
    // Ceil/floor so a rounded halo texel cannot leak into the uploaded crop.
    let x0 = (uv.min.x * w as f32)
        .ceil()
        .clamp(0.0, w.saturating_sub(1) as f32) as u32;
    let y0 = (uv.min.y * h as f32)
        .ceil()
        .clamp(0.0, h.saturating_sub(1) as f32) as u32;
    let x1 = (uv.max.x * w as f32)
        .floor()
        .clamp((x0 + 1) as f32, w as f32) as u32;
    let y1 = (uv.max.y * h as f32)
        .floor()
        .clamp((y0 + 1) as f32, h as f32) as u32;
    if x0 == 0 && y0 == 0 && x1 == w && y1 == h {
        return (w, h, rgb.to_vec(), full);
    }
    let nw = x1 - x0;
    let nh = y1 - y0;
    let mut out = Vec::with_capacity(nw as usize * nh as usize * 3);
    let stride = w as usize * 3;
    for y in y0..y1 {
        let s = y as usize * stride + x0 as usize * 3;
        out.extend_from_slice(&rgb[s..s + nw as usize * 3]);
    }
    (nw, nh, out, full)
}

/// Pipeline geometry: rotate, then flip H, then flip V.
#[derive(Clone, Copy)]
struct ViewOrient {
    rot: i32,
    flip_h: bool,
    flip_v: bool,
}

impl ViewOrient {
    fn from_parts(rot: i32, flip_h: bool, flip_v: bool) -> Self {
        Self {
            rot: rot.rem_euclid(360),
            flip_h,
            flip_v,
        }
    }

    fn matches(self, other: Self) -> bool {
        self.rot == other.rot && self.flip_h == other.flip_h && self.flip_v == other.flip_v
    }
}

fn preview_desired_orient(entry: &ImageEntry) -> ViewOrient {
    ViewOrient::from_parts(
        entry.options.rotation_degrees,
        entry.options.flip_horizontal,
        entry.options.flip_vertical,
    )
}

fn preview_baked_orient(entry: &ImageEntry) -> ViewOrient {
    ViewOrient::from_parts(
        entry.preview_texture_rotation,
        entry.preview_texture_flip_h,
        entry.preview_texture_flip_v,
    )
}

fn preview_view_rotation(entry: &ImageEntry) -> i32 {
    (entry.options.rotation_degrees - entry.preview_texture_rotation).rem_euclid(360)
}

fn preview_view_geometry_pending(entry: &ImageEntry) -> bool {
    !preview_desired_orient(entry).matches(preview_baked_orient(entry))
}

/// Dest UV after rotation `rot` → source UV (texture unrotated).
fn unrotate_uv(u: f32, v: f32, rot: i32) -> egui::Pos2 {
    match rot.rem_euclid(360) {
        90 => egui::pos2(v, 1.0 - u),
        180 => egui::pos2(1.0 - u, 1.0 - v),
        270 => egui::pos2(1.0 - v, u),
        _ => egui::pos2(u, v),
    }
}

fn rotate_uv(u: f32, v: f32, rot: i32) -> egui::Pos2 {
    unrotate_uv(u, v, (360 - rot.rem_euclid(360)) % 360)
}

/// Display UV in desired-output space → UV in the baked preview texture.
fn map_display_uv_to_tex(u: f32, v: f32, desired: ViewOrient, baked: ViewOrient) -> egui::Pos2 {
    let mut x = u;
    let mut y = v;
    if desired.flip_v {
        y = 1.0 - y;
    }
    if desired.flip_h {
        x = 1.0 - x;
    }
    let src = unrotate_uv(x, y, desired.rot);
    let mid = rotate_uv(src.x, src.y, baked.rot);
    let mut tx = mid.x;
    let mut ty = mid.y;
    if baked.flip_h {
        tx = 1.0 - tx;
    }
    if baked.flip_v {
        ty = 1.0 - ty;
    }
    egui::pos2(tx, ty)
}

fn unrotate_px(dx: i32, dy: i32, dw: i32, dh: i32, rot: i32) -> (i32, i32) {
    match rot.rem_euclid(360) {
        90 => (dy, dw - 1 - dx),
        180 => (dw - 1 - dx, dh - 1 - dy),
        270 => (dh - 1 - dy, dx),
        _ => (dx, dy),
    }
}

fn rotate_px(sx: i32, sy: i32, sw: i32, sh: i32, rot: i32) -> (i32, i32) {
    match rot.rem_euclid(360) {
        90 => (sh - 1 - sy, sx),
        180 => (sw - 1 - sx, sh - 1 - sy),
        270 => (sy, sw - 1 - sx),
        _ => (sx, sy),
    }
}

fn display_to_tex_px(
    dx: u32,
    dy: u32,
    tex_w: u32,
    tex_h: u32,
    desired: ViewOrient,
    baked: ViewOrient,
) -> (u32, u32) {
    let view_rot = (desired.rot - baked.rot).rem_euclid(360);
    let (disp_w, disp_h) = if view_rot == 90 || view_rot == 270 {
        (tex_h, tex_w)
    } else {
        (tex_w, tex_h)
    };
    let mut x = dx.min(disp_w.saturating_sub(1)) as i32;
    let mut y = dy.min(disp_h.saturating_sub(1)) as i32;
    let dw = disp_w as i32;
    let dh = disp_h as i32;
    if desired.flip_v {
        y = dh - 1 - y;
    }
    if desired.flip_h {
        x = dw - 1 - x;
    }
    let (sx, sy) = unrotate_px(x, y, dw, dh, desired.rot);
    let (sw, sh) = if desired.rot == 90 || desired.rot == 270 {
        (dh, dw)
    } else {
        (dw, dh)
    };
    let (bx, by) = rotate_px(sx, sy, sw, sh, baked.rot);
    let mut tx = bx;
    let mut ty = by;
    if baked.flip_h {
        tx = tex_w as i32 - 1 - tx;
    }
    if baked.flip_v {
        ty = tex_h as i32 - 1 - ty;
    }
    (
        tx.clamp(0, tex_w.saturating_sub(1) as i32) as u32,
        ty.clamp(0, tex_h.saturating_sub(1) as i32) as u32,
    )
}

fn paint_preview_image(
    painter: &egui::Painter,
    tex: egui::TextureId,
    dest: egui::Rect,
    display_uv: egui::Rect,
    desired: ViewOrient,
    baked: ViewOrient,
) {
    if desired.matches(baked) {
        painter.image(tex, dest, display_uv, egui::Color32::WHITE);
        return;
    }
    let l = display_uv.min.x;
    let t = display_uv.min.y;
    let r = display_uv.max.x;
    let b = display_uv.max.y;
    let color = egui::Color32::WHITE;
    let mut mesh = egui::Mesh::with_texture(tex);
    let i0 = mesh.vertices.len() as u32;
    for (pos, uv) in [
        (dest.left_top(), map_display_uv_to_tex(l, t, desired, baked)),
        (
            dest.right_top(),
            map_display_uv_to_tex(r, t, desired, baked),
        ),
        (
            dest.right_bottom(),
            map_display_uv_to_tex(r, b, desired, baked),
        ),
        (
            dest.left_bottom(),
            map_display_uv_to_tex(l, b, desired, baked),
        ),
    ] {
        mesh.vertices.push(egui::epaint::Vertex { pos, uv, color });
    }
    mesh.add_triangle(i0, i0 + 1, i0 + 2);
    mesh.add_triangle(i0, i0 + 2, i0 + 3);
    painter.add(egui::Shape::mesh(mesh));
}

/// 1.0 = 1 preview-buffer pixel : 1 screen point (same units as the zoom %).
fn preview_pixel_scale(base_scale: f32, zoom: f32) -> f32 {
    base_scale * zoom
}

/// Tiles run at every zoom, including fit / zoom-out. The screen proxy is
/// always softer than full-res tiles (CFA box-downsample before demosaic).
fn proxy_is_soft(_pixel_scale: f32) -> bool {
    true
}

/// NEAREST only when magnifying the proxy; LINEAR for fit / minification.
fn want_nearest_filter(pixel_scale: f32) -> bool {
    pixel_scale > 1.0
}

/// LINEAR + mipmaps so minifying a grainy buffer averages like a downscaled export.
/// Without mipmaps, bilinear picks isolated grain peaks and the grain looks bigger.
fn preview_minify_texture_options() -> egui::TextureOptions {
    egui::TextureOptions::LINEAR.with_mipmap_mode(Some(egui::TextureFilter::Linear))
}

/// Full-res tiles: NEAREST when magnified (true 1:1), LINEAR + mipmaps when
/// minified so fit / zoom-out does not alias film grain into sparkle.
fn tile_texture_options() -> egui::TextureOptions {
    egui::TextureOptions {
        magnification: egui::TextureFilter::Nearest,
        minification: egui::TextureFilter::Linear,
        wrap_mode: egui::TextureWrapMode::ClampToEdge,
        mipmap_mode: Some(egui::TextureFilter::Linear),
    }
}

/// Place a tile on the virtual image, then expand / snap so a 1 px proxy
/// sliver cannot show on the top or right after pixel snapping.
fn tile_draw_rect(vir_rect: egui::Rect, uv: egui::Rect, img_w: f32, img_h: f32) -> egui::Rect {
    let mut r = egui::Rect::from_min_max(
        egui::pos2(
            vir_rect.left() + uv.min.x * img_w,
            vir_rect.top() + uv.min.y * img_h,
        ),
        egui::pos2(
            vir_rect.left() + uv.max.x * img_w,
            vir_rect.top() + uv.max.y * img_h,
        ),
    );
    r = r.expand(2.0);
    if uv.min.x <= 1e-5 {
        r.min.x = vir_rect.left();
    }
    if uv.min.y <= 1e-5 {
        r.min.y = vir_rect.top();
    }
    if uv.max.x >= 1.0 - 1e-5 {
        r.max.x = vir_rect.right();
    }
    if uv.max.y >= 1.0 - 1e-5 {
        r.max.y = vir_rect.bottom();
    }
    r
}

fn visible_priority_tiles_ready(
    tiles: &[PreviewTile],
    grid: &VisibleTileGrid,
    opt_hash: u64,
) -> bool {
    for iy in grid.iy0..=grid.iy1 {
        for ix in grid.ix0..=grid.ix1 {
            if !grid.is_priority(ix, iy) {
                continue;
            }
            if !tiles
                .iter()
                .any(|t| t.ix == ix && t.iy == iy && t.options_hash == opt_hash)
            {
                return false;
            }
        }
    }
    true
}

/// Working preview size: canvas × DPI, floored at 640 and capped at 1920×1200.
/// Zoom does not change this; full-res mode requests the export pipeline size.
fn preview_working_limits(canvas: Option<(f32, f32)>, ppp: f32, full_res: bool) -> (u32, u32) {
    if full_res {
        return (u32::MAX, u32::MAX);
    }
    let min_side = PREVIEW_MIN_SIDE as f32;
    let (base_w, base_h) = canvas
        .map(|(w, h)| {
            let pw = (w * ppp).round().clamp(min_side, PREVIEW_MAX_WIDTH as f32);
            let ph = (h * ppp).round().clamp(min_side, PREVIEW_MAX_HEIGHT as f32);
            (pw, ph)
        })
        .unwrap_or((PREVIEW_MAX_WIDTH as f32, PREVIEW_MAX_HEIGHT as f32));
    (base_w as u32, base_h as u32)
}

fn preview_invalidation_hash(
    path: &PathBuf,
    options: &PipelineOptions,
    full_res: bool,
    max_w: u32,
    max_h: u32,
) -> u64 {
    let base_hash = options_hash_for(path, options);
    let mut hh = DefaultHasher::new();
    base_hash.hash(&mut hh);
    full_res.hash(&mut hh);
    max_w.hash(&mut hh);
    max_h.hash(&mut hh);
    hh.finish()
}

/// Options + full-res flag only. Working size is not included so draft and screen share a key.
fn preview_options_hash(path: &PathBuf, options: &PipelineOptions, full_res: bool) -> u64 {
    preview_invalidation_hash(path, options, full_res, 0, 0)
}

fn preview_draft_limits() -> (u32, u32) {
    (PREVIEW_DRAFT_MAX, PREVIEW_DRAFT_MAX)
}

fn crop_histogram_hash(opts: &PipelineOptions) -> u64 {
    let mut h = DefaultHasher::new();
    opts.apply_crop.hash(&mut h);
    opts.crop_rect.hash(&mut h);
    opts.crop_rect_reference_size.hash(&mut h);
    h.finish()
}

fn image_entry(
    id: u64,
    path: PathBuf,
    options: PipelineOptions,
    export_format: ExportFormat,
) -> ImageEntry {
    ImageEntry {
        id,
        path,
        options,
        preview_texture: None,
        preview_texture_nearest: false,
        preview_texture_rotation: 0,
        preview_texture_flip_h: false,
        preview_texture_flip_v: false,
        preview_hash: 0,
        preview_options_hash: 0,
        preview_lod: PreviewLod::Draft,
        preview_screen_wh: (0, 0),
        preview_screen_requested_wh: (0, 0),
        tile_cache: Vec::new(),
        draft_step_cache: None,
        screen_step_cache: None,
        preview_full_rgb: None,
        preview_input_size: None,
        raw_source_size: None,
        preview_zoom: 1.0,
        preview_pan: egui::vec2(0.5, 0.5),
        thumbnail_texture: None,
        histogram: None,
        histogram_crop_hash: 0,
        export_format,
        raw_debug_report: None,
        pipeline_debug_log: None,
        cached_sensor: None,
        scene_stats: None,
        preview_step_cache: None,
        process_tab: ProcessTab::Input,
        dust_strokes: Vec::new(),
        dust_reference_size: None,
        dust_mask: Vec::new(),
        dust_mask_size: (0, 0),
        dust_view: DustView::Edit,
        dust_tool: Some(DustTool::Pen),
        dust_brush_radius: 8.0,
        dust_detect: 1.0,
        dust_feather: 6.0,
        dust_grain: 1.0,
        dust_grain_size: 2.0,
        dust_infill: DustInfill::PatchMatch,
        dust_tile: 3,
        dust_match: 2.0,
        dust_overlay_texture: None,
        dust_overlay_dirty: true,
    }
}

fn is_importable_image(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            IMPORT_EXTENSIONS
                .iter()
                .any(|ok| e.eq_ignore_ascii_case(ok))
        })
        .unwrap_or(false)
}

/// Files with a supported extension, plus supported files in dropped folders (one level).
fn collect_importable_paths(paths: impl IntoIterator<Item = PathBuf>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for path in paths {
        if path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&path) {
                let mut files: Vec<PathBuf> = entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.is_file() && is_importable_image(p))
                    .collect();
                files.sort();
                out.extend(files);
            }
        } else if is_importable_image(&path) {
            out.push(path);
        }
    }
    out
}

fn apply_runtime_gui_defaults(opts: &mut PipelineOptions) {
    opts.use_gpu = cfg!(feature = "gpu");
    opts.debug_pipeline_step = 6;
    opts.debug_preview_simple_debayer = false;
    opts.verbose_debug = false;
    opts.pinned_zone = None;
    opts.flat_field_path = None;
    opts.dust_mask = None;
}

fn normalize_runtime_options(opts: &mut PipelineOptions) {
    opts.use_gpu = false;
    opts.debug_pipeline_step = 6;
    opts.debug_preview_simple_debayer = false;
    opts.verbose_debug = false;
    opts.pinned_zone = None;
    opts.flat_field_path = None;
    opts.dust_mask = None;
}

fn normalize_persistable_options(opts: &mut PipelineOptions) {
    normalize_runtime_options(opts);
    opts.dust_mask_hash = 0;
    opts.dust_mask = None;
    opts.dust_strokes.clear();
    opts.dust_reference_size = None;
    opts.dust_uv = None;
    opts.dust_heal = DustHealParams::default();
}

fn persistable_project_images(images: &[ImageEntry]) -> Vec<ProjectImage> {
    images
        .iter()
        .map(|e| {
            let mut options = e.options.clone();
            normalize_persistable_options(&mut options);
            let dust = {
                let dust = ProjectDust {
                    reference_size: e.dust_reference_size.unwrap_or(e.dust_mask_size),
                    strokes: e.dust_strokes.clone(),
                    heal: entry_dust_heal(e),
                    brush_radius: e.dust_brush_radius,
                };
                if dust.is_empty() {
                    None
                } else {
                    Some(dust)
                }
            };
            ProjectImage {
                path: e.path.clone(),
                export_format: e.export_format.to_project(),
                options,
                dust,
            }
        })
        .collect()
}

fn edit_snapshot(entry: &ImageEntry) -> EditSnapshot {
    let mut options = entry.options.clone();
    normalize_runtime_options(&mut options);
    EditSnapshot {
        options,
        export_format: entry.export_format,
        dust_strokes: entry.dust_strokes.clone(),
        dust_reference_size: entry.dust_reference_size,
    }
}

fn restore_edit_snapshot(entry: &mut ImageEntry, snap: &EditSnapshot) {
    let use_gpu = entry.options.use_gpu;
    let debug_pipeline_step = entry.options.debug_pipeline_step;
    let debug_preview_simple_debayer = entry.options.debug_preview_simple_debayer;
    let verbose_debug = entry.options.verbose_debug;
    let pinned_zone = entry.options.pinned_zone;
    let flat_field_path = entry.options.flat_field_path.clone();
    entry.options = snap.options.clone();
    entry.options.use_gpu = use_gpu;
    entry.options.debug_pipeline_step = debug_pipeline_step;
    entry.options.debug_preview_simple_debayer = debug_preview_simple_debayer;
    entry.options.verbose_debug = verbose_debug;
    entry.options.pinned_zone = pinned_zone;
    entry.options.flat_field_path = flat_field_path;
    entry.export_format = snap.export_format;
    entry.dust_strokes = snap.dust_strokes.clone();
    entry.dust_reference_size = snap.dust_reference_size;
    rebuild_dust_raster(entry);
}

fn rebuild_dust_raster(entry: &mut ImageEntry) {
    let (w, h) = entry
        .dust_reference_size
        .or_else(|| entry.preview_input_size.map(|s| (s[0], s[1])))
        .unwrap_or((0, 0));
    if w == 0 || h == 0 {
        entry.dust_mask.clear();
        entry.dust_mask_size = (0, 0);
        entry.dust_overlay_dirty = true;
        return;
    }
    let mask = rasterize_strokes(
        &entry.dust_strokes,
        w,
        h,
        entry.dust_reference_size.unwrap_or((w, h)),
    );
    entry.dust_mask = mask.data;
    entry.dust_mask_size = (w, h);
    entry.dust_overlay_dirty = true;
}

fn dust_source_wh(entry: &ImageEntry, preview_w: u32, preview_h: u32) -> (f32, f32) {
    entry
        .raw_source_size
        .map(|[w, h]| (w as f32, h as f32))
        .unwrap_or((preview_w.max(1) as f32, preview_h.max(1) as f32))
}

fn ensure_dust_working_mask(entry: &mut ImageEntry, w: u32, h: u32) {
    if w == 0 || h == 0 {
        return;
    }
    if entry.dust_strokes.is_empty() {
        if let Some([sw, sh]) = entry.raw_source_size {
            if sw > 0 && sh > 0 {
                entry.dust_reference_size = Some((sw, sh));
            }
        }
    }
    if entry.dust_reference_size.is_none() {
        entry.dust_reference_size = Some((w, h));
    }
    if entry.dust_mask_size == (w, h) && entry.dust_mask.len() == w as usize * h as usize {
        return;
    }
    let mask = rasterize_strokes(
        &entry.dust_strokes,
        w,
        h,
        entry.dust_reference_size.unwrap_or((w, h)),
    );
    entry.dust_mask = mask.data;
    entry.dust_mask_size = (w, h);
    entry.dust_overlay_dirty = true;
}

fn apply_project_dust(entry: &mut ImageEntry, dust: ProjectDust) {
    entry.dust_detect = dust.heal.detect;
    entry.dust_feather = dust.heal.feather;
    entry.dust_grain = dust.heal.grain;
    entry.dust_grain_size = dust.heal.grain_sigma;
    entry.dust_infill = dust.heal.infill;
    entry.dust_tile = dust.heal.tile;
    entry.dust_match = dust.heal.match_loosen;
    entry.dust_brush_radius = dust.brush_radius;
    if !dust.strokes.is_empty() {
        entry.dust_strokes = dust.strokes;
        entry.dust_reference_size = Some(dust.reference_size);
        rebuild_dust_raster(entry);
    }
}

fn entry_dust_heal(entry: &ImageEntry) -> DustHealParams {
    let _ = entry.dust_grain_size;
    DustHealParams {
        detect: entry.dust_detect,
        feather: entry.dust_feather,
        grain: entry.dust_grain,
        grain_sigma: 2.0,
        infill: entry.dust_infill,
        tile: entry.dust_tile,
        match_loosen: entry.dust_match,
    }
}

const DUST_ERASER_RED: egui::Color32 = egui::Color32::from_rgb(220, 64, 64);

/// Classic block eraser (the icon font has no eraser glyph).
fn paint_eraser_icon(painter: &egui::Painter, center: egui::Pos2, size: f32, color: egui::Color32) {
    let s = size.max(8.0) * 0.5;
    let (sin, cos) = 35.0_f32.to_radians().sin_cos();
    let rot = |x: f32, y: f32| {
        egui::pos2(
            center.x + x * cos - y * sin,
            center.y + x * sin + y * cos,
        )
    };
    let body = [
        rot(-s * 1.05, -s * 0.55),
        rot(s * 0.85, -s * 0.55),
        rot(s * 1.05, s * 0.55),
        rot(-s * 0.85, s * 0.55),
    ];
    painter.add(egui::Shape::convex_polygon(
        body.to_vec(),
        color,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(48, 16, 16)),
    ));
    let band = [
        rot(s * 0.32, -s * 0.55),
        rot(s * 0.58, -s * 0.55),
        rot(s * 0.78, s * 0.55),
        rot(s * 0.52, s * 0.55),
    ];
    painter.add(egui::Shape::convex_polygon(
        band.to_vec(),
        egui::Color32::from_rgb(236, 140, 140),
        egui::Stroke::NONE,
    ));
}

fn oriented_export_size(entry: &ImageEntry) -> Option<(u32, u32)> {
    let [w, h] = entry.raw_source_size?;
    Some(oriented_sensor_size(w, h, entry.options.rotation_degrees))
}

fn attach_export_dust(opts: &mut PipelineOptions, entry: &ImageEntry) {
    if entry.dust_strokes.is_empty() {
        opts.dust_mask_hash = 0;
        opts.dust_mask = None;
        opts.dust_strokes.clear();
        opts.dust_reference_size = None;
        opts.dust_uv = None;
        return;
    }
    opts.dust_heal = entry_dust_heal(entry);
    opts.dust_mask_hash = hash_dust(&entry.dust_strokes, opts.dust_heal);
    opts.dust_strokes = entry.dust_strokes.clone();
    opts.dust_reference_size = entry
        .dust_reference_size
        .or(oriented_export_size(entry))
        .filter(|(w, h)| *w > 0 && *h > 0);
    opts.dust_mask = None;
    opts.dust_uv = None;
}

impl C41Gui {
    fn make_image_entry(
        &mut self,
        path: PathBuf,
        options: PipelineOptions,
        export_format: ExportFormat,
    ) -> ImageEntry {
        let id = self.next_image_id;
        self.next_image_id = self.next_image_id.wrapping_add(1);
        let entry = image_entry(id, path, options, export_format);
        self.history.track(id, edit_snapshot(&entry));
        entry
    }

    fn add_image_paths(&mut self, paths: impl IntoIterator<Item = PathBuf>) -> usize {
        let mut added = 0usize;
        for p in collect_importable_paths(paths) {
            if !self.images.iter().any(|e| e.path == p) {
                let entry = self.make_image_entry(p, default_options(), ExportFormat::Tiff16);
                self.images.push(entry);
                if self.selected_index.is_none() {
                    self.selected_index = Some(self.images.len() - 1);
                    self.full_res_preview_active = false;
                }
                added += 1;
            }
        }
        if added > 0 {
            self.status = format!("{} file(s)", self.images.len());
        }
        added
    }

    fn undo_edits(&mut self) {
        if let Some(states) = self.history.undo() {
            self.apply_history_restore(states);
            self.status = "Undo".to_string();
        }
    }

    fn redo_edits(&mut self) {
        if let Some(states) = self.history.redo() {
            self.apply_history_restore(states);
            self.status = "Redo".to_string();
        }
    }

    fn apply_history_restore(&mut self, states: Vec<(u64, EditSnapshot)>) {
        let mut first_idx = None;
        for (id, snap) in states {
            if let Some(idx) = self.images.iter().position(|e| e.id == id) {
                restore_edit_snapshot(&mut self.images[idx], &snap);
                if first_idx.is_none() {
                    first_idx = Some(idx);
                }
            }
        }
        if let Some(idx) = first_idx {
            self.selected_index = Some(idx);
            self.full_res_preview_active = false;
        }
    }

    fn commit_edit_history(&mut self, ctx: &egui::Context) {
        if self.auto_job.as_ref().is_some_and(|j| j.batch) {
            return;
        }
        let pointer_down = ctx.input(|i| i.pointer.any_down());
        let interacting = pointer_down || self.rect_dragging || self.dust_painting;
        let skip_id = if interacting {
            self.selected_index
                .and_then(|i| self.images.get(i).map(|e| e.id))
        } else {
            None
        };
        let current: Vec<(u64, EditSnapshot)> = self
            .images
            .iter()
            .map(|e| (e.id, edit_snapshot(e)))
            .collect();
        self.history.commit_settled(&current, skip_id.as_ref());
    }

    fn request_project_save(&mut self, save_as: bool) {
        if !save_as {
            if let Some(path) = self.project_path.clone() {
                let _ = self.write_project_to_path(&path);
                return;
            }
        }
        self.pending_project_save_as = true;
    }

    fn persistable_project(&self) -> Vec<ProjectImage> {
        persistable_project_images(&self.images)
    }

    fn mark_project_clean(&mut self) {
        self.clean_project = self.persistable_project();
    }

    fn project_is_dirty(&self) -> bool {
        self.persistable_project() != self.clean_project
    }

    fn request_leave_current_project(
        &mut self,
        ctx: &egui::Context,
        action: PendingLeaveAction,
    ) {
        if !self.project_is_dirty() {
            self.dispatch_leave_action(ctx, action);
            return;
        }
        self.save_before_leave = Some(action);
    }

    fn dispatch_leave_action(&mut self, ctx: &egui::Context, action: PendingLeaveAction) {
        match action {
            PendingLeaveAction::LoadDialog => {
                self.pending_project_load = true;
            }
            PendingLeaveAction::LoadRecent(path) => {
                self.pending_recent_load = Some(path);
            }
            PendingLeaveAction::Quit => {
                self.close_confirmed = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
    }

    fn confirm_save_then_leave(&mut self, ctx: &egui::Context, action: PendingLeaveAction) {
        if let Some(path) = self.project_path.clone() {
            if self.write_project_to_path(&path) {
                self.dispatch_leave_action(ctx, action);
            }
            return;
        }
        self.pending_save_then_leave = Some(action);
        self.pending_project_save_as = true;
    }

    fn write_project_to_path(&mut self, path: &Path) -> bool {
        let images = self.persistable_project();
        match save_project(&images, path) {
            Ok(()) => {
                self.project_path = Some(path.to_path_buf());
                self.clean_project = images;
                self.remember_recent_project(path);
                self.status = format!("Saved project: {}", path.display());
                true
            }
            Err(e) => {
                self.status = format!("Failed to save project: {e}");
                false
            }
        }
    }

    fn run_project_save_as_dialog(&mut self) -> bool {
        let mut dialog = rfd::FileDialog::new()
            .add_filter("Oxid Project", &[PROJECT_EXTENSION])
            .add_filter("JSON", &["json"]);
        if let Some(ref existing) = self.project_path {
            if let Some(parent) = existing.parent() {
                dialog = dialog.set_directory(parent);
            }
            if let Some(name) = existing.file_name() {
                dialog = dialog.set_file_name(name.to_string_lossy());
            }
        } else if let Some(first) = self.images.first() {
            if let Some(parent) = first.path.parent() {
                dialog = dialog.set_directory(parent);
            }
        }
        if let Some(mut path) = dialog.save_file() {
            if path.extension().is_none() {
                path.set_extension(PROJECT_EXTENSION);
            }
            return self.write_project_to_path(&path);
        }
        false
    }

    fn run_project_load_dialog(&mut self) {
        let mut dialog = rfd::FileDialog::new().add_filter(
            "Oxid Project",
            &[PROJECT_EXTENSION, PROJECT_EXTENSION_LEGACY, "json"],
        );
        if let Some(ref existing) = self.project_path {
            if let Some(parent) = existing.parent() {
                dialog = dialog.set_directory(parent);
            }
        } else if let Some(first) = self.images.first() {
            if let Some(parent) = first.path.parent() {
                dialog = dialog.set_directory(parent);
            }
        }
        if let Some(path) = dialog.pick_file() {
            self.load_project_from_path(path);
        }
    }

    fn load_project_from_path(&mut self, path: PathBuf) {
        match load_project(&path) {
            Ok(loaded) => self.apply_loaded_project(path, loaded),
            Err(e) => {
                if !path.exists() {
                    self.forget_recent_project(&path);
                }
                self.status = format!("Failed to load project: {e}");
            }
        }
    }

    fn remember_recent_project(&mut self, path: &Path) {
        let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        self.recent_projects.retain(|p| p != &path);
        self.recent_projects.insert(0, path);
        self.recent_projects.truncate(RECENT_PROJECTS_MAX);
        save_recent_projects(&self.recent_projects);
    }

    fn forget_recent_project(&mut self, path: &Path) {
        let canonical = path.canonicalize().ok();
        self.recent_projects
            .retain(|p| p != path && canonical.as_ref() != Some(p));
        save_recent_projects(&self.recent_projects);
    }

    fn apply_loaded_project(&mut self, path: PathBuf, loaded: LoadedProject) {
        self.preview_gen = self.preview_gen.wrapping_add(1);
        self.tile_gen = self.tile_gen.wrapping_add(1);
        self.preview_receiver = None;
        self.preview_started_at = None;
        self.preview_job_hash = None;
        self.preview_job_live = false;
        self.tile_receiver = None;
        self.tile_inflight = None;
        self.tile_failed.clear();
        self.thumbnail_receiver = None;
        self.thumbnail_pending.clear();
        self.full_res_preview_active = false;

        let missing_count = loaded.missing.len();
        let total = loaded.images.len() + missing_count;
        self.history.clear();
        self.images.clear();
        for img in loaded.images {
            let mut opts = img.options;
            apply_runtime_gui_defaults(&mut opts);
            let mut entry = self.make_image_entry(
                img.path,
                opts,
                ExportFormat::from_project(img.export_format),
            );
            if let Some(dust) = img.dust {
                apply_project_dust(&mut entry, dust);
            }
            self.images.push(entry);
        }
        self.selected_index = if self.images.is_empty() {
            None
        } else {
            Some(0)
        };
        self.project_path = Some(path.clone());
        self.mark_project_clean();
        self.remember_recent_project(&path);

        if missing_count == 0 {
            self.status = format!("Loaded {} image(s)", self.images.len());
        } else {
            self.status = format!(
                "Loaded {} of {} images ({} missing)",
                self.images.len(),
                total,
                missing_count
            );
        }
    }

    fn bake_preview_options(&self, entry: &ImageEntry) -> PipelineOptions {
        let mut options = entry.options.clone();
        options.flat_field_path = self.flat_field_path.clone();
        if !self.full_res_preview_active {
            if let Some((_, stats)) = entry.scene_stats {
                if let Some(dmin) = stats.dmin {
                    options.dmin_mode = DminMode::Fixed;
                    options.dmin_fixed = Some(dmin);
                }
                if entry.options.auto_wb {
                    if let Some((ar, ag, ab)) = stats.auto_wb {
                        options.auto_wb = false;
                        options.apply_white_balance = true;
                        options.wb_r *= ar;
                        options.wb_g *= ag;
                        options.wb_b *= ab;
                    }
                }
                options.pinned_zone = stats.zone;
            }
        }
        options.dust_heal = entry_dust_heal(entry);
        if self.dust_should_apply(entry) {
            options.dust_mask_hash = hash_dust(&entry.dust_strokes, options.dust_heal);
            options.dust_strokes = entry.dust_strokes.clone();
            options.dust_reference_size = entry
                .dust_reference_size
                .or(Some(entry.dust_mask_size))
                .filter(|(w, h)| *w > 0 && *h > 0);
            options.dust_mask = None;
            options.dust_uv = None;
        } else {
            options.dust_mask_hash = 0;
            options.dust_mask = None;
            options.dust_strokes.clear();
            options.dust_reference_size = None;
            options.dust_uv = None;
        }
        options
    }

    fn dust_should_apply(&self, entry: &ImageEntry) -> bool {
        if entry.dust_strokes.is_empty() {
            return false;
        }
        if self.mode == UIMode::Process {
            return match entry.process_tab {
                ProcessTab::Input | ProcessTab::Develop => false,
                ProcessTab::Dust => entry.dust_view == DustView::Process,
                ProcessTab::Export => true,
            };
        }
        true
    }

    fn begin_dust_brush_resize(&mut self, ctx: &egui::Context, radius: f32, pos: egui::Pos2) {
        self.dust_brush_resize = Some((radius, pos, 0.0));
        self.dust_painting = false;
        ctx.send_viewport_cmd(egui::ViewportCommand::CursorGrab(egui::CursorGrab::Locked));
        ctx.send_viewport_cmd(egui::ViewportCommand::CursorVisible(false));
    }

    fn end_dust_brush_resize(&mut self, ctx: &egui::Context) {
        if self.dust_brush_resize.take().is_some() {
            ctx.send_viewport_cmd(egui::ViewportCommand::CursorGrab(egui::CursorGrab::None));
            ctx.send_viewport_cmd(egui::ViewportCommand::CursorVisible(true));
        }
    }

    fn ensure_sensor_and_scene_stats(&mut self, index: usize) -> bool {
        if index >= self.images.len() {
            return false;
        }
        let path = self.images[index].path.clone();
        if self.images[index].cached_sensor.is_none() {
            match load_sensor_from_path(&path) {
                Ok(sensor) => {
                    self.images[index].cached_sensor = Some(Arc::new(sensor));
                }
                Err(e) => {
                    self.status = format!("Failed to load sensor data: {}", e);
                    return false;
                }
            }
        }
        if !self.full_res_preview_active {
            let mut options = self.images[index].options.clone();
            options.flat_field_path = self.flat_field_path.clone();
            let key = preview_scene_stats_key(&options);
            let stale = self.images[index].scene_stats.as_ref().map(|(k, _)| *k) != Some(key);
            if stale {
                if let Some(sensor) = self.images[index].cached_sensor.clone() {
                    match compute_preview_scene_stats(sensor.as_ref(), &options) {
                        Ok(stats) => self.images[index].scene_stats = Some((key, stats)),
                        Err(_) => self.images[index].scene_stats = None,
                    }
                }
            }
        }
        self.evict_unselected_heavy_caches();
        true
    }

    /// Drop full-res sensor / step caches / tiles / RGB buffers on every image.
    /// Keeps strip thumbnails and the current preview texture so the UI stays visible.
    fn release_heavy_caches(&mut self) {
        self.preview_gen = self.preview_gen.wrapping_add(1);
        self.tile_gen = self.tile_gen.wrapping_add(1);
        self.preview_receiver = None;
        self.preview_started_at = None;
        self.preview_job_hash = None;
        self.preview_job_live = false;
        self.pending_preview_key = None;
        self.pending_preview_since = None;
        self.tile_receiver = None;
        self.tile_inflight = None;
        self.tile_failed.clear();
        for entry in &mut self.images {
            entry.cached_sensor = None;
            entry.preview_step_cache = None;
            entry.draft_step_cache = None;
            entry.screen_step_cache = None;
            entry.tile_cache.clear();
            entry.preview_full_rgb = None;
        }
    }

    /// Keep `cached_sensor` and step caches for the selected image only.
    fn evict_unselected_heavy_caches(&mut self) {
        let keep = self.selected_index;
        for (i, entry) in self.images.iter_mut().enumerate() {
            if Some(i) == keep {
                continue;
            }
            entry.cached_sensor = None;
            entry.preview_step_cache = None;
            entry.draft_step_cache = None;
            entry.screen_step_cache = None;
            entry.tile_cache.clear();
        }
    }

    fn request_preview_for(&mut self, index: usize, ctx: &egui::Context, lod: PreviewLod) {
        // Auto's last step must not remosaic the full sensor on the UI thread
        // (that froze the progress dialog at 97%). Use cached scene stats instead.
        if self.auto_waiting_preview() != Some(index) && !self.ensure_sensor_and_scene_stats(index)
        {
            return;
        }
        let path = self.images[index].path.clone();
        let mut options = self.bake_preview_options(&self.images[index]);
        let capture_debug = self.capture_pipeline_debug_next;
        self.capture_pipeline_debug_next = false;
        options.verbose_debug = capture_debug;

        let (max_width, max_height) = match lod {
            PreviewLod::Draft => preview_draft_limits(),
            PreviewLod::Screen => {
                preview_working_limits(self.preview_canvas_size, ctx.pixels_per_point(), false)
            }
            PreviewLod::FullRes => (u32::MAX, u32::MAX),
        };
        if lod == PreviewLod::Screen {
            self.images[index].preview_screen_requested_wh = (max_width, max_height);
        }
        let options_hash = preview_options_hash(
            &path,
            &self.images[index].options,
            self.full_res_preview_active,
        );
        self.preview_job_hash = Some(options_hash);
        self.preview_job_live = false;

        let cache = match lod {
            PreviewLod::Draft => self.images[index].draft_step_cache.clone(),
            PreviewLod::Screen | PreviewLod::FullRes => {
                self.images[index].screen_step_cache.clone()
            }
        };
        let sensor = self.images[index].cached_sensor.clone();
        let gen = self.preview_gen;
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
                sensor.as_deref(),
            );
            #[cfg(not(feature = "gpu"))]
            let res = c41_raw_tool::process_one_to_preview_with_cache(
                &path,
                &options,
                max_width,
                max_height,
                cache.as_ref(),
                capture_debug,
                sensor.as_deref(),
            );
            let res =
                res.map(
                    |(input_w, input_h, w, h, rgb, dbg_log, new_cache)| PreviewJobResult {
                        gen,
                        lod,
                        index,
                        options_hash,
                        input_w,
                        input_h,
                        w,
                        h,
                        rgb,
                        dbg_log,
                        captured_debug: capture_debug,
                        new_cache,
                    },
                );
            let _ = tx.send(res);
        });
        ctx.request_repaint();
    }

    fn live_cache_limits(&self, index: usize, ctx: &egui::Context) -> (u32, u32) {
        let requested = self.images[index].preview_screen_requested_wh;
        if requested != (0, 0) {
            requested
        } else {
            preview_working_limits(self.preview_canvas_size, ctx.pixels_per_point(), false)
        }
    }

    fn live_preview_cache(&self, index: usize) -> Option<&PreviewStepCache> {
        let entry = self.images.get(index)?;
        entry
            .preview_step_cache
            .as_ref()
            .or(entry.screen_step_cache.as_ref())
            .or(entry.draft_step_cache.as_ref())
    }

    fn live_preview_available(&self, index: usize, ctx: &egui::Context) -> bool {
        let Some(cache) = self.live_preview_cache(index) else {
            return false;
        };
        let options = self.bake_preview_options(&self.images[index]);
        let (max_w, max_h) = self.live_cache_limits(index, ctx);
        cached_start_step(&self.images[index].path, &options, max_w, max_h, cache) >= 4
    }

    /// Apply remaining pipeline steps from the cached preview buffers. No remosaic.
    fn request_live_preview_for(&mut self, index: usize, ctx: &egui::Context) {
        let Some(cache) = self.live_preview_cache(index).cloned() else {
            return;
        };
        let path = self.images[index].path.clone();
        let options = self.bake_preview_options(&self.images[index]);
        let (max_width, max_height) = self.live_cache_limits(index, ctx);
        let options_hash = preview_options_hash(
            &path,
            &self.images[index].options,
            self.full_res_preview_active,
        );
        self.preview_job_hash = Some(options_hash);
        self.preview_job_live = true;
        let gen = self.preview_gen;
        let lod = self.images[index].preview_lod;
        #[cfg(feature = "gpu")]
        let gpu_arc = self.gpu_pipeline.clone();
        let (tx, rx) = mpsc::channel();
        self.preview_receiver = Some(rx);
        self.preview_started_at = Some(Instant::now());
        thread::spawn(move || {
            #[cfg(feature = "gpu")]
            let applied = apply_preview_from_cache_gpu(
                &path,
                &options,
                max_width,
                max_height,
                &cache,
                gpu_arc.as_deref(),
            );
            #[cfg(not(feature = "gpu"))]
            let applied = apply_preview_from_cache(&path, &options, max_width, max_height, &cache);
            let res = match applied {
                Some((input_w, input_h, w, h, rgb, new_cache)) => Ok(PreviewJobResult {
                    gen,
                    lod,
                    index,
                    options_hash,
                    input_w,
                    input_h,
                    w,
                    h,
                    rgb,
                    dbg_log: String::new(),
                    captured_debug: false,
                    new_cache,
                }),
                None => Err(anyhow::anyhow!("Live preview cache miss")),
            };
            let _ = tx.send(res);
        });
        ctx.request_repaint();
    }

    fn request_tile_for(&mut self, index: usize, ix: i32, iy: i32, ctx: &egui::Context) {
        if !self.ensure_sensor_and_scene_stats(index) {
            return;
        }
        if !self.full_res_preview_active && self.images[index].scene_stats.is_none() {
            return;
        }
        let path = self.images[index].path.clone();
        let entry = &self.images[index];
        let Some(sensor) = entry.cached_sensor.clone() else {
            return;
        };
        let (sw, sh) = sensor.dimensions();
        let (ow, oh) = oriented_sensor_size(sw, sh, entry.options.rotation_degrees);
        if ow == 0 || oh == 0 {
            return;
        }
        let (origin_x, origin_y, grid_w, grid_h) = tile_space(entry, ow, oh);
        let x = origin_x.saturating_add((ix as u32).saturating_mul(PREVIEW_TILE_SIZE));
        let y = origin_y.saturating_add((iy as u32).saturating_mul(PREVIEW_TILE_SIZE));
        if x >= ow || y >= oh || x >= origin_x + grid_w || y >= origin_y + grid_h {
            return;
        }
        let tw = PREVIEW_TILE_SIZE.min(ow - x).min(origin_x + grid_w - x);
        let th = PREVIEW_TILE_SIZE.min(oh - y).min(origin_y + grid_h - y);
        let halo = if entry.options.bujack_enabled {
            PREVIEW_TILE_HALO.max(entry.options.bujack_radius.ceil() as u32 + 8)
        } else {
            PREVIEW_TILE_HALO
        };
        let crop = match crop_sensor_for_oriented_rect(
            sensor.as_ref(),
            x,
            y,
            tw,
            th,
            entry.options.rotation_degrees,
            entry.options.flip_horizontal,
            entry.options.flip_vertical,
            halo,
        ) {
            Ok(c) => c,
            Err(e) => {
                self.status = format!("Tile crop failed: {}", e);
                self.tile_failed.push((self.tile_gen, ix, iy));
                return;
            }
        };
        // Place the requested 512² (plus 1 px overlap), not the halo crop.
        // Drawing the halo showed C-41 edge fringes as red seams.
        // Display UV is in tile-space (crop when active, else full sensor).
        let owf = ow as f32;
        let ohf = oh as f32;
        let gwf = (grid_w as f32).max(1.0);
        let ghf = (grid_h as f32).max(1.0);
        let overlap = 1.0;
        let sensor_l = (x as f32 - overlap).max(0.0);
        let sensor_t = (y as f32 - overlap).max(0.0);
        let sensor_r = ((x + tw) as f32 + overlap).min(owf);
        let sensor_b = ((y + th) as f32 + overlap).min(ohf);
        let inner_l = ((sensor_l - origin_x as f32) / gwf).clamp(0.0, 1.0);
        let inner_t = ((sensor_t - origin_y as f32) / ghf).clamp(0.0, 1.0);
        let inner_r = ((sensor_r - origin_x as f32) / gwf).clamp(0.0, 1.0);
        let inner_b = ((sensor_b - origin_y as f32) / ghf).clamp(0.0, 1.0);
        let uv =
            egui::Rect::from_min_max(egui::pos2(inner_l, inner_t), egui::pos2(inner_r, inner_b));
        let sensor_uv_l = sensor_l / owf;
        let sensor_uv_t = sensor_t / ohf;
        let sensor_uv_r = sensor_r / owf;
        let sensor_uv_b = sensor_b / ohf;
        let crop_uw = (crop.uv_right - crop.uv_left).max(1e-6);
        let crop_uh = (crop.uv_bottom - crop.uv_top).max(1e-6);
        let tex_uv = egui::Rect::from_min_max(
            egui::pos2(
                ((sensor_uv_l - crop.uv_left) / crop_uw).clamp(0.0, 1.0),
                ((sensor_uv_t - crop.uv_top) / crop_uh).clamp(0.0, 1.0),
            ),
            egui::pos2(
                ((sensor_uv_r - crop.uv_left) / crop_uw).clamp(0.0, 1.0),
                ((sensor_uv_b - crop.uv_top) / crop_uh).clamp(0.0, 1.0),
            ),
        );
        let mut options = self.bake_preview_options(entry);
        options.verbose_debug = false;
        if self.dust_should_apply(entry) && !entry.dust_strokes.is_empty() {
            options.dust_uv = Some((crop.uv_left, crop.uv_top, crop.uv_right, crop.uv_bottom));
        }
        // Same hash the cache lookup uses, so a finished tile is never "missing".
        let options_hash = entry.preview_options_hash;
        let gen = self.tile_gen;
        let tile_sensor = crop.sensor;
        #[cfg(feature = "gpu")]
        let gpu_arc = self.gpu_pipeline.clone();
        let (tx, rx) = mpsc::channel();
        self.tile_receiver = Some(rx);
        self.tile_inflight = Some((index, ix, iy));
        thread::spawn(move || {
            #[cfg(feature = "gpu")]
            let res = process_one_to_preview_with_cache_gpu(
                &path,
                &options,
                u32::MAX,
                u32::MAX,
                None,
                false,
                gpu_arc.as_deref(),
                Some(&tile_sensor),
            );
            #[cfg(not(feature = "gpu"))]
            let res = c41_raw_tool::process_one_to_preview_with_cache(
                &path,
                &options,
                u32::MAX,
                u32::MAX,
                None,
                false,
                Some(&tile_sensor),
            );
            let res = res.map(|(_iw, _ih, w, h, rgb, _dbg, _cache)| TileJobResult {
                gen,
                index,
                ix,
                iy,
                options_hash,
                w,
                h,
                rgb,
                uv,
                tex_uv,
            });
            let _ = tx.send(res);
        });
        ctx.request_repaint();
    }

    fn visible_tile_grid(&self, idx: usize) -> Option<VisibleTileGrid> {
        let entry = self.images.get(idx)?;
        let (tex_w, tex_h, _) = entry.preview_full_rgb.as_ref()?;
        let (canvas_w, canvas_h) = self.preview_canvas_size?;
        // Same oriented frame the tile crop uses, so indices match request_tile_for.
        let (ow, oh) = if let Some(s) = entry.cached_sensor.as_ref() {
            oriented_sensor_size(
                s.dimensions().0,
                s.dimensions().1,
                entry.options.rotation_degrees,
            )
        } else if let Some([w, h]) = entry.raw_source_size {
            (w, h)
        } else {
            return None;
        };
        if ow == 0 || oh == 0 || canvas_w <= 0.0 || canvas_h <= 0.0 {
            return None;
        }
        // Same display size as the draw path (crop when active).
        let (full_w, full_h) = preview_display_wh(entry, *tex_w, *tex_h);
        let full_w_f = (full_w as f32).max(1.0);
        let full_h_f = (full_h as f32).max(1.0);
        let zoom = entry.preview_zoom.max(1.0);
        let base_scale = (canvas_w / full_w_f).min(canvas_h / full_h_f);
        let img_w = (full_w_f * base_scale * zoom).max(1.0);
        let img_h = (full_h_f * base_scale * zoom).max(1.0);
        let (_, _, grid_w, grid_h) = tile_space(entry, ow, oh);
        // Same canvas ∩ virtual-image UV as the draw path.
        let pan_x = entry.preview_pan.x.clamp(0.0, 1.0);
        let pan_y = entry.preview_pan.y.clamp(0.0, 1.0);
        let vir_left = canvas_w * 0.5 - pan_x * img_w;
        let vir_top = canvas_h * 0.5 - pan_y * img_h;
        let vis_left = vir_left.max(0.0);
        let vis_top = vir_top.max(0.0);
        let vis_right = (vir_left + img_w).min(canvas_w);
        let vis_bottom = (vir_top + img_h).min(canvas_h);
        if vis_right <= vis_left || vis_bottom <= vis_top {
            return None;
        }
        let uv_l = ((vis_left - vir_left) / img_w).clamp(0.0, 1.0);
        let uv_t = ((vis_top - vir_top) / img_h).clamp(0.0, 1.0);
        let uv_r = ((vis_right - vir_left) / img_w).clamp(0.0, 1.0);
        let uv_b = ((vis_bottom - vir_top) / img_h).clamp(0.0, 1.0);
        // 2 screen pixels in UV so a sliver after zoom-out still counts.
        // Not a full tile — that used to inflate past PREVIEW_TILE_MAX.
        let pad_u = 2.0 / img_w;
        let pad_v = 2.0 / img_h;
        let uv_l = (uv_l - pad_u).clamp(0.0, 1.0);
        let uv_t = (uv_t - pad_v).clamp(0.0, 1.0);
        let uv_r = (uv_r + pad_u).clamp(0.0, 1.0);
        let uv_b = (uv_b + pad_v).clamp(0.0, 1.0);
        let tiles_x = ((grid_w + PREVIEW_TILE_SIZE - 1) / PREVIEW_TILE_SIZE) as i32;
        let tiles_y = ((grid_h + PREVIEW_TILE_SIZE - 1) / PREVIEW_TILE_SIZE) as i32;
        let (raw_ix0, raw_iy0, raw_ix1, raw_iy1) = tile_range_intersecting(
            uv_l * grid_w as f32,
            uv_t * grid_h as f32,
            uv_r * grid_w as f32,
            uv_b * grid_h as f32,
            PREVIEW_TILE_SIZE as f32,
            tiles_x,
            tiles_y,
        );
        let (ix0, iy0, ix1, iy1) = include_image_edge_tiles(
            uv_l, uv_t, uv_r, uv_b, tiles_x, tiles_y, raw_ix0, raw_iy0, raw_ix1, raw_iy1,
        );
        let nx = (ix1 - ix0 + 1) as usize;
        let ny = (iy1 - iy0 + 1) as usize;
        let core_n = nx.saturating_mul(ny);
        let grid = VisibleTileGrid {
            ix0,
            iy0,
            ix1,
            iy1,
            opt_hash: entry.preview_options_hash,
            proxy_soft: proxy_is_soft(base_scale * zoom),
            tiles_fit: core_n <= PREVIEW_TILE_MAX,
            core_n,
        };
        Some(grid)
    }

    fn tile_cache_limit(&self, idx: usize) -> usize {
        let Some(g) = self.visible_tile_grid(idx) else {
            return PREVIEW_TILE_LRU;
        };
        // Over-cap views still hold a center-first window of PREVIEW_TILE_MAX.
        g.core_n.max(PREVIEW_TILE_LRU).min(PREVIEW_TILE_MAX)
    }

    fn drop_tiles_outside_view(&mut self, idx: usize) {
        // Evict off-screen tiles only when over the cap so pan/zoom keeps the cache.
        self.evict_tile_cache(idx);
    }

    fn evict_tile_cache(&mut self, idx: usize) {
        if idx >= self.images.len() {
            return;
        }
        let limit = self.tile_cache_limit(idx);
        let grid = self.visible_tile_grid(idx);
        let cache = &mut self.images[idx].tile_cache;
        if cache.len() <= limit {
            return;
        }
        let Some(grid) = grid else {
            cache.truncate(limit);
            return;
        };
        let mut i = cache.len();
        while cache.len() > limit && i > 0 {
            i -= 1;
            let t = &cache[i];
            let visible = t.options_hash == grid.opt_hash && grid.is_priority(t.ix, t.iy);
            if !visible {
                cache.remove(i);
            }
        }
    }

    /// Next missing 1:1 tile, or None when the proxy is still sharp.
    /// Over-cap views still fetch the center-first priority window.
    fn visible_tile_to_request(&self, idx: usize) -> Option<(i32, i32)> {
        let grid = self.visible_tile_grid(idx)?;
        if !grid.proxy_soft {
            return None;
        }
        let entry = self.images.get(idx)?;
        // Scan missing priority tiles so zoom-out edges inside the cap window
        // are fetched even when the full core exceeds PREVIEW_TILE_MAX.
        let cx = (grid.ix0 + grid.ix1) as f32 * 0.5;
        let cy = (grid.iy0 + grid.iy1) as f32 * 0.5;
        let mut best: Option<(i32, i32, f32)> = None;
        for iy in grid.iy0..=grid.iy1 {
            for ix in grid.ix0..=grid.ix1 {
                if !grid.is_priority(ix, iy) {
                    continue;
                }
                if self.tile_inflight == Some((idx, ix, iy)) {
                    continue;
                }
                // Silhouette tiles are retried — a single failed top-row crop
                // used to blacklist the whole row for the rest of the gen.
                let on_silhouette =
                    iy == 0 || iy == grid.iy0 || iy == grid.iy1 || ix == grid.ix0 || ix == grid.ix1;
                if !on_silhouette
                    && self
                        .tile_failed
                        .iter()
                        .any(|&(g, fx, fy)| g == self.tile_gen && fx == ix && fy == iy)
                {
                    continue;
                }
                let have = entry
                    .tile_cache
                    .iter()
                    .any(|t| t.ix == ix && t.iy == iy && t.options_hash == grid.opt_hash);
                if have {
                    continue;
                }
                let mut d = (ix as f32 - cx).hypot(iy as f32 - cy);
                // Top sensor row first. Center-first left it last, and a
                // wrong iy0=1 meant it was never in the set at all.
                if iy == 0 || iy == grid.iy0 {
                    d -= 1.0e6;
                } else if iy == grid.iy1 || ix == grid.ix0 || ix == grid.ix1 {
                    d -= 1.0e5;
                }
                if best.map(|(_, _, bd)| d < bd).unwrap_or(true) {
                    best = Some((ix, iy, d));
                }
            }
        }
        best.map(|(ix, iy, _)| (ix, iy))
    }

    fn mark_preview_view_changed(&mut self) {
        // Pan/zoom only changes which tiles are visible. Do not bump tile_gen —
        // in-flight tiles stay valid and the cache is not thrown away.
        self.preview_view_dragging = true;
        self.preview_view_changed_at = Some(Instant::now());
    }

    fn preview_view_settling(&self) -> bool {
        self.preview_view_changed_at
            .map(|t| t.elapsed() < Duration::from_millis(PREVIEW_VIEW_SETTLE_MS))
            .unwrap_or(false)
    }

    fn preview_options_dirty(&self, idx: usize) -> bool {
        let Some(entry) = self.images.get(idx) else {
            return false;
        };
        let hash_now =
            preview_options_hash(&entry.path, &entry.options, self.full_res_preview_active);
        entry.preview_options_hash != hash_now
    }

    /// Instant: only update options. The current texture is drawn rotated until
    /// one coalesced pipeline job replaces it. No pixel work on the UI thread.
    fn apply_rotate_click(&mut self, idx: usize, clockwise: bool, ctx: &egui::Context) {
        if idx >= self.images.len() {
            return;
        }

        {
            let entry = &mut self.images[idx];
            let preview_size = entry.preview_input_size.map(|[w, h]| (w, h));
            if let Some(rect) = entry.options.dmin_rect {
                let source_size = entry.options.dmin_rect_reference_size.or(preview_size);
                if let Some((w, h)) = source_size {
                    entry.options.dmin_rect = Some(rotate_dmin_rect_90(rect, w, h, clockwise));
                    entry.options.dmin_rect_reference_size = Some((h, w));
                }
            }
            if let Some(rect) = entry.options.crop_rect {
                let source_size = entry.options.crop_rect_reference_size.or(preview_size);
                if let Some((w, h)) = source_size {
                    entry.options.crop_rect = Some(rotate_dmin_rect_90(rect, w, h, clockwise));
                    entry.options.crop_rect_reference_size = Some((h, w));
                }
            }
            let delta = if clockwise { 90 } else { -90 };
            entry.options.rotation_degrees =
                (entry.options.rotation_degrees + delta).rem_euclid(360);
            if let Some([iw, ih]) = entry.preview_input_size {
                entry.preview_input_size = Some([ih, iw]);
            }
            if let Some([rw, rh]) = entry.raw_source_size {
                entry.raw_source_size = Some([rh, rw]);
            }
            let (sw, sh) = entry.preview_screen_wh;
            entry.preview_screen_wh = (sh, sw);
            entry.tile_cache.clear();
        }

        self.begin_geometry_coalesce(ctx);
    }

    /// Instant: only update options. The current texture is drawn flipped until
    /// one coalesced pipeline job replaces it. No pixel work on the UI thread.
    fn apply_flip_click(&mut self, idx: usize, horizontal: bool, ctx: &egui::Context) {
        if idx >= self.images.len() {
            return;
        }

        {
            let entry = &mut self.images[idx];
            let preview_size = entry.preview_input_size.map(|[w, h]| (w, h));
            if horizontal {
                if let Some(rect) = entry.options.dmin_rect {
                    let source_size = entry.options.dmin_rect_reference_size.or(preview_size);
                    if let Some((w, h)) = source_size {
                        entry.options.dmin_rect = Some(flip_rect_horizontal(rect, w, h));
                    }
                }
                if let Some(rect) = entry.options.crop_rect {
                    let source_size = entry.options.crop_rect_reference_size.or(preview_size);
                    if let Some((w, h)) = source_size {
                        entry.options.crop_rect = Some(flip_rect_horizontal(rect, w, h));
                    }
                }
                entry.options.flip_horizontal = !entry.options.flip_horizontal;
            } else {
                if let Some(rect) = entry.options.dmin_rect {
                    let source_size = entry.options.dmin_rect_reference_size.or(preview_size);
                    if let Some((w, h)) = source_size {
                        entry.options.dmin_rect = Some(flip_rect_vertical(rect, w, h));
                    }
                }
                if let Some(rect) = entry.options.crop_rect {
                    let source_size = entry.options.crop_rect_reference_size.or(preview_size);
                    if let Some((w, h)) = source_size {
                        entry.options.crop_rect = Some(flip_rect_vertical(rect, w, h));
                    }
                }
                entry.options.flip_vertical = !entry.options.flip_vertical;
            }
            entry.tile_cache.clear();
        }

        self.begin_geometry_coalesce(ctx);
    }

    fn begin_geometry_coalesce(&mut self, ctx: &egui::Context) {
        self.preview_gen = self.preview_gen.wrapping_add(1);
        self.preview_receiver = None;
        self.preview_started_at = None;
        self.preview_job_live = false;
        self.tile_gen = self.tile_gen.wrapping_add(1);
        self.tile_inflight = None;
        self.tile_failed.clear();
        self.pending_preview_key = None;
        self.pending_preview_since = None;
        self.geometry_coalesce_until =
            Some(Instant::now() + Duration::from_millis(GEOMETRY_COALESCE_MS));
        ctx.request_repaint();
    }

    fn start_export(&mut self, ctx: &egui::Context, selected_only: bool) {
        if self.heavy_job_running() {
            return;
        }
        let Some(output_dir) = self.output_dir.clone() else {
            return;
        };
        let export_template = self
            .selected_index
            .filter(|&i| i < self.images.len())
            .map(|i| self.images[i].options.clone())
            .or_else(|| self.images.first().map(|img| img.options.clone()));
        let Some(export_template) = export_template else {
            return;
        };

        let jobs: Vec<ExportJobSpec> = if selected_only {
            match self.selected_index {
                Some(idx) if idx < self.images.len() => {
                    let img = &self.images[idx];
                    let mut opts = img.options.clone();
                    opts.flat_field_path = self.flat_field_path.clone();
                    attach_export_dust(&mut opts, img);
                    vec![ExportJobSpec {
                        path: img.path.clone(),
                        options: opts,
                        source_size: img.raw_source_size.map(|[w, h]| (w, h)),
                    }]
                }
                _ => return,
            }
        } else {
            self.images
                .iter()
                .map(|img| {
                    let mut opts = img.options.clone();
                    opts.flat_field_path = self.flat_field_path.clone();
                    attach_export_dust(&mut opts, img);
                    opts.format = export_template.format;
                    opts.write_exr = export_template.write_exr;
                    opts.write_jpeg = export_template.write_jpeg;
                    opts.write_jpeg_only = export_template.write_jpeg_only;
                    opts.export_aces_exr = export_template.export_aces_exr;
                    opts.write_aces2065_only = export_template.write_aces2065_only;
                    ExportJobSpec {
                        path: img.path.clone(),
                        options: opts,
                        source_size: img.raw_source_size.map(|[w, h]| (w, h)),
                    }
                })
                .collect()
        };
        if jobs.is_empty() {
            return;
        }

        self.release_heavy_caches();

        let total = jobs.len();
        let control = Arc::new(ExportControl::new(total));
        let (tx, rx) = mpsc::channel();
        let control_thread = control.clone();
        thread::spawn(move || {
            match process_export_jobs(&jobs, &output_dir, Some(control_thread.as_ref())) {
                Ok(()) => {
                    let _ = tx.send(ExportJobOutcome::Done {
                        count: control_thread.completed(),
                    });
                }
                Err(e) if e.downcast_ref::<ExportCancelled>().is_some() => {
                    let _ = tx.send(ExportJobOutcome::Cancelled {
                        completed: control_thread.completed(),
                    });
                }
                Err(e) => {
                    let _ = tx.send(ExportJobOutcome::Error(e.to_string()));
                }
            }
        });

        self.export_job = Some(ExportJob {
            receiver: rx,
            started_at: Instant::now(),
            control,
        });
        self.status = if selected_only {
            "Exporting selected…".to_string()
        } else {
            format!("Exporting {} images…", total)
        };
        ctx.request_repaint();
    }

    fn poll_export_job(&mut self, ctx: &egui::Context) {
        let Some(job) = self.export_job.as_ref() else {
            return;
        };
        match job.receiver.try_recv() {
            Ok(ExportJobOutcome::Done { count }) => {
                self.export_job = None;
                self.status = if count == 1 {
                    "Exported 1 image.".to_string()
                } else {
                    format!("Exported {} images.", count)
                };
            }
            Ok(ExportJobOutcome::Cancelled { completed }) => {
                self.export_job = None;
                self.status = if completed == 0 {
                    "Export cancelled.".to_string()
                } else {
                    format!("Export cancelled after {} image(s).", completed)
                };
            }
            Ok(ExportJobOutcome::Error(e)) => {
                self.export_job = None;
                self.status = format!("Error: {}", e);
            }
            Err(mpsc::TryRecvError::Empty) => {
                ctx.request_repaint();
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                self.export_job = None;
                self.status = "Export stopped unexpectedly.".to_string();
            }
        }
    }

    fn show_export_progress(&mut self, ctx: &egui::Context) {
        let Some(job) = self.export_job.as_ref() else {
            return;
        };
        if job.started_at.elapsed() < Duration::from_millis(EXPORT_PROGRESS_DELAY_MS) {
            ctx.request_repaint_after(Duration::from_millis(
                EXPORT_PROGRESS_DELAY_MS
                    .saturating_sub(job.started_at.elapsed().as_millis() as u64),
            ));
            return;
        }

        let control = job.control.clone();
        let progress = control.snapshot();
        let cancelling = control.is_cancelled();
        let fraction = progress.fraction.clamp(0.0, 1.0);
        let shown = if progress.completed < progress.total && !progress.file_name.is_empty() {
            progress.completed.saturating_add(1).min(progress.total)
        } else {
            progress.completed.min(progress.total)
        };
        let label = if progress.total > 1 {
            format!("{}  ({}/{})", progress.file_name, shown, progress.total)
        } else if !progress.file_name.is_empty() {
            progress.file_name.clone()
        } else {
            "Exporting…".to_string()
        };
        let stage = if cancelling {
            "Cancelling…".to_string()
        } else {
            progress.stage
        };

        egui::Window::new("Exporting")
            .collapsible(false)
            .resizable(false)
            .movable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.set_min_width(320.0);
                ui.add_space(4.0);
                ui.label(egui::RichText::new(&label).strong());
                ui.add_space(6.0);
                ui.add(
                    egui::ProgressBar::new(fraction)
                        .desired_width(320.0)
                        .show_percentage(),
                );
                ui.add_space(4.0);
                ui.label(egui::RichText::new(&stage).small());
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let cancel = ui.add_enabled(!cancelling, egui::Button::new("Cancel"));
                        if cancel.clicked() {
                            control.request_cancel();
                        }
                    });
                });
            });
    }

    fn open_batch_export_dialog(&mut self) {
        if self.images.is_empty() || self.heavy_job_running() {
            return;
        }
        let template = self
            .selected_index
            .filter(|&i| i < self.images.len())
            .and_then(|i| self.images.get(i))
            .or_else(|| self.images.first());
        let Some(img) = template else {
            return;
        };
        self.batch_export_dialog = Some(BatchExportDialog {
            format: img.export_format,
            write_jpeg: img.options.write_jpeg && img.export_format != ExportFormat::Jpeg,
        });
    }

    fn apply_batch_export_dialog(&mut self, dialog: &BatchExportDialog) {
        let idx = self
            .selected_index
            .filter(|&i| i < self.images.len())
            .unwrap_or(0);
        let Some(img) = self.images.get_mut(idx) else {
            return;
        };
        img.export_format = dialog.format;
        apply_export_format_to_options(&mut img.options, dialog.format);
        if dialog.format != ExportFormat::Jpeg {
            img.options.write_jpeg = dialog.write_jpeg;
        }
    }

    fn show_batch_export_dialog(&mut self, ctx: &egui::Context) {
        let Some(mut dialog) = self.batch_export_dialog.take() else {
            return;
        };
        if dialog.format == ExportFormat::Jpeg {
            dialog.write_jpeg = false;
        }

        let mut close = false;
        let mut start = false;
        let ready = !self.images.is_empty() && !self.heavy_job_running();
        let out_label = self
            .output_dir
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "No output folder".to_string());

        egui::Window::new("Export")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.set_min_width(320.0);
                ui.add_space(4.0);
                export_format_combo(ui, &mut dialog.format);
                ui.add_enabled(
                    dialog.format != ExportFormat::Jpeg,
                    egui::Checkbox::new(&mut dialog.write_jpeg, "Also export JPG"),
                );
                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);
                if ui
                    .button(theme::icon_label(theme::FOLDER, "Output folder…"))
                    .clicked()
                {
                    if let Some(path) = rfd::FileDialog::new().pick_folder() {
                        self.output_dir = Some(path);
                    }
                }
                ui.label(egui::RichText::new(out_label).small());
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add_enabled(
                                ready,
                                egui::Button::new(theme::icon_label(theme::DOWNLOAD, "Export")),
                            )
                            .clicked()
                        {
                            if self.output_dir.is_none() {
                                if let Some(path) = rfd::FileDialog::new().pick_folder() {
                                    self.output_dir = Some(path);
                                }
                            }
                            if self.output_dir.is_some() {
                                start = true;
                                close = true;
                            }
                        }
                        if ui
                            .button(theme::icon_label(theme::CANCEL, "Cancel"))
                            .clicked()
                        {
                            close = true;
                        }
                    });
                });
            });

        if start {
            self.apply_batch_export_dialog(&dialog);
            self.start_export(ctx, false);
        }
        if !close {
            self.batch_export_dialog = Some(dialog);
        }
    }

    fn show_save_before_leave_dialog(&mut self, ctx: &egui::Context) {
        let Some(action) = self.save_before_leave.take() else {
            return;
        };
        let mut keep = true;
        let mut yes = false;
        let mut no = false;
        egui::Window::new("Save project")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.set_min_width(320.0);
                ui.add_space(4.0);
                ui.label("Do you wish to save your project before closing it?");
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Yes").clicked() {
                            yes = true;
                            keep = false;
                        }
                        if ui.button("No").clicked() {
                            no = true;
                            keep = false;
                        }
                        if ui.button("Cancel").clicked() {
                            keep = false;
                        }
                    });
                });
            });
        if yes {
            self.confirm_save_then_leave(ctx, action);
        } else if no {
            self.dispatch_leave_action(ctx, action);
        } else if keep {
            self.save_before_leave = Some(action);
        }
    }

    fn start_auto(&mut self, ctx: &egui::Context) {
        if self.heavy_job_running() || self.auto_job.is_some() {
            return;
        }
        let Some(index) = self.selected_index else {
            self.status = "Auto: select an image first.".to_string();
            return;
        };
        if index >= self.images.len() {
            return;
        }
        if !self.ensure_sensor_and_scene_stats(index) {
            return;
        }

        let path = self.images[index].path.clone();
        let options = self.bake_preview_options(&self.images[index]);
        let after_step3 = self.images[index]
            .preview_step_cache
            .as_ref()
            .and_then(|c| c.after_step3.as_ref().map(|(_, buf)| buf.clone()))
            .or_else(|| {
                self.images[index]
                    .screen_step_cache
                    .as_ref()
                    .and_then(|c| c.after_step3.as_ref().map(|(_, buf)| buf.clone()))
            })
            .or_else(|| {
                self.images[index]
                    .draft_step_cache
                    .as_ref()
                    .and_then(|c| c.after_step3.as_ref().map(|(_, buf)| buf.clone()))
            });
        let cache = self.images[index]
            .preview_step_cache
            .clone()
            .or_else(|| self.images[index].screen_step_cache.clone())
            .or_else(|| self.images[index].draft_step_cache.clone());
        let sensor = self.images[index].cached_sensor.clone();
        let (max_width, max_height) = preview_draft_limits();

        let progress = Arc::new(Mutex::new(AutoProgressState {
            fraction: 0.0,
            message: "Preparing analysis…".to_string(),
            log: vec!["Preparing analysis…".to_string()],
            file_name: String::new(),
            completed: 0,
            total: 1,
        }));
        let (tx, rx) = mpsc::channel();
        let progress_worker = progress.clone();
        let ctx_worker = ctx.clone();
        thread::spawn(move || {
            let report = |message: &str, fraction: f32, log_line: Option<&str>| {
                if let Ok(mut p) = progress_worker.lock() {
                    p.fraction = fraction.clamp(0.0, 1.0);
                    p.message = message.to_string();
                    if let Some(line) = log_line {
                        if p.log.last().map(|s| s.as_str()) != Some(line) {
                            p.log.push(line.to_string());
                        }
                    }
                }
                ctx_worker.request_repaint();
            };

            let buf = match after_step3 {
                Some(img) => img,
                None => {
                    report("Preparing analysis…", 0.02, Some("Preparing analysis…"));
                    match process_one_to_preview_with_cache(
                        &path,
                        &options,
                        max_width,
                        max_height,
                        cache.as_ref(),
                        false,
                        sensor.as_deref(),
                    ) {
                        Ok((_, _, _, _, _, _, new_cache)) => match new_cache.after_step3 {
                            Some((_, img)) => img,
                            None => {
                                let _ = tx.send(AutoJobOutcome::Error {
                                    message: "Auto: no D-min buffer to analyse.".to_string(),
                                });
                                return;
                            }
                        },
                        Err(e) => {
                            let _ = tx.send(AutoJobOutcome::Error {
                                message: format!("Auto: failed to load preview ({e})"),
                            });
                            return;
                        }
                    }
                }
            };

            let mut on_progress = |message: &str, fraction: f32, log_line: Option<&str>| {
                report(message, fraction, log_line);
            };
            match auto_tune(&buf, &options, &mut on_progress) {
                Ok(result) => {
                    let _ = tx.send(AutoJobOutcome::Done { index, result });
                }
                Err(e) => {
                    let _ = tx.send(AutoJobOutcome::Error {
                        message: format!("Auto: {e}"),
                    });
                }
            }
            ctx_worker.request_repaint();
        });

        self.auto_job = Some(AutoJob {
            receiver: rx,
            progress,
            cancel: None,
            batch: false,
            title: "Auto",
            applying_preview: None,
            ticker_stop: Arc::new(AtomicBool::new(false)),
        });
        ctx.request_repaint();
    }

    fn heavy_job_running(&self) -> bool {
        self.export_job.is_some()
            || self
                .auto_job
                .as_ref()
                .is_some_and(|j| j.batch || j.applying_preview.is_some())
    }

    fn auto_waiting_preview(&self) -> Option<usize> {
        self.auto_job.as_ref().and_then(|j| j.applying_preview)
    }

    fn set_auto_progress(&self, message: &str, fraction: f32) {
        let Some(job) = self.auto_job.as_ref() else {
            return;
        };
        if let Ok(mut p) = job.progress.lock() {
            p.fraction = fraction.clamp(0.0, 1.0);
            p.message = message.to_string();
            if p.log.last().map(|s| s.as_str()) != Some(message) {
                p.log.push(message.to_string());
            }
        }
    }

    fn begin_auto_preview_wait(&mut self, index: usize, ctx: &egui::Context) {
        self.set_auto_progress("Applying settings…", 0.97);
        if index >= self.images.len() {
            self.auto_job = None;
            return;
        }
        if let Some(job) = self.auto_job.as_mut() {
            job.applying_preview = Some(index);
        }
        if let Some(stop) = self.auto_job.as_ref().map(|j| j.ticker_stop.clone()) {
            let ctx_tick = ctx.clone();
            thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    ctx_tick.request_repaint();
                    thread::sleep(Duration::from_millis(50));
                }
            });
        }
        self.preview_gen = self.preview_gen.wrapping_add(1);
        self.preview_receiver = None;
        self.preview_started_at = None;
        self.preview_job_live = false;
        self.pending_preview_key = None;
        self.pending_preview_since = None;
        self.start_auto_preview_job(index, ctx);
        ctx.request_repaint();
    }

    /// Apply Auto settings on a worker, reporting 97 / 98 / 99 / 100 as each
    /// remaining pipeline chunk finishes so the dialog can paint.
    fn start_auto_preview_job(&mut self, index: usize, ctx: &egui::Context) {
        if index >= self.images.len() {
            return;
        }
        let path = self.images[index].path.clone();
        let options = self.bake_preview_options(&self.images[index]);
        let options_hash = preview_options_hash(
            &path,
            &self.images[index].options,
            self.full_res_preview_active,
        );
        let (max_width, max_height) =
            preview_working_limits(self.preview_canvas_size, ctx.pixels_per_point(), false);
        let cache = self.live_preview_cache(index).cloned();
        let sensor = self.images[index].cached_sensor.clone();
        let gen = self.preview_gen;
        let progress = self.auto_job.as_ref().map(|j| j.progress.clone());
        let ctx_worker = ctx.clone();
        let (tx, rx) = mpsc::channel();
        self.preview_receiver = Some(rx);
        self.preview_started_at = Some(Instant::now());
        self.preview_job_hash = Some(options_hash);
        self.preview_job_live = cache
            .as_ref()
            .is_some_and(|c| cached_start_step(&path, &options, max_width, max_height, c) >= 4);

        thread::spawn(move || {
            let mut report = |message: &str, fraction: f32| {
                if let Some(progress) = &progress {
                    if let Ok(mut p) = progress.lock() {
                        p.fraction = fraction.clamp(0.0, 1.0);
                        p.message = message.to_string();
                        if p.log.last().map(|s| s.as_str()) != Some(message) {
                            p.log.push(message.to_string());
                        }
                    }
                }
                ctx_worker.request_repaint();
            };
            report("Applying settings…", 0.97);

            let applied = cache.as_ref().and_then(|c| {
                apply_preview_from_cache_on_progress(
                    &path,
                    &options,
                    max_width,
                    max_height,
                    c,
                    &mut report,
                )
            });
            let res = if let Some((input_w, input_h, w, h, rgb, new_cache)) = applied {
                Ok(PreviewJobResult {
                    gen,
                    lod: PreviewLod::Screen,
                    index,
                    options_hash,
                    input_w,
                    input_h,
                    w,
                    h,
                    rgb,
                    dbg_log: String::new(),
                    captured_debug: false,
                    new_cache,
                })
            } else {
                process_one_to_preview_with_cache_on_progress(
                    &path,
                    &options,
                    max_width,
                    max_height,
                    cache.as_ref(),
                    false,
                    sensor.as_deref(),
                    &mut report,
                )
                .map(|(input_w, input_h, w, h, rgb, dbg_log, new_cache)| {
                    PreviewJobResult {
                        gen,
                        lod: PreviewLod::Screen,
                        index,
                        options_hash,
                        input_w,
                        input_h,
                        w,
                        h,
                        rgb,
                        dbg_log,
                        captured_debug: false,
                        new_cache,
                    }
                })
            };
            let _ = tx.send(res);
            ctx_worker.request_repaint();
        });
    }

    fn finish_auto_if_preview_ready(&mut self, ctx: &egui::Context) {
        let Some(idx) = self.auto_job.as_ref().and_then(|j| j.applying_preview) else {
            return;
        };
        if idx >= self.images.len() {
            self.auto_job = None;
            return;
        }
        let hash_now = preview_options_hash(
            &self.images[idx].path,
            &self.images[idx].options,
            self.full_res_preview_active,
        );
        let progress_done = self
            .auto_job
            .as_ref()
            .and_then(|j| j.progress.lock().ok())
            .is_some_and(|p| p.fraction >= 1.0);
        let ready = self.preview_receiver.is_none()
            && self.images[idx].preview_texture.is_some()
            && (self.images[idx].preview_options_hash == hash_now || progress_done);
        if ready {
            self.set_auto_progress("Done", 1.0);
            self.auto_job = None;
        } else {
            ctx.request_repaint();
        }
    }

    fn apply_auto_result_to_path(&mut self, path: &Path, result: &AutoTuneResult) -> bool {
        let Some(entry) = self.images.iter_mut().find(|e| e.path == path) else {
            return false;
        };
        result.apply_to(&mut entry.options);
        entry.preview_hash = 0;
        entry.preview_options_hash = 0;
        true
    }

    fn start_batch_auto(&mut self, ctx: &egui::Context) {
        if self.heavy_job_running() || self.auto_job.is_some() {
            return;
        }
        if self.images.is_empty() {
            self.status = "Auto Develop: add images first.".to_string();
            return;
        }

        self.release_heavy_caches();

        let jobs: Vec<(PathBuf, PipelineOptions)> = self
            .images
            .iter()
            .map(|img| {
                let mut opts = img.options.clone();
                opts.flat_field_path = self.flat_field_path.clone();
                (img.path.clone(), opts)
            })
            .collect();
        let total = jobs.len();
        let cancel = Arc::new(AtomicBool::new(false));
        let completed = Arc::new(AtomicUsize::new(0));
        let progress = Arc::new(Mutex::new(AutoProgressState {
            fraction: 0.0,
            message: "Starting…".to_string(),
            log: Vec::new(),
            file_name: String::new(),
            completed: 0,
            total,
        }));
        let (tx, rx) = mpsc::channel();
        let workers = if total > 1 { 2 } else { 1 };
        let cancel_thread = cancel.clone();
        let progress_thread = progress.clone();
        let ctx_thread = ctx.clone();

        thread::spawn(move || {
            let queue = Arc::new(Mutex::new(jobs.into_iter().collect::<VecDeque<_>>()));
            let errors = Arc::new(AtomicUsize::new(0));

            thread::scope(|scope| {
                for _ in 0..workers {
                    let tx = tx.clone();
                    let ctx = ctx_thread.clone();
                    let cancel_thread = cancel_thread.clone();
                    let progress_thread = progress_thread.clone();
                    let completed = completed.clone();
                    let queue = queue.clone();
                    let errors = errors.clone();
                    scope.spawn(move || loop {
                        if cancel_thread.load(Ordering::Relaxed) {
                            return;
                        }
                        let Some((path, opts)) = queue.lock().ok().and_then(|mut q| q.pop_front())
                        else {
                            return;
                        };
                        let name = path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("image")
                            .to_string();
                        if let Ok(mut p) = progress_thread.lock() {
                            p.file_name = name.clone();
                            p.message = "Loading…".to_string();
                        }
                        ctx.request_repaint();

                        if cancel_thread.load(Ordering::Relaxed) {
                            return;
                        }

                        let done_at_start = completed.load(Ordering::Relaxed);
                        let mut on_progress = |message: &str, frac: f32, _log: Option<&str>| {
                            if let Ok(mut p) = progress_thread.lock() {
                                p.message = message.to_string();
                                p.file_name = name.clone();
                                p.completed = done_at_start;
                                let n = total.max(1) as f32;
                                p.fraction = (done_at_start as f32 + frac.clamp(0.0, 1.0)) / n;
                            }
                            ctx.request_repaint();
                        };

                        match run_auto_for_path(&path, &opts, &mut on_progress) {
                            Ok(result) => {
                                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                                if let Ok(mut p) = progress_thread.lock() {
                                    p.completed = done;
                                    p.fraction = done as f32 / total.max(1) as f32;
                                }
                                let _ = tx.send(AutoJobOutcome::FileDone { path, result });
                            }
                            Err(_) => {
                                errors.fetch_add(1, Ordering::Relaxed);
                                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                                if let Ok(mut p) = progress_thread.lock() {
                                    p.completed = done;
                                    p.fraction = done as f32 / total.max(1) as f32;
                                    p.message = format!("Skipped {}", name);
                                }
                            }
                        }
                        ctx.request_repaint();
                    });
                }
            });

            let done = completed.load(Ordering::Relaxed);
            let err_n = errors.load(Ordering::Relaxed);
            if cancel_thread.load(Ordering::Relaxed) {
                let _ = tx.send(AutoJobOutcome::Cancelled { completed: done });
            } else {
                let _ = tx.send(AutoJobOutcome::BatchDone {
                    completed: done.saturating_sub(err_n),
                    errors: err_n,
                });
            }
            ctx_thread.request_repaint();
        });

        self.auto_job = Some(AutoJob {
            receiver: rx,
            progress,
            cancel: Some(cancel),
            batch: true,
            title: "Auto Develop",
            applying_preview: None,
            ticker_stop: Arc::new(AtomicBool::new(false)),
        });
        self.status = format!("Auto Develop: {} images…", total);
        ctx.request_repaint();
    }

    fn apply_crop_result_to_path(&mut self, path: &Path, result: &AutoCropResult) -> bool {
        let Some(entry) = self.images.iter_mut().find(|e| e.path == path) else {
            return false;
        };
        entry.options.crop_rect = Some(result.rect);
        entry.options.crop_rect_reference_size = Some(result.reference_size);
        entry.options.apply_crop = true;
        true
    }

    fn start_batch_crop(&mut self, ctx: &egui::Context) {
        if self.heavy_job_running() || self.auto_job.is_some() {
            return;
        }
        if self.images.is_empty() {
            self.status = "Auto Crop: add images first.".to_string();
            return;
        }

        self.release_heavy_caches();

        let jobs: Vec<(PathBuf, PipelineOptions)> = self
            .images
            .iter()
            .map(|img| {
                let mut opts = img.options.clone();
                opts.flat_field_path = self.flat_field_path.clone();
                (img.path.clone(), opts)
            })
            .collect();
        let total = jobs.len();
        let cancel = Arc::new(AtomicBool::new(false));
        let completed = Arc::new(AtomicUsize::new(0));
        let progress = Arc::new(Mutex::new(AutoProgressState {
            fraction: 0.0,
            message: "Starting…".to_string(),
            log: Vec::new(),
            file_name: String::new(),
            completed: 0,
            total,
        }));
        let (tx, rx) = mpsc::channel();
        let workers = if total > 1 { 2 } else { 1 };
        let cancel_thread = cancel.clone();
        let progress_thread = progress.clone();
        let ctx_thread = ctx.clone();

        thread::spawn(move || {
            let queue = Arc::new(Mutex::new(jobs.into_iter().collect::<VecDeque<_>>()));
            let errors = Arc::new(AtomicUsize::new(0));

            thread::scope(|scope| {
                for _ in 0..workers {
                    let tx = tx.clone();
                    let ctx = ctx_thread.clone();
                    let cancel_thread = cancel_thread.clone();
                    let progress_thread = progress_thread.clone();
                    let completed = completed.clone();
                    let queue = queue.clone();
                    let errors = errors.clone();
                    scope.spawn(move || loop {
                        if cancel_thread.load(Ordering::Relaxed) {
                            return;
                        }
                        let Some((path, opts)) = queue.lock().ok().and_then(|mut q| q.pop_front())
                        else {
                            return;
                        };
                        let name = path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("image")
                            .to_string();
                        if let Ok(mut p) = progress_thread.lock() {
                            p.file_name = name.clone();
                            p.message = "Loading…".to_string();
                        }
                        ctx.request_repaint();

                        if cancel_thread.load(Ordering::Relaxed) {
                            return;
                        }

                        let done_at_start = completed.load(Ordering::Relaxed);
                        let mut on_progress = |message: &str, frac: f32, _log: Option<&str>| {
                            if let Ok(mut p) = progress_thread.lock() {
                                p.message = message.to_string();
                                p.file_name = name.clone();
                                p.completed = done_at_start;
                                let n = total.max(1) as f32;
                                p.fraction = (done_at_start as f32 + frac.clamp(0.0, 1.0)) / n;
                            }
                            ctx.request_repaint();
                        };

                        match run_auto_crop_for_path(&path, &opts, &mut on_progress) {
                            Ok(result) => {
                                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                                if let Ok(mut p) = progress_thread.lock() {
                                    p.completed = done;
                                    p.fraction = done as f32 / total.max(1) as f32;
                                }
                                let _ = tx.send(AutoJobOutcome::CropFileDone { path, result });
                            }
                            Err(_) => {
                                errors.fetch_add(1, Ordering::Relaxed);
                                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                                if let Ok(mut p) = progress_thread.lock() {
                                    p.completed = done;
                                    p.fraction = done as f32 / total.max(1) as f32;
                                    p.message = format!("Skipped {}", name);
                                }
                            }
                        }
                        ctx.request_repaint();
                    });
                }
            });

            let done = completed.load(Ordering::Relaxed);
            let err_n = errors.load(Ordering::Relaxed);
            if cancel_thread.load(Ordering::Relaxed) {
                let _ = tx.send(AutoJobOutcome::Cancelled { completed: done });
            } else {
                let _ = tx.send(AutoJobOutcome::BatchDone {
                    completed: done.saturating_sub(err_n),
                    errors: err_n,
                });
            }
            ctx_thread.request_repaint();
        });

        self.auto_job = Some(AutoJob {
            receiver: rx,
            progress,
            cancel: Some(cancel),
            batch: true,
            title: "Auto Crop",
            applying_preview: None,
            ticker_stop: Arc::new(AtomicBool::new(false)),
        });
        self.status = format!("Auto Crop: {} images…", total);
        ctx.request_repaint();
    }

    fn poll_auto_job(&mut self, ctx: &egui::Context) {
        loop {
            let waiting_preview = self
                .auto_job
                .as_ref()
                .is_some_and(|j| j.applying_preview.is_some());
            let recv = {
                let Some(job) = self.auto_job.as_ref() else {
                    return;
                };
                job.receiver.try_recv()
            };
            let outcome = match recv {
                Ok(outcome) => outcome,
                Err(mpsc::TryRecvError::Empty) => {
                    self.finish_auto_if_preview_ready(ctx);
                    ctx.request_repaint();
                    return;
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    if waiting_preview {
                        self.finish_auto_if_preview_ready(ctx);
                        return;
                    }
                    self.auto_job = None;
                    self.status = "Auto stopped unexpectedly.".to_string();
                    return;
                }
            };
            match outcome {
                AutoJobOutcome::Done { index, result } => {
                    if index < self.images.len() {
                        result.apply_to(&mut self.images[index].options);
                        self.images[index].preview_hash = 0;
                        self.images[index].preview_options_hash = 0;
                        let hardness = result.lut_in_mid - 1.0;
                        let density = 1.0 + result.curve_offset;
                        let grade = if result.curve_gamma > 0.0 {
                            result.curve_gamma / 2.5
                        } else {
                            1.0
                        };
                        self.status = format!(
                            "Auto: γ {:.2}, density {:.2}, grade {:.2}, toe {:+.2}, hardness {:+.2}",
                            result.film_gamma, density, grade, result.toe_strength, hardness
                        );
                        self.begin_auto_preview_wait(index, ctx);
                    } else {
                        self.auto_job = None;
                    }
                    return;
                }
                AutoJobOutcome::FileDone { path, result } => {
                    self.apply_auto_result_to_path(&path, &result);
                    ctx.request_repaint();
                }
                AutoJobOutcome::CropFileDone { path, result } => {
                    self.apply_crop_result_to_path(&path, &result);
                    ctx.request_repaint();
                }
                AutoJobOutcome::BatchDone { completed, errors } => {
                    let title = self.auto_job.as_ref().map(|j| j.title).unwrap_or("Batch");
                    self.auto_job = None;
                    self.preview_gen = self.preview_gen.wrapping_add(1);
                    self.status = if errors == 0 {
                        format!("{title}: finished {} image(s).", completed)
                    } else {
                        format!(
                            "{title}: finished {} image(s), {} failed.",
                            completed, errors
                        )
                    };
                    ctx.request_repaint();
                    return;
                }
                AutoJobOutcome::Cancelled { completed } => {
                    let title = self.auto_job.as_ref().map(|j| j.title).unwrap_or("Batch");
                    self.auto_job = None;
                    self.preview_gen = self.preview_gen.wrapping_add(1);
                    self.status = if completed == 0 {
                        format!("{title} cancelled.")
                    } else {
                        format!("{title} cancelled after {} image(s).", completed)
                    };
                    ctx.request_repaint();
                    return;
                }
                AutoJobOutcome::Error { message } => {
                    self.auto_job = None;
                    self.status = message;
                    return;
                }
            }
        }
    }

    fn show_auto_progress(&mut self, ctx: &egui::Context) {
        let Some(job) = self.auto_job.as_ref() else {
            return;
        };
        let batch = job.batch;
        let cancel = job.cancel.clone();
        let snap = job
            .progress
            .lock()
            .map(|g| g.clone())
            .unwrap_or_else(|e| e.into_inner().clone());
        let cancelling = cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed));

        let title = job.title;
        if job.applying_preview.is_some() {
            ctx.request_repaint();
        }
        let fraction = snap.fraction.clamp(0.0, 1.0);
        let percent_text = format!("{}%", (fraction * 100.0).round() as i32);
        let shown = if snap.completed < snap.total && !snap.file_name.is_empty() {
            snap.completed.saturating_add(1).min(snap.total)
        } else {
            snap.completed.min(snap.total)
        };
        let heading = if batch && snap.total > 1 {
            format!("{}  ({}/{})", snap.file_name, shown, snap.total)
        } else if batch && !snap.file_name.is_empty() {
            snap.file_name.clone()
        } else {
            snap.message.clone()
        };
        let stage = if cancelling {
            "Cancelling…".to_string()
        } else if batch {
            snap.message.clone()
        } else {
            String::new()
        };

        egui::Window::new(title)
            .collapsible(false)
            .resizable(false)
            .movable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.set_min_width(320.0);
                ui.add_space(4.0);
                ui.label(egui::RichText::new(&heading).strong());
                ui.add_space(6.0);
                ui.add(
                    egui::ProgressBar::new(fraction)
                        .desired_width(320.0)
                        .text(percent_text),
                );
                if batch {
                    if !stage.is_empty() {
                        ui.add_space(4.0);
                        ui.label(egui::RichText::new(&stage).small());
                    }
                    ui.add_space(10.0);
                    ui.horizontal(|ui| {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let btn = ui.add_enabled(!cancelling, egui::Button::new("Cancel"));
                            if btn.clicked() {
                                if let Some(c) = &cancel {
                                    c.store(true, Ordering::Relaxed);
                                }
                            }
                        });
                    });
                } else {
                    ui.add_space(8.0);
                    for line in &snap.log {
                        ui.label(
                            egui::RichText::new(line)
                                .small()
                                .color(egui::Color32::from_gray(170)),
                        );
                    }
                    ui.add_space(6.0);
                }
            });
    }
}

impl eframe::App for C41Gui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Re-apply each frame so backends that reset visuals stay on the lab theme.
        theme::apply(ctx);

        if ctx.input(|i| i.viewport().close_requested()) && !self.close_confirmed {
            if self.project_is_dirty() {
                ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                if self.pending_save_then_leave.is_none() {
                    self.request_leave_current_project(ctx, PendingLeaveAction::Quit);
                }
            }
        }

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

        if ctx.input_mut(|i| i.consume_shortcut(&PROJECT_SAVE_AS_SHORTCUT)) {
            self.request_project_save(true);
        } else if ctx.input_mut(|i| i.consume_shortcut(&PROJECT_SAVE_SHORTCUT)) {
            self.request_project_save(false);
        }
        if ctx.input_mut(|i| i.consume_shortcut(&PROJECT_LOAD_SHORTCUT)) {
            self.request_leave_current_project(ctx, PendingLeaveAction::LoadDialog);
        }
        if ctx.input_mut(|i| i.consume_shortcut(&REDO_SHORTCUT)) {
            self.redo_edits();
        } else if ctx.input_mut(|i| i.consume_shortcut(&UNDO_SHORTCUT)) {
            self.undo_edits();
        }

        self.rect_dragging = false;
        self.preview_canvas_pointer = false;

        if self.ui_icons.logo.is_none() {
            self.ui_icons.logo = load_icon_texture(ctx, "ui_icon_logo", ICON_LOGO_PATH);
        }

        // Deferred output-LUT file dialog (runs outside the egui panel render loop
        // to avoid macOS NSOpenPanel re-entrance crashes on repeated opens).
        if self.pending_project_save_as {
            self.pending_project_save_as = false;
            let saved = self.run_project_save_as_dialog();
            if let Some(action) = self.pending_save_then_leave.take() {
                if saved {
                    self.dispatch_leave_action(ctx, action);
                }
            }
        }
        if self.pending_project_load {
            self.pending_project_load = false;
            self.run_project_load_dialog();
        }
        if let Some(path) = self.pending_recent_load.take() {
            self.load_project_from_path(path);
        }

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
                                self.status =
                                    format!("Failed to parse output LUT {}: {}", path.display(), e);
                            }
                        },
                        None => {
                            self.status = "Output LUT: file dialog cancelled.".into();
                        }
                    }
                }
            }
        }

        let dropped_files = ctx.input_mut(|i| std::mem::take(&mut i.raw.dropped_files));
        if !dropped_files.is_empty() {
            let paths: Vec<PathBuf> = dropped_files.into_iter().filter_map(|f| f.path).collect();
            if !paths.is_empty() && self.add_image_paths(paths) == 0 {
                self.status = "No supported files in drop (RAW, PNG, JPEG, TIFF).".into();
            }
        }

        // Poll preview worker
        if let Some(rx) = self.preview_receiver.as_ref() {
            match rx.try_recv() {
                Ok(Ok(job)) => {
                    let was_live = self.preview_job_live;
                    self.preview_receiver = None;
                    self.preview_started_at = None;
                    self.preview_job_live = false;
                    if job.gen == self.preview_gen && job.index < self.images.len() {
                        let idx = job.index;
                        // Fit / first frame: LINEAR + mipmaps. Draw path recreates if pixel scale needs NEAREST.
                        let image = rgb_u8_to_color_image(job.w, job.h, &job.rgb);
                        let tex = ctx.load_texture(
                            format!("preview_full_{}", idx),
                            image,
                            preview_minify_texture_options(),
                        );
                        self.images[idx].preview_texture = Some(tex);
                        self.images[idx].preview_texture_nearest = false;
                        self.images[idx].preview_hash = job.options_hash;
                        self.images[idx].preview_options_hash = job.options_hash;
                        if self.auto_waiting_preview() == Some(idx) {
                            self.set_auto_progress("Done", 1.0);
                            self.auto_job = None;
                        }
                        self.images[idx].preview_lod = job.lod;
                        if job.lod == PreviewLod::Screen || job.lod == PreviewLod::FullRes {
                            self.images[idx].preview_screen_wh = (job.w, job.h);
                            self.images[idx].screen_step_cache = Some(job.new_cache.clone());
                        } else {
                            self.images[idx].draft_step_cache = Some(job.new_cache.clone());
                            let screen_wh = self.images[idx].preview_screen_wh;
                            if screen_wh == (0, 0) || screen_wh == (job.w, job.h) {
                                self.images[idx].screen_step_cache = Some(job.new_cache.clone());
                            }
                        }
                        self.images[idx].preview_full_rgb = Some((job.w, job.h, job.rgb.clone()));
                        self.images[idx].preview_step_cache = Some(job.new_cache);
                        self.images[idx].preview_input_size = Some([job.w, job.h]);
                        if job.input_w > 0 && job.input_h > 0 {
                            self.images[idx].raw_source_size = Some([job.input_w, job.input_h]);
                        }
                        self.images[idx].preview_texture_rotation =
                            self.images[idx].options.rotation_degrees;
                        self.images[idx].preview_texture_flip_h =
                            self.images[idx].options.flip_horizontal;
                        self.images[idx].preview_texture_flip_v =
                            self.images[idx].options.flip_vertical;
                        if self.images[idx].thumbnail_texture.is_none() {
                            if let Some(thumb_image) =
                                make_thumbnail_from_rgb(&job.rgb, job.w, job.h, THUMB_MAX_SIZE)
                            {
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
                        }
                        {
                            let hist = compute_histogram_from_rgb(
                                &job.rgb,
                                job.w,
                                job.h,
                                &self.images[idx].options,
                                job.input_w,
                                job.input_h,
                            );
                            let crop_h = crop_histogram_hash(&self.images[idx].options);
                            self.images[idx].histogram = Some(hist);
                            self.images[idx].histogram_crop_hash = crop_h;
                        }
                        if job.captured_debug {
                            self.images[idx].pipeline_debug_log = Some(job.dbg_log);
                        }
                        // Live apply already wrote the new backdrop + step cache.
                        // Drop pre-slider tiles so they cannot cover that result;
                        // tiling starts from the updated cache. A remosaic keeps
                        // same-hash tiles (screen refine must not wipe the cache).
                        if was_live {
                            self.images[idx].tile_cache.clear();
                            self.tile_gen = self.tile_gen.wrapping_add(1);
                            self.tile_inflight = None;
                            self.tile_failed.clear();
                        } else {
                            self.images[idx]
                                .tile_cache
                                .retain(|t| t.options_hash == job.options_hash);
                        }
                    }
                }
                Ok(Err(e)) => {
                    self.preview_receiver = None;
                    self.preview_started_at = None;
                    self.preview_job_live = false;
                    self.status = format!("Preview error: {}", e);
                    if self.auto_waiting_preview().is_some() {
                        self.auto_job = None;
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.preview_receiver = None;
                    self.preview_started_at = None;
                    self.preview_job_live = false;
                }
            }
        }

        if let Some(rx) = self.tile_receiver.as_ref() {
            match rx.try_recv() {
                Ok(Ok(job)) => {
                    self.tile_receiver = None;
                    self.tile_inflight = None;
                    if job.gen == self.tile_gen && job.index < self.images.len() {
                        let idx = job.index;
                        let (tw, th, rgb, tex_uv) =
                            crop_rgb_u8_to_uv(job.w, job.h, &job.rgb, job.tex_uv);
                        let image = rgb_u8_to_color_image(tw, th, &rgb);
                        let tex = ctx.load_texture(
                            format!("preview_tile_{}_{}_{}", idx, job.ix, job.iy),
                            image,
                            tile_texture_options(),
                        );
                        let tile = PreviewTile {
                            ix: job.ix,
                            iy: job.iy,
                            options_hash: job.options_hash,
                            texture: tex,
                            uv: job.uv,
                            tex_uv,
                        };
                        let cache = &mut self.images[idx].tile_cache;
                        cache.retain(|t| !(t.ix == tile.ix && t.iy == tile.iy));
                        cache.insert(0, tile);
                        self.evict_tile_cache(idx);
                        ctx.request_repaint();
                    }
                }
                Ok(Err(e)) => {
                    if let Some((_, ix, iy)) = self.tile_inflight.take() {
                        self.tile_failed.push((self.tile_gen, ix, iy));
                    }
                    self.tile_receiver = None;
                    self.status = format!("Tile error: {}", e);
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.tile_receiver = None;
                    self.tile_inflight = None;
                }
            }
        }

        self.poll_export_job(ctx);
        self.poll_auto_job(ctx);

        // Request thumbnail for one image at a time (strip icons).
        if !self.heavy_job_running() && self.thumbnail_receiver.is_none() {
            if let Some(entry) = self.images.iter().find(|e| {
                e.thumbnail_texture.is_none() && !self.thumbnail_pending.contains(&e.path)
            }) {
                let path = entry.path.clone();
                let mut options = entry.options.clone();
                options.flat_field_path = self.flat_field_path.clone();
                let (tx, rx) = mpsc::channel();
                self.thumbnail_receiver = Some(rx);
                self.thumbnail_pending.insert(path.clone());
                thread::spawn(move || {
                    let result =
                        process_one_to_preview(&path, &options, THUMB_MAX_SIZE, THUMB_MAX_SIZE)
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
                            format!(
                                "thumb_{}",
                                path.display()
                                    .to_string()
                                    .replace('\\', "_")
                                    .replace('/', "_")
                            ),
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

        // ---- Archive menu (macOS inset clears the hidden-title-bar traffic lights) ----
        egui::TopBottomPanel::top("menu_bar")
            .frame({
                let mut frame = egui::Frame::side_top_panel(&ctx.style());
                frame.inner_margin.top += 5.0;
                frame.inner_margin.bottom += 5.0;
                frame
            })
            .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                if cfg!(target_os = "macos") {
                    ui.add_space(MENU_BAR_MACOS_INSET);
                }
                egui::menu::menu_custom_button(
                    ui,
                    egui::Button::new(theme::icon_label(theme::ARCHIVE, "Archive")).frame(false),
                    |ui| {
                    ui.menu_button(theme::icon_label(theme::FOLDER, "Project"), |ui| {
                        if ui
                            .add(
                                egui::Button::new(theme::icon_label(theme::DOWNLOAD, "Save")).shortcut_text(
                                    ui.ctx().format_shortcut(&PROJECT_SAVE_SHORTCUT),
                                ),
                            )
                            .clicked()
                        {
                            self.request_project_save(false);
                            ui.close_menu();
                        }
                        if ui
                            .add(
                                egui::Button::new(theme::icon_label(theme::DOWNLOAD, "Save As...")).shortcut_text(
                                    ui.ctx().format_shortcut(&PROJECT_SAVE_AS_SHORTCUT),
                                ),
                            )
                            .clicked()
                        {
                            self.request_project_save(true);
                            ui.close_menu();
                        }
                        if ui
                            .add(
                                egui::Button::new(theme::icon_label(theme::FOLDER, "Load")).shortcut_text(
                                    ui.ctx().format_shortcut(&PROJECT_LOAD_SHORTCUT),
                                ),
                            )
                            .clicked()
                        {
                            self.request_leave_current_project(
                                ui.ctx(),
                                PendingLeaveAction::LoadDialog,
                            );
                            ui.close_menu();
                        }
                    });
                    ui.menu_button(theme::icon_label(theme::HISTORY, "Recent"), |ui| {
                        if self.recent_projects.is_empty() {
                            ui.add_enabled(false, egui::Button::new("No recent projects"));
                        } else {
                            let recent = self.recent_projects.clone();
                            for path in recent {
                                let label = recent_project_label(&path);
                                if ui
                                    .button(label)
                                    .on_hover_text(path.display().to_string())
                                    .clicked()
                                {
                                    self.request_leave_current_project(
                                        ui.ctx(),
                                        PendingLeaveAction::LoadRecent(path),
                                    );
                                    ui.close_menu();
                                }
                            }
                        }
                    });
                    ui.menu_button(theme::icon_label(theme::INVENTORY, "Batch"), |ui| {
                        let enabled = !self.images.is_empty()
                            && !self.heavy_job_running()
                            && self.auto_job.is_none();
                        if ui
                            .add_enabled(
                                enabled,
                                egui::Button::new(theme::icon_label(theme::AUTO_FIX, "Auto Develop")),
                            )
                            .on_hover_text(
                                "Run Auto on every loaded image (Film γ, density, grade, toe, hardness, saturation).",
                            )
                            .clicked()
                        {
                            self.start_batch_auto(ui.ctx());
                            ui.close_menu();
                        }
                        if ui
                            .add_enabled(
                                enabled,
                                egui::Button::new(theme::icon_label(theme::CROP, "Crop")),
                            )
                            .on_hover_text(
                                "Auto-crop every loaded image (detect the film frame, exclude holder and lightbox).",
                            )
                            .clicked()
                        {
                            self.start_batch_crop(ui.ctx());
                            ui.close_menu();
                        }
                        if ui
                            .add_enabled(
                                enabled,
                                egui::Button::new(theme::icon_label(theme::DOWNLOAD, "Export")),
                            )
                            .on_hover_text(
                                "Export every loaded image. Choose format and optional JPEG sidecar.",
                            )
                            .clicked()
                        {
                            self.open_batch_export_dialog();
                            ui.close_menu();
                        }
                    });
                },
                );
                egui::menu::menu_custom_button(
                    ui,
                    egui::Button::new(theme::icon_label(theme::EDIT, "Edit")).frame(false),
                    |ui| {
                    if ui
                        .add_enabled(
                            self.history.can_undo(),
                            egui::Button::new(theme::icon_label(theme::UNDO, "Undo")).shortcut_text(
                                ui.ctx().format_shortcut(&UNDO_SHORTCUT),
                            ),
                        )
                        .clicked()
                    {
                        self.undo_edits();
                        ui.close_menu();
                    }
                    if ui
                        .add_enabled(
                            self.history.can_redo(),
                            egui::Button::new(theme::icon_label(theme::REDO, "Redo")).shortcut_text(
                                ui.ctx().format_shortcut(&REDO_SHORTCUT),
                            ),
                        )
                        .clicked()
                    {
                        self.redo_edits();
                        ui.close_menu();
                    }
                },
                );
            });
        });

        // ---- Bottom panel: image strip + global output / convert ----
        egui::TopBottomPanel::bottom("bottom_panel")
            .min_height(BOTTOM_PANEL_HEIGHT)
            .show(ctx, |ui| {
                ui.vertical(|ui| {
                    ui.add_space(14.0);
                    ui.horizontal(|ui| {
                        if ui
                            .button(theme::icon_label(theme::ADD, "Add image…"))
                            .clicked()
                        {
                            if let Some(paths) = rfd::FileDialog::new()
                                .add_filter("RAW & images", IMPORT_EXTENSIONS)
                                .pick_files()
                            {
                                self.add_image_paths(paths);
                            }
                        }
                        ui.label(
                            egui::RichText::new("or drop files here")
                                .small()
                                .color(egui::Color32::from_gray(150)),
                        );
                    });

                    ui.add_space(10.0);

                    let mut to_remove = Vec::new();
                    let mut selection_changed = false;
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

                                let card_response =
                                    ui.allocate_ui(egui::vec2(CARD_WIDTH, CARD_HEIGHT), |ui| {
                                        let card_rect = ui.available_rect_before_wrap();
                                        let id = ui.make_persistent_id(("strip_card", i));
                                        let interact_resp =
                                            ui.interact(card_rect, id, egui::Sense::click());

                                        // Card background and border (drawn first so content is on top)
                                        let stroke = if selected {
                                            egui::Stroke::new(
                                                2.0,
                                                egui::Color32::from_rgb(100, 150, 255),
                                            )
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
                                            .allocate_new_ui(
                                                egui::UiBuilder::new().max_rect(x_rect),
                                                |ui| {
                                                    ui.add(
                                                        egui::Button::new(
                                                            egui::RichText::new(theme::CLOSE)
                                                                .size(14.0),
                                                        )
                                                        .frame(false),
                                                    )
                                                    .clicked()
                                                },
                                            )
                                            .inner;

                                        // Content area: thumbnail + name, clipped to card (below X row)
                                        let content_top =
                                            card_rect.top() + CARD_PADDING + X_BUTTON_SIZE + 2.0;
                                        let content_rect = egui::Rect::from_min_max(
                                            egui::pos2(
                                                card_rect.left() + CARD_PADDING,
                                                content_top,
                                            ),
                                            egui::pos2(
                                                card_rect.right() - CARD_PADDING,
                                                card_rect.bottom() - CARD_PADDING,
                                            ),
                                        );
                                        ui.allocate_new_ui(
                                            egui::UiBuilder::new().max_rect(content_rect),
                                            |ui| {
                                                ui.set_clip_rect(card_rect);
                                                ui.vertical_centered(|ui| {
                                                    if let Some(ref thumb) = entry.thumbnail_texture
                                                    {
                                                        let size = thumb.size();
                                                        let (w, h) =
                                                            (size[0] as f32, size[1] as f32);
                                                        let scale = (THUMB_SIZE / w)
                                                            .min(THUMB_SIZE / h)
                                                            .min(1.0);
                                                        ui.image((
                                                            thumb.id(),
                                                            egui::vec2(w * scale, h * scale),
                                                        ));
                                                    } else {
                                                        ui.allocate_space(egui::vec2(
                                                            THUMB_SIZE, THUMB_SIZE,
                                                        ));
                                                    }
                                                    ui.add_space(2.0);
                                                    ui.label(
                                                    egui::RichText::new(&display_name).small(),
                                                )
                                                .on_hover_text(entry.path.display().to_string());
                                                });
                                            },
                                        );

                                        (interact_resp, x_clicked)
                                    });

                                let (interact_resp, x_clicked) = card_response.inner;

                                if x_clicked {
                                    to_remove.push(i);
                                } else if interact_resp.clicked() {
                                    self.selected_index = Some(i);
                                    self.full_res_preview_active = false;
                                    selection_changed = true;
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
                        self.tile_receiver = None;
                        self.tile_inflight = None;
                        self.full_res_preview_active = false;
                        for &i in &to_remove {
                            if let Some(e) = self.images.get(i) {
                                self.thumbnail_pending.remove(&e.path);
                            }
                        }
                    }
                    for i in to_remove.into_iter().rev() {
                        if let Some(id) = self.images.get(i).map(|e| e.id) {
                            self.history.forget(&id);
                        }
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
                    if selection_changed || had_removals {
                        self.evict_unselected_heavy_caches();
                    }
                });
            });

        // ---- Right panel: mode toggle + per-image settings / calibration ----
        let mut auto_crop_requested = false;
        let mut auto_tune_requested = false;
        let mut arm_wb_picker = false;
        let mut disarm_wb_picker = false;
        let wb_picker_armed = self.wb_picker_armed;
        let available = ctx.available_rect();
        let panel_max_w = available
            .width()
            .min(RIGHT_PANEL_MAX_WIDTH)
            .max(RIGHT_PANEL_MIN_WIDTH);
        self.right_panel_width = self
            .right_panel_width
            .clamp(RIGHT_PANEL_MIN_WIDTH, panel_max_w);
        let resize_id = egui::Id::new("settings_panel").with("__resize");
        if let Some(resp) = ctx.read_response(resize_id) {
            if resp.dragged() {
                if let Some(pointer) = resp.interact_pointer_pos() {
                    self.right_panel_width =
                        (available.right() - pointer.x).clamp(RIGHT_PANEL_MIN_WIDTH, panel_max_w);
                }
            }
        }
        egui::SidePanel::right("settings_panel")
            .resizable(true)
            .exact_width(self.right_panel_width)
            .show(ctx, |ui| {
                // SidePanel persists the content min-rect. Lock to the allocated
                // width so sliders/paths/combos cannot snap the panel to max.
                ui.set_width(ui.available_width());
                ui.spacing_mut().item_spacing = egui::vec2(5.0, 8.0);
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                ui.add_space(16.0);
                ui.horizontal(|ui| {
                    ui.add_space(10.0);
                    ui.selectable_value(&mut self.mode, UIMode::Process, "Process");
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

                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        ui.add_space(14.0);
                        ui.vertical(|ui| {
                ui.set_width((ui.available_width() - 8.0).max(80.0));
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
                    ui.horizontal_wrapped(|ui| {
                        ui.selectable_value(
                            &mut entry.process_tab,
                            ProcessTab::Input,
                            theme::icon_label(theme::IMAGE, "Input"),
                        );
                        ui.selectable_value(
                            &mut entry.process_tab,
                            ProcessTab::Develop,
                            theme::icon_label(theme::TUNE, "Develop"),
                        );
                        ui.selectable_value(
                            &mut entry.process_tab,
                            ProcessTab::Dust,
                            theme::icon_label(theme::HEALING, "Dust"),
                        );
                        ui.selectable_value(
                            &mut entry.process_tab,
                            ProcessTab::Export,
                            theme::icon_label(theme::DOWNLOAD, "Export"),
                        );
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
                        ui.horizontal_wrapped(|ui| {
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
                    ui.horizontal_wrapped(|ui| {
                        if ui
                            .button("Capture pipeline log for current settings")
                            .on_hover_text("Snapshot current settings into the pipeline debug log")
                            .clicked()
                        {
                            self.capture_pipeline_debug_next = true;
                            entry.preview_hash = 0;
                            entry.preview_options_hash = 0;
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
                    ui.label(egui::RichText::new("Preset").strong());
                    ui.add_space(4.0);
                    ui.horizontal_wrapped(|ui| {
                        let auto_busy = self.auto_job.is_some();
                        if ui
                            .add_enabled(
                                !auto_busy,
                                egui::Button::new(theme::icon_label(theme::AUTO_FIX, "Auto")),
                            )
                            .on_hover_text(
                                "Set Film γ, density, grade, toe, hardness, and saturation from the histogram. Applies Lab 1.5 and max warmth.",
                            )
                            .clicked()
                        {
                            auto_tune_requested = true;
                        }
                        if ui
                            .button(theme::icon_label(theme::DOWNLOAD, "Export JSON…"))
                            .on_hover_text(
                                "Save current Develop settings (exposure, WB, color, zones, output, De-Bujack) as a JSON preset. Crop, D-min, and export format are not included.",
                            )
                            .clicked()
                        {
                            let base_dir = std::env::current_dir()
                                .unwrap_or_else(|_| PathBuf::from("."))
                                .join("presets");
                            let _ = std::fs::create_dir_all(&base_dir);
                            let default_name = entry
                                .path
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .map(|s| format!("{s}.json"))
                                .unwrap_or_else(|| "preset.json".to_string());
                            if let Some(path) = rfd::FileDialog::new()
                                .set_directory(&base_dir)
                                .add_filter("JSON preset", &["json"])
                                .set_file_name(&default_name)
                                .save_file()
                            {
                                match save_develop_preset(opts, &path) {
                                    Ok(()) => {
                                        self.status = format!(
                                            "Saved develop preset: {}",
                                            path.display()
                                        );
                                    }
                                    Err(e) => {
                                        self.status = format!("Failed to save preset: {e}");
                                    }
                                }
                            }
                        }
                        if ui
                            .button(theme::icon_label(theme::UPLOAD, "Import JSON…"))
                            .on_hover_text(
                                "Load Develop settings from a JSON preset onto this image. Crop, D-min, and export format stay as they are.",
                            )
                            .clicked()
                        {
                            let base_dir = std::env::current_dir()
                                .unwrap_or_else(|_| PathBuf::from("."))
                                .join("presets");
                            let mut dialog = rfd::FileDialog::new()
                                .add_filter("JSON preset", &["json"]);
                            if base_dir.is_dir() {
                                dialog = dialog.set_directory(&base_dir);
                            }
                            if let Some(path) = dialog.pick_file() {
                                match load_develop_preset(&path) {
                                    Ok(preset) => {
                                        preset.apply_to(opts);
                                        entry.preview_hash = 0;
                                        entry.preview_options_hash = 0;
                                        let label = if preset.name.trim().is_empty() {
                                            path.display().to_string()
                                        } else {
                                            preset.name
                                        };
                                        self.status = format!("Loaded develop preset: {label}");
                                    }
                                    Err(e) => {
                                        self.status = format!("Failed to load preset: {e}");
                                    }
                                }
                            }
                        }
                    });
                    ui.add_space(6.0);
                    ui.separator();
                    ui.add_space(4.0);

                    // ════════════════════════════════════════════════════════
                    // GROUP 1 — Exposure  (primary editing controls)
                    // ════════════════════════════════════════════════════════
                    ui.label(theme::icon_label(theme::EXPOSURE, "Exposure").strong());
                    ui.add_space(4.0);
                    {
                        let mut exp = exposure_from_opts(opts);
                        let mut changed = false;
                        changed |= theme::slider_row(ui, "Density", &mut exp.density, 0.5..=1.5, 2)
                            .changed();
                        changed |= theme::slider_row(ui, "Grade", &mut exp.grade, 0.5..=2.0, 2)
                            .changed();
                        changed |=
                            theme::slider_row(ui, "Shadows", &mut exp.shadows, -0.3..=0.3, 3)
                                .changed();
                        changed |= theme::slider_row(
                            ui,
                            "Highlights",
                            &mut exp.highlights,
                            -0.5..=0.5,
                            3,
                        )
                        .changed();
                        changed |=
                            theme::slider_row(ui, "Hardness", &mut exp.hardness, -0.5..=0.5, 3)
                                .changed();
                        if changed {
                            apply_exposure_to_opts(&exp, opts);
                        }

                        if matches!(opts.output_stage, OutputStage::FilmPrint) {
                            ui.add_space(4.0);
                            ui.label("Print balance (CMY)")
                                .on_hover_text("Per-channel cyan, magenta, yellow adjustments for print balance");
                            let mut pb = print_balance_from_opts(opts);
                            let mut pb_changed = false;
                            pb_changed |=
                                theme::slider_row(ui, "C", &mut pb.cyan, -0.5..=0.5, 2).changed();
                            pb_changed |=
                                theme::slider_row(ui, "M", &mut pb.magenta, -0.5..=0.5, 2)
                                    .changed();
                            pb_changed |=
                                theme::slider_row(ui, "Y", &mut pb.yellow, -0.5..=0.5, 2)
                                    .changed();
                            if pb_changed {
                                apply_print_balance_to_opts(&pb, opts);
                            }
                        }
                        if theme::section_reset(ui) {
                            apply_exposure_to_opts(
                                &ExposureParams {
                                    density: 1.0,
                                    grade: 1.0,
                                    shadows: 0.0,
                                    highlights: 0.0,
                                    hardness: 0.0,
                                },
                                opts,
                            );
                            apply_print_balance_to_opts(
                                &PrintBalance {
                                    cyan: 0.0,
                                    magenta: 0.0,
                                    yellow: 0.0,
                                },
                                opts,
                            );
                        }
                    }
                    ui.add_space(6.0);
                    ui.separator();

                    // ════════════════════════════════════════════════════════
                    // Highlight roll-off (Reinhard) — order matches workflow
                    // ════════════════════════════════════════════════════════
                    let cr_rolloff = ui.collapsing(
                        theme::icon_label(theme::FILTER_HDR, "Highlight roll-off"),
                        |ui| {
                        ui.add_space(4.0);
                        theme::slider_row(ui, "Strength", &mut opts.highlight_rolloff, 0.0..=3.0, 2);
                        theme::slider_row(
                            ui,
                            "Knee",
                            &mut opts.highlight_rolloff_d_mid,
                            0.5..=3.0,
                            2,
                        );
                        if theme::section_reset(ui) {
                            opts.highlight_rolloff = 0.0;
                            opts.highlight_rolloff_d_mid = 1.5;
                        }
                    });
                    cr_rolloff.header_response.on_hover_text("Reinhard-style compression in density space to mask noise in skies and dense negative areas.");

                    // ════════════════════════════════════════════════════════
                    // Tone shaping (advanced shadow/highlight)
                    // ════════════════════════════════════════════════════════
                    let cr_tone = ui.collapsing(
                        theme::icon_label(theme::CONTRAST, "Tone shaping"),
                        |ui| {
                        ui.add_space(4.0);
                        theme::slider_row(ui, "Toe", &mut opts.toe_strength, -0.5..=0.5, 2);
                        theme::slider_row(
                            ui,
                            "Shoulder",
                            &mut opts.shoulder_strength,
                            -0.5..=0.5,
                            2,
                        );
                        theme::slider_row(
                            ui,
                            "Shadow cast",
                            &mut opts.shadow_cast_strength,
                            0.0..=1.0,
                            2,
                        );
                        if theme::section_reset(ui) {
                            opts.toe_strength = 0.0;
                            opts.shoulder_strength = 0.0;
                            opts.shadow_cast_strength = 0.0;
                        }
                    });
                    cr_tone.header_response.on_hover_text("Toe/shoulder: softer shadows and highlights. Shadow cast: auto-neutralize color cast in shadows.");

                    // ════════════════════════════════════════════════════════
                    // White balance & color neutrality
                    // ════════════════════════════════════════════════════════
                    ui.collapsing(theme::icon_label(theme::WB_AUTO, "White balance"), |ui| {
                        let mut wb_mode = opts.wb_mode;
                        egui::ComboBox::from_id_salt(ui.id().with("wb_mode"))
                            .selected_text(match wb_mode {
                                WbMode::None => "None",
                                WbMode::Auto => "Auto",
                                WbMode::Picker => "Picker",
                                WbMode::Manual => "Manual",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut wb_mode, WbMode::None, "None");
                                ui.selectable_value(&mut wb_mode, WbMode::Auto, "Auto");
                                ui.selectable_value(&mut wb_mode, WbMode::Picker, "Picker");
                                ui.selectable_value(&mut wb_mode, WbMode::Manual, "Manual");
                            });
                        if wb_mode != opts.wb_mode {
                            opts.wb_mode = wb_mode;
                            if wb_mode == WbMode::Picker {
                                reset_wb_for_picker(opts);
                                arm_wb_picker = true;
                            } else {
                                sync_wb_flags_from_mode(opts);
                                disarm_wb_picker = true;
                            }
                        }

                        match opts.wb_mode {
                            WbMode::None => {}
                            WbMode::Auto => {
                                ui.label(
                                    egui::RichText::new("Per-channel median equalization.")
                                        .small()
                                        .color(egui::Color32::from_gray(160)),
                                );
                            }
                            WbMode::Picker => {
                                if wb_picker_armed {
                                    ui.add(
                                        egui::Label::new(
                                            egui::RichText::new(
                                                "Click the preview to sample a 4×4 neutral.",
                                            )
                                            .small()
                                            .color(egui::Color32::from_rgb(180, 220, 120)),
                                        )
                                        .selectable(false),
                                    );
                                } else if ui
                                    .button(theme::icon_label(theme::COLORIZE, "Pick whitepoint"))
                                    .clicked()
                                {
                                    reset_wb_for_picker(opts);
                                    arm_wb_picker = true;
                                }
                            }
                            WbMode::Manual => {
                                let mut k = opts.temp_k.unwrap_or(5500.0);
                                theme::slider_row_with(ui, "Temp", &mut k, 2500.0..=9000.0, |s| {
                                    s.suffix(" K")
                                });
                                opts.temp_k = Some(k);
                            }
                        }
                        if theme::section_reset(ui) {
                            opts.wb_mode = WbMode::Auto;
                            sync_wb_flags_from_mode(opts);
                            disarm_wb_picker = true;
                        }
                    });

                    // ════════════════════════════════════════════════════════
                    // Color character & separation
                    // ════════════════════════════════════════════════════════
                    let cr_color = ui.collapsing(theme::icon_label(theme::PALETTE, "Color"), |ui| {
                        ui.add_space(4.0);
                        theme::slider_row(ui, "Saturation", &mut opts.saturation, 0.7..=1.6, 2);
                        theme::slider_row(ui, "Warmth", &mut opts.highlight_warmth, 0.0..=0.6, 2);
                        ui.checkbox(&mut opts.apply_lab, "Lab separation");
                        ui.add_enabled_ui(opts.apply_lab, |ui| {
                            theme::slider_row(
                                ui,
                                "Separation",
                                &mut opts.lab_separation,
                                -2.0..=2.0,
                                2,
                            );
                        });
                        theme::slider_row(
                            ui,
                            "Skin magenta",
                            &mut opts.skin_magenta_shift,
                            0.0..=1.0,
                            2,
                        );
                        if theme::section_reset(ui) {
                            opts.saturation = 1.0;
                            opts.highlight_warmth = 0.0;
                            opts.apply_lab = true;
                            opts.lab_separation = 1.0;
                            opts.skin_magenta_shift = 0.0;
                        }
                    });
                    cr_color.header_response.on_hover_text("Saturation: density chroma. Warmth: golden highlights. Lab: mid-chroma separation in a/b. Skin magenta: rotates lips/eye magenta toward orange.");

                    // ════════════════════════════════════════════════════════
                    // Color zones (per-channel shadow/mid/highlight)
                    // ════════════════════════════════════════════════════════
                    let cr_zones = ui.collapsing(
                        theme::icon_label(theme::LAYERS, "Color zones"),
                        |ui| {
                        ui.add_space(4.0);

                        ui.label(egui::RichText::new("Shadows").strong());
                        theme::slider_row(ui, "Gain", &mut opts.zone_shadow_gain, -0.5..=0.5, 3);
                        theme::slider_row(
                            ui,
                            "Saturation",
                            &mut opts.zone_shadow_saturation,
                            0.5..=1.6,
                            2,
                        );
                        theme::slider_row(
                            ui,
                            "Gain R",
                            &mut opts.color_shadow_gain_r,
                            -0.3..=0.3,
                            3,
                        );
                        theme::slider_row(
                            ui,
                            "Gain G",
                            &mut opts.color_shadow_gain_g,
                            -0.3..=0.3,
                            3,
                        );
                        theme::slider_row(
                            ui,
                            "Gain B",
                            &mut opts.color_shadow_gain_b,
                            -0.3..=0.3,
                            3,
                        );

                        ui.add_space(4.0);
                        ui.label(egui::RichText::new("Midtones").strong());
                        theme::slider_row(ui, "Gain", &mut opts.zone_mid_gain, -0.5..=0.5, 3);
                        theme::slider_row(
                            ui,
                            "Saturation",
                            &mut opts.zone_mid_saturation,
                            0.5..=1.6,
                            2,
                        );
                        theme::slider_row(ui, "Gain R", &mut opts.color_mid_gain_r, -0.3..=0.3, 3);
                        theme::slider_row(ui, "Gain G", &mut opts.color_mid_gain_g, -0.3..=0.3, 3);
                        theme::slider_row(ui, "Gain B", &mut opts.color_mid_gain_b, -0.3..=0.3, 3);

                        ui.add_space(4.0);
                        ui.label(egui::RichText::new("Highlights").strong());
                        theme::slider_row(ui, "Gain", &mut opts.zone_highlight_gain, -0.5..=0.5, 3);
                        theme::slider_row(
                            ui,
                            "Saturation",
                            &mut opts.zone_highlight_saturation,
                            0.5..=1.6,
                            2,
                        );
                        theme::slider_row(
                            ui,
                            "Gain R",
                            &mut opts.color_highlight_gain_r,
                            -0.3..=0.3,
                            3,
                        );
                        theme::slider_row(
                            ui,
                            "Gain G",
                            &mut opts.color_highlight_gain_g,
                            -0.3..=0.3,
                            3,
                        );
                        theme::slider_row(
                            ui,
                            "Gain B",
                            &mut opts.color_highlight_gain_b,
                            -0.3..=0.3,
                            3,
                        );

                        if theme::section_reset(ui) {
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
                                if ui
                                    .button(theme::icon_label(theme::CROP, "Auto"))
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
                        theme::slider_row(ui, "Film γ", &mut opts.film_gamma, 0.4..=2.0, 2)
                            .label
                            .on_hover_text(
                                "C-41 γ ≈ 0.55–0.75. Converts density → scene log-exposure.",
                            );
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
                                theme::slider_row(
                                    ui,
                                    "Border buffer",
                                    &mut opts.auto_norm_buffer,
                                    0.1..=0.3,
                                    2,
                                )
                                .label
                                .on_hover_text("Automatic per-channel percentile normalization. Finds film base density (p0.5) and normalizes. Border buffer excludes edges from analysis.");
                            }
                        }

                        if opts.dmin_mode != DminMode::Off {
                            ui.add_space(4.0);
                            ui.separator();
                            ui.label("Flat-field override (luminance calibration)");
                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .button(theme::icon_label(theme::FOLDER, "Load flat-field map…"))
                                    .clicked()
                                {
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
                    let cr_bujack = ui.collapsing(theme::icon_label(theme::BLUR, "De-Bujack"), |ui| {
                        ui.checkbox(&mut opts.bujack_enabled, "De-Bujack")
                            .on_hover_text(
                                "Non-local OkLab compensation for diminishing returns. \
                                 After the output transform and looks, before encode. \
                                 Off by default.",
                            );
                        ui.add_enabled_ui(opts.bujack_enabled, |ui| {
                            ui.add_space(4.0);
                            theme::slider_row(ui, "Knee L", &mut opts.bujack_k_l, 0.05..=0.60, 3)
                                .label
                                .on_hover_text("Where lightness differences start to flatten. Smaller = more aggressive.");
                            theme::slider_row(ui, "Knee C", &mut opts.bujack_k_c, 0.05..=0.60, 3)
                                .label
                                .on_hover_text("Same knee, applied to the (a,b) chroma vector.");
                            theme::slider_row(ui, "Strength", &mut opts.bujack_strength, 0.0..=1.5, 2)
                                .label
                                .on_hover_text("Dry/wet mix. Above 1.0 over-corrects.");
                            theme::slider_row_with(
                                ui,
                                "Radius",
                                &mut opts.bujack_radius,
                                2.0..=48.0,
                                |s| s.fixed_decimals(0).suffix(" px"),
                            )
                            .label
                            .on_hover_text(
                                "Bilateral radius in pixels of the current buffer. \
                                 Small = across edges, large = across the frame. \
                                 Preview is smaller than export, so the same number \
                                 covers more of the frame in preview.",
                            );
                            theme::slider_row(ui, "Edge preserve", &mut opts.bujack_edge, 0.03..=1.0, 2)
                                .label
                                .on_hover_text("Bilateral range σ. Low keeps edges out of the base (less halo); 1.0 ≈ Gaussian.");
                        });
                        if theme::section_reset(ui) {
                            opts.bujack_enabled = false;
                            opts.bujack_k_l = 0.25;
                            opts.bujack_k_c = 0.30;
                            opts.bujack_strength = 0.2;
                            opts.bujack_radius = 16.0;
                            opts.bujack_edge = 0.25;
                        }
                    });
                    cr_bujack.header_response.on_hover_text(
                        "Stretches large OkLab differences from a local mean. \
                         Pointwise grades cannot undo diminishing returns.",
                    );

                    ui.collapsing(theme::icon_label(theme::MOVIE, "Output"), |ui| {
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
                                theme::slider_row(ui, "R", &mut opts.fp_offset_r, -0.3..=0.3, 3)
                                    .label
                                    .on_hover_text("Per-channel offsets (exposure shift)");
                                theme::slider_row(ui, "G", &mut opts.fp_offset_g, -0.3..=0.3, 3);
                                theme::slider_row(ui, "B", &mut opts.fp_offset_b, -0.3..=0.3, 3);

                                ui.add_space(4.0);
                                theme::slider_row(ui, "R", &mut opts.fp_gamma_r, 0.7..=1.5, 2)
                                    .label
                                    .on_hover_text("Per-channel gamma (contrast)");
                                theme::slider_row(ui, "G", &mut opts.fp_gamma_g, 0.7..=1.5, 2);
                                theme::slider_row(ui, "B", &mut opts.fp_gamma_b, 0.7..=1.5, 2);

                                ui.add_space(4.0);
                                theme::slider_row(
                                    ui,
                                    "Color bleed",
                                    &mut opts.fp_color_bleed,
                                    0.0..=0.3,
                                    2,
                                );
                                theme::slider_row(
                                    ui,
                                    "Vibrance",
                                    &mut opts.fp_vibrance,
                                    0.0..=1.0,
                                    2,
                                );
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
                                if ui
                                    .button(theme::icon_label(theme::FOLDER, "Browse output LUT…"))
                                    .clicked()
                                {
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

                        if theme::section_reset(ui) {
                            let d = PipelineOptions::default();
                            opts.no_curve = d.no_curve;
                            opts.output_stage = d.output_stage;
                            opts.curve_pivot = d.curve_pivot;
                            opts.fp_offset_r = d.fp_offset_r;
                            opts.fp_offset_g = d.fp_offset_g;
                            opts.fp_offset_b = d.fp_offset_b;
                            opts.fp_gamma_r = d.fp_gamma_r;
                            opts.fp_gamma_g = d.fp_gamma_g;
                            opts.fp_gamma_b = d.fp_gamma_b;
                            opts.fp_color_bleed = d.fp_color_bleed;
                            opts.fp_vibrance = d.fp_vibrance;
                            opts.output_lut_encoding = d.output_lut_encoding;
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
                        .on_hover_text("Set the profile name / film stock and notes, then create the color profile in one step (matrix + 3D LUT saved as .oxid).");
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
                                            .add_filter("Oxid profile", &[calibration::PROFILE_EXTENSION])
                                            .set_file_name(&(name.clone() + "." + calibration::PROFILE_EXTENSION))
                                            .save_file()
                                        {
                                            match calibration::save_c41_profile(&profile, &save_path) {
                                                Ok(()) => {
                                                    self.status = format!(
                                                        "Created .oxid profile (MSE {:.6}): {}",
                                                        mse,
                                                        save_path.display()
                                                    );
                                                }
                                                Err(e) => {
                                                    self.status = format!("Failed to save .oxid profile: {}", e);
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

                if self.mode == UIMode::Process && entry.process_tab == ProcessTab::Dust {
                    ui.add_space(12.0);
                    ui.separator();
                    ui.add_space(8.0);
                    ui.heading("Dust");
                    ui.add_space(8.0);

                    ui.label(egui::RichText::new("View").strong());
                    ui.horizontal(|ui| {
                        ui.selectable_value(
                            &mut entry.dust_view,
                            DustView::Disable,
                            theme::icon_label(theme::CANCEL, "Disable"),
                        )
                        .on_hover_text(format!(
                            "Disable heal ({})",
                            ui.ctx().format_shortcut(&DUST_DISABLE_SHORTCUT)
                        ));
                        ui.selectable_value(
                            &mut entry.dust_view,
                            DustView::Edit,
                            theme::icon_label(theme::EDIT, "Edit"),
                        )
                        .on_hover_text(format!(
                            "Edit ({})",
                            ui.ctx().format_shortcut(&DUST_EDIT_SHORTCUT)
                        ));
                        ui.selectable_value(
                            &mut entry.dust_view,
                            DustView::Process,
                            theme::icon_label(theme::AUTO_FIX, "Process"),
                        )
                        .on_hover_text(format!(
                            "Process ({})",
                            ui.ctx().format_shortcut(&DUST_PROCESS_SHORTCUT)
                        ));
                    });
                    ui.add_space(8.0);

                    ui.add_enabled_ui(entry.dust_view == DustView::Edit, |ui| {
                        ui.label(egui::RichText::new("Tool").strong());
                        ui.horizontal(|ui| {
                            let pen_resp = ui
                                .selectable_label(
                                    entry.dust_tool == Some(DustTool::Pen),
                                    "   Pen",
                                )
                                .on_hover_text("Pen (P)");
                            ui.painter().circle_stroke(
                                egui::pos2(pen_resp.rect.left() + 9.0, pen_resp.rect.center().y),
                                4.5,
                                egui::Stroke::new(1.25, egui::Color32::WHITE),
                            );
                            if pen_resp.clicked() {
                                entry.dust_tool = Some(DustTool::Pen);
                            }
                            let eraser_sel = entry.dust_tool == Some(DustTool::Eraser);
                            let eraser_resp = ui
                                .selectable_label(
                                    eraser_sel,
                                    egui::RichText::new("   Eraser").color(DUST_ERASER_RED),
                                )
                                .on_hover_text("Eraser (E)");
                            paint_eraser_icon(
                                ui.painter(),
                                egui::pos2(
                                    eraser_resp.rect.left() + 9.0,
                                    eraser_resp.rect.center().y,
                                ),
                                11.0,
                                DUST_ERASER_RED,
                            );
                            if eraser_resp.clicked() {
                                entry.dust_tool = Some(DustTool::Eraser);
                            }
                        });
                        ui.add_space(6.0);
                        theme::slider_row(
                            ui,
                            "Size",
                            &mut entry.dust_brush_radius,
                            1.0..=DUST_BRUSH_RADIUS_MAX,
                            0,
                        );
                    });

                    if ui.input(|i| i.key_pressed(egui::Key::OpenBracket)) {
                        entry.dust_brush_radius = (entry.dust_brush_radius - 1.0).max(1.0);
                    }
                    if ui.input(|i| i.key_pressed(egui::Key::CloseBracket)) {
                        entry.dust_brush_radius =
                            (entry.dust_brush_radius + 1.0).min(DUST_BRUSH_RADIUS_MAX);
                    }
                    if !ui.ctx().wants_keyboard_input() {
                        if ui.input_mut(|i| i.consume_shortcut(&DUST_DISABLE_SHORTCUT)) {
                            entry.dust_view = DustView::Disable;
                            self.dust_painting = false;
                            if self.dust_brush_resize.take().is_some() {
                                ui.ctx().send_viewport_cmd(egui::ViewportCommand::CursorGrab(
                                    egui::CursorGrab::None,
                                ));
                                ui.ctx()
                                    .send_viewport_cmd(egui::ViewportCommand::CursorVisible(true));
                            }
                        }
                        if ui.input_mut(|i| i.consume_shortcut(&DUST_PROCESS_SHORTCUT)) {
                            entry.dust_view = DustView::Process;
                            self.dust_painting = false;
                            if self.dust_brush_resize.take().is_some() {
                                ui.ctx().send_viewport_cmd(egui::ViewportCommand::CursorGrab(
                                    egui::CursorGrab::None,
                                ));
                                ui.ctx()
                                    .send_viewport_cmd(egui::ViewportCommand::CursorVisible(true));
                            }
                        }
                        if ui.input_mut(|i| i.consume_shortcut(&DUST_EDIT_SHORTCUT)) {
                            entry.dust_view = DustView::Edit;
                            self.dust_painting = false;
                            if self.dust_brush_resize.take().is_some() {
                                ui.ctx().send_viewport_cmd(egui::ViewportCommand::CursorGrab(
                                    egui::CursorGrab::None,
                                ));
                                ui.ctx()
                                    .send_viewport_cmd(egui::ViewportCommand::CursorVisible(true));
                            }
                        }
                        if ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape))
                        {
                            entry.dust_tool = None;
                            self.dust_painting = false;
                        } else if !self.dust_painting {
                            if ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::P)) {
                                entry.dust_tool = Some(DustTool::Pen);
                            }
                            if ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::E)) {
                                entry.dust_tool = Some(DustTool::Eraser);
                            }
                        }
                    }

                    ui.add_space(8.0);
                    ui.label(egui::RichText::new("Heal").strong());
                    egui::ComboBox::from_label("Infill")
                        .selected_text(entry.dust_infill.label())
                        .show_ui(ui, |ui| {
                            for value in [
                                DustInfill::PatchMatch,
                                DustInfill::WaveFunction,
                                DustInfill::Telea,
                            ] {
                                ui.selectable_value(
                                    &mut entry.dust_infill,
                                    value,
                                    value.label(),
                                );
                            }
                        });
                    theme::slider_row(ui, "Feather", &mut entry.dust_feather, 0.0..=12.0, 0);
                    theme::slider_row(ui, "Grain", &mut entry.dust_grain, 0.0..=3.0, 1);
                    match entry.dust_infill {
                        DustInfill::Telea => {
                            ui.small("Grain is the high-pass of film next to the stroke. 1.0 is a 1:1 copy.");
                        }
                        DustInfill::WaveFunction => {
                            let mut tile = entry.dust_tile as f32;
                            theme::slider_row(ui, "Tile", &mut tile, 2.0..=5.0, 0);
                            entry.dust_tile = tile.round().clamp(2.0, 5.0) as u8;
                            theme::slider_row(ui, "Match", &mut entry.dust_match, 1.0..=4.0, 1);
                            ui.small("Match is how strongly local direction is used for the structure fill.");
                            ui.small("Grain adds statistical film grain on the hole (NLF + clump spectrum). 1.0 matches measured σ.");
                        }
                        DustInfill::PatchMatch => {
                            theme::slider_row(ui, "Match", &mut entry.dust_match, 1.0..=4.0, 1);
                            ui.small("Match is how far and how loosely PatchMatch may search nearby film.");
                            ui.small("Copies a 7×7-matched patch from a color-gated collar.");
                            ui.small("Grain adds statistical film grain on the hole and feather (NLF + clump spectrum). 1.0 matches measured σ.");
                            ui.small("Feather fades the copied patch into film next to the stroke.");
                        }
                    }
                    ui.small("The pen is the hole. Size the brush to the speck; feather fades the rim.");
                    ui.add_space(8.0);
                    let has_mask = !entry.dust_strokes.is_empty();
                    if ui
                        .add_enabled(has_mask, egui::Button::new("Clear mask"))
                        .clicked()
                    {
                        entry.dust_strokes.clear();
                        entry.dust_mask.fill(0);
                        entry.dust_overlay_dirty = true;
                    }
                    if has_mask {
                        ui.small(format!("{} stroke(s)", entry.dust_strokes.len()));
                    } else {
                        ui.small("Paint over dust, then switch to Process.");
                    }
                    ui.small(format!(
                        "P pen · E eraser · Esc deselect · {} disable · {} edit · {} process · [ ] or Ctrl+drag size · Space pan.",
                        ui.ctx().format_shortcut(&DUST_DISABLE_SHORTCUT),
                        ui.ctx().format_shortcut(&DUST_EDIT_SHORTCUT),
                        ui.ctx().format_shortcut(&DUST_PROCESS_SHORTCUT),
                    ));
                }

                if self.mode == UIMode::Process && entry.process_tab == ProcessTab::Export {
                ui.add_space(12.0);
                ui.separator();
                ui.add_space(8.0);
                ui.heading("Export");
                ui.add_space(8.0);

                // Per-image export options
                export_format_combo(ui, &mut entry.export_format);
                apply_export_format_to_options(opts, entry.export_format);

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
                if ui
                    .button(theme::icon_label(theme::FOLDER, "Output folder…"))
                    .clicked()
                {
                    if let Some(path) = rfd::FileDialog::new().pick_folder() {
                        self.output_dir = Some(path);
                    }
                }
                ui.label(egui::RichText::new(out_label).small());

                let exporting = self.heavy_job_running();
                let ready = !self.images.is_empty() && self.output_dir.is_some() && !exporting;
                let selected_ready = self.selected_index.is_some()
                    && self.selected_index.unwrap() < self.images.len()
                    && self.output_dir.is_some()
                    && !exporting;
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .add_enabled(
                            ready,
                            egui::Button::new(theme::icon_label(theme::DOWNLOAD, "Convert all")),
                        )
                        .clicked()
                    {
                        self.start_export(ui.ctx(), false);
                    }
                    if ui
                        .add_enabled(
                            selected_ready,
                            egui::Button::new(theme::icon_label(theme::DOWNLOAD, "Export selected")),
                        )
                        .clicked()
                    {
                        self.start_export(ui.ctx(), true);
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

        if auto_tune_requested {
            self.start_auto(ctx);
        }
        if arm_wb_picker {
            self.wb_picker_armed = true;
        }
        if disarm_wb_picker {
            self.wb_picker_armed = false;
        }

        // ---- Auto-crop: detect on post–D-min linear T after sidebar borrow is released ----
        if auto_crop_requested {
            if let Some(idx) = self.selected_index {
                if idx < self.images.len() {
                    let after_step3 =
                        self.images[idx]
                            .preview_step_cache
                            .as_ref()
                            .and_then(|c| c.after_step3.as_ref().map(|(_, buf)| buf.clone()))
                            .or_else(|| {
                                self.images[idx].screen_step_cache.as_ref().and_then(|c| {
                                    c.after_step3.as_ref().map(|(_, buf)| buf.clone())
                                })
                            })
                            .or_else(|| {
                                self.images[idx].draft_step_cache.as_ref().and_then(|c| {
                                    c.after_step3.as_ref().map(|(_, buf)| buf.clone())
                                })
                            });
                    if let Some(buf) = after_step3 {
                        let dmin_rect = self.images[idx].options.dmin_rect;
                        let dmin_ref = self.images[idx].options.dmin_rect_reference_size;
                        match detect_crop(&buf, dmin_rect, dmin_ref) {
                            Some(result) => {
                                self.images[idx].options.crop_rect = Some(result.rect);
                                self.images[idx].options.crop_rect_reference_size =
                                    Some(result.reference_size);
                                self.images[idx].options.apply_crop = true;
                                if result.confidence == CropConfidence::Low {
                                    self.status = "Auto crop: weak edge — check crop.".to_string();
                                }
                            }
                            None => {
                                self.status =
                                    "Auto crop: no clear frame boundary found.".to_string();
                            }
                        }
                    } else {
                        self.status =
                            "Auto crop: waiting for preview to finish processing.".to_string();
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
                        const BOTTOM_PADDING: f32 = 8.0;
                        const IMAGE_PREVIEW_BOTTOM_PADDING: f32 = 16.0;
                        const TOP_PADDING: f32 = 17.0;
                        const INFO_ROW_HEIGHT: f32 = 18.0;

                        let other_reserved = IMAGE_PREVIEW_BOTTOM_PADDING
                            + INFO_ROW_HEIGHT
                            + CONTROL_ROW_HEIGHT
                            + BOTTOM_PADDING
                            + BOTTOM_PADDING
                            + TOP_PADDING;
                        let hist_max = (available.height() - other_reserved - 60.0)
                            .clamp(HISTOGRAM_MIN_HEIGHT, HISTOGRAM_MAX_HEIGHT);
                        self.histogram_height = self
                            .histogram_height
                            .clamp(HISTOGRAM_MIN_HEIGHT, hist_max);
                        let hist_h = self.histogram_height;

                        let reserved_bottom = IMAGE_PREVIEW_BOTTOM_PADDING
                            + INFO_ROW_HEIGHT
                            + CONTROL_ROW_HEIGHT
                            + BOTTOM_PADDING
                            + hist_h
                            + BOTTOM_PADDING;
                        // TOP_PADDING is allocated above the canvas; reserve it or the
                        // stored size is taller than the drawn rect and edge tiles miss.
                        let canvas_h =
                            (available.height() - reserved_bottom - TOP_PADDING).max(60.0);
                        let canvas_w = available.width();

                        // Extract image dims with fallback so the layout is stable before
                        // the first preview arrives (no jump when data loads in).
                        let _view_rot = preview_view_rotation(&self.images[idx]);
                        let desired_orient = preview_desired_orient(&self.images[idx]);
                        let baked_orient = preview_baked_orient(&self.images[idx]);
                        let geometry_pending = preview_view_geometry_pending(&self.images[idx]);
                        let (full_w, full_h) = if let Some((w, h, _)) = &self.images[idx].preview_full_rgb {
                            preview_display_wh(&self.images[idx], *w, *h)
                        } else {
                            self.images[idx].preview_input_size
                                .map(|s| (s[0], s[1]))
                                .unwrap_or((canvas_w as u32, (canvas_w * 2.0 / 3.0) as u32))
                        };
                        let full_w_f = (full_w as f32).max(1.0);
                        let full_h_f = (full_h as f32).max(1.0);

                        // Allocate the full canvas area — always, so the layout never jumps.
                        ui.add_space(TOP_PADDING);
                        let (canvas_rect, canvas_resp) = ui.allocate_exact_size(
                            egui::vec2(canvas_w, canvas_h),
                            egui::Sense::click_and_drag(),
                        );
                        let canvas_w = canvas_rect.width();
                        let canvas_h = canvas_rect.height();
                        let canvas_changed = self.preview_canvas_size.map(|(ow, oh)| {
                            (ow - canvas_w).abs() > 1.0 || (oh - canvas_h).abs() > 1.0
                        });
                        self.preview_canvas_size = Some((canvas_w, canvas_h));
                        if canvas_changed == Some(true) {
                            self.mark_preview_view_changed();
                        }
                        let canvas_painter = ui.painter_at(canvas_rect);
                        canvas_painter.rect_filled(canvas_rect, 0.0, egui::Color32::from_gray(30));

                        // Base scale: image size at zoom=1.0 to fit within canvas.
                        let base_scale = (canvas_w / full_w_f).min(canvas_h / full_h_f);

                        let zoom = self.images[idx].preview_zoom.max(1.0);
                        let pixel_scale = preview_pixel_scale(base_scale, zoom);
                        let img_w = full_w_f * pixel_scale;
                        let img_h = full_h_f * pixel_scale;

                        // Recreate texture only when LINEAR/NEAREST must flip (pixel scale crosses 1×).
                        let want_nearest = want_nearest_filter(pixel_scale);
                        if self.images[idx].preview_texture.is_some()
                            && self.images[idx].preview_texture_nearest != want_nearest
                        {
                            if let Some((fw, fh, rgb)) = self.images[idx].preview_full_rgb.as_ref() {
                                let image = rgb_u8_to_color_image(*fw, *fh, rgb);
                                let tex_opts = if want_nearest {
                                    egui::TextureOptions::NEAREST
                                } else {
                                    preview_minify_texture_options()
                                };
                                let tex = ui.ctx().load_texture(
                                    format!("preview_full_{}", idx),
                                    image,
                                    tex_opts,
                                );
                                self.images[idx].preview_texture = Some(tex);
                                self.images[idx].preview_texture_nearest = want_nearest;
                            }
                        }
                        let tex_opt = self.images[idx].preview_texture.clone();

                        // Pan: which image-normalized point sits at canvas center.
                        let pan_x = self.images[idx].preview_pan.x.clamp(0.0, 1.0);
                        let pan_y = self.images[idx].preview_pan.y.clamp(0.0, 1.0);

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
                            self.preview_canvas_pointer = canvas_resp.is_pointer_button_down_on()
                                || canvas_resp.dragged();
                            let grid_now = self.visible_tile_grid(idx);
                            let proxy_soft = grid_now.as_ref().map(|g| g.proxy_soft).unwrap_or(false);
                            let hide_tiles = !proxy_soft
                                || self.preview_options_dirty(idx)
                                || geometry_pending;
                            let opt_hash = self.images[idx].preview_options_hash;
                            let tiles_complete = !hide_tiles
                                && grid_now.as_ref().is_some_and(|g| {
                                    visible_priority_tiles_ready(
                                        &self.images[idx].tile_cache,
                                        g,
                                        opt_hash,
                                    )
                                });
                            // Once 1:1 tiles cover the view, skip the CFA proxy.
                            // It is more saturated; a 1 px gap on the top/right
                            // used to read as a chroma band after tiles finished.
                            if vis_rect.width() > 0.0 && vis_rect.height() > 0.0 && !tiles_complete {
                                let uv_l = (vis_rect.left()   - vir_rect.left()) / img_w;
                                let uv_t = (vis_rect.top()    - vir_rect.top())  / img_h;
                                let uv_r = (vis_rect.right()  - vir_rect.left()) / img_w;
                                let uv_b = (vis_rect.bottom() - vir_rect.top())  / img_h;
                                let display_uv = egui::Rect::from_min_max(
                                    egui::pos2(uv_l, uv_t),
                                    egui::pos2(uv_r, uv_b),
                                );
                                let uv = if let Some((tw, th, _)) = self.images[idx].preview_full_rgb {
                                    preview_crop_uv(&self.images[idx], tw, th).map(|c| {
                                        egui::Rect::from_min_max(
                                            egui::pos2(
                                                c.min.x + display_uv.min.x * (c.max.x - c.min.x),
                                                c.min.y + display_uv.min.y * (c.max.y - c.min.y),
                                            ),
                                            egui::pos2(
                                                c.min.x + display_uv.max.x * (c.max.x - c.min.x),
                                                c.min.y + display_uv.max.y * (c.max.y - c.min.y),
                                            ),
                                        )
                                    }).unwrap_or(display_uv)
                                } else {
                                    display_uv
                                };
                                paint_preview_image(
                                    &canvas_painter,
                                    tex.id(),
                                    vis_rect,
                                    uv,
                                    desired_orient,
                                    baked_orient,
                                );
                            }

                            // 1:1 tiles stay up during click / pan / zoom (UVs follow the view).
                            // Hide only when develop options or rotate/flip are stale.
                            if !hide_tiles {
                                let opt_hash = self.images[idx].preview_options_hash;
                                for tile in &self.images[idx].tile_cache {
                                    if tile.options_hash != opt_hash {
                                        continue;
                                    }
                                    // Draw by screen intersection, not grid.contains —
                                    // a tight grid used to hide real edge tiles after zoom.
                                    let tile_rect = tile_draw_rect(vir_rect, tile.uv, img_w, img_h);
                                    let tvis = tile_rect.intersect(canvas_rect);
                                    if tvis.width() <= 0.0 || tvis.height() <= 0.0 {
                                        continue;
                                    }
                                    let tw = tile_rect.width().max(1.0);
                                    let th = tile_rect.height().max(1.0);
                                    let fx0 = (tvis.left() - tile_rect.left()) / tw;
                                    let fy0 = (tvis.top() - tile_rect.top()) / th;
                                    let fx1 = (tvis.right() - tile_rect.left()) / tw;
                                    let fy1 = (tvis.bottom() - tile_rect.top()) / th;
                                    let tu = tile.tex_uv;
                                    let tuv = egui::Rect::from_min_max(
                                        egui::pos2(
                                            tu.min.x + fx0 * (tu.max.x - tu.min.x),
                                            tu.min.y + fy0 * (tu.max.y - tu.min.y),
                                        ),
                                        egui::pos2(
                                            tu.min.x + fx1 * (tu.max.x - tu.min.x),
                                            tu.min.y + fy1 * (tu.max.y - tu.min.y),
                                        ),
                                    );
                                    canvas_painter.image(
                                        tile.texture.id(),
                                        tvis,
                                        tuv,
                                        egui::Color32::WHITE,
                                    );
                                }
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

                        let dust_edit = self.mode == UIMode::Process
                            && self.images[idx].process_tab == ProcessTab::Dust
                            && self.images[idx].dust_view == DustView::Edit;
                        if dust_edit {
                            ensure_dust_working_mask(&mut self.images[idx], full_w, full_h);
                            let space_down = ui.input(|i| i.key_down(egui::Key::Space));
                            let ctrl_down = ui.input(|i| i.modifiers.ctrl);
                            let dust_tool = self.images[idx].dust_tool;
                            let primary_down = canvas_resp.is_pointer_button_down_on()
                                && ui.input(|i| i.pointer.primary_down())
                                && !ui.input(|i| i.pointer.middle_down())
                                && !space_down;
                            if dust_tool.is_some() && primary_down && ctrl_down {
                                if let Some(pos) = canvas_resp.interact_pointer_pos() {
                                    if let Some((start_r, _, accum)) = self.dust_brush_resize {
                                        let d = ui.input(|i| i.pointer.delta());
                                        let accum = accum + d.x - d.y;
                                        if let Some((_, _, acc)) = self.dust_brush_resize.as_mut() {
                                            *acc = accum;
                                        }
                                        self.images[idx].dust_brush_radius = (start_r
                                            + accum * 0.15)
                                            .clamp(1.0, DUST_BRUSH_RADIUS_MAX);
                                        ui.ctx().send_viewport_cmd(
                                            egui::ViewportCommand::CursorGrab(
                                                egui::CursorGrab::Locked,
                                            ),
                                        );
                                        ui.ctx().request_repaint();
                                    } else {
                                        self.begin_dust_brush_resize(
                                            ui.ctx(),
                                            self.images[idx].dust_brush_radius,
                                            pos,
                                        );
                                    }
                                }
                            } else if self.dust_brush_resize.is_some() && !primary_down {
                                self.end_dust_brush_resize(ui.ctx());
                            }
                            let pointer_down = dust_tool.is_some()
                                && primary_down
                                && !ctrl_down
                                && self.dust_brush_resize.is_none();
                            if pointer_down {
                                if let Some(pos) = canvas_resp.interact_pointer_pos() {
                                    if vir_rect.contains(pos) && canvas_rect.contains(pos) {
                                        let (ix, iy) = screen_to_image(pos.x, pos.y);
                                        let entry = &mut self.images[idx];
                                        let (src_w, src_h) =
                                            dust_source_wh(entry, full_w, full_h);
                                        let (rw, rh) = entry
                                            .dust_reference_size
                                            .unwrap_or((src_w as u32, src_h as u32));
                                        let sx = rw as f32 / full_w_f;
                                        let sy = rh as f32 / full_h_f;
                                        let pt = (ix * sx, iy * sy);
                                        let radius_ref = entry.dust_brush_radius
                                            * ((rw as f32 / src_w) + (rh as f32 / src_h))
                                            * 0.5;
                                        let prev_img = if !self.dust_painting {
                                            None
                                        } else {
                                            entry.dust_strokes.last().and_then(|s| {
                                                s.points.last().copied().map(|(px, py)| {
                                                    (px / sx, py / sy)
                                                })
                                            })
                                        };
                                        let tool = dust_tool.expect("pointer_down requires a tool");
                                        if !self.dust_painting {
                                            entry.dust_strokes.push(DustStroke {
                                                tool,
                                                radius: radius_ref,
                                                points: vec![pt],
                                            });
                                            self.dust_painting = true;
                                        } else if let Some(stroke) = entry.dust_strokes.last_mut()
                                        {
                                            if stroke
                                                .points
                                                .last()
                                                .map(|p| (p.0 - pt.0).hypot(p.1 - pt.1) > 0.25)
                                                .unwrap_or(true)
                                            {
                                                stroke.points.push(pt);
                                            }
                                        }
                                        let mut working = DustMask {
                                            width: entry.dust_mask_size.0,
                                            height: entry.dust_mask_size.1,
                                            data: std::mem::take(&mut entry.dust_mask),
                                        };
                                        let (src_w, _) = dust_source_wh(entry, full_w, full_h);
                                        let radius = entry.dust_brush_radius
                                            * (full_w_f / src_w.max(1.0));
                                        if let Some((ox, oy)) = prev_img {
                                            let dx = ix - ox;
                                            let dy = iy - oy;
                                            let dist = (dx * dx + dy * dy).sqrt();
                                            let steps = (dist * 2.0).ceil().max(1.0) as i32;
                                            for s in 1..=steps {
                                                let t = s as f32 / steps as f32;
                                                stamp_disc(
                                                    &mut working,
                                                    ox + dx * t,
                                                    oy + dy * t,
                                                    radius,
                                                    tool,
                                                );
                                            }
                                        } else {
                                            stamp_disc(&mut working, ix, iy, radius, tool);
                                        }
                                        entry.dust_mask = working.data;
                                        entry.dust_overlay_dirty = true;
                                    }
                                }
                            } else if self.dust_painting {
                                self.dust_painting = false;
                            }

                            let vis_rect = vir_rect.intersect(canvas_rect);
                            if vis_rect.width() > 0.0 && vis_rect.height() > 0.0 {
                                let entry = &mut self.images[idx];
                                if entry.dust_overlay_dirty
                                    || entry.dust_overlay_texture.is_none()
                                {
                                    if entry.dust_mask_size.0 > 0
                                        && entry.dust_mask.len()
                                            == entry.dust_mask_size.0 as usize
                                                * entry.dust_mask_size.1 as usize
                                    {
                                        let (mw, mh) = entry.dust_mask_size;
                                        let mut pixels =
                                            Vec::with_capacity(entry.dust_mask.len());
                                        for &c in &entry.dust_mask {
                                            let a = (c as u16 * 120 / 255) as u8;
                                            pixels.push(
                                                egui::Color32::from_rgba_unmultiplied(
                                                    220, 36, 36, a,
                                                ),
                                            );
                                        }
                                        let image = egui::ColorImage {
                                            size: [mw as usize, mh as usize],
                                            pixels,
                                        };
                                        entry.dust_overlay_texture = Some(
                                            ui.ctx().load_texture(
                                                format!("dust_overlay_{}", idx),
                                                image,
                                                egui::TextureOptions::LINEAR,
                                            ),
                                        );
                                        entry.dust_overlay_dirty = false;
                                    }
                                }
                                if let Some(tex) = entry.dust_overlay_texture.as_ref() {
                                    let uv_l = (vis_rect.left() - vir_rect.left()) / img_w;
                                    let uv_t = (vis_rect.top() - vir_rect.top()) / img_h;
                                    let uv_r = (vis_rect.right() - vir_rect.left()) / img_w;
                                    let uv_b = (vis_rect.bottom() - vir_rect.top()) / img_h;
                                    canvas_painter.image(
                                        tex.id(),
                                        vis_rect,
                                        egui::Rect::from_min_max(
                                            egui::pos2(uv_l, uv_t),
                                            egui::pos2(uv_r, uv_b),
                                        ),
                                        egui::Color32::WHITE,
                                    );
                                }
                            }

                            let hovering_canvas = canvas_resp.hovered();
                            let tool_armed = dust_tool.is_some();
                            if hovering_canvas || self.dust_brush_resize.is_some() {
                                if space_down && self.dust_brush_resize.is_none() {
                                    let grabbing = ui.input(|i| i.pointer.primary_down())
                                        || ui.input(|i| i.pointer.middle_down());
                                    ui.ctx().set_cursor_icon(if grabbing {
                                        egui::CursorIcon::Grabbing
                                    } else {
                                        egui::CursorIcon::Grab
                                    });
                                } else if tool_armed {
                                    ui.ctx().set_cursor_icon(egui::CursorIcon::None);
                                }
                            }
                            if !space_down && tool_armed {
                                let pos = self
                                    .dust_brush_resize
                                    .map(|(_, lock, _)| lock)
                                    .or_else(|| {
                                        canvas_resp.hover_pos().filter(|p| {
                                            vir_rect.contains(*p) && canvas_rect.contains(*p)
                                        })
                                    });
                                if let Some(pos) = pos {
                                    let (src_w, _) =
                                        dust_source_wh(&self.images[idx], full_w, full_h);
                                    let radius_screen = self.images[idx].dust_brush_radius
                                        * (img_w / src_w.max(1.0));
                                    let eraser = dust_tool == Some(DustTool::Eraser);
                                    let color = if eraser {
                                        DUST_ERASER_RED
                                    } else {
                                        egui::Color32::WHITE
                                    };
                                    canvas_painter.circle_stroke(
                                        pos,
                                        radius_screen.max(2.0),
                                        egui::Stroke::new(1.5, color),
                                    );
                                    if eraser {
                                        paint_eraser_icon(
                                            &canvas_painter,
                                            pos + egui::vec2(14.0, -16.0),
                                            15.0,
                                            DUST_ERASER_RED,
                                        );
                                    }
                                }
                            }
                        } else if self.dust_painting || self.dust_brush_resize.is_some() {
                            self.dust_painting = false;
                            self.end_dust_brush_resize(ui.ctx());
                        }

                        ui.add_space(IMAGE_PREVIEW_BOTTOM_PADDING);

                        // Loading spinner overlay — drawn entirely via canvas_painter so it
                        // never touches the UI layout cursor (which would shift the histogram).
                        // Also show when no preview data is available yet (first load).
                        if self.images[idx].preview_texture.is_none()
                            && (show_loader || has_inflight)
                            && self.auto_waiting_preview().is_none()
                        {
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
                        if canvas_resp.hovered()
                            && ui.input(|i| i.raw_scroll_delta.y != 0.0 && !i.modifiers.shift)
                        {
                            self.mark_preview_view_changed();
                        }

                        // Pan with left drag (when no rect handle hit) or middle drag.
                        let dust_edit = self.mode == UIMode::Process
                            && self.images[idx].process_tab == ProcessTab::Dust
                            && self.images[idx].dust_view == DustView::Edit;
                        let space_down = ui.input(|i| i.key_down(egui::Key::Space));
                        let middle_drag = ui.input(|i| i.pointer.middle_down()) && canvas_resp.dragged();
                        let space_drag = space_down
                            && canvas_resp.dragged()
                            && ui.input(|i| i.pointer.primary_down());
                        let left_drag = canvas_resp.dragged()
                            && !self.rect_dragging
                            && !self.dust_painting
                            && self.dust_brush_resize.is_none()
                            && (!dust_edit
                                || space_down
                                || self.images[idx].dust_tool.is_none());
                        if middle_drag || space_drag || left_drag {
                            let delta = canvas_resp.drag_delta();
                            {
                                let entry = &mut self.images[idx];
                                entry.preview_pan.x =
                                    (entry.preview_pan.x - delta.x / img_w).clamp(0.0, 1.0);
                                entry.preview_pan.y =
                                    (entry.preview_pan.y - delta.y / img_h).clamp(0.0, 1.0);
                            }
                            self.mark_preview_view_changed();
                        }

                        // White balance picker: 4×4 loupe + click to set gains.
                        let picker_armed = self.wb_picker_armed
                            && self.images[idx].options.wb_mode == WbMode::Picker;
                        if picker_armed {
                            if let Some(pos) = canvas_resp
                                .hover_pos()
                                .filter(|p| vir_rect.contains(*p) && canvas_rect.contains(*p))
                            {
                                let (px_f, py_f) = screen_to_image(pos.x, pos.y);
                                let px = (px_f as u32).min(full_w.saturating_sub(1));
                                let py = (py_f as u32).min(full_h.saturating_sub(1));
                                let [sr, sg, sb] = self.images[idx]
                                    .preview_full_rgb
                                    .as_ref()
                                    .map(|(w, h, rgb)| {
                                        let (tx, ty) = display_to_tex_px(
                                            px,
                                            py,
                                            *w,
                                            *h,
                                            desired_orient,
                                            baked_orient,
                                        );
                                        sample_rgb_u8_4x4(rgb, *w, *h, tx, ty)
                                    })
                                    .unwrap_or([128, 128, 128]);
                                let radius = 16.0;
                                let mut loupe = pos + egui::vec2(radius + 14.0, 0.0);
                                if loupe.x + radius > canvas_rect.right() {
                                    loupe.x = pos.x - radius - 14.0;
                                }
                                canvas_painter.circle_filled(
                                    loupe,
                                    radius,
                                    egui::Color32::from_rgb(sr, sg, sb),
                                );
                                canvas_painter.circle_stroke(
                                    loupe,
                                    radius,
                                    egui::Stroke::new(2.0, egui::Color32::from_gray(240)),
                                );
                                canvas_painter.circle_stroke(
                                    loupe,
                                    radius - 2.0,
                                    egui::Stroke::new(1.0, egui::Color32::from_gray(40)),
                                );
                            }

                            if canvas_resp.clicked() {
                                if let Some(pos) = canvas_resp.interact_pointer_pos() {
                                    let (px_f, py_f) = screen_to_image(pos.x, pos.y);
                                    if let Some(ref cache) = self.images[idx].preview_step_cache {
                                        if let Some((_, ref buf)) = cache.after_step3 {
                                            let (bh, bw, _) = buf.dim();
                                            let (tex_w, tex_h) = self.images[idx]
                                                .preview_full_rgb
                                                .as_ref()
                                                .map(|(w, h, _)| (*w, *h))
                                                .unwrap_or((full_w, full_h));
                                            let px = (px_f as u32).min(full_w.saturating_sub(1));
                                            let py = (py_f as u32).min(full_h.saturating_sub(1));
                                            let (tx, ty) = display_to_tex_px(
                                                px,
                                                py,
                                                tex_w,
                                                tex_h,
                                                desired_orient,
                                                baked_orient,
                                            );
                                            let x = ((tx as f32 + 0.5) / tex_w.max(1) as f32
                                                * bw as f32)
                                                .floor()
                                                .clamp(0.0, bw.saturating_sub(1) as f32)
                                                as usize;
                                            let y = ((ty as f32 + 0.5) / tex_h.max(1) as f32
                                                * bh as f32)
                                                .floor()
                                                .clamp(0.0, bh.saturating_sub(1) as f32)
                                                as usize;
                                            let (tr, tg, tb) = sample_array3_4x4(buf, x, y);
                                            let dr = -(tr.max(1e-10) as f64).log10() as f32;
                                            let dg = -(tg.max(1e-10) as f64).log10() as f32;
                                            let db = -(tb.max(1e-10) as f64).log10() as f32;
                                            let (wb_r, wb_g, wb_b) =
                                                color::density_to_wb_gains(dr, dg, db);
                                            let opts = &mut self.images[idx].options;
                                            opts.wb_r = wb_r;
                                            opts.wb_g = wb_g;
                                            opts.wb_b = wb_b;
                                            opts.apply_white_balance = true;
                                            opts.auto_wb = false;
                                            self.wb_picker_armed = false;
                                            self.status = format!(
                                                "WB set from 4×4 sample (R={:.3} G={:.3} B={:.3})",
                                                wb_r, wb_g, wb_b
                                            );
                                        } else {
                                            self.status =
                                                "WB picker: no cache (re-run preview first)."
                                                    .to_string();
                                        }
                                    } else {
                                        self.status =
                                            "WB picker: no cache (re-run preview first).".to_string();
                                    }
                                }
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
                        // Skip when apply_crop — the view is already the crop.
                        if self.mode == UIMode::Process
                            && !self.images.get(idx).is_some_and(|e| e.options.apply_crop)
                        {
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
                                                || entry.process_tab == ProcessTab::Dust
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
                            let zoom_pct = pixel_scale * 100.0;

                            let mut info_text = if let Some((cw, ch)) = crop_dims {
                                format!(
                                    "{} × {}  ->  {} × {}  ·  {:.0}%",
                                    src_w, src_h, cw, ch, zoom_pct
                                )
                            } else {
                                format!("{} × {}  ·  {:.0}%", src_w, src_h, zoom_pct)
                            };
                            let (scr_w, scr_h) = preview_working_limits(
                                self.preview_canvas_size,
                                ctx.pixels_per_point(),
                                false,
                            );
                            if self.images[idx].preview_lod == PreviewLod::Draft
                                && !self.full_res_preview_active
                                && zoom <= 1.0
                                && (has_inflight || scr_w > PREVIEW_DRAFT_MAX || scr_h > PREVIEW_DRAFT_MAX)
                            {
                                info_text.push_str("  ·  Refining…");
                            }
                            if proxy_is_soft(pixel_scale)
                                && !self.preview_options_dirty(idx)
                                && !self.preview_view_dragging
                                && !self.preview_view_settling()
                                && (self.tile_receiver.is_some()
                                    || !self.images[idx].tile_cache.is_empty())
                            {
                                let n = self.images[idx].tile_cache.len();
                                if self.tile_receiver.is_some() {
                                    info_text.push_str(&format!("  ·  1:1 tiles ({n})…"));
                                } else {
                                    info_text.push_str(&format!("  ·  1:1 tiles ({n})"));
                                }
                            }

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
                                .add(
                                    egui::Button::new(theme::icon_label(
                                        theme::HD,
                                        "Full resolution preview",
                                    ))
                                    .selected(self.full_res_preview_active),
                                )
                                .on_hover_text("Use full resolution export pipeline for preview. Deactivates when adjusting settings or switching images.")
                                .clicked();
                            if full_res_clicked {
                                self.full_res_preview_active = !self.full_res_preview_active;
                                if self.full_res_preview_active {
                                    self.full_res_preview_button_clicked = true;
                                    self.images[idx].preview_hash = 0;
                                    self.images[idx].preview_options_hash = 0;
                                    self.preview_gen = self.preview_gen.wrapping_add(1);
                                    self.tile_gen = self.tile_gen.wrapping_add(1);
                                    self.tile_inflight = None;
                                    self.tile_failed.clear();
                                    self.images[idx].tile_cache.clear();
                                }
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                // Order: rotate right, rotate left, mirror right, mirror left (right to left)
                                let rotate_right =
                                    theme::icon_button(ui, theme::ROTATE_RIGHT, "Rotate right");
                                let rotate_left =
                                    theme::icon_button(ui, theme::ROTATE_LEFT, "Rotate left");
                                // Count on press so a second click is not lost waiting for release
                                // while a preview job starts.
                                let mirror_v = theme::icon_button(
                                    ui,
                                    theme::FLIP_V,
                                    "Mirror right (flip vertical)",
                                );
                                let mirror_h = theme::icon_button(
                                    ui,
                                    theme::FLIP_H,
                                    "Mirror left (flip horizontal)",
                                );
                                // Count on press so a second click is not lost waiting for release
                                // while a preview job starts.
                                let pressed = ui.input(|i| i.pointer.primary_pressed());
                                if pressed && rotate_right.hovered() {
                                    self.apply_rotate_click(idx, true, ui.ctx());
                                } else if pressed && rotate_left.hovered() {
                                    self.apply_rotate_click(idx, false, ui.ctx());
                                } else if pressed && mirror_v.hovered() {
                                    self.apply_flip_click(idx, false, ui.ctx());
                                } else if pressed && mirror_h.hovered() {
                                    self.apply_flip_click(idx, true, ui.ctx());
                                }
                            });
                        });
                        ui.add_space(BOTTOM_PADDING);
                        if let Some((r_hist, g_hist, b_hist)) = self.images[idx].histogram {
                            let (hist_rect, _) = ui.allocate_exact_size(
                                egui::vec2(ui.available_width(), hist_h),
                                egui::Sense::hover(),
                            );
                            let resize_rect = egui::Rect::from_x_y_ranges(
                                hist_rect.x_range(),
                                (hist_rect.top() - 5.0)..=(hist_rect.top() + 5.0),
                            );
                            let resize_resp = ui.interact(
                                resize_rect,
                                ui.id().with("histogram_resize"),
                                egui::Sense::drag(),
                            );
                            if resize_resp.hovered() || resize_resp.dragged() {
                                ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
                            }
                            if resize_resp.dragged() {
                                self.histogram_height = (self.histogram_height
                                    - resize_resp.drag_delta().y)
                                    .clamp(HISTOGRAM_MIN_HEIGHT, hist_max);
                            }
                            let grip_color = if resize_resp.dragged() {
                                egui::Color32::from_rgb(110, 140, 170)
                            } else if resize_resp.hovered() {
                                egui::Color32::from_gray(160)
                            } else {
                                egui::Color32::from_gray(80)
                            };
                            let grip_w = 28.0;
                            ui.painter().hline(
                                (hist_rect.center().x - grip_w * 0.5)
                                    ..=(hist_rect.center().x + grip_w * 0.5),
                                hist_rect.top() + 2.0,
                                egui::Stroke::new(2.0, grip_color),
                            );
                            let painter = ui.painter_at(hist_rect);
                            let rect = hist_rect;
                            let draw_rect = rect.shrink(1.0);

                            let scale_at_full = histogram_y_scale(&r_hist, &g_hist, &b_hist);

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
                            // per-channel curve + fill. Shared robust peak scale.
                            let draw_channel =
                                |hist: &[u32; 256],
                                 line_color: egui::Color32,
                                 fill_color: egui::Color32,
                                 painter: &egui::Painter| {
                                    let mut curve_points = Vec::with_capacity(256);
                                    let mut clipped = [false; 256];
                                    let w = draw_rect.width().max(1.0);
                                    for i in 0..256 {
                                        let x = draw_rect.left()
                                            + (i as f32 / 255.0) * w;
                                        let raw = hist[i] as f32 / scale_at_full;
                                        clipped[i] = raw > 1.0;
                                        let h_norm = raw.min(1.0);
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

                                    // 1px cap on clipped bins so overflow is not a silent flat ceiling.
                                    let cap_y = draw_rect.top();
                                    let bin_w = (w / 255.0).max(1.0);
                                    let cap_stroke = egui::Stroke::new(1.0, line_color);
                                    for i in 0..256 {
                                        if !clipped[i] {
                                            continue;
                                        }
                                        let x = draw_rect.left() + (i as f32 / 255.0) * w;
                                        painter.line_segment(
                                            [
                                                egui::pos2(x - bin_w * 0.5, cap_y),
                                                egui::pos2(x + bin_w * 0.5, cap_y),
                                            ],
                                            cap_stroke,
                                        );
                                    }
                                };

                            draw_channel(
                                &r_hist,
                                egui::Color32::from_rgba_unmultiplied(220, 70, 70, 140),
                                egui::Color32::from_rgba_unmultiplied(200, 0, 0, 18),
                                &painter,
                            );
                            draw_channel(
                                &g_hist,
                                egui::Color32::from_rgba_unmultiplied(80, 200, 80, 140),
                                egui::Color32::from_rgba_unmultiplied(0, 200, 0, 18),
                                &painter,
                            );
                            draw_channel(
                                &b_hist,
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
                    ui.label("Drop RAW or image files here, or use Add image…");
                });
            }
        });

        // Crop is overlay-only: refresh the histogram from the current preview, no pipeline.
        if let Some(idx) = self.selected_index {
            if idx < self.images.len() {
                let crop_h = crop_histogram_hash(&self.images[idx].options);
                if self.images[idx].histogram_crop_hash != crop_h {
                    if let Some((w, h, rgb)) = self.images[idx].preview_full_rgb.clone() {
                        let input = self.images[idx].raw_source_size.unwrap_or([w, h]);
                        let hist = compute_histogram_from_rgb(
                            &rgb,
                            w,
                            h,
                            &self.images[idx].options,
                            input[0],
                            input[1],
                        );
                        self.images[idx].histogram = Some(hist);
                        self.images[idx].histogram_crop_hash = crop_h;
                    }
                }
            }
        }

        // Draft → screen refine. Runs after UI so drag/release is available for debounce.
        if !self.heavy_job_running() {
            if let Some(idx) = self.selected_index {
                if idx < self.images.len() {
                    let (screen_w, screen_h) = preview_working_limits(
                        self.preview_canvas_size,
                        ctx.pixels_per_point(),
                        self.full_res_preview_active,
                    );
                    let apply = self.dust_should_apply(&self.images[idx]);
                    {
                        let entry = &mut self.images[idx];
                        entry.options.dust_heal = entry_dust_heal(entry);
                        entry.options.dust_mask_hash = if apply {
                            hash_dust(&entry.dust_strokes, entry.options.dust_heal)
                        } else {
                            0
                        };
                        entry.options.dust_mask = None;
                    }
                    let hash_now = preview_options_hash(
                        &self.images[idx].path,
                        &self.images[idx].options,
                        self.full_res_preview_active,
                    );
                    if self.capture_pipeline_debug_next {
                        self.images[idx].preview_options_hash = 0;
                    }
                    let have_rgb = self.images[idx].preview_full_rgb.is_some();
                    let have_current =
                        have_rgb && self.images[idx].preview_options_hash == hash_now;
                    let need_options = !have_current;
                    let lod = self.images[idx].preview_lod;
                    let grid_now = self.visible_tile_grid(idx);
                    let proxy_soft = grid_now.as_ref().map(|g| g.proxy_soft).unwrap_or(false);
                    let tiles_fit = grid_now.as_ref().map(|g| g.tiles_fit).unwrap_or(false);
                    let wfc_dust = apply
                        && self.images[idx].dust_infill == DustInfill::WaveFunction;
                    let (req_sw, req_sh) = self.images[idx].preview_screen_requested_wh;
                    // Do not compare CFA output size to the request cap — downsample often
                    // comes in smaller and that used to restart screen refine forever,
                    // which blocked 1:1 tiles. Skip refine when tiles can cover the view;
                    // still refine the proxy when the visible grid exceeds PREVIEW_TILE_MAX.
                    let pointer_down = ctx.input(|i| i.pointer.any_down());
                    // Canvas press is not a slider. Live 50 ms debounce is only
                    // for develop widgets (and crop/d-min rects use rect_dragging).
                    let slider_dragging = pointer_down
                        && !self.rect_dragging
                        && !self.dust_painting
                        && !self.preview_view_dragging
                        && !self.preview_canvas_pointer;
                    let need_screen = have_current
                        && !self.full_res_preview_active
                        && (!tiles_fit || wfc_dust)
                        && !slider_dragging
                        && (lod == PreviewLod::Draft
                            || (lod == PreviewLod::Screen
                                && (req_sw + 64 < screen_w || req_sh + 64 < screen_h)));

                    if need_options {
                        let now = Instant::now();
                        let key = (idx, hash_now);
                        if self.pending_preview_key != Some(key) {
                            self.pending_preview_key = Some(key);
                            self.pending_preview_since = Some(now);
                            // Slider/options changed: drop the old tile cache now
                            // so release cannot flash pre-slider tiles over the
                            // live backdrop. New tiles start after the cache apply.
                            self.images[idx].tile_cache.clear();
                            self.tile_gen = self.tile_gen.wrapping_add(1);
                            self.tile_inflight = None;
                            self.tile_failed.clear();
                        }

                        let geometry_pending = self
                            .geometry_coalesce_until
                            .map(|t| Instant::now() < t)
                            .unwrap_or(false);
                        let can_live = !geometry_pending
                            && !self.rect_dragging
                            && !self.dust_painting
                            && self.live_preview_available(idx, ctx);
                        // Any pointer-down (including rotate/flip buttons) used to look like a
                        // slider drag and start a preview after 50ms — that blocked further clicks.
                        let debounce_ms = if geometry_pending {
                            GEOMETRY_COALESCE_MS
                        } else if slider_dragging {
                            PREVIEW_LIVE_DEBOUNCE_MS
                        } else {
                            PREVIEW_DEBOUNCE_MS
                        };
                        let auto_waiting = self.auto_waiting_preview() == Some(idx);
                        let settled = auto_waiting
                            || self
                                .pending_preview_since
                                .map(|t| {
                                    now.saturating_duration_since(t)
                                        >= Duration::from_millis(debounce_ms)
                                })
                                .unwrap_or(false);

                        if can_live && self.preview_receiver.is_some() && !self.preview_job_live {
                            // Drop a slow full remosaic so the live cache path can start.
                            self.preview_gen = self.preview_gen.wrapping_add(1);
                            self.preview_receiver = None;
                            self.preview_started_at = None;
                            self.preview_job_live = false;
                        }

                        if can_live && self.preview_receiver.is_none() {
                            if self.full_res_preview_active && !self.full_res_preview_button_clicked
                            {
                                self.full_res_preview_active = false;
                            }
                            self.request_live_preview_for(idx, ctx);
                            self.full_res_preview_button_clicked = false;
                            self.pending_preview_since = None;
                        } else if self.preview_receiver.is_none()
                            && !self.rect_dragging
                            && !self.dust_painting
                            && !geometry_pending
                            && !pointer_down
                            && settled
                        {
                            if self.full_res_preview_active && !self.full_res_preview_button_clicked
                            {
                                self.full_res_preview_active = false;
                            }
                            let lod = if self.full_res_preview_active && !slider_dragging {
                                PreviewLod::FullRes
                            } else if slider_dragging {
                                PreviewLod::Draft
                            } else {
                                // Fit / settled view: screen-res proxy (2× demosaic + filter).
                                PreviewLod::Screen
                            };
                            self.request_preview_for(idx, ctx, lod);
                            self.full_res_preview_button_clicked = false;
                            self.pending_preview_since = None;
                            self.geometry_coalesce_until = None;
                        } else {
                            ctx.request_repaint_after(Duration::from_millis(16));
                        }
                    } else if need_screen && self.preview_receiver.is_none() {
                        self.pending_preview_key = None;
                        self.pending_preview_since = None;
                        self.request_preview_for(idx, ctx, PreviewLod::Screen);
                    } else if self.pending_preview_key.map(|(i, _)| i) == Some(idx) {
                        self.pending_preview_key = None;
                        self.pending_preview_since = None;
                    }

                    if !pointer_down {
                        self.preview_view_dragging = false;
                    }
                    let view_settling = self.preview_view_settling();
                    // 1:1 tiles: after pan/zoom settle, at every zoom including fit.
                    // Wait for slider release so the live backdrop is committed first.
                    // Over-cap views still fetch a center-first window of PREVIEW_TILE_MAX.
                    if proxy_soft
                        && self.preview_receiver.is_none()
                        && self.tile_receiver.is_none()
                        && have_current
                        && !self.full_res_preview_active
                        && !self.rect_dragging
                        && !self.preview_view_dragging
                        && !view_settling
                        && !need_options
                        && !slider_dragging
                        && !(wfc_dust && lod == PreviewLod::Draft)
                    {
                        self.drop_tiles_outside_view(idx);
                        if let Some(missing) = self.visible_tile_to_request(idx) {
                            self.request_tile_for(idx, missing.0, missing.1, ctx);
                        }
                    }
                    if self.tile_receiver.is_some() {
                        ctx.request_repaint();
                    } else if view_settling {
                        if let Some(t) = self.preview_view_changed_at {
                            let left = PREVIEW_VIEW_SETTLE_MS
                                .saturating_sub(t.elapsed().as_millis() as u64)
                                .max(16);
                            ctx.request_repaint_after(Duration::from_millis(left));
                        }
                    }
                }
            } else {
                self.pending_preview_key = None;
                self.pending_preview_since = None;
            }
        }

        self.show_export_progress(ctx);
        self.show_auto_progress(ctx);
        self.show_batch_export_dialog(ctx);
        self.show_save_before_leave_dialog(ctx);
        self.commit_edit_history(ctx);

        if ctx.input(|i| !i.raw.hovered_files.is_empty()) {
            let screen = ctx.screen_rect();
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("file_drop_overlay"),
            ));
            painter.rect_filled(
                screen,
                0.0,
                egui::Color32::from_rgba_unmultiplied(20, 20, 20, 170),
            );
            let inset = screen.shrink(18.0);
            painter.rect_stroke(
                inset,
                8.0,
                egui::Stroke::new(2.0, egui::Color32::from_rgb(100, 150, 255)),
            );
            painter.text(
                screen.center(),
                egui::Align2::CENTER_CENTER,
                "Drop RAW or image files to add them",
                egui::FontId::proportional(20.0),
                egui::Color32::from_gray(240),
            );
        }
    }
}

#[cfg(test)]
mod tile_grid_tests {
    use super::{
        crop_rgb_u8_to_uv, egui, include_image_edge_tiles, preview_minify_texture_options,
        preview_pixel_scale, proxy_is_soft, tile_draw_rect, tile_range_intersecting,
        tile_texture_options, want_nearest_filter, VisibleTileGrid, PREVIEW_TILE_MAX,
        PREVIEW_TILE_SIZE,
    };

    #[test]
    fn exclusive_end_on_tile_boundary_does_not_drop_previous_tile() {
        let t = PREVIEW_TILE_SIZE as f32;
        let (ix0, _, ix1, _) = tile_range_intersecting(0.0, 0.0, t, t, t, 16, 16);
        assert_eq!((ix0, ix1), (0, 0));
    }

    #[test]
    fn sliver_past_tile_boundary_includes_new_edge_tile() {
        let t = PREVIEW_TILE_SIZE as f32;
        // Zoom-out exposes 0.1 sensor px of the next column — used to miss with floor(px-eps).
        let (ix0, _, ix1, _) = tile_range_intersecting(10.0, 0.0, t + 0.1, t, t, 16, 16);
        assert_eq!(ix0, 0);
        assert_eq!(ix1, 1);
    }

    #[test]
    fn floor_minus_eps_regression_still_covers_new_column() {
        let t = PREVIEW_TILE_SIZE as f32;
        // Exclusive UV just past a boundary, then minus 1e-3, used to snap back a tile.
        let px1 = t + 0.0005;
        let old_ix1 = ((px1 - 1e-3) / t).floor() as i32;
        assert_eq!(old_ix1, 0, "documents the old miss");
        let (_, _, ix1, _) = tile_range_intersecting(0.0, 0.0, px1, t, t, 16, 16);
        assert_eq!(ix1, 1);
    }

    #[test]
    fn tiles_on_at_fit_and_100_and_110_percent() {
        assert!(proxy_is_soft(0.25), "fit / zoom-out must request 1:1 tiles");
        assert!(proxy_is_soft(0.99));
        assert!(proxy_is_soft(1.0), "100% must request 1:1 tiles");
        assert!(proxy_is_soft(1.1), "110% must request 1:1 tiles");
        assert!(proxy_is_soft(preview_pixel_scale(0.25, 1.0)));
        assert!(proxy_is_soft(preview_pixel_scale(0.5, 2.0)));
    }

    #[test]
    fn nearest_only_when_magnifying_proxy() {
        assert!(!want_nearest_filter(0.5));
        assert!(!want_nearest_filter(1.0));
        assert!(want_nearest_filter(1.1));
    }

    #[test]
    fn crop_rgb_drops_halo_and_uses_full_tex_uv() {
        let w = 4u32;
        let h = 3u32;
        let mut rgb = Vec::new();
        for y in 0..h {
            for x in 0..w {
                rgb.extend_from_slice(&[x as u8, y as u8, 9]);
            }
        }
        let uv = egui::Rect::from_min_max(egui::pos2(0.25, 1.0 / 3.0), egui::pos2(0.75, 2.0 / 3.0));
        let (cw, ch, out, tex_uv) = crop_rgb_u8_to_uv(w, h, &rgb, uv);
        assert_eq!((cw, ch), (2, 1));
        assert_eq!(out, vec![1, 1, 9, 2, 1, 9]);
        assert_eq!(tex_uv.min, egui::pos2(0.0, 0.0));
        assert_eq!(tex_uv.max, egui::pos2(1.0, 1.0));
    }

    #[test]
    fn over_cap_grid_keeps_visible_silhouette() {
        let g = VisibleTileGrid {
            ix0: 0,
            iy0: 0,
            ix1: 20,
            iy1: 12,
            opt_hash: 0,
            proxy_soft: true,
            tiles_fit: false,
            core_n: PREVIEW_TILE_MAX + 40,
        };
        assert!(g.is_priority(10, 0), "top row must stay");
        assert!(g.is_priority(10, 12), "bottom row must stay");
        assert!(g.is_priority(0, 6), "left column must stay");
        assert!(g.is_priority(20, 6), "right column must stay");
        assert!(g.is_priority(10, 6), "center stays");
    }

    #[test]
    fn fit_view_always_includes_top_sensor_row() {
        // Simulates preview-UV mapping that skipped row 0 (iy0=1).
        let (ix0, iy0, ix1, iy1) =
            include_image_edge_tiles(0.0, 0.0, 1.0, 1.0, 12, 8, 0, 1, 11, 7);
        assert_eq!(iy0, 0, "image top must request sensor row 0");
        assert_eq!((ix0, ix1, iy1), (0, 11, 7));
    }

    #[test]
    fn tile_draw_rect_snaps_to_virtual_image_edges() {
        let vir = egui::Rect::from_min_max(egui::pos2(10.0, 20.0), egui::pos2(110.0, 80.0));
        let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        let r = tile_draw_rect(vir, uv, 100.0, 60.0);
        assert_eq!(r.min, vir.min);
        assert_eq!(r.max, vir.max);
    }

    #[test]
    fn minify_preview_and_tiles_use_mipmaps() {
        let preview = preview_minify_texture_options();
        assert_eq!(preview.minification, egui::TextureFilter::Linear);
        assert_eq!(preview.mipmap_mode, Some(egui::TextureFilter::Linear));
        let tile = tile_texture_options();
        assert_eq!(tile.magnification, egui::TextureFilter::Nearest);
        assert_eq!(tile.minification, egui::TextureFilter::Linear);
        assert_eq!(tile.mipmap_mode, Some(egui::TextureFilter::Linear));
    }
}
