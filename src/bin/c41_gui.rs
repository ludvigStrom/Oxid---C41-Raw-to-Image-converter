//! C-41 RAW Tool GUI: three-panel layout — center preview, right per-image settings, bottom image strip + global output/convert.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use c41_raw_tool::{
    blur_flat_field,
    calibration,
    demosaic,
    dmin,
    load_flat_field_linear,
    lut3d,
    png_reader,
    process_files,
    process_one_to_preview,
    raw_reader,
    tiff_export,
    PipelineOptions,
    Rect,
    TiffFormat,
};
use eframe::egui;

const PREVIEW_MAX_WIDTH: u32 = 1920;
const PREVIEW_MAX_HEIGHT: u32 = 1200;
const BOTTOM_PANEL_HEIGHT: f32 = 120.0;
const RIGHT_PANEL_WIDTH: f32 = 330.0;

fn main() -> eframe::Result<()> {
    eframe::run_native(
        "C-41 RAW Tool",
        eframe::NativeOptions::default(),
        Box::new(|_| Ok(Box::new(C41Gui::default()))),
    )
}

struct ImageEntry {
    path: PathBuf,
    options: PipelineOptions,
    preview_texture: Option<egui::TextureHandle>,
    preview_hash: u64,
    // Per-channel histograms (R, G, B) over 0–255
    histogram: Option<([u32; 256], [u32; 256], [u32; 256])>,
    export_format: ExportFormat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExportFormat {
    Tiff16,
    Tiff32,
    Dng,
    Exr,
    /// TIFF 16-bit display + linear ACES2065-1 EXR (only when Use ACEScg is on).
    ExrAces2065,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UIMode {
    Process,
    Calibrate,
    LuminanceCalibrate,
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
    preview_receiver: Option<mpsc::Receiver<anyhow::Result<(usize, u32, u32, Vec<u8>)>>>,
    preview_started_at: Option<Instant>,
    mode: UIMode,
    calibration_overlay: CalibrationOverlay,
    calibration_result: Option<([[f32; 3]; 3], f32)>, // (matrix, mse)
    calibration_profile_name: String,
    calibration_light_source: String,
    calibration_profiles: Vec<(PathBuf, calibration::CalibrationProfile)>,
    selected_profile_idx: Option<usize>,
    /// Luminance calibration: path and linearized flat-field image (RAW → demosaic only).
    flat_field_path: Option<PathBuf>,
    flat_field_image: Option<ndarray::Array3<f32>>,
    /// Camera IDT profiles loaded from camera_idt/ (path, profile).
    idt_profiles: Vec<(PathBuf, c41_raw_tool::aces::IdtProfile)>,
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
        // RAW formats handled by LibRaw; assume Bayer sensor and RGGB.
        "arw" | "nef" | "nrw" | "cr2" | "cr3" | "crw" | "dng" | "raf" | "orf" | "rw2" => {
            let bayer = raw_reader::load_raw_as_ndarray(path)?;
            demosaic::demosaic_bilinear(&bayer, demosaic::BayerPattern::Rggb)?
        }
        "png" => png_reader::load_png_as_ndarray(path)?,
        _ => anyhow::bail!("Unsupported extension for calibration"),
    };

    if let Some((r, g, b)) = opts.dmin_fixed {
        dmin::neutralize_with_medians(&mut image, r, g, b)?;
    } else if let Some(rect) = opts.dmin_rect {
        dmin::neutralize(&mut image, rect.x, rect.y, rect.width, rect.height)?;
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
        dmin_rect: None,
        dmin_fixed: Some((0.635294, 0.635294, 0.623529)),
        format: TiffFormat::Float32,
        write_exr: false,
        write_jpeg: false,
        no_invert: false,
        no_curve: false,
        wb_r: 1.15,
        wb_g: 0.88,
        wb_b: 1.0,
        curve_offset: 0.0,
        curve_gamma: 2.5,
        curve_pivot: 3.0,
        curve_white: 0.745,
        apply_color_profile: true,
        density_matrix: [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        flat_field_path: None,
        use_acescg: false,
        idt_matrix: c41_raw_tool::aces::IDT_IDENTITY,
        export_aces_exr: false,
        lut3d_path: None,
    }
}

fn options_hash_for(path: &PathBuf, opts: &PipelineOptions) -> u64 {
    let mut h = DefaultHasher::new();
    path.display().to_string().hash(&mut h);
    opts.apply_dmin.hash(&mut h);
    opts.apply_white_balance.hash(&mut h);
    opts.apply_color_profile.hash(&mut h);
    opts.dmin_rect.hash(&mut h);
    if let Some((r, g, b)) = opts.dmin_fixed {
        r.to_bits().hash(&mut h);
        g.to_bits().hash(&mut h);
        b.to_bits().hash(&mut h);
    }
    (opts.wb_r.to_bits(), opts.wb_g.to_bits(), opts.wb_b.to_bits()).hash(&mut h);
    opts.no_curve.hash(&mut h);
    opts.no_invert.hash(&mut h);
    opts.curve_offset.to_bits().hash(&mut h);
    opts.curve_gamma.to_bits().hash(&mut h);
    opts.curve_pivot.to_bits().hash(&mut h);
    opts.curve_white.to_bits().hash(&mut h);
    (opts.format as u8).hash(&mut h);
    opts.write_exr.hash(&mut h);
    for row in &opts.density_matrix {
        for v in row {
            v.to_bits().hash(&mut h);
        }
    }
    opts.flat_field_path.as_ref().map(|p| p.display().to_string()).hash(&mut h);
    opts.use_acescg.hash(&mut h);
    opts.export_aces_exr.hash(&mut h);
    for row in &opts.idt_matrix {
        for v in row {
            v.to_bits().hash(&mut h);
        }
    }
    opts.lut3d_path.as_ref().map(|p| p.display().to_string()).hash(&mut h);
    h.finish()
}

impl C41Gui {
    fn request_preview_for(&mut self, index: usize, ctx: &egui::Context) {
        if index >= self.images.len() {
            return;
        }
        let path = self.images[index].path.clone();
        let mut options = self.images[index].options.clone();
        options.flat_field_path = self.flat_field_path.clone();
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
            .map(|(w, h, rgb)| (index, w, h, rgb));
            let _ = tx.send(res);
        });
        ctx.request_repaint();
    }
}

impl eframe::App for C41Gui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll preview worker
        if let Some(rx) = self.preview_receiver.as_ref() {
            match rx.try_recv() {
                Ok(Ok((idx, w, h, rgb))) => {
                    self.preview_receiver = None;
                    self.preview_started_at = None;
                    if idx < self.images.len() {
                        let size = [w as usize, h as usize];
                        let mut r_hist = [0u32; 256];
                        let mut g_hist = [0u32; 256];
                        let mut b_hist = [0u32; 256];
                        let pixels: Vec<egui::Color32> = rgb
                            .chunks_exact(3)
                            .map(|c| {
                                let r = c[0] as usize;
                                let g = c[1] as usize;
                                let b = c[2] as usize;
                                r_hist[r] += 1;
                                g_hist[g] += 1;
                                b_hist[b] += 1;
                                egui::Color32::from_rgb(c[0], c[1], c[2])
                            })
                            .collect();
                        let image = egui::ColorImage { size, pixels };
                        let tex = ctx.load_texture(
                            format!("preview_{}", idx),
                            image,
                            egui::TextureOptions::default(),
                        );
                        let hash = options_hash_for(&self.images[idx].path, &self.images[idx].options);
                        self.images[idx].preview_texture = Some(tex);
                        self.images[idx].preview_hash = hash;
                        self.images[idx].histogram = Some((r_hist, g_hist, b_hist));
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

        // If selected image has no preview or options changed, request a new one.
        if let Some(idx) = self.selected_index {
            if idx < self.images.len() && self.preview_receiver.is_none() {
                let hash_now = options_hash_for(&self.images[idx].path, &self.images[idx].options);
                let need_new = self.images[idx].preview_texture.is_none()
                    || self.images[idx].preview_hash != hash_now;
                if need_new {
                    self.request_preview_for(idx, ctx);
                }
            }
        }

        // ---- Bottom panel: image strip + global output / convert ----
        egui::TopBottomPanel::bottom("bottom_panel")
            .min_height(BOTTOM_PANEL_HEIGHT)
            .show(ctx, |ui| {
                ui.vertical(|ui| {
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
                                            histogram: None,
                                            export_format: ExportFormat::Tiff16,
                                        });
                                        if self.selected_index.is_none() {
                                            self.selected_index = Some(self.images.len() - 1);
                                        }
                                    }
                                }
                                self.status = format!("{} file(s)", self.images.len());
                            }
                        }

                        ui.separator();

                    });

                    ui.add_space(4.0);

                    egui::ScrollArea::horizontal().show(ui, |ui| {
                        let mut to_remove = Vec::new();
                        ui.horizontal(|ui| {
                            for (i, entry) in self.images.iter().enumerate() {
                                let name = entry
                                    .path
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or("?");
                                let selected = self.selected_index == Some(i);
                                let resp = ui
                                    .selectable_label(selected, name)
                                    .on_hover_text(entry.path.display().to_string());
                                if resp.clicked() {
                                    self.selected_index = Some(i);
                                }
                                if ui.small_button("✕").clicked() {
                                    to_remove.push(i);
                                }
                            }
                        });
                        if !to_remove.is_empty() {
                            self.preview_receiver = None;
                        }
                        for i in to_remove.into_iter().rev() {
                            self.images.remove(i);
                            if self.selected_index == Some(i) {
                                self.selected_index = None;
                            } else if self.selected_index.map(|s| s > i).unwrap_or(false) {
                                self.selected_index = self.selected_index.map(|s| s - 1);
                            }
                        }
                    });
                });
            });

        // ---- Right panel: mode toggle + per-image settings / calibration ----
        egui::SidePanel::right("settings_panel")
            .resizable(false)
            .exact_width(RIGHT_PANEL_WIDTH)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.mode, UIMode::Process, "Process");
                    ui.selectable_value(&mut self.mode, UIMode::Calibrate, "Color calibration");
                    ui.selectable_value(
                        &mut self.mode,
                        UIMode::LuminanceCalibrate,
                        "Luminance calibration",
                    );
                });
                ui.separator();
                ui.add_space(4.0);

                match self.mode {
                    UIMode::Process => {
                        ui.heading("Image Settings");
                    }
                    UIMode::Calibrate => {
                        ui.heading("Color calibration");
                    }
                    UIMode::LuminanceCalibrate => {
                        ui.heading("Luminance calibration");
                    }
                }
                ui.add_space(4.0);

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
                ui.add_space(4.0);

                // D-min, White balance, and Print curve apply only to normal processing.
                // In Luminance calibration we only load a reference frame (raw → demosaic → blur); no conversion settings.
                if self.mode != UIMode::LuminanceCalibrate {
                    ui.checkbox(&mut opts.apply_dmin, "D-min");
                    if opts.apply_dmin {
                    ui.collapsing("D-min settings", |ui| {
                        // Option 1: classic D-min (fixed or crop) when no flat-field override is set.
                        let mut use_fixed = opts.dmin_fixed.is_some();
                        ui.checkbox(&mut use_fixed, "Use fixed D-min (R,G,B)");
                        if use_fixed {
                            if opts.dmin_fixed.is_none() {
                                opts.dmin_fixed = Some((0.635294, 0.635294, 0.623529));
                            }
                            let (mut r, mut g, mut b) = opts.dmin_fixed.unwrap();
                            ui.horizontal(|ui| {
                                ui.label("R");
                                ui.add(egui::DragValue::new(&mut r).range(0.0..=1.0).speed(0.01));
                                ui.label("G");
                                ui.add(egui::DragValue::new(&mut g).range(0.0..=1.0).speed(0.01));
                                ui.label("B");
                                ui.add(egui::DragValue::new(&mut b).range(0.0..=1.0).speed(0.01));
                            });
                            opts.dmin_fixed = Some((r, g, b));
                            opts.dmin_rect = None;
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
                            ui.label(
                                egui::RichText::new(format!("Flat-field: {}", p.display())).small(),
                            );
                        } else {
                            ui.label(egui::RichText::new("No flat-field override set.").small());
                        }
                    });
                    }

                    ui.checkbox(&mut opts.apply_white_balance, "White balance");
                    if opts.apply_white_balance {
                    ui.collapsing("White balance settings", |ui| {
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

                    // ACES: run pipeline in ACEScg. Export ACES2065-1 is chosen in the Export dropdown.
                    let use_acescg_prev = opts.use_acescg;
                    ui.checkbox(&mut opts.use_acescg, "Use ACEScg");
                    if use_acescg_prev && !opts.use_acescg {
                        opts.idt_matrix = c41_raw_tool::aces::IDT_IDENTITY;
                    }
                    if opts.use_acescg {
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
                                // Refresh list when dropdown is open
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
                                        ui.add(egui::DragValue::new(&mut m[row][col]).speed(0.05));
                                    }
                                }
                            });
                        });
                    }

                    let mut apply_curve = !opts.no_curve;
                    ui.checkbox(&mut apply_curve, "Print curve");
                    opts.no_curve = !apply_curve;
                    if apply_curve {
                    ui.collapsing("Print curve settings", |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Offset");
                            ui.add(
                                egui::DragValue::new(&mut opts.curve_offset)
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
                                egui::DragValue::new(&mut opts.curve_pivot)
                                    .range(0.1..=10.0)
                                    .speed(0.1),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("White");
                            ui.add(egui::Slider::new(&mut opts.curve_white, 0.3..=1.0));
                        });
                    });
                    }
                }

                if self.mode == UIMode::Calibrate {
                    ui.separator();
                    ui.heading("Solve color calibration");
                    ui.add_space(4.0);

                    if ui.button("Solve 3×3 matrix from chart").clicked() {
                        let path = entry.path.clone();
                        let opts_clone = calibration_opts_snapshot.clone();
                        match load_linear_transmittance_for_calibration(&path, &opts_clone) {
                            Ok(image_lin) => {
                                // Step 3.2: sample 24 patches (medians) from linear transmittance.
                                let centers_norm =
                                    compute_patch_centers_normalized(self.calibration_overlay.corners);
                                let patches_linear =
                                    sample_patch_medians(&image_lin, &centers_norm, 5.0);

                                // Step 3.3: convert to density.
                                let measured_density =
                                    calibration::linear_to_density_24(patches_linear);
                                let reference_density =
                                    calibration::reference_density_24();

                                // Phase 4: OLS solver.
                                match calibration::solve_density_matrix_ols(
                                    measured_density,
                                    reference_density,
                                ) {
                                    Some((m, mse)) => {
                                        self.calibration_result = Some((m, mse));
                                        opts.density_matrix = m;
                                        self.status = format!(
                                            "Solved color calibration matrix (MSE {:.6}) applied to this image.",
                                            mse
                                        );
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
                        ui.add_space(4.0);
                        ui.label(format!("MSE: {:.6}", mse));
                        ui.monospace(format!(
                            "Matrix:\n[{:.6}, {:.6}, {:.6}]\n[{:.6}, {:.6}, {:.6}]\n[{:.6}, {:.6}, {:.6}]",
                            m[0][0], m[0][1], m[0][2],
                            m[1][0], m[1][1], m[1][2],
                            m[2][0], m[2][1], m[2][2],
                        ));

                        ui.add_space(4.0);
                        ui.label("Profile name / film stock");
                        if self.calibration_profile_name.is_empty() {
                            if let Some(stem) = entry
                                .path
                                .file_stem()
                                .and_then(|s| s.to_str())
                            {
                                self.calibration_profile_name = stem.to_string();
                            }
                        }
                        ui.text_edit_singleline(&mut self.calibration_profile_name);

                        ui.label("Light source notes");
                        ui.text_edit_singleline(&mut self.calibration_light_source);

                        if ui.button("Save color calibration profile…").clicked() {
                            let dmin_snapshot = calibration_opts_snapshot.dmin_fixed;
                            let name = if self.calibration_profile_name.trim().is_empty() {
                                "profile".to_string()
                            } else {
                                self.calibration_profile_name.trim().to_string()
                            };

                            let profile = calibration::CalibrationProfile {
                                name: name.clone(),
                                light_source: self.calibration_light_source.clone(),
                                matrix: m,
                                dmin_medians: dmin_snapshot,
                            };

                            let base_dir = std::env::current_dir()
                                .unwrap_or_else(|_| PathBuf::from("."))
                                .join("profiles");
                            let _ = std::fs::create_dir_all(&base_dir);

                            if let Some(path) = rfd::FileDialog::new()
                                .set_directory(&base_dir)
                                .set_file_name(&(name.clone() + ".json"))
                                .save_file()
                            {
                                match calibration::save_profile_to_path(&profile, &path) {
                                    Ok(()) => {
                                        self.status =
                                            format!("Saved color calibration profile to {}", path.display());
                                    }
                                    Err(e) => {
                                        self.status =
                                            format!("Failed to save color calibration profile: {}", e);
                                    }
                                }
                            }
                        }
                    }

                    ui.separator();
                    ui.heading("3D LUT");
                    ui.label("Generate a .cube file from the current matrix; apply it in the Process tab.");
                    if ui.button("Generate 3D LUT from current matrix…").clicked() {
                        let matrix = opts.density_matrix;
                        let lut = lut3d::Lut3d::generate_from_matrix(&matrix, 17, 4.0);
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("CUBE LUT", &["cube"])
                            .set_file_name("density_matrix.cube")
                            .save_file()
                        {
                            match lut3d::write_cube(&lut, &path) {
                                Ok(()) => {
                                    self.status = format!("Saved 3D LUT (17³) to {}", path.display());
                                }
                                Err(e) => {
                                    self.status = format!("Failed to save 3D LUT: {}", e);
                                }
                            }
                        }
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
                            if let Some((_, p)) = self.calibration_profiles.get(i) {
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
                                for (i, (_, profile)) in
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
                                    ui.label("No .json profiles in profiles/");
                                }
                            });

                        // Apply selection to current image options.
                        if current_idx == usize::MAX {
                            self.selected_profile_idx = None;
                        } else if let Some((_, profile)) =
                            self.calibration_profiles.get(current_idx).cloned()
                        {
                            self.selected_profile_idx = Some(current_idx);
                            opts.density_matrix = profile.matrix;
                            if let Some(dmin) = profile.dmin_medians {
                                opts.dmin_fixed = Some(dmin);
                                opts.dmin_rect = None;
                            }
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

                ui.separator();
                ui.heading("Export");
                ui.add_space(4.0);

                // Per-image export options
                let label = match entry.export_format {
                    ExportFormat::Tiff16 => "TIFF 16‑bit",
                    ExportFormat::Tiff32 => "TIFF 32‑bit float",
                    ExportFormat::Dng => "DNG (not yet implemented)",
                    ExportFormat::Exr => "EXR (32‑bit float)",
                    ExportFormat::ExrAces2065 => "TIFF 16‑bit + EXR ACES2065-1",
                };
                egui::ComboBox::from_label("Output format")
                    .selected_text(label)
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_label(matches!(entry.export_format, ExportFormat::Tiff16), "TIFF 16‑bit")
                            .clicked()
                        {
                            entry.export_format = ExportFormat::Tiff16;
                        }
                        if ui
                            .selectable_label(matches!(entry.export_format, ExportFormat::Tiff32), "TIFF 32‑bit float")
                            .clicked()
                        {
                            entry.export_format = ExportFormat::Tiff32;
                        }
                        if ui
                            .selectable_label(matches!(entry.export_format, ExportFormat::Dng), "DNG")
                            .clicked()
                        {
                            entry.export_format = ExportFormat::Dng;
                        }
                        if ui
                            .selectable_label(matches!(entry.export_format, ExportFormat::Exr), "EXR (32‑bit float)")
                            .clicked()
                        {
                            entry.export_format = ExportFormat::Exr;
                        }
                        // EXR ACES2065-1: only valid when Use ACEScg is on; greyed out otherwise.
                        let aces_selected = matches!(entry.export_format, ExportFormat::ExrAces2065);
                        if ui
                            .add_enabled(
                                opts.use_acescg,
                                egui::SelectableLabel::new(aces_selected, "TIFF 16‑bit + EXR ACES2065-1"),
                            )
                            .clicked()
                        {
                            entry.export_format = ExportFormat::ExrAces2065;
                        }
                    });

                // If Use ACEScg is off but user had picked ACES2065 export, revert to TIFF 16.
                if !opts.use_acescg && entry.export_format == ExportFormat::ExrAces2065 {
                    entry.export_format = ExportFormat::Tiff16;
                }

                // Keep PipelineOptions.format and export_aces_exr in sync with export dropdown
                match entry.export_format {
                    ExportFormat::Tiff16 => {
                        opts.format = TiffFormat::U16;
                        opts.write_exr = false;
                        opts.export_aces_exr = false;
                    }
                    ExportFormat::Tiff32 => {
                        opts.format = TiffFormat::Float32;
                        opts.write_exr = false;
                        opts.export_aces_exr = false;
                    }
                    ExportFormat::Exr => {
                        opts.format = TiffFormat::Float32;
                        opts.write_exr = true;
                        opts.export_aces_exr = false;
                    }
                    ExportFormat::Dng => { /* pipeline uses TIFF; DNG is placeholder */ }
                    ExportFormat::ExrAces2065 => {
                        opts.format = TiffFormat::U16;
                        opts.write_exr = false;
                        opts.export_aces_exr = true;
                    }
                }

                if opts.no_curve {
                    ui.checkbox(&mut opts.no_invert, "Invert (1-x)");
                } else {
                    ui.label("Inversion handled by print curve.");
                }

                ui.checkbox(&mut opts.write_jpeg, "Also export JPG");

                ui.add_space(8.0);

                // Global export: output folder + convert all
                ui.separator();
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
                if ui.add_enabled(ready, egui::Button::new("Convert all")).clicked() {
                    let output_dir = self.output_dir.clone().unwrap();
                    let mut err: Option<anyhow::Error> = None;
                    for img in &self.images {
                        if matches!(img.export_format, ExportFormat::Dng) {
                            err = Some(anyhow::anyhow!("DNG export is not implemented yet"));
                            break;
                        }
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

                if !self.status.is_empty() {
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new(&self.status).small());
                }
            });

        // ---- Central panel: preview + histogram ----
        egui::CentralPanel::default().show(ctx, |ui| {
            let has_inflight = self.preview_receiver.is_some();
            let show_loader = has_inflight
                && self
                    .preview_started_at
                    .map(|t| t.elapsed() >= Duration::from_millis(250))
                    .unwrap_or(true);

            if show_loader {
                ui.vertical_centered(|ui| {
                    ui.add_space(ui.available_height() / 2.0 - 40.0);
                    ui.spinner();
                    ui.label("Loading preview…");
                });
            } else if let Some(idx) = self.selected_index {
                if idx < self.images.len() {
                    if let Some(ref tex) = self.images[idx].preview_texture {
                        let size = tex.size();
                        let (w, h) = (size[0] as f32, size[1] as f32);
                        let available = ui.available_rect_before_wrap();
                        let scale = (available.width() / w).min((available.height() - 80.0) / h).min(1.0);
                        let display_size = egui::vec2(w * scale, h * scale);
                        let image_resp = ui.image((tex.id(), display_size));
                        let image_rect = image_resp.rect;

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
                        ui.add_space(8.0);
                        if let Some((r_hist, g_hist, b_hist)) = &self.images[idx].histogram {
                            // Histogram aligned to bottom of the central panel.
                            let available = ui.available_rect_before_wrap();
                            let h_hist = 72.0;
                            let rect = egui::Rect::from_min_max(
                                egui::pos2(available.left(), available.bottom() - h_hist),
                                egui::pos2(available.right(), available.bottom()),
                            );
                            let painter = ui.painter_at(rect);

                            let max_r = r_hist.iter().copied().max().unwrap_or(1) as f32;
                            let max_g = g_hist.iter().copied().max().unwrap_or(1) as f32;
                            let max_b = b_hist.iter().copied().max().unwrap_or(1) as f32;
                            let max_total = r_hist
                                .iter()
                                .zip(g_hist.iter())
                                .zip(b_hist.iter())
                                .map(|((&r, &g), &b)| r + g + b)
                                .max()
                                .unwrap_or(1) as f32;
                            let max_all = max_r.max(max_g).max(max_b).max(max_total).max(1.0);
                            let w_bin = rect.width() / 256.0;

                            // Draw combined + channel histograms as vertical lines.
                            for i in 0..256 {
                                let x = rect.left() + (i as f32 + 0.5) * w_bin;
                                let y_base = rect.bottom();

                                let draw_channel = |count: u32, color: egui::Color32, painter: &egui::Painter| {
                                    if count == 0 {
                                        return;
                                    }
                                    let h_norm = (count as f32 / max_all).clamp(0.0, 1.0);
                                    let y_top = y_base - rect.height() * h_norm;
                                    painter.line_segment(
                                        [egui::pos2(x, y_base), egui::pos2(x, y_top)],
                                        egui::Stroke::new(1.0, color),
                                    );
                                };

                                // Combined (R+G+B) in gray, drawn first.
                                let total = r_hist[i] + g_hist[i] + b_hist[i];
                                if total > 0 {
                                    let h_norm = (total as f32 / max_all).clamp(0.0, 1.0);
                                    let y_top = y_base - rect.height() * h_norm;
                                    painter.line_segment(
                                        [egui::pos2(x, y_base), egui::pos2(x, y_top)],
                                        egui::Stroke::new(
                                            2.0,
                                            egui::Color32::from_rgba_premultiplied(200, 200, 200, 220),
                                        ),
                                    );
                                }

                                // Individual channels on top.
                                draw_channel(r_hist[i], egui::Color32::RED, &painter);
                                draw_channel(g_hist[i], egui::Color32::GREEN, &painter);
                                draw_channel(b_hist[i], egui::Color32::BLUE, &painter);
                            }

                            // Axes: X (bottom) and Y (left), in black.
                            let axis_color = egui::Color32::BLACK;
                            let stroke = egui::Stroke::new(1.0, axis_color);
                            painter.line_segment(
                                [egui::pos2(rect.left(), rect.bottom()), egui::pos2(rect.right(), rect.bottom())],
                                stroke,
                            );
                            painter.line_segment(
                                [egui::pos2(rect.left(), rect.bottom()), egui::pos2(rect.left(), rect.top())],
                                stroke,
                            );
                        }
                        return;
                    }
                }
                ui.vertical_centered(|ui| {
                    ui.add_space(ui.available_height() / 2.0 - 20.0);
                    ui.label("Preview not ready yet.");
                });
            } else {
                ui.vertical_centered(|ui| {
                    ui.add_space(ui.available_height() / 2.0 - 20.0);
                    ui.label("Select an image in the strip below to see a preview.");
                });
            }
        });
    }
}

