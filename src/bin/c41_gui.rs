//! C-41 RAW Tool GUI: three-panel layout — center preview, right per-image settings, bottom image strip + global output/convert.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;

use c41_raw_tool::{process_files, process_one_to_preview, PipelineOptions, Rect, TiffFormat};
use eframe::egui;

const PREVIEW_MAX_WIDTH: u32 = 1920;
const PREVIEW_MAX_HEIGHT: u32 = 1200;
const BOTTOM_PANEL_HEIGHT: f32 = 120.0;
const RIGHT_PANEL_WIDTH: f32 = 280.0;

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
}

struct C41Gui {
    images: Vec<ImageEntry>,
    selected_index: Option<usize>,
    output_dir: Option<PathBuf>,
    status: String,
    preview_receiver: Option<mpsc::Receiver<anyhow::Result<(usize, u32, u32, Vec<u8>)>>>,
}

impl Default for C41Gui {
    fn default() -> Self {
        Self {
            images: Vec::new(),
            selected_index: None,
            output_dir: None,
            status: String::new(),
            preview_receiver: None,
        }
    }
}

fn default_options() -> PipelineOptions {
    PipelineOptions {
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
        density_matrix: [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    }
}

fn options_hash_for(path: &PathBuf, opts: &PipelineOptions) -> u64 {
    let mut h = DefaultHasher::new();
    path.display().to_string().hash(&mut h);
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
    h.finish()
}

impl C41Gui {
    fn request_preview_for(&mut self, index: usize, ctx: &egui::Context) {
        if index >= self.images.len() {
            return;
        }
        let path = self.images[index].path.clone();
        let options = self.images[index].options.clone();
        let (tx, rx) = mpsc::channel();
        self.preview_receiver = Some(rx);
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
                    self.status = format!("Preview error: {}", e);
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.preview_receiver = None;
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

        // ---- Right panel: per-image settings + export ----
        egui::SidePanel::right("settings_panel")
            .resizable(false)
            .exact_width(RIGHT_PANEL_WIDTH)
            .show(ctx, |ui| {
                ui.heading("Image Settings");
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

                ui.collapsing("D-min", |ui| {
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
                });

                ui.collapsing("White balance", |ui| {
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

                ui.collapsing("Print curve", |ui| {
                    let mut apply_curve = !opts.no_curve;
                    ui.checkbox(&mut apply_curve, "Apply curve");
                    opts.no_curve = !apply_curve;
                    if apply_curve {
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
                    }
                });

                ui.separator();
                ui.heading("Export");
                ui.add_space(4.0);

                // Per-image export options
                let label = match entry.export_format {
                    ExportFormat::Tiff16 => "TIFF 16‑bit",
                    ExportFormat::Tiff32 => "TIFF 32‑bit float",
                    ExportFormat::Dng => "DNG (not yet implemented)",
                    ExportFormat::Exr => "EXR (32‑bit float)",
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
                    });

                // Keep PipelineOptions.format in sync for TIFF exports
                match entry.export_format {
                    ExportFormat::Tiff16 => {
                        opts.format = TiffFormat::U16;
                        opts.write_exr = false;
                    }
                    ExportFormat::Tiff32 => {
                        opts.format = TiffFormat::Float32;
                        opts.write_exr = false;
                    }
                    ExportFormat::Exr => {
                        opts.format = TiffFormat::Float32;
                        opts.write_exr = true;
                    }
                    ExportFormat::Dng => { /* pipeline uses TIFF; DNG is placeholder */ }
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
                        if let Err(e) = process_files(&[img.path.clone()], &output_dir, &img.options) {
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
            if self.preview_receiver.is_some() {
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
                        ui.image((tex.id(), display_size));
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

