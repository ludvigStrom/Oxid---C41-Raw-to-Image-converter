//! C-41 RAW Tool GUI: three-panel layout — center preview, right settings, bottom image strip.

use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;

use c41_raw_tool::{process_files, process_one_to_preview, PipelineOptions, Rect, TiffFormat};
use eframe::egui;

const PREVIEW_MAX_WIDTH: u32 = 1920;
const PREVIEW_MAX_HEIGHT: u32 = 1200;
const BOTTOM_PANEL_HEIGHT: f32 = 100.0;
const RIGHT_PANEL_WIDTH: f32 = 280.0;

fn main() -> eframe::Result<()> {
    eframe::run_native(
        "C-41 RAW Tool",
        eframe::NativeOptions::default(),
        Box::new(|_| Ok(Box::new(C41Gui::default()))),
    )
}

struct C41Gui {
    input_paths: Vec<PathBuf>,
    selected_index: Option<usize>,
    output_dir: Option<PathBuf>,
    use_fixed_dmin: bool,
    dmin_r: f32,
    dmin_g: f32,
    dmin_b: f32,
    dmin_x: u32,
    dmin_y: u32,
    dmin_w: u32,
    dmin_h: u32,
    wb_r: f32,
    wb_g: f32,
    wb_b: f32,
    apply_curve: bool,
    curve_offset: f32,
    curve_gamma: f32,
    curve_pivot: f32,
    curve_white: f32,
    no_invert: bool,
    format_32f: bool,
    write_exr: bool,
    status: String,
    /// When set, we're waiting for preview bytes; when we receive, we upload to texture.
    preview_receiver: Option<mpsc::Receiver<anyhow::Result<(u32, u32, Vec<u8>)>>>,
    preview_texture: Option<egui::TextureHandle>,
    /// Options used for the current preview (so we know when to invalidate).
    preview_options_hash: u64,
}

impl Default for C41Gui {
    fn default() -> Self {
        Self {
            input_paths: Vec::new(),
            selected_index: None,
            output_dir: None,
            use_fixed_dmin: true,
            // Defaults tuned for ARW D-min (measured once via CLI)
            dmin_r: 0.016297,
            dmin_g: 0.031067,
            dmin_b: 0.026215,
            dmin_x: 35,
            dmin_y: 15,
            dmin_w: 20,
            dmin_h: 20,
            wb_r: 1.15,
            wb_g: 0.88,
            wb_b: 1.0,
            apply_curve: true,
            curve_offset: 0.0,
            curve_gamma: 2.5,
            curve_pivot: 3.0,
            curve_white: 0.745,
            no_invert: false,
            format_32f: true,
            write_exr: false,
            status: String::new(),
            preview_receiver: None,
            preview_texture: None,
            preview_options_hash: 0,
        }
    }
}

impl C41Gui {
    fn options(&self) -> PipelineOptions {
        PipelineOptions {
            dmin_rect: if self.use_fixed_dmin {
                None
            } else {
                Some(Rect {
                    x: self.dmin_x,
                    y: self.dmin_y,
                    width: self.dmin_w,
                    height: self.dmin_h,
                })
            },
            dmin_fixed: if self.use_fixed_dmin {
                Some((self.dmin_r, self.dmin_g, self.dmin_b))
            } else {
                None
            },
            format: if self.format_32f {
                TiffFormat::Float32
            } else {
                TiffFormat::U16
            },
            write_exr: self.write_exr,
            no_invert: self.no_invert,
            no_curve: !self.apply_curve,
            wb_r: self.wb_r,
            wb_g: self.wb_g,
            wb_b: self.wb_b,
            curve_offset: self.curve_offset,
            curve_gamma: self.curve_gamma,
            curve_pivot: self.curve_pivot,
            curve_white: self.curve_white,
        }
    }

    fn options_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.use_fixed_dmin.hash(&mut h);
        self.dmin_r.to_bits().hash(&mut h);
        self.dmin_g.to_bits().hash(&mut h);
        self.dmin_b.to_bits().hash(&mut h);
        self.dmin_x.hash(&mut h);
        self.dmin_y.hash(&mut h);
        self.dmin_w.hash(&mut h);
        self.dmin_h.hash(&mut h);
        self.wb_r.to_bits().hash(&mut h);
        self.wb_g.to_bits().hash(&mut h);
        self.wb_b.to_bits().hash(&mut h);
        self.apply_curve.hash(&mut h);
        self.curve_offset.to_bits().hash(&mut h);
        self.curve_gamma.to_bits().hash(&mut h);
        self.curve_pivot.to_bits().hash(&mut h);
        self.curve_white.to_bits().hash(&mut h);
        self.no_invert.hash(&mut h);
        self.format_32f.hash(&mut h);
        h.finish()
    }

    fn request_preview(&mut self, path: PathBuf, options: PipelineOptions, ctx: &egui::Context) {
        let (tx, rx) = mpsc::channel();
        self.preview_receiver = Some(rx);
        // Keep previous texture visible until new one is ready to avoid flash
        thread::spawn(move || {
            let result = process_one_to_preview(
                &path,
                &options,
                PREVIEW_MAX_WIDTH,
                PREVIEW_MAX_HEIGHT,
            );
            let _ = tx.send(result);
        });
        ctx.request_repaint();
    }
}

impl eframe::App for C41Gui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let options = self.options();
        let current_hash = self.options_hash();

        // Poll preview worker
        if let Some(rx) = self.preview_receiver.as_ref() {
            match rx.try_recv() {
                Ok(Ok((w, h, rgb))) => {
                    self.preview_receiver = None;
                    let size = [w as usize, h as usize];
                    let pixels: Vec<egui::Color32> = rgb
                        .chunks_exact(3)
                        .map(|c| egui::Color32::from_rgb(c[0], c[1], c[2]))
                        .collect();
                    let image = egui::ColorImage {
                        size,
                        pixels,
                    };
                    self.preview_texture = Some(ctx.load_texture(
                        "preview",
                        image,
                        egui::TextureOptions::default(),
                    ));
                    self.preview_options_hash = current_hash;
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

        // If we have a selection and no pending receiver and (no texture or options changed), start preview
        if self.preview_receiver.is_none() {
            if let Some(idx) = self.selected_index {
                if idx < self.input_paths.len() {
                    let path = self.input_paths[idx].clone();
                    let need_new = self.preview_texture.is_none()
                        || self.preview_options_hash != current_hash;
                    if need_new {
                        self.request_preview(path, options.clone(), ctx);
                    }
                }
            } else {
                self.preview_texture = None;
            }
        }

        // ---- Bottom panel: image strip ----
        egui::TopBottomPanel::bottom("bottom_panel")
            .min_height(BOTTOM_PANEL_HEIGHT)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Add image…").clicked() {
                        if let Some(paths) = rfd::FileDialog::new()
                            .add_filter("ARW & PNG", &["arw", "png"])
                            .pick_files()
                        {
                            for p in paths {
                                if !self.input_paths.contains(&p) {
                                    self.input_paths.push(p);
                                }
                            }
                            self.status = format!("{} file(s)", self.input_paths.len());
                        }
                    }
                    ui.separator();
                    egui::ScrollArea::horizontal().show(ui, |ui| {
                        let mut to_remove = Vec::new();
                        ui.horizontal(|ui| {
                            for (i, path) in self.input_paths.iter().enumerate() {
                                let name = path
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or("?");
                                let selected = self.selected_index == Some(i);
                                let resp = ui
                                    .selectable_label(selected, name)
                                    .on_hover_text(path.display().to_string());
                                if resp.clicked() {
                                    self.selected_index = Some(i);
                                }
                                if ui.small_button("✕").clicked() {
                                    to_remove.push(i);
                                }
                            }
                        });
                        for i in to_remove.into_iter().rev() {
                            self.input_paths.remove(i);
                            if self.selected_index == Some(i) {
                                self.selected_index = None;
                            } else if self.selected_index.map(|s| s > i).unwrap_or(false) {
                                self.selected_index = self.selected_index.map(|s| s - 1);
                            }
                        }
                    });
                });
            });

        // ---- Right panel: settings ----
        egui::SidePanel::right("settings_panel")
            .resizable(false)
            .exact_width(RIGHT_PANEL_WIDTH)
            .show(ctx, |ui| {
                ui.heading("Settings");
                ui.add_space(4.0);

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
                ui.add_space(8.0);

                ui.collapsing("D-min", |ui| {
                    ui.checkbox(&mut self.use_fixed_dmin, "Use fixed D-min (R,G,B)");
                    if self.use_fixed_dmin {
                        ui.horizontal(|ui| {
                            ui.label("R");
                            ui.add(egui::DragValue::new(&mut self.dmin_r).range(0.0..=1.0).speed(0.01));
                            ui.label("G");
                            ui.add(egui::DragValue::new(&mut self.dmin_g).range(0.0..=1.0).speed(0.01));
                            ui.label("B");
                            ui.add(egui::DragValue::new(&mut self.dmin_b).range(0.0..=1.0).speed(0.01));
                        });
                    } else {
                        ui.horizontal(|ui| {
                            ui.label("x,y,w,h");
                            ui.add(egui::DragValue::new(&mut self.dmin_x).speed(1));
                            ui.add(egui::DragValue::new(&mut self.dmin_y).speed(1));
                            ui.add(egui::DragValue::new(&mut self.dmin_w).speed(1));
                            ui.add(egui::DragValue::new(&mut self.dmin_h).speed(1));
                        });
                    }
                });

                ui.collapsing("White balance", |ui| {
                    ui.horizontal(|ui| {
                        ui.label("R");
                        ui.add(egui::Slider::new(&mut self.wb_r, 0.5..=2.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("G");
                        ui.add(egui::Slider::new(&mut self.wb_g, 0.5..=2.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("B");
                        ui.add(egui::Slider::new(&mut self.wb_b, 0.5..=2.0));
                    });
                });

                ui.collapsing("Print curve", |ui| {
                    ui.checkbox(&mut self.apply_curve, "Apply curve");
                    if self.apply_curve {
                        ui.horizontal(|ui| {
                            ui.label("Offset");
                            ui.add(egui::DragValue::new(&mut self.curve_offset).range(-2.0..=2.0).speed(0.05));
                        });
                        ui.horizontal(|ui| {
                            ui.label("Gamma");
                            ui.add(egui::Slider::new(&mut self.curve_gamma, 0.5..=5.0));
                        });
                        ui.horizontal(|ui| {
                            ui.label("Pivot");
                            ui.add(egui::DragValue::new(&mut self.curve_pivot).range(0.1..=10.0).speed(0.1));
                        });
                        ui.horizontal(|ui| {
                            ui.label("White");
                            ui.add(egui::Slider::new(&mut self.curve_white, 0.3..=1.0));
                        });
                    } else {
                        ui.checkbox(&mut self.no_invert, "Invert (1-x)");
                        ui.checkbox(&mut self.format_32f, "32-bit float (else 16-bit)");
                    }
                });

                ui.checkbox(&mut self.write_exr, "Also write EXR");
                ui.add_space(8.0);

                let ready = !self.input_paths.is_empty() && self.output_dir.is_some();
                if ui.add_enabled(ready, egui::Button::new("Convert all")).clicked() {
                    let output_dir = self.output_dir.clone().unwrap();
                    match process_files(&self.input_paths, &output_dir, &options) {
                        Ok(()) => self.status = "Done.".to_string(),
                        Err(e) => self.status = format!("Error: {}", e),
                    }
                }

                if !self.status.is_empty() {
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new(&self.status).small());
                }
            });

        // ---- Central panel: preview ----
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.preview_receiver.is_some() {
                ui.vertical_centered(|ui| {
                    ui.add_space(ui.available_height() / 2.0 - 20.0);
                    ui.spinner();
                    ui.label("Loading preview…");
                });
            } else if let Some(ref tex) = self.preview_texture {
                let size = tex.size();
                let (w, h) = (size[0] as f32, size[1] as f32);
                let available = ui.available_rect_before_wrap();
                let scale = (available.width() / w).min(available.height() / h).min(1.0);
                let display_size = egui::vec2(w * scale, h * scale);
                ui.image((tex.id(), display_size));
            } else {
                ui.vertical_centered(|ui| {
                    ui.add_space(ui.available_height() / 2.0 - 20.0);
                    ui.label("Select an image in the strip below to see a preview.");
                });
            }
        });
    }
}
