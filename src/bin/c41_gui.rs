//! Minimal GUI for C-41 RAW tool. Run with: cargo run --bin c41-gui --features gui

use std::path::PathBuf;

use c41_raw_tool::{process_files, PipelineOptions, Rect, TiffFormat};
use eframe::egui;

fn main() -> eframe::Result<()> {
    eframe::run_native(
        "C-41 RAW Tool",
        eframe::NativeOptions::default(),
        Box::new(|_| Ok(Box::new(C41Gui::default()))),
    )
}

struct C41Gui {
    input_paths: Vec<PathBuf>,
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
}

impl Default for C41Gui {
    fn default() -> Self {
        Self {
            input_paths: Vec::new(),
            output_dir: None,
            use_fixed_dmin: true,
            dmin_r: 0.635294,
            dmin_g: 0.635294,
            dmin_b: 0.623529,
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
        }
    }
}

impl eframe::App for C41Gui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("C-41 RAW Tool");
            ui.add_space(8.0);

            // Files
            if ui.button("Select files…").clicked() {
                if let Some(paths) = rfd::FileDialog::new()
                    .add_filter("ARW & PNG", &["arw", "png"])
                    .pick_files()
                {
                    self.input_paths = paths;
                    self.status = format!("{} file(s) selected", self.input_paths.len());
                }
            }
            if !self.input_paths.is_empty() {
                ui.label(format!("{} file(s)", self.input_paths.len()));
            }
            ui.add_space(4.0);

            // Output
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
            ui.label(out_label);
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

            let ready = !self.input_paths.is_empty()
                && self.output_dir.is_some()
                && (self.use_fixed_dmin || true);

            if ui.add_enabled(ready, egui::Button::new("Convert")).clicked() {
                let output_dir = self.output_dir.clone().unwrap();
                let options = PipelineOptions {
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
                };

                match process_files(&self.input_paths, &output_dir, &options) {
                    Ok(()) => self.status = "Done.".to_string(),
                    Err(e) => self.status = format!("Error: {}", e),
                }
            }

            if !self.status.is_empty() {
                ui.add_space(4.0);
                ui.label(&self.status);
            }
        });
    }
}
