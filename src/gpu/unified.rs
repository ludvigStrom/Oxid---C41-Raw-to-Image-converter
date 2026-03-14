//! Unified GPU pipeline: run steps 4→5→6 with a single upload and readback.
//!
//! Supports partial runs (start from step 5 or 6) for the cached preview path.
//! CPU precomputes reduction-based analysis (auto WB, shadow cast) for step 4;
//! everything else runs on GPU.

use std::sync::Arc;

use ndarray::Array3;
use wgpu::util::DeviceExt;

use super::step4::Step4Pipeline;
use super::step5::Step5Pipeline;
use super::step6::Step6Pipeline;
use super::GpuContext;
use crate::curve::{self, PrintCurveParams};
use crate::density_ops;
use crate::lut3d::Lut3d;
use crate::options::DminMode;
use crate::pipeline::Step6Display;
use crate::stats;
use crate::{OutputLutEncoding, OutputStage, PipelineOptions};

/// Unified GPU pipeline holding compiled shaders for steps 4, 5, and 6.
/// Created once at startup, reused for every preview / export.
pub struct GpuPipeline {
    pub ctx: Arc<GpuContext>,
    step4: Step4Pipeline,
    step5: Step5Pipeline,
    step6: Step6Pipeline,
}

impl GpuPipeline {
    /// Try to initialize the full GPU pipeline. Returns `None` if no GPU adapter.
    pub fn try_new() -> Option<Self> {
        let ctx = Arc::new(GpuContext::try_new()?);
        let step4 = Step4Pipeline::new(&ctx);
        let step5 = Step5Pipeline::new(&ctx);
        let step6 = Step6Pipeline::new(&ctx);
        Some(Self {
            ctx,
            step4,
            step5,
            step6,
        })
    }

    /// Run pipeline steps from `start_step` through 6 on the GPU.
    ///
    /// - `start_step == 4`: image is transmittance; runs T→D→WB→step5→step6
    /// - `start_step == 5`: image is density (after step 4); runs step5→step6
    /// - `start_step >= 6`: image is calibrated density (after step 5); runs step6 only
    ///
    /// Returns the step 6 display output. The `image` is NOT modified in place
    /// (the GPU works on its own copy). If step 4 or 5 results are needed for
    /// caching, callers should use the individual pipelines instead.
    pub fn run_from_step(
        &self,
        image: &Array3<f32>,
        start_step: u32,
        options: &PipelineOptions,
        lut3d: Option<&Lut3d>,
        ra4_params: &PrintCurveParams,
        output_lut_cube: Option<&Lut3d>,
    ) -> anyhow::Result<Step6Display> {
        let (height, width, channels) = image.dim();
        anyhow::ensure!(channels == 3, "Expected 3-channel image");

        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let pixel_count = width * height;
        let buf_len = pixel_count * 3;
        let buf_bytes = (buf_len * std::mem::size_of::<f32>()) as u64;

        let max_buf = device.limits().max_buffer_size;
        // Step 6 needs two buffers (input + output), so check 2x.
        if buf_bytes * 2 > max_buf {
            anyhow::bail!(
                "Image buffer ({:.1} MB x2) exceeds GPU max_buffer_size ({:.1} MB); falling back to CPU",
                buf_bytes as f64 / 1048576.0,
                max_buf as f64 / 1048576.0,
            );
        }

        // ── Flatten image ──
        let mut flat: Vec<f32> = Vec::with_capacity(buf_len);
        for y in 0..height {
            for x in 0..width {
                flat.push(image[[y, x, 0]]);
                flat.push(image[[y, x, 1]]);
                flat.push(image[[y, x, 2]]);
            }
        }

        let image_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("unified_image"),
            contents: bytemuck::cast_slice(&flat),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC,
        });

        let workgroups = (pixel_count as u32 + 255) / 256;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("unified_encoder"),
        });

        // ── Step 4 ──
        if start_step <= 4 && options.debug_pipeline_step >= 4 {
            let step4_params = self.build_step4_params(image, options, width, height);
            let step4_params_buf =
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("u_step4_params"),
                    contents: bytemuck::bytes_of(&step4_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let step4_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("u_step4_bg"),
                layout: self.step4.bind_group_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: step4_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: image_buf.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("u_step4_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.step4.pipeline());
            pass.set_bind_group(0, &step4_bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // ── Step 5 ──
        if start_step <= 5 && options.debug_pipeline_step >= 5 {
            let (step5_params_buf, step5_lut_buf) =
                self.build_step5_buffers(device, options, lut3d, width, height);
            let step5_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("u_step5_bg"),
                layout: self.step5.bind_group_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: step5_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: image_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: step5_lut_buf.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("u_step5_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.step5.pipeline());
            pass.set_bind_group(0, &step5_bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // ── Step 6 ──
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("u_step6_output"),
            size: buf_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        if options.debug_pipeline_step >= 6 {
            let (step6_params_buf, step6_curve_lut_buf, step6_output_lut_buf) =
                self.build_step6_buffers(device, options, ra4_params, output_lut_cube, width, height);
            let step6_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("u_step6_bg"),
                layout: self.step6.bind_group_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: step6_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: image_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: step6_curve_lut_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: step6_output_lut_buf.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("u_step6_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.step6.pipeline());
            pass.set_bind_group(0, &step6_bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // ── Readback ──
        let readback_source = if options.debug_pipeline_step >= 6 {
            &output_buf
        } else {
            &image_buf
        };

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("u_staging"),
            size: buf_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(readback_source, 0, &staging_buf, 0, buf_bytes);

        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| anyhow::anyhow!("GPU map channel disconnected"))?
            .map_err(|e| anyhow::anyhow!("GPU buffer map failed: {:?}", e))?;

        let data = slice.get_mapped_range();
        let result: &[f32] = bytemuck::cast_slice(&data);

        let display = if options.debug_pipeline_step < 6 {
            let mut out = Array3::<f32>::zeros((height, width, 3));
            for y in 0..height {
                for x in 0..width {
                    let i = (y * width + x) * 3;
                    out[[y, x, 0]] = result[i];
                    out[[y, x, 1]] = result[i + 1];
                    out[[y, x, 2]] = result[i + 2];
                }
            }
            Step6Display::PassthroughDensity(out)
        } else {
            match options.output_stage {
                OutputStage::Ra4 | OutputStage::FilmPrint => {
                    let mut out = Array3::<u16>::zeros((height, width, 3));
                    for y in 0..height {
                        for x in 0..width {
                            let i = (y * width + x) * 3;
                            out[[y, x, 0]] =
                                (result[i].clamp(0.0, 1.0) * 65535.0).round() as u16;
                            out[[y, x, 1]] =
                                (result[i + 1].clamp(0.0, 1.0) * 65535.0).round() as u16;
                            out[[y, x, 2]] =
                                (result[i + 2].clamp(0.0, 1.0) * 65535.0).round() as u16;
                        }
                    }
                    Step6Display::U16(out)
                }
                OutputStage::None | OutputStage::Lut2383 => {
                    let mut out = Array3::<f32>::zeros((height, width, 3));
                    for y in 0..height {
                        for x in 0..width {
                            let i = (y * width + x) * 3;
                            out[[y, x, 0]] = result[i];
                            out[[y, x, 1]] = result[i + 1];
                            out[[y, x, 2]] = result[i + 2];
                        }
                    }
                    Step6Display::F32(out)
                }
            }
        };

        drop(data);
        staging_buf.unmap();

        Ok(display)
    }
}

// ── Private helpers for building per-step GPU buffers ──

// Re-export the Pod structs from step modules.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Step4Params {
    width: u32,
    height: u32,
    s_r: f32,
    s_g: f32,
    s_b: f32,
    off_r: f32,
    off_g: f32,
    off_b: f32,
    shadow_cast_active: u32,
    cr: f32,
    cg: f32,
    cb: f32,
    shadow_cast_strength: f32,
    inv_threshold: f32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Step5Params {
    width: u32,
    height: u32,
    use_lut: u32,
    lut_size: u32,
    lut_d_max: f32,
    saturation: f32,
    zone_shadows: f32,
    zone_highlights: f32,
    color_shadows_r: f32,
    color_shadows_g: f32,
    color_shadows_b: f32,
    color_mids_r: f32,
    color_mids_g: f32,
    color_mids_b: f32,
    color_highlights_r: f32,
    color_highlights_g: f32,
    color_highlights_b: f32,
    _pad_pre_mat: [f32; 3],
    mat_r0: [f32; 3],
    _pad0: f32,
    mat_r1: [f32; 3],
    _pad1: f32,
    mat_r2: [f32; 3],
    _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Step6Params {
    width: u32,
    height: u32,
    mode: u32,
    d_max: f32,
    lut_in_black: f32,
    lut_in_white: f32,
    lut_in_mid: f32,
    levels_active: u32,
    white_point: f32,
    toe_strength: f32,
    shoulder_strength: f32,
    soft_clip: f32,
    highlight_warmth: f32,
    apply_lab: u32,
    lab_separation: f32,
    no_invert: u32,
    color_bleed: f32,
    vibrance: f32,
    output_lut_encoding: u32,
    output_lut_size: u32,
    use_output_lut: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

impl GpuPipeline {
    fn build_step4_params(
        &self,
        transmittance_image: &Array3<f32>,
        options: &PipelineOptions,
        width: usize,
        height: usize,
    ) -> Step4Params {
        let mut density_tmp = transmittance_image.clone();
        density_tmp.mapv_inplace(|t| (-(t.max(1e-10_f32)).log10()).max(0.0));

        let (auto_s_r, auto_s_g, auto_s_b) =
            if options.auto_wb && options.dmin_mode != DminMode::Off {
                let wb_stats = stats::wb_channel_stats(&density_tmp, options);
                let med_r = wb_stats[0].2.max(1e-4);
                let med_g = wb_stats[1].2.max(1e-4);
                let med_b = wb_stats[2].2.max(1e-4);
                let mean_d = (med_r + med_g + med_b) / 3.0;
                (mean_d / med_r, mean_d / med_g, mean_d / med_b)
            } else {
                (1.0, 1.0, 1.0)
            };

        let (man_s_r, man_s_g, man_s_b) = if options.apply_white_balance {
            (options.wb_r, options.wb_g, options.wb_b)
        } else {
            (1.0, 1.0, 1.0)
        };

        let inv_gamma = 1.0 / options.film_gamma.max(0.1);
        let s_r = auto_s_r * man_s_r * inv_gamma;
        let s_g = auto_s_g * man_s_g * inv_gamma;
        let s_b = auto_s_b * man_s_b * inv_gamma;

        let (off_r, off_g, off_b) = if let Some(k) = options.temp_k {
            let (tr, tg, tb) = crate::color::temp_k_to_wb_gains(k);
            (
                -(tr.max(1e-6) as f64).log10() as f32,
                -(tg.max(1e-6) as f64).log10() as f32,
                -(tb.max(1e-6) as f64).log10() as f32,
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        let (shadow_cast_active, cr, cg, cb) = if options.shadow_cast_strength > 0.0 {
            density_tmp
                .slice_mut(ndarray::s![.., .., 0])
                .mapv_inplace(|v| v * s_r + off_r);
            density_tmp
                .slice_mut(ndarray::s![.., .., 1])
                .mapv_inplace(|v| v * s_g + off_g);
            density_tmp
                .slice_mut(ndarray::s![.., .., 2])
                .mapv_inplace(|v| v * s_b + off_b);
            let (cr, cg, cb) = density_ops::analyze_shadow_cast(&density_tmp, 0.8);
            (1u32, cr, cg, cb)
        } else {
            (0u32, 0.0, 0.0, 0.0)
        };

        Step4Params {
            width: width as u32,
            height: height as u32,
            s_r,
            s_g,
            s_b,
            off_r,
            off_g,
            off_b,
            shadow_cast_active,
            cr,
            cg,
            cb,
            shadow_cast_strength: options.shadow_cast_strength,
            inv_threshold: 1.0 / 0.8_f32,
            _pad0: 0,
            _pad1: 0,
        }
    }

    fn build_step5_buffers(
        &self,
        device: &wgpu::Device,
        options: &PipelineOptions,
        lut3d: Option<&Lut3d>,
        width: usize,
        height: usize,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let m = if options.apply_color_profile {
            options.density_matrix
        } else {
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        };

        let (use_lut, lut_size, lut_d_max) = match lut3d {
            Some(lut) => (1u32, lut.size as u32, lut.d_max),
            None => (0u32, 1u32, 1.0f32),
        };

        let params = Step5Params {
            width: width as u32,
            height: height as u32,
            use_lut,
            lut_size,
            lut_d_max,
            saturation: options.saturation,
            zone_shadows: options.zone_shadows,
            zone_highlights: options.zone_highlights,
            color_shadows_r: options.color_shadows_r,
            color_shadows_g: options.color_shadows_g,
            color_shadows_b: options.color_shadows_b,
            color_mids_r: options.color_mids_r,
            color_mids_g: options.color_mids_g,
            color_mids_b: options.color_mids_b,
            color_highlights_r: options.color_highlights_r,
            color_highlights_g: options.color_highlights_g,
            color_highlights_b: options.color_highlights_b,
            _pad_pre_mat: [0.0; 3],
            mat_r0: m[0],
            _pad0: 0.0,
            mat_r1: m[1],
            _pad1: 0.0,
            mat_r2: m[2],
            _pad2: 0.0,
        };

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("u_step5_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let lut_gpu_data: Vec<[f32; 4]> = match lut3d {
            Some(lut) => lut
                .data_slice()
                .iter()
                .map(|&[r, g, b]| [r, g, b, 0.0])
                .collect(),
            None => vec![[0.0; 4]],
        };
        let lut_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("u_step5_lut"),
            contents: bytemuck::cast_slice(&lut_gpu_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        (params_buf, lut_buf)
    }

    fn build_step6_buffers(
        &self,
        device: &wgpu::Device,
        options: &PipelineOptions,
        ra4_params: &PrintCurveParams,
        output_lut_cube: Option<&Lut3d>,
        width: usize,
        height: usize,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
        let mode = match options.output_stage {
            OutputStage::Ra4 => 0u32,
            OutputStage::FilmPrint => 1u32,
            OutputStage::None => 2u32,
            OutputStage::Lut2383 => 3u32,
        };
        let d_max = 4.0_f32;
        let levels_active = options.lut_in_black != 0.0
            || options.lut_in_white != 1.0
            || (options.lut_in_mid - 1.0).abs() > 1e-6;

        let fp_params = crate::post_curve::build_film_print_params(options);

        let params = Step6Params {
            width: width as u32,
            height: height as u32,
            mode,
            d_max,
            lut_in_black: options.lut_in_black,
            lut_in_white: options.lut_in_white,
            lut_in_mid: options.lut_in_mid,
            levels_active: if levels_active { 1 } else { 0 },
            white_point: options.curve_white,
            toe_strength: options.toe_strength,
            shoulder_strength: options.shoulder_strength,
            soft_clip: options.soft_clip,
            highlight_warmth: options.highlight_warmth,
            apply_lab: if options.apply_lab { 1 } else { 0 },
            lab_separation: options.lab_separation,
            no_invert: if options.no_invert { 1 } else { 0 },
            color_bleed: fp_params.color_bleed,
            vibrance: fp_params.vibrance,
            output_lut_encoding: match options.output_lut_encoding {
                OutputLutEncoding::CineonLog => 0,
                OutputLutEncoding::Rec709 => 1,
                OutputLutEncoding::LinearDensity => 2,
            },
            output_lut_size: output_lut_cube.map(|l| l.size as u32).unwrap_or(1),
            use_output_lut: if output_lut_cube.is_some() && mode == 3 { 1 } else { 0 },
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("u_step6_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let curve_lut_data: Vec<f32> = match options.output_stage {
            OutputStage::Ra4 => {
                let lut = curve::build_density_to_ra4_lut(*ra4_params, d_max);
                lut.iter().map(|&v| v as f32 / 65535.0).collect()
            }
            OutputStage::FilmPrint => {
                let luts = build_film_print_luts(&fp_params, d_max);
                let mut data = Vec::with_capacity(65536 * 3);
                for ch in 0..3 {
                    for &v in &luts[ch] {
                        data.push(v as f32 / 65535.0);
                    }
                }
                data
            }
            _ => vec![0.0_f32],
        };
        let curve_lut_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("u_step6_curve_lut"),
            contents: bytemuck::cast_slice(&curve_lut_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_lut_data: Vec<[f32; 4]> = match output_lut_cube {
            Some(lut) if mode == 3 => lut
                .data_slice()
                .iter()
                .map(|&[r, g, b]| [r, g, b, 0.0])
                .collect(),
            _ => vec![[0.0; 4]],
        };
        let output_lut_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("u_step6_output_lut"),
            contents: bytemuck::cast_slice(&output_lut_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        (params_buf, curve_lut_buf, output_lut_buf)
    }
}

fn build_film_print_luts(
    params: &crate::curve::FilmPrintParams,
    d_max: f32,
) -> [Vec<u16>; 3] {
    let mut luts: [Vec<u16>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for ch in 0..3 {
        let offset = params.base.offset + params.offset_rgb[ch];
        let gamma = (params.base.gamma * params.gamma_rgb[ch]).max(0.1);
        let pivot = params.base.pivot;
        let lut: Vec<u16> = (0..65536)
            .map(|i| {
                let d = (i as f32 / 65535.0) * d_max;
                let d = d.max(0.0);
                let log_exposure = d + offset;
                let linear_exposure = (10.0_f32).powf(log_exposure);
                let eg = linear_exposure.powf(gamma);
                let pg = pivot.powf(gamma);
                let mut y = (eg / (eg + pg)).clamp(0.0, 1.0);
                const SHOULDER_START: f32 = 0.93;
                if y > SHOULDER_START {
                    let t = (y - SHOULDER_START) / (1.0 - SHOULDER_START);
                    let t_shaped = 1.0 - (1.0 - t).powf(1.5);
                    y = SHOULDER_START + t_shaped * (1.0 - SHOULDER_START);
                }
                (y * 65535.0).round() as u16
            })
            .collect();
        luts[ch] = lut;
    }
    luts
}
