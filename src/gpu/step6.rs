//! GPU step 6: all output stages (Ra4, FilmPrint, None, Lut2383) with post-curve ops.

use std::sync::Arc;

use ndarray::Array3;
use wgpu::util::DeviceExt;

use super::GpuContext;
use crate::curve::{self, PrintCurveParams, FilmPrintParams};
use crate::lut3d::Lut3d;
use crate::pipeline::Step6Display;
use crate::{OutputLutEncoding, OutputStage, PipelineOptions};

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
    skin_magenta_shift: f32,
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

pub struct Step6Pipeline {
    compute_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    ctx: Arc<GpuContext>,
}

impl Step6Pipeline {
    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.compute_pipeline
    }
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bgl
    }
}

impl Step6Pipeline {
    pub fn new(ctx: &Arc<GpuContext>) -> Self {
        let shader_source = include_str!("step6.wgsl");
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("step6_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("step6_bgl"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("step6_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("step6_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            compute_pipeline: pipeline,
            bgl: bind_group_layout,
            ctx: Arc::clone(ctx),
        }
    }

    /// Run step 6 on the GPU.
    pub fn run(
        &self,
        image: &Array3<f32>,
        options: &PipelineOptions,
        ra4_params: &PrintCurveParams,
        output_lut_cube: Option<&Lut3d>,
    ) -> anyhow::Result<Step6Display> {
        let (height, width, channels) = image.dim();
        anyhow::ensure!(channels == 3, "Expected 3-channel image");

        let buf_bytes = (width * height * 3 * std::mem::size_of::<f32>()) as u64;
        let max_buf = self.ctx.device.limits().max_buffer_size;
        if buf_bytes * 2 > max_buf {
            anyhow::bail!("Image buffer exceeds GPU max_buffer_size; falling back to CPU");
        }

        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

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
            skin_magenta_shift: options.skin_magenta_shift,
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

        let pixel_count = width * height;
        let buf_len = pixel_count * 3;

        let mut flat: Vec<f32> = Vec::with_capacity(buf_len);
        for y in 0..height {
            for x in 0..width {
                flat.push(image[[y, x, 0]]);
                flat.push(image[[y, x, 1]]);
                flat.push(image[[y, x, 2]]);
            }
        }

        let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("step6_input"),
            contents: bytemuck::cast_slice(&flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("step6_output"),
            size: (buf_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("step6_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Build curve LUT buffer
        let curve_lut_data: Vec<f32> = match options.output_stage {
            OutputStage::Ra4 => {
                let lut = curve::build_density_to_ra4_lut(*ra4_params, d_max);
                lut.iter().map(|&v| v as f32 / 65535.0).collect()
            }
            OutputStage::FilmPrint => {
                let luts = build_film_print_luts_pub(&fp_params, d_max);
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
            label: Some("step6_curve_lut"),
            contents: bytemuck::cast_slice(&curve_lut_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // 3D output LUT buffer
        let output_lut_data: Vec<[f32; 4]> = match output_lut_cube {
            Some(lut) if mode == 3 => lut
                .data_slice()
                .iter()
                .map(|&[r, g, b]| [r, g, b, 0.0])
                .collect(),
            _ => vec![[0.0; 4]],
        };
        let output_lut_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("step6_output_lut"),
            contents: bytemuck::cast_slice(&output_lut_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("step6_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: input_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: curve_lut_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: output_lut_buf.as_entire_binding() },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("step6_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("step6_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            const MAX_WG: u32 = 65535;
            let total_wg = (pixel_count as u32 + 255) / 256;
            let wg_x = total_wg.min(MAX_WG);
            let wg_y = (total_wg + MAX_WG - 1) / MAX_WG;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("step6_staging"),
            size: (buf_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            &output_buf,
            0,
            &staging_buf,
            0,
            (buf_len * std::mem::size_of::<f32>()) as u64,
        );

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

        let display = match options.output_stage {
            OutputStage::Ra4 | OutputStage::FilmPrint => {
                let mut out = Array3::<u16>::zeros((height, width, 3));
                for y in 0..height {
                    for x in 0..width {
                        let i = (y * width + x) * 3;
                        out[[y, x, 0]] = (result[i].clamp(0.0, 1.0) * 65535.0).round() as u16;
                        out[[y, x, 1]] = (result[i + 1].clamp(0.0, 1.0) * 65535.0).round() as u16;
                        out[[y, x, 2]] = (result[i + 2].clamp(0.0, 1.0) * 65535.0).round() as u16;
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
        };

        drop(data);
        staging_buf.unmap();

        Ok(display)
    }
}

/// Build per-channel density→u16 LUTs for Film Print (public wrapper for GPU use).
fn build_film_print_luts_pub(params: &FilmPrintParams, d_max: f32) -> [Vec<u16>; 3] {
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
