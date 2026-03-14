//! GPU step 4: T→D, WB, film gamma, temp_k, shadow cast correction.
//!
//! Option A: all reductions (auto WB medians, shadow cast analysis) are computed on
//! CPU from a temporary density copy. GPU receives precomputed scalars and does the
//! per-pixel transform in one pass.

use std::sync::Arc;

use ndarray::Array3;
use wgpu::util::DeviceExt;

use super::GpuContext;
use crate::density_ops;
use crate::options::DminMode;
use crate::stats;
use crate::PipelineOptions;

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

pub struct Step4Pipeline {
    compute_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    ctx: Arc<GpuContext>,
}

impl Step4Pipeline {
    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.compute_pipeline
    }
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bgl
    }
}

impl Step4Pipeline {
    pub fn new(ctx: &Arc<GpuContext>) -> Self {
        let shader_source = include_str!("step4.wgsl");
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("step4_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("step4_bgl"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                    label: Some("step4_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("step4_pipeline"),
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

    /// Run step 4 on the GPU. Modifies `image` in-place (transmittance → density).
    ///
    /// CPU precomputes all reduction-based values (auto WB medians, shadow cast
    /// analysis) on a temporary density copy, then the GPU does the full per-pixel
    /// transform in one dispatch.
    pub fn run(
        &self,
        image: &mut Array3<f32>,
        options: &PipelineOptions,
    ) -> anyhow::Result<()> {
        let (height, width, channels) = image.dim();
        anyhow::ensure!(channels == 3, "Expected 3-channel image");

        let buf_bytes = (width * height * 3 * std::mem::size_of::<f32>()) as u64;
        let max_buf = self.ctx.device.limits().max_buffer_size;
        if buf_bytes > max_buf {
            anyhow::bail!("Image buffer exceeds GPU max_buffer_size; falling back to CPU");
        }

        // ── CPU precomputation ──
        // Temporary density copy for stats + shadow cast analysis.
        let mut density_tmp = image.clone();
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
            // Apply scale + offset to the temp density to match CPU state before
            // shadow cast analysis.
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

        drop(density_tmp);

        // ── GPU dispatch ──
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let params = Step4Params {
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

        let image_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("step4_image"),
            contents: bytemuck::cast_slice(&flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("step4_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("step4_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: image_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("step4_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("step4_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (pixel_count as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("step4_staging"),
            size: (buf_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            &image_buf,
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

        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) * 3;
                image[[y, x, 0]] = result[i];
                image[[y, x, 1]] = result[i + 1];
                image[[y, x, 2]] = result[i + 2];
            }
        }

        drop(data);
        staging_buf.unmap();

        Ok(())
    }
}
