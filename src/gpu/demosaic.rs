//! GPU demosaic: RGGB quality (edge-aware green + color-difference R/B).
//!
//! Two-pass compute: pass 1 interpolates green, pass 2 computes R and B from (R-G), (B-G).
//! Falls back to CPU for non-RGGB Bayer and X-Trans.

use std::sync::Arc;

use ndarray::Array3;
use wgpu::util::DeviceExt;

use super::GpuContext;
use crate::demosaic::{demosaic_quality, BayerPattern, CfaPattern};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DemosaicParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

pub struct DemosaicPipeline {
    pass1_pipeline: wgpu::ComputePipeline,
    pass2_pipeline: wgpu::ComputePipeline,
    pass1_bgl: wgpu::BindGroupLayout,
    pass2_bgl: wgpu::BindGroupLayout,
    ctx: Arc<GpuContext>,
}

impl DemosaicPipeline {
    pub fn new(ctx: &Arc<GpuContext>) -> Self {
        let shader_source = include_str!("demosaic.wgsl");
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("demosaic_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        // Pass 1: params, bayer (read), g_plane (read_write)
        let pass1_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("demosaic_pass1_bgl"),
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
                ],
            });

        // Pass 2: params, bayer (read), g_plane (read), rgb (read_write)
        let pass2_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("demosaic_pass2_bgl"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
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

        let pass1_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("demosaic_pass1_layout"),
                bind_group_layouts: &[&pass1_bgl],
                push_constant_ranges: &[],
            });

        let pass2_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("demosaic_pass2_layout"),
                bind_group_layouts: &[&pass2_bgl],
                push_constant_ranges: &[],
            });

        let pass1_pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("demosaic_pass1"),
                layout: Some(&pass1_layout),
                module: &shader_module,
                entry_point: Some("pass1_green"),
                compilation_options: Default::default(),
                cache: None,
            });

        let pass2_pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("demosaic_pass2"),
                layout: Some(&pass2_layout),
                module: &shader_module,
                entry_point: Some("pass2_rgb"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            pass1_pipeline,
            pass2_pipeline,
            pass1_bgl,
            pass2_bgl,
            ctx: Arc::clone(ctx),
        }
    }

    /// Run RGGB quality demosaic on the GPU. Returns RGB (H, W, 3).
    /// Only supports RGGB Bayer; caller must fall back to CPU for other patterns.
    pub fn run_rggb(&self, bayer: &Array3<f32>) -> anyhow::Result<Array3<f32>> {
        let (height, width, c) = bayer.dim();
        anyhow::ensure!(c == 1, "Bayer must be (H, W, 1)");

        let pixel_count = width * height;
        let bayer_bytes = (pixel_count * std::mem::size_of::<f32>()) as u64;
        let rgb_bytes = (pixel_count * 3 * std::mem::size_of::<f32>()) as u64;
        let max_buf = self.ctx.device.limits().max_buffer_size;
        if bayer_bytes.max(rgb_bytes) > max_buf {
            anyhow::bail!("Bayer/RGB buffer exceeds GPU max; falling back to CPU");
        }

        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let bayer_flat: Vec<f32> = bayer.iter().copied().collect();
        let params = DemosaicParams {
            width: width as u32,
            height: height as u32,
            _pad0: 0,
            _pad1: 0,
        };

        let bayer_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("demosaic_bayer"),
            contents: bytemuck::cast_slice(&bayer_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let g_plane_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("demosaic_g_plane"),
            size: bayer_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("demosaic_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let rgb_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("demosaic_rgb"),
            size: rgb_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let pass1_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("demosaic_pass1_bg"),
            layout: &self.pass1_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bayer_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: g_plane_buf.as_entire_binding(),
                },
            ],
        });

        let pass2_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("demosaic_pass2_bg"),
            layout: &self.pass2_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bayer_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: g_plane_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: rgb_buf.as_entire_binding(),
                },
            ],
        });

        const MAX_WG: u32 = 65535;
        let total_wg = (pixel_count as u32 + 255) / 256;
        let wg_x = total_wg.min(MAX_WG);
        let wg_y = (total_wg + MAX_WG - 1) / MAX_WG;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("demosaic_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("demosaic_pass1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pass1_pipeline);
            pass.set_bind_group(0, &pass1_bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("demosaic_pass2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pass2_pipeline);
            pass.set_bind_group(0, &pass2_bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("demosaic_staging"),
            size: rgb_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&rgb_buf, 0, &staging_buf, 0, rgb_bytes);

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

        let mut out = Array3::<f32>::zeros((height, width, 3));
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) * 3;
                out[[y, x, 0]] = result[i];
                out[[y, x, 1]] = result[i + 1];
                out[[y, x, 2]] = result[i + 2];
            }
        }

        drop(data);
        staging_buf.unmap();

        Ok(out)
    }
}

/// Demosaic using GPU when possible (RGGB), CPU otherwise.
pub fn demosaic_gpu_or_cpu(
    bayer: &Array3<f32>,
    pattern: CfaPattern,
    gpu_pipeline: Option<&DemosaicPipeline>,
) -> anyhow::Result<Array3<f32>> {
    let use_gpu = gpu_pipeline.is_some()
        && matches!(pattern, CfaPattern::Bayer(BayerPattern::Rggb));

    if use_gpu {
        if let Ok(rgb) = gpu_pipeline.unwrap().run_rggb(bayer) {
            return Ok(rgb.mapv(|v| v.max(0.0)));
        }
    }

    let mut rgb = demosaic_quality(bayer, pattern)?;
    rgb.mapv_inplace(|v| v.max(0.0));
    Ok(rgb)
}
