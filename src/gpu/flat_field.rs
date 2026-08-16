//! GPU flat-field division: image /= flat (per-pixel).
//! CPU does resize_flat_field; GPU does the divide + clamp.

use std::sync::Arc;

use ndarray::Array3;
use wgpu::util::DeviceExt;

use super::GpuContext;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlatFieldParams {
    width: u32,
    height: u32,
    eps: f32,
    _pad: u32,
}

pub struct FlatFieldPipeline {
    compute_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    ctx: Arc<GpuContext>,
}

impl FlatFieldPipeline {
    pub fn new(ctx: &Arc<GpuContext>) -> Self {
        let shader = include_str!("flat_field.wgsl");
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("flat_field_shader"),
                source: wgpu::ShaderSource::Wgsl(shader.into()),
            });

        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("flat_field_bgl"),
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

        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("flat_field_layout"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let compute_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("flat_field_pipeline"),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        Self {
            compute_pipeline,
            bgl,
            ctx: Arc::clone(ctx),
        }
    }

    /// Apply flat-field division on GPU. `flat_resampled` must match image dims.
    pub fn run(&self, image: &mut Array3<f32>, flat_resampled: &Array3<f32>) -> anyhow::Result<()> {
        let (height, width, channels) = image.dim();
        anyhow::ensure!(channels == 3, "Expected 3-channel image");
        let (fh, fw, fc) = flat_resampled.dim();
        anyhow::ensure!(
            fh == height && fw == width && fc == 3,
            "Flat field must match image dimensions"
        );

        let pixel_count = width * height;
        let buf_len = pixel_count * 3;
        let buf_bytes = (buf_len * std::mem::size_of::<f32>()) as u64;
        if buf_bytes > self.ctx.device.limits().max_buffer_size {
            anyhow::bail!("Buffer exceeds GPU max; falling back to CPU");
        }

        let params = FlatFieldParams {
            width: width as u32,
            height: height as u32,
            eps: 1.0e-6,
            _pad: 0,
        };

        let mut img_flat = Vec::with_capacity(buf_len);
        for y in 0..height {
            for x in 0..width {
                img_flat.push(image[[y, x, 0]]);
                img_flat.push(image[[y, x, 1]]);
                img_flat.push(image[[y, x, 2]]);
            }
        }

        let mut flat_flat = Vec::with_capacity(buf_len);
        for y in 0..height {
            for x in 0..width {
                flat_flat.push(flat_resampled[[y, x, 0]]);
                flat_flat.push(flat_resampled[[y, x, 1]]);
                flat_flat.push(flat_resampled[[y, x, 2]]);
            }
        }

        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let image_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("flat_field_image"),
            contents: bytemuck::cast_slice(&img_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let flat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("flat_field_flat"),
            contents: bytemuck::cast_slice(&flat_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("flat_field_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flat_field_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: flat_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: image_buf.as_entire_binding(),
                },
            ],
        });

        const MAX_WG: u32 = 65535;
        let total_wg = (pixel_count as u32 + 255) / 256;
        let wg_x = total_wg.min(MAX_WG);
        let wg_y = (total_wg + MAX_WG - 1) / MAX_WG;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat_field_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flat_field_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat_field_staging"),
            size: buf_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&image_buf, 0, &staging, 0, buf_bytes);

        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
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
        staging.unmap();

        Ok(())
    }
}
