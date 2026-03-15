//! GPU step 5: density matrix / 3D LUT, highlight spread, saturation, zone adjustments.
//!
//! Runs the same math as `pipeline::step_5_calibration` and `density_ops` but on the GPU.

use ndarray::Array3;
use std::sync::Arc;

use super::GpuContext;
use crate::lut3d::Lut3d;
use crate::PipelineOptions;

/// Uniform params layout matching the WGSL struct (std140-compatible).
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
    zone_shadow_gain: f32,
    zone_mid_gain: f32,
    zone_highlight_gain: f32,
    color_shadow_gain_r: f32,
    color_shadow_gain_g: f32,
    color_shadow_gain_b: f32,
    color_mid_gain_r: f32,
    color_mid_gain_g: f32,
    color_mid_gain_b: f32,
    color_highlight_gain_r: f32,
    color_highlight_gain_g: f32,
    color_highlight_gain_b: f32,
    curve_offset: f32,
    d_zone_min: f32,
    d_zone_max: f32,
    highlight_rolloff: f32,
    highlight_rolloff_d_mid: f32,
    _pad_before_mat: [f32; 2], // align mat_r0 to 16-byte boundary (offset 144)
    // mat row 0 + padding
    mat_r0: [f32; 3],
    _pad0: f32,
    // mat row 1 + padding
    mat_r1: [f32; 3],
    _pad1: f32,
    // mat row 2 + padding
    mat_r2: [f32; 3],
    _pad2: f32,
}

/// Cached GPU pipeline for step 5 (reuse across frames).
pub struct Step5Pipeline {
    compute_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    ctx: Arc<GpuContext>,
}

impl Step5Pipeline {
    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.compute_pipeline
    }
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bgl
    }
}

impl Step5Pipeline {
    pub fn new(ctx: &Arc<GpuContext>) -> Self {
        let shader_source = include_str!("step5.wgsl");
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("step5_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("step5_bgl"),
                    entries: &[
                        // binding 0: uniform params
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
                        // binding 1: image storage buffer (read_write)
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
                        // binding 2: LUT data (read-only storage)
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
                    ],
                });

        let pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("step5_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("step5_pipeline"),
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

    /// Run step 5 on the GPU, modifying `image` in-place.
    /// Returns `Ok(())` on success, or an error if something goes wrong.
    pub fn run(
        &self,
        image: &mut Array3<f32>,
        options: &PipelineOptions,
        lut3d: Option<&Lut3d>,
    ) -> anyhow::Result<()> {
        let (height, width, channels) = image.dim();
        anyhow::ensure!(channels == 3, "Expected 3-channel image");

        let buf_bytes = (width * height * 3 * std::mem::size_of::<f32>()) as u64;
        let max_buf = self.ctx.device.limits().max_buffer_size;
        if buf_bytes > max_buf {
            anyhow::bail!("Image buffer exceeds GPU max_buffer_size; falling back to CPU");
        }

        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let m = if options.apply_color_profile {
            options.density_matrix
        } else {
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        };

        let (use_lut, lut_size, lut_d_max) = match lut3d {
            Some(lut) => (1u32, lut.size as u32, lut.d_max),
            None => (0u32, 1u32, 1.0f32),
        };

        let (d_zone_min, d_zone_max) =
            crate::density_ops::zone_density_range(image, options.curve_offset);

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
            zone_shadow_gain: options.zone_shadow_gain,
            zone_mid_gain: options.zone_mid_gain,
            zone_highlight_gain: options.zone_highlight_gain,
            color_shadow_gain_r: options.color_shadow_gain_r,
            color_shadow_gain_g: options.color_shadow_gain_g,
            color_shadow_gain_b: options.color_shadow_gain_b,
            color_mid_gain_r: options.color_mid_gain_r,
            color_mid_gain_g: options.color_mid_gain_g,
            color_mid_gain_b: options.color_mid_gain_b,
            color_highlight_gain_r: options.color_highlight_gain_r,
            color_highlight_gain_g: options.color_highlight_gain_g,
            color_highlight_gain_b: options.color_highlight_gain_b,
            curve_offset: options.curve_offset,
            d_zone_min,
            d_zone_max,
            highlight_rolloff: options.highlight_rolloff,
            highlight_rolloff_d_mid: options.highlight_rolloff_d_mid,
            _pad_before_mat: [0.0; 2],
            mat_r0: m[0],
            _pad0: 0.0,
            mat_r1: m[1],
            _pad1: 0.0,
            mat_r2: m[2],
            _pad2: 0.0,
        };

        // Flatten image to contiguous f32 slice (H*W*3).
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
            label: Some("step5_image"),
            contents: bytemuck::cast_slice(&flat),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("step5_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // LUT buffer: if no LUT, upload a dummy 1-entry buffer.
        let lut_gpu_data: Vec<[f32; 4]> = match lut3d {
            Some(lut) => lut
                .data_slice()
                .iter()
                .map(|&[r, g, b]| [r, g, b, 0.0])
                .collect(),
            None => vec![[0.0; 4]],
        };
        let lut_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("step5_lut"),
            contents: bytemuck::cast_slice(&lut_gpu_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("step5_bg"),
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: lut_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("step5_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("step5_pass"),
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

        // Copy result to a readable staging buffer.
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("step5_staging"),
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

        // Map and read back.
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

use wgpu::util::DeviceExt;
