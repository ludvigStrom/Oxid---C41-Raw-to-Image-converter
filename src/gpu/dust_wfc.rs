//! GPU coherent exemplar copy for Wave-function dust heal.
//! Candidate list stays on the CPU; per-pixel propagate/search matches `dust_wfc`.

use std::sync::Arc;

use ndarray::Array3;
use wgpu::util::DeviceExt;

use super::GpuContext;
use crate::dust::{
    connected_components, prepare_dust_heal, DustHealParams, DustHealPrep, DustMask,
};
use crate::dust_wfc::{
    blend_offset_seams, build_candidates, composite_and_grain, exemplar_fill, hole_pass_count,
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    height: u32,
    n: u32,
    n_cand: u32,
    color_gate: f32,
    rim_r: f32,
    rim_g: f32,
    rim_b: f32,
}

pub struct DustWfcPipeline {
    compute_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    ctx: Arc<GpuContext>,
}

impl DustWfcPipeline {
    pub fn new(ctx: &Arc<GpuContext>) -> Self {
        let shader = include_str!("dust_wfc.wgsl");
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("dust_wfc_shader"),
                source: wgpu::ShaderSource::Wgsl(shader.into()),
            });

        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("dust_wfc_bgl"),
                entries: &[
                    bind_uniform(0),
                    bind_storage(1, true),
                    bind_storage(2, true),
                    bind_storage(3, true),
                    bind_storage(4, true),
                    bind_storage(5, false),
                    bind_storage(6, true),
                    bind_storage(7, false),
                    bind_storage(8, true),
                ],
            });

        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("dust_wfc_layout"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let compute_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("dust_wfc_pipeline"),
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

    pub fn run(
        &self,
        image: &mut Array3<f32>,
        mask: &DustMask,
        params: DustHealParams,
    ) -> anyhow::Result<()> {
        let Some(prep) = prepare_dust_heal(image, mask, params) else {
            return Ok(());
        };
        self.run_prep(image, &prep)
    }

    fn run_prep(&self, image: &mut Array3<f32>, prep: &DustHealPrep) -> anyhow::Result<()> {
        let DustHealPrep {
            tight,
            dilated,
            alpha,
            grain,
            tile,
            loosen,
            w,
            h,
        } = prep;
        let w = *w;
        let h = *h;
        let n = (*tile as usize).clamp(2, 5);
        let n_pix = w * h;

        let max_buf = self.ctx.device.limits().max_buffer_size;
        let img_bytes = (n_pix * 3 * 4) as u64;
        let fill_bytes = (n_pix * 4 * 4) as u64;
        if img_bytes + fill_bytes * 2 > max_buf {
            anyhow::bail!("Dust WFC buffers exceed GPU max; falling back to CPU");
        }

        let mut img_flat = Vec::with_capacity(n_pix * 3);
        for y in 0..h {
            for x in 0..w {
                img_flat.push(image[(y, x, 0)]);
                img_flat.push(image[(y, x, 1)]);
                img_flat.push(image[(y, x, 2)]);
            }
        }
        let tight_u: Vec<u32> = tight.iter().map(|&t| u32::from(t)).collect();

        let device = &self.ctx.device;
        let image_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dust_wfc_image"),
            contents: bytemuck::cast_slice(&img_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let tight_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dust_wfc_tight"),
            contents: bytemuck::cast_slice(&tight_u),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let mut fill = vec![None; n_pix];
        for component in connected_components(tight, w, h) {
            if component.is_empty() {
                continue;
            }
            let mut hole = vec![false; n_pix];
            for &i in &component {
                hole[i] = true;
            }
            let (cands, color_gate, rim) = build_candidates(image, tight, &hole, *loosen, w, h);
            if cands.is_empty() {
                let (mut colors, srcs) =
                    exemplar_fill(image, tight, &component, &[], color_gate, rim, n, w, h);
                blend_offset_seams(
                    image, tight, &component, &mut colors, &srcs, color_gate, rim, w, h,
                );
                for (&i, c) in component.iter().zip(colors) {
                    fill[i] = Some(c);
                }
                continue;
            }
            self.fill_component(
                image,
                tight,
                &image_buf,
                &tight_buf,
                &component,
                &cands,
                color_gate,
                rim,
                n,
                w,
                h,
                n_pix,
                &mut fill,
            )?;
        }

        composite_and_grain(image, &fill, dilated, alpha, *grain, w, h);
        Ok(())
    }

    fn fill_component(
        &self,
        image: &Array3<f32>,
        tight: &[bool],
        image_buf: &wgpu::Buffer,
        tight_buf: &wgpu::Buffer,
        component: &[usize],
        cands: &[(u16, u16)],
        color_gate: f32,
        rim: (f32, f32, f32),
        n: usize,
        w: usize,
        h: usize,
        n_pix: usize,
        fill: &mut [Option<(f32, f32, f32)>],
    ) -> anyhow::Result<()> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let mut hole_mask = vec![0u32; n_pix];
        for &i in component {
            hole_mask[i] = 1;
        }
        let mut cand_flat = Vec::with_capacity(cands.len() * 2);
        for &(x, y) in cands {
            cand_flat.push(x as u32);
            cand_flat.push(y as u32);
        }

        let params = Params {
            width: w as u32,
            height: h as u32,
            n: n as u32,
            n_cand: cands.len() as u32,
            color_gate,
            rim_r: rim.0,
            rim_g: rim.1,
            rim_b: rim.2,
        };

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dust_wfc_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let hole_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dust_wfc_hole"),
            contents: bytemuck::cast_slice(&hole_mask),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let cand_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dust_wfc_cands"),
            contents: bytemuck::cast_slice(&cand_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let fill_zeros = vec![0.0f32; n_pix * 4];
        let src_zeros = vec![0u32; n_pix];
        let fill_bytes = (fill_zeros.len() * std::mem::size_of::<f32>()) as u64;
        let fill_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dust_wfc_fill_a"),
            contents: bytemuck::cast_slice(&fill_zeros),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let fill_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dust_wfc_fill_b"),
            contents: bytemuck::cast_slice(&fill_zeros),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let src_bytes = (src_zeros.len() * std::mem::size_of::<u32>()) as u64;
        let src_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dust_wfc_src_a"),
            contents: bytemuck::cast_slice(&src_zeros),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let src_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dust_wfc_src_b"),
            contents: bytemuck::cast_slice(&src_zeros),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let bg_ab = self.bind(
            device, &params_buf, image_buf, tight_buf, &hole_buf, &fill_a, &fill_b, &src_a,
            &src_b, &cand_buf,
        );
        let bg_ba = self.bind(
            device, &params_buf, image_buf, tight_buf, &hole_buf, &fill_b, &fill_a, &src_b,
            &src_a, &cand_buf,
        );

        const MAX_WG: u32 = 65535;
        let total_wg = (n_pix as u32 + 255) / 256;
        let wg_x = total_wg.min(MAX_WG);
        let wg_y = (total_wg + MAX_WG - 1) / MAX_WG;
        let passes = hole_pass_count(component, w, n);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dust_wfc_encoder"),
        });
        for pass in 0..passes {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dust_wfc_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, if pass % 2 == 0 { &bg_ab } else { &bg_ba }, &[]);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        let last_fill = if passes % 2 == 1 { &fill_b } else { &fill_a };
        let last_src = if passes % 2 == 1 { &src_b } else { &src_a };
        let fill_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dust_wfc_fill_staging"),
            size: fill_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let src_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dust_wfc_src_staging"),
            size: src_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(last_fill, 0, &fill_staging, 0, fill_bytes);
        encoder.copy_buffer_to_buffer(last_src, 0, &src_staging, 0, src_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let fill_slice = fill_staging.slice(..);
        let src_slice = src_staging.slice(..);
        let (tx_f, rx_f) = std::sync::mpsc::channel();
        let (tx_s, rx_s) = std::sync::mpsc::channel();
        fill_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx_f.send(r);
        });
        src_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx_s.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        rx_f.recv()
            .map_err(|_| anyhow::anyhow!("GPU map channel disconnected"))?
            .map_err(|e| anyhow::anyhow!("GPU buffer map failed: {:?}", e))?;
        rx_s.recv()
            .map_err(|_| anyhow::anyhow!("GPU map channel disconnected"))?
            .map_err(|e| anyhow::anyhow!("GPU buffer map failed: {:?}", e))?;

        let fill_data = fill_slice.get_mapped_range();
        let src_data = src_slice.get_mapped_range();
        let result: &[f32] = bytemuck::cast_slice(&fill_data);
        let packed: &[u32] = bytemuck::cast_slice(&src_data);
        let mut colors = vec![(0.0f32, 0.0f32, 0.0f32); component.len()];
        let mut srcs = vec![None; component.len()];
        let mut placed = vec![false; component.len()];
        for (k, &i) in component.iter().enumerate() {
            let b = i * 4;
            if result[b + 3] > 0.5 {
                colors[k] = (result[b], result[b + 1], result[b + 2]);
                placed[k] = true;
            }
            let p = packed[i];
            if p != 0 {
                let sx = ((p >> 16) - 1) as u16;
                let sy = ((p & 0xFFFF) - 1) as u16;
                srcs[k] = Some((sx, sy));
            }
        }
        drop(fill_data);
        drop(src_data);
        fill_staging.unmap();
        src_staging.unmap();

        blend_offset_seams(
            image, tight, component, &mut colors, &srcs, color_gate, rim, w, h,
        );
        for (k, &i) in component.iter().enumerate() {
            if placed[k] {
                fill[i] = Some(colors[k]);
            }
        }
        Ok(())
    }

    fn bind(
        &self,
        device: &wgpu::Device,
        params: &wgpu::Buffer,
        image: &wgpu::Buffer,
        tight: &wgpu::Buffer,
        hole: &wgpu::Buffer,
        prev_fill: &wgpu::Buffer,
        next_fill: &wgpu::Buffer,
        prev_src: &wgpu::Buffer,
        next_src: &wgpu::Buffer,
        cands: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dust_wfc_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: image.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tight.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: hole.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: prev_fill.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: next_fill.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: prev_src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: next_src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: cands.as_entire_binding(),
                },
            ],
        })
    }
}

fn bind_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bind_storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
