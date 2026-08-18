//! GPU Jacobi tile pick for Wave-function dust heal.
//! Harvest stays on the CPU; per-pixel score/pick matches `dust_wfc`.

use std::sync::Arc;

use ndarray::Array3;
use wgpu::util::DeviceExt;

use super::GpuContext;
use crate::dust::{
    connected_components, prepare_dust_heal, DustHealParams, DustHealPrep, DustMask,
};
use crate::dust_wfc::{
    build_library, composite_and_grain, flatten_tiles, hole_pass_count, parallel_fill, MEAN_W,
    SCORE_BAND, TAU_PENALTY,
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    height: u32,
    n: u32,
    n_tiles: u32,
    tau: f32,
    mean_w: f32,
    tau_penalty: f32,
    score_band: f32,
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

    /// Fill holes with the same tile pick as the CPU path. Falls back via `Err`.
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
            let (tiles, tau_base) = build_library(image, tight, &hole, n, w, h);
            let tau = (tau_base * loosen).max(1.0e-6);
            if tiles.is_empty() {
                let (colors, _) = parallel_fill(image, tight, &component, &[], tau, n, w, h);
                for (&i, c) in component.iter().zip(colors) {
                    fill[i] = Some(c);
                }
                continue;
            }
            self.fill_component(
                &image_buf,
                &tight_buf,
                &component,
                &tiles,
                tau,
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
        image_buf: &wgpu::Buffer,
        tight_buf: &wgpu::Buffer,
        component: &[usize],
        tiles: &[crate::dust_wfc::Tile],
        tau: f32,
        n: usize,
        w: usize,
        h: usize,
        n_pix: usize,
        fill: &mut [Option<(f32, f32, f32)>],
    ) -> anyhow::Result<()> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let mut active = vec![0u32; n_pix];
        for &i in component {
            active[i] = 1;
        }
        let tile_flat = flatten_tiles(tiles);
        anyhow::ensure!(!tile_flat.is_empty(), "empty tile buffer");

        let params = Params {
            width: w as u32,
            height: h as u32,
            n: n as u32,
            n_tiles: tiles.len() as u32,
            tau,
            mean_w: MEAN_W,
            tau_penalty: TAU_PENALTY,
            score_band: SCORE_BAND,
        };

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dust_wfc_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let active_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dust_wfc_active"),
            contents: bytemuck::cast_slice(&active),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let tiles_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dust_wfc_tiles"),
            contents: bytemuck::cast_slice(&tile_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let fill_zeros = vec![0.0f32; n_pix * 4];
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

        let bg_ab = self.bind(device, &params_buf, image_buf, tight_buf, &active_buf, &fill_a, &fill_b, &tiles_buf);
        let bg_ba = self.bind(device, &params_buf, image_buf, tight_buf, &active_buf, &fill_b, &fill_a, &tiles_buf);

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

        let last = if passes % 2 == 1 { &fill_b } else { &fill_a };
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dust_wfc_staging"),
            size: fill_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(last, 0, &staging, 0, fill_bytes);
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
        for &i in component {
            let b = i * 4;
            if result[b + 3] > 0.5 {
                fill[i] = Some((result[b], result[b + 1], result[b + 2]));
            }
        }
        drop(data);
        staging.unmap();
        let _ = (w, h);
        Ok(())
    }

    fn bind(
        &self,
        device: &wgpu::Device,
        params: &wgpu::Buffer,
        image: &wgpu::Buffer,
        tight: &wgpu::Buffer,
        active: &wgpu::Buffer,
        prev: &wgpu::Buffer,
        next: &wgpu::Buffer,
        tiles: &wgpu::Buffer,
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
                    resource: active.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: prev.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: next.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: tiles.as_entire_binding(),
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
