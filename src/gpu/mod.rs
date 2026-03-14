//! Optional GPU pipeline acceleration via wgpu.
//!
//! Enabled with the `gpu` Cargo feature. Provides step 5 (density matrix / 3D LUT,
//! highlight spread, saturation, zone adjustments) on the GPU, with the same math
//! as the CPU reference in `pipeline.rs` and `density_ops.rs`.

pub mod step5;

use std::sync::Arc;

/// Shared wgpu context: device + queue. Created once, reused across frames.
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

impl GpuContext {
    /// Try to initialize a GPU context. Returns `None` if no suitable adapter is found
    /// (headless, old drivers, etc.). This is the only place wgpu init happens.
    pub fn try_new() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("c41_gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;

        Some(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }
}
