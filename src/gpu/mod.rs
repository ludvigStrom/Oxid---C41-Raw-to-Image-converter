//! Optional GPU pipeline acceleration via wgpu.
//!
//! Enabled with the `gpu` Cargo feature. Provides demosaic (RGGB quality),
//! Wave-function dust heal, and steps 4–6 (T→D, calibration, render) on the GPU,
//! with the same math as the CPU reference.

pub mod demosaic;
pub mod dust_wfc;
pub mod flat_field;
pub mod step3_dmin;
pub mod step4;
pub mod step5;
pub mod step6;
pub mod unified;

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

        // Request larger buffer limits for high-zoom previews. Fall back to
        // default limits if the adapter doesn't support them.
        let mut limits = wgpu::Limits::default();
        let adapter_limits = adapter.limits();
        limits.max_buffer_size = adapter_limits.max_buffer_size.max(limits.max_buffer_size);
        limits.max_storage_buffer_binding_size = adapter_limits
            .max_storage_buffer_binding_size
            .max(limits.max_storage_buffer_binding_size);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("c41_gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
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
