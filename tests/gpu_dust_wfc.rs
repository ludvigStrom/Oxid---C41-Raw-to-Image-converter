//! CPU-vs-GPU comparison for Wave-function dust heal.

#![cfg(feature = "gpu")]

use c41_raw_tool::gpu::unified::GpuPipeline;
use c41_raw_tool::{
    apply_dust_removal_with, DustHealParams, DustInfill, DustMask, DustTool,
};
use ndarray::Array3;

fn grainy_field(w: usize, h: usize) -> Array3<f32> {
    let mut img = Array3::<f32>::from_elem((h, w, 3), 0.42);
    for y in 0..h {
        for x in 0..w {
            let n = ((x.wrapping_mul(13) + y.wrapping_mul(29)) % 11) as f32 * 0.03 - 0.15;
            img[(y, x, 0)] = (0.44 + n).clamp(0.0, 1.0);
            img[(y, x, 1)] = (0.41 + n * 0.8).clamp(0.0, 1.0);
            img[(y, x, 2)] = (0.39 + n * 0.6).clamp(0.0, 1.0);
        }
    }
    img
}

fn wfc_params() -> DustHealParams {
    DustHealParams {
        infill: DustInfill::WaveFunction,
        feather: 1.0,
        grain: 0.0,
        tile: 3,
        match_loosen: 2.5,
        ..DustHealParams::default()
    }
}

#[test]
fn gpu_wfc_matches_cpu() {
    let Some(gpu) = GpuPipeline::try_new() else {
        eprintln!("No GPU; skip");
        return;
    };

    let mut cpu = grainy_field(48, 48);
    cpu[(24, 24, 0)] = 1.0;
    cpu[(24, 24, 1)] = 1.0;
    cpu[(24, 24, 2)] = 1.0;
    let mut gpu_img = cpu.clone();

    let mut mask = DustMask::new(48, 48);
    c41_raw_tool::stamp_disc(&mut mask, 24.0, 24.0, 5.0, DustTool::Pen);

    apply_dust_removal_with(&mut cpu, &mask, wfc_params());
    gpu.dust_wfc
        .run(&mut gpu_img, &mask, wfc_params())
        .expect("GPU WFC");

    let mut max_diff = 0.0f32;
    let mut n_big = 0u32;
    let mut hole_cpu = 0.0f32;
    let mut hole_gpu = 0.0f32;
    let mut hole_n = 0.0f32;
    for y in 0..48 {
        for x in 0..48 {
            let d = (0..3)
                .map(|c| (cpu[(y, x, c)] - gpu_img[(y, x, c)]).abs())
                .fold(0.0f32, f32::max);
            max_diff = max_diff.max(d);
            if mask.data[y * 48 + x] >= 16 {
                hole_cpu += cpu[(y, x, 0)];
                hole_gpu += gpu_img[(y, x, 0)];
                hole_n += 1.0;
                if d > 0.01 {
                    n_big += 1;
                }
            } else {
                assert!(
                    d < 1e-5,
                    "pixel outside paint changed ({x},{y} diff={d})"
                );
            }
        }
    }
    eprintln!("  dust WFC max diff: {max_diff:.8} big={n_big} hole_mean cpu={} gpu={}", hole_cpu / hole_n, hole_gpu / hole_n);
    assert!(cpu[(24, 24, 0)] < 0.85, "CPU must heal speck");
    assert!(gpu_img[(24, 24, 0)] < 0.85, "GPU must heal speck");
    assert!(
        (hole_cpu / hole_n - hole_gpu / hole_n).abs() < 0.03,
        "hole mean drifted between CPU and GPU"
    );
    assert!(
        max_diff <= 0.08,
        "CPU/GPU WFC mismatch (max_diff={max_diff})"
    );
}
