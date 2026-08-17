//! CPU-vs-GPU comparison tests for pipeline step 4 (T→D, WB, shadow cast).

#![cfg(feature = "gpu")]

use std::sync::Arc;

use c41_raw_tool::gpu::{step4::Step4Pipeline, GpuContext};
use c41_raw_tool::{pipeline, PipelineOptions};
use ndarray::Array3;

fn make_transmittance_image(width: usize, height: usize) -> Array3<f32> {
    let mut img = Array3::<f32>::zeros((height, width, 3));
    for y in 0..height {
        for x in 0..width {
            let u = x as f32 / width as f32;
            let v = y as f32 / height as f32;
            img[[y, x, 0]] = (u * 0.9 + 0.05).clamp(0.001, 1.0);
            img[[y, x, 1]] = (v * 0.85 + 0.08).clamp(0.001, 1.0);
            img[[y, x, 2]] = (((u + v) * 0.5) * 0.8 + 0.1).clamp(0.001, 1.0);
        }
    }
    img
}

fn compare_images(cpu: &Array3<f32>, gpu: &Array3<f32>, tolerance: f32, label: &str) {
    assert_eq!(cpu.dim(), gpu.dim(), "{}: dim mismatch", label);
    let (h, w, c) = cpu.dim();
    let mut max_diff: f32 = 0.0;
    let mut max_coords = (0, 0, 0);
    let mut above = 0u64;
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let diff = (cpu[[y, x, ch]] - gpu[[y, x, ch]]).abs();
                if diff > max_diff {
                    max_diff = diff;
                    max_coords = (y, x, ch);
                }
                if diff > tolerance {
                    above += 1;
                }
            }
        }
    }
    eprintln!(
        "  {}: max_diff={:.8} at ({},{},ch={}), {} above tol {:.6}",
        label, max_diff, max_coords.0, max_coords.1, max_coords.2, above, tolerance
    );
    if above > 0 {
        panic!(
            "{}: {} pixels exceed {:.6} tolerance (max_diff={:.8}). CPU={:.8}, GPU={:.8}",
            label,
            above,
            tolerance,
            max_diff,
            cpu[[max_coords.0, max_coords.1, max_coords.2]],
            gpu[[max_coords.0, max_coords.1, max_coords.2]],
        );
    }
}

fn get_gpu() -> Option<(Arc<GpuContext>, Step4Pipeline)> {
    let ctx = Arc::new(GpuContext::try_new()?);
    let pipeline = Step4Pipeline::new(&ctx);
    Some((ctx, pipeline))
}

#[test]
fn step4_basic_t_to_d() {
    let (_ctx, gpu_pipe) = match get_gpu() {
        Some(v) => v,
        None => {
            eprintln!("No GPU; skip");
            return;
        }
    };
    let img = make_transmittance_image(120, 80);
    let mut opts = PipelineOptions::default();
    opts.auto_wb = false;
    opts.apply_white_balance = false;
    opts.film_gamma = 1.0;
    opts.shadow_cast_strength = 0.0;

    let mut cpu_img = img.clone();
    pipeline::step_4_t_to_d_wb(&mut cpu_img, &opts);

    let mut gpu_img = img.clone();
    gpu_pipe.run(&mut gpu_img, &opts).expect("GPU step4 basic");

    compare_images(&cpu_img, &gpu_img, 1e-5, "basic T→D");
}

#[test]
fn step4_manual_wb() {
    let (_ctx, gpu_pipe) = match get_gpu() {
        Some(v) => v,
        None => {
            eprintln!("No GPU; skip");
            return;
        }
    };
    let img = make_transmittance_image(100, 80);
    let mut opts = PipelineOptions::default();
    opts.auto_wb = false;
    opts.apply_white_balance = true;
    opts.wb_r = 1.1;
    opts.wb_g = 0.95;
    opts.wb_b = 1.05;
    opts.film_gamma = 0.65;
    opts.shadow_cast_strength = 0.0;

    let mut cpu_img = img.clone();
    pipeline::step_4_t_to_d_wb(&mut cpu_img, &opts);

    let mut gpu_img = img.clone();
    gpu_pipe
        .run(&mut gpu_img, &opts)
        .expect("GPU step4 manual WB");

    compare_images(&cpu_img, &gpu_img, 1e-5, "manual WB + film gamma");
}

#[test]
fn step4_auto_wb() {
    let (_ctx, gpu_pipe) = match get_gpu() {
        Some(v) => v,
        None => {
            eprintln!("No GPU; skip");
            return;
        }
    };
    let img = make_transmittance_image(100, 80);
    let mut opts = PipelineOptions::default();
    opts.auto_wb = true;
    opts.apply_white_balance = true;
    opts.wb_r = 1.0;
    opts.wb_g = 1.0;
    opts.wb_b = 1.0;
    opts.film_gamma = 0.65;
    opts.shadow_cast_strength = 0.0;

    let mut cpu_img = img.clone();
    pipeline::step_4_t_to_d_wb(&mut cpu_img, &opts);

    let mut gpu_img = img.clone();
    gpu_pipe
        .run(&mut gpu_img, &opts)
        .expect("GPU step4 auto WB");

    compare_images(&cpu_img, &gpu_img, 1e-5, "auto WB");
}

#[test]
fn step4_temp_k() {
    let (_ctx, gpu_pipe) = match get_gpu() {
        Some(v) => v,
        None => {
            eprintln!("No GPU; skip");
            return;
        }
    };
    let img = make_transmittance_image(80, 60);
    let mut opts = PipelineOptions::default();
    opts.auto_wb = false;
    opts.apply_white_balance = true;
    opts.film_gamma = 0.65;
    opts.temp_k = Some(4500.0);
    opts.shadow_cast_strength = 0.0;

    let mut cpu_img = img.clone();
    pipeline::step_4_t_to_d_wb(&mut cpu_img, &opts);

    let mut gpu_img = img.clone();
    gpu_pipe.run(&mut gpu_img, &opts).expect("GPU step4 temp_k");

    compare_images(&cpu_img, &gpu_img, 1e-5, "temp_k");
}

#[test]
fn step4_shadow_cast() {
    let (_ctx, gpu_pipe) = match get_gpu() {
        Some(v) => v,
        None => {
            eprintln!("No GPU; skip");
            return;
        }
    };
    let img = make_transmittance_image(100, 80);
    let mut opts = PipelineOptions::default();
    opts.auto_wb = true;
    opts.apply_white_balance = true;
    opts.film_gamma = 0.65;
    opts.shadow_cast_strength = 0.8;

    let mut cpu_img = img.clone();
    pipeline::step_4_t_to_d_wb(&mut cpu_img, &opts);

    let mut gpu_img = img.clone();
    gpu_pipe
        .run(&mut gpu_img, &opts)
        .expect("GPU step4 shadow cast");

    compare_images(&cpu_img, &gpu_img, 1e-5, "shadow cast");
}

#[test]
fn step4_full_options() {
    let (_ctx, gpu_pipe) = match get_gpu() {
        Some(v) => v,
        None => {
            eprintln!("No GPU; skip");
            return;
        }
    };
    let img = make_transmittance_image(120, 80);
    let mut opts = PipelineOptions::default();
    opts.auto_wb = true;
    opts.apply_white_balance = true;
    opts.wb_r = 1.05;
    opts.wb_g = 0.98;
    opts.wb_b = 1.02;
    opts.film_gamma = 0.60;
    opts.temp_k = Some(5500.0);
    opts.shadow_cast_strength = 0.6;

    let mut cpu_img = img.clone();
    pipeline::step_4_t_to_d_wb(&mut cpu_img, &opts);

    let mut gpu_img = img.clone();
    gpu_pipe.run(&mut gpu_img, &opts).expect("GPU step4 full");

    compare_images(&cpu_img, &gpu_img, 1e-5, "full options");
}
