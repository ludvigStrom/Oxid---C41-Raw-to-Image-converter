//! CPU-vs-GPU comparison tests for step 3 (flat-field divide and D-min divide).

#![cfg(feature = "gpu")]

use std::sync::Arc;

use c41_raw_tool::gpu::{flat_field::FlatFieldPipeline, step3_dmin::Step3DminPipeline, GpuContext};
use c41_raw_tool::pipeline::{step_3_dmin, step_3_dmin_gpu};
use c41_raw_tool::PipelineOptions;
use ndarray::Array3;

fn make_image(width: usize, height: usize) -> Array3<f32> {
    let mut img = Array3::<f32>::zeros((height, width, 3));
    for y in 0..height {
        for x in 0..width {
            let u = x as f32 / width as f32;
            let v = y as f32 / height as f32;
            img[[y, x, 0]] = (u * 0.8 + 0.1).clamp(0.01, 1.0);
            img[[y, x, 1]] = (v * 0.7 + 0.15).clamp(0.01, 1.0);
            img[[y, x, 2]] = (((u + v) * 0.5) * 0.6 + 0.2).clamp(0.01, 1.0);
        }
    }
    img
}

fn compare_step3(cpu: &Array3<f32>, gpu: &Array3<f32>, tol: f32) {
    assert_eq!(cpu.dim(), gpu.dim());
    let mut max_diff = 0.0f32;
    for (a, b) in cpu.iter().zip(gpu.iter()) {
        max_diff = max_diff.max((a - b).abs());
    }
    eprintln!("  step3 max diff: {:.8}", max_diff);
    assert!(max_diff <= tol, "max_diff {} > tol {}", max_diff, tol);
}

#[test]
fn step3_dmin_fixed() {
    let ctx = match GpuContext::try_new() {
        Some(c) => Arc::new(c),
        None => {
            eprintln!("No GPU; skip");
            return;
        }
    };
    let step3 = Step3DminPipeline::new(&ctx);

    let mut opts = PipelineOptions::default();
    opts.debug_pipeline_step = 3;
    opts.dmin_mode = c41_raw_tool::DminMode::Fixed;
    opts.dmin_fixed = Some((0.5, 0.6, 0.4));

    let mut cpu_img = make_image(120, 80);
    let mut gpu_img = cpu_img.clone();

    step_3_dmin(&mut cpu_img, &opts, None).unwrap();

    let gpu_step3 = c41_raw_tool::gpu::unified::Step3Gpu {
        flat_field: FlatFieldPipeline::new(&ctx),
        step3_dmin: step3,
    };
    step_3_dmin_gpu(&mut gpu_img, &opts, None, Some(&gpu_step3)).unwrap();

    compare_step3(&cpu_img, &gpu_img, 1e-5);
}

#[test]
fn step3_dmin_sample_region() {
    let ctx = match GpuContext::try_new() {
        Some(c) => Arc::new(c),
        None => {
            eprintln!("No GPU; skip");
            return;
        }
    };
    let step3 = Step3DminPipeline::new(&ctx);

    let mut opts = PipelineOptions::default();
    opts.debug_pipeline_step = 3;
    opts.dmin_mode = c41_raw_tool::DminMode::SampleRegion;
    opts.dmin_rect = Some(c41_raw_tool::Rect {
        x: 0,
        y: 0,
        width: 40,
        height: 30,
    });
    opts.dmin_neutral_only = false;

    let mut cpu_img = make_image(120, 80);
    let mut gpu_img = cpu_img.clone();

    step_3_dmin(&mut cpu_img, &opts, None).unwrap();

    let gpu_step3 = c41_raw_tool::gpu::unified::Step3Gpu {
        flat_field: FlatFieldPipeline::new(&ctx),
        step3_dmin: step3,
    };
    step_3_dmin_gpu(&mut gpu_img, &opts, None, Some(&gpu_step3)).unwrap();

    compare_step3(&cpu_img, &gpu_img, 1e-5);
}

#[test]
fn step3_dmin_auto_percentile() {
    let ctx = match GpuContext::try_new() {
        Some(c) => Arc::new(c),
        None => {
            eprintln!("No GPU; skip");
            return;
        }
    };
    let step3 = Step3DminPipeline::new(&ctx);

    let mut opts = PipelineOptions::default();
    opts.debug_pipeline_step = 3;
    opts.dmin_mode = c41_raw_tool::DminMode::AutoPercentile;
    opts.auto_norm_buffer = 0.05;

    let mut cpu_img = make_image(120, 80);
    let mut gpu_img = cpu_img.clone();

    step_3_dmin(&mut cpu_img, &opts, None).unwrap();

    let gpu_step3 = c41_raw_tool::gpu::unified::Step3Gpu {
        flat_field: FlatFieldPipeline::new(&ctx),
        step3_dmin: step3,
    };
    step_3_dmin_gpu(&mut gpu_img, &opts, None, Some(&gpu_step3)).unwrap();

    compare_step3(&cpu_img, &gpu_img, 1e-5);
}

#[test]
fn step3_flat_field() {
    let ctx = match GpuContext::try_new() {
        Some(c) => Arc::new(c),
        None => {
            eprintln!("No GPU; skip");
            return;
        }
    };

    let mut opts = PipelineOptions::default();
    opts.debug_pipeline_step = 3;
    opts.dmin_mode = c41_raw_tool::DminMode::Fixed;
    opts.dmin_fixed = Some((1.0, 1.0, 1.0));
    let flat = make_image(120, 80).mapv(|v| v * 0.5 + 0.3);

    let mut cpu_img = make_image(120, 80);
    let mut gpu_img = cpu_img.clone();

    step_3_dmin(&mut cpu_img, &opts, Some(&flat)).unwrap();

    let gpu_step3 = c41_raw_tool::gpu::unified::Step3Gpu {
        flat_field: FlatFieldPipeline::new(&ctx),
        step3_dmin: Step3DminPipeline::new(&ctx),
    };
    step_3_dmin_gpu(&mut gpu_img, &opts, Some(&flat), Some(&gpu_step3)).unwrap();

    compare_step3(&cpu_img, &gpu_img, 1e-5);
}
