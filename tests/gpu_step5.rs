//! CPU-vs-GPU comparison test for pipeline step 5.
//!
//! Generates a synthetic density image, runs step 5 on both CPU and GPU,
//! and asserts the results match within 1e-5.

#![cfg(feature = "gpu")]

use std::sync::Arc;

use c41_raw_tool::gpu::{GpuContext, step5::Step5Pipeline};
use c41_raw_tool::lut3d::Lut3d;
use c41_raw_tool::{pipeline, PipelineOptions};
use ndarray::Array3;

fn make_test_image(width: usize, height: usize) -> Array3<f32> {
    let mut img = Array3::<f32>::zeros((height, width, 3));
    for y in 0..height {
        for x in 0..width {
            let u = x as f32 / width as f32;
            let v = y as f32 / height as f32;
            img[[y, x, 0]] = u * 2.5;
            img[[y, x, 1]] = v * 2.5;
            img[[y, x, 2]] = ((u + v) * 0.5) * 2.5;
        }
    }
    img
}

fn compare_images(cpu: &Array3<f32>, gpu: &Array3<f32>, tolerance: f32) {
    let (h, w, c) = cpu.dim();
    assert_eq!(cpu.dim(), gpu.dim(), "Dimension mismatch");
    let mut max_diff: f32 = 0.0;
    let mut max_diff_coords = (0, 0, 0);
    let mut diffs_above_tol = 0u64;

    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let diff = (cpu[[y, x, ch]] - gpu[[y, x, ch]]).abs();
                if diff > max_diff {
                    max_diff = diff;
                    max_diff_coords = (y, x, ch);
                }
                if diff > tolerance {
                    diffs_above_tol += 1;
                }
            }
        }
    }

    if diffs_above_tol > 0 || max_diff > tolerance {
        panic!(
            "CPU vs GPU mismatch: max_diff={:.8} at ({},{},ch={}), {} pixels above tolerance {}. \
             CPU={:.8}, GPU={:.8}",
            max_diff,
            max_diff_coords.0,
            max_diff_coords.1,
            max_diff_coords.2,
            diffs_above_tol,
            tolerance,
            cpu[[max_diff_coords.0, max_diff_coords.1, max_diff_coords.2]],
            gpu[[max_diff_coords.0, max_diff_coords.1, max_diff_coords.2]],
        );
    }
    eprintln!("  max diff: {:.8}", max_diff);
}

fn get_gpu_context() -> Option<Arc<GpuContext>> {
    GpuContext::try_new().map(Arc::new)
}

#[test]
fn step5_matrix_cpu_vs_gpu() {
    let ctx = match get_gpu_context() {
        Some(c) => c,
        None => {
            eprintln!("No GPU adapter found; skipping test");
            return;
        }
    };

    let pipeline_gpu = Step5Pipeline::new(&ctx);

    let mut opts = PipelineOptions::default();
    opts.apply_color_profile = true;
    opts.density_matrix = [
        [1.2, -0.1, -0.05],
        [-0.08, 1.1, -0.02],
        [-0.03, -0.07, 1.15],
    ];
    opts.saturation = 1.3;
    opts.zone_shadows = 0.1;
    opts.zone_highlights = -0.05;
    opts.color_shadows_r = 0.02;
    opts.color_mids_g = -0.01;
    opts.color_highlights_b = 0.03;

    let img = make_test_image(128, 96);

    let mut cpu_img = img.clone();
    pipeline::step_5_calibration(&mut cpu_img, &opts, None);

    let mut gpu_img = img.clone();
    pipeline_gpu.run(&mut gpu_img, &opts, None).expect("GPU step 5 failed");

    eprintln!("step5_matrix_cpu_vs_gpu:");
    compare_images(&cpu_img, &gpu_img, 1e-5);
}

#[test]
fn step5_lut_cpu_vs_gpu() {
    let ctx = match get_gpu_context() {
        Some(c) => c,
        None => {
            eprintln!("No GPU adapter found; skipping test");
            return;
        }
    };

    let pipeline_gpu = Step5Pipeline::new(&ctx);

    let matrix = [
        [1.15, -0.1, -0.05],
        [-0.05, 1.1, -0.05],
        [-0.02, -0.08, 1.1],
    ];
    let lut = Lut3d::generate_from_matrix(&matrix, 17, 4.0);

    let mut opts = PipelineOptions::default();
    opts.apply_color_profile = true;
    opts.saturation = 1.1;
    opts.zone_shadows = 0.05;

    let img = make_test_image(100, 80);

    let mut cpu_img = img.clone();
    pipeline::step_5_calibration(&mut cpu_img, &opts, Some(&lut));

    let mut gpu_img = img.clone();
    pipeline_gpu.run(&mut gpu_img, &opts, Some(&lut)).expect("GPU step 5 with LUT failed");

    eprintln!("step5_lut_cpu_vs_gpu:");
    compare_images(&cpu_img, &gpu_img, 1e-5);
}

#[test]
fn step5_identity_cpu_vs_gpu() {
    let ctx = match get_gpu_context() {
        Some(c) => c,
        None => {
            eprintln!("No GPU adapter found; skipping test");
            return;
        }
    };

    let pipeline_gpu = Step5Pipeline::new(&ctx);

    let opts = PipelineOptions::default();
    let img = make_test_image(64, 48);

    let mut cpu_img = img.clone();
    pipeline::step_5_calibration(&mut cpu_img, &opts, None);

    let mut gpu_img = img.clone();
    pipeline_gpu.run(&mut gpu_img, &opts, None).expect("GPU step 5 identity failed");

    eprintln!("step5_identity_cpu_vs_gpu:");
    compare_images(&cpu_img, &gpu_img, 1e-5);
}
