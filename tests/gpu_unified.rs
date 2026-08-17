//! CPU-vs-GPU comparison tests for the unified pipeline (steps 4→5→6).

#![cfg(feature = "gpu")]

use c41_raw_tool::curve::PrintCurveParams;
use c41_raw_tool::gpu::unified::GpuPipeline;
use c41_raw_tool::pipeline::{self, Step6Display};
use c41_raw_tool::{OutputStage, PipelineOptions};
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

fn compare_u16(cpu: &Step6Display, gpu: &Step6Display, tolerance_lsb: u16, label: &str) {
    let (cpu_img, gpu_img) = match (cpu, gpu) {
        (Step6Display::U16(c), Step6Display::U16(g)) => (c, g),
        _ => panic!("{}: expected U16 output from both paths", label),
    };
    assert_eq!(cpu_img.dim(), gpu_img.dim(), "{}: dim mismatch", label);
    let (h, w, c) = cpu_img.dim();
    let mut max_diff: u16 = 0;
    let mut max_coords = (0, 0, 0);
    let mut above = 0u64;
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let cv = cpu_img[[y, x, ch]];
                let gv = gpu_img[[y, x, ch]];
                let diff = if cv >= gv { cv - gv } else { gv - cv };
                if diff > max_diff {
                    max_diff = diff;
                    max_coords = (y, x, ch);
                }
                if diff > tolerance_lsb {
                    above += 1;
                }
            }
        }
    }
    eprintln!(
        "  {}: max_diff={} LSB at ({},{},ch={}), {} above tol {}",
        label, max_diff, max_coords.0, max_coords.1, max_coords.2, above, tolerance_lsb
    );
    if above > 0 {
        panic!(
            "{}: {} pixels exceed {} LSB tolerance (max={}). CPU={}, GPU={}",
            label,
            above,
            tolerance_lsb,
            max_diff,
            cpu_img[[max_coords.0, max_coords.1, max_coords.2]],
            gpu_img[[max_coords.0, max_coords.1, max_coords.2]],
        );
    }
}

fn compare_f32(cpu: &Step6Display, gpu: &Step6Display, tolerance: f32, label: &str) {
    let (cpu_img, gpu_img) = match (cpu, gpu) {
        (Step6Display::F32(c), Step6Display::F32(g)) => (c, g),
        _ => panic!("{}: expected F32 output from both paths", label),
    };
    assert_eq!(cpu_img.dim(), gpu_img.dim(), "{}: dim mismatch", label);
    let (h, w, c) = cpu_img.dim();
    let mut max_diff: f32 = 0.0;
    let mut max_coords = (0, 0, 0);
    let mut above = 0u64;
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let diff = (cpu_img[[y, x, ch]] - gpu_img[[y, x, ch]]).abs();
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
        "  {}: max_diff={:.8} at ({},{},ch={}), {} above tol {}",
        label, max_diff, max_coords.0, max_coords.1, max_coords.2, above, tolerance
    );
    if above > 0 {
        panic!(
            "{}: {} pixels exceed {:.6} tolerance (max={:.8}). CPU={:.8}, GPU={:.8}",
            label,
            above,
            tolerance,
            max_diff,
            cpu_img[[max_coords.0, max_coords.1, max_coords.2]],
            gpu_img[[max_coords.0, max_coords.1, max_coords.2]],
        );
    }
}

/// Run the full CPU pipeline (steps 4→5→6) on a transmittance image.
fn run_cpu_4_5_6(image: &Array3<f32>, options: &PipelineOptions) -> Step6Display {
    let mut img = image.clone();
    pipeline::step_4_t_to_d_wb(&mut img, options);
    let lut3d = options
        .lut3d_path
        .as_ref()
        .and_then(|p| c41_raw_tool::lut3d::read_cube(p).ok());
    pipeline::step_5_calibration(&mut img, options, lut3d.as_ref());
    let ra4 = PrintCurveParams {
        offset: options.curve_offset,
        gamma: options.curve_gamma,
        pivot: options.curve_pivot,
    };
    let output_lut = options
        .output_lut_cube
        .as_ref()
        .and_then(|p| c41_raw_tool::lut3d::read_cube(p).ok());
    pipeline::step_6_render(&img, options, &ra4, output_lut.as_ref())
}

#[test]
fn unified_ra4_from_step4() {
    let gpu = match GpuPipeline::try_new() {
        Some(g) => g,
        None => {
            eprintln!("No GPU; skip");
            return;
        }
    };
    let img = make_transmittance_image(120, 80);
    let opts = PipelineOptions::default();
    let ra4 = PrintCurveParams {
        offset: opts.curve_offset,
        gamma: opts.curve_gamma,
        pivot: opts.curve_pivot,
    };

    let cpu_result = run_cpu_4_5_6(&img, &opts);
    let gpu_result = gpu
        .run_from_step(&img, 4, &opts, None, &ra4, None)
        .expect("unified Ra4 from step4");
    // End-to-end: step4 precision (~2e-7) compounds through step5→step6 LUT lookup.
    compare_u16(&cpu_result, &gpu_result, 4, "unified Ra4 from step4");
}

#[test]
fn unified_ra4_full_ops() {
    let gpu = match GpuPipeline::try_new() {
        Some(g) => g,
        None => {
            eprintln!("No GPU; skip");
            return;
        }
    };
    let img = make_transmittance_image(100, 80);
    let mut opts = PipelineOptions::default();
    opts.wb_r = 1.05;
    opts.wb_g = 0.98;
    opts.wb_b = 1.02;
    opts.film_gamma = 0.60;
    opts.shadow_cast_strength = 0.5;
    opts.saturation = 1.3;
    opts.zone_shadows = 0.1;
    opts.toe_strength = 0.2;
    opts.soft_clip = 0.88;
    opts.highlight_warmth = 0.3;

    let ra4 = PrintCurveParams {
        offset: opts.curve_offset,
        gamma: opts.curve_gamma,
        pivot: opts.curve_pivot,
    };

    let cpu_result = run_cpu_4_5_6(&img, &opts);
    let gpu_result = gpu
        .run_from_step(&img, 4, &opts, None, &ra4, None)
        .expect("unified Ra4 full ops");
    // End-to-end with shadow cast + saturation + zones + post-curve ops.
    compare_u16(&cpu_result, &gpu_result, 10, "unified Ra4 full ops");
}

#[test]
fn unified_none_from_step4() {
    let gpu = match GpuPipeline::try_new() {
        Some(g) => g,
        None => {
            eprintln!("No GPU; skip");
            return;
        }
    };
    let img = make_transmittance_image(80, 60);
    let mut opts = PipelineOptions::default();
    opts.output_stage = OutputStage::None;

    let ra4 = PrintCurveParams {
        offset: opts.curve_offset,
        gamma: opts.curve_gamma,
        pivot: opts.curve_pivot,
    };

    let cpu_result = run_cpu_4_5_6(&img, &opts);
    let gpu_result = gpu
        .run_from_step(&img, 4, &opts, None, &ra4, None)
        .expect("unified None from step4");
    compare_f32(&cpu_result, &gpu_result, 1e-5, "unified None from step4");
}

#[test]
fn unified_from_step5() {
    let gpu = match GpuPipeline::try_new() {
        Some(g) => g,
        None => {
            eprintln!("No GPU; skip");
            return;
        }
    };
    // Make a density image (as if step 4 already ran)
    let mut img = make_transmittance_image(100, 80);
    let opts = PipelineOptions::default();
    pipeline::step_4_t_to_d_wb(&mut img, &opts);

    let ra4 = PrintCurveParams {
        offset: opts.curve_offset,
        gamma: opts.curve_gamma,
        pivot: opts.curve_pivot,
    };

    // CPU: steps 5→6
    let mut cpu_img = img.clone();
    pipeline::step_5_calibration(&mut cpu_img, &opts, None);
    let cpu_result = pipeline::step_6_render(&cpu_img, &opts, &ra4, None);

    // GPU: unified from step 5
    let gpu_result = gpu
        .run_from_step(&img, 5, &opts, None, &ra4, None)
        .expect("unified from step5");
    compare_u16(&cpu_result, &gpu_result, 2, "unified from step5");
}

#[test]
fn unified_from_step6() {
    let gpu = match GpuPipeline::try_new() {
        Some(g) => g,
        None => {
            eprintln!("No GPU; skip");
            return;
        }
    };
    // Make a calibrated density image (as if steps 4+5 already ran)
    let mut img = make_transmittance_image(80, 60);
    let opts = PipelineOptions::default();
    pipeline::step_4_t_to_d_wb(&mut img, &opts);
    pipeline::step_5_calibration(&mut img, &opts, None);

    let ra4 = PrintCurveParams {
        offset: opts.curve_offset,
        gamma: opts.curve_gamma,
        pivot: opts.curve_pivot,
    };

    let cpu_result = pipeline::step_6_render(&img, &opts, &ra4, None);
    let gpu_result = gpu
        .run_from_step(&img, 6, &opts, None, &ra4, None)
        .expect("unified from step6");
    compare_u16(&cpu_result, &gpu_result, 2, "unified from step6");
}

#[test]
fn gpu_preview_populates_step4_and_step5_cache() {
    let gpu = match GpuPipeline::try_new() {
        Some(g) => g,
        None => {
            eprintln!("No GPU; skip");
            return;
        }
    };

    let dir = std::env::temp_dir();
    let path = dir.join("c41_live_slider_cache_test.png");
    let mut img = image::RgbImage::new(16, 12);
    for (x, y, p) in img.enumerate_pixels_mut() {
        *p = image::Rgb([(x * 12) as u8, (y * 16) as u8, 80]);
    }
    img.save(&path).expect("write temp png");

    let mut opts = PipelineOptions::default();
    opts.use_gpu = true;
    let (_, _, _, _, _, _, cache) = c41_raw_tool::process_one_to_preview_with_cache_gpu(
        &path,
        &opts,
        16,
        12,
        None,
        false,
        Some(&gpu),
        None,
    )
    .expect("gpu preview");
    assert!(cache.after_step3.is_some(), "step 3 cache");
    assert!(cache.after_step4.is_some(), "step 4 cache");
    assert!(cache.after_step5.is_some(), "step 5 cache");

    opts.curve_offset = 0.2;
    let live = c41_raw_tool::apply_preview_from_cache_gpu(&path, &opts, 16, 12, &cache, Some(&gpu));
    assert!(live.is_some(), "live apply from step 6");
    let _ = std::fs::remove_file(&path);
}
