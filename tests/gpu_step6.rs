//! CPU-vs-GPU comparison tests for pipeline step 6 (all output stages).

#![cfg(feature = "gpu")]

use std::sync::Arc;

use c41_raw_tool::curve::PrintCurveParams;
use c41_raw_tool::gpu::{step6::Step6Pipeline, GpuContext};
use c41_raw_tool::lut3d::Lut3d;
use c41_raw_tool::pipeline::{self, Step6Display};
use c41_raw_tool::{OutputLutEncoding, OutputStage, PipelineOptions};
use ndarray::Array3;

fn make_density_image(width: usize, height: usize) -> Array3<f32> {
    let mut img = Array3::<f32>::zeros((height, width, 3));
    for y in 0..height {
        for x in 0..width {
            let u = x as f32 / width as f32;
            let v = y as f32 / height as f32;
            img[[y, x, 0]] = u * 3.5;
            img[[y, x, 1]] = v * 3.5;
            img[[y, x, 2]] = ((u + v) * 0.5) * 3.5;
        }
    }
    img
}

fn make_display_lut(size: usize) -> Lut3d {
    let m = [
        [1.1, -0.05, -0.05],
        [-0.05, 1.1, -0.05],
        [-0.05, -0.05, 1.1],
    ];
    Lut3d::generate_from_matrix(&m, size, 1.0)
}

fn get_gpu() -> Option<(Arc<GpuContext>, Step6Pipeline)> {
    let ctx = Arc::new(GpuContext::try_new()?);
    let pipeline = Step6Pipeline::new(&ctx);
    Some((ctx, pipeline))
}

fn compare_u16(
    cpu: &Step6Display,
    gpu: &Step6Display,
    tolerance_lsb: u16,
    label: &str,
) {
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
            "{}: {} pixels exceed {} LSB tolerance (max_diff={}). CPU={}, GPU={}",
            label,
            above,
            tolerance_lsb,
            max_diff,
            cpu_img[[max_coords.0, max_coords.1, max_coords.2]],
            gpu_img[[max_coords.0, max_coords.1, max_coords.2]],
        );
    }
}

fn compare_f32(
    cpu: &Step6Display,
    gpu: &Step6Display,
    tolerance: f32,
    label: &str,
) {
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
            "{}: {} pixels exceed {:.6} tolerance (max_diff={:.8}). CPU={:.8}, GPU={:.8}",
            label,
            above,
            tolerance,
            max_diff,
            cpu_img[[max_coords.0, max_coords.1, max_coords.2]],
            gpu_img[[max_coords.0, max_coords.1, max_coords.2]],
        );
    }
}

// ─── RA-4 ───

#[test]
fn step6_ra4_basic() {
    let (_ctx, gpu_pipe) = match get_gpu() {
        Some(v) => v,
        None => { eprintln!("No GPU; skip"); return; }
    };
    let img = make_density_image(120, 80);
    let opts = PipelineOptions::default();
    let ra4 = PrintCurveParams {
        offset: opts.curve_offset,
        gamma: opts.curve_gamma,
        pivot: opts.curve_pivot,
    };

    let cpu = pipeline::step_6_render(&img, &opts, &ra4, None);
    let gpu = gpu_pipe.run(&img, &opts, &ra4, None).expect("GPU step6 Ra4 basic");
    compare_u16(&cpu, &gpu, 2, "Ra4 basic");
}

#[test]
fn step6_ra4_full_ops() {
    let (_ctx, gpu_pipe) = match get_gpu() {
        Some(v) => v,
        None => { eprintln!("No GPU; skip"); return; }
    };
    let img = make_density_image(100, 80);
    let mut opts = PipelineOptions::default();
    opts.lut_in_black = 0.02;
    opts.lut_in_white = 0.95;
    opts.lut_in_mid = 1.2;
    opts.curve_white = 1.05;
    opts.toe_strength = 0.3;
    opts.shoulder_strength = -0.2;
    opts.soft_clip = 0.88;
    opts.highlight_warmth = 0.4;
    opts.apply_lab = true;
    opts.lab_separation = 0.5;
    opts.skin_magenta_shift = 0.5;

    let ra4 = PrintCurveParams {
        offset: opts.curve_offset,
        gamma: opts.curve_gamma,
        pivot: opts.curve_pivot,
    };

    let cpu = pipeline::step_6_render(&img, &opts, &ra4, None);
    let gpu = gpu_pipe.run(&img, &opts, &ra4, None).expect("GPU step6 Ra4 full");
    // Lab separation and skin magenta shift use pow()/cbrt/trig with GPU precision, plus
    // the GPU avoids intermediate u16 quantization that CPU does between each
    // post-curve op. 10 LSB = 0.015% — imperceptible.
    compare_u16(&cpu, &gpu, 10, "Ra4 full ops");
}

// ─── FilmPrint ───

#[test]
fn step6_film_print_basic() {
    let (_ctx, gpu_pipe) = match get_gpu() {
        Some(v) => v,
        None => { eprintln!("No GPU; skip"); return; }
    };
    let img = make_density_image(100, 80);
    let mut opts = PipelineOptions::default();
    opts.output_stage = OutputStage::FilmPrint;
    opts.fp_offset_r = 0.05;
    opts.fp_offset_g = -0.03;
    opts.fp_offset_b = 0.02;
    opts.fp_gamma_r = 1.1;
    opts.fp_gamma_g = 0.95;
    opts.fp_gamma_b = 1.05;
    opts.fp_color_bleed = 0.12;
    opts.fp_vibrance = 0.4;

    let ra4 = PrintCurveParams {
        offset: opts.curve_offset,
        gamma: opts.curve_gamma,
        pivot: opts.curve_pivot,
    };

    let cpu = pipeline::step_6_render(&img, &opts, &ra4, None);
    let gpu = gpu_pipe.run(&img, &opts, &ra4, None).expect("GPU step6 FilmPrint");
    compare_u16(&cpu, &gpu, 2, "FilmPrint basic");
}

#[test]
fn step6_film_print_full_ops() {
    let (_ctx, gpu_pipe) = match get_gpu() {
        Some(v) => v,
        None => { eprintln!("No GPU; skip"); return; }
    };
    let img = make_density_image(100, 80);
    let mut opts = PipelineOptions::default();
    opts.output_stage = OutputStage::FilmPrint;
    opts.lut_in_black = 0.01;
    opts.lut_in_white = 0.98;
    opts.lut_in_mid = 0.9;
    opts.fp_offset_r = 0.05;
    opts.fp_gamma_r = 1.1;
    opts.fp_color_bleed = 0.15;
    opts.fp_vibrance = 0.5;
    opts.curve_white = 1.1;
    opts.toe_strength = 0.2;
    opts.shoulder_strength = -0.15;
    opts.soft_clip = 0.85;
    opts.highlight_warmth = 0.3;
    opts.apply_lab = true;
    opts.lab_separation = 0.4;
    opts.skin_magenta_shift = 0.4;

    let ra4 = PrintCurveParams {
        offset: opts.curve_offset,
        gamma: opts.curve_gamma,
        pivot: opts.curve_pivot,
    };

    let cpu = pipeline::step_6_render(&img, &opts, &ra4, None);
    let gpu = gpu_pipe.run(&img, &opts, &ra4, None).expect("GPU step6 FilmPrint full");
    compare_u16(&cpu, &gpu, 10, "FilmPrint full ops");
}

// ─── None ───

#[test]
fn step6_none_mode() {
    let (_ctx, gpu_pipe) = match get_gpu() {
        Some(v) => v,
        None => { eprintln!("No GPU; skip"); return; }
    };
    let img = make_density_image(100, 80);
    let mut opts = PipelineOptions::default();
    opts.output_stage = OutputStage::None;

    let ra4 = PrintCurveParams {
        offset: opts.curve_offset,
        gamma: opts.curve_gamma,
        pivot: opts.curve_pivot,
    };

    let cpu = pipeline::step_6_render(&img, &opts, &ra4, None);
    let gpu = gpu_pipe.run(&img, &opts, &ra4, None).expect("GPU step6 None");
    compare_f32(&cpu, &gpu, 1e-6, "None mode");
}

#[test]
fn step6_none_no_invert() {
    let (_ctx, gpu_pipe) = match get_gpu() {
        Some(v) => v,
        None => { eprintln!("No GPU; skip"); return; }
    };
    let img = make_density_image(64, 48);
    let mut opts = PipelineOptions::default();
    opts.output_stage = OutputStage::None;
    opts.no_invert = true;

    let ra4 = PrintCurveParams {
        offset: opts.curve_offset,
        gamma: opts.curve_gamma,
        pivot: opts.curve_pivot,
    };

    let cpu = pipeline::step_6_render(&img, &opts, &ra4, None);
    let gpu = gpu_pipe.run(&img, &opts, &ra4, None).expect("GPU step6 None no_invert");
    compare_f32(&cpu, &gpu, 1e-6, "None no_invert");
}

// ─── Lut2383 ───

#[test]
fn step6_lut2383_cineon() {
    let (_ctx, gpu_pipe) = match get_gpu() {
        Some(v) => v,
        None => { eprintln!("No GPU; skip"); return; }
    };
    let img = make_density_image(80, 60);
    let lut = make_display_lut(17);
    let mut opts = PipelineOptions::default();
    opts.output_stage = OutputStage::Lut2383;
    opts.output_lut_encoding = OutputLutEncoding::CineonLog;
    opts.lut_in_black = 0.01;
    opts.lut_in_white = 0.95;

    let ra4 = PrintCurveParams {
        offset: opts.curve_offset,
        gamma: opts.curve_gamma,
        pivot: opts.curve_pivot,
    };

    let cpu = pipeline::step_6_render(&img, &opts, &ra4, Some(&lut));
    let gpu = gpu_pipe.run(&img, &opts, &ra4, Some(&lut)).expect("GPU step6 Lut2383 Cineon");
    compare_f32(&cpu, &gpu, 1e-5, "Lut2383 CineonLog");
}

#[test]
fn step6_lut2383_rec709() {
    let (_ctx, gpu_pipe) = match get_gpu() {
        Some(v) => v,
        None => { eprintln!("No GPU; skip"); return; }
    };
    let img = make_density_image(80, 60);
    let lut = make_display_lut(17);
    let mut opts = PipelineOptions::default();
    opts.output_stage = OutputStage::Lut2383;
    opts.output_lut_encoding = OutputLutEncoding::Rec709;
    opts.lut_in_mid = 1.2;

    let ra4 = PrintCurveParams {
        offset: opts.curve_offset,
        gamma: opts.curve_gamma,
        pivot: opts.curve_pivot,
    };

    let cpu = pipeline::step_6_render(&img, &opts, &ra4, Some(&lut));
    let gpu = gpu_pipe.run(&img, &opts, &ra4, Some(&lut)).expect("GPU step6 Lut2383 Rec709");
    compare_f32(&cpu, &gpu, 5e-5, "Lut2383 Rec709");
}

#[test]
fn step6_lut2383_full_ops() {
    let (_ctx, gpu_pipe) = match get_gpu() {
        Some(v) => v,
        None => { eprintln!("No GPU; skip"); return; }
    };
    let img = make_density_image(80, 60);
    let lut = make_display_lut(17);
    let mut opts = PipelineOptions::default();
    opts.output_stage = OutputStage::Lut2383;
    opts.output_lut_encoding = OutputLutEncoding::CineonLog;
    opts.lut_in_black = 0.02;
    opts.lut_in_white = 0.9;
    opts.lut_in_mid = 1.1;
    opts.soft_clip = 0.85;
    opts.highlight_warmth = 0.3;
    opts.apply_lab = true;
    opts.lab_separation = 0.5;
    opts.skin_magenta_shift = 0.5;

    let ra4 = PrintCurveParams {
        offset: opts.curve_offset,
        gamma: opts.curve_gamma,
        pivot: opts.curve_pivot,
    };

    let cpu = pipeline::step_6_render(&img, &opts, &ra4, Some(&lut));
    let gpu = gpu_pipe.run(&img, &opts, &ra4, Some(&lut)).expect("GPU step6 Lut2383 full");
    compare_f32(&cpu, &gpu, 1e-4, "Lut2383 full ops");
}
