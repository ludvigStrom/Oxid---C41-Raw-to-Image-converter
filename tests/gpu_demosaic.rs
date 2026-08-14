//! CPU-vs-GPU comparison tests for demosaic (RGGB quality).

#![cfg(feature = "gpu")]

use std::sync::Arc;

use c41_raw_tool::demosaic::{demosaic_quality, BayerPattern, CfaPattern};
use c41_raw_tool::gpu::{demosaic::DemosaicPipeline, GpuContext};
use ndarray::Array3;

/// Build a small RGGB Bayer pattern. (y,x): R at (even,even), G at (even,odd)/(odd,even), B at (odd,odd).
fn make_rggb_bayer(width: usize, height: usize) -> Array3<f32> {
    let mut bayer = Array3::<f32>::zeros((height, width, 1));
    for y in 0..height {
        for x in 0..width {
            let v = (x as f32 / width as f32 * 0.9 + 0.05).clamp(0.01, 1.0);
            bayer[[y, x, 0]] = v;
        }
    }
    bayer
}

fn compare_demosaic_outputs(cpu: &Array3<f32>, gpu: &Array3<f32>, tolerance: f32) {
    assert_eq!(cpu.dim(), gpu.dim(), "dimension mismatch");
    let (h, w, c) = cpu.dim();
    let mut max_diff = 0.0f32;
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let d = (cpu[[y, x, ch]] - gpu[[y, x, ch]]).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
    }
    eprintln!("  max diff: {:.8}", max_diff);
    assert!(
        max_diff <= tolerance,
        "max_diff {} exceeds tolerance {}",
        max_diff,
        tolerance
    );
}

#[test]
fn demosaic_rggb_small() {
    let ctx = match GpuContext::try_new() {
        Some(gctx) => Arc::new(gctx),
        None => {
            eprintln!("No GPU; skip");
            return;
        }
    };
    let pipeline = DemosaicPipeline::new(&ctx);

    let bayer = make_rggb_bayer(64, 48);
    let pattern = CfaPattern::Bayer(BayerPattern::Rggb);

    let cpu_rgb = demosaic_quality(&bayer, pattern).unwrap();
    let gpu_rgb = pipeline.run_rggb(&bayer).unwrap();

    // GPU uses same math; expect very small difference (floating point order)
    compare_demosaic_outputs(&cpu_rgb, &gpu_rgb, 1e-4);
}

#[test]
fn demosaic_rggb_larger() {
    let ctx = match GpuContext::try_new() {
        Some(gctx) => Arc::new(gctx),
        None => {
            eprintln!("No GPU; skip");
            return;
        }
    };
    let pipeline = DemosaicPipeline::new(&ctx);

    let bayer = make_rggb_bayer(320, 240);
    let pattern = CfaPattern::Bayer(BayerPattern::Rggb);

    let cpu_rgb = demosaic_quality(&bayer, pattern).unwrap();
    let gpu_rgb = pipeline.run_rggb(&bayer).unwrap();

    compare_demosaic_outputs(&cpu_rgb, &gpu_rgb, 1e-4);
}

#[test]
fn demosaic_gpu_or_cpu_fallback() {
    // When GPU fails or pattern is non-RGGB, should fall back to CPU
    let bayer = make_rggb_bayer(32, 32);
    let pattern = CfaPattern::Bayer(BayerPattern::Rggb);

    let cpu_direct = demosaic_quality(&bayer, pattern).unwrap();
    let via_helper = c41_raw_tool::gpu::demosaic::demosaic_gpu_or_cpu(&bayer, pattern, None).unwrap();

    assert_eq!(cpu_direct.dim(), via_helper.dim());
    let max_diff: f32 = cpu_direct
        .iter()
        .zip(via_helper.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(max_diff < 1e-6, "CPU fallback should match direct CPU");
}
