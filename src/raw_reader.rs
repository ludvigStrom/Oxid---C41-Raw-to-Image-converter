use std::fmt::Write as _;
use std::path::Path;

use anyhow::{bail, Context, Result};
use ndarray::{Array2, Array3, Axis};
use rawloader::RawImageData;

use crate::demosaic::{BayerPattern, CfaPattern};

/// Check that the CFA at `(origin_row, origin_col)` is a valid 2×2 Bayer pattern
/// by verifying the 2×2 block repeats consistently over a 6×6 area.
/// Returns `false` for X-Trans (6×6 repeat) or other non-Bayer CFAs.
fn is_bayer_cfa(cfa: &rawloader::CFA, origin_row: usize, origin_col: usize) -> bool {
    let p00 = cfa.color_at(origin_row, origin_col);
    let p01 = cfa.color_at(origin_row, origin_col + 1);
    let p10 = cfa.color_at(origin_row + 1, origin_col);
    let p11 = cfa.color_at(origin_row + 1, origin_col + 1);

    for r in 0..6 {
        for c in 0..6 {
            let expected = match (r % 2, c % 2) {
                (0, 0) => p00,
                (0, 1) => p01,
                (1, 0) => p10,
                (1, 1) => p11,
                _ => unreachable!(),
            };
            if cfa.color_at(origin_row + r, origin_col + c) != expected {
                return false;
            }
        }
    }
    true
}

/// Check whether the CFA at `(origin_row, origin_col)` is an X-Trans 6×6 pattern
/// by verifying the 6×6 tile repeats correctly over a 12×12 area (two periods).
fn is_xtrans_cfa(cfa: &rawloader::CFA, origin_row: usize, origin_col: usize) -> bool {
    // Read the 6×6 reference tile.
    let mut tile = [[0usize; 6]; 6];
    for r in 0..6 {
        for c in 0..6 {
            tile[r][c] = cfa.color_at(origin_row + r, origin_col + c);
        }
    }
    // Verify it repeats over a 12×12 area.
    for r in 0..12 {
        for c in 0..12 {
            if cfa.color_at(origin_row + r, origin_col + c) != tile[r % 6][c % 6] {
                return false;
            }
        }
    }
    true
}

/// Extract the 6×6 X-Trans tile as seen from the crop origin `(crop_top, crop_left)`.
///
/// The returned tile satisfies `tile[y % 6][x % 6] == cfa.color_at(crop_top + y, crop_left + x)`
/// for any cropped-image coordinate `(y, x)`.
fn extract_xtrans_tile(
    cfa: &rawloader::CFA,
    crop_top: usize,
    crop_left: usize,
) -> [[u8; 6]; 6] {
    let mut tile = [[0u8; 6]; 6];
    for r in 0..6 {
        for c in 0..6 {
            tile[r][c] = cfa.color_at(crop_top + r, crop_left + c) as u8;
        }
    }
    tile
}

/// Detect the Bayer pattern by checking a 2×2 block at the given (row, col) origin.
fn detect_bayer_pattern_at(cfa: &rawloader::CFA, row: usize, col: usize) -> Option<BayerPattern> {
    let tl = cfa.color_at(row, col);
    let tr = cfa.color_at(row, col + 1);
    let bl = cfa.color_at(row + 1, col);
    let br = cfa.color_at(row + 1, col + 1);
    match (tl, tr, bl, br) {
        (0, 1, 1, 2) => Some(BayerPattern::Rggb),
        (1, 0, 2, 1) => Some(BayerPattern::Grbg),
        (1, 2, 0, 1) => Some(BayerPattern::Gbrg),
        (2, 1, 1, 0) => Some(BayerPattern::Bggr),
        _ => None,
    }
}

/// Load a RAW file into a strictly linear `Array3<f32>` with proper normalization.
///
/// Processing steps:
/// 1. Subtract per-channel **black level** (removes dark current / sensor bias).
/// 2. Divide by a **single global scale** (max white − min black) to preserve
///    inter-channel color ratios.
/// 3. Apply **camera white balance** at the Bayer (pre-demosaic) level. This
///    roughly balances R/G/B so the demosaic algorithm sees comparable channel
///    levels. Without this, the blue channel (heavily attenuated by the C-41
///    orange base) has very poor SNR and the demosaic produces extreme artifacts.
///    With per-channel D-min, camera WB cancels out mathematically.
/// 4. Apply sensor crops (forced even for Bayer alignment).
/// 5. Detect and validate the CFA pattern (rejects X-Trans / non-Bayer sensors).
///
/// Returns `(bayer_array, detected_pattern)`.
pub fn load_raw_as_ndarray(path: &Path) -> Result<(Array3<f32>, CfaPattern)> {
    let raw_image = rawloader::decode_file(path)
        .with_context(|| format!("rawloader failed to decode {}", path.display()))?;

    let width = raw_image.width;
    let height = raw_image.height;

    if raw_image.cpp != 1 {
        bail!(
            "Expected Bayer/CFA raw (cpp=1), got cpp={}. Unsupported format.",
            raw_image.cpp
        );
    }

    let [crop_top, crop_right, crop_bottom, crop_left] = raw_image.crops;

    // Force crops even so the Bayer 2×2 block alignment is preserved.
    let crop_top = crop_top & !1;
    let crop_left = crop_left & !1;
    let crop_right = crop_right & !1;
    let crop_bottom = crop_bottom & !1;
    let cropped_w = width.saturating_sub(crop_left + crop_right) & !1;
    let cropped_h = height.saturating_sub(crop_top + crop_bottom) & !1;

    if cropped_w == 0 || cropped_h == 0 {
        bail!(
            "Crop [{},{},{},{}] leaves no pixels in {}x{} image",
            crop_top, crop_right, crop_bottom, crop_left, width, height
        );
    }

    let cfa = &raw_image.cfa;

    // Detect the CFA type: standard 2×2 Bayer or Fujifilm 6×6 X-Trans.
    let pattern: CfaPattern = if is_bayer_cfa(cfa, crop_top, crop_left) {
        let bayer = detect_bayer_pattern_at(cfa, crop_top, crop_left)
            .unwrap_or(BayerPattern::Rggb);
        CfaPattern::Bayer(bayer)
    } else if is_xtrans_cfa(cfa, crop_top, crop_left) {
        let tile = extract_xtrans_tile(cfa, crop_top, crop_left);
        CfaPattern::XTrans(tile)
    } else {
        bail!(
            "Unsupported CFA pattern in {}. Only standard 2×2 Bayer \
             (RGGB/GRBG/GBRG/BGGR) and Fujifilm X-Trans 6×6 are supported.",
            path.display()
        );
    };

    let blacks = raw_image.blacklevels.map(|v| v as f32);
    let whites = raw_image.whitelevels.map(|v| v as f32);

    // Single global scale preserves inter-channel color ratios.
    let min_black = blacks.iter().copied().reduce(f32::min).unwrap_or(0.0);
    let max_white = whites.iter().copied().reduce(f32::max).unwrap_or(1.0);
    let global_scale = (max_white - min_black).max(1.0);

    // Camera WB is applied at the Bayer (pre-demosaic) level to roughly balance
    // channels. This is critical for demosaic quality: the orange C-41 base
    // transmits ~3-4× more red than blue, so without WB the B channel has very
    // poor SNR and the demosaic produces severe artifacts and negative values.
    //
    // With per-channel D-min, camera WB cancels out mathematically (both the
    // image and the base sample are scaled by the same factor), so it does NOT
    // affect the final color — it only improves the quality of the intermediate
    // demosaiced data.
    let wb = raw_image.wb_coeffs;
    let wb_valid = wb[0].is_finite()
        && wb[1].is_finite()
        && wb[2].is_finite()
        && wb[0] > 0.0
        && wb[1] > 0.0
        && wb[2] > 0.0;
    let (wb_r, wb_g, wb_b) = if wb_valid {
        let g = wb[1];
        (wb[0] / g, 1.0f32, wb[2] / g)
    } else {
        (1.0, 1.0, 1.0)
    };

    let expected_len = width * height;

    let float_data: Vec<f32> = match &raw_image.data {
        RawImageData::Integer(data) => {
            if data.len() != expected_len {
                bail!(
                    "Unexpected raw buffer length: expected {} pixels, got {}",
                    expected_len,
                    data.len()
                );
            }
            let mut out = Vec::with_capacity(cropped_w * cropped_h);
            for row in 0..cropped_h {
                let src_row = row + crop_top;
                for col in 0..cropped_w {
                    let src_col = col + crop_left;
                    let ch = cfa.color_at(src_row, src_col);
                    let black = blacks[ch];
                    let channel_wb = match ch {
                        0 => wb_r,
                        2 => wb_b,
                        _ => wb_g,
                    };
                    let raw = data[src_row * width + src_col] as f32;
                    let val = (raw - black) / global_scale * channel_wb;
                    out.push(val.max(0.0));
                }
            }
            out
        }
        RawImageData::Float(data) => {
            if data.len() != expected_len {
                bail!(
                    "Unexpected raw buffer length: expected {} pixels, got {}",
                    expected_len,
                    data.len()
                );
            }
            let mut out = Vec::with_capacity(cropped_w * cropped_h);
            for row in 0..cropped_h {
                let src_row = row + crop_top;
                for col in 0..cropped_w {
                    let src_col = col + crop_left;
                    let ch = cfa.color_at(src_row, src_col);
                    let black = blacks[ch];
                    let channel_wb = match ch {
                        0 => wb_r,
                        2 => wb_b,
                        _ => wb_g,
                    };
                    let raw = data[src_row * width + src_col];
                    let val = (raw - black) / global_scale * channel_wb;
                    out.push(val.max(0.0));
                }
            }
            out
        }
    };

    let array2: Array2<f32> =
        Array2::from_shape_vec((cropped_h, cropped_w), float_data).with_context(|| {
            format!(
                "Failed to reshape RAW data into 2D array (height={}, width={})",
                cropped_h, cropped_w
            )
        })?;

    let array3: Array3<f32> = array2.insert_axis(Axis(2));
    Ok((array3, pattern))
}

/// Build a human-readable report of rawloader decode info for one RAW file.
/// This is used by both CLI (`debug-raw`) and GUI Debug tab.
pub fn debug_raw_report(path: &Path) -> Result<String> {
    let mut out = String::new();
    writeln!(&mut out, "=== rawloader debug: {} ===", path.display()).ok();
    writeln!(&mut out).ok();

    let raw = rawloader::decode_file(path)
        .with_context(|| format!("rawloader failed to decode {}", path.display()))?;

    writeln!(&mut out, "make:        {}", raw.make).ok();
    writeln!(&mut out, "model:       {}", raw.model).ok();
    writeln!(&mut out, "width:       {}", raw.width).ok();
    writeln!(&mut out, "height:      {}", raw.height).ok();
    writeln!(&mut out, "cpp:         {} (expect 1 for Bayer)", raw.cpp).ok();
    writeln!(
        &mut out,
        "crops:       [top={}, right={}, bottom={}, left={}]",
        raw.crops[0], raw.crops[1], raw.crops[2], raw.crops[3]
    )
    .ok();
    writeln!(
        &mut out,
        "blacklevels: [R={}, G={}, B={}, E={}]",
        raw.blacklevels[0], raw.blacklevels[1], raw.blacklevels[2], raw.blacklevels[3]
    )
    .ok();
    writeln!(
        &mut out,
        "whitelevels: [R={}, G={}, B={}, E={}]",
        raw.whitelevels[0], raw.whitelevels[1], raw.whitelevels[2], raw.whitelevels[3]
    )
    .ok();
    writeln!(
        &mut out,
        "wb_coeffs:   [R={}, G={}, B={}, E={}]",
        raw.wb_coeffs[0], raw.wb_coeffs[1], raw.wb_coeffs[2], raw.wb_coeffs[3]
    )
    .ok();

    let cfa = &raw.cfa;
    let (ct, cl) = (raw.crops[0], raw.crops[3]);
    writeln!(&mut out).ok();
    writeln!(&mut out, "CFA at crop origin (top={}, left={}):", ct, cl).ok();
    writeln!(
        &mut out,
        "  (0,0)={} (0,1)={} (1,0)={} (1,1)={}  (0=R, 1=G, 2=B, 3=E)",
        cfa.color_at(ct, cl),
        cfa.color_at(ct, cl + 1),
        cfa.color_at(ct + 1, cl),
        cfa.color_at(ct + 1, cl + 1)
    )
    .ok();
    let crop_top = ct & !1;
    let crop_left = cl & !1;
    let is_bayer = is_bayer_cfa(cfa, crop_top, crop_left);
    writeln!(
        &mut out,
        "  is_bayer_cfa (2x2 repeat over 6x6): {}",
        is_bayer
    )
    .ok();
    if let Some(p) = detect_bayer_pattern_at(cfa, crop_top, crop_left) {
        writeln!(&mut out, "  detected pattern: {:?}", p).ok();
    } else {
        writeln!(&mut out, "  detected pattern: (none - not standard Bayer)").ok();
    }

    let n = raw.width * raw.height;
    writeln!(&mut out).ok();
    writeln!(&mut out, "Raw data: {} pixels", n).ok();

    match &raw.data {
        RawImageData::Integer(buf) => {
            if buf.len() != n {
                writeln!(
                    &mut out,
                    "  WARNING: buffer length {} != width*height {}",
                    buf.len(),
                    n
                )
                .ok();
            } else {
                let min = *buf.iter().min().unwrap_or(&0);
                let max = *buf.iter().max().unwrap_or(&0);
                let mut sorted = buf.to_vec();
                sorted.sort_unstable();
                let median = sorted[n / 2];
                writeln!(&mut out, "  min={} max={} median={}", min, max, median).ok();
                let (cy, cx) = (raw.height / 2, raw.width / 2);
                writeln!(&mut out, "  center 3x3 block (row {} col {}):", cy, cx).ok();
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let y = (cy as i32 + dy).max(0).min(raw.height as i32 - 1) as usize;
                        let x = (cx as i32 + dx).max(0).min(raw.width as i32 - 1) as usize;
                        let v = buf[y * raw.width + x];
                        let ch = cfa.color_at(y, x);
                        let ch_name = match ch {
                            0 => "R",
                            1 => "G",
                            2 => "B",
                            _ => "E",
                        };
                        write!(&mut out, "  {} ({})", v, ch_name).ok();
                    }
                    writeln!(&mut out).ok();
                }
            }
        }
        RawImageData::Float(buf) => {
            if buf.len() != n {
                writeln!(
                    &mut out,
                    "  WARNING: buffer length {} != width*height {}",
                    buf.len(),
                    n
                )
                .ok();
            } else {
                let min = buf.iter().copied().fold(f32::INFINITY, f32::min);
                let max = buf.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sorted = buf.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = sorted[n / 2];
                writeln!(&mut out, "  min={} max={} median={}", min, max, median).ok();
                let (cy, cx) = (raw.height / 2, raw.width / 2);
                writeln!(&mut out, "  center 3x3 block (row {} col {}):", cy, cx).ok();
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let y = (cy as i32 + dy).max(0).min(raw.height as i32 - 1) as usize;
                        let x = (cx as i32 + dx).max(0).min(raw.width as i32 - 1) as usize;
                        let v = buf[y * raw.width + x];
                        let ch = cfa.color_at(y, x);
                        let ch_name = match ch {
                            0 => "R",
                            1 => "G",
                            2 => "B",
                            _ => "E",
                        };
                        write!(&mut out, "  {:.4} ({})", v, ch_name).ok();
                    }
                    writeln!(&mut out).ok();
                }
            }
        }
    }

    writeln!(&mut out).ok();
    writeln!(&mut out, "=== end debug ===").ok();
    Ok(out)
}

/// Debug rawloader's decode of a RAW file: print metadata and sample values to stdout.
pub fn debug_raw(path: &Path) -> Result<()> {
    let report = debug_raw_report(path)?;
    println!("{}", report);
    Ok(())
}
