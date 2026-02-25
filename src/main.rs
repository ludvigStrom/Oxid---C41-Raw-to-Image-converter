use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use ndarray;

mod curve;
mod demosaic;
mod dmin;
mod exr_export;
mod inversion;
mod png_reader;
mod raw_reader;
mod tiff_export;

/// Rectangle for D-min sampling (in pixel coordinates).
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// CLI definition.
#[derive(Parser, Debug)]
#[command(
    name = "c41-raw-tool",
    about = "High-performance, strictly linear RAW processor for C-41 narrowband RGB scans",
    long_about = None
)]
struct Cli {
    /// Input directory containing .ARW (RAW) and/or .png files
    #[arg(short = 'i', long = "input-dir")]
    input_dir: PathBuf,

    /// Output directory for 16-bit TIFFs
    #[arg(short = 'o', long = "output-dir")]
    output_dir: PathBuf,

    /// D-min crop rectangle as X,Y,WIDTH,HEIGHT (pixels)
    ///
    /// Example: --dmin-rect 50,50,200,200
    #[arg(long = "dmin-rect", value_parser = parse_rect, value_name = "X,Y,WIDTH,HEIGHT")]
    dmin_rect: Option<Rect>,

    /// Output TIFF format: 32f (float, max data) or 16 (integer, display/print)
    ///
    /// 32f = 32-bit float, no clamping or quantization. Best for archival.
    /// 16  = 16-bit integer, clamp to [0,1]. Smaller, widely compatible.
    #[arg(long = "format", default_value = "32f", value_parser = parse_format, value_name = "32f|16")]
    format: tiff_export::TiffFormat,

    /// Also write an OpenEXR (.exr) file alongside the TIFF.
    /// Uses 32-bit float RGB in [0, 1].
    #[arg(long = "write-exr", action = clap::ArgAction::SetTrue)]
    write_exr: bool,

    /// Skip negative→positive inversion (output stays as negative after D-min)
    #[arg(long = "no-invert", action = clap::ArgAction::SetTrue)]
    no_invert: bool,

    /// Skip physical print curve (output stays as linear transmittance; use --format for float/16 export).
    /// When skipped, --no-invert controls whether 1.0-input is applied.
    #[arg(long = "no-curve", action = clap::ArgAction::SetTrue)]
    no_curve: bool,

    /// Red channel gain multiplier (applied after D-min, before curve). Default 1.0.
    #[arg(long = "wb-r", default_value = "1.0", value_name = "N")]
    wb_r: f32,

    /// Green channel gain multiplier (applied after D-min, before curve). Default 1.0.
    #[arg(long = "wb-g", default_value = "1.0", value_name = "N")]
    wb_g: f32,

    /// Blue channel gain multiplier (applied after D-min, before curve). Default 1.0.
    #[arg(long = "wb-b", default_value = "1.0", value_name = "N")]
    wb_b: f32,

    /// Print exposure offset (log-domain shift). Higher = brighter print. Default 0.0.
    #[arg(long = "curve-offset", default_value = "0.0", value_name = "N")]
    curve_offset: f32,

    /// Paper grade / contrast gamma. Higher = harder paper. Default 2.5.
    #[arg(long = "curve-gamma", default_value = "2.5", value_parser = parse_curve_gamma, value_name = "N")]
    curve_gamma: f32,

    /// Half-saturation exposure for RA-4 S-curve. Default 3.0.
    #[arg(long = "curve-pivot", default_value = "3.0", value_name = "N")]
    curve_pivot: f32,

    /// Normalized code that should map to display white after the curve (0–1).
    /// Example: 0.745 ≈ 190/255. Default 1.0 (no additional scaling).
    #[arg(long = "curve-white", default_value = "1.0", value_name = "N")]
    curve_white: f32,
}

/// Parse a rectangle of the form "x,y,width,height".
fn parse_rect(s: &str) -> Result<Rect, String> {
    let parts: Vec<_> = s.split(',').collect();
    if parts.len() != 4 {
        return Err("expected X,Y,WIDTH,HEIGHT".to_string());
    }

    let parse_u32 = |p: &str| p.trim().parse::<u32>().map_err(|e| e.to_string());

    let x = parse_u32(parts[0])?;
    let y = parse_u32(parts[1])?;
    let width = parse_u32(parts[2])?;
    let height = parse_u32(parts[3])?;

    if width == 0 || height == 0 {
        return Err("WIDTH and HEIGHT must be > 0".to_string());
    }

    Ok(Rect { x, y, width, height })
}

/// Parse curve gamma: float in 0.5..=5.0.
fn parse_curve_gamma(s: &str) -> Result<f32, String> {
    let g: f32 = s.trim().parse().map_err(|_| "curve-gamma must be a number".to_string())?;
    if !(0.5..=5.0).contains(&g) {
        return Err("curve-gamma must be between 0.5 and 5.0".to_string());
    }
    Ok(g)
}

/// Parse output format: 32f or 16.
fn parse_format(s: &str) -> Result<tiff_export::TiffFormat, String> {
    match s.trim().to_lowercase().as_str() {
        "32f" | "32" | "float" => Ok(tiff_export::TiffFormat::Float32),
        "16" | "u16" => Ok(tiff_export::TiffFormat::U16),
        _ => Err("format must be 32f (float) or 16 (integer)".to_string()),
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Ensure output directory exists
    fs::create_dir_all(&cli.output_dir)
        .with_context(|| format!("Failed to create output directory {}", cli.output_dir.display()))?;

    println!("Input directory:  {}", cli.input_dir.display());
    println!("Output directory: {}", cli.output_dir.display());
    println!("D-min rect:       {:?}", cli.dmin_rect);
    println!("Output format:    {:?}", cli.format);
    println!("Write EXR:        {}", cli.write_exr);
    println!("Invert (neg→pos): {}", if cli.no_curve { format!("{}", !cli.no_invert) } else { "via curve (log domain)".to_string() });
    println!("WB gains:         R={} G={} B={}", cli.wb_r, cli.wb_g, cli.wb_b);
    println!(
        "Print curve:     {} (offset={}, gamma={}, pivot={}, white={})",
        !cli.no_curve,
        cli.curve_offset,
        cli.curve_gamma,
        cli.curve_pivot,
        cli.curve_white
    );

    // One LUT for all images when curve is enabled. The curve handles inversion in log/density domain.
    let lut = (!cli.no_curve).then(|| curve::generate_16bit_lut(cli.curve_offset, cli.curve_gamma, cli.curve_pivot));

    // Iterate over input directory: .arw (RAW) and .png
    let entries = fs::read_dir(&cli.input_dir)
        .with_context(|| format!("Failed to read input directory {}", cli.input_dir.display()))?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if !path.is_file() {
            continue;
        }

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();

        let mut image = match ext.as_str() {
            "arw" => {
                println!("\nLoading RAW file: {}", path.display());
                let bayer = raw_reader::load_raw_as_ndarray(&path)?;
                let (h, w, c) = bayer.dim();
                println!("Loaded RAW: height={}, width={}, channels={}", h, w, c);
                let rgb = demosaic::demosaic_bilinear(&bayer, demosaic::BayerPattern::Rggb)?;
                let (h, w, c) = rgb.dim();
                println!("Demosaiced to RGB: height={}, width={}, channels={}", h, w, c);
                rgb
            }
            "png" => {
                println!("\nLoading PNG: {}", path.display());
                let rgb = png_reader::load_png_as_ndarray(&path)?;
                let (h, w, c) = rgb.dim();
                println!("Loaded PNG: height={}, width={}, channels={}", h, w, c);
                rgb
            }
            _ => continue,
        };

        // D-min neutralization (sample unexposed border, divide by median R/G/B)
        if let Some(rect) = cli.dmin_rect {
            dmin::neutralize(&mut image, rect.x, rect.y, rect.width, rect.height)?;
            println!("D-min neutralized with rect {:?}", rect);
        }

        // Per-channel white balance gains (compensate narrowband LED imbalance)
        if cli.wb_r != 1.0 || cli.wb_g != 1.0 || cli.wb_b != 1.0 {
            image.slice_mut(ndarray::s![.., .., 0]).mapv_inplace(|v| v * cli.wb_r);
            image.slice_mut(ndarray::s![.., .., 1]).mapv_inplace(|v| v * cli.wb_g);
            image.slice_mut(ndarray::s![.., .., 2]).mapv_inplace(|v| v * cli.wb_b);
            println!("Applied WB gains: R={} G={} B={}", cli.wb_r, cli.wb_g, cli.wb_b);
        }

        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("image");
        let out_path = cli.output_dir.join(format!("{}.tiff", stem));
        let exr_path = cli.output_dir.join(format!("{}.exr", stem));

        if let Some(ref lut) = lut {
            // Physical print curve: inversion is implicit in the density domain.
            // Do NOT apply the linear 1.0 - input here.
            let image_u16 = curve::apply_curve_and_quantize(&image, lut, cli.curve_white);
            tiff_export::write_tiff_u16(&image_u16, &out_path)?;
            println!("Applied print film curve, wrote {}", out_path.display());

            if cli.write_exr {
                exr_export::write_exr_u16(&image_u16, &exr_path)?;
                println!("Also wrote EXR {}", exr_path.display());
            }
        } else {
            // No curve: optionally apply the simple linear inversion for quick preview
            if !cli.no_invert {
                inversion::invert(&mut image);
                println!("Inverted (linear 1.0 - input)");
            }
            tiff_export::write_tiff(&image, &out_path, cli.format)?;
            println!("Wrote {}", out_path.display());

            if cli.write_exr {
                exr_export::write_exr_f32(&image, &exr_path)?;
                println!("Also wrote EXR {}", exr_path.display());
            }
        }
    }

    Ok(())
}