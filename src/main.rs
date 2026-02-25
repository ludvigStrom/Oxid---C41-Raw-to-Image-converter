use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;

mod curve;
mod demosaic;
mod dmin;
mod inversion;
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
    /// Input directory containing Sony .ARW files
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

    /// Skip negative→positive inversion (output stays as negative after D-min)
    #[arg(long = "no-invert", action = clap::ArgAction::SetTrue)]
    no_invert: bool,

    /// Skip universal tone curve (output stays linear; use --format for float/16 export)
    #[arg(long = "no-curve", action = clap::ArgAction::SetTrue)]
    no_curve: bool,
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
    println!("Invert (neg→pos): {}", !cli.no_invert);
    println!("Tone curve:      {}", !cli.no_curve);

    // One LUT for all images when curve is enabled
    let lut = (!cli.no_curve).then(|| curve::generate_16bit_lut(curve::DEFAULT_STEEPNESS));

    // Iterate over input directory, picking up .arw files
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

        if ext != "arw" {
            continue;
        }

        println!("\nLoading RAW file: {}", path.display());

        // Load raw Bayer into (H, W, 1)
        let bayer = raw_reader::load_raw_as_ndarray(&path)?;
        let (h, w, c) = bayer.dim();
        println!("Loaded RAW: height={}, width={}, channels={}", h, w, c);

        // Demosaic to linear RGB (H, W, 3) — Sony a7R II = RGGB
        let mut image = demosaic::demosaic_bilinear(&bayer, demosaic::BayerPattern::Rggb)?;
        let (h, w, c) = image.dim();
        println!("Demosaiced to RGB: height={}, width={}, channels={}", h, w, c);

        // D-min neutralization (sample unexposed border, divide by median R/G/B)
        if let Some(rect) = cli.dmin_rect {
            dmin::neutralize(&mut image, rect.x, rect.y, rect.width, rect.height)?;
            println!("D-min neutralized with rect {:?}", rect);
        }

        // Inversion: negative → positive (1.0 - input)
        if !cli.no_invert {
            inversion::invert(&mut image);
            println!("Inverted (negative → positive)");
        }

        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("image");
        let out_path = cli.output_dir.join(format!("{}.tiff", stem));

        // Tone curve (LUT) → u16, or direct f32/16 export
        if let Some(ref lut) = lut {
            let image_u16 = curve::apply_curve_and_quantize(&image, lut);
            tiff_export::write_tiff_u16(&image_u16, &out_path)?;
            println!("Applied tone curve, wrote {}", out_path.display());
        } else {
            tiff_export::write_tiff(&image, &out_path, cli.format)?;
            println!("Wrote {}", out_path.display());
        }
    }

    Ok(())
}