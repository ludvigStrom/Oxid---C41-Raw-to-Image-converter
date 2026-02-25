use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;

use c41_raw_tool::{process_files, PipelineOptions, Rect, TiffFormat};

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
    #[arg(long = "dmin-rect", value_parser = parse_rect, value_name = "X,Y,WIDTH,HEIGHT")]
    dmin_rect: Option<Rect>,

    /// Fixed D-min medians R,G,B (linear [0,1]); bypass crop measurement.
    #[arg(long = "dmin-fixed", value_parser = parse_dmin_fixed, value_name = "R,G,B")]
    dmin_fixed: Option<(f32, f32, f32)>,

    /// Output TIFF format: 32f (float) or 16 (integer). Only with --no-curve.
    #[arg(long = "format", default_value = "32f", value_parser = parse_format, value_name = "32f|16")]
    format: TiffFormat,

    /// Also write OpenEXR (.exr) alongside TIFF.
    #[arg(long = "write-exr", action = clap::ArgAction::SetTrue)]
    write_exr: bool,

    #[arg(long = "no-invert", action = clap::ArgAction::SetTrue)]
    no_invert: bool,

    #[arg(long = "no-curve", action = clap::ArgAction::SetTrue)]
    no_curve: bool,

    #[arg(long = "wb-r", default_value = "1.0")]
    wb_r: f32,
    #[arg(long = "wb-g", default_value = "1.0")]
    wb_g: f32,
    #[arg(long = "wb-b", default_value = "1.0")]
    wb_b: f32,

    #[arg(long = "curve-offset", default_value = "0.0")]
    curve_offset: f32,
    #[arg(long = "curve-gamma", default_value = "2.5", value_parser = parse_curve_gamma)]
    curve_gamma: f32,
    #[arg(long = "curve-pivot", default_value = "3.0")]
    curve_pivot: f32,
    #[arg(long = "curve-white", default_value = "1.0")]
    curve_white: f32,
}

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

fn parse_dmin_fixed(s: &str) -> Result<(f32, f32, f32), String> {
    let parts: Vec<_> = s.split(',').collect();
    if parts.len() != 3 {
        return Err("expected R,G,B".to_string());
    }
    let parse_f32 = |p: &str| p.trim().parse::<f32>().map_err(|e| e.to_string());
    Ok((parse_f32(parts[0])?, parse_f32(parts[1])?, parse_f32(parts[2])?))
}

fn parse_curve_gamma(s: &str) -> Result<f32, String> {
    let g: f32 = s.trim().parse().map_err(|_| "curve-gamma must be a number".to_string())?;
    if !(0.5..=5.0).contains(&g) {
        return Err("curve-gamma must be between 0.5 and 5.0".to_string());
    }
    Ok(g)
}

fn parse_format(s: &str) -> Result<TiffFormat, String> {
    match s.trim().to_lowercase().as_str() {
        "32f" | "32" | "float" => Ok(TiffFormat::Float32),
        "16" | "u16" => Ok(TiffFormat::U16),
        _ => Err("format must be 32f (float) or 16 (integer)".to_string()),
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    println!("Input directory:  {}", cli.input_dir.display());
    println!("Output directory: {}", cli.output_dir.display());
    println!("D-min rect:       {:?}", cli.dmin_rect);
    println!("D-min fixed:      {:?}", cli.dmin_fixed);
    println!("Format:           {:?}", cli.format);
    println!("Write EXR:        {}", cli.write_exr);
    println!("WB gains:         R={} G={} B={}", cli.wb_r, cli.wb_g, cli.wb_b);
    println!(
        "Print curve:      {} (offset={}, gamma={}, pivot={}, white={})",
        !cli.no_curve, cli.curve_offset, cli.curve_gamma, cli.curve_pivot, cli.curve_white
    );

    let paths: Vec<PathBuf> = fs::read_dir(&cli.input_dir)
        .with_context(|| format!("Failed to read input directory {}", cli.input_dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .filter(|p| {
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            ext.eq_ignore_ascii_case("arw") || ext.eq_ignore_ascii_case("png")
        })
        .collect();

    println!("Found {} file(s) to process.", paths.len());

    let options = PipelineOptions {
        dmin_rect: cli.dmin_rect,
        dmin_fixed: cli.dmin_fixed,
        format: cli.format,
        write_exr: cli.write_exr,
        no_invert: cli.no_invert,
        no_curve: cli.no_curve,
        wb_r: cli.wb_r,
        wb_g: cli.wb_g,
        wb_b: cli.wb_b,
        curve_offset: cli.curve_offset,
        curve_gamma: cli.curve_gamma,
        curve_pivot: cli.curve_pivot,
        curve_white: cli.curve_white,
    };

    process_files(&paths, &cli.output_dir, &options)?;
    println!("Done.");
    Ok(())
}
