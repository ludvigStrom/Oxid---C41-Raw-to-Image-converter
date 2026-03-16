use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;

use c41_raw_tool::{process_files, PipelineOptions, Rect, TiffFormat, OutputStage, OutputLutEncoding, DminMode};
use c41_raw_tool::raw_reader;

#[derive(Parser, Debug)]
#[command(
    name = "c41-raw-tool",
    about = "High-performance, strictly linear RAW processor for C-41 narrowband RGB scans",
    long_about = None
)]
enum Cli {
    /// Process RAW/PNG files from a directory (default workflow)
    Convert(ConvertArgs),

    /// Validate rawloader decode of a single RAW file (ARW, RAF, etc.); print metadata and sample values
    DebugRaw(DebugRawArgs),
}

#[derive(Parser, Debug)]
struct DebugRawArgs {
    /// Path to a single RAW file (e.g. .arw, .raf)
    #[arg(value_name = "FILE")]
    file: PathBuf,
}

#[derive(Parser, Debug)]
struct ConvertArgs {
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

    /// 3×3 density-domain calibration matrix (row-major), as 9 comma-separated values:
    /// C00,C01,C02,C10,C11,C12,C20,C21,C22
    /// Defaults to the identity matrix.
    #[arg(
        long = "density-matrix",
        value_parser = parse_density_matrix,
        value_name = "C00,C01,C02,C10,C11,C12,C20,C21,C22",
        default_value = "1,0,0,0,1,0,0,0,1"
    )]
    density_matrix: [f32; 9],

    /// Flat-field reference: path to a RAW file of an unexposed, developed frame (same film stock) for luminance calibration.
    #[arg(long = "flat-field", value_name = "PATH")]
    flat_field: Option<PathBuf>,

    /// 3×3 IDT matrix (row-major), 9 comma-separated values. Default: identity; optional camera_idt/ profiles.
    #[arg(
        long = "idt-matrix",
        value_parser = parse_density_matrix,
        value_name = "M00,M01,M02,M10,M11,M12,M20,M21,M22",
        default_value = "1,0,0,0,1,0,0,0,1"
    )]
    idt_matrix: [f32; 9],

    /// Also write linear ACES2065-1 EXR (e.g. image_aces2065-1.exr) for VFX/archival.
    #[arg(long = "export-aces-exr", action = clap::ArgAction::SetTrue)]
    export_aces_exr: bool,
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

fn parse_density_matrix(s: &str) -> Result<[f32; 9], String> {
    let parts: Vec<_> = s.split(',').collect();
    if parts.len() != 9 {
        return Err(
            "expected 9 comma-separated values: C00,C01,C02,C10,C11,C12,C20,C21,C22".to_string(),
        );
    }
    let mut vals = [0.0_f32; 9];
    for (i, p) in parts.iter().enumerate() {
        vals[i] = p
            .trim()
            .parse::<f32>()
            .map_err(|e| format!("invalid float at position {}: {}", i, e))?;
    }
    Ok(vals)
}

fn main() -> anyhow::Result<()> {
    match Cli::parse() {
        Cli::DebugRaw(args) => {
            raw_reader::debug_raw(&args.file)?;
            Ok(())
        }
        Cli::Convert(cli) => run_convert(cli),
    }
}

fn run_convert(cli: ConvertArgs) -> anyhow::Result<()> {
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
    println!("Density matrix:  {:?}", cli.density_matrix);
    println!("Flat field:      {:?}", cli.flat_field);
    println!("Export ACES2065-1 EXR: {}", cli.export_aces_exr);

    let paths: Vec<PathBuf> = fs::read_dir(&cli.input_dir)
        .with_context(|| format!("Failed to read input directory {}", cli.input_dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .filter(|p| {
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "arw" | "nef" | "nrw" | "cr2" | "cr3" | "crw" | "dng" | "raf" | "orf" | "rw2" | "png" | "jpeg" | "jpg" | "tiff" | "tif"
            )
        })
        .collect();

    println!("Found {} file(s) to process.", paths.len());

    let dmin_mode = if cli.dmin_fixed.is_some() {
        DminMode::Fixed
    } else if cli.dmin_rect.is_some() {
        DminMode::SampleRegion
    } else {
        DminMode::Fixed
    };

    let options = PipelineOptions {
        dmin_mode,
        auto_norm_buffer: 0.2,
        apply_white_balance: true,
        auto_wb: true,
        wb_mode: c41_raw_tool::WbMode::Auto,
        film_gamma: 0.65,
        apply_color_profile: true,
        dmin_rect: cli.dmin_rect,
        dmin_rect_reference_size: None,
        apply_crop: false,
        crop_rect: None,
        crop_rect_reference_size: None,
        dmin_fixed: cli.dmin_fixed,
        dmin_neutral_only: false,
        format: cli.format,
        write_exr: cli.write_exr,
        write_jpeg: false,
        write_jpeg_only: false,
        no_invert: cli.no_invert,
        no_curve: cli.no_curve,
        wb_r: cli.wb_r,
        wb_g: cli.wb_g,
        wb_b: cli.wb_b,
        temp_k: None,
        curve_offset: cli.curve_offset,
        curve_gamma: cli.curve_gamma,
        curve_pivot: cli.curve_pivot,
        curve_white: cli.curve_white,
        density_matrix: [
            [cli.density_matrix[0], cli.density_matrix[1], cli.density_matrix[2]],
            [cli.density_matrix[3], cli.density_matrix[4], cli.density_matrix[5]],
            [cli.density_matrix[6], cli.density_matrix[7], cli.density_matrix[8]],
        ],
        flat_field_path: cli.flat_field,
        export_aces_exr: cli.export_aces_exr,
        write_aces2065_only: false,
        lut3d_path: None,
        output_stage: if cli.no_curve {
            OutputStage::None
        } else {
            OutputStage::Ra4
        },
        output_lut_cube: None,
        output_lut_encoding: OutputLutEncoding::CineonLog,
        lut_in_black: 0.0,
        lut_in_white: 1.0,
        lut_in_mid: 1.0,
        fp_offset_r: 0.0,
        fp_offset_g: 0.0,
        fp_offset_b: 0.0,
        fp_gamma_r: 1.0,
        fp_gamma_g: 1.0,
        fp_gamma_b: 1.0,
        fp_color_bleed: 0.08,
        fp_vibrance: 0.3,
        saturation: 1.0,
        toe_strength: 0.0,
        shoulder_strength: 0.0,
        shadow_cast_strength: 0.0,
        zone_shadows: 0.0,
        zone_highlights: 0.0,
        zone_shadow_gain: 0.0,
        zone_mid_gain: 0.0,
        zone_highlight_gain: 0.0,
        color_shadow_gain_r: 0.0,
        color_shadow_gain_g: 0.0,
        color_shadow_gain_b: 0.0,
        color_mid_gain_r: 0.0,
        color_mid_gain_g: 0.0,
        color_mid_gain_b: 0.0,
        color_highlight_gain_r: 0.0,
        color_highlight_gain_g: 0.0,
        color_highlight_gain_b: 0.0,
        zone_shadow_saturation: 1.0,
        zone_mid_saturation: 1.0,
        zone_highlight_saturation: 1.0,
        highlight_rolloff: 0.0,
        highlight_rolloff_d_mid: 1.5,
        highlight_warmth: 0.0,
        soft_clip: 0.0,
        apply_lab: false,
        lab_separation: 0.0,
        rotation_degrees: 0,
        debug_pipeline_step: 6,
        debug_preview_simple_debayer: false,
        verbose_debug: false,
        use_gpu: false,
    };

    process_files(&paths, &cli.output_dir, &options)?;
    println!("Done.");
    Ok(())
}
