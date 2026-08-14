//! Pipeline option types: rects, D-min mode, output stage, and full pipeline options.

use std::path::PathBuf;

use crate::tiff_export::TiffFormat;

/// Rectangle for D-min sampling (pixel coordinates).
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// White balance mode: Auto (per-channel median equalization) or Picker (sample a point to set neutral).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WbMode {
    /// Use auto white balance (per-channel density median equalization when D-min is on).
    Auto,
    /// Use manual WB gains; can set them via eyedropper (white/gray/black point) or sliders.
    Picker,
}

/// D-min method selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DminMode {
    /// D-min correction disabled.
    Off,
    /// Fixed R, G, B values (manually entered or pasted).
    Fixed,
    /// Sample a rectangle region to compute per-channel medians.
    SampleRegion,
    /// Automatic per-channel percentile-based normalization (negPy style).
    /// Finds D-min floor via low percentile in log-density space.
    AutoPercentile,
}

/// Final render/output stage selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputStage {
    /// RA-4 print emulation (current behavior).
    Ra4,
    /// Film Print: per-channel RA-4 curves with color bleed and vibrance.
    FilmPrint,
    /// Generic display-space 3D LUT (e.g. Cineon→Kodak 2383 D65 cube).
    Lut2383,
    /// No print/display stage; density or linear mapping out.
    None,
}

/// Encoding expected by the output LUT. Determines the pre-transform
/// applied to density values before feeding them into the cube.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputLutEncoding {
    /// Cineon log: printing density ÷ 2.046 (10-bit Cineon full-scale).
    /// Use with Resolve-style "Kodak 2383" cubes whose input is Cineon Film Log.
    CineonLog,
    /// Density-as-code-value + sRGB OETF: `OETF(D / 2.5)` then levels.
    /// This is **not** a scene-linear Rec.709 conversion (no primaries hop).
    /// Use with cubes that expect gamma-encoded Rec.709-looking input.
    Rec709,
    /// Linear normalized density: D ÷ D_max (2.5).
    /// Use with generic cubes that expect linear 0–1 RGB.
    LinearDensity,
}

/// All pipeline options (CLI flags / GUI state).
#[derive(Debug, Clone)]
pub struct PipelineOptions {
    /// Which D-min method to use (Off / Fixed / SampleRegion / AutoPercentile).
    pub dmin_mode: DminMode,
    /// Border exclusion ratio for AutoPercentile mode (0.0–0.3).
    pub auto_norm_buffer: f32,
    /// When false, white balance gains are not applied.
    pub apply_white_balance: bool,
    /// When true, automatically equalize per-channel density medians after D-min
    /// using multiplicative correction (per-channel gamma). Preserves D=0 black point.
    pub auto_wb: bool,
    /// White balance mode: Auto (median equalization) or Picker (eyedropper + manual sliders).
    pub wb_mode: WbMode,
    /// C-41 film negative gamma. Scene log-exposure = density / film_gamma.
    /// Typical values: 0.55–0.75 for C-41, ~0.65 default. Applied as D *= 1/gamma
    /// to decompress density into scene-relative log-exposure before the paper curve.
    pub film_gamma: f32,
    pub dmin_rect: Option<Rect>,
    /// When set, the rect is in pixels for this (width, height). Used to scale rect when exporting at full size.
    pub dmin_rect_reference_size: Option<(u32, u32)>,
    /// When false, crop is skipped and full frame is exported.
    pub apply_crop: bool,
    /// Optional export crop rectangle in pixel coordinates.
    pub crop_rect: Option<Rect>,
    /// Reference size for `crop_rect` coordinates (width, height), used to scale to current image size.
    pub crop_rect_reference_size: Option<(u32, u32)>,
    pub dmin_fixed: Option<(f32, f32, f32)>,
    /// When true and using rect, divide all channels by the same value (geometric mean of medians) to remove density without shifting color.
    pub dmin_neutral_only: bool,
    pub format: TiffFormat,
    pub write_exr: bool,
    pub write_jpeg: bool,
    /// When true, output is JPEG only (no TIFF). Implies JPEG write; "Also export JPG" is irrelevant.
    pub write_jpeg_only: bool,
    pub no_invert: bool,
    pub no_curve: bool,
    pub wb_r: f32,
    pub wb_g: f32,
    pub wb_b: f32,
    /// When set, multiply WB by gains derived from this color temperature (K). e.g. 5500 = daylight, 3000 = tungsten.
    pub temp_k: Option<f32>,
    pub curve_offset: f32,
    pub curve_gamma: f32,
    pub curve_pivot: f32,
    pub curve_white: f32,
    /// When false, the 3×3 density matrix is ignored (identity is used instead).
    pub apply_color_profile: bool,
    pub density_matrix: [[f32; 3]; 3],
    /// Path to a RAW flat-field (unexposed) frame for luminance calibration. Optional.
    pub flat_field_path: Option<PathBuf>,
    /// Camera IDT (linear camera RGB → ACEScg). Identity = stay in camera RGB;
    /// the ACEScg→Rec.709 matrix is not applied. Non-identity: IDT then
    /// ACEScg→linear Rec.709 so D-min / density / RA-4 run in Rec.709.
    pub idt_matrix: [[f32; 3]; 3],
    /// When true, also write a linear ACES2065-1 EXR alongside display output.
    pub export_aces_exr: bool,
    /// When true, output is only ACES2065-1 EXR (32-bit float); no TIFF/JPEG.
    pub write_aces2065_only: bool,
    /// Optional 3D LUT (density domain) used instead of the density matrix when set.
    /// If present, applied after T→D, before D→RA-4. Generated via "Generate 3D LUT" from current matrix.
    pub lut3d_path: Option<PathBuf>,
    /// Output stage: RA-4 curve, generic display-space 3D LUT, or raw density/no-curve display.
    pub output_stage: OutputStage,
    /// Generic display-space 3D LUT (normalized 0–1 RGB in/out) applied in the
    /// final output stage when `output_stage == OutputStage::Lut2383`.
    pub output_lut_cube: Option<PathBuf>,
    /// Encoding expected by `output_lut_cube`. Determines the density→code-value
    /// pre-transform before the cube is applied.
    pub output_lut_encoding: OutputLutEncoding,
    /// Pre-LUT levels: black point (input density that maps to 0). Default 0.0.
    pub lut_in_black: f32,
    /// Pre-LUT levels: white point (input density that maps to 1). Default 1.0.
    pub lut_in_white: f32,
    /// Pre-LUT levels: midpoint gamma. Redistributes tones between black and
    /// white: v_out = v_in^(1/mid). 1.0 = linear (no change), >1.0 = brighter
    /// midtones, <1.0 = darker midtones. Default 1.0.
    pub lut_in_mid: f32,
    // ── Film Print stage parameters ──────────────────────────────────
    /// Per-channel offset deltas added to `curve_offset` (log-exposure shift).
    /// Positive = brighter in that channel. Default 0.0.
    pub fp_offset_r: f32,
    pub fp_offset_g: f32,
    pub fp_offset_b: f32,
    /// Per-channel gamma multipliers applied to `curve_gamma`.
    /// 1.0 = same as global, <1 = softer (less contrast), >1 = harder. Default 1.0.
    pub fp_gamma_r: f32,
    pub fp_gamma_g: f32,
    pub fp_gamma_b: f32,
    /// Inter-channel density bleed (0.0–0.5). Mixes adjacent channel densities
    /// before the curve to simulate dye-layer crosstalk in real photo paper.
    /// 0.0 = no bleed, 0.1 = subtle, 0.3 = heavy. Default 0.08.
    pub fp_color_bleed: f32,
    /// Post-curve vibrance: luminance-aware saturation that boosts muted colors
    /// more than already-saturated ones. 0.0 = off, 1.0 = strong. Default 0.3.
    pub fp_vibrance: f32,
    /// Output rotation in degrees: 0, 90, 180, or 270 (applied after load/demosaic).
    pub rotation_degrees: i32,
    /// Flip horizontal (mirror left–right). Applied after rotation.
    pub flip_horizontal: bool,
    /// Flip vertical (mirror top–bottom). Applied after rotation.
    pub flip_vertical: bool,
    /// Debug: only run pipeline up to this step (1..=6). Preview and export use this. See TODO_DEBUG.md.
    pub debug_pipeline_step: u32,
    /// Density-domain saturation boost applied before the RA-4 curve.
    /// Scales per-channel density deviation from the neutral axis:
    ///   D_ch = D_mean + sat * (D_ch - D_mean)
    /// 1.0 = neutral (no change), >1.0 = more saturated. Default 1.2.
    /// Compensates for the S-curve's tendency to compress channel differences.
    pub saturation: f32,
    /// Toe shaping strength for the RA-4 curve. Positive = softer toe (opens
    /// shadows), negative = harder toe (deeper blacks). Dimensionless -1..1.
    pub toe_strength: f32,
    /// Shoulder shaping strength for the RA-4 curve. Positive = softer
    /// shoulder (protects highlights), negative = harder shoulder (snappier
    /// highlights). Dimensionless -1..1.
    pub shoulder_strength: f32,
    /// Shadow cast auto-correction strength. Detects residual per-channel color
    /// imbalance in the shadow (low-density) zone and corrects toward neutral.
    /// 0.0 = off, 1.0 = full correction. Applied after WB, before density matrix.
    pub shadow_cast_strength: f32,
    /// Gaussian-masked shadow zone density offset. Positive = brighten shadows
    /// (adds density in low-D region → brighter through RA-4), negative = darken.
    /// Applied per-pixel weighted by a Gaussian centered on the shadow zone.
    pub zone_shadows: f32,
    /// Gaussian-masked highlight zone density offset. Positive = brighten highlights
    /// (adds density in high-D region), negative = darken highlights.
    pub zone_highlights: f32,
    /// Per-zone gain (multiplicative). 0 = no change; e.g. 0.2 = 20% brighter in that zone.
    /// Applied before zone offsets. Shadow/mid/highlight zones use the same Gaussian masks as offsets.
    pub zone_shadow_gain: f32,
    pub zone_mid_gain: f32,
    pub zone_highlight_gain: f32,
    /// Per-channel gain in each zone. 0 = no change. Combines with global zone gain per channel.
    pub color_shadow_gain_r: f32,
    pub color_shadow_gain_g: f32,
    pub color_shadow_gain_b: f32,
    pub color_mid_gain_r: f32,
    pub color_mid_gain_g: f32,
    pub color_mid_gain_b: f32,
    pub color_highlight_gain_r: f32,
    pub color_highlight_gain_g: f32,
    pub color_highlight_gain_b: f32,
    /// Per-zone saturation (1.0 = no change). Blended by shadow/mid/highlight masks.
    pub zone_shadow_saturation: f32,
    pub zone_mid_saturation: f32,
    pub zone_highlight_saturation: f32,
    /// Reinhard-style highlight roll-off in density space (0.0 = off, 1.0 = full).
    /// Compresses high densities to mask noise in dense negative areas (skies).
    pub highlight_rolloff: f32,
    /// Knee density for highlight roll-off. Densities above this are compressed toward it. Default 1.5.
    pub highlight_rolloff_d_mid: f32,
    /// Post-curve highlight warmth (Noritsu/Frontier style).
    /// Adds a golden/warm tint to neutral highlights while leaving saturated
    /// colors (blue sky, red etc.) untouched.
    /// 0.0 = neutral, 0.3–0.6 = subtle lab-scan warmth. Default 0.4.
    pub highlight_warmth: f32,
    /// Post-curve soft knee for specular highlights (display space).
    /// Value in [0.6, 0.99]: lower = earlier/stronger rolloff, higher = later/subtler.
    /// 0.0 or >= 0.999 = effectively disabled.
    pub soft_clip: f32,
    /// Enable display-space Lab adjustments (separation, future vibrance).
    pub apply_lab: bool,
    /// Lab-space separation strength (0 = off). Boosts mid-chroma colors in
    /// the a/b plane to increase color separation without affecting neutrals.
    pub lab_separation: f32,
    /// Skin magenta shift (0 = off). Rotates magenta/red hues in LAB toward orange
    /// to correct scanner cast in lips and eye areas. 0.3–0.8 typical.
    pub skin_magenta_shift: f32,
    /// When true and input is PNG/TIFF (not RAW), invert values before T→D: T = 1 − V (per channel).
    /// Use for synthetic negatives that store "display negative" (positive with orange) instead of
    /// transmittance-like camera scan values. Pipeline then treats inverted values as transmittance.
    pub synthetic_negative_input: bool,
    /// Debug preview mode: for RAW files, show only a simple bilinear demosaic
    /// (plus optional rotation) and skip the rest of the pipeline.
    pub debug_preview_simple_debayer: bool,
    /// When true, compute per-step channel statistics (min/max/median) in the
    /// debug log. Expensive (sorts entire image per channel per step). Only
    /// enable when the Debug tab is active.
    pub verbose_debug: bool,
    /// When true (and the `gpu` feature is enabled), offload eligible pipeline steps to the GPU.
    pub use_gpu: bool,
    /// De-Bujack: non-local OkLab difference stretch after the output transform.
    /// Off by default. Runs after step 6, before encode.
    pub bujack_enabled: bool,
    /// Lightness knee k_L. Where L-differences start to flatten. Smaller = more aggressive.
    pub bujack_k_l: f32,
    /// Chroma knee k_C. Same knee, applied to the (a,b) chroma vector.
    pub bujack_k_c: f32,
    /// Dry/wet mix. 0 = off, 1 = full inverse-response, >1 over-corrects.
    pub bujack_strength: f32,
    /// Bilateral radius in pixels of the current buffer (preview or export).
    pub bujack_radius: f32,
    /// Bilateral range σ in OkLab. Low keeps edges out of the base (less halo).
    pub bujack_edge: f32,
    /// When set, zone masks use these full-frame percentiles (d_min, p33, p66, d_max)
    /// at curve_offset 0 instead of measuring the current buffer. Preview/tiles only.
    pub pinned_zone: Option<(f32, f32, f32, f32)>,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            dmin_mode: DminMode::AutoPercentile,
            auto_norm_buffer: 0.2,
            apply_white_balance: true,
            auto_wb: true,
            wb_mode: WbMode::Auto,
            film_gamma: 0.65,
            dmin_rect: None,
            dmin_rect_reference_size: None,
            apply_crop: false,
            crop_rect: None,
            crop_rect_reference_size: None,
            dmin_fixed: None,
            dmin_neutral_only: false,
            format: TiffFormat::Float32,
            write_exr: false,
            write_jpeg: false,
            write_jpeg_only: false,
            no_invert: false,
            no_curve: false,
            wb_r: 1.0,
            wb_g: 1.0,
            wb_b: 1.0,
            temp_k: None,
            curve_offset: 0.0,
            curve_gamma: 2.5,
            curve_pivot: 3.0,
            curve_white: 1.0,
            apply_color_profile: false,
            density_matrix: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            idt_matrix: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            flat_field_path: None,
            export_aces_exr: false,
            write_aces2065_only: false,
            lut3d_path: None,
            output_stage: OutputStage::Ra4,
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
            soft_clip: 0.93,
            apply_lab: true,
            lab_separation: 1.0,
            skin_magenta_shift: 0.0,
            rotation_degrees: 0,
            flip_horizontal: false,
            flip_vertical: false,
            synthetic_negative_input: false,
            debug_pipeline_step: 6,
            debug_preview_simple_debayer: false,
            verbose_debug: false,
            use_gpu: false,
            bujack_enabled: false,
            bujack_k_l: 0.25,
            bujack_k_c: 0.30,
            bujack_strength: 0.2,
            bujack_radius: 16.0,
            bujack_edge: 0.25,
            pinned_zone: None,
        }
    }
}
