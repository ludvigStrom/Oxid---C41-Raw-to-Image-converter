//! Develop-tab look presets: serialize / deserialize `PipelineOptions` fields
//! that belong in Develop (exposure, WB, color, zones, output, De-Bujack).
//!
//! Input geometry (crop, rotation), D-min sample rects, export format, and
//! debug flags are left unchanged when a preset is applied.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::options::{OutputLutEncoding, OutputStage, PipelineOptions, WbMode};

/// Current develop-preset schema version. Bump when the JSON shape changes
/// in a way that needs a migration in `load_develop_preset`.
pub const PRESET_VERSION: u32 = 1;

/// Serializable Develop-tab settings. Missing fields fall back to defaults
/// so older presets keep loading after new knobs are added.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DevelopPreset {
    pub version: u32,
    pub name: String,
    pub apply_white_balance: bool,
    pub auto_wb: bool,
    pub wb_mode: WbMode,
    pub wb_r: f32,
    pub wb_g: f32,
    pub wb_b: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temp_k: Option<f32>,
    pub curve_offset: f32,
    pub curve_gamma: f32,
    pub curve_pivot: f32,
    pub curve_white: f32,
    pub no_curve: bool,
    pub output_stage: OutputStage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_lut_cube: Option<PathBuf>,
    pub output_lut_encoding: OutputLutEncoding,
    pub lut_in_black: f32,
    pub lut_in_white: f32,
    pub lut_in_mid: f32,
    pub fp_offset_r: f32,
    pub fp_offset_g: f32,
    pub fp_offset_b: f32,
    pub fp_gamma_r: f32,
    pub fp_gamma_g: f32,
    pub fp_gamma_b: f32,
    pub fp_color_bleed: f32,
    pub fp_vibrance: f32,
    pub saturation: f32,
    pub toe_strength: f32,
    pub shoulder_strength: f32,
    pub shadow_cast_strength: f32,
    pub zone_shadows: f32,
    pub zone_highlights: f32,
    pub zone_shadow_gain: f32,
    pub zone_mid_gain: f32,
    pub zone_highlight_gain: f32,
    pub color_shadow_gain_r: f32,
    pub color_shadow_gain_g: f32,
    pub color_shadow_gain_b: f32,
    pub color_mid_gain_r: f32,
    pub color_mid_gain_g: f32,
    pub color_mid_gain_b: f32,
    pub color_highlight_gain_r: f32,
    pub color_highlight_gain_g: f32,
    pub color_highlight_gain_b: f32,
    pub zone_shadow_saturation: f32,
    pub zone_mid_saturation: f32,
    pub zone_highlight_saturation: f32,
    pub highlight_rolloff: f32,
    pub highlight_rolloff_d_mid: f32,
    pub highlight_warmth: f32,
    pub soft_clip: f32,
    pub apply_lab: bool,
    pub lab_separation: f32,
    pub skin_magenta_shift: f32,
    pub bujack_enabled: bool,
    pub bujack_k_l: f32,
    pub bujack_k_c: f32,
    pub bujack_strength: f32,
    pub bujack_radius: f32,
    pub bujack_edge: f32,
}

impl Default for DevelopPreset {
    fn default() -> Self {
        Self::from_options(&PipelineOptions::default())
    }
}

impl DevelopPreset {
    pub fn from_options(opts: &PipelineOptions) -> Self {
        Self {
            version: PRESET_VERSION,
            name: String::new(),
            apply_white_balance: opts.apply_white_balance,
            auto_wb: opts.auto_wb,
            wb_mode: opts.wb_mode,
            wb_r: opts.wb_r,
            wb_g: opts.wb_g,
            wb_b: opts.wb_b,
            temp_k: opts.temp_k,
            curve_offset: opts.curve_offset,
            curve_gamma: opts.curve_gamma,
            curve_pivot: opts.curve_pivot,
            curve_white: opts.curve_white,
            no_curve: opts.no_curve,
            output_stage: opts.output_stage,
            output_lut_cube: opts.output_lut_cube.clone(),
            output_lut_encoding: opts.output_lut_encoding,
            lut_in_black: opts.lut_in_black,
            lut_in_white: opts.lut_in_white,
            lut_in_mid: opts.lut_in_mid,
            fp_offset_r: opts.fp_offset_r,
            fp_offset_g: opts.fp_offset_g,
            fp_offset_b: opts.fp_offset_b,
            fp_gamma_r: opts.fp_gamma_r,
            fp_gamma_g: opts.fp_gamma_g,
            fp_gamma_b: opts.fp_gamma_b,
            fp_color_bleed: opts.fp_color_bleed,
            fp_vibrance: opts.fp_vibrance,
            saturation: opts.saturation,
            toe_strength: opts.toe_strength,
            shoulder_strength: opts.shoulder_strength,
            shadow_cast_strength: opts.shadow_cast_strength,
            zone_shadows: opts.zone_shadows,
            zone_highlights: opts.zone_highlights,
            zone_shadow_gain: opts.zone_shadow_gain,
            zone_mid_gain: opts.zone_mid_gain,
            zone_highlight_gain: opts.zone_highlight_gain,
            color_shadow_gain_r: opts.color_shadow_gain_r,
            color_shadow_gain_g: opts.color_shadow_gain_g,
            color_shadow_gain_b: opts.color_shadow_gain_b,
            color_mid_gain_r: opts.color_mid_gain_r,
            color_mid_gain_g: opts.color_mid_gain_g,
            color_mid_gain_b: opts.color_mid_gain_b,
            color_highlight_gain_r: opts.color_highlight_gain_r,
            color_highlight_gain_g: opts.color_highlight_gain_g,
            color_highlight_gain_b: opts.color_highlight_gain_b,
            zone_shadow_saturation: opts.zone_shadow_saturation,
            zone_mid_saturation: opts.zone_mid_saturation,
            zone_highlight_saturation: opts.zone_highlight_saturation,
            highlight_rolloff: opts.highlight_rolloff,
            highlight_rolloff_d_mid: opts.highlight_rolloff_d_mid,
            highlight_warmth: opts.highlight_warmth,
            soft_clip: opts.soft_clip,
            apply_lab: opts.apply_lab,
            lab_separation: opts.lab_separation,
            skin_magenta_shift: opts.skin_magenta_shift,
            bujack_enabled: opts.bujack_enabled,
            bujack_k_l: opts.bujack_k_l,
            bujack_k_c: opts.bujack_k_c,
            bujack_strength: opts.bujack_strength,
            bujack_radius: opts.bujack_radius,
            bujack_edge: opts.bujack_edge,
        }
    }

    /// Overwrite Develop-tab fields. Crop, D-min, rotation, export, and debug
    /// state on `opts` are left as-is.
    pub fn apply_to(&self, opts: &mut PipelineOptions) {
        opts.apply_white_balance = self.apply_white_balance;
        opts.auto_wb = self.auto_wb;
        opts.wb_mode = self.wb_mode;
        opts.wb_r = self.wb_r;
        opts.wb_g = self.wb_g;
        opts.wb_b = self.wb_b;
        opts.temp_k = self.temp_k;
        opts.curve_offset = self.curve_offset;
        opts.curve_gamma = self.curve_gamma;
        opts.curve_pivot = self.curve_pivot;
        opts.curve_white = self.curve_white;
        opts.no_curve = self.no_curve;
        opts.output_stage = self.output_stage;
        opts.output_lut_cube = self.output_lut_cube.clone();
        opts.output_lut_encoding = self.output_lut_encoding;
        opts.lut_in_black = self.lut_in_black;
        opts.lut_in_white = self.lut_in_white;
        opts.lut_in_mid = self.lut_in_mid;
        opts.fp_offset_r = self.fp_offset_r;
        opts.fp_offset_g = self.fp_offset_g;
        opts.fp_offset_b = self.fp_offset_b;
        opts.fp_gamma_r = self.fp_gamma_r;
        opts.fp_gamma_g = self.fp_gamma_g;
        opts.fp_gamma_b = self.fp_gamma_b;
        opts.fp_color_bleed = self.fp_color_bleed;
        opts.fp_vibrance = self.fp_vibrance;
        opts.saturation = self.saturation;
        opts.toe_strength = self.toe_strength;
        opts.shoulder_strength = self.shoulder_strength;
        opts.shadow_cast_strength = self.shadow_cast_strength;
        opts.zone_shadows = self.zone_shadows;
        opts.zone_highlights = self.zone_highlights;
        opts.zone_shadow_gain = self.zone_shadow_gain;
        opts.zone_mid_gain = self.zone_mid_gain;
        opts.zone_highlight_gain = self.zone_highlight_gain;
        opts.color_shadow_gain_r = self.color_shadow_gain_r;
        opts.color_shadow_gain_g = self.color_shadow_gain_g;
        opts.color_shadow_gain_b = self.color_shadow_gain_b;
        opts.color_mid_gain_r = self.color_mid_gain_r;
        opts.color_mid_gain_g = self.color_mid_gain_g;
        opts.color_mid_gain_b = self.color_mid_gain_b;
        opts.color_highlight_gain_r = self.color_highlight_gain_r;
        opts.color_highlight_gain_g = self.color_highlight_gain_g;
        opts.color_highlight_gain_b = self.color_highlight_gain_b;
        opts.zone_shadow_saturation = self.zone_shadow_saturation;
        opts.zone_mid_saturation = self.zone_mid_saturation;
        opts.zone_highlight_saturation = self.zone_highlight_saturation;
        opts.highlight_rolloff = self.highlight_rolloff;
        opts.highlight_rolloff_d_mid = self.highlight_rolloff_d_mid;
        opts.highlight_warmth = self.highlight_warmth;
        opts.soft_clip = self.soft_clip;
        opts.apply_lab = self.apply_lab;
        opts.lab_separation = self.lab_separation;
        opts.skin_magenta_shift = self.skin_magenta_shift;
        opts.bujack_enabled = self.bujack_enabled;
        opts.bujack_k_l = self.bujack_k_l;
        opts.bujack_k_c = self.bujack_k_c;
        opts.bujack_strength = self.bujack_strength;
        opts.bujack_radius = self.bujack_radius;
        opts.bujack_edge = self.bujack_edge;
        crate::options::sync_wb_flags_from_mode(opts);
    }
}

/// Write a pretty-printed develop preset JSON. Uses `name` if set, otherwise
/// the file stem of `path`.
pub fn save_develop_preset(opts: &PipelineOptions, path: &Path) -> anyhow::Result<()> {
    let mut preset = DevelopPreset::from_options(opts);
    if preset.name.trim().is_empty() {
        preset.name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("preset")
            .to_string();
    }
    let json = serde_json::to_string_pretty(&preset)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Load a develop preset from JSON. Unknown fields are ignored; missing
/// fields use `PipelineOptions` defaults.
pub fn load_develop_preset(path: &Path) -> anyhow::Result<DevelopPreset> {
    let text = std::fs::read_to_string(path)?;
    let preset: DevelopPreset = serde_json::from_str(&text)
        .map_err(|e| anyhow::anyhow!("Invalid develop preset JSON: {e}"))?;
    Ok(preset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::{DminMode, Rect};

    #[test]
    fn roundtrip_preserves_develop_fields() {
        let mut opts = PipelineOptions::default();
        opts.curve_offset = 0.12;
        opts.curve_gamma = 3.1;
        opts.saturation = 1.25;
        opts.wb_mode = WbMode::Picker;
        opts.wb_r = 1.08;
        opts.temp_k = Some(4200.0);
        opts.output_stage = OutputStage::FilmPrint;
        opts.fp_vibrance = 0.55;
        opts.zone_shadow_gain = 0.1;
        opts.bujack_enabled = true;
        opts.highlight_rolloff = 0.8;

        let json = serde_json::to_string(&DevelopPreset::from_options(&opts)).unwrap();
        let loaded: DevelopPreset = serde_json::from_str(&json).unwrap();
        let mut applied = PipelineOptions::default();
        loaded.apply_to(&mut applied);

        assert_eq!(applied.curve_offset, 0.12);
        assert_eq!(applied.curve_gamma, 3.1);
        assert_eq!(applied.saturation, 1.25);
        assert_eq!(applied.wb_mode, WbMode::Picker);
        assert_eq!(applied.wb_r, 1.08);
        assert!(!applied.auto_wb);
        assert_eq!(applied.temp_k, None);
        assert_eq!(applied.output_stage, OutputStage::FilmPrint);
        assert_eq!(applied.fp_vibrance, 0.55);
        assert_eq!(applied.zone_shadow_gain, 0.1);
        assert!(applied.bujack_enabled);
        assert_eq!(applied.highlight_rolloff, 0.8);
    }

    #[test]
    fn apply_leaves_input_and_export_alone() {
        let mut opts = PipelineOptions::default();
        opts.apply_crop = true;
        opts.crop_rect = Some(Rect {
            x: 10,
            y: 20,
            width: 100,
            height: 80,
        });
        opts.rotation_degrees = 90;
        opts.flip_horizontal = true;
        opts.dmin_mode = DminMode::Fixed;
        opts.dmin_fixed = Some((0.2, 0.1, 0.05));
        opts.write_jpeg = true;
        opts.film_gamma = 0.72;
        opts.debug_pipeline_step = 4;

        let preset = DevelopPreset::from_options(&PipelineOptions::default());
        preset.apply_to(&mut opts);

        assert!(opts.apply_crop);
        assert_eq!(opts.crop_rect.unwrap().width, 100);
        assert_eq!(opts.rotation_degrees, 90);
        assert!(opts.flip_horizontal);
        assert_eq!(opts.dmin_mode, DminMode::Fixed);
        assert_eq!(opts.dmin_fixed, Some((0.2, 0.1, 0.05)));
        assert!(opts.write_jpeg);
        assert_eq!(opts.film_gamma, 0.72);
        assert_eq!(opts.debug_pipeline_step, 4);
    }

    #[test]
    fn missing_fields_use_defaults() {
        let loaded: DevelopPreset = serde_json::from_str(r#"{"name":"partial","saturation":1.4}"#)
            .unwrap();
        assert_eq!(loaded.name, "partial");
        assert_eq!(loaded.saturation, 1.4);
        assert_eq!(loaded.curve_gamma, PipelineOptions::default().curve_gamma);
        assert_eq!(loaded.wb_mode, WbMode::Auto);
        assert_eq!(loaded.output_stage, OutputStage::Ra4);
    }
}
