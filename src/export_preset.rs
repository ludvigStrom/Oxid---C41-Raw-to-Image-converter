//! Export-tab presets: format, ICC, intent, JPEG quality, filename template.
//!
//! Look knobs stay in [`crate::preset::DevelopPreset`]. Paper/printer proof
//! settings are a view option and are not stored here.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::options::{CmsIntent, OutputIcc, PipelineOptions};
use crate::tiff_export::TiffFormat;

pub const EXPORT_PRESET_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExportPresetFormat {
    #[default]
    Tiff16,
    Tiff32,
    Exr,
    Jpeg,
    ExrAces2065,
}

impl ExportPresetFormat {
    pub fn from_options(opts: &PipelineOptions) -> Self {
        if opts.write_aces2065_only {
            Self::ExrAces2065
        } else if opts.write_jpeg_only {
            Self::Jpeg
        } else if opts.write_exr {
            Self::Exr
        } else if opts.format == TiffFormat::Float32 {
            Self::Tiff32
        } else {
            Self::Tiff16
        }
    }

    pub fn apply_to(self, opts: &mut PipelineOptions) {
        match self {
            Self::Tiff16 => {
                opts.format = TiffFormat::U16;
                opts.write_exr = false;
                opts.write_jpeg_only = false;
                opts.export_aces_exr = false;
                opts.write_aces2065_only = false;
            }
            Self::Tiff32 => {
                opts.format = TiffFormat::Float32;
                opts.write_exr = false;
                opts.write_jpeg_only = false;
                opts.export_aces_exr = false;
                opts.write_aces2065_only = false;
            }
            Self::Exr => {
                opts.format = TiffFormat::Float32;
                opts.write_exr = true;
                opts.write_jpeg_only = false;
                opts.export_aces_exr = false;
                opts.write_aces2065_only = false;
            }
            Self::Jpeg => {
                opts.format = TiffFormat::U16;
                opts.write_exr = false;
                opts.write_jpeg = false;
                opts.write_jpeg_only = true;
                opts.export_aces_exr = false;
                opts.write_aces2065_only = false;
            }
            Self::ExrAces2065 => {
                opts.format = TiffFormat::Float32;
                opts.write_exr = false;
                opts.write_jpeg = false;
                opts.write_jpeg_only = false;
                opts.export_aces_exr = false;
                opts.write_aces2065_only = true;
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ExportPreset {
    pub version: u32,
    pub name: String,
    pub format: ExportPresetFormat,
    pub write_jpeg: bool,
    pub jpeg_quality: u8,
    pub output_icc: OutputIcc,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_icc_path: Option<PathBuf>,
    pub export_intent: CmsIntent,
    pub export_bpc: bool,
    pub filename_template: String,
}

impl Default for ExportPreset {
    fn default() -> Self {
        Self::from_options(&PipelineOptions::default())
    }
}

impl ExportPreset {
    pub fn from_options(opts: &PipelineOptions) -> Self {
        Self {
            version: EXPORT_PRESET_VERSION,
            name: opts.export_preset_name.clone(),
            format: ExportPresetFormat::from_options(opts),
            write_jpeg: opts.write_jpeg && !opts.write_jpeg_only,
            jpeg_quality: opts.jpeg_quality,
            output_icc: opts.output_icc,
            output_icc_path: opts.output_icc_path.clone(),
            export_intent: opts.export_intent,
            export_bpc: opts.export_bpc,
            filename_template: opts.filename_template.clone(),
        }
    }

    /// Overwrite Export-tab fields only.
    pub fn apply_to(&self, opts: &mut PipelineOptions) {
        self.format.apply_to(opts);
        if !opts.write_jpeg_only && !opts.write_aces2065_only {
            opts.write_jpeg = self.write_jpeg;
        }
        opts.jpeg_quality = self.jpeg_quality.clamp(1, 100);
        opts.output_icc = self.output_icc;
        opts.output_icc_path = self.output_icc_path.clone();
        opts.export_intent = self.export_intent;
        opts.export_bpc = self.export_bpc;
        opts.filename_template = if self.filename_template.trim().is_empty() {
            "{stem}".to_string()
        } else {
            self.filename_template.clone()
        };
        opts.export_preset_name = self.name.clone();
    }
}

pub fn save_export_preset(opts: &PipelineOptions, path: &Path) -> anyhow::Result<()> {
    let mut preset = ExportPreset::from_options(opts);
    if preset.name.trim().is_empty() {
        preset.name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("export")
            .to_string();
    }
    let json = serde_json::to_string_pretty(&preset)?;
    std::fs::write(path, json)?;
    Ok(())
}

pub fn load_export_preset(path: &Path) -> anyhow::Result<ExportPreset> {
    let text = std::fs::read_to_string(path)?;
    let preset: ExportPreset = serde_json::from_str(&text)
        .map_err(|e| anyhow::anyhow!("Invalid export preset JSON: {e}"))?;
    Ok(preset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_leaves_develop_alone() {
        let mut opts = PipelineOptions::default();
        opts.curve_gamma = 3.1;
        opts.output_icc = OutputIcc::DisplayP3;
        opts.filename_template = "{date}_{stem}".to_string();
        opts.write_jpeg = true;
        opts.jpeg_quality = 90;

        let preset = ExportPreset::from_options(&opts);
        let mut applied = PipelineOptions::default();
        applied.curve_gamma = 3.1;
        preset.apply_to(&mut applied);

        assert_eq!(applied.output_icc, OutputIcc::DisplayP3);
        assert_eq!(applied.filename_template, "{date}_{stem}");
        assert!(applied.write_jpeg);
        assert_eq!(applied.jpeg_quality, 90);
        assert_eq!(applied.curve_gamma, 3.1);
    }
}
