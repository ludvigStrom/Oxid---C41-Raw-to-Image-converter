//! Post-step-6 ICC color management (LittleCMS).
//!
//! Working space is linear Rec.709 print RGB. This module does not change
//! pipeline steps 4–6. sRGB export keeps the existing OETF + IEC profile so
//! current files stay bit-identical.

use std::path::Path;
use std::sync::OnceLock;

use anyhow::{bail, Context, Result};
use lcms2::{Flags, Intent, PixelFormat, Profile, ToneCurve, Transform, CIExyY, CIExyYTRIPLE};

use crate::color_space;
use crate::options::{CmsIntent, CmsTarget, OutputIcc, PipelineOptions};
use crate::pipeline::Step6Display;

const D65: CIExyY = CIExyY {
    x: 0.3127,
    y: 0.3290,
    Y: 1.0,
};

/// IEC 61966-2.1 parametric type-4 (sRGB / Display P3 transfer).
fn srgb_tone_curve() -> Result<ToneCurve> {
    ToneCurve::new_parametric(4, &[2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045])
        .map_err(|e| anyhow::anyhow!("sRGB tone curve: {e}"))
}

fn rgb_profile(white: &CIExyY, primaries: &CIExyYTRIPLE, curve: &ToneCurve) -> Result<Profile> {
    Profile::new_rgb(white, primaries, &[curve, curve, curve])
        .map_err(|e| anyhow::anyhow!("ICC RGB profile: {e}"))
}

fn linear_rec709_profile() -> Result<Profile> {
    let curve = ToneCurve::new(1.0);
    let primaries = CIExyYTRIPLE {
        Red: CIExyY {
            x: 0.64,
            y: 0.33,
            Y: 1.0,
        },
        Green: CIExyY {
            x: 0.30,
            y: 0.60,
            Y: 1.0,
        },
        Blue: CIExyY {
            x: 0.15,
            y: 0.06,
            Y: 1.0,
        },
    };
    rgb_profile(&D65, &primaries, &curve)
}

fn display_p3_profile() -> Result<Profile> {
    let curve = srgb_tone_curve()?;
    let primaries = CIExyYTRIPLE {
        Red: CIExyY {
            x: 0.680,
            y: 0.320,
            Y: 1.0,
        },
        Green: CIExyY {
            x: 0.265,
            y: 0.690,
            Y: 1.0,
        },
        Blue: CIExyY {
            x: 0.150,
            y: 0.060,
            Y: 1.0,
        },
    };
    rgb_profile(&D65, &primaries, &curve)
}

fn adobe_rgb_profile() -> Result<Profile> {
    // Adobe RGB (1998) specified gamma 2 + 51/256 = 2.19921875.
    let curve = ToneCurve::new(2.199_218_75);
    let primaries = CIExyYTRIPLE {
        Red: CIExyY {
            x: 0.6400,
            y: 0.3300,
            Y: 1.0,
        },
        Green: CIExyY {
            x: 0.2100,
            y: 0.7100,
            Y: 1.0,
        },
        Blue: CIExyY {
            x: 0.1500,
            y: 0.0600,
            Y: 1.0,
        },
    };
    rgb_profile(&D65, &primaries, &curve)
}

fn profile_icc(profile: &Profile) -> Result<Vec<u8>> {
    profile
        .icc()
        .map_err(|e| anyhow::anyhow!("serialize ICC: {e}"))
}

fn bundled_display_p3_icc() -> Result<&'static [u8]> {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    if BYTES.get().is_none() {
        let p = display_p3_profile()?;
        let _ = BYTES.set(profile_icc(&p)?);
    }
    Ok(BYTES.get().map(|v| v.as_slice()).unwrap_or(&[]))
}

fn bundled_adobe_rgb_icc() -> Result<&'static [u8]> {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    if BYTES.get().is_none() {
        let p = adobe_rgb_profile()?;
        let _ = BYTES.set(profile_icc(&p)?);
    }
    Ok(BYTES.get().map(|v| v.as_slice()).unwrap_or(&[]))
}

/// Short label for `{profile}` and the Export tab.
pub fn output_icc_label(kind: OutputIcc, custom: Option<&Path>) -> String {
    match kind {
        OutputIcc::Srgb => "sRGB".to_string(),
        OutputIcc::DisplayP3 => "Display P3".to_string(),
        OutputIcc::AdobeRgb => "Adobe RGB".to_string(),
        OutputIcc::Custom => custom
            .and_then(|p| p.file_stem())
            .and_then(|s| s.to_str())
            .unwrap_or("custom")
            .to_string(),
    }
}

/// ICC bytes to embed in an export file.
pub fn output_icc_bytes(kind: OutputIcc, custom: Option<&Path>) -> Result<Vec<u8>> {
    match kind {
        OutputIcc::Srgb => Ok(color_space::SRGB_ICC.to_vec()),
        OutputIcc::DisplayP3 => Ok(bundled_display_p3_icc()?.to_vec()),
        OutputIcc::AdobeRgb => Ok(bundled_adobe_rgb_icc()?.to_vec()),
        OutputIcc::Custom => {
            let path = custom.context("Custom output ICC selected but no file is set")?;
            std::fs::read(path)
                .with_context(|| format!("Failed to read ICC {}", path.display()))
        }
    }
}

fn load_profile_bytes(bytes: &[u8]) -> Result<Profile> {
    Profile::new_icc(bytes).map_err(|e| anyhow::anyhow!("Invalid ICC profile: {e}"))
}

fn lcms_intent(intent: CmsIntent) -> Intent {
    match intent {
        CmsIntent::Perceptual => Intent::Perceptual,
        CmsIntent::Relative => Intent::RelativeColorimetric,
        CmsIntent::Absolute => Intent::AbsoluteColorimetric,
    }
}

fn transform_flags(bpc: bool, gamut: bool, proof: bool) -> Flags {
    let mut flags = Flags::default();
    if bpc {
        flags = flags | Flags::BLACKPOINT_COMPENSATION;
    }
    if gamut {
        flags = flags | Flags::GAMUT_CHECK;
    }
    if proof {
        flags = flags | Flags::SOFT_PROOFING;
    }
    flags
}

fn as_rgb_f32(linear: &[f32]) -> Vec<[f32; 3]> {
    linear
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect()
}

fn linear_from_u16(img: &ndarray::Array3<u16>) -> Vec<f32> {
    img.iter().map(|v| *v as f32 / 65535.0).collect()
}

fn linear_from_encoded_f32(img: &ndarray::Array3<f32>) -> Vec<f32> {
    img.iter()
        .map(|v| color_space::srgb_to_linear(v.clamp(0.0, 1.0)))
        .collect()
}

fn transform_linear_to_u8(
    linear: &[f32],
    dest_icc: &[u8],
    intent: CmsIntent,
    bpc: bool,
) -> Result<Vec<u8>> {
    let src = linear_rec709_profile()?;
    let dest = load_profile_bytes(dest_icc)?;
    let xf: Transform<[f32; 3], [u8; 3]> = Transform::new_flags(
        &src,
        PixelFormat::RGB_FLT,
        &dest,
        PixelFormat::RGB_8,
        lcms_intent(intent),
        transform_flags(bpc, false, false),
    )
    .map_err(|e| anyhow::anyhow!("ICC transform: {e}"))?;
    let input = as_rgb_f32(linear);
    let mut out = vec![[0u8; 3]; input.len()];
    xf.transform_pixels(&input, &mut out);
    Ok(out.into_iter().flatten().collect())
}

fn transform_linear_to_u16(
    linear: &[f32],
    dest_icc: &[u8],
    intent: CmsIntent,
    bpc: bool,
) -> Result<Vec<u16>> {
    let src = linear_rec709_profile()?;
    let dest = load_profile_bytes(dest_icc)?;
    let xf: Transform<[f32; 3], [u16; 3]> = Transform::new_flags(
        &src,
        PixelFormat::RGB_FLT,
        &dest,
        PixelFormat::RGB_16,
        lcms_intent(intent),
        transform_flags(bpc, false, false),
    )
    .map_err(|e| anyhow::anyhow!("ICC transform: {e}"))?;
    let input = as_rgb_f32(linear);
    let mut out = vec![[0u16; 3]; input.len()];
    xf.transform_pixels(&input, &mut out);
    Ok(out.into_iter().flatten().collect())
}

fn proof_linear_to_u8(
    linear: &[f32],
    paper_icc: &[u8],
    monitor_icc: &[u8],
    proof_intent: CmsIntent,
    paper_white: bool,
    gamut: bool,
) -> Result<Vec<u8>> {
    let src = linear_rec709_profile()?;
    let paper = load_profile_bytes(paper_icc)?;
    let monitor = load_profile_bytes(monitor_icc)?;
    let display_intent = if paper_white {
        Intent::AbsoluteColorimetric
    } else {
        Intent::RelativeColorimetric
    };
    let xf: Transform<[f32; 3], [u8; 3]> = Transform::new_proofing(
        &src,
        PixelFormat::RGB_FLT,
        &monitor,
        PixelFormat::RGB_8,
        &paper,
        lcms_intent(proof_intent),
        display_intent,
        transform_flags(true, gamut, true),
    )
    .map_err(|e| anyhow::anyhow!("ICC proofing transform: {e}"))?;
    let input = as_rgb_f32(linear);
    let mut out = vec![[0u8; 3]; input.len()];
    xf.transform_pixels(&input, &mut out);
    Ok(out.into_iter().flatten().collect())
}

fn srgb_u8_from_u16(img: &ndarray::Array3<u16>) -> Vec<u8> {
    img.iter()
        .map(|v| color_space::linear_to_srgb_u8(*v as f32 / 65535.0))
        .collect()
}

fn clamp_f32_u8(img: &ndarray::Array3<f32>) -> Vec<u8> {
    img.iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
}

/// True when TIFF/JPEG should go through LittleCMS instead of the sRGB OETF.
pub fn uses_output_cms(options: &PipelineOptions) -> bool {
    !matches!(options.output_icc, OutputIcc::Srgb)
}

/// Encode linear Rec.709 print RGB to the output ICC as 8-bit + profile bytes.
pub fn export_u8_from_linear(
    linear: &[f32],
    options: &PipelineOptions,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let icc = output_icc_bytes(options.output_icc, options.output_icc_path.as_deref())?;
    if matches!(options.output_icc, OutputIcc::Srgb) {
        let rgb: Vec<u8> = linear
            .iter()
            .map(|v| color_space::linear_to_srgb_u8(*v))
            .collect();
        return Ok((rgb, icc));
    }
    let rgb = transform_linear_to_u8(
        linear,
        &icc,
        options.export_intent,
        options.export_bpc,
    )?;
    Ok((rgb, icc))
}

/// Encode linear Rec.709 print RGB to the output ICC as 16-bit + profile bytes.
pub fn export_u16_from_linear(
    linear: &[f32],
    options: &PipelineOptions,
) -> Result<(Vec<u16>, Vec<u8>)> {
    let icc = output_icc_bytes(options.output_icc, options.output_icc_path.as_deref())?;
    if matches!(options.output_icc, OutputIcc::Srgb) {
        let rgb: Vec<u16> = linear
            .iter()
            .map(|v| color_space::linear_to_srgb_u16(*v))
            .collect();
        return Ok((rgb, icc));
    }
    let rgb = transform_linear_to_u16(
        linear,
        &icc,
        options.export_intent,
        options.export_bpc,
    )?;
    Ok((rgb, icc))
}

fn display_u8_from_linear(linear: &[f32], options: &PipelineOptions) -> Result<Vec<u8>> {
    match options.cms_target {
        CmsTarget::Export => Ok(export_u8_from_linear(linear, options)?.0),
        CmsTarget::Display => {
            let monitor = options.preview_monitor_icc.as_deref().map(|v| v.as_slice());
            if options.soft_proof {
                let paper_path = options
                    .proof_icc_path
                    .as_ref()
                    .context("Soft proof is on but no paper/printer ICC is set")?;
                let paper = std::fs::read(paper_path).with_context(|| {
                    format!("Failed to read proof ICC {}", paper_path.display())
                })?;
                let dest = match monitor {
                    Some(m) if !m.is_empty() => m.to_vec(),
                    _ => color_space::SRGB_ICC.to_vec(),
                };
                return proof_linear_to_u8(
                    linear,
                    &paper,
                    &dest,
                    options.proof_intent,
                    options.proof_paper_white,
                    options.proof_gamut_warning,
                );
            }
            if let Some(monitor) = monitor {
                if !monitor.is_empty() {
                    return transform_linear_to_u8(
                        linear,
                        monitor,
                        CmsIntent::Relative,
                        true,
                    );
                }
            }
            Ok(linear
                .iter()
                .map(|v| color_space::linear_to_srgb_u8(*v))
                .collect())
        }
    }
}

/// Preview / contact-sheet encode. Falls back to the sRGB OETF on CMS errors
/// so a bad ICC never blanks the viewer.
pub fn encode_preview(display: &Step6Display, options: &PipelineOptions) -> Vec<u8> {
    match encode_preview_inner(display, options) {
        Ok(rgb) => rgb,
        Err(_) => crate::pipeline::step6_display_to_u8(display),
    }
}

fn encode_preview_inner(display: &Step6Display, options: &PipelineOptions) -> Result<Vec<u8>> {
    match display {
        Step6Display::PassthroughDensity(_) => {
            Ok(crate::pipeline::step6_display_to_u8(display))
        }
        Step6Display::U16(img) => {
            if !needs_cms_preview(options) {
                return Ok(srgb_u8_from_u16(img));
            }
            display_u8_from_linear(&linear_from_u16(img), options)
        }
        Step6Display::F32(img) => {
            if options.output_stage == crate::options::OutputStage::None {
                return Ok(clamp_f32_u8(img));
            }
            if !needs_cms_preview(options) {
                return Ok(clamp_f32_u8(img));
            }
            // Lut2383 is already display-referred; hop through linear Rec.709.
            display_u8_from_linear(&linear_from_encoded_f32(img), options)
        }
    }
}

fn needs_cms_preview(options: &PipelineOptions) -> bool {
    match options.cms_target {
        CmsTarget::Export => uses_output_cms(options),
        CmsTarget::Display => {
            options.soft_proof
                || options
                    .preview_monitor_icc
                    .as_ref()
                    .is_some_and(|m| !m.is_empty())
        }
    }
}

/// Export a RA-4 / FilmPrint u16 buffer through the output ICC.
pub fn export_from_u16(
    img: &ndarray::Array3<u16>,
    options: &PipelineOptions,
) -> Result<(Vec<u16>, Vec<u8>)> {
    export_u16_from_linear(&linear_from_u16(img), options)
}

/// Export already-encoded f32 display RGB (Lut2383) through the output ICC.
pub fn export_from_encoded_f32(
    img: &ndarray::Array3<f32>,
    options: &PipelineOptions,
) -> Result<(Vec<u8>, Vec<u8>)> {
    export_u8_from_linear(&linear_from_encoded_f32(img), options)
}

/// Parse a user ICC just enough to reject junk before export.
pub fn validate_icc_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("Failed to read ICC {}", path.display()))?;
    if bytes.len() < 128 || &bytes[36..40] != b"acsp" {
        bail!("Not a valid ICC profile: {}", path.display());
    }
    let profile = load_profile_bytes(&bytes)?;
    let name = profile
        .info(lcms2::InfoType::Description, lcms2::Locale::none())
        .unwrap_or_else(|| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("ICC")
                .to_string()
        });
    Ok(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_profiles_are_icc_v2_or_v4() {
        for bytes in [bundled_display_p3_icc().unwrap(), bundled_adobe_rgb_icc().unwrap()] {
            assert!(bytes.len() >= 128);
            assert_eq!(&bytes[36..40], b"acsp");
            let size = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
            assert_eq!(size, bytes.len());
        }
    }

    #[test]
    fn srgb_export_matches_hand_oetf() {
        let lin = [0.18f32, 0.18, 0.18];
        let opts = PipelineOptions::default();
        let (rgb, icc) = export_u8_from_linear(&lin, &opts).unwrap();
        assert_eq!(icc, color_space::SRGB_ICC);
        let expect = color_space::linear_to_srgb_u8(0.18);
        assert_eq!(rgb, [expect, expect, expect]);
    }

    #[test]
    fn p3_export_builds_and_embeds_p3() {
        let lin = [0.18f32, 0.18, 0.18];
        let mut opts = PipelineOptions::default();
        opts.output_icc = OutputIcc::DisplayP3;
        let (rgb, icc) = export_u8_from_linear(&lin, &opts).unwrap();
        assert_eq!(rgb.len(), 3);
        assert_eq!(&icc[36..40], b"acsp");
        assert_ne!(icc, color_space::SRGB_ICC);
        // Neutral mid-gray should stay roughly neutral.
        let max = rgb.iter().copied().max().unwrap().saturating_sub(rgb.iter().copied().min().unwrap());
        assert!(max < 8, "neutral drifted {rgb:?}");
    }

    #[test]
    fn proofing_transform_builds() {
        let lin = [0.4f32, 0.35, 0.3];
        let paper = bundled_adobe_rgb_icc().unwrap();
        let monitor = color_space::SRGB_ICC;
        let out = proof_linear_to_u8(&lin, paper, monitor, CmsIntent::Relative, true, false)
            .unwrap();
        assert_eq!(out.len(), 3);
    }
}
