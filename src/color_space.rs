//! Explicit color-state types and conversions for the C-41 pipeline.
//!
//! Every buffer in the pipeline occupies one `(domain, primaries, transfer)` state.
//! Conversions must name both the source and destination. Silent reinterpretation
//! (linear print RGB shown as sRGB-encoded, density treated as ACEScg) is what
//! produced the purple/magenta display cast.
//!
//! Canonical path (identity IDT, default):
//!
//! ```text
//! RAW  → Camera linear T          (domain=T, primaries=Camera, transfer=Linear)
//! PNG8 → Rec.709 linear T         (sRGB decode on 8-bit; 16-bit/float stay linear)
//! IDT  → Rec.709 linear T         (only when idt_matrix ≠ I: Camera→ACEScg→Rec.709)
//! Dmin → Rec.709/Camera linear T
//! T→D  → Density                  (D = −log10(T); inversion lives here)
//! RA-4 → Rec.709/Camera linear print
//! Out  → sRGB encoded             (OETF at preview / JPEG / 16-bit display TIFF)
//! EXR  → linear print             (no OETF)
//! ```

use ndarray::Array3;

use crate::aces;

/// What the sample values represent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorDomain {
    /// Linear transmittance (film + base), typically after D-min.
    Transmittance,
    /// Optical density D = −log10(T). High D = more dye = brighter print.
    Density,
    /// Scene/print-referred linear RGB (RA-4 / FilmPrint output).
    LinearPrint,
    /// Display-referred code values (sRGB OETF or LUT cube codes).
    DisplayEncoded,
}

/// RGB primaries of the working buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorPrimaries {
    /// Demosaiced camera RGB. Not Rec.709 unless an IDT was applied.
    Camera,
    /// Rec.709 / sRGB primaries, D65.
    Rec709,
    /// ACEScg (AP1), linear.
    ACEScg,
    /// ACES2065-1 (AP0), linear.
    ACES2065,
}

/// Transfer function of the stored samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferFn {
    Linear,
    /// IEC 61966-2-1 sRGB OETF (display encode) or EOTF (decode).
    Srgb,
    /// Optical density (log10).
    DensityLog,
}

/// Tagged color state. Used for documentation and debug checks, not stored per pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorState {
    pub domain: ColorDomain,
    pub primaries: ColorPrimaries,
    pub transfer: TransferFn,
}

impl ColorState {
    pub const CAMERA_LINEAR_T: Self = Self {
        domain: ColorDomain::Transmittance,
        primaries: ColorPrimaries::Camera,
        transfer: TransferFn::Linear,
    };
    pub const REC709_LINEAR_T: Self = Self {
        domain: ColorDomain::Transmittance,
        primaries: ColorPrimaries::Rec709,
        transfer: TransferFn::Linear,
    };
    pub const CAMERA_DENSITY: Self = Self {
        domain: ColorDomain::Density,
        primaries: ColorPrimaries::Camera,
        transfer: TransferFn::DensityLog,
    };
    pub const REC709_DENSITY: Self = Self {
        domain: ColorDomain::Density,
        primaries: ColorPrimaries::Rec709,
        transfer: TransferFn::DensityLog,
    };
    pub const LINEAR_PRINT: Self = Self {
        domain: ColorDomain::LinearPrint,
        primaries: ColorPrimaries::Rec709,
        transfer: TransferFn::Linear,
    };
    pub const SRGB_DISPLAY: Self = Self {
        domain: ColorDomain::DisplayEncoded,
        primaries: ColorPrimaries::Rec709,
        transfer: TransferFn::Srgb,
    };
}

/// Working-space primaries after the optional IDT hop.
pub fn working_primaries(idt: &[[f32; 3]; 3]) -> ColorPrimaries {
    if aces::is_identity(idt) {
        ColorPrimaries::Camera
    } else {
        ColorPrimaries::Rec709
    }
}

/// sRGB OETF (linear → encoded). Input is clamped to [0, 1].
#[inline]
pub fn linear_to_srgb(v: f32) -> f32 {
    let x = v.clamp(0.0, 1.0);
    if x <= 0.003_130_8 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

/// sRGB EOTF (encoded → linear). Input is clamped to [0, 1].
#[inline]
pub fn srgb_to_linear(v: f32) -> f32 {
    let x = v.clamp(0.0, 1.0);
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear [0, 1] → 8-bit sRGB code value.
#[inline]
pub fn linear_to_srgb_u8(v: f32) -> u8 {
    (linear_to_srgb(v) * 255.0).round() as u8
}

/// Linear [0, 1] → 16-bit sRGB code value.
#[inline]
pub fn linear_to_srgb_u16(v: f32) -> u16 {
    (linear_to_srgb(v) * 65535.0).round() as u16
}

/// Encode a linear print-RGB u16 buffer to sRGB in place (display TIFF).
pub fn encode_linear_u16_to_srgb(image: &mut Array3<u16>) {
    let (h, w, c) = image.dim();
    assert_eq!(c, 3);
    let scale = 1.0 / 65535.0_f32;
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let lin = image[[y, x, ch]] as f32 * scale;
                image[[y, x, ch]] = linear_to_srgb_u16(lin);
            }
        }
    }
}

/// Apply a camera IDT and convert into Rec.709 linear working space.
///
/// Identity IDT is a no-op: the buffer stays camera RGB. A real IDT is
/// camera → ACEScg, then ACEScg → linear Rec.709. The ACEScg→sRGB matrix is
/// never applied to identity data (that was the old teal/cyan bug).
pub fn apply_input_idt_to_working_space(image: &mut Array3<f32>, idt: &[[f32; 3]; 3]) {
    if aces::is_identity(idt) {
        return;
    }
    aces::apply_idt(image, idt);
    aces::linear_acescg_to_linear_srgb(image);
    image.mapv_inplace(|v| v.max(0.0));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aces::{ACESCG_TO_LINEAR_SRGB, IDT_IDENTITY};

    #[test]
    fn oetf_roundtrip_mid_gray() {
        let lin = 0.18;
        let enc = linear_to_srgb(lin);
        let back = srgb_to_linear(enc);
        assert!((back - lin).abs() < 1e-5, "roundtrip {back} vs {lin}");
    }

    #[test]
    fn identity_idt_is_noop() {
        let mut img = Array3::<f32>::from_shape_fn((1, 1, 3), |(_, _, c)| [0.4, 0.5, 0.6][c]);
        apply_input_idt_to_working_space(&mut img, &IDT_IDENTITY);
        assert!((img[[0, 0, 0]] - 0.4).abs() < 1e-6);
        assert!((img[[0, 0, 1]] - 0.5).abs() < 1e-6);
        assert!((img[[0, 0, 2]] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn acescg_to_srgb_preserves_neutral() {
        let mut img = Array3::<f32>::from_elem((1, 1, 3), 0.5);
        aces::apply_idt(&mut img, &ACESCG_TO_LINEAR_SRGB);
        assert!((img[[0, 0, 0]] - 0.5).abs() < 1e-4);
        assert!((img[[0, 0, 1]] - 0.5).abs() < 1e-4);
        assert!((img[[0, 0, 2]] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn linear_as_srgb_exaggerates_green_deficit() {
        // A slight G deficit in linear, displayed without OETF, is decoded by
        // the monitor as sRGB. That per-channel gamma deepens magenta/purple.
        let r: f32 = 0.40;
        let g: f32 = 0.34;
        let b: f32 = 0.40;
        let displayed_r = r.powf(2.2);
        let displayed_g = g.powf(2.2);
        let displayed_b = b.powf(2.2);
        let lin_rg = r / g;
        let disp_rg = displayed_r / displayed_g;
        assert!(
            disp_rg > lin_rg,
            "display gamma should increase R/G ({disp_rg} vs {lin_rg})"
        );
        let _ = displayed_b;
    }
}
