//! Step-level cache for preview: reuse results of pipeline stages so only steps
//! after a changed option are re-run. Used by the GUI for fast preview updates.

use std::hash::{Hash, Hasher};
use std::path::Path;

use ndarray::Array3;

use crate::options::{PipelineOptions, Rect};

/// Cache slots for each pipeline stage. Each slot stores the hash that produced it
/// and the image buffer after that step (so we can run from the next step).
#[derive(Clone, Default)]
pub struct PreviewStepCache {
    /// After load + demosaic + rotate. Includes true source (pre-rotation) dimensions.
    pub after_load: Option<(u64, Array3<f32>, u32, u32)>,
    /// After step 3 (D-min / flat-field).
    pub after_step3: Option<(u64, Array3<f32>)>,
    /// After step 4 (T→D, WB, film γ, shadow cast).
    pub after_step4: Option<(u64, Array3<f32>)>,
    /// After step 5 (density matrix / LUT, saturation, zones).
    pub after_step5: Option<(u64, Array3<f32>)>,
}

fn hash_f32(h: &mut impl Hasher, v: f32) {
    v.to_bits().hash(h);
}

fn hash_rect(h: &mut impl Hasher, r: &Rect) {
    r.x.hash(h);
    r.y.hash(h);
    r.width.hash(h);
    r.height.hash(h);
}

/// Hash of everything that affects output after load (path, rotation, preview size, simple-debayer).
/// Used as cache key for the "after load" slot.
pub fn hash_after_load(
    path: &Path,
    opts: &PipelineOptions,
    max_width: u32,
    max_height: u32,
) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    path.display().to_string().hash(&mut h);
    opts.rotation_degrees.hash(&mut h);
    opts.flip_horizontal.hash(&mut h);
    opts.flip_vertical.hash(&mut h);
    opts.synthetic_negative_input.hash(&mut h);
    max_width.hash(&mut h);
    max_height.hash(&mut h);
    opts.debug_preview_simple_debayer.hash(&mut h);
    for row in &opts.idt_matrix {
        for &v in row {
            hash_f32(&mut h, v);
        }
    }
    h.finish()
}

/// Hash for "after step 3": everything that affects steps 1..3 (load + D-min/flat-field).
pub fn hash_after_step3(
    path: &Path,
    opts: &PipelineOptions,
    max_width: u32,
    max_height: u32,
) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    hash_after_load(path, opts, max_width, max_height).hash(&mut h);
    opts.dmin_mode.hash(&mut h);
    opts.dmin_rect_reference_size.hash(&mut h);
    opts.dmin_neutral_only.hash(&mut h);
    opts.auto_norm_buffer.to_bits().hash(&mut h);
    if let Some(r) = opts.dmin_rect {
        hash_rect(&mut h, &r);
    }
    if let Some((r, g, b)) = opts.dmin_fixed {
        hash_f32(&mut h, r);
        hash_f32(&mut h, g);
        hash_f32(&mut h, b);
    }
    opts.flat_field_path
        .as_ref()
        .map(|p| p.display().to_string())
        .hash(&mut h);
    opts.dust_mask_hash.hash(&mut h);
    opts.dust_heal.detect.to_bits().hash(&mut h);
    opts.dust_heal.feather.to_bits().hash(&mut h);
    opts.dust_heal.grain.to_bits().hash(&mut h);
    opts.dust_heal.grain_sigma.to_bits().hash(&mut h);
    opts.dust_heal.infill.hash(&mut h);
    h.finish()
}

/// Hash for "after step 4": everything that affects steps 1..4 (+ WB, film γ, shadow cast).
pub fn hash_after_step4(
    path: &Path,
    opts: &PipelineOptions,
    max_width: u32,
    max_height: u32,
) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    hash_after_step3(path, opts, max_width, max_height).hash(&mut h);
    opts.wb_mode.hash(&mut h);
    opts.auto_wb.hash(&mut h);
    opts.apply_white_balance.hash(&mut h);
    hash_f32(&mut h, opts.wb_r);
    hash_f32(&mut h, opts.wb_g);
    hash_f32(&mut h, opts.wb_b);
    hash_f32(&mut h, opts.film_gamma);
    opts.temp_k.map(|k| hash_f32(&mut h, k));
    hash_f32(&mut h, opts.shadow_cast_strength);
    h.finish()
}

/// Hash for "after step 5": everything that affects steps 1..5 (+ matrix, LUT, saturation, zones).
pub fn hash_after_step5(
    path: &Path,
    opts: &PipelineOptions,
    max_width: u32,
    max_height: u32,
) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    hash_after_step4(path, opts, max_width, max_height).hash(&mut h);
    opts.apply_color_profile.hash(&mut h);
    for row in &opts.density_matrix {
        for &v in row {
            hash_f32(&mut h, v);
        }
    }
    opts.lut3d_path
        .as_ref()
        .map(|p| p.display().to_string())
        .hash(&mut h);
    hash_f32(&mut h, opts.saturation);
    hash_f32(&mut h, opts.zone_shadows);
    hash_f32(&mut h, opts.zone_highlights);
    hash_f32(&mut h, opts.zone_shadow_gain);
    hash_f32(&mut h, opts.zone_mid_gain);
    hash_f32(&mut h, opts.zone_highlight_gain);
    hash_f32(&mut h, opts.color_shadow_gain_r);
    hash_f32(&mut h, opts.color_shadow_gain_g);
    hash_f32(&mut h, opts.color_shadow_gain_b);
    hash_f32(&mut h, opts.color_mid_gain_r);
    hash_f32(&mut h, opts.color_mid_gain_g);
    hash_f32(&mut h, opts.color_mid_gain_b);
    hash_f32(&mut h, opts.color_highlight_gain_r);
    hash_f32(&mut h, opts.color_highlight_gain_g);
    hash_f32(&mut h, opts.color_highlight_gain_b);
    hash_f32(&mut h, opts.zone_shadow_saturation);
    hash_f32(&mut h, opts.zone_mid_saturation);
    hash_f32(&mut h, opts.zone_highlight_saturation);
    hash_f32(&mut h, opts.highlight_rolloff);
    hash_f32(&mut h, opts.highlight_rolloff_d_mid);
    opts.debug_pipeline_step.hash(&mut h);
    h.finish()
}

/// Earliest pipeline step that must be re-run given `cache` and current options.
///
/// `1` = nothing usable (reload), `3` = from D-min, `4`/`5`/`6` = from that step.
pub fn cached_start_step(
    path: &Path,
    opts: &PipelineOptions,
    max_width: u32,
    max_height: u32,
    cache: &PreviewStepCache,
) -> u8 {
    let h1 = hash_after_load(path, opts, max_width, max_height);
    let h3 = hash_after_step3(path, opts, max_width, max_height);
    let h4 = hash_after_step4(path, opts, max_width, max_height);
    let h5 = hash_after_step5(path, opts, max_width, max_height);
    let mut start = 1u8;
    if cache.after_load.as_ref().is_some_and(|(h, ..)| *h == h1) {
        start = 3;
    }
    if start <= 3 && cache.after_step3.as_ref().is_some_and(|(h, _)| *h == h3) {
        start = 4;
    }
    if start <= 4 && cache.after_step4.as_ref().is_some_and(|(h, _)| *h == h4) {
        start = 5;
    }
    if start <= 5 && cache.after_step5.as_ref().is_some_and(|(h, _)| *h == h5) {
        start = 6;
    }
    start
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::DminMode;
    use crate::PipelineOptions;
    use ndarray::Array3;

    #[test]
    fn hash_after_load_deterministic() {
        let opts = PipelineOptions::default();
        let h1 = hash_after_load(Path::new("/a.raw"), &opts, 800, 600);
        let h2 = hash_after_load(Path::new("/a.raw"), &opts, 800, 600);
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_after_step3_differs_with_dust() {
        let mut opts = PipelineOptions::default();
        let h_off = hash_after_step3(Path::new("/a.raw"), &opts, 800, 600);
        opts.dust_mask_hash = 42;
        let h_dust = hash_after_step3(Path::new("/a.raw"), &opts, 800, 600);
        assert_ne!(h_off, h_dust);
    }

    #[test]
    fn hash_after_step3_differs_with_dmin() {
        let mut opts = PipelineOptions::default();
        let h_off = hash_after_step3(Path::new("/a.raw"), &opts, 800, 600);
        opts.dmin_mode = DminMode::Fixed;
        opts.dmin_fixed = Some((0.1, 0.2, 0.3));
        let h_fixed = hash_after_step3(Path::new("/a.raw"), &opts, 800, 600);
        assert_ne!(h_off, h_fixed);
    }

    #[test]
    fn hash_after_step5_unchanged_by_step6_curve() {
        let mut opts = PipelineOptions::default();
        let path = Path::new("/a.raw");
        let h5 = hash_after_step5(path, &opts, 800, 600);
        opts.curve_offset = 0.25;
        opts.curve_gamma = 1.4;
        opts.lut_in_mid = 1.1;
        opts.toe_strength = 0.2;
        assert_eq!(h5, hash_after_step5(path, &opts, 800, 600));
    }

    #[test]
    fn cached_start_step_is_6_when_only_curve_changes() {
        let path = Path::new("/a.raw");
        let opts = PipelineOptions::default();
        let buf = Array3::<f32>::zeros((4, 4, 3));
        let cache = PreviewStepCache {
            after_step5: Some((hash_after_step5(path, &opts, 800, 600), buf)),
            ..PreviewStepCache::default()
        };
        let mut live = opts.clone();
        live.curve_offset = 0.3;
        assert_eq!(cached_start_step(path, &live, 800, 600, &cache), 6);
    }

    #[test]
    fn cached_start_step_is_5_when_saturation_changes() {
        let path = Path::new("/a.raw");
        let opts = PipelineOptions::default();
        let buf = Array3::<f32>::zeros((4, 4, 3));
        let cache = PreviewStepCache {
            after_step4: Some((hash_after_step4(path, &opts, 800, 600), buf.clone())),
            after_step5: Some((hash_after_step5(path, &opts, 800, 600), buf)),
            ..PreviewStepCache::default()
        };
        let mut live = opts.clone();
        live.saturation = 0.4;
        assert_eq!(cached_start_step(path, &live, 800, 600, &cache), 5);
    }
}
