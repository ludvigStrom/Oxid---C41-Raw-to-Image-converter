//! Shared pipeline step logic used by both batch export and GUI preview.
//!
//! Steps 3–6 are implemented here once; `process_files` and `process_one_to_preview`
//! call them and handle load/save or preview conversion.

use anyhow::Result;
use ndarray::Array3;

use crate::curve;

/// Minimum transmittance to avoid log(0). Same as curve::transmittance_to_density threshold.
const T_MIN: f32 = 1e-10;

/// Apply synthetic negative inversion: T = 1 − V (per channel), clamped to [T_MIN, 1].
/// Use when input is a synthetic negative that stores "display negative" (positive * orange in sRGB)
/// instead of transmittance. After inversion, pipeline treats values as transmittance for T→D.
pub fn apply_synthetic_negative_invert(image: &mut Array3<f32>) {
    image.mapv_inplace(|v| (1.0 - v).clamp(T_MIN, 1.0));
}
use crate::flat_field;
use crate::lut3d;
use crate::scale_dmin_rect;
use crate::DminMode;
use crate::OutputLutEncoding;
use crate::OutputStage;
use crate::PipelineOptions;

/// Result of step 6: either intermediate density (step < 6) or final display buffer.
pub enum Step6Display {
    /// Pipeline stopped before step 6; image is still density.
    PassthroughDensity(Array3<f32>),
    U16(Array3<u16>),
    F32(Array3<f32>),
}

/// Step 3: D-min / flat-field. Modifies image in place.
pub fn step_3_dmin(
    image: &mut Array3<f32>,
    options: &PipelineOptions,
    flat_field_map: Option<&Array3<f32>>,
) -> Result<()> {
    if options.debug_pipeline_step < 3 || options.dmin_mode == DminMode::Off {
        return Ok(());
    }
    if let Some(flat) = flat_field_map {
        flat_field::apply_flat_field_division(image, flat);
    } else {
        match options.dmin_mode {
            DminMode::Fixed => {
                if let Some((r, g, b)) = options.dmin_fixed {
                    crate::dmin::neutralize_with_medians(image, r, g, b)?;
                }
            }
            DminMode::SampleRegion => {
                if let Some(rect) = options.dmin_rect {
                    let (h, w, _) = image.dim();
                    let (x, y, rw, rh) =
                        scale_dmin_rect(rect, options.dmin_rect_reference_size, w as u32, h as u32);
                    crate::dmin::neutralize(image, x, y, rw, rh, options.dmin_neutral_only)?;
                }
            }
            DminMode::AutoPercentile => {
                crate::dmin::auto_percentile_normalize(image, options.auto_norm_buffer)?;
            }
            DminMode::Off => {}
        }
    }
    image.mapv_inplace(|v| v.max(0.0));
    Ok(())
}

/// Step 3 with GPU: flat-field divide and D-min divide on GPU; rect/percentile on CPU.
#[cfg(feature = "gpu")]
pub fn step_3_dmin_gpu(
    image: &mut Array3<f32>,
    options: &PipelineOptions,
    flat_field_map: Option<&Array3<f32>>,
    gpu_step3: Option<&crate::gpu::unified::Step3Gpu>,
) -> Result<()> {
    if options.debug_pipeline_step < 3 || options.dmin_mode == DminMode::Off {
        return Ok(());
    }

    if let Some(flat) = flat_field_map {
        let (h, w, _) = image.dim();
        let flat_resampled = flat_field::resize_flat_field(flat, h, w);
        if let Some(gpu) = gpu_step3 {
            if gpu.flat_field.run(image, &flat_resampled).is_ok() {
                image.mapv_inplace(|v| v.max(0.0));
                return Ok(());
            }
        }
        flat_field::apply_flat_field_division(image, &flat_resampled);
    } else {
        let (div_r, div_g, div_b) = match options.dmin_mode {
            DminMode::Fixed => {
                if let Some((r, g, b)) = options.dmin_fixed {
                    (r, g, b)
                } else {
                    (1.0, 1.0, 1.0)
                }
            }
            DminMode::SampleRegion => {
                if let Some(rect) = options.dmin_rect {
                    let (h, w, _) = image.dim();
                    let (x, y, rw, rh) =
                        scale_dmin_rect(rect, options.dmin_rect_reference_size, w as u32, h as u32);
                    crate::dmin::compute_neutralize_divisors(
                        image,
                        x,
                        y,
                        rw,
                        rh,
                        options.dmin_neutral_only,
                    )?
                } else {
                    (1.0, 1.0, 1.0)
                }
            }
            DminMode::AutoPercentile => {
                crate::dmin::compute_auto_percentile_divisors(image, options.auto_norm_buffer)?
            }
            DminMode::Off => (1.0, 1.0, 1.0),
        };

        if let Some(gpu) = gpu_step3 {
            let do_divide = (div_r - 1.0).abs() > 1e-9
                || (div_g - 1.0).abs() > 1e-9
                || (div_b - 1.0).abs() > 1e-9;
            if do_divide && gpu.step3_dmin.run(image, div_r, div_g, div_b).is_ok() {
                image.mapv_inplace(|v| v.max(0.0));
                return Ok(());
            }
        }

        crate::dmin::neutralize_with_medians(image, div_r, div_g, div_b)?;
    }

    image.mapv_inplace(|v| v.max(0.0));
    Ok(())
}

/// Step 4 on GPU (if available). Falls back to CPU if the GPU pass fails.
#[cfg(feature = "gpu")]
pub fn step_4_t_to_d_wb_gpu(
    image: &mut Array3<f32>,
    options: &PipelineOptions,
    gpu_step4: Option<&crate::gpu::step4::Step4Pipeline>,
) {
    if options.debug_pipeline_step < 4 {
        return;
    }
    if let Some(pipeline) = gpu_step4 {
        if pipeline.run(image, options).is_ok() {
            return;
        }
    }
    step_4_t_to_d_wb(image, options);
}

/// Step 4: T → D → WB (multiplicative) → Film γ → shadow cast. Modifies image in place.
pub fn step_4_t_to_d_wb(image: &mut Array3<f32>, options: &PipelineOptions) {
    if options.debug_pipeline_step < 4 {
        return;
    }
    image.mapv_inplace(|t| (-(t.max(1e-10_f32)).log10()).max(0.0));

    let (auto_s_r, auto_s_g, auto_s_b) = if options.auto_wb && options.dmin_mode != DminMode::Off {
        let stats = crate::stats::wb_channel_stats(image, options);
        let med_r = stats[0].2.max(1e-4);
        let med_g = stats[1].2.max(1e-4);
        let med_b = stats[2].2.max(1e-4);
        let mean_d = (med_r + med_g + med_b) / 3.0;
        (mean_d / med_r, mean_d / med_g, mean_d / med_b)
    } else {
        (1.0, 1.0, 1.0)
    };

    let (man_s_r, man_s_g, man_s_b) = if options.apply_white_balance {
        (options.wb_r, options.wb_g, options.wb_b)
    } else {
        (1.0, 1.0, 1.0)
    };

    let inv_gamma = 1.0 / options.film_gamma.max(0.1);
    let s_r = auto_s_r * man_s_r * inv_gamma;
    let s_g = auto_s_g * man_s_g * inv_gamma;
    let s_b = auto_s_b * man_s_b * inv_gamma;

    image
        .slice_mut(ndarray::s![.., .., 0])
        .mapv_inplace(|v| v * s_r);
    image
        .slice_mut(ndarray::s![.., .., 1])
        .mapv_inplace(|v| v * s_g);
    image
        .slice_mut(ndarray::s![.., .., 2])
        .mapv_inplace(|v| v * s_b);

    if let Some(k) = options.temp_k {
        let (tr, tg, tb) = crate::color::temp_k_to_wb_gains(k);
        let off_r = -(tr.max(1e-6) as f64).log10() as f32;
        let off_g = -(tg.max(1e-6) as f64).log10() as f32;
        let off_b = -(tb.max(1e-6) as f64).log10() as f32;
        image
            .slice_mut(ndarray::s![.., .., 0])
            .mapv_inplace(|v| v + off_r);
        image
            .slice_mut(ndarray::s![.., .., 1])
            .mapv_inplace(|v| v + off_g);
        image
            .slice_mut(ndarray::s![.., .., 2])
            .mapv_inplace(|v| v + off_b);
    }

    if options.shadow_cast_strength > 0.0 {
        let thresh = crate::density_ops::SHADOW_CAST_THRESHOLD;
        let cast = crate::density_ops::analyze_shadow_cast(image, thresh);
        crate::density_ops::apply_shadow_cast_correction(
            image,
            cast,
            options.shadow_cast_strength,
            thresh,
        );
    }
}

/// Step 5 on GPU (if available). Falls back to CPU if the GPU pass fails.
#[cfg(feature = "gpu")]
pub fn step_5_calibration_gpu(
    image: &mut Array3<f32>,
    options: &PipelineOptions,
    lut3d: Option<&lut3d::Lut3d>,
    gpu_step5: Option<&crate::gpu::step5::Step5Pipeline>,
) {
    if options.debug_pipeline_step < 5 {
        return;
    }
    if let Some(pipeline) = gpu_step5 {
        if pipeline.run(image, options, lut3d).is_ok() {
            return;
        }
    }
    step_5_calibration(image, options, lut3d);
}

fn step5_profile_matrix(options: &PipelineOptions) -> [[f32; 3]; 3] {
    if options.apply_color_profile {
        options.density_matrix
    } else {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }
}

fn step5_matrix_is_identity(m: [[f32; 3]; 3]) -> bool {
    (m[0][0] - 1.0).abs() < 1e-6
        && m[0][1].abs() < 1e-6
        && m[0][2].abs() < 1e-6
        && m[1][0].abs() < 1e-6
        && (m[1][1] - 1.0).abs() < 1e-6
        && m[1][2].abs() < 1e-6
        && m[2][0].abs() < 1e-6
        && m[2][1].abs() < 1e-6
        && (m[2][2] - 1.0).abs() < 1e-6
}

/// Matrix/LUT + clamp + highlight-spread for one density pixel. Same order as CPU step 5.
pub(crate) fn step5_prep_pixel(
    dr: f32,
    dg: f32,
    db: f32,
    options: &PipelineOptions,
    lut3d: Option<&lut3d::Lut3d>,
) -> [f32; 3] {
    let [r, g, b] = if let Some(lut) = lut3d {
        lut.sample_density(dr, dg, db)
    } else {
        let m = step5_profile_matrix(options);
        if step5_matrix_is_identity(m) {
            [dr, dg, db]
        } else {
            [
                m[0][0] * dr + m[0][1] * dg + m[0][2] * db,
                m[1][0] * dr + m[1][1] * dg + m[1][2] * db,
                m[2][0] * dr + m[2][1] * dg + m[2][2] * db,
            ]
        }
    };
    crate::density_ops::limit_highlight_density_spread_pixel(r.max(0.0), g.max(0.0), b.max(0.0))
}

/// Zone percentiles as CPU step 5 measures them: after matrix/LUT and highlight spread.
///
/// `from_transmittance` is for unified GPU start-at-step-4 (image is still T).
/// Pinned zones are used as-is.
pub(crate) fn zone_range_for_gpu_step5(
    image: &Array3<f32>,
    options: &PipelineOptions,
    lut3d: Option<&lut3d::Lut3d>,
    from_transmittance: bool,
) -> crate::density_ops::ZonePercentiles {
    if options.pinned_zone.is_some() {
        return crate::density_ops::zone_range_for_options(image, options);
    }
    if from_transmittance {
        let mut density = image.clone();
        step_4_t_to_d_wb(&mut density, options);
        return zone_range_after_step5_prep(&density, options, lut3d);
    }
    zone_range_after_step5_prep(image, options, lut3d)
}

fn zone_range_after_step5_prep(
    image: &Array3<f32>,
    options: &PipelineOptions,
    lut3d: Option<&lut3d::Lut3d>,
) -> crate::density_ops::ZonePercentiles {
    let (h, w, _) = image.dim();
    let n = h * w;
    if n == 0 {
        return crate::density_ops::zone_percentiles_from_samples(Vec::new());
    }
    let step = (n / 4096).max(1);
    let d_effs: Vec<f32> = (0..n)
        .step_by(step)
        .map(|i| {
            let y = i / w;
            let x = i % w;
            let [r, g, b] = step5_prep_pixel(
                image[[y, x, 0]],
                image[[y, x, 1]],
                image[[y, x, 2]],
                options,
                lut3d,
            );
            (r + g + b) / 3.0 + options.curve_offset
        })
        .collect();
    crate::density_ops::zone_percentiles_from_samples(d_effs)
}

/// Step 5: Density matrix / 3D LUT, limit highlight spread, saturation, zone adjustments.
pub fn step_5_calibration(
    image: &mut Array3<f32>,
    options: &PipelineOptions,
    lut3d: Option<&lut3d::Lut3d>,
) {
    if options.debug_pipeline_step < 5 {
        return;
    }
    let m = step5_profile_matrix(options);

    if let Some(lut) = lut3d {
        let (h, w, _) = image.dim();
        for y in 0..h {
            for x in 0..w {
                let dr = image[[y, x, 0]];
                let dg = image[[y, x, 1]];
                let db = image[[y, x, 2]];
                let [or, og, ob] = lut.sample_density(dr, dg, db);
                image[[y, x, 0]] = or;
                image[[y, x, 1]] = og;
                image[[y, x, 2]] = ob;
            }
        }
    } else if !step5_matrix_is_identity(m) {
        let (h, w, _) = image.dim();
        for y in 0..h {
            for x in 0..w {
                let dr = image[[y, x, 0]];
                let dg = image[[y, x, 1]];
                let db = image[[y, x, 2]];
                image[[y, x, 0]] = m[0][0] * dr + m[0][1] * dg + m[0][2] * db;
                image[[y, x, 1]] = m[1][0] * dr + m[1][1] * dg + m[1][2] * db;
                image[[y, x, 2]] = m[2][0] * dr + m[2][1] * dg + m[2][2] * db;
            }
        }
    }
    image.mapv_inplace(|v| v.max(0.0));
    crate::density_ops::limit_highlight_density_spread(image);

    let pinned_zone = Some(crate::density_ops::zone_range_for_options(image, options));
    crate::density_ops::apply_zone_density_saturation(
        image,
        options.curve_offset,
        options.saturation,
        options.zone_shadow_saturation,
        options.zone_mid_saturation,
        options.zone_highlight_saturation,
        pinned_zone,
    );
    crate::density_ops::apply_zone_density_adjustments(
        image,
        options.curve_offset,
        options.zone_shadows,
        options.zone_highlights,
        options.zone_shadow_gain,
        options.zone_mid_gain,
        options.zone_highlight_gain,
        [
            options.color_shadow_gain_r,
            options.color_shadow_gain_g,
            options.color_shadow_gain_b,
        ],
        [
            options.color_mid_gain_r,
            options.color_mid_gain_g,
            options.color_mid_gain_b,
        ],
        [
            options.color_highlight_gain_r,
            options.color_highlight_gain_g,
            options.color_highlight_gain_b,
        ],
        pinned_zone,
    );
    crate::density_ops::apply_reinhard_highlight_rolloff(
        image,
        options.highlight_rolloff_d_mid,
        options.highlight_rolloff,
    );
}

/// Step 6 on GPU (if available). Falls back to CPU if the GPU pass fails.
#[cfg(feature = "gpu")]
pub fn step_6_render_gpu(
    image: &Array3<f32>,
    options: &PipelineOptions,
    ra4_params: &curve::PrintCurveParams,
    output_lut_cube: Option<&lut3d::Lut3d>,
    gpu_step6: Option<&crate::gpu::step6::Step6Pipeline>,
) -> Step6Display {
    if options.debug_pipeline_step < 6 {
        return Step6Display::PassthroughDensity(image.clone());
    }
    if let Some(pipeline) = gpu_step6 {
        if let Ok(display) = pipeline.run(image, options, ra4_params, output_lut_cube) {
            return display;
        }
    }
    step_6_render(image, options, ra4_params, output_lut_cube)
}

fn density_levels_active(options: &PipelineOptions) -> bool {
    options.lut_in_black != 0.0
        || options.lut_in_white != 1.0
        || (options.lut_in_mid - 1.0).abs() > 1e-6
}

fn apply_ra4_levels_in_place(image: &mut Array3<f32>, options: &PipelineOptions) {
    crate::color::apply_density_levels(
        image,
        4.0,
        options.lut_in_black,
        options.lut_in_white,
        options.lut_in_mid,
    );
    image.mapv_inplace(|v| v * 4.0);
}

fn finish_u16_display(mut image_u16: Array3<u16>, options: &PipelineOptions) -> Step6Display {
    crate::post_curve::apply_toe_shoulder_u16(
        &mut image_u16,
        options.toe_strength,
        options.shoulder_strength,
    );
    crate::post_curve::apply_soft_knee_u16(&mut image_u16, options.soft_clip);
    if options.apply_lab {
        crate::color::apply_lab_separation_u16(&mut image_u16, options.lab_separation);
    }
    crate::color::apply_skin_magenta_shift_u16(&mut image_u16, options.skin_magenta_shift);
    crate::post_curve::apply_highlight_warmth_u16(&mut image_u16, options.highlight_warmth);
    Step6Display::U16(image_u16)
}

fn finish_none_display(mut display: Array3<f32>, options: &PipelineOptions) -> Step6Display {
    if !options.no_invert {
        const D_DISP_MAX: f32 = 2.5;
        display.mapv_inplace(|v| (v / D_DISP_MAX).clamp(0.0, 1.0));
    }
    Step6Display::F32(display)
}

fn finish_lut2383_display(
    mut display: Array3<f32>,
    options: &PipelineOptions,
    output_lut_cube: Option<&lut3d::Lut3d>,
) -> Step6Display {
    match options.output_lut_encoding {
        OutputLutEncoding::Rec709 => {
            crate::color::density_to_rec709_leveled(
                &mut display,
                options.lut_in_black,
                options.lut_in_white,
                options.lut_in_mid,
            );
        }
        enc => {
            let d_max = match enc {
                OutputLutEncoding::CineonLog => 2.046_f32,
                OutputLutEncoding::LinearDensity => 2.5_f32,
                OutputLutEncoding::Rec709 => unreachable!(),
            };
            crate::color::apply_density_levels(
                &mut display,
                d_max,
                options.lut_in_black,
                options.lut_in_white,
                options.lut_in_mid,
            );
        }
    }
    if let Some(lut) = output_lut_cube {
        crate::post_curve::apply_output_cube_rgb(&mut display, lut);
    }
    let encoded_srgb = options.output_lut_encoding == OutputLutEncoding::Rec709;
    if options.apply_lab {
        crate::color::apply_lab_separation_f32(&mut display, options.lab_separation, encoded_srgb);
    }
    crate::color::apply_skin_magenta_shift_f32(
        &mut display,
        options.skin_magenta_shift,
        encoded_srgb,
    );
    crate::post_curve::apply_soft_knee_f32(&mut display, options.soft_clip);
    crate::post_curve::apply_highlight_warmth_f32(&mut display, options.highlight_warmth);
    Step6Display::F32(display)
}

/// Step 6: render to display buffer (RA-4, FilmPrint, None, or Lut2383).
/// If debug_pipeline_step < 6, returns PassthroughDensity with the current image.
pub fn step_6_render(
    image: &Array3<f32>,
    options: &PipelineOptions,
    ra4_params: &curve::PrintCurveParams,
    output_lut_cube: Option<&lut3d::Lut3d>,
) -> Step6Display {
    if options.debug_pipeline_step < 6 {
        return Step6Display::PassthroughDensity(image.clone());
    }

    match options.output_stage {
        OutputStage::Ra4 => {
            if density_levels_active(options) {
                let mut leveled = image.clone();
                apply_ra4_levels_in_place(&mut leveled, options);
                finish_u16_display(
                    curve::apply_ra4_from_density(&leveled, *ra4_params, 4.0, options.curve_white),
                    options,
                )
            } else {
                finish_u16_display(
                    curve::apply_ra4_from_density(image, *ra4_params, 4.0, options.curve_white),
                    options,
                )
            }
        }
        OutputStage::FilmPrint => {
            let fp_params = crate::post_curve::build_film_print_params(options);
            if density_levels_active(options) {
                let mut leveled = image.clone();
                apply_ra4_levels_in_place(&mut leveled, options);
                finish_u16_display(
                    curve::apply_film_print_from_density(&leveled, &fp_params, 4.0),
                    options,
                )
            } else {
                finish_u16_display(
                    curve::apply_film_print_from_density(image, &fp_params, 4.0),
                    options,
                )
            }
        }
        OutputStage::None => finish_none_display(image.clone(), options),
        OutputStage::Lut2383 => finish_lut2383_display(image.clone(), options, output_lut_cube),
    }
}

/// Export-only step 6: takes ownership so levels run in place and the density
/// buffer can be dropped before TIFF/JPEG write.
pub fn step_6_render_owned(
    mut image: Array3<f32>,
    options: &PipelineOptions,
    ra4_params: &curve::PrintCurveParams,
    output_lut_cube: Option<&lut3d::Lut3d>,
) -> Step6Display {
    if options.debug_pipeline_step < 6 {
        return Step6Display::PassthroughDensity(image);
    }

    match options.output_stage {
        OutputStage::Ra4 => {
            if density_levels_active(options) {
                apply_ra4_levels_in_place(&mut image, options);
            }
            let image_u16 =
                curve::apply_ra4_from_density(&image, *ra4_params, 4.0, options.curve_white);
            drop(image);
            finish_u16_display(image_u16, options)
        }
        OutputStage::FilmPrint => {
            let fp_params = crate::post_curve::build_film_print_params(options);
            if density_levels_active(options) {
                apply_ra4_levels_in_place(&mut image, options);
            }
            let image_u16 = curve::apply_film_print_from_density(&image, &fp_params, 4.0);
            drop(image);
            finish_u16_display(image_u16, options)
        }
        OutputStage::None => finish_none_display(image, options),
        OutputStage::Lut2383 => finish_lut2383_display(image, options, output_lut_cube),
    }
}

/// De-Bujack after step 6: display-referred, before grain / sharpen / encode.
///
/// Skipped when disabled, when the pipeline stopped before step 6, or when the
/// buffer is still density (`None` / passthrough). RA-4 / FilmPrint are linear
/// print RGB. Lut2383 Rec.709 is sRGB-encoded and is decoded for the OkLab pass.
pub fn apply_bujack(display: &mut Step6Display, options: &PipelineOptions) {
    if !options.bujack_enabled || options.bujack_strength <= 0.0 {
        return;
    }
    if options.debug_pipeline_step < 6 {
        return;
    }
    match display {
        Step6Display::PassthroughDensity(_) => {}
        Step6Display::U16(img) => {
            crate::bujack::apply_to_u16_linear(img, options);
        }
        Step6Display::F32(img) => {
            if options.output_stage == OutputStage::None {
                return;
            }
            let encoded_srgb = options.output_stage == OutputStage::Lut2383
                && options.output_lut_encoding == OutputLutEncoding::Rec709;
            crate::bujack::apply_to_f32(img, options, encoded_srgb);
        }
    }
}

/// Convert Step6Display to u8 RGB for preview.
///
/// RA-4 / FilmPrint `U16` is linear print RGB — apply the sRGB OETF here.
/// Showing linear as sRGB-encoded (the old `v >> 8` path) is the main purple
/// cast: monitor EOTF exaggerates any green deficit into magenta.
/// Lut2383 / None `F32` is already display-referred code values.
pub fn step6_display_to_u8(display: &Step6Display) -> Vec<u8> {
    match display {
        Step6Display::PassthroughDensity(img) => img
            .iter()
            .map(|v| ((*v / 2.5).clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect(),
        Step6Display::U16(img) => img
            .iter()
            .map(|v| crate::color_space::linear_to_srgb_u8(*v as f32 / 65535.0))
            .collect(),
        Step6Display::F32(img) => img
            .iter()
            .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect(),
    }
}
