//! One-shot Auto grade: search Film γ, density, grade, toe, and hardness
//! from a display histogram, then solve saturation from midtone density chroma.

use ndarray::Array3;

use crate::curve::PrintCurveParams;
use crate::density_ops;
use crate::lut3d;
use crate::pipeline;
use crate::PipelineOptions;

const PROXY_MAX_SIDE: usize = 384;
const TARGET_CENTROID: f32 = 118.0;
const TARGET_P1: f32 = 8.0;
const TARGET_MID_CHROMA: f32 = 0.10;
const GAMMA_LO: f32 = 0.55;
const GAMMA_HI: f32 = 0.75;
const OFFSET_LO: f32 = -0.12;
const OFFSET_HI: f32 = 0.12;
const GRADE_LO: f32 = 0.80;
const GRADE_HI: f32 = 1.25;
const TOE_LO: f32 = -0.50;
const TOE_HI: f32 = 0.50;
const MID_LO: f32 = 1.00;
const MID_HI: f32 = 1.35;
const SAT_LO: f32 = 0.90;
const SAT_HI: f32 = 1.35;
const HOUSE_LAB: f32 = 1.5;
const HOUSE_WARMTH: f32 = 0.6;

/// Settings Auto writes onto [`PipelineOptions`].
#[derive(Debug, Clone, Copy)]
pub struct AutoTuneResult {
    pub film_gamma: f32,
    pub curve_offset: f32,
    pub curve_gamma: f32,
    pub toe_strength: f32,
    pub lut_in_mid: f32,
    pub highlight_rolloff: f32,
    pub saturation: f32,
    pub apply_lab: bool,
    pub lab_separation: f32,
    pub bujack_enabled: bool,
    pub highlight_warmth: f32,
}

impl AutoTuneResult {
    pub fn apply_to(&self, opts: &mut PipelineOptions) {
        opts.film_gamma = self.film_gamma;
        opts.curve_offset = self.curve_offset;
        opts.curve_gamma = self.curve_gamma;
        opts.toe_strength = self.toe_strength;
        opts.lut_in_mid = self.lut_in_mid;
        opts.highlight_rolloff = self.highlight_rolloff;
        opts.saturation = self.saturation;
        opts.apply_lab = self.apply_lab;
        opts.lab_separation = self.lab_separation;
        opts.bujack_enabled = self.bujack_enabled;
        opts.highlight_warmth = self.highlight_warmth;
    }
}

/// Progress callback: current phase title, 0–1 fraction, optional extra log line.
pub type AutoProgressCb<'a> = dyn FnMut(&str, f32, Option<&str>) + 'a;

/// Run Auto on a post-D-min (step 3) transmittance buffer.
///
/// `after_step3` is cropped and downsampled internally. `options` supplies WB,
/// D-min-independent look, crop, LUTs, and print-curve settings. Knobs Auto
/// owns are reset before the search.
pub fn auto_tune(
    after_step3: &Array3<f32>,
    options: &PipelineOptions,
    on_progress: &mut AutoProgressCb<'_>,
) -> anyhow::Result<AutoTuneResult> {
    on_progress("Preparing analysis…", 0.02, Some("Preparing analysis…"));

    let cropped = crop_to_rect(after_step3, options);
    let proxy = downsample_to_max_side(&cropped, PROXY_MAX_SIDE);
    let (ph, pw, pc) = proxy.dim();
    if ph < 8 || pw < 8 || pc != 3 {
        anyhow::bail!("Auto: image is too small to analyse");
    }

    let mut opts = prepare_search_options(options);
    let mut eval = ProxyEval::new(proxy, &opts)?;

    on_progress("Preparing analysis…", 0.08, None);

    on_progress(
        "Finding optimal Film γ…",
        0.08,
        Some("Finding optimal Film γ…"),
    );
    opts.film_gamma = search_1d(GAMMA_LO, GAMMA_HI, 7, 5, |g, i, n| {
        opts.film_gamma = g;
        let frac = 0.08 + 0.14 * (i as f32 / n.max(1) as f32);
        on_progress("Finding optimal Film γ…", frac, None);
        gamma_score(&eval.metrics(&opts))
    });
    opts.film_gamma = opts.film_gamma.clamp(GAMMA_LO, GAMMA_HI);

    on_progress("Adjusting density…", 0.22, Some("Adjusting density…"));
    opts.curve_offset = search_1d(OFFSET_LO, OFFSET_HI, 7, 5, |o, i, n| {
        opts.curve_offset = o;
        let frac = 0.22 + 0.12 * (i as f32 / n.max(1) as f32);
        on_progress("Adjusting density…", frac, None);
        gamma_score(&eval.metrics(&opts))
    });
    opts.curve_offset = opts.curve_offset.clamp(OFFSET_LO, OFFSET_HI);

    on_progress("Setting paper grade…", 0.34, Some("Setting paper grade…"));
    let grade = search_1d(GRADE_LO, GRADE_HI, 7, 5, |g, i, n| {
        opts.curve_gamma = 2.5 * g;
        let frac = 0.34 + 0.12 * (i as f32 / n.max(1) as f32);
        on_progress("Setting paper grade…", frac, None);
        grade_score(&eval.metrics(&opts))
    });
    opts.curve_gamma = (2.5 * grade.clamp(GRADE_LO, GRADE_HI)).clamp(2.0, 3.125);

    on_progress("Adjusting exposure…", 0.46, Some("Adjusting exposure…"));
    opts.toe_strength = search_1d(TOE_LO, TOE_HI, 7, 5, |t, i, n| {
        opts.toe_strength = t;
        let frac = 0.46 + 0.10 * (i as f32 / n.max(1) as f32);
        on_progress("Adjusting exposure…", frac, None);
        toe_score(&eval.metrics(&opts))
    });
    opts.toe_strength = opts.toe_strength.clamp(TOE_LO, TOE_HI);

    on_progress("Stretching midtones…", 0.56, Some("Stretching midtones…"));
    opts.lut_in_mid = search_1d(MID_LO, MID_HI, 7, 5, |m, i, n| {
        opts.lut_in_mid = m;
        let frac = 0.56 + 0.10 * (i as f32 / n.max(1) as f32);
        on_progress("Stretching midtones…", frac, None);
        hardness_score(&eval.metrics(&opts))
    });
    opts.lut_in_mid = opts.lut_in_mid.clamp(MID_LO, MID_HI);

    on_progress("Refining shadows…", 0.66, Some("Refining shadows…"));
    opts.toe_strength = search_1d(TOE_LO, TOE_HI, 7, 5, |t, i, n| {
        opts.toe_strength = t;
        let frac = 0.66 + 0.10 * (i as f32 / n.max(1) as f32);
        on_progress("Refining shadows…", frac, None);
        toe_score(&eval.metrics(&opts))
    });
    opts.toe_strength = opts.toe_strength.clamp(TOE_LO, TOE_HI);

    on_progress("Refining density…", 0.76, Some("Refining density…"));
    opts.curve_offset = search_1d(OFFSET_LO, OFFSET_HI, 7, 5, |o, i, n| {
        opts.curve_offset = o;
        let frac = 0.76 + 0.10 * (i as f32 / n.max(1) as f32);
        on_progress("Refining density…", frac, None);
        gamma_score(&eval.metrics(&opts))
    });
    opts.curve_offset = opts.curve_offset.clamp(OFFSET_LO, OFFSET_HI);

    on_progress("Adjusting saturation…", 0.88, Some("Adjusting saturation…"));
    opts.saturation = 1.0;
    let measured = eval.midtone_chroma(&opts);
    let mut sat = solve_saturation(measured, TARGET_MID_CHROMA);
    let clip_base = eval.metrics(&opts).clip_hi;
    opts.saturation = sat;
    let mut clip_sat = eval.metrics(&opts).clip_hi;
    while sat > SAT_LO + 0.02 && clip_sat > clip_base + 0.0005 {
        sat = (sat - 0.05).max(SAT_LO);
        opts.saturation = sat;
        clip_sat = eval.metrics(&opts).clip_hi;
    }
    opts.saturation = sat;

    on_progress("Applying settings…", 0.97, Some("Applying settings…"));

    Ok(AutoTuneResult {
        film_gamma: opts.film_gamma,
        curve_offset: opts.curve_offset,
        curve_gamma: opts.curve_gamma,
        toe_strength: opts.toe_strength,
        lut_in_mid: opts.lut_in_mid,
        highlight_rolloff: 0.0,
        saturation: opts.saturation,
        apply_lab: true,
        lab_separation: HOUSE_LAB,
        bujack_enabled: false,
        highlight_warmth: HOUSE_WARMTH,
    })
}

fn prepare_search_options(src: &PipelineOptions) -> PipelineOptions {
    let mut opts = src.clone();
    opts.film_gamma = 0.65;
    opts.curve_offset = 0.0;
    opts.curve_gamma = 2.5;
    opts.toe_strength = 0.0;
    opts.lut_in_mid = 1.0;
    opts.highlight_rolloff = 0.0;
    opts.saturation = 1.0;
    opts.apply_lab = true;
    opts.lab_separation = HOUSE_LAB;
    opts.highlight_warmth = HOUSE_WARMTH;
    opts.bujack_enabled = false;
    opts
}

struct ProxyEval {
    t: Array3<f32>,
    lut3d: Option<lut3d::Lut3d>,
    output_lut: Option<lut3d::Lut3d>,
    after4_gamma: Option<f32>,
    after4: Option<Array3<f32>>,
    after5_key: Option<(u32, u32, u32)>,
    after5: Option<Array3<f32>>,
}

impl ProxyEval {
    fn new(t: Array3<f32>, opts: &PipelineOptions) -> anyhow::Result<Self> {
        let lut3d = opts
            .lut3d_path
            .as_ref()
            .and_then(|p| lut3d::read_cube(p).ok());
        let output_lut = opts
            .output_lut_cube
            .as_ref()
            .and_then(|p| lut3d::read_cube(p).ok());
        Ok(Self {
            t,
            lut3d,
            output_lut,
            after4_gamma: None,
            after4: None,
            after5_key: None,
            after5: None,
        })
    }

    fn density_after4(&mut self, opts: &PipelineOptions) -> &Array3<f32> {
        if self.after4_gamma != Some(opts.film_gamma) {
            let mut img = self.t.clone();
            pipeline::step_4_t_to_d_wb(&mut img, opts);
            self.after4 = Some(img);
            self.after4_gamma = Some(opts.film_gamma);
            self.after5_key = None;
        }
        self.after4.as_ref().expect("after4")
    }

    fn density_after5(&mut self, opts: &PipelineOptions) -> &Array3<f32> {
        let key = (
            opts.film_gamma.to_bits(),
            opts.saturation.to_bits(),
            opts.highlight_rolloff.to_bits(),
        );
        if self.after5_key != Some(key) {
            let mut img = self.density_after4(opts).clone();
            pipeline::step_5_calibration(&mut img, opts, self.lut3d.as_ref());
            self.after5 = Some(img);
            self.after5_key = Some(key);
        }
        self.after5.as_ref().expect("after5")
    }

    fn metrics(&mut self, opts: &PipelineOptions) -> HistMetrics {
        let density = self.density_after5(opts).clone();
        let ra4 = PrintCurveParams {
            offset: opts.curve_offset,
            gamma: opts.curve_gamma,
            pivot: opts.curve_pivot,
        };
        let display = pipeline::step_6_render(&density, opts, &ra4, self.output_lut.as_ref());
        let rgb = pipeline::step6_display_to_u8(&display);
        let (h, w, _) = density.dim();
        metrics_from_rgb(&rgb, w as u32, h as u32)
    }

    fn midtone_chroma(&mut self, opts: &PipelineOptions) -> f32 {
        let density = self.density_after5(opts);
        midtone_chroma(density, opts.curve_offset)
    }
}

#[derive(Debug, Clone, Copy)]
struct HistMetrics {
    centroid: f32,
    clip_hi: f32,
    clip_lo: f32,
    p1: f32,
    p50: f32,
    p99: f32,
}

fn metrics_from_rgb(rgb: &[u8], _w: u32, _h: u32) -> HistMetrics {
    let mut luma = [0u32; 256];
    let mut r_h = [0u32; 256];
    let mut g_h = [0u32; 256];
    let mut b_h = [0u32; 256];
    let mut n = 0u32;
    let mut sum = 0.0f64;
    for c in rgb.chunks_exact(3) {
        r_h[c[0] as usize] += 1;
        g_h[c[1] as usize] += 1;
        b_h[c[2] as usize] += 1;
        let y = (c[0] as u32 + c[1] as u32 + c[2] as u32) / 3;
        luma[y as usize] += 1;
        sum += y as f64;
        n += 1;
    }
    let n_f = n.max(1) as f32;
    let clip_hi = {
        let cr: u32 = r_h[253..=255].iter().sum();
        let cg: u32 = g_h[253..=255].iter().sum();
        let cb: u32 = b_h[253..=255].iter().sum();
        cr.max(cg).max(cb) as f32 / n_f
    };
    let clip_lo = {
        let cr: u32 = r_h[0..=2].iter().sum();
        let cg: u32 = g_h[0..=2].iter().sum();
        let cb: u32 = b_h[0..=2].iter().sum();
        cr.max(cg).max(cb) as f32 / n_f
    };
    HistMetrics {
        centroid: (sum / n.max(1) as f64) as f32,
        clip_hi,
        clip_lo,
        p1: percentile(&luma, n, 0.01),
        p50: percentile(&luma, n, 0.50),
        p99: percentile(&luma, n, 0.99),
    }
}

fn percentile(hist: &[u32; 256], n: u32, p: f32) -> f32 {
    if n == 0 {
        return 0.0;
    }
    let target = (n as f32 * p).ceil().max(1.0) as u32;
    let mut acc = 0u32;
    for (i, &c) in hist.iter().enumerate() {
        acc += c;
        if acc >= target {
            return i as f32;
        }
    }
    255.0
}

fn gamma_score(m: &HistMetrics) -> f32 {
    (m.centroid - TARGET_CENTROID).abs() / 128.0 + 8.0 * m.clip_hi + 2.0 * m.clip_lo
}

fn toe_score(m: &HistMetrics) -> f32 {
    let mut s = (m.p1 - TARGET_P1).abs() / 255.0;
    if m.clip_lo > 0.0015 {
        s += 4.0 * (m.clip_lo - 0.0015);
    }
    if m.p1 < 2.0 {
        s += 0.3;
    }
    s
}

fn hardness_score(m: &HistMetrics) -> f32 {
    (m.p50 - TARGET_CENTROID).abs() / 128.0 + 4.0 * m.clip_hi
}

fn grade_score(m: &HistMetrics) -> f32 {
    let spread = ((m.p99 - m.p1) / 255.0).clamp(0.0, 1.0);
    (m.centroid - TARGET_CENTROID).abs() / 128.0 + 6.0 * m.clip_hi + 0.35 * (1.0 - spread)
}

fn solve_saturation(measured: f32, target: f32) -> f32 {
    if measured < 1e-5 {
        return 1.0;
    }
    (target / measured).clamp(SAT_LO, SAT_HI)
}

fn midtone_chroma(image: &Array3<f32>, curve_offset: f32) -> f32 {
    let z = density_ops::zone_density_range(image, curve_offset);
    let (h, w, _) = image.dim();
    let n = h * w;
    if n == 0 {
        return 0.0;
    }
    let step = (n / 4096).max(1);
    let mut sum = 0.0f32;
    let mut count = 0.0f32;
    for i in (0..n).step_by(step) {
        let y = i / w;
        let x = i % w;
        let dr = image[[y, x, 0]];
        let dg = image[[y, x, 1]];
        let db = image[[y, x, 2]];
        let mean = (dr + dg + db) / 3.0;
        let d_eff = mean + curve_offset;
        if d_eff >= z.d_p33 && d_eff <= z.d_p66 {
            sum += ((dr - mean).abs() + (dg - mean).abs() + (db - mean).abs()) / 3.0;
            count += 1.0;
        }
    }
    if count > 0.0 {
        sum / count
    } else {
        0.0
    }
}

fn search_1d(
    lo: f32,
    hi: f32,
    coarse: usize,
    refine: usize,
    mut score: impl FnMut(f32, usize, usize) -> f32,
) -> f32 {
    let mut best_x = lo;
    let mut best_s = f32::INFINITY;
    let total = coarse + refine;
    for i in 0..coarse {
        let t = if coarse <= 1 {
            0.0
        } else {
            i as f32 / (coarse - 1) as f32
        };
        let x = lo + (hi - lo) * t;
        let s = score(x, i, total);
        if s < best_s {
            best_s = s;
            best_x = x;
        }
    }
    let span = (hi - lo) / (coarse as f32).max(1.0);
    let rlo = (best_x - span).max(lo);
    let rhi = (best_x + span).min(hi);
    for i in 0..refine {
        let t = if refine <= 1 {
            0.5
        } else {
            i as f32 / (refine - 1) as f32
        };
        let x = rlo + (rhi - rlo) * t;
        let s = score(x, coarse + i, total);
        if s < best_s {
            best_s = s;
            best_x = x;
        }
    }
    best_x
}

fn downsample_to_max_side(image: &Array3<f32>, max_side: usize) -> Array3<f32> {
    let (h, w, _) = image.dim();
    if h == 0 || w == 0 {
        return image.clone();
    }
    let long = h.max(w);
    if long <= max_side {
        return image.clone();
    }
    let new_w = ((w * max_side) / long).max(1);
    let new_h = ((h * max_side) / long).max(1);
    let mut out = Array3::<f32>::zeros((new_h, new_w, 3));
    for y in 0..new_h {
        let y0 = y * h / new_h;
        let y1 = ((y + 1) * h / new_h).max(y0 + 1);
        for x in 0..new_w {
            let x0 = x * w / new_w;
            let x1 = ((x + 1) * w / new_w).max(x0 + 1);
            let mut r = 0.0;
            let mut g = 0.0;
            let mut b = 0.0;
            let mut n = 0.0;
            for yy in y0..y1 {
                for xx in x0..x1 {
                    r += image[[yy, xx, 0]];
                    g += image[[yy, xx, 1]];
                    b += image[[yy, xx, 2]];
                    n += 1.0;
                }
            }
            out[[y, x, 0]] = r / n;
            out[[y, x, 1]] = g / n;
            out[[y, x, 2]] = b / n;
        }
    }
    out
}

fn crop_to_rect(image: &Array3<f32>, options: &PipelineOptions) -> Array3<f32> {
    if !options.apply_crop {
        return image.clone();
    }
    let Some(rect) = options.crop_rect else {
        return image.clone();
    };
    let (h, w, _) = image.dim();
    let (x, y, rw, rh) =
        crate::scale_dmin_rect(rect, options.crop_rect_reference_size, w as u32, h as u32);
    let x0 = (x as usize).min(w.saturating_sub(1));
    let y0 = (y as usize).min(h.saturating_sub(1));
    let x1 = ((x + rw) as usize).min(w).max(x0 + 1);
    let y1 = ((y + rh) as usize).min(h).max(y0 + 1);
    if x1 <= x0 + 4 || y1 <= y0 + 4 {
        return image.clone();
    }
    image.slice(ndarray::s![y0..y1, x0..x1, ..]).to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray_rgb(value: u8, n: usize) -> Vec<u8> {
        vec![value, value, value].repeat(n)
    }

    #[test]
    fn centroid_of_flat_gray() {
        let rgb = gray_rgb(118, 100);
        let m = metrics_from_rgb(&rgb, 10, 10);
        assert!((m.centroid - 118.0).abs() < 0.5);
        assert_eq!(m.clip_hi, 0.0);
        assert_eq!(m.clip_lo, 0.0);
        assert!((m.p50 - 118.0).abs() < 0.5);
    }

    #[test]
    fn clip_hi_detects_channel_rail() {
        let mut rgb = gray_rgb(128, 100);
        for i in 0..5 {
            rgb[i * 3] = 255;
        }
        let m = metrics_from_rgb(&rgb, 10, 10);
        assert!((m.clip_hi - 0.05).abs() < 1e-4);
    }

    #[test]
    fn gamma_score_prefers_no_clip_over_perfect_center() {
        let centered_clip = HistMetrics {
            centroid: 118.0,
            clip_hi: 0.05,
            clip_lo: 0.0,
            p1: 10.0,
            p50: 118.0,
            p99: 250.0,
        };
        let left_safe = HistMetrics {
            centroid: 100.0,
            clip_hi: 0.0,
            clip_lo: 0.0,
            p1: 10.0,
            p50: 100.0,
            p99: 200.0,
        };
        assert!(gamma_score(&left_safe) < gamma_score(&centered_clip));
    }

    #[test]
    fn toe_score_prefers_black_touch() {
        let gap = HistMetrics {
            centroid: 128.0,
            clip_hi: 0.0,
            clip_lo: 0.0,
            p1: 40.0,
            p50: 128.0,
            p99: 200.0,
        };
        let touch = HistMetrics {
            centroid: 128.0,
            clip_hi: 0.0,
            clip_lo: 0.0005,
            p1: 8.0,
            p50: 128.0,
            p99: 200.0,
        };
        let crush = HistMetrics {
            centroid: 128.0,
            clip_hi: 0.0,
            clip_lo: 0.05,
            p1: 0.0,
            p50: 128.0,
            p99: 200.0,
        };
        assert!(toe_score(&touch) < toe_score(&gap));
        assert!(toe_score(&touch) < toe_score(&crush));
    }

    #[test]
    fn saturation_solve_is_linear() {
        assert!((solve_saturation(0.10, 0.11) - 1.10).abs() < 1e-5);
        assert!((solve_saturation(0.01, 0.11) - SAT_HI).abs() < 1e-5);
        assert!((solve_saturation(0.20, 0.11) - SAT_LO).abs() < 1e-5);
        assert!((solve_saturation(0.0, 0.11) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn search_1d_finds_parabola_minimum() {
        let x = search_1d(0.0, 1.0, 9, 7, |v, _, _| (v - 0.37) * (v - 0.37));
        assert!((x - 0.37).abs() < 0.03);
    }

    #[test]
    fn grade_score_prefers_wider_unclipped_hist() {
        let narrow = HistMetrics {
            centroid: 118.0,
            clip_hi: 0.0,
            clip_lo: 0.0,
            p1: 80.0,
            p50: 118.0,
            p99: 160.0,
        };
        let wide = HistMetrics {
            centroid: 118.0,
            clip_hi: 0.0,
            clip_lo: 0.0,
            p1: 12.0,
            p50: 118.0,
            p99: 240.0,
        };
        let clipped_wide = HistMetrics {
            centroid: 118.0,
            clip_hi: 0.04,
            clip_lo: 0.0,
            p1: 8.0,
            p50: 118.0,
            p99: 254.0,
        };
        assert!(grade_score(&wide) < grade_score(&narrow));
        assert!(grade_score(&narrow) < grade_score(&clipped_wide));
    }

    #[test]
    fn prepare_search_resets_owned_knobs() {
        let mut src = PipelineOptions::default();
        src.highlight_rolloff = 1.2;
        src.curve_offset = 0.3;
        src.curve_gamma = 4.0;
        src.bujack_enabled = true;
        let opts = prepare_search_options(&src);
        assert_eq!(opts.highlight_rolloff, 0.0);
        assert_eq!(opts.curve_offset, 0.0);
        assert_eq!(opts.curve_gamma, 2.5);
        assert!(!opts.bujack_enabled);
    }
}
