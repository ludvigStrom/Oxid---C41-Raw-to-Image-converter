//! Film-frame auto crop on post–D-min linear transmittance.
//!
//! Detects the image area (excluding holder, lightbox, rebate, and sprockets)
//! from an `after_step3` RGB buffer. Independent of RA-4 / display look.

use ndarray::Array3;

use crate::options::Rect;

/// Max side length of the analysis proxy. Matches Auto's working scale.
pub const CROP_PROXY_MAX_SIDE: usize = 384;

/// How sure the detector is that the rect is the real frame edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CropConfidence {
    High,
    Medium,
    Low,
}

/// What the outer perimeter looks like (majority vote of the four sides).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurroundClass {
    DarkHolder,
    BrightLight,
    FilmRebate,
    Mixed,
}

/// Still-format guess from the cropped aspect ratio, if it snapped or matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilmFormat {
    Still35,
    Square,
    SixBySeven,
    FourByThree,
    FourByFive,
}

/// Successful auto-crop. `rect` is in the input buffer's pixel space.
#[derive(Debug, Clone, Copy)]
pub struct AutoCropResult {
    pub rect: Rect,
    pub reference_size: (u32, u32),
    pub confidence: CropConfidence,
    pub surround: SurroundClass,
    pub format_hint: Option<FilmFormat>,
}

/// Probe stats for crop-benchmark debugging.
pub(crate) fn crop_probe(image: &Array3<f32>) -> String {
    let proxy = downsample_to_max_side(image, CROP_PROXY_MAX_SIDE);
    let (h, w, _) = proxy.dim();
    let t = pixel_transmittance(&proxy);
    let t_max = pixel_max(&proxy);
    let tex = pixel_texture(&t, w, h);
    let light_level = find_bright_mode(&t_max, w, h);
    let base = estimate_base(&t, &t_max, w, h, None, light_level);
    let tex_thresh = adaptive_tex_thresh(&tex, w, h);
    let (outer, inner) = outer_inner_means(&t, w, h);
    let polarity = detect_polarity(&t, w, h);
    let seed = bounds_from_bright_seed(&t, &tex, w, h);
    let image_px = {
        let holder_cap = (0.07 * base).clamp(0.05, 0.10);
        let light_floor = (1.18 * base).max(1.12);
        let rebate_floor = 0.86 * base;
        let dye_cap = 0.62 * base;
        let mut image_px = vec![false; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                image_px[i] = classify_image_pixel(
                    t[i],
                    t_max[i],
                    tex[i],
                    holder_cap,
                    light_floor,
                    rebate_floor,
                    dye_cap,
                    tex_thresh,
                    light_level,
                    outer,
                );
            }
        }
        mark_sprocket_rows_as_border(&t, &mut image_px, w, h, base);
        bounds_from_image_frac(&image_px, w, h, 0.18)
    };
    let fmt = |b: Option<(usize, usize, usize, usize)>| match b {
        Some((l, r, t0, b0)) => format!(
            "[{:.3},{:.3},{:.3},{:.3}] a={:.2}",
            l as f32 / w as f32,
            t0 as f32 / h as f32,
            r as f32 / w as f32,
            b0 as f32 / h as f32,
            (r.saturating_sub(l) * b0.saturating_sub(t0)) as f32 / (w * h) as f32
        ),
        None => "none".to_string(),
    };
    format!(
        "{w}x{h} pol={polarity:?} outer={outer:.3} inner={inner:.3} base={base:.3} texT={tex_thresh:.4} light={light_level:?} seed={} px={}",
        fmt(seed),
        fmt(image_px)
    )
}

/// Detect a tight image-area crop on a linear-transmittance RGB buffer.
///
/// `dmin_rect` is an optional film-base sample (any reference size). It is a
/// hint for base transmittance, not a requirement.
pub fn detect_crop(
    image: &Array3<f32>,
    dmin_rect: Option<Rect>,
    dmin_rect_reference_size: Option<(u32, u32)>,
) -> Option<AutoCropResult> {
    let (h, w, c) = image.dim();
    if c != 3 || w < 16 || h < 16 {
        return None;
    }
    let src_w = w as u32;
    let src_h = h as u32;

    let proxy = downsample_to_max_side(image, CROP_PROXY_MAX_SIDE);
    let (ph, pw, _) = proxy.dim();
    if pw < 16 || ph < 16 {
        return None;
    }

    let dmin_proxy = dmin_rect.map(|r| {
        let (x, y, rw, rh) = crate::scale_dmin_rect(r, dmin_rect_reference_size, src_w, src_h);
        let sx = pw as f32 / src_w as f32;
        let sy = ph as f32 / src_h as f32;
        Rect {
            x: (x as f32 * sx).round() as u32,
            y: (y as f32 * sy).round() as u32,
            width: (rw as f32 * sx).round().max(1.0) as u32,
            height: (rh as f32 * sy).round().max(1.0) as u32,
        }
    });

    let detected = detect_on_proxy(&proxy, dmin_proxy)?;
    let scale_x = src_w as f32 / pw as f32;
    let scale_y = src_h as f32 / ph as f32;
    let rect = Rect {
        x: (detected.rect.x as f32 * scale_x).round() as u32,
        y: (detected.rect.y as f32 * scale_y).round() as u32,
        width: (detected.rect.width as f32 * scale_x).round().max(1.0) as u32,
        height: (detected.rect.height as f32 * scale_y).round().max(1.0) as u32,
    };
    let rect = clamp_rect(rect, src_w, src_h)?;
    Some(AutoCropResult {
        rect,
        reference_size: (src_w, src_h),
        confidence: detected.confidence,
        surround: detected.surround,
        format_hint: detected.format_hint,
    })
}

fn detect_on_proxy(image: &Array3<f32>, dmin_rect: Option<Rect>) -> Option<AutoCropResult> {
    let (h, w, _) = image.dim();
    let t = pixel_transmittance(image);
    let t_max = pixel_max(image);
    let tex = pixel_texture(&t, w, h);
    let light_level = find_bright_mode(&t_max, w, h);
    let base = estimate_base(&t, &t_max, w, h, dmin_rect, light_level);
    let surround = classify_surround(&t_max, w, h, base, light_level);
    let tex_thresh = adaptive_tex_thresh(&tex, w, h);

    let holder_cap = (0.07 * base).clamp(0.05, 0.10);
    let light_floor = (1.18 * base).max(1.12);
    let rebate_floor = 0.86 * base;
    let dye_cap = 0.62 * base;
    let (outer_t, _) = outer_inner_means(&t, w, h);

    let mut image_px = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            image_px[i] = classify_image_pixel(
                t[i],
                t_max[i],
                tex[i],
                holder_cap,
                light_floor,
                rebate_floor,
                dye_cap,
                tex_thresh,
                light_level,
                outer_t,
            );
        }
    }

    mark_sprocket_rows_as_border(&t, &mut image_px, w, h, base);

    let polarity = detect_polarity(&t, w, h);
    let surround = match polarity {
        Polarity::FilmBrighter => SurroundClass::DarkHolder,
        Polarity::SurroundBrighter => SurroundClass::BrightLight,
        Polarity::Unclear => surround,
    };

    const FRAC_MIN: f32 = 0.18;
    let image_px_bounds = bounds_from_image_frac(&image_px, w, h, FRAC_MIN);
    let gate_bounds = bounds_from_gate_edges(&t, w, h, polarity);
    let (left, right, top, bottom) = if polarity == Polarity::FilmBrighter {
        let seed = bounds_from_bright_seed(&t, &tex, w, h)?;
        let grown = grow_bright_seed(&t, w, h, seed, tex_thresh, light_floor);
        pick_bounds(&t, w, h, grown, image_px_bounds, gate_bounds)
    } else {
        pick_bounds(&t, w, h, image_px_bounds?, None, gate_bounds)
    };

    if right <= left || bottom <= top {
        return None;
    }

    let (mut left, mut right, mut top, mut bottom, format_hint) =
        snap_aspect(left, right, top, bottom);

    inset_bounds(&mut left, &mut right, &mut top, &mut bottom, w, h);

    if right <= left || bottom <= top {
        return None;
    }

    let rw = right - left;
    let rh = bottom - top;
    let min_side = (w.min(h) / 20).max(16);
    if rw < min_side || rh < min_side {
        return None;
    }

    let margin_l = left as f32 / w as f32;
    let margin_r = (w - right) as f32 / w as f32;
    let margin_t = top as f32 / h as f32;
    let margin_b = (h - bottom) as f32 / h as f32;
    let min_margin = margin_l.min(margin_r).min(margin_t).min(margin_b);
    let max_margin = margin_l.max(margin_r).max(margin_t).max(margin_b);
    let strong_edge = min_margin > 0.025 && max_margin < 0.45;
    let surround_ok = !matches!(surround, SurroundClass::Mixed);

    let confidence = if strong_edge && surround_ok {
        CropConfidence::High
    } else if strong_edge || surround_ok || min_margin > 0.015 {
        CropConfidence::Medium
    } else {
        CropConfidence::Low
    };

    Some(AutoCropResult {
        rect: Rect {
            x: left as u32,
            y: top as u32,
            width: rw as u32,
            height: rh as u32,
        },
        reference_size: (w as u32, h as u32),
        confidence,
        surround,
        format_hint,
    })
}

fn classify_image_pixel(
    t: f32,
    t_max: f32,
    tex: f32,
    holder_cap: f32,
    light_floor: f32,
    rebate_floor: f32,
    dye_cap: f32,
    tex_thresh: f32,
    light_level: Option<f32>,
    outer_t: f32,
) -> bool {
    if t < holder_cap {
        return false;
    }
    // Lightbox (including chromatic narrowband) is never image, even if textured.
    let in_bright = light_level.map(|l| t_max >= l * 0.88).unwrap_or(false);
    if t_max > light_floor || in_bright {
        return false;
    }
    // Mid-T surround matches the outer band; do not promote it via dye_cap.
    if (t - outer_t).abs() < 0.08 && tex < tex_thresh {
        return false;
    }
    if tex >= tex_thresh {
        return true;
    }
    // Flat dye (clear sky on the neg) is still image; flat near-base is rebate.
    t < dye_cap && t >= holder_cap || (t < rebate_floor && tex >= tex_thresh * 0.45)
}

fn pixel_transmittance(image: &Array3<f32>) -> Vec<f32> {
    let (h, w, _) = image.dim();
    let mut t = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            t[y * w + x] = (image[[y, x, 0]] + image[[y, x, 1]] + image[[y, x, 2]]) * (1.0 / 3.0);
        }
    }
    t
}

fn pixel_max(image: &Array3<f32>) -> Vec<f32> {
    let (h, w, _) = image.dim();
    let mut t = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            t[y * w + x] = image[[y, x, 0]].max(image[[y, x, 1]]).max(image[[y, x, 2]]);
        }
    }
    t
}

/// High-end clump on the outer frame = open lightbox, not film base.
fn find_bright_mode(t_max: &[f32], w: usize, h: usize) -> Option<f32> {
    if t_max.len() != w * h || w < 16 || h < 16 {
        return None;
    }
    let p98 = percentile(t_max, 0.98);
    if p98 < 0.40 {
        return None;
    }
    let cut = p98 * 0.88;
    let bw = ((w as f32 * 0.10) as usize).clamp(2, w / 5);
    let bh = ((h as f32 * 0.10) as usize).clamp(2, h / 5);

    let mut bright_n = 0usize;
    let mut outer_n = 0usize;
    let mut outer_bright = 0usize;
    let mut inner_n = 0usize;
    let mut inner_bright = 0usize;
    let mut outer_bright_vals: Vec<f32> = Vec::new();
    let mut inner_film: Vec<f32> = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let v = t_max[y * w + x];
            let on_edge = x < bw || x >= w - bw || y < bh || y >= h - bh;
            let in_inner = x >= w / 4 && x < 3 * w / 4 && y >= h / 4 && y < 3 * h / 4;
            let bright = v >= cut;
            if bright {
                bright_n += 1;
            }
            if on_edge {
                outer_n += 1;
                if bright {
                    outer_bright += 1;
                    outer_bright_vals.push(v);
                }
            }
            if in_inner {
                inner_n += 1;
                if bright {
                    inner_bright += 1;
                }
                if v > 0.12 && v < cut {
                    inner_film.push(v);
                }
            }
        }
    }

    if bright_n < (t_max.len() as f32 * 0.08) as usize {
        return None;
    }
    let frac_outer = if outer_n == 0 {
        0.0
    } else {
        outer_bright as f32 / outer_n as f32
    };
    let frac_inner = if inner_n == 0 {
        0.0
    } else {
        inner_bright as f32 / inner_n as f32
    };
    // Lightbox lives on the border; fogged / full-frame image is bright in the middle.
    if frac_outer < 0.40 || frac_inner > 0.35 {
        return None;
    }
    if inner_film.len() < (inner_n as f32 * 0.15) as usize {
        return None;
    }
    let film_body = percentile(&inner_film, 0.72);
    if p98 < film_body * 1.15 && (p98 - film_body) < 0.12 {
        return None;
    }
    if outer_bright_vals.is_empty() {
        return None;
    }
    Some(percentile(&outer_bright_vals, 0.50).max(p98 * 0.90))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Polarity {
    /// Dark holder / mask / vignette; the film rectangle is brighter in T.
    FilmBrighter,
    /// Open lightbox; the surround is brighter than the film.
    SurroundBrighter,
    Unclear,
}

fn outer_inner_means(t: &[f32], w: usize, h: usize) -> (f32, f32) {
    let bw = ((w as f32 * 0.06) as usize).clamp(2, w / 8);
    let bh = ((h as f32 * 0.06) as usize).clamp(2, h / 8);
    let mut outer = 0.0;
    let mut on = 0.0;
    let mut inner = 0.0;
    let mut inn = 0.0;
    for y in 0..h {
        for x in 0..w {
            let v = t[y * w + x];
            let on_edge = x < bw || x >= w - bw || y < bh || y >= h - bh;
            let in_inner = x >= w / 4 && x < 3 * w / 4 && y >= h / 4 && y < 3 * h / 4;
            if on_edge {
                outer += v;
                on += 1.0;
            } else if in_inner {
                inner += v;
                inn += 1.0;
            }
        }
    }
    (
        if on > 0.0 { outer / on } else { 0.0 },
        if inn > 0.0 { inner / inn } else { 0.0 },
    )
}

fn detect_polarity(t: &[f32], w: usize, h: usize) -> Polarity {
    let (outer, inner) = outer_inner_means(t, w, h);
    if inner > outer * 1.18 && inner - outer > 0.08 {
        Polarity::FilmBrighter
    } else if outer > inner * 1.12 && outer - inner > 0.08 {
        Polarity::SurroundBrighter
    } else {
        Polarity::Unclear
    }
}

fn axis_percentile(t: &[f32], w: usize, h: usize, rows: bool, p: f32) -> Vec<f32> {
    if rows {
        (0..h)
            .map(|y| percentile(&t[y * w..(y + 1) * w], p))
            .collect()
    } else {
        (0..w)
            .map(|x| {
                let mut col = Vec::with_capacity(h);
                for y in 0..h {
                    col.push(t[y * w + x]);
                }
                percentile(&col, p)
            })
            .collect()
    }
}

fn tex_row_mean(tex: &[f32], w: usize, h: usize) -> Vec<f32> {
    (0..h)
        .map(|y| tex[y * w..(y + 1) * w].iter().sum::<f32>() / w as f32)
        .collect()
}

fn tex_col_mean(tex: &[f32], w: usize, h: usize) -> Vec<f32> {
    (0..w)
        .map(|x| {
            let mut s = 0.0;
            for y in 0..h {
                s += tex[y * w + x];
            }
            s / h as f32
        })
        .collect()
}

/// Peel a thin brighter rebate off the ends of a bright run.
/// Rebate sits above the film-body T; a dark scene does not.
fn trim_edge_rebate(
    profile_t: &[f32],
    _tex_mean: &[f32],
    start: usize,
    end: usize,
) -> (usize, usize) {
    if end <= start + 4 || profile_t.is_empty() {
        return (start, end);
    }
    let body = percentile(&profile_t[start..end], 0.50);
    let rebate_t = body * 1.06;
    let max_walk = ((end - start) / 6).max(3).min(profile_t.len() / 8);
    let mut new_start = start;
    let mut walked = 0usize;
    while new_start + 2 < end && walked < max_walk && profile_t[new_start] > rebate_t {
        new_start += 1;
        walked += 1;
    }
    let mut new_end = end;
    walked = 0;
    while new_end > new_start + 2 && walked < max_walk && profile_t[new_end - 1] > rebate_t {
        new_end -= 1;
        walked += 1;
    }
    (new_start, new_end)
}

fn strip_mean(t: &[f32], w: usize, h: usize, x0: usize, x1: usize, y0: usize, y1: usize) -> f32 {
    let x1 = x1.min(w);
    let y1 = y1.min(h);
    if x0 >= x1 || y0 >= y1 {
        return 0.0;
    }
    let mut s = 0.0;
    let mut n = 0.0;
    for y in y0..y1 {
        for x in x0..x1 {
            s += t[y * w + x];
            n += 1.0;
        }
    }
    if n > 0.0 {
        s / n
    } else {
        0.0
    }
}

fn has_gate(t: &[f32], w: usize, h: usize, rows: bool, index: usize, a: usize, b: usize) -> bool {
    let band = 6usize;
    if rows {
        if index >= h || b <= a + 4 {
            return false;
        }
        let inside = strip_mean(t, w, h, a, (a + band).min(b), index, index + 1);
        let outside = if a >= band {
            strip_mean(t, w, h, a - band, a, index, index + 1)
        } else if b + band <= w {
            strip_mean(t, w, h, b, b + band, index, index + 1)
        } else {
            return false;
        };
        (inside - outside).abs() > 0.06
    } else {
        if index >= w || b <= a + 4 {
            return false;
        }
        let inside = strip_mean(t, w, h, index, index + 1, a, (a + band).min(b));
        let outside = if a >= band {
            strip_mean(t, w, h, index, index + 1, a - band, a)
        } else if b + band <= h {
            strip_mean(t, w, h, index, index + 1, b, b + band)
        } else {
            return false;
        };
        (inside - outside).abs() > 0.06
    }
}

fn bounds_from_image_frac(
    image_px: &[bool],
    w: usize,
    h: usize,
    frac_min: f32,
) -> Option<(usize, usize, usize, usize)> {
    let row_frac: Vec<f32> = (0..h)
        .map(|y| {
            let count = (0..w).filter(|&x| image_px[y * w + x]).count();
            count as f32 / w as f32
        })
        .collect();
    let col_frac: Vec<f32> = (0..w)
        .map(|x| {
            let count = (0..h).filter(|&y| image_px[y * w + x]).count();
            count as f32 / h as f32
        })
        .collect();
    let (top, bottom) = longest_run(&row_frac, frac_min, h / 2)?;
    let (left, right) = longest_run(&col_frac, frac_min, w / 2)?;
    if right <= left || bottom <= top {
        None
    } else {
        Some((left, right, top, bottom))
    }
}

/// High-T seed inside the film. Dark scene content is recovered by
/// [`grow_bright_seed`], not by this threshold.
fn bounds_from_bright_seed(
    t: &[f32],
    tex: &[f32],
    w: usize,
    h: usize,
) -> Option<(usize, usize, usize, usize)> {
    let row_p = axis_percentile(t, w, h, true, 0.60);
    let col_p = axis_percentile(t, w, h, false, 0.60);
    let (outer_t, inner_t) = outer_inner_means(t, w, h);
    let thresh = outer_t * 0.35 + inner_t * 0.65;
    let row_ok: Vec<f32> = row_p
        .iter()
        .map(|&v| if v > thresh { 1.0 } else { 0.0 })
        .collect();
    let col_ok: Vec<f32> = col_p
        .iter()
        .map(|&v| if v > thresh { 1.0 } else { 0.0 })
        .collect();
    let (top, bottom) = longest_run(&row_ok, 0.5, h / 2)?;
    let (left, right) = longest_run(&col_ok, 0.5, w / 2)?;
    let row_tex = tex_row_mean(tex, w, h);
    let col_tex = tex_col_mean(tex, w, h);
    let (top, bottom) = trim_edge_rebate(&row_p, &row_tex, top, bottom);
    let (left, right) = trim_edge_rebate(&col_p, &col_tex, left, right);
    if right <= left || bottom <= top {
        None
    } else {
        Some((left, right, top, bottom))
    }
}

fn seed_looks_complete(
    w: usize,
    h: usize,
    (left, right, top, bottom): (usize, usize, usize, usize),
) -> bool {
    if right <= left || bottom <= top {
        return false;
    }
    let rw = (right - left) as f32;
    let rh = (bottom - top) as f32;
    let area = (rw * rh) / (w.max(1) * h.max(1)) as f32;
    let ratio = rw / rh.max(1.0);
    let aspect_err = [1.333, 1.5, 0.75, 0.667]
        .into_iter()
        .map(|target| (ratio - target).abs() / target)
        .fold(1.0f32, f32::min);
    area >= 0.40 && rw / w as f32 >= 0.52 && rh / h as f32 >= 0.45 && aspect_err < 0.10
}

fn grow_bright_seed(
    t: &[f32],
    w: usize,
    h: usize,
    seed: (usize, usize, usize, usize),
    tex_thresh: f32,
    light_floor: f32,
) -> (usize, usize, usize, usize) {
    let (mut left, mut right, mut top, mut bottom) = seed;
    let side_wide =
        left as f32 / w as f32 > 0.175 || (w.saturating_sub(right)) as f32 / w as f32 > 0.175;
    if side_wide || !seed_looks_complete(w, h, seed) {
        // Recover width first. Vertical expand uses left/right as the gate
        // span; a half-frame seed cannot grow into a dark sky above it.
        let (l, r) = expand_into_dark_film(
            t,
            w,
            h,
            false,
            left,
            right,
            top,
            bottom,
            tex_thresh,
            light_floor,
        );
        left = l;
        right = r;
    }
    let (t0, b0) = expand_axis(t, w, h, true, top, bottom, left, right, tex_thresh);
    top = t0;
    bottom = b0;
    let tall_margin =
        top as f32 / h as f32 > 0.14 || (h.saturating_sub(bottom)) as f32 / h as f32 > 0.14;
    if tall_margin {
        let (t0, b0) = expand_into_dark_film(
            t,
            w,
            h,
            true,
            top,
            bottom,
            left,
            right,
            tex_thresh,
            light_floor,
        );
        top = t0;
        bottom = b0;
    }
    (left, right, top, bottom)
}

fn bounds_from_gate_edges(
    t: &[f32],
    w: usize,
    h: usize,
    polarity: Polarity,
) -> Option<(usize, usize, usize, usize)> {
    let col_p = axis_percentile(t, w, h, false, 0.50);
    let row_p = axis_percentile(t, w, h, true, 0.50);
    let film_brighter = !matches!(polarity, Polarity::SurroundBrighter);
    let (left, right) = best_gate_pair(&col_p, 0.52, 0.86, film_brighter)?;
    let (top, bottom) = best_gate_pair(&row_p, 0.45, 0.92, film_brighter)?;
    if right <= left || bottom <= top {
        None
    } else {
        Some((left, right, top, bottom))
    }
}

fn smooth_profile(values: &[f32], radius: usize) -> Vec<f32> {
    let n = values.len();
    let r = radius.max(1);
    (0..n)
        .map(|i| {
            let a = i.saturating_sub(r);
            let b = (i + r + 1).min(n);
            let s: f32 = values[a..b].iter().sum();
            s / (b - a) as f32
        })
        .collect()
}

/// Strongest opposite-edge pair whose span looks like a still frame.
fn best_gate_pair(
    profile: &[f32],
    min_span: f32,
    max_span: f32,
    rise_then_fall: bool,
) -> Option<(usize, usize)> {
    let n = profile.len();
    if n < 24 {
        return None;
    }
    let sm = smooth_profile(profile, 2);
    let step = 4usize;
    let mut deriv = vec![0.0f32; n];
    for i in 0..n.saturating_sub(step) {
        deriv[i] = sm[i + step] - sm[i];
    }
    let lo = ((n as f32 * min_span).round() as usize).max(8);
    let hi = ((n as f32 * max_span).round() as usize).min(n - 1);
    let mut best = None;
    let mut best_s = 0.12f32;
    for l in 0..(n / 2) {
        let r0 = (l + lo).min(n);
        let r1 = (l + hi).min(n);
        for r in r0..r1 {
            let left_d = deriv[l];
            let right_d = deriv[r.saturating_sub(step)];
            let s = if rise_then_fall {
                left_d.max(0.0) + (-right_d).max(0.0)
            } else {
                (-left_d).max(0.0) + right_d.max(0.0)
            };
            if s > best_s {
                best_s = s;
                best = Some((l, r));
            }
        }
    }
    best
}

fn pick_bounds(
    t: &[f32],
    w: usize,
    h: usize,
    grown: (usize, usize, usize, usize),
    image_px: Option<(usize, usize, usize, usize)>,
    gate: Option<(usize, usize, usize, usize)>,
) -> (usize, usize, usize, usize) {
    let area =
        |c: (usize, usize, usize, usize)| (c.1.saturating_sub(c.0)) * (c.3.saturating_sub(c.2));
    let frame = (w * h).max(1) as f32;
    if seed_looks_complete(w, h, grown) {
        if let Some(px) = image_px {
            let pa = area(px) as f32 / frame;
            if pa > 0.88 {
                return grown;
            }
        }
        return grown;
    }

    let score = |c: (usize, usize, usize, usize)| {
        let area_frac = area(c) as f32 / frame;
        if area_frac < 0.22 || area_frac > 0.92 {
            return -1.0;
        }
        let rw = c.1.saturating_sub(c.0).max(1) as f32;
        let rh = c.3.saturating_sub(c.2).max(1) as f32;
        let ratio = rw / rh;
        let aspect = [1.333, 1.5, 0.75, 0.667]
            .into_iter()
            .map(|target| (ratio - target).abs() / target)
            .fold(1.0f32, f32::min);
        let gates = count_side_gates(t, w, h, c) as f32;
        0.35 * gates + (1.0 - aspect).max(0.0) + area_frac.min(0.72)
    };

    let mut best = grown;
    let mut best_s = score(grown);
    if let Some(px) = image_px {
        let s = score(px);
        if s > best_s + 0.20 {
            best_s = s;
            best = px;
        }
    }
    // Gate edges are a last resort for highlight islands. They can lock
    // onto rebate / sprocket steps, so ignore them when a decent seed exists.
    let grown_area = area(grown) as f32 / frame;
    if grown_area < 0.30 {
        if let Some(g) = gate {
            let s = score(g);
            if seed_looks_complete(w, h, g) && s > best_s + 0.20 {
                best = g;
            }
        }
    }
    best
}

fn count_side_gates(
    t: &[f32],
    w: usize,
    h: usize,
    (left, right, top, bottom): (usize, usize, usize, usize),
) -> u32 {
    let band = 6usize;
    let mut n = 0u32;
    let strong = |inside: f32, outside: f32| (inside - outside).abs() > 0.06;
    if left >= band {
        let inside = strip_mean(t, w, h, left, (left + band).min(right), top, bottom);
        let outside = strip_mean(t, w, h, left - band, left, top, bottom);
        if strong(inside, outside) {
            n += 1;
        }
    }
    if right + band <= w {
        let inside = strip_mean(
            t,
            w,
            h,
            right.saturating_sub(band).max(left),
            right,
            top,
            bottom,
        );
        let outside = strip_mean(t, w, h, right, right + band, top, bottom);
        if strong(inside, outside) {
            n += 1;
        }
    }
    if top >= band {
        let inside = strip_mean(t, w, h, left, right, top, (top + band).min(bottom));
        let outside = strip_mean(t, w, h, left, right, top - band, top);
        if strong(inside, outside) {
            n += 1;
        }
    }
    if bottom + band <= h {
        let inside = strip_mean(
            t,
            w,
            h,
            left,
            right,
            bottom.saturating_sub(band).max(top),
            bottom,
        );
        let outside = strip_mean(t, w, h, left, right, bottom, bottom + band);
        if strong(inside, outside) {
            n += 1;
        }
    }
    n
}

/// Grow a bright-film seed into darker image content. Stops at holder,
/// lightbox, and flat mid-T surround (inverted-looking white border).
fn expand_into_dark_film(
    t: &[f32],
    w: usize,
    h: usize,
    rows: bool,
    start: usize,
    end: usize,
    other0: usize,
    other1: usize,
    tex_thresh: f32,
    light_floor: f32,
) -> (usize, usize) {
    let n = if rows { h } else { w };
    let holder_cap = 0.08;
    let rebate_floor = 0.86;
    let max_walk = (n / 3).max(12);
    let line_mean = |i: usize| {
        if rows {
            strip_mean(t, w, h, other0, other1, i, i + 1)
        } else {
            strip_mean(t, w, h, i, i + 1, other0, other1)
        }
    };
    let line_tex = |i: usize| {
        let m = line_mean(i);
        if rows {
            let mut s = 0.0;
            let mut c = 0.0;
            for x in other0..other1.min(w) {
                s += (t[i * w + x] - m).abs();
                c += 1.0;
            }
            if c > 0.0 {
                s / c
            } else {
                0.0
            }
        } else {
            let mut s = 0.0;
            let mut c = 0.0;
            for y in other0..other1.min(h) {
                s += (t[y * w + i] - m).abs();
                c += 1.0;
            }
            if c > 0.0 {
                s / c
            } else {
                0.0
            }
        }
    };
    let mut body_vals = Vec::new();
    for i in start..end.min(n) {
        body_vals.push(line_mean(i));
    }
    let body = if body_vals.is_empty() {
        0.5
    } else {
        percentile(&body_vals, 0.50)
    };
    let (outer, _) = outer_inner_means(t, w, h);
    let include = |i: usize| {
        if i >= n {
            return false;
        }
        let m = line_mean(i);
        if m < holder_cap {
            return false;
        }
        if m > light_floor {
            return false;
        }
        if (m - outer).abs() < 0.08 {
            return false;
        }
        let tx = line_tex(i);
        if m > body * 1.06 {
            return false;
        }
        if tx < tex_thresh * 0.7 && m > rebate_floor {
            return false;
        }
        // Mid-T uniform surround: darker than the bright seed, little grain.
        if m < body * 0.55 && tx < tex_thresh * 0.85 {
            return false;
        }
        true
    };
    let mut new_start = start;
    let mut walked = 0usize;
    while new_start > 0 && walked < max_walk && include(new_start - 1) {
        new_start -= 1;
        walked += 1;
    }
    let mut new_end = end;
    walked = 0;
    while new_end < n && walked < max_walk && include(new_end) {
        new_end += 1;
        walked += 1;
    }
    (new_start, new_end)
}

/// Grow a bright-film run into low-T content (sky, stage lights) that still
/// shares the same left/right (or top/bottom) gate. Stop at holder or rebate.
fn expand_axis(
    t: &[f32],
    w: usize,
    h: usize,
    rows: bool,
    start: usize,
    end: usize,
    other0: usize,
    other1: usize,
    tex_thresh: f32,
) -> (usize, usize) {
    let n = if rows { h } else { w };
    let holder_cap = 0.08;
    let rebate_floor = 0.86;
    let max_walk = (n / 4).max(8);
    let line_mean = |i: usize| {
        if rows {
            strip_mean(t, w, h, other0, other1, i, i + 1)
        } else {
            strip_mean(t, w, h, i, i + 1, other0, other1)
        }
    };
    let line_tex = |i: usize| {
        // Cheap: mean abs diff from line mean along the interior.
        let m = line_mean(i);
        if rows {
            let mut s = 0.0;
            let mut c = 0.0;
            for x in other0..other1.min(w) {
                s += (t[i * w + x] - m).abs();
                c += 1.0;
            }
            if c > 0.0 {
                s / c
            } else {
                0.0
            }
        } else {
            let mut s = 0.0;
            let mut c = 0.0;
            for y in other0..other1.min(h) {
                s += (t[y * w + i] - m).abs();
                c += 1.0;
            }
            if c > 0.0 {
                s / c
            } else {
                0.0
            }
        }
    };
    let include = |i: usize| {
        if i >= n {
            return false;
        }
        let m = line_mean(i);
        if m < holder_cap {
            return false;
        }
        if line_tex(i) < tex_thresh * 0.7 && m > rebate_floor {
            return false;
        }
        has_gate(t, w, h, rows, i, other0, other1)
    };
    let mut new_start = start;
    let mut walked = 0usize;
    while new_start > 0 && walked < max_walk && include(new_start - 1) {
        new_start -= 1;
        walked += 1;
    }
    let mut new_end = end;
    walked = 0;
    while new_end < n && walked < max_walk && include(new_end) {
        new_end += 1;
        walked += 1;
    }
    (new_start, new_end)
}

fn pixel_texture(t: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut tex = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let c = t[y * w + x];
            let mut acc = 0.0;
            let mut n = 0.0;
            for dy in -1i32..=1 {
                let yy = y as i32 + dy;
                if yy < 0 || yy >= h as i32 {
                    continue;
                }
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let xx = x as i32 + dx;
                    if xx < 0 || xx >= w as i32 {
                        continue;
                    }
                    acc += (t[yy as usize * w + xx as usize] - c).abs();
                    n += 1.0;
                }
            }
            tex[y * w + x] = if n > 0.0 { acc / n } else { 0.0 };
        }
    }
    tex
}

fn estimate_base(
    t: &[f32],
    t_max: &[f32],
    w: usize,
    h: usize,
    dmin_rect: Option<Rect>,
    light_level: Option<f32>,
) -> f32 {
    if let Some(r) = dmin_rect {
        let x0 = (r.x as usize).min(w.saturating_sub(1));
        let y0 = (r.y as usize).min(h.saturating_sub(1));
        let x1 = (r.x as usize + r.width as usize).min(w).max(x0 + 1);
        let y1 = (r.y as usize + r.height as usize).min(h).max(y0 + 1);
        let mut vals = Vec::with_capacity((x1 - x0) * (y1 - y0));
        for y in y0..y1 {
            for x in x0..x1 {
                vals.push(t[y * w + x]);
            }
        }
        if !vals.is_empty() {
            return percentile(&vals, 0.5).clamp(0.2, 1.6);
        }
    }

    if let Some(light) = light_level {
        let cut = light * 0.85;
        let vals: Vec<f32> = t
            .iter()
            .zip(t_max.iter())
            .filter_map(
                |(&v, &mx)| {
                    if v > 0.12 && mx < cut {
                        Some(v)
                    } else {
                        None
                    }
                },
            )
            .collect();
        if vals.len() >= t.len() / 10 {
            return percentile(&vals, 0.72).clamp(0.25, 1.5);
        }
    }

    let mut vals: Vec<f32> = t
        .iter()
        .copied()
        .filter(|&v| v > 0.12 && v < 1.35)
        .collect();
    if vals.is_empty() {
        return 1.0;
    }
    percentile(&mut vals, 0.72).clamp(0.25, 1.5)
}

fn classify_surround(
    t_max: &[f32],
    w: usize,
    h: usize,
    base: f32,
    light_level: Option<f32>,
) -> SurroundClass {
    let bw = ((w as f32 * 0.04) as usize).clamp(2, w / 8);
    let bh = ((h as f32 * 0.04) as usize).clamp(2, h / 8);
    let sides = [
        strip_class(t_max, w, h, 0, w, 0, bh, base, light_level),
        strip_class(t_max, w, h, 0, w, h - bh, h, base, light_level),
        strip_class(t_max, w, h, 0, bw, bh, h - bh, base, light_level),
        strip_class(t_max, w, h, w - bw, w, bh, h - bh, base, light_level),
    ];
    vote_surround(&sides)
}

fn strip_class(
    t_max: &[f32],
    w: usize,
    h: usize,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    base: f32,
    light_level: Option<f32>,
) -> SurroundClass {
    let x1 = x1.min(w);
    let y1 = y1.min(h);
    if x0 >= x1 || y0 >= y1 {
        return SurroundClass::Mixed;
    }
    let mut sum = 0.0;
    let mut sq = 0.0;
    let mut n = 0.0;
    for y in y0..y1 {
        for x in x0..x1 {
            let v = t_max[y * w + x];
            sum += v;
            sq += v * v;
            n += 1.0;
        }
    }
    if n < 1.0 {
        return SurroundClass::Mixed;
    }
    let mean = sum / n;
    let var = (sq / n - mean * mean).max(0.0);
    let std = var.sqrt();
    let holder_cap = (0.22 * base).max(0.12);
    let light_floor = (1.16 * base).max(1.10);
    let in_bright = light_level.map(|l| mean >= l * 0.88).unwrap_or(false);
    if mean < holder_cap && std < 0.12 {
        SurroundClass::DarkHolder
    } else if (mean > light_floor || in_bright) && std < 0.35 {
        SurroundClass::BrightLight
    } else if (mean - base).abs() < 0.22 * base.max(0.4) && std < 0.14 {
        SurroundClass::FilmRebate
    } else {
        SurroundClass::Mixed
    }
}

fn vote_surround(sides: &[SurroundClass; 4]) -> SurroundClass {
    let mut holder = 0;
    let mut light = 0;
    let mut rebate = 0;
    for s in sides {
        match s {
            SurroundClass::DarkHolder => holder += 1,
            SurroundClass::BrightLight => light += 1,
            SurroundClass::FilmRebate => rebate += 1,
            SurroundClass::Mixed => {}
        }
    }
    if holder >= 2 && holder >= light && holder >= rebate {
        SurroundClass::DarkHolder
    } else if light >= 2 && light >= holder && light >= rebate {
        SurroundClass::BrightLight
    } else if rebate >= 2 && rebate >= holder && rebate >= light {
        SurroundClass::FilmRebate
    } else {
        SurroundClass::Mixed
    }
}

fn adaptive_tex_thresh(tex: &[f32], w: usize, h: usize) -> f32 {
    let bw = ((w as f32 * 0.08) as usize).clamp(2, w / 5);
    let bh = ((h as f32 * 0.08) as usize).clamp(2, h / 5);
    let mut outer = Vec::new();
    let mut inner = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let v = tex[y * w + x];
            let on_edge = x < bw || x >= w - bw || y < bh || y >= h - bh;
            if on_edge {
                outer.push(v);
            } else if x >= w / 4 && x < 3 * w / 4 && y >= h / 4 && y < 3 * h / 4 {
                inner.push(v);
            }
        }
    }
    let o = if outer.is_empty() {
        0.01
    } else {
        percentile(&outer, 0.5)
    };
    let i = if inner.is_empty() {
        0.04
    } else {
        percentile(&inner, 0.5)
    };
    // Bias toward the quieter (outer) side so weak grain on a fogged base still counts.
    (o * 0.65 + i * 0.35).clamp(0.005, 0.085)
}

/// Mark 35mm sprocket bands as non-image so holes cannot pull the crop out.
fn mark_sprocket_rows_as_border(t: &[f32], image_px: &mut [bool], w: usize, h: usize, base: f32) {
    let band = (h / 5).max(8);
    for y in 0..h {
        if y >= band && y < h - band {
            continue;
        }
        if !is_sprocket_line(t, w, y, true, base) {
            continue;
        }
        for x in 0..w {
            image_px[y * w + x] = false;
        }
    }
    let band_x = (w / 5).max(8);
    for x in 0..w {
        if x >= band_x && x < w - band_x {
            continue;
        }
        if !is_sprocket_line(t, w, x, false, base) {
            continue;
        }
        for y in 0..h {
            image_px[y * w + x] = false;
        }
    }
}

fn is_sprocket_line(t: &[f32], w: usize, index: usize, row: bool, base: f32) -> bool {
    let n = if row { w } else { t.len() / w };
    if n < 32 {
        return false;
    }
    let sample = |i: usize| {
        if row {
            t[index * w + i]
        } else {
            t[i * w + index]
        }
    };
    let dark_cap = (0.20 * base).max(0.12);
    let mut dark = false;
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;
    for i in 0..n {
        let is_dark = sample(i) < dark_cap;
        if is_dark && !dark {
            start = i;
            dark = true;
        } else if !is_dark && dark {
            runs.push((start, i));
            dark = false;
        }
    }
    if dark {
        runs.push((start, n));
    }
    if runs.len() < 3 {
        return false;
    }
    let widths: Vec<usize> = runs.iter().map(|(a, b)| b - a).collect();
    let gaps: Vec<usize> = runs.windows(2).map(|w| w[1].0 - w[0].1).collect();
    if gaps.len() < 2 {
        return false;
    }
    let mean_w = widths.iter().sum::<usize>() as f32 / widths.len() as f32;
    let mean_g = gaps.iter().sum::<usize>() as f32 / gaps.len() as f32;
    if mean_w < 1.5 || mean_g < 4.0 || mean_w > mean_g {
        return false;
    }
    let cv = |vals: &[usize], mean: f32| {
        if mean <= 0.0 {
            return 1.0;
        }
        let var = vals
            .iter()
            .map(|&v| {
                let d = v as f32 - mean;
                d * d
            })
            .sum::<f32>()
            / vals.len() as f32;
        var.sqrt() / mean
    };
    cv(&widths, mean_w) < 0.55 && cv(&gaps, mean_g) < 0.45
}

fn longest_run(frac: &[f32], min_frac: f32, center: usize) -> Option<(usize, usize)> {
    let n = frac.len();
    if n == 0 {
        return None;
    }
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut start = None;
    for (i, &f) in frac.iter().enumerate() {
        if f >= min_frac {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start.take() {
            runs.push((s, i));
        }
    }
    if let Some(s) = start {
        runs.push((s, n));
    }
    if runs.is_empty() {
        // Fall back: treat the whole axis as image (tight frame).
        return Some((0, n));
    }
    let center_run = runs
        .iter()
        .copied()
        .find(|(s, e)| *s <= center && center < *e);
    if let Some((s, e)) = center_run {
        if (e - s) as f32 >= n as f32 * 0.28 {
            return Some((s, e));
        }
    }
    runs.into_iter().max_by_key(|(s, e)| e - s)
}

fn snap_aspect(
    left: usize,
    right: usize,
    top: usize,
    bottom: usize,
) -> (usize, usize, usize, usize, Option<FilmFormat>) {
    let rw = (right - left).max(1);
    let rh = (bottom - top).max(1);
    let ratio = rw as f32 / rh as f32;
    const CANDIDATES: [(f32, FilmFormat); 9] = [
        (3.0 / 2.0, FilmFormat::Still35),
        (2.0 / 3.0, FilmFormat::Still35),
        (1.0, FilmFormat::Square),
        (6.0 / 7.0, FilmFormat::SixBySeven),
        (7.0 / 6.0, FilmFormat::SixBySeven),
        (4.0 / 3.0, FilmFormat::FourByThree),
        (3.0 / 4.0, FilmFormat::FourByThree),
        (5.0 / 4.0, FilmFormat::FourByFive),
        (4.0 / 5.0, FilmFormat::FourByFive),
    ];
    let mut best: Option<(f32, FilmFormat)> = None;
    for (target, fmt) in CANDIDATES {
        let err = (ratio - target).abs() / target;
        if err < 0.045 && best.map(|(e, _)| err < e).unwrap_or(true) {
            best = Some((err, fmt));
        }
    }
    let Some((err, fmt)) = best else {
        return (left, right, top, bottom, None);
    };
    if err < 0.012 {
        return (left, right, top, bottom, Some(fmt));
    }
    let target = CANDIDATES
        .iter()
        .filter(|(_, f)| *f == fmt)
        .min_by(|(a, _), (b, _)| {
            (ratio - *a)
                .abs()
                .partial_cmp(&(ratio - *b).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(r, _)| *r)
        .unwrap_or(ratio);

    let mut left = left;
    let mut right = right;
    let mut top = top;
    let mut bottom = bottom;
    let cur_w = (right - left) as f32;
    let cur_h = (bottom - top) as f32;
    let target_w = cur_h * target;
    let target_h = cur_w / target;
    // Shrink the side that overshoots; do not expand into the border.
    if target_w < cur_w {
        let shrink = ((cur_w - target_w) * 0.5).round() as usize;
        left += shrink;
        right = right.saturating_sub(shrink).max(left + 1);
    } else if target_h < cur_h {
        let shrink = ((cur_h - target_h) * 0.5).round() as usize;
        top += shrink;
        bottom = bottom.saturating_sub(shrink).max(top + 1);
    }
    (left, right, top, bottom, Some(fmt))
}

fn inset_bounds(
    left: &mut usize,
    right: &mut usize,
    top: &mut usize,
    bottom: &mut usize,
    w: usize,
    h: usize,
) {
    if *right <= *left || *bottom <= *top || *right > w || *bottom > h {
        return;
    }
    let min_side = (*right - *left).min(*bottom - *top);
    let margin = (*left).min(w - *right).min(*top).min(h - *bottom);
    let inset = ((min_side as f32 * 0.006).round() as usize)
        .min(margin / 3)
        .min(6);
    if inset == 0 {
        return;
    }
    *left += inset;
    *right = right.saturating_sub(inset).max(*left + 1);
    *top += inset;
    *bottom = bottom.saturating_sub(inset).max(*top + 1);
}

fn clamp_rect(rect: Rect, w: u32, h: u32) -> Option<Rect> {
    if w == 0 || h == 0 {
        return None;
    }
    let x = rect.x.min(w.saturating_sub(1));
    let y = rect.y.min(h.saturating_sub(1));
    let width = rect.width.min(w.saturating_sub(x)).max(1);
    let height = rect.height.min(h.saturating_sub(y)).max(1);
    if width < 8 || height < 8 {
        return None;
    }
    Some(Rect {
        x,
        y,
        width,
        height,
    })
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

fn percentile(values: &[f32], p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p = p.clamp(0.0, 1.0);
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = p * (n - 1) as f32;
    let i = idx.floor() as usize;
    let frac = idx - i as f32;
    if i >= n - 1 {
        sorted[n - 1]
    } else {
        sorted[i] * (1.0 - frac) + sorted[i + 1] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set_rgb(img: &mut Array3<f32>, x: usize, y: usize, v: f32) {
        img[[y, x, 0]] = v;
        img[[y, x, 1]] = v;
        img[[y, x, 2]] = v;
    }

    fn fill_rect(img: &mut Array3<f32>, x0: usize, y0: usize, x1: usize, y1: usize, v: f32) {
        for y in y0..y1 {
            for x in x0..x1 {
                set_rgb(img, x, y, v);
            }
        }
    }

    /// Structured image content so texture distinguishes dye from rebate.
    fn fill_image_pattern(
        img: &mut Array3<f32>,
        x0: usize,
        y0: usize,
        x1: usize,
        y1: usize,
        lo: f32,
        hi: f32,
    ) {
        for y in y0..y1 {
            for x in x0..x1 {
                let u = (x - x0) as f32 * 0.21 + (y - y0) as f32 * 0.13;
                let v = lo + (hi - lo) * (0.5 + 0.5 * (u).sin());
                set_rgb(img, x, y, v);
            }
        }
    }

    fn canvas(w: usize, h: usize, fill: f32) -> Array3<f32> {
        let mut img = Array3::<f32>::zeros((h, w, 3));
        fill_rect(&mut img, 0, 0, w, h, fill);
        img
    }

    fn assert_rect_close(got: Rect, exp: (u32, u32, u32, u32), tol: u32) {
        let (x, y, w, h) = exp;
        assert!(got.x.abs_diff(x) <= tol, "x {} vs {} (tol {tol})", got.x, x);
        assert!(got.y.abs_diff(y) <= tol, "y {} vs {} (tol {tol})", got.y, y);
        assert!(
            got.width.abs_diff(w) <= tol,
            "w {} vs {} (tol {tol})",
            got.width,
            w
        );
        assert!(
            got.height.abs_diff(h) <= tol,
            "h {} vs {} (tol {tol})",
            got.height,
            h
        );
    }

    fn detect(img: &Array3<f32>) -> AutoCropResult {
        detect_crop(img, None, None).expect("auto crop should find a frame")
    }

    #[test]
    fn dark_holder_rebate_image() {
        let (w, h) = (320, 240);
        let mut img = canvas(w, h, 0.02);
        fill_rect(&mut img, 24, 24, w - 24, h - 24, 1.0);
        fill_image_pattern(&mut img, 40, 40, w - 40, h - 40, 0.32, 0.68);
        let r = detect(&img);
        assert_rect_close(r.rect, (40, 40, 240, 160), 10);
        assert_eq!(r.surround, SurroundClass::DarkHolder);
        assert_ne!(r.confidence, CropConfidence::Low);
    }

    #[test]
    fn bright_surround_rebate_image() {
        let (w, h) = (320, 240);
        let mut img = canvas(w, h, 1.55);
        fill_rect(&mut img, 24, 24, w - 24, h - 24, 1.0);
        fill_image_pattern(&mut img, 40, 40, w - 40, h - 40, 0.30, 0.65);
        let r = detect(&img);
        assert_rect_close(r.rect, (40, 40, 240, 160), 10);
        assert_eq!(r.surround, SurroundClass::BrightLight);
    }

    #[test]
    fn holder_only_no_rebate() {
        let (w, h) = (320, 240);
        let mut img = canvas(w, h, 0.02);
        fill_image_pattern(&mut img, 36, 36, w - 36, h - 36, 0.28, 0.70);
        let r = detect(&img);
        assert_rect_close(r.rect, (36, 36, 248, 168), 10);
        assert_eq!(r.surround, SurroundClass::DarkHolder);
    }

    #[test]
    fn thin_border() {
        let (w, h) = (320, 240);
        let mut img = canvas(w, h, 0.02);
        // ~2.5% border, no rebate.
        fill_image_pattern(&mut img, 8, 6, w - 8, h - 6, 0.30, 0.66);
        let r = detect(&img);
        assert_rect_close(r.rect, (8, 6, 304, 228), 12);
    }

    #[test]
    fn off_center_frame() {
        let (w, h) = (320, 240);
        let mut img = canvas(w, h, 0.02);
        // Image in the top-left; canvas centre sits in the holder.
        fill_image_pattern(&mut img, 12, 10, 152, 120, 0.30, 0.68);
        let r = detect(&img);
        assert_rect_close(r.rect, (12, 10, 140, 110), 12);
    }

    #[test]
    fn dark_sky_at_image_edge() {
        let (w, h) = (320, 240);
        let mut img = canvas(w, h, 0.02);
        fill_rect(&mut img, 28, 28, w - 28, h - 28, 1.0);
        fill_image_pattern(&mut img, 44, 44, w - 44, h - 44, 0.34, 0.62);
        // Dense (low-T) sky band at the top of the image — must stay inside the crop.
        fill_rect(&mut img, 44, 44, w - 44, 72, 0.16);
        let r = detect(&img);
        assert!(r.rect.y <= 50, "sky band was cropped away: y={}", r.rect.y);
        assert!(
            r.rect.y + r.rect.height >= 190,
            "crop too short: {:?}",
            r.rect
        );
    }

    #[test]
    fn fogged_base() {
        let (w, h) = (320, 240);
        let mut img = canvas(w, h, 0.02);
        fill_rect(&mut img, 24, 24, w - 24, h - 24, 1.0);
        // Low-contrast dye with higher-frequency grain (fogged / thin negs).
        for y in 40..h - 40 {
            for x in 40..w - 40 {
                let u = (x as f32) * 0.55 + (y as f32) * 0.41;
                let v = 0.88 + 0.07 * (0.5 + 0.5 * u.sin());
                img[[y, x, 0]] = v;
                img[[y, x, 1]] = v;
                img[[y, x, 2]] = v;
            }
        }
        let r = detect(&img);
        assert_rect_close(r.rect, (40, 40, 240, 160), 12);
    }

    #[test]
    fn vignette() {
        let (w, h) = (320, 240);
        let mut img = canvas(w, h, 0.02);
        fill_rect(&mut img, 24, 24, w - 24, h - 24, 1.0);
        fill_image_pattern(&mut img, 40, 40, w - 40, h - 40, 0.32, 0.66);
        let cx = (w as f32 - 1.0) * 0.5;
        let cy = (h as f32 - 1.0) * 0.5;
        for y in 0..h {
            for x in 0..w {
                let nx = (x as f32 - cx) / cx;
                let ny = (y as f32 - cy) / cy;
                let v = 0.74 + 0.26 * (1.0 - (nx * nx + ny * ny).min(1.0));
                for c in 0..3 {
                    img[[y, x, c]] *= v;
                }
            }
        }
        let r = detect(&img);
        assert_rect_close(r.rect, (40, 40, 240, 160), 14);
    }

    #[test]
    fn sprocket_holes_35mm() {
        let (w, h) = (360, 240);
        let mut img = canvas(w, h, 0.02);
        fill_rect(&mut img, 16, 12, w - 16, h - 12, 1.0);
        fill_image_pattern(&mut img, 16, 36, w - 16, h - 36, 0.30, 0.64);
        // Periodic holes in the top/bottom rebate (35mm long edges).
        for &(y0, y1) in &[(12usize, 32usize), (h - 32, h - 12)] {
            let mut x = 24;
            while x + 8 < w - 16 {
                fill_rect(&mut img, x, y0, x + 8, y1, 0.02);
                x += 28;
            }
        }
        let r = detect(&img);
        assert!(
            r.rect.y >= 28 && r.rect.y <= 44,
            "sprocket band not trimmed: y={}",
            r.rect.y
        );
        assert!(
            r.rect.y + r.rect.height <= (h as u32 - 28),
            "bottom sprockets included: {:?}",
            r.rect
        );
        assert!(r.rect.width >= 300, "width too tight: {}", r.rect.width);
    }

    #[test]
    fn tightly_framed_low_confidence() {
        let (w, h) = (320, 240);
        let mut img = canvas(w, h, 0.02);
        fill_image_pattern(&mut img, 2, 2, w - 2, h - 2, 0.30, 0.66);
        let r = detect(&img);
        assert!(r.rect.width >= 280, "width {}", r.rect.width);
        assert!(r.rect.height >= 200, "height {}", r.rect.height);
        assert_eq!(r.confidence, CropConfidence::Low);
    }

    #[test]
    fn dmin_rect_hint_is_optional() {
        let (w, h) = (320, 240);
        let mut img = canvas(w, h, 0.02);
        fill_rect(&mut img, 24, 24, w - 24, h - 24, 1.0);
        fill_image_pattern(&mut img, 40, 40, w - 40, h - 40, 0.32, 0.68);
        let hint = Rect {
            x: 26,
            y: 26,
            width: 10,
            height: 8,
        };
        let with =
            detect_crop(&img, Some(hint), Some((w as u32, h as u32))).expect("crop with dmin hint");
        let without = detect(&img);
        assert_rect_close(
            with.rect,
            (
                without.rect.x,
                without.rect.y,
                without.rect.width,
                without.rect.height,
            ),
            6,
        );
    }

    #[test]
    fn large_noisy_white_surround() {
        let (w, h) = (320, 240);
        let mut img = Array3::<f32>::zeros((h, w, 3));
        // ~40% of the frame is open lightbox at T≈1 with LED-like texture.
        for y in 0..h {
            for x in 0..w {
                let n = 1.0
                    + 0.035 * ((x as f32) * 0.31).sin()
                    + 0.025 * ((y as f32) * 0.27).sin()
                    + 0.015 * (((x + y) % 7) as f32 / 6.0);
                set_rgb(&mut img, x, y, n);
            }
        }
        fill_image_pattern(&mut img, 70, 60, 250, 180, 0.28, 0.62);
        let r = detect(&img);
        assert_rect_close(r.rect, (70, 60, 180, 120), 14);
        assert!(
            r.rect.x >= 50 && r.rect.x + r.rect.width <= 270,
            "crop swallowed the lightbox: {:?}",
            r.rect
        );
        assert_eq!(r.surround, SurroundClass::BrightLight);
    }

    #[test]
    fn chromatic_blue_narrowband_surround() {
        let (w, h) = (320, 240);
        let mut img = Array3::<f32>::zeros((h, w, 3));
        // Film-base D-min turns open narrowband LED blue: mean T ~1.27, max 1.8.
        for y in 0..h {
            for x in 0..w {
                let n = 0.04 * ((x as f32) * 0.29).sin() + 0.03 * ((y as f32) * 0.23).sin();
                img[[y, x, 0]] = 0.90 + n;
                img[[y, x, 1]] = 1.10 + n;
                img[[y, x, 2]] = 1.80 + n;
            }
        }
        fill_image_pattern(&mut img, 70, 60, 250, 180, 0.28, 0.62);
        let r = detect(&img);
        assert_rect_close(r.rect, (70, 60, 180, 120), 14);
        assert!(
            r.rect.x >= 50 && r.rect.x + r.rect.width <= 270,
            "crop swallowed the blue lightbox: {:?}",
            r.rect
        );
        assert_eq!(r.surround, SurroundClass::BrightLight);
    }

    #[test]
    fn dark_surround_bright_film() {
        // DSC00931-style: inverted preview looks white around a dark frame,
        // but after D-min the surround is low-T and the film (thin/dark scene) is high-T.
        let (w, h) = (320, 240);
        let mut img = canvas(w, h, 0.22);
        for y in 0..h {
            for x in 0..w {
                let n = 0.03 * ((x as f32) * 0.19).sin();
                set_rgb(&mut img, x, y, 0.22 + n);
            }
        }
        fill_image_pattern(&mut img, 36, 16, 278, 218, 0.80, 0.96);
        fill_rect(&mut img, 36, 170, 278, 218, 0.14);
        let r = detect(&img);
        assert_rect_close(r.rect, (36, 16, 242, 202), 16);
        assert!(
            r.rect.x >= 20 && r.rect.x + r.rect.width <= 300,
            "crop included the dark surround: {:?}",
            r.rect
        );
        assert_eq!(r.surround, SurroundClass::DarkHolder);
    }

    #[test]
    fn dark_and_bright_halves_inside_holder() {
        // Concert-style: one side of the film is dense (low T), the other is thin.
        // The bright-seed path must grow back into the dark half.
        let (w, h) = (320, 240);
        let mut img = canvas(w, h, 0.03);
        fill_image_pattern(&mut img, 36, 28, 156, 212, 0.22, 0.40);
        fill_image_pattern(&mut img, 156, 28, 278, 212, 0.78, 0.94);
        let r = detect(&img);
        assert_rect_close(r.rect, (36, 28, 242, 184), 18);
        assert!(
            r.rect.x <= 48 && r.rect.x + r.rect.width >= 260,
            "missed the dark half: {:?}",
            r.rect
        );
    }
}
