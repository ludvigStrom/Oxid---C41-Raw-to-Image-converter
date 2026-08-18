//! User-painted dust mask and punched-hole Telea + grain healing.

use std::hash::{Hash, Hasher};

use ndarray::Array3;
use serde::{Deserialize, Serialize};

/// Brush used while painting the dust mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DustTool {
    #[default]
    Pen,
    Eraser,
}

/// One paint or erase stroke in post-rotation image space.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DustStroke {
    pub tool: DustTool,
    pub radius: f32,
    pub points: Vec<(f32, f32)>,
}

/// Persisted dust state for a project image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ProjectDust {
    pub reference_size: (u32, u32),
    #[serde(default)]
    pub strokes: Vec<DustStroke>,
}

impl ProjectDust {
    pub fn is_empty(&self) -> bool {
        self.strokes.is_empty()
    }
}

/// Raster coverage mask (0 = clean, 255 = fully marked).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DustMask {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl DustMask {
    pub fn new(width: u32, height: u32) -> Self {
        let n = (width as usize).saturating_mul(height as usize);
        Self {
            width,
            height,
            data: vec![0; n],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0 || self.data.iter().all(|&v| v == 0)
    }

    pub fn hash(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.width.hash(&mut h);
        self.height.hash(&mut h);
        self.data.hash(&mut h);
        h.finish()
    }

    fn index(&self, x: u32, y: u32) -> Option<usize> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(y as usize * self.width as usize + x as usize)
    }
}

/// Stable hash of the stroke list (source of truth for cache keys).
pub fn hash_strokes(strokes: &[DustStroke]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    strokes.len().hash(&mut h);
    for stroke in strokes {
        stroke.tool.hash(&mut h);
        stroke.radius.to_bits().hash(&mut h);
        stroke.points.len().hash(&mut h);
        for (x, y) in &stroke.points {
            x.to_bits().hash(&mut h);
            y.to_bits().hash(&mut h);
        }
    }
    h.finish()
}

/// Stamp a soft disc into `mask`. Coordinates are in mask-pixel space.
pub fn stamp_disc(mask: &mut DustMask, cx: f32, cy: f32, radius: f32, tool: DustTool) {
    if mask.width == 0 || mask.height == 0 || radius <= 0.0 {
        return;
    }
    let r = radius.max(0.5);
    let feather = 1.5_f32.min(r);
    let x0 = (cx - r).floor().max(0.0) as i32;
    let y0 = (cy - r).floor().max(0.0) as i32;
    let x1 = (cx + r).ceil().min(mask.width as f32 - 1.0) as i32;
    let y1 = (cy + r).ceil().min(mask.height as f32 - 1.0) as i32;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > r {
                continue;
            }
            let cover = if dist <= r - feather {
                255u8
            } else {
                let t = ((r - dist) / feather).clamp(0.0, 1.0);
                (t * t * (3.0 - 2.0 * t) * 255.0).round() as u8
            };
            if cover == 0 {
                continue;
            }
            let Some(i) = mask.index(x as u32, y as u32) else {
                continue;
            };
            match tool {
                DustTool::Pen => {
                    mask.data[i] = mask.data[i].max(cover);
                }
                DustTool::Eraser => {
                    mask.data[i] = mask.data[i].saturating_sub(cover);
                }
            }
        }
    }
}

fn scale_point(x: f32, y: f32, radius: f32, from: (u32, u32), to: (u32, u32)) -> (f32, f32, f32) {
    if from.0 == 0 || from.1 == 0 {
        return (x, y, radius);
    }
    let sx = to.0 as f32 / from.0 as f32;
    let sy = to.1 as f32 / from.1 as f32;
    (x * sx, y * sy, radius * (sx + sy) * 0.5)
}

/// Replay strokes into a new mask at `width` × `height`.
///
/// `reference_size` is the image size the stroke coordinates were painted in.
pub fn rasterize_strokes(
    strokes: &[DustStroke],
    width: u32,
    height: u32,
    reference_size: (u32, u32),
) -> DustMask {
    let mut mask = DustMask::new(width, height);
    if width == 0 || height == 0 {
        return mask;
    }
    let target = (width, height);
    for stroke in strokes {
        if stroke.points.is_empty() || stroke.radius <= 0.0 {
            continue;
        }
        let mut prev: Option<(f32, f32, f32)> = None;
        for &(x, y) in &stroke.points {
            let (px, py, pr) = scale_point(x, y, stroke.radius, reference_size, target);
            if let Some((ox, oy, _)) = prev {
                let dx = px - ox;
                let dy = py - oy;
                let dist = (dx * dx + dy * dy).sqrt();
                let steps = (dist * 2.0).ceil().max(1.0) as i32;
                for s in 1..=steps {
                    let t = s as f32 / steps as f32;
                    stamp_disc(&mut mask, ox + dx * t, oy + dy * t, pr, stroke.tool);
                }
            } else {
                stamp_disc(&mut mask, px, py, pr, stroke.tool);
            }
            prev = Some((px, py, pr));
        }
    }
    mask
}

/// Nearest-neighbor scale. Used when the process buffer size differs from the mask.
pub fn scale_mask(mask: &DustMask, new_w: u32, new_h: u32) -> DustMask {
    if mask.width == new_w && mask.height == new_h {
        return mask.clone();
    }
    let mut out = DustMask::new(new_w, new_h);
    if mask.width == 0 || mask.height == 0 || new_w == 0 || new_h == 0 {
        return out;
    }
    for y in 0..new_h {
        let sy = (y as u64 * mask.height as u64 / new_h as u64) as u32;
        for x in 0..new_w {
            let sx = (x as u64 * mask.width as u64 / new_w as u64) as u32;
            if let (Some(si), Some(di)) = (mask.index(sx, sy), out.index(x, y)) {
                out.data[di] = mask.data[si];
            }
        }
    }
    out
}

/// Crop `mask` to a normalized UV rectangle and scale to `out_w` × `out_h`.
pub fn crop_mask_uv(
    mask: &DustMask,
    u0: f32,
    v0: f32,
    u1: f32,
    v1: f32,
    out_w: u32,
    out_h: u32,
) -> DustMask {
    let mut out = DustMask::new(out_w, out_h);
    if mask.width == 0 || mask.height == 0 || out_w == 0 || out_h == 0 {
        return out;
    }
    let u0 = u0.clamp(0.0, 1.0);
    let v0 = v0.clamp(0.0, 1.0);
    let u1 = u1.clamp(u0, 1.0);
    let v1 = v1.clamp(v0, 1.0);
    let src_w = ((u1 - u0) * mask.width as f32).max(1.0);
    let src_h = ((v1 - v0) * mask.height as f32).max(1.0);
    for y in 0..out_h {
        let sy = (v0 * mask.height as f32 + (y as f32 + 0.5) * src_h / out_h as f32) as i32;
        let sy = sy.clamp(0, mask.height as i32 - 1) as u32;
        for x in 0..out_w {
            let sx = (u0 * mask.width as f32 + (x as f32 + 0.5) * src_w / out_w as f32) as i32;
            let sx = sx.clamp(0, mask.width as i32 - 1) as u32;
            if let (Some(si), Some(di)) = (mask.index(sx, sy), out.index(x, y)) {
                out.data[di] = mask.data[si];
            }
        }
    }
    out
}

const MASK_ON: u8 = 16;
const MIN_SPECK_AREA: usize = 2;

/// Rim fade and grain for [`apply_dust_removal_with`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DustHealParams {
    /// Unused by heal (the paint is the hole). Kept for cache-hash compatibility.
    pub detect: f32,
    /// Fade width in pixels *outside* the painted hole (0 = core only).
    pub feather: f32,
    /// Scale on nearby high-pass grain (0 = structure only, 1 = 1:1 copy).
    pub grain: f32,
    /// Unused by heal (σ is estimated from nearby film). Kept for cache-hash compatibility.
    pub grain_sigma: f32,
}

impl Default for DustHealParams {
    fn default() -> Self {
        Self {
            detect: 1.0,
            feather: 6.0,
            grain: 1.5,
            grain_sigma: 2.0,
        }
    }
}

impl DustHealParams {
    pub fn hash(self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.detect.to_bits().hash(&mut h);
        self.feather.to_bits().hash(&mut h);
        self.grain.to_bits().hash(&mut h);
        self.grain_sigma.to_bits().hash(&mut h);
        h.finish()
    }
}

/// Hash strokes plus heal params (cache key when Process is on).
pub fn hash_dust(strokes: &[DustStroke], params: DustHealParams) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    hash_strokes(strokes).hash(&mut h);
    params.hash().hash(&mut h);
    h.finish()
}

/// Heal using default detect/feather/grain.
pub fn apply_dust_removal(image: &mut Array3<f32>, mask: &DustMask) {
    apply_dust_removal_with(image, mask, DustHealParams::default());
}

/// Replace dust in the painted stroke. Punched Telea / H+V on a masked
/// low-pass; grain is the high-pass of nearby film.
pub fn apply_dust_removal_with(image: &mut Array3<f32>, mask: &DustMask, params: DustHealParams) {
    if mask.is_empty() {
        return;
    }
    let (h, w, c) = image.dim();
    if c < 3 || w == 0 || h == 0 {
        return;
    }
    let scaled = if mask.width == w as u32 && mask.height == h as u32 {
        None
    } else {
        Some(scale_mask(mask, w as u32, h as u32))
    };
    let mask = scaled.as_ref().unwrap_or(mask);
    heal_spots(image, mask, params);
}

fn heal_spots(image: &mut Array3<f32>, mask: &DustMask, params: DustHealParams) {
    let (h, w, _) = image.dim();
    let n = w * h;
    let mut roi = vec![false; n];
    for (i, &c) in mask.data.iter().enumerate() {
        if c >= MASK_ON {
            roi[i] = true;
        }
    }
    if !roi.iter().any(|&v| v) {
        return;
    }

    let _ = params.detect;
    let _ = params.grain_sigma;
    // The painted stroke is the hole. Detect is not a gate.
    let tight = roi;

    let feather = params.feather.clamp(0.0, 16.0);
    let grain_amount = params.grain.clamp(0.0, 3.0);
    let dilate_r = feather.ceil() as i32;
    let dilated = dilate(&tight, w, h, dilate_r);
    let not_tight: Vec<bool> = tight.iter().map(|&t| !t).collect();
    let dist_from_tight = dist_inside(&not_tight, w, h);
    let mut alpha = vec![0.0f32; n];
    for i in 0..n {
        if tight[i] {
            alpha[i] = 1.0;
        } else if dilated[i] {
            alpha[i] = 1.0 - smoothstep(0.0, feather, dist_from_tight[i]);
        }
    }

    let grain_sigma = estimate_grain_sigma(image, &dilated, w, h);
    let low = blur_rgb_masked(image, &dilated, grain_sigma);
    let structure = fill_structure_telea(&low, &dilated, w, h);
    for component in connected_components(&dilated, w, h) {
        heal_component(
            image,
            &low,
            &structure,
            &component,
            &dilated,
            &alpha,
            grain_amount,
            w,
            h,
        );
    }
}

/// Split frequency from film next to the hole: finer grain → smaller σ.
fn estimate_grain_sigma(image: &Array3<f32>, hole: &[bool], w: usize, h: usize) -> f32 {
    let outer = dilate(hole, w, h, 10);
    let mut x0 = w;
    let mut y0 = h;
    let mut x1 = 0usize;
    let mut y1 = 0usize;
    let mut any = false;
    for i in 0..w * h {
        if outer[i] && !hole[i] {
            any = true;
            x0 = x0.min(i % w);
            y0 = y0.min(i / w);
            x1 = x1.max(i % w);
            y1 = y1.max(i / w);
        }
    }
    if !any {
        return 1.6;
    }

    let luma = |x: usize, y: usize| {
        0.2126 * image[(y, x, 0)] + 0.7152 * image[(y, x, 1)] + 0.0722 * image[(y, x, 2)]
    };
    let mut hp = vec![0.0f32; w * h];
    let mut has_hp = vec![false; w * h];
    let mut acc_var = 0.0f32;
    let mut n_var = 0i32;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let i = y * w + x;
            if hole[i] || !outer[i] {
                continue;
            }
            let mut s = 0.0f32;
            let mut n = 0i32;
            for dy in -2i32..=2 {
                for dx in -2i32..=2 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let ni = ny as usize * w + nx as usize;
                    if hole[ni] {
                        continue;
                    }
                    s += luma(nx as usize, ny as usize);
                    n += 1;
                }
            }
            if n < 4 {
                continue;
            }
            let v = luma(x, y) - s / n as f32;
            hp[i] = v;
            has_hp[i] = true;
            acc_var += v * v;
            n_var += 1;
        }
    }
    if n_var < 12 || acc_var < 1.0e-8 {
        return 1.6;
    }

    let mut acc_c1 = 0.0f32;
    let mut n_c1 = 0i32;
    for y in y0..=y1 {
        for x in x0..x1 {
            let i = y * w + x;
            if has_hp[i] && has_hp[i + 1] {
                acc_c1 += hp[i] * hp[i + 1];
                n_c1 += 1;
            }
        }
    }
    let var = acc_var / n_var as f32;
    let c1 = if n_c1 > 0 { acc_c1 / n_c1 as f32 } else { 0.0 };
    let corr = (c1 / var).clamp(0.0, 1.0);
    (0.95 + 1.7 * corr).clamp(0.8, 2.6)
}

#[allow(dead_code)]
fn refine_speck_mask(
    image: &Array3<f32>,
    roi: &[bool],
    w: usize,
    h: usize,
    detect: f32,
) -> Vec<bool> {
    let detect = detect.clamp(0.35, 2.5);
    let dog_thresh = (0.022 / detect).max(0.005);
    let mad_k = 2.4 / detect;
    let clip_lo = 0.025;
    let clip_hi = 0.975;

    let n = w * h;
    let mut pr = vec![0.0f32; n];
    let mut pg = vec![0.0f32; n];
    let mut pb = vec![0.0f32; n];
    let mut lum = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let r = image[(y, x, 0)];
            let g = image[(y, x, 1)];
            let b = image[(y, x, 2)];
            pr[i] = r;
            pg[i] = g;
            pb[i] = b;
            lum[i] = 0.3 * r + 0.6 * g + 0.1 * b;
        }
    }
    let blur_s = blur_plane(&lum, w, h, 0.7);
    let blur_l = blur_plane(&lum, w, h, 2.2);

    let mut residual_at = vec![0.0f32; n];
    let mut ring_res = Vec::new();
    let mut all_res = Vec::new();
    for i in 0..n {
        if !roi[i] {
            continue;
        }
        let x = i % w;
        let y = i / w;
        let mr = local_median(&pr, w, h, x, y, 3);
        let mg = local_median(&pg, w, h, x, y, 3);
        let mb = local_median(&pb, w, h, x, y, 3);
        let ml = local_median(&lum, w, h, x, y, 3);
        let chan = (pr[i] - mr).abs().max((pg[i] - mg).abs()).max((pb[i] - mb).abs());
        let chroma = (pr[i] - mr).abs() + (pg[i] - mg).abs() + (pb[i] - mb).abs();
        let r = (lum[i] - ml).abs().max(chan).max(chroma * 0.45);
        residual_at[i] = r;
        all_res.push(r);
        if roi_is_ring(roi, w, h, x, y) {
            ring_res.push(r);
        }
    }
    let mad_src = if ring_res.len() >= 8 {
        ring_res
    } else {
        robust_interior(&all_res)
    };
    let mad = {
        let mut m = mad_src.clone();
        let med = median_f32(&mut m);
        let mut dev: Vec<f32> = mad_src.iter().map(|v| (v - med).abs()).collect();
        median_f32(&mut dev).max(0.005)
    };

    let mut tight = vec![false; n];
    for i in 0..n {
        if !roi[i] {
            continue;
        }
        let dog = (blur_s[i] - blur_l[i]).abs();
        let clip = lum[i] <= clip_lo || lum[i] >= clip_hi || pr[i].max(pg[i]).max(pb[i]) >= clip_hi;
        if dog >= dog_thresh || residual_at[i] >= mad_k * mad || clip {
            tight[i] = true;
        }
    }

    // Close 1px gaps so a soft core stays one speck.
    let tight = erode(&dilate(&tight, w, h, 1), w, h, 1);

    let mut kept = vec![false; n];
    for comp in connected_components(&tight, w, h) {
        if comp.len() >= MIN_SPECK_AREA {
            for i in comp {
                if roi[i] {
                    kept[i] = true;
                }
            }
        }
    }
    kept
}

/// Inner ~3.5 px of the stroke (centroid via repeated erode). Used when detect
/// finds no speck so a large search brush does not become the hole.
#[allow(dead_code)]
fn force_middle(roi: &[bool], w: usize, h: usize) -> Vec<bool> {
    let area = roi.iter().filter(|&&v| v).count() as f32;
    if area < 1.0 {
        return roi.to_vec();
    }
    let r = (area / std::f32::consts::PI).sqrt();
    const KEEP_R: f32 = 3.5;
    let erode_r = (r - KEEP_R).ceil().max(1.0) as i32;
    let mid = erode(roi, w, h, erode_r);
    if mid.iter().any(|&v| v) {
        return mid;
    }
    let mid = erode(roi, w, h, 1);
    if mid.iter().any(|&v| v) {
        mid
    } else {
        roi.to_vec()
    }
}

#[allow(dead_code)]
fn roi_is_ring(roi: &[bool], w: usize, h: usize, x: usize, y: usize) -> bool {
    let x = x as i32;
    let y = y as i32;
    for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] {
        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
            return true;
        }
        if !roi[ny as usize * w + nx as usize] {
            return true;
        }
    }
    false
}

/// Drop the brightest residuals so a tight brush on the speck does not inflate MAD.
#[allow(dead_code)]
fn robust_interior(residuals: &[f32]) -> Vec<f32> {
    if residuals.len() < 12 {
        return residuals.to_vec();
    }
    let mut s = residuals.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let keep = (s.len() * 4 / 5).max(8);
    s.truncate(keep);
    s
}

#[allow(dead_code)]
fn local_median(lum: &[f32], w: usize, h: usize, x: usize, y: usize, radius: i32) -> f32 {
    let mut vals = Vec::with_capacity(((2 * radius + 1) * (2 * radius + 1)) as usize);
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                continue;
            }
            vals.push(lum[ny as usize * w + nx as usize]);
        }
    }
    median_f32(&mut vals)
}

fn blur_plane(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 || w == 0 || h == 0 {
        return src.to_vec();
    }
    let kernel = gauss_kernel(sigma);
    let half = kernel.len() / 2;
    let mut temp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (i, &k) in kernel.iter().enumerate() {
                let xi = (x as i32 + i as i32 - half as i32).clamp(0, w as i32 - 1) as usize;
                acc += src[y * w + xi] * k;
            }
            temp[y * w + x] = acc;
        }
    }
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (i, &k) in kernel.iter().enumerate() {
                let yi = (y as i32 + i as i32 - half as i32).clamp(0, h as i32 - 1) as usize;
                acc += temp[yi * w + x] * k;
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// Separable Gaussian that never reads `hole` pixels (renormalize the kernel).
fn blur_plane_masked(src: &[f32], hole: &[bool], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 || w == 0 || h == 0 {
        return src.to_vec();
    }
    let kernel = gauss_kernel(sigma);
    let half = kernel.len() / 2;
    let mut temp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            let mut wsum = 0.0;
            for (i, &k) in kernel.iter().enumerate() {
                let xi = (x as i32 + i as i32 - half as i32).clamp(0, w as i32 - 1) as usize;
                if hole[y * w + xi] {
                    continue;
                }
                acc += src[y * w + xi] * k;
                wsum += k;
            }
            temp[y * w + x] = if wsum > 1.0e-8 { acc / wsum } else { src[y * w + x] };
        }
    }
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            let mut wsum = 0.0;
            for (i, &k) in kernel.iter().enumerate() {
                let yi = (y as i32 + i as i32 - half as i32).clamp(0, h as i32 - 1) as usize;
                if hole[yi * w + x] {
                    continue;
                }
                acc += temp[yi * w + x] * k;
                wsum += k;
            }
            out[y * w + x] = if wsum > 1.0e-8 { acc / wsum } else { temp[y * w + x] };
        }
    }
    out
}

fn blur_rgb_masked(image: &Array3<f32>, hole: &[bool], sigma: f32) -> Array3<f32> {
    let (h, w, _) = image.dim();
    let mut planes = [
        vec![0.0f32; w * h],
        vec![0.0f32; w * h],
        vec![0.0f32; w * h],
    ];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            planes[0][i] = image[(y, x, 0)];
            planes[1][i] = image[(y, x, 1)];
            planes[2][i] = image[(y, x, 2)];
        }
    }
    let br = blur_plane_masked(&planes[0], hole, w, h, sigma);
    let bg = blur_plane_masked(&planes[1], hole, w, h, sigma);
    let bb = blur_plane_masked(&planes[2], hole, w, h, sigma);
    let mut out = Array3::<f32>::zeros((h, w, 3));
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            out[(y, x, 0)] = br[i];
            out[(y, x, 1)] = bg[i];
            out[(y, x, 2)] = bb[i];
        }
    }
    out
}

fn gauss_kernel(sigma: f32) -> Vec<f32> {
    let half = (3.0 * sigma).ceil().max(1.0) as usize;
    let len = 2 * half + 1;
    let mut k = Vec::with_capacity(len);
    let mut sum = 0.0f32;
    for i in 0..len {
        let x = i as f32 - half as f32;
        let w = (-x * x / (2.0 * sigma * sigma)).exp();
        k.push(w);
        sum += w;
    }
    for w in &mut k {
        *w /= sum;
    }
    k
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn dist_inside(on: &[bool], w: usize, h: usize) -> Vec<f32> {
    const INF: f32 = 1.0e8;
    let n = w * h;
    let mut d = vec![INF; n];
    for i in 0..n {
        if !on[i] {
            d[i] = 0.0;
        }
    }
    let s2 = std::f32::consts::SQRT_2;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !on[i] {
                continue;
            }
            let mut best = d[i];
            if x > 0 {
                best = best.min(d[i - 1] + 1.0);
            }
            if y > 0 {
                best = best.min(d[i - w] + 1.0);
            }
            if x > 0 && y > 0 {
                best = best.min(d[i - w - 1] + s2);
            }
            if x + 1 < w && y > 0 {
                best = best.min(d[i - w + 1] + s2);
            }
            d[i] = best;
        }
    }
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let i = y * w + x;
            if !on[i] {
                continue;
            }
            let mut best = d[i];
            if x + 1 < w {
                best = best.min(d[i + 1] + 1.0);
            }
            if y + 1 < h {
                best = best.min(d[i + w] + 1.0);
            }
            if x + 1 < w && y + 1 < h {
                best = best.min(d[i + w + 1] + s2);
            }
            if x > 0 && y + 1 < h {
                best = best.min(d[i + w - 1] + s2);
            }
            d[i] = best;
        }
    }
    d
}

fn erode(on: &[bool], w: usize, h: usize, radius: i32) -> Vec<bool> {
    if radius <= 0 {
        return on.to_vec();
    }
    let r2 = radius * radius;
    let mut out = vec![false; w * h];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            if !on[y as usize * w + x as usize] {
                continue;
            }
            let mut ok = true;
            'n: for dy in -radius..=radius {
                for dx in -radius..=radius {
                    if dx * dx + dy * dy > r2 {
                        continue;
                    }
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        ok = false;
                        break 'n;
                    }
                    if !on[ny as usize * w + nx as usize] {
                        ok = false;
                        break 'n;
                    }
                }
            }
            out[y as usize * w + x as usize] = ok;
        }
    }
    out
}

fn dilate(on: &[bool], w: usize, h: usize, radius: i32) -> Vec<bool> {
    let mut out = on.to_vec();
    if radius <= 0 {
        return out;
    }
    let r2 = radius * radius;
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            if !on[y as usize * w + x as usize] {
                continue;
            }
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    if dx * dx + dy * dy > r2 {
                        continue;
                    }
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    out[ny as usize * w + nx as usize] = true;
                }
            }
        }
    }
    out
}

fn connected_components(on: &[bool], w: usize, h: usize) -> Vec<Vec<usize>> {
    let mut seen = vec![false; w * h];
    let mut out = Vec::new();
    for start in 0..w * h {
        if !on[start] || seen[start] {
            continue;
        }
        let mut stack = vec![start];
        let mut comp = Vec::new();
        seen[start] = true;
        while let Some(p) = stack.pop() {
            comp.push(p);
            let x = (p % w) as i32;
            let y = (p / w) as i32;
            for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] {
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    continue;
                }
                let ni = ny as usize * w + nx as usize;
                if on[ni] && !seen[ni] {
                    seen[ni] = true;
                    stack.push(ni);
                }
            }
        }
        out.push(comp);
    }
    out
}

fn heal_component(
    image: &mut Array3<f32>,
    low: &Array3<f32>,
    structure: &Array3<f32>,
    component: &[usize],
    forbidden: &[bool],
    alpha: &[f32],
    grain_amount: f32,
    w: usize,
    h: usize,
) {
    if component.is_empty() {
        return;
    }

    let ring = component_ring(component, forbidden, w, h);
    let offset = nearby_grain_offset(image, low, &ring, forbidden, w, h);
    let residuals = ring_residuals(image, low, &ring);
    let ring_xy: Vec<(i32, i32)> = ring
        .iter()
        .map(|&i| ((i % w) as i32, (i / w) as i32))
        .collect();

    let mut fills = Vec::with_capacity(component.len());
    for &i in component {
        let x = i % w;
        let y = i / w;
        let color = rgb_at(structure, x, y);
        let hsh = pixel_hash(x, y);
        // Cardinal ±1 on about half the pixels; the rest stay aligned so
        // grain does not look fully scrambled.
        let (jx, jy) = match hsh % 8 {
            0 => (-1, 0),
            1 => (1, 0),
            2 => (0, -1),
            3 => (0, 1),
            _ => (0, 0),
        };
        let x = x as i32;
        let y = y as i32;
        let grain = if let Some((dx, dy)) = offset {
            residual_at(image, low, forbidden, x + dx + jx, y + dy + jy, w, h)
                .or_else(|| residual_at(image, low, forbidden, x + dx, y + dy, w, h))
        } else {
            None
        };
        let (gr, gg, gb) = grain.unwrap_or_else(|| {
            pick_nearby_ring_residual(x, y, hsh, &ring_xy, &residuals)
        });
        fills.push((
            color.0 + grain_amount * gr,
            color.1 + grain_amount * gg,
            color.2 + grain_amount * gb,
        ));
    }
    for (&i, (r, g, b)) in component.iter().zip(fills) {
        let a = alpha[i];
        if a <= 1.0e-5 {
            continue;
        }
        let x = i % w;
        let y = i / w;
        image[(y, x, 0)] = image[(y, x, 0)] * (1.0 - a) + r * a;
        image[(y, x, 1)] = image[(y, x, 1)] * (1.0 - a) + g * a;
        image[(y, x, 2)] = image[(y, x, 2)] * (1.0 - a) + b * a;
    }
}

fn pixel_hash(x: usize, y: usize) -> u32 {
    let mut n = (x as u32)
        .wrapping_mul(0x9E37_79B9)
        ^ (y as u32).wrapping_mul(0x85EB_CA6B);
    n ^= n >> 16;
    n = n.wrapping_mul(0x7FEB_352D);
    n ^= n >> 15;
    n
}

fn residual_at(
    image: &Array3<f32>,
    low: &Array3<f32>,
    forbidden: &[bool],
    x: i32,
    y: i32,
    w: usize,
    h: usize,
) -> Option<(f32, f32, f32)> {
    if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
        return None;
    }
    let i = y as usize * w + x as usize;
    if forbidden[i] {
        return None;
    }
    let x = x as usize;
    let y = y as usize;
    Some((
        image[(y, x, 0)] - low[(y, x, 0)],
        image[(y, x, 1)] - low[(y, x, 1)],
        image[(y, x, 2)] - low[(y, x, 2)],
    ))
}

fn rgb_at(img: &Array3<f32>, x: usize, y: usize) -> (f32, f32, f32) {
    (img[(y, x, 0)], img[(y, x, 1)], img[(y, x, 2)])
}

fn lerp3(a: (f32, f32, f32), b: (f32, f32, f32), t: f32) -> (f32, f32, f32) {
    (
        a.0 + (b.0 - a.0) * t,
        a.1 + (b.1 - a.1) * t,
        a.2 + (b.2 - a.2) * t,
    )
}

fn hole_run_h(hole: &[bool], w: usize, x: usize, y: usize) -> (Option<usize>, Option<usize>) {
    let row = y * w;
    let mut left = None;
    if x > 0 {
        let mut xi = x;
        while xi > 0 {
            xi -= 1;
            if !hole[row + xi] {
                left = Some(xi);
                break;
            }
        }
    }
    let mut right = None;
    let mut xi = x + 1;
    while xi < w {
        if !hole[row + xi] {
            right = Some(xi);
            break;
        }
        xi += 1;
    }
    (left, right)
}

fn hole_run_v(hole: &[bool], w: usize, h: usize, x: usize, y: usize) -> (Option<usize>, Option<usize>) {
    let mut top = None;
    if y > 0 {
        let mut yi = y;
        while yi > 0 {
            yi -= 1;
            if !hole[yi * w + x] {
                top = Some(yi);
                break;
            }
        }
    }
    let mut bottom = None;
    let mut yi = y + 1;
    while yi < h {
        if !hole[yi * w + x] {
            bottom = Some(yi);
            break;
        }
        yi += 1;
    }
    (top, bottom)
}

/// Lerp `low` from known endpoints. One-sided copies that edge with a large gap
/// so a real two-sided lerp in the other axis wins the blend.
fn interp_axis(
    low: &Array3<f32>,
    a: Option<usize>,
    b: Option<usize>,
    pos: usize,
    fixed: usize,
    horizontal: bool,
) -> Option<((f32, f32, f32), f32)> {
    let sample = |p: usize| {
        if horizontal {
            rgb_at(low, p, fixed)
        } else {
            rgb_at(low, fixed, p)
        }
    };
    match (a, b) {
        (Some(p0), Some(p1)) if p1 != p0 => {
            let t = (pos as f32 - p0 as f32) / (p1 as f32 - p0 as f32);
            Some((lerp3(sample(p0), sample(p1), t), (p1.abs_diff(p0)) as f32))
        }
        (Some(p0), Some(_)) => Some((sample(p0), 1.0)),
        (Some(p0), None) => Some((sample(p0), 1.0e6)),
        (None, Some(p1)) => Some((sample(p1), 1.0e6)),
        (None, None) => None,
    }
}

const TELEA_RADIUS: i32 = 4;

/// Telea-style weights over **original** known film only. Filled pixels are
/// never sampled again (that was the inward smear). No gradient term — that
/// continued any leftover halo into the hole. Interior pixels fall back to H+V.
fn fill_structure_telea(image: &Array3<f32>, hole: &[bool], w: usize, h: usize) -> Array3<f32> {
    let n = w * h;
    let mut out = image.clone();
    let mut original = vec![false; n];
    let mut pixels = Vec::new();
    for i in 0..n {
        if hole[i] {
            pixels.push(i);
        } else {
            original[i] = true;
        }
    }
    if pixels.is_empty() {
        return out;
    }
    let t = dist_inside(hole, w, h);
    pixels.sort_by(|&a, &b| t[a].partial_cmp(&t[b]).unwrap_or(std::cmp::Ordering::Equal));
    for i in pixels {
        let x = i % w;
        let y = i / w;
        let color = telea_estimate(image, &original, &t, x, y, w, h)
            .or_else(|| structure_hv(image, hole, x, y, w, h))
            .unwrap_or((0.0, 0.0, 0.0));
        out[(y, x, 0)] = color.0;
        out[(y, x, 1)] = color.1;
        out[(y, x, 2)] = color.2;
    }
    out
}

fn telea_estimate(
    img: &Array3<f32>,
    original: &[bool],
    t: &[f32],
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) -> Option<(f32, f32, f32)> {
    let (nx, ny) = normal_t(t, x, y, w, h);
    let tp = t[y * w + x];
    let mut acc = (0.0f32, 0.0f32, 0.0f32);
    let mut wsum = 0.0f32;
    for dy in -TELEA_RADIUS..=TELEA_RADIUS {
        for dx in -TELEA_RADIUS..=TELEA_RADIUS {
            if dx == 0 && dy == 0 {
                continue;
            }
            if dx * dx + dy * dy > TELEA_RADIUS * TELEA_RADIUS {
                continue;
            }
            let qx = x as i32 + dx;
            let qy = y as i32 + dy;
            if qx < 0 || qy < 0 || qx >= w as i32 || qy >= h as i32 {
                continue;
            }
            let qx = qx as usize;
            let qy = qy as usize;
            if !original[qy * w + qx] {
                continue;
            }
            let ddx = x as f32 - qx as f32;
            let ddy = y as f32 - qy as f32;
            let len2 = ddx * ddx + ddy * ddy;
            if len2 < 1.0e-8 {
                continue;
            }
            let len = len2.sqrt();
            let dir = (ddx / len * nx + ddy / len * ny).max(0.0);
            let dst = 1.0 / len2;
            let lev = 1.0 / (1.0 + (tp - t[qy * w + qx]).abs());
            let wgt = dir * dst * lev;
            if wgt <= 1.0e-12 {
                continue;
            }
            let iq = rgb_at(img, qx, qy);
            acc.0 += wgt * iq.0;
            acc.1 += wgt * iq.1;
            acc.2 += wgt * iq.2;
            wsum += wgt;
        }
    }
    if wsum > 1.0e-8 {
        Some((acc.0 / wsum, acc.1 / wsum, acc.2 / wsum))
    } else {
        None
    }
}

fn normal_t(t: &[f32], x: usize, y: usize, w: usize, h: usize) -> (f32, f32) {
    let i = y * w + x;
    let xm = if x > 0 { t[i - 1] } else { t[i] };
    let xp = if x + 1 < w { t[i + 1] } else { t[i] };
    let ym = if y > 0 { t[i - w] } else { t[i] };
    let yp = if y + 1 < h { t[i + w] } else { t[i] };
    let gx = (xp - xm) * 0.5;
    let gy = (yp - ym) * 0.5;
    let len = (gx * gx + gy * gy).sqrt();
    if len < 1.0e-6 {
        (0.0, 0.0)
    } else {
        (gx / len, gy / len)
    }
}

/// Weighted H+V lerp of known endpoints. Shorter span wins.
fn structure_hv(
    low: &Array3<f32>,
    hole: &[bool],
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) -> Option<(f32, f32, f32)> {
    let (xl, xr) = hole_run_h(hole, w, x, y);
    let (yt, yb) = hole_run_v(hole, w, h, x, y);
    let horiz = interp_axis(low, xl, xr, x, y, true);
    let vert = interp_axis(low, yt, yb, y, x, false);
    match (horiz, vert) {
        (Some((hc, gap_h)), Some((vc, gap_v))) => {
            let w_h = 1.0 / gap_h.max(1.0);
            let w_v = 1.0 / gap_v.max(1.0);
            let s = w_h + w_v;
            Some(lerp3(hc, vc, w_v / s))
        }
        (Some((hc, _)), None) => Some(hc),
        (None, Some((vc, _))) => Some(vc),
        (None, None) => None,
    }
}

fn ring_residuals(image: &Array3<f32>, low: &Array3<f32>, ring: &[usize]) -> Vec<(f32, f32, f32)> {
    let w = image.dim().1;
    ring.iter()
        .map(|&i| {
            let x = i % w;
            let y = i / w;
            (
                image[(y, x, 0)] - low[(y, x, 0)],
                image[(y, x, 1)] - low[(y, x, 1)],
                image[(y, x, 2)] - low[(y, x, 2)],
            )
        })
        .collect()
}

fn component_ring(component: &[usize], forbidden: &[bool], w: usize, h: usize) -> Vec<usize> {
    let mut ring = Vec::new();
    let mut seen = vec![false; w * h];
    for &i in component {
        let x = (i % w) as i32;
        let y = (i / w) as i32;
        for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] {
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                continue;
            }
            let ni = ny as usize * w + nx as usize;
            if !forbidden[ni] && !seen[ni] {
                seen[ni] = true;
                ring.push(ni);
            }
        }
    }
    ring
}

fn median_f32(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    if v.len() % 2 == 1 {
        v[mid]
    } else {
        (v[mid - 1] + v[mid]) * 0.5
    }
}

fn pick_nearby_ring_residual(
    x: i32,
    y: i32,
    hsh: u32,
    ring_xy: &[(i32, i32)],
    residuals: &[(f32, f32, f32)],
) -> (f32, f32, f32) {
    if residuals.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut best = [(i32::MAX, 0usize); 4];
    for (k, &(rx, ry)) in ring_xy.iter().enumerate() {
        let d = (rx - x) * (rx - x) + (ry - y) * (ry - y);
        if d < best[3].0 {
            best[3] = (d, k);
            best.sort_unstable_by_key(|p| p.0);
        }
    }
    let n = best.iter().filter(|p| p.0 < i32::MAX).count().max(1);
    residuals[best[hsh as usize % n].1]
}

/// Copy a coherent grain patch from film just outside the hole.
fn nearby_grain_offset(
    image: &Array3<f32>,
    low: &Array3<f32>,
    ring: &[usize],
    forbidden: &[bool],
    w: usize,
    h: usize,
) -> Option<(i32, i32)> {
    if ring.is_empty() {
        return None;
    }
    let mut candidates = [(0i32, 0i32); 48];
    let mut n = 0usize;
    for r in 3i32..=8 {
        for (dx, dy) in [
            (r, 0),
            (-r, 0),
            (0, r),
            (0, -r),
            (r, r),
            (r, -r),
            (-r, r),
            (-r, -r),
        ] {
            candidates[n] = (dx, dy);
            n += 1;
        }
    }
    let mut best: Option<(f32, (i32, i32))> = None;
    for &(dx, dy) in &candidates[..n] {
        let mut ssd = 0.0f32;
        let mut count = 0i32;
        for &i in ring {
            let x = (i % w) as i32 + dx;
            let y = (i / w) as i32 + dy;
            if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
                continue;
            }
            let sx = x as usize;
            let sy = y as usize;
            if forbidden[sy * w + sx] {
                continue;
            }
            let rx = i % w;
            let ry = i / w;
            let d0 = (image[(ry, rx, 0)] - low[(ry, rx, 0)]) - (image[(sy, sx, 0)] - low[(sy, sx, 0)]);
            let d1 = (image[(ry, rx, 1)] - low[(ry, rx, 1)]) - (image[(sy, sx, 1)] - low[(sy, sx, 1)]);
            let d2 = (image[(ry, rx, 2)] - low[(ry, rx, 2)]) - (image[(sy, sx, 2)] - low[(sy, sx, 2)]);
            ssd += d0 * d0 + d1 * d1 + d2 * d2;
            count += 1;
        }
        if count < (ring.len() as i32 / 2).max(4) {
            continue;
        }
        let score = ssd / count as f32;
        if best.map(|(b, _)| score < b).unwrap_or(true) {
            best = Some((score, (dx, dy)));
        }
    }
    best.map(|(_, off)| off)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn stamp_and_erase_disc() {
        let mut mask = DustMask::new(32, 32);
        stamp_disc(&mut mask, 16.0, 16.0, 4.0, DustTool::Pen);
        assert!(mask.data.iter().any(|&v| v > 200));
        stamp_disc(&mut mask, 16.0, 16.0, 6.0, DustTool::Eraser);
        assert!(mask.data.iter().all(|&v| v < 8));
    }

    #[test]
    fn rasterize_scales_from_reference() {
        let strokes = vec![DustStroke {
            tool: DustTool::Pen,
            radius: 2.0,
            points: vec![(8.0, 8.0)],
        }];
        let small = rasterize_strokes(&strokes, 16, 16, (16, 16));
        let big = rasterize_strokes(&strokes, 32, 32, (16, 16));
        assert!(small.data[8 * 16 + 8] > 200);
        assert!(big.data[16 * 32 + 16] > 200);
        assert_ne!(hash_strokes(&strokes), 0);
    }

    #[test]
    fn scale_mask_nearest_preserves_spot() {
        let mut src = DustMask::new(4, 4);
        src.data[1 * 4 + 1] = 255;
        let dst = scale_mask(&src, 8, 8);
        assert_eq!(dst.data[2 * 8 + 2], 255);
        assert_eq!(dst.data[0], 0);
    }

    #[test]
    fn crop_mask_uv_keeps_center_spot() {
        let mut src = DustMask::new(10, 10);
        src.data[5 * 10 + 5] = 255;
        let crop = crop_mask_uv(&src, 0.4, 0.4, 0.7, 0.7, 6, 6);
        assert!(crop.data.iter().any(|&v| v == 255));
    }

    #[test]
    fn inpaint_removes_synthetic_spot() {
        let mut img = Array3::<f32>::from_elem((24, 24, 3), 0.4);
        for y in 0..24 {
            for x in 0..24 {
                img[(y, x, 0)] = 0.2 + x as f32 * 0.02;
                img[(y, x, 1)] = 0.3;
                img[(y, x, 2)] = 0.5 - y as f32 * 0.01;
            }
        }
        let expected = [img[(12, 12, 0)], img[(12, 12, 1)], img[(12, 12, 2)]];
        img[(12, 12, 0)] = 1.0;
        img[(12, 12, 1)] = 0.0;
        img[(12, 12, 2)] = 1.0;
        img[(12, 13, 0)] = 1.0;
        img[(13, 12, 0)] = 1.0;

        let mut mask = DustMask::new(24, 24);
        stamp_disc(&mut mask, 12.0, 12.0, 2.0, DustTool::Pen);
        apply_dust_removal(&mut img, &mask);

        assert!((img[(12, 12, 0)] - expected[0]).abs() < 0.18);
        assert!((img[(12, 12, 1)] - expected[1]).abs() < 0.18);
        assert!((img[(12, 12, 2)] - expected[2]).abs() < 0.18);
        assert!(
            img[(12, 12, 0)] < 0.7,
            "spot should be replaced, not smeared"
        );
    }

    #[test]
    fn empty_mask_is_noop() {
        let mut img = Array3::<f32>::from_elem((4, 4, 3), 0.25);
        apply_dust_removal(&mut img, &DustMask::new(4, 4));
        assert!((img[(0, 0, 0)] - 0.25).abs() < 1e-6);
    }

    fn gradient_image(h: usize, w: usize) -> Array3<f32> {
        let mut img = Array3::<f32>::zeros((h, w, 3));
        for y in 0..h {
            for x in 0..w {
                img[(y, x, 0)] = 0.2 + x as f32 * 0.015;
                img[(y, x, 1)] = 0.35;
                img[(y, x, 2)] = 0.55 - y as f32 * 0.008;
            }
        }
        img
    }

    #[test]
    fn painted_disc_is_the_hole() {
        let mut img = gradient_image(48, 48);
        let outside = img[(24, 5, 0)];
        img[(24, 24, 0)] = 1.0;
        img[(24, 24, 1)] = 1.0;
        img[(24, 24, 2)] = 1.0;
        img[(24, 25, 0)] = 1.0;
        img[(25, 24, 0)] = 1.0;
        img[(25, 25, 0)] = 1.0;
        img[(24, 11, 0)] = 0.95;
        img[(24, 11, 1)] = 0.05;
        img[(24, 11, 2)] = 0.05;

        let mut mask = DustMask::new(48, 48);
        stamp_disc(&mut mask, 24.0, 24.0, 16.0, DustTool::Pen);
        apply_dust_removal_with(
            &mut img,
            &mask,
            DustHealParams {
                detect: 1.0,
                feather: 2.0,
                grain: 0.0,
                grain_sigma: 0.8,
            },
        );

        assert!(
            (img[(24, 5, 0)] - outside).abs() < 1e-5,
            "pixels outside the paint must stay"
        );
        assert!(img[(24, 24, 0)] < 0.85, "the painted core must be healed");
        assert!(
            img[(24, 11, 0)] < 0.7,
            "interior of a large brush is now part of the hole (got {})",
            img[(24, 11, 0)]
        );
    }

    #[test]
    fn faint_paint_is_healed_without_detect() {
        let mut img = Array3::<f32>::from_elem((40, 40, 3), 0.35);
        img[(20, 20, 0)] = 0.37;
        img[(20, 20, 1)] = 0.37;
        img[(20, 20, 2)] = 0.37;
        img[(20, 21, 0)] = 0.37;
        img[(21, 20, 0)] = 0.37;
        let outside = img[(20, 4, 0)];

        let mut mask = DustMask::new(40, 40);
        stamp_disc(&mut mask, 20.0, 20.0, 12.0, DustTool::Pen);
        apply_dust_removal_with(
            &mut img,
            &mask,
            DustHealParams {
                detect: 0.5,
                feather: 1.0,
                grain: 0.0,
                grain_sigma: 0.8,
            },
        );
        assert!(
            (img[(20, 20, 0)] - 0.35).abs() < 0.03,
            "faint specks must still be punched when the mask is trusted (got {})",
            img[(20, 20, 0)]
        );
        assert!(
            (img[(20, 4, 0)] - outside).abs() < 1e-5,
            "pixels outside the paint must stay"
        );
    }

    #[test]
    fn healed_core_matches_local_color() {
        let mut img = gradient_image(40, 40);
        let local = [img[(20, 20, 0)], img[(20, 20, 1)], img[(20, 20, 2)]];
        img[(20, 20, 0)] = 0.0;
        img[(20, 20, 1)] = 0.0;
        img[(20, 20, 2)] = 0.0;
        img[(20, 21, 0)] = 0.0;
        img[(21, 20, 0)] = 0.0;

        let mut mask = DustMask::new(40, 40);
        stamp_disc(&mut mask, 20.0, 20.0, 5.0, DustTool::Pen);
        apply_dust_removal(&mut img, &mask);

        assert!((img[(20, 20, 0)] - local[0]).abs() < 0.12);
        assert!((img[(20, 20, 1)] - local[1]).abs() < 0.12);
        assert!((img[(20, 20, 2)] - local[2]).abs() < 0.12);
    }

    #[test]
    fn healed_pixels_keep_grain_variance() {
        let mut img = gradient_image(36, 36);
        for y in 0..36 {
            for x in 0..36 {
                let n = ((x * 17 + y * 31) % 7) as f32 * 0.012 - 0.036;
                img[(y, x, 0)] = (img[(y, x, 0)] + n).clamp(0.0, 1.0);
                img[(y, x, 1)] = (img[(y, x, 1)] + n * 0.8).clamp(0.0, 1.0);
                img[(y, x, 2)] = (img[(y, x, 2)] + n * 0.6).clamp(0.0, 1.0);
            }
        }
        img[(18, 18, 0)] = 1.0;
        img[(18, 18, 1)] = 1.0;
        img[(18, 18, 2)] = 1.0;
        img[(18, 19, 0)] = 1.0;
        img[(19, 18, 0)] = 1.0;

        let mut mask = DustMask::new(36, 36);
        stamp_disc(&mut mask, 18.0, 18.0, 4.0, DustTool::Pen);
        apply_dust_removal(&mut img, &mask);

        let mut healed = Vec::new();
        for y in 17..20 {
            for x in 17..20 {
                healed.push(img[(y, x, 0)]);
            }
        }
        let spread = healed.iter().copied().fold(0.0f32, f32::max)
            - healed.iter().copied().fold(1.0f32, f32::min);
        assert!(
            spread > 0.008,
            "healed patch should not be a flat median (spread={spread})"
        );
    }

    #[test]
    fn hash_dust_includes_params() {
        let strokes = vec![DustStroke {
            tool: DustTool::Pen,
            radius: 3.0,
            points: vec![(1.0, 1.0)],
        }];
        let a = hash_dust(&strokes, DustHealParams::default());
        let b = hash_dust(
            &strokes,
            DustHealParams {
                detect: 1.4,
                feather: 4.0,
                grain: 1.5,
                grain_sigma: 0.8,
            },
        );
        let c = hash_dust(
            &strokes,
            DustHealParams {
                grain: 2.5,
                ..DustHealParams::default()
            },
        );
        assert_ne!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn detect_picks_faint_colored_two_pixel_speck() {
        let mut img = gradient_image(40, 40);
        let before = [img[(20, 20, 0)], img[(20, 20, 1)], img[(20, 20, 2)]];
        // Same max-channel-ish luma, but a cyan tint — old max(R,G,B) detect missed this.
        img[(20, 20, 1)] = (before[1] + 0.11).min(1.0);
        img[(20, 20, 2)] = (before[2] + 0.11).min(1.0);
        img[(20, 21, 1)] = (img[(20, 21, 1)] + 0.11).min(1.0);
        img[(20, 21, 2)] = (img[(20, 21, 2)] + 0.11).min(1.0);

        let mut mask = DustMask::new(40, 40);
        stamp_disc(&mut mask, 20.0, 20.0, 8.0, DustTool::Pen);
        apply_dust_removal_with(
            &mut img,
            &mask,
            DustHealParams {
                detect: 1.0,
                feather: 2.0,
                grain: 0.0,
                grain_sigma: 0.8,
            },
        );
        assert!(
            (img[(20, 20, 1)] - before[1]).abs() < 0.08,
            "faint colored speck should be detected and replaced (g={})",
            img[(20, 20, 1)]
        );
    }

    #[test]
    fn large_feather_fully_replaces_speck_core() {
        let mut img = gradient_image(48, 48);
        let local = [img[(24, 24, 0)], img[(24, 24, 1)], img[(24, 24, 2)]];
        img[(24, 24, 0)] = 1.0;
        img[(24, 24, 1)] = 1.0;
        img[(24, 24, 2)] = 1.0;
        img[(24, 25, 0)] = 1.0;
        img[(25, 24, 0)] = 1.0;
        img[(25, 25, 0)] = 1.0;

        let mut mask = DustMask::new(48, 48);
        stamp_disc(&mut mask, 24.0, 24.0, 6.0, DustTool::Pen);
        apply_dust_removal_with(
            &mut img,
            &mask,
            DustHealParams {
                detect: 1.0,
                feather: 12.0,
                grain: 0.0,
                grain_sigma: 0.8,
            },
        );

        assert!(
            (img[(24, 24, 0)] - local[0]).abs() < 0.12,
            "core must be fully replaced even with a wide feather (got {})",
            img[(24, 24, 0)]
        );
        assert!(
            img[(24, 24, 0)] < 0.75,
            "must not leak the white speck through the core (got {})",
            img[(24, 24, 0)]
        );
    }

    fn heal_core_spread(img: &Array3<f32>, cx: usize, cy: usize) -> f32 {
        let mut vals = Vec::new();
        for y in cy.saturating_sub(1)..=cy + 1 {
            for x in cx.saturating_sub(1)..=cx + 1 {
                vals.push(img[(y, x, 0)]);
            }
        }
        vals.iter().copied().fold(0.0f32, f32::max)
            - vals.iter().copied().fold(1.0f32, f32::min)
    }

    #[test]
    fn grain_amount_scales_residual() {
        let mut base = gradient_image(40, 40);
        for y in 0..40 {
            for x in 0..40 {
                let n = ((x * 17 + y * 31) % 7) as f32 * 0.02 - 0.06;
                base[(y, x, 0)] = (base[(y, x, 0)] + n).clamp(0.0, 1.0);
                base[(y, x, 1)] = (base[(y, x, 1)] + n * 0.8).clamp(0.0, 1.0);
                base[(y, x, 2)] = (base[(y, x, 2)] + n * 0.6).clamp(0.0, 1.0);
            }
        }
        base[(20, 20, 0)] = 1.0;
        base[(20, 20, 1)] = 1.0;
        base[(20, 20, 2)] = 1.0;
        base[(20, 21, 0)] = 1.0;
        base[(21, 20, 0)] = 1.0;

        let mut mask = DustMask::new(40, 40);
        stamp_disc(&mut mask, 20.0, 20.0, 5.0, DustTool::Pen);

        let mut none = base.clone();
        apply_dust_removal_with(
            &mut none,
            &mask,
            DustHealParams {
                detect: 1.0,
                feather: 2.0,
                grain: 0.0,
                grain_sigma: 0.8,
            },
        );
        let mut heavy = base.clone();
        apply_dust_removal_with(
            &mut heavy,
            &mask,
            DustHealParams {
                detect: 1.0,
                feather: 2.0,
                grain: 3.0,
                grain_sigma: 0.8,
            },
        );

        let spread_none = heal_core_spread(&none, 20, 20);
        let spread_heavy = heal_core_spread(&heavy, 20, 20);
        assert!(
            spread_heavy > spread_none + 0.008,
            "grain amount should scale residual (none={spread_none}, heavy={spread_heavy})"
        );
    }

    #[test]
    fn grain_comes_from_nearby_highpass() {
        let mut img = Array3::<f32>::from_elem((48, 48, 3), 0.4);
        for y in 0..48 {
            for x in 0..48 {
                let r = ((x as i32 - 24).pow(2) + (y as i32 - 24).pow(2)) as f32;
                let r = r.sqrt();
                if (6.0..14.0).contains(&r) {
                    let n = ((x * 13 + y * 29) % 5) as f32 * 0.04 - 0.08;
                    img[(y, x, 0)] = (0.4 + n).clamp(0.0, 1.0);
                    img[(y, x, 1)] = (0.4 + n).clamp(0.0, 1.0);
                    img[(y, x, 2)] = (0.4 + n).clamp(0.0, 1.0);
                }
            }
        }
        img[(24, 24, 0)] = 1.0;
        img[(24, 24, 1)] = 1.0;
        img[(24, 24, 2)] = 1.0;
        img[(24, 25, 0)] = 1.0;
        img[(25, 24, 0)] = 1.0;

        let mut mask = DustMask::new(48, 48);
        stamp_disc(&mut mask, 24.0, 24.0, 5.0, DustTool::Pen);
        apply_dust_removal(&mut img, &mask);

        let spread = heal_core_spread(&img, 24, 24);
        assert!(
            spread > 0.015,
            "hole should inherit nearby high-pass grain (spread={spread})"
        );
    }

    #[test]
    fn hv_lerp_follows_step_edge_not_ring_median() {
        // Left 0.2 / right 0.8. A tall hole sits on the 0.8 side, touching the step.
        // Short H span → H lerp ~0.5. A ring median would be ~0.8.
        let mut low = Array3::<f32>::zeros((40, 40, 3));
        for y in 0..40 {
            for x in 0..40 {
                let v = if x < 20 { 0.2 } else { 0.8 };
                low[(y, x, 0)] = v;
                low[(y, x, 1)] = v;
                low[(y, x, 2)] = v;
            }
        }
        let mut hole = vec![false; 40 * 40];
        for y in 8..=32 {
            for x in 20..=22 {
                hole[y * 40 + x] = true;
            }
        }
        let s = structure_hv(&low, &hole, 21, 20, 40, 40).expect("hole has endpoints");
        assert!(
            (s.0 - 0.54).abs() < 0.03,
            "H+V should recover the step (weighted ~0.54), not a flat 0.8 ring (got {})",
            s.0
        );

        let mut img = low.clone();
        let mut mask = DustMask::new(40, 40);
        for y in 8..=32 {
            for x in 20..=22 {
                img[(y, x, 0)] = 1.0;
                img[(y, x, 1)] = 1.0;
                img[(y, x, 2)] = 1.0;
                mask.data[y * 40 + x] = 255;
            }
        }
        apply_dust_removal_with(
            &mut img,
            &mask,
            DustHealParams {
                detect: 1.0,
                feather: 0.0,
                grain: 0.0,
                grain_sigma: 0.6,
            },
        );
        let healed = img[(20, 21, 0)];
        assert!(
            healed < 0.72 && healed > 0.35,
            "heal should follow the step, not the 0.8 ring (got {healed})"
        );
    }

    #[test]
    fn telea_does_not_smear_speck_inward() {
        let mut img = Array3::<f32>::from_elem((32, 32, 3), 0.4);
        img[(16, 16, 0)] = 1.0;
        img[(16, 16, 1)] = 1.0;
        img[(16, 16, 2)] = 1.0;
        img[(16, 17, 0)] = 1.0;
        img[(17, 16, 0)] = 1.0;
        img[(17, 17, 0)] = 1.0;

        let mut mask = DustMask::new(32, 32);
        stamp_disc(&mut mask, 16.0, 16.0, 4.0, DustTool::Pen);
        apply_dust_removal_with(
            &mut img,
            &mask,
            DustHealParams {
                detect: 1.0,
                feather: 2.0,
                grain: 0.0,
                grain_sigma: 0.8,
            },
        );
        let v = img[(16, 16, 0)];
        assert!(
            (v - 0.4).abs() < 0.12,
            "punched Telea must reconstruct the surround, not the speck (got {v})"
        );
        assert!(v < 0.65, "must not smear the white speck inward (got {v})");
    }

    #[test]
    fn telea_ignores_hole_values_on_diagonal() {
        let mut img = Array3::<f32>::zeros((40, 40, 3));
        for y in 0..40 {
            for x in 0..40 {
                let v = if y < x { 0.2 } else { 0.8 };
                img[(y, x, 0)] = v;
                img[(y, x, 1)] = v;
                img[(y, x, 2)] = v;
            }
        }
        let mut hole = vec![false; 40 * 40];
        for y in 18..=22 {
            for x in 18..=22 {
                hole[y * 40 + x] = true;
                img[(y, x, 0)] = 1.0;
                img[(y, x, 1)] = 1.0;
                img[(y, x, 2)] = 1.0;
            }
        }
        let s = fill_structure_telea(&img, &hole, 40, 40);
        let v = s[(20, 18, 0)];
        assert!(
            v < 0.85,
            "must not read the punched 1.0 hole (got {v})"
        );
        assert!(
            (v - 0.2).abs() < 0.15 || (v - 0.8).abs() < 0.15,
            "Telea should stay on one side of the diagonal (got {v})"
        );
    }

    fn plane_spread(img: &Array3<f32>, x0: usize, y0: usize, x1: usize, y1: usize) -> f32 {
        let mut lo = 1.0f32;
        let mut hi = 0.0f32;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let v = img[(y, x, 0)];
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        hi - lo
    }

    #[test]
    fn telea_on_low_does_not_drag_grain() {
        let mut img = Array3::<f32>::from_elem((36, 36, 3), 0.4);
        for y in 0..36 {
            for x in 0..36 {
                let n = ((x * 17 + y * 31) % 11) as f32 * 0.03 - 0.15;
                img[(y, x, 0)] = (0.4 + n).clamp(0.0, 1.0);
                img[(y, x, 1)] = (0.4 + n * 0.8).clamp(0.0, 1.0);
                img[(y, x, 2)] = (0.4 + n * 0.6).clamp(0.0, 1.0);
            }
        }
        let surround = plane_spread(&img, 2, 2, 8, 8);
        img[(18, 18, 0)] = 1.0;
        img[(18, 18, 1)] = 1.0;
        img[(18, 18, 2)] = 1.0;
        img[(18, 19, 0)] = 1.0;
        img[(19, 18, 0)] = 1.0;

        let mut mask = DustMask::new(36, 36);
        stamp_disc(&mut mask, 18.0, 18.0, 4.0, DustTool::Pen);
        apply_dust_removal_with(
            &mut img,
            &mask,
            DustHealParams {
                detect: 1.0,
                feather: 2.0,
                grain: 0.0,
                grain_sigma: 2.0,
            },
        );
        let hole = plane_spread(&img, 17, 17, 19, 19);
        assert!(
            hole < surround * 0.55,
            "grain=0 should leave a smooth hole, not dragged edge grains (hole={hole}, surround={surround})"
        );
    }
}
