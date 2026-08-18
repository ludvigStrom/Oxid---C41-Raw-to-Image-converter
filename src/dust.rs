//! User-painted dust mask and nearby-patch dust healing on linear transmittance.

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
const MIN_SPECK_AREA: usize = 4;

/// Detection strength, rim fade, and grain for [`apply_dust_removal_with`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DustHealParams {
    /// Higher = pick up fainter specks inside the brush (0.5–2.5).
    pub detect: f32,
    /// Fade width in pixels *outside* the tight speck (0 = core only).
    pub feather: f32,
    /// Scale on the high-pass residual (0 = color only, 1 = match surround).
    pub grain: f32,
    /// Blur σ that splits low/high for grain (lower keeps finer texture).
    pub grain_sigma: f32,
}

impl Default for DustHealParams {
    fn default() -> Self {
        Self {
            detect: 1.0,
            feather: 6.0,
            grain: 1.0,
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

/// Replace dust inside the painted ROI: tighten to the speck, copy grain,
/// match surrounding color, fade only outside the tight core.
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

    let tight = refine_speck_mask(image, &roi, w, h, params.detect);
    if !tight.iter().any(|&v| v) {
        return;
    }

    let feather = params.feather.clamp(0.0, 16.0);
    let grain_amount = params.grain.clamp(0.0, 3.0);
    let grain_sigma = params.grain_sigma.clamp(0.6, 4.0);
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

    let low = blur_rgb(image, grain_sigma);
    for component in connected_components(&dilated, w, h) {
        heal_component(
            image,
            &low,
            &component,
            &dilated,
            &alpha,
            grain_amount,
            w,
            h,
        );
    }
}

fn refine_speck_mask(
    image: &Array3<f32>,
    roi: &[bool],
    w: usize,
    h: usize,
    detect: f32,
) -> Vec<bool> {
    let detect = detect.clamp(0.35, 2.5);
    let dog_thresh = (0.032 / detect).max(0.006);
    let mad_k = 3.0 / detect;
    let clip_lo = 0.025;
    let clip_hi = 0.975;

    let n = w * h;
    let mut lum = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let r = image[(y, x, 0)];
            let g = image[(y, x, 1)];
            let b = image[(y, x, 2)];
            lum[y * w + x] = r.max(g).max(b);
        }
    }
    let blur_s = blur_plane(&lum, w, h, 0.8);
    let blur_l = blur_plane(&lum, w, h, 2.4);

    let mut residuals = Vec::new();
    let mut residual_at = vec![0.0f32; n];
    for i in 0..n {
        if !roi[i] {
            continue;
        }
        let med = local_median(&lum, w, h, i % w, i / w, 3);
        let r = (lum[i] - med).abs();
        residual_at[i] = r;
        residuals.push(r);
    }
    let mad = {
        let mut m = residuals.clone();
        let med = median_f32(&mut m);
        let mut dev: Vec<f32> = residuals.iter().map(|v| (v - med).abs()).collect();
        median_f32(&mut dev).max(0.006)
    };

    let mut tight = vec![false; n];
    for i in 0..n {
        if !roi[i] {
            continue;
        }
        let dog = (blur_s[i] - blur_l[i]).abs();
        let clip = lum[i] <= clip_lo || lum[i] >= clip_hi;
        if dog >= dog_thresh || residual_at[i] >= mad_k * mad || clip {
            tight[i] = true;
        }
    }

    let mut kept = vec![false; n];
    for comp in connected_components(&tight, w, h) {
        if comp.len() >= MIN_SPECK_AREA {
            for i in comp {
                kept[i] = true;
            }
        }
    }
    if kept.iter().any(|&v| v) {
        return kept;
    }
    let fallback = erode(roi, w, h, 2);
    if fallback.iter().any(|&v| v) {
        fallback
    } else {
        erode(roi, w, h, 1)
    }
}

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

fn blur_rgb(image: &Array3<f32>, sigma: f32) -> Array3<f32> {
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
    let br = blur_plane(&planes[0], w, h, sigma);
    let bg = blur_plane(&planes[1], w, h, sigma);
    let bb = blur_plane(&planes[2], w, h, sigma);
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
    let mut x0 = w;
    let mut y0 = h;
    let mut x1 = 0usize;
    let mut y1 = 0usize;
    for &i in component {
        let x = i % w;
        let y = i / w;
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    let span = (x1 - x0 + 1).max(y1 - y0 + 1).max(6) as i32;

    let ring = component_ring(component, forbidden, w, h);
    let fallback = ring_median(low, &ring);
    let offset = best_patch_offset(image, &ring, forbidden, w, h, span);
    let residuals = ring_residuals(image, low, &ring);

    let mut fills = Vec::with_capacity(component.len());
    for (k, &i) in component.iter().enumerate() {
        let x = i % w;
        let y = i / w;
        let color = structure_hv(low, forbidden, x, y, w, h).unwrap_or(fallback);
        let x = x as i32;
        let y = y as i32;
        let grain = if let Some((dx, dy)) = offset {
            let sx = x + dx;
            let sy = y + dy;
            if sx >= 0 && sy >= 0 && sx < w as i32 && sy < h as i32 {
                let si = sy as usize * w + sx as usize;
                if !forbidden[si] {
                    Some((
                        image[(sy as usize, sx as usize, 0)] - low[(sy as usize, sx as usize, 0)],
                        image[(sy as usize, sx as usize, 1)] - low[(sy as usize, sx as usize, 1)],
                        image[(sy as usize, sx as usize, 2)] - low[(sy as usize, sx as usize, 2)],
                    ))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        let (gr, gg, gb) = grain.unwrap_or_else(|| {
            if residuals.is_empty() {
                (0.0, 0.0, 0.0)
            } else {
                residuals[k % residuals.len()]
            }
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

/// Weighted H+V lerp of `low` across the hole. Shorter span wins.
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

fn ring_median(image: &Array3<f32>, ring: &[usize]) -> (f32, f32, f32) {
    if ring.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let w = image.dim().1;
    let mut rs = Vec::with_capacity(ring.len());
    let mut gs = Vec::with_capacity(ring.len());
    let mut bs = Vec::with_capacity(ring.len());
    for &i in ring {
        let x = i % w;
        let y = i / w;
        rs.push(image[(y, x, 0)]);
        gs.push(image[(y, x, 1)]);
        bs.push(image[(y, x, 2)]);
    }
    (
        median_f32(&mut rs),
        median_f32(&mut gs),
        median_f32(&mut bs),
    )
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

fn best_patch_offset(
    image: &Array3<f32>,
    ring: &[usize],
    forbidden: &[bool],
    w: usize,
    h: usize,
    span: i32,
) -> Option<(i32, i32)> {
    if ring.is_empty() {
        return None;
    }
    let r = span.max(4);
    let candidates = [
        (r, 0),
        (-r, 0),
        (0, r),
        (0, -r),
        (r, r),
        (r, -r),
        (-r, r),
        (-r, -r),
        (r + r / 2, 0),
        (-(r + r / 2), 0),
        (0, r + r / 2),
        (0, -(r + r / 2)),
        (r / 2, r),
        (-r / 2, r),
        (r / 2, -r),
        (-r / 2, -r),
        (r, r / 2),
        (-r, r / 2),
        (r, -r / 2),
        (-r, -r / 2),
    ];
    let mut best: Option<(f32, (i32, i32))> = None;
    for (dx, dy) in candidates {
        if dx == 0 && dy == 0 {
            continue;
        }
        let mut ssd = 0.0f32;
        let mut n = 0i32;
        for &i in ring {
            let x = (i % w) as i32 + dx;
            let y = (i / w) as i32 + dy;
            if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
                continue;
            }
            let si = y as usize * w + x as usize;
            if forbidden[si] {
                continue;
            }
            let dx0 = image[(i / w, i % w, 0)] - image[(y as usize, x as usize, 0)];
            let dx1 = image[(i / w, i % w, 1)] - image[(y as usize, x as usize, 1)];
            let dx2 = image[(i / w, i % w, 2)] - image[(y as usize, x as usize, 2)];
            ssd += dx0 * dx0 + dx1 * dx1 + dx2 * dx2;
            n += 1;
        }
        if n < (ring.len() as i32 / 2).max(4) {
            continue;
        }
        let score = ssd / n as f32;
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
    fn tight_mask_leaves_far_roi_pixels() {
        let mut img = gradient_image(48, 48);
        let before = img[(24, 11, 0)];
        img[(24, 24, 0)] = 1.0;
        img[(24, 24, 1)] = 1.0;
        img[(24, 24, 2)] = 1.0;
        img[(24, 25, 0)] = 1.0;
        img[(25, 24, 0)] = 1.0;
        img[(25, 25, 0)] = 1.0;

        let mut mask = DustMask::new(48, 48);
        stamp_disc(&mut mask, 24.0, 24.0, 16.0, DustTool::Pen);
        apply_dust_removal(&mut img, &mask);

        assert!(
            (img[(24, 11, 0)] - before).abs() < 1e-5,
            "pixels far from the speck inside the brush must stay"
        );
        assert!(img[(24, 24, 0)] < 0.85, "the white speck must be healed");
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
}
