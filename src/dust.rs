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
const DILATE_RADIUS: i32 = 2;

/// Replace painted dust by copying nearby clean texture (spot-heal).
///
/// Telea-style diffusion smears the speck. This instead finds a nearby offset
/// whose border matches, then copies that patch in — grain stays, the spot goes.
pub fn apply_dust_removal(image: &mut Array3<f32>, mask: &DustMask) {
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
    heal_spots(image, mask);
}

fn heal_spots(image: &mut Array3<f32>, mask: &DustMask) {
    let (h, w, _) = image.dim();
    let n = w * h;
    let mut marked = vec![false; n];
    for (i, &c) in mask.data.iter().enumerate() {
        if c >= MASK_ON {
            marked[i] = true;
        }
    }
    if !marked.iter().any(|&v| v) {
        return;
    }
    let forbidden = dilate(&marked, w, h, DILATE_RADIUS);
    for component in connected_components(&forbidden, w, h) {
        heal_component(image, &component, &forbidden, w, h);
    }
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
    component: &[usize],
    forbidden: &[bool],
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
    let rw = (x1 - x0 + 1) as i32;
    let rh = (y1 - y0 + 1) as i32;
    let span = rw.max(rh).max(6);

    let ring = component_ring(component, forbidden, w, h);
    let median = ring_median(image, &ring);
    let offset = best_patch_offset(image, &ring, forbidden, w, h, span);

    let mut fills = Vec::with_capacity(component.len());
    for &i in component {
        let x = (i % w) as i32;
        let y = (i / w) as i32;
        let rgb = if let Some((dx, dy)) = offset {
            let sx = x + dx;
            let sy = y + dy;
            if sx >= 0 && sy >= 0 && sx < w as i32 && sy < h as i32 {
                let si = sy as usize * w + sx as usize;
                if !forbidden[si] {
                    Some((
                        image[(sy as usize, sx as usize, 0)],
                        image[(sy as usize, sx as usize, 1)],
                        image[(sy as usize, sx as usize, 2)],
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
        fills.push(rgb.unwrap_or(median));
    }
    for (&i, (r, g, b)) in component.iter().zip(fills) {
        let x = i % w;
        let y = i / w;
        image[(y, x, 0)] = r;
        image[(y, x, 1)] = g;
        image[(y, x, 2)] = b;
    }
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
    (median_f32(&mut rs), median_f32(&mut gs), median_f32(&mut bs))
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

        assert!((img[(12, 12, 0)] - expected[0]).abs() < 0.12);
        assert!((img[(12, 12, 1)] - expected[1]).abs() < 0.12);
        assert!((img[(12, 12, 2)] - expected[2]).abs() < 0.12);
        assert!(img[(12, 12, 0)] < 0.55, "spot should be replaced, not smeared");
    }

    #[test]
    fn empty_mask_is_noop() {
        let mut img = Array3::<f32>::from_elem((4, 4, 3), 0.25);
        apply_dust_removal(&mut img, &DustMask::new(4, 4));
        assert!((img[(0, 0, 0)] - 0.25).abs() < 1e-6);
    }
}
