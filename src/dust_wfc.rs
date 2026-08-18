//! Coherent exemplar copy for dust holes.
//!
//! Color-gated film near the stroke is the only legal source. Placement
//! prefers a neighbor’s source offset (rigid grain patch), then a short
//! overlap search. The GPU path runs the same kernel.

use ndarray::Array3;
use rayon::prelude::*;

use crate::dust::{connected_components, dilate, pixel_hash, rgb_at, structure_hv};

const RIM_R: i32 = 8;
const GRAIN_AMP: f32 = 0.035;
pub(crate) const MAX_PASSES: u32 = 48;
pub(crate) const MAX_CANDIDATES: usize = 4096;

const DIRS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

fn tile_n(tile: u8) -> usize {
    (tile as usize).clamp(2, 5)
}

pub(crate) fn search_radius(loosen: f32) -> i32 {
    (10.0 + 4.0 * loosen.clamp(1.0, 4.0)).round() as i32
}

fn rgb_ssd(a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
    let d0 = a.0 - b.0;
    let d1 = a.1 - b.1;
    let d2 = a.2 - b.2;
    (d0 * d0 + d1 * d1 + d2 * d2) / 3.0
}

fn hash_unit(h: u32) -> f32 {
    (h as f32) / (u32::MAX as f32)
}

fn synth_grain(x: usize, y: usize, amount: f32) -> (f32, f32, f32) {
    let h = pixel_hash(x, y);
    let amp = GRAIN_AMP * amount;
    (
        (hash_unit(h) * 2.0 - 1.0) * amp,
        (hash_unit(h.wrapping_mul(0x85EB_CA6B)) * 2.0 - 1.0) * amp,
        (hash_unit(h.wrapping_mul(0xC2B2_AE35)) * 2.0 - 1.0) * amp,
    )
}

fn median_f32(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.01;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// Sharp exemplar fill on `tight`, composite with `alpha`, then optional hash grain.
pub(crate) fn heal_wfc(
    image: &mut Array3<f32>,
    tight: &[bool],
    dilated: &[bool],
    alpha: &[f32],
    grain_amount: f32,
    tile: u8,
    match_loosen: f32,
    w: usize,
    h: usize,
) {
    let n_pix = w * h;
    let n = tile_n(tile);
    let loosen = match_loosen.clamp(1.0, 4.0);
    let mut fill = vec![None; n_pix];
    for component in connected_components(tight, w, h) {
        fill_component(image, tight, &component, &mut fill, n, loosen, w, h);
    }
    composite_and_grain(image, &fill, dilated, alpha, grain_amount, w, h);
}

pub(crate) fn composite_and_grain(
    image: &mut Array3<f32>,
    fill: &[Option<(f32, f32, f32)>],
    dilated: &[bool],
    alpha: &[f32],
    grain_amount: f32,
    w: usize,
    h: usize,
) {
    let n_pix = w * h;
    for i in 0..n_pix {
        let a = alpha[i];
        if a <= 1.0e-5 {
            continue;
        }
        let Some((r, g, b)) = fill[i] else {
            continue;
        };
        let x = i % w;
        let y = i / w;
        image[(y, x, 0)] = image[(y, x, 0)] * (1.0 - a) + r * a;
        image[(y, x, 1)] = image[(y, x, 1)] * (1.0 - a) + g * a;
        image[(y, x, 2)] = image[(y, x, 2)] * (1.0 - a) + b * a;
    }

    let grain_amount = grain_amount.clamp(0.0, 3.0);
    if grain_amount <= 1.0e-5 {
        return;
    }
    for i in 0..n_pix {
        if !dilated[i] {
            continue;
        }
        let a = alpha[i];
        if a <= 1.0e-5 {
            continue;
        }
        let x = i % w;
        let y = i / w;
        let (gr, gg, gb) = synth_grain(x, y, grain_amount);
        image[(y, x, 0)] = (image[(y, x, 0)] + gr * a).clamp(0.0, 1.0);
        image[(y, x, 1)] = (image[(y, x, 1)] + gg * a).clamp(0.0, 1.0);
        image[(y, x, 2)] = (image[(y, x, 2)] + gb * a).clamp(0.0, 1.0);
    }
}

fn fill_component(
    image: &Array3<f32>,
    tight: &[bool],
    component: &[usize],
    fill: &mut [Option<(f32, f32, f32)>],
    n: usize,
    loosen: f32,
    w: usize,
    h: usize,
) {
    if component.is_empty() {
        return;
    }
    let mut hole = vec![false; w * h];
    for &i in component {
        hole[i] = true;
    }
    let (candidates, color_gate, rim_mean) =
        build_candidates(image, tight, &hole, loosen, w, h);
    let (mut colors, srcs) = exemplar_fill(
        image,
        tight,
        component,
        &candidates,
        color_gate,
        rim_mean,
        n,
        w,
        h,
    );
    blend_offset_seams(
        image,
        tight,
        component,
        &mut colors,
        &srcs,
        color_gate,
        rim_mean,
        w,
        h,
    );
    for (&i, color) in component.iter().zip(colors) {
        fill[i] = Some(color);
    }
}

pub(crate) fn build_candidates(
    image: &Array3<f32>,
    tight: &[bool],
    hole: &[bool],
    loosen: f32,
    w: usize,
    h: usize,
) -> (Vec<(u16, u16)>, f32, (f32, f32, f32)) {
    let rim = dilate(hole, w, h, RIM_R);
    let search_r = search_radius(loosen);
    let search = dilate(hole, w, h, search_r);

    let mut rim_acc = (0.0f32, 0.0f32, 0.0f32);
    let mut rim_n = 0.0f32;
    let mut rim_colors = Vec::new();
    for i in 0..w * h {
        if !rim[i] || tight[i] {
            continue;
        }
        let c = rgb_at(image, i % w, i / w);
        rim_acc.0 += c.0;
        rim_acc.1 += c.1;
        rim_acc.2 += c.2;
        rim_n += 1.0;
        rim_colors.push(c);
    }
    let rim_mean = if rim_n > 0.0 {
        (rim_acc.0 / rim_n, rim_acc.1 / rim_n, rim_acc.2 / rim_n)
    } else {
        (0.4, 0.4, 0.4)
    };
    let color_scale = if rim_colors.is_empty() {
        0.02
    } else {
        median_f32(rim_colors.iter().map(|&c| rgb_ssd(c, rim_mean)).collect())
    };
    let color_gate = (color_scale * loosen).max(1.0e-4);

    let mut prev = hole.to_vec();
    let mut xy = Vec::new();
    for r in 1..=search_r {
        let ring = dilate(hole, w, h, r);
        for i in 0..w * h {
            if !ring[i] || prev[i] || tight[i] || !search[i] {
                continue;
            }
            let c = rgb_at(image, i % w, i / w);
            if rgb_ssd(c, rim_mean) > color_gate {
                continue;
            }
            xy.push(((i % w) as u16, (i / w) as u16));
            if xy.len() >= MAX_CANDIDATES {
                return (xy, color_gate, rim_mean);
            }
        }
        prev = ring;
    }
    (xy, color_gate, rim_mean)
}

fn color_legal(c: (f32, f32, f32), rim: (f32, f32, f32), gate: f32) -> bool {
    rgb_ssd(c, rim) <= gate
}

fn source_ok(
    image: &Array3<f32>,
    tight: &[bool],
    sx: i32,
    sy: i32,
    w: usize,
    h: usize,
    rim: (f32, f32, f32),
    gate: f32,
) -> Option<(u16, u16)> {
    if sx < 0 || sy < 0 || sx >= w as i32 || sy >= h as i32 {
        return None;
    }
    let sx = sx as usize;
    let sy = sy as usize;
    if tight[sy * w + sx] {
        return None;
    }
    let c = rgb_at(image, sx, sy);
    if !color_legal(c, rim, gate) {
        return None;
    }
    Some((sx as u16, sy as u16))
}

fn sample_known(
    x: usize,
    y: usize,
    n: usize,
    image: &Array3<f32>,
    placed: &[Option<(f32, f32, f32)>],
    tight: &[bool],
    w: usize,
    h: usize,
) -> Vec<(bool, (f32, f32, f32))> {
    let off = n as i32 / 2;
    let mut out = vec![(false, (0.0, 0.0, 0.0)); n * n];
    for ty in 0..n {
        for tx in 0..n {
            let px = x as i32 + tx as i32 - off;
            let py = y as i32 + ty as i32 - off;
            if px < 0 || py < 0 || px >= w as i32 || py >= h as i32 {
                continue;
            }
            let px = px as usize;
            let py = py as usize;
            let i = py * w + px;
            let k = ty * n + tx;
            if let Some(c) = placed.get(i).and_then(|c| *c) {
                out[k] = (true, c);
            } else if !tight[i] {
                out[k] = (true, rgb_at(image, px, py));
            }
        }
    }
    out
}

fn overlap_ssd(
    image: &Array3<f32>,
    tight: &[bool],
    known: &[(bool, (f32, f32, f32))],
    sx: usize,
    sy: usize,
    n: usize,
    w: usize,
    h: usize,
) -> f32 {
    let off = n as i32 / 2;
    let mut s = 0.0f32;
    let mut c = 0.0f32;
    for ty in 0..n {
        for tx in 0..n {
            let k = ty * n + tx;
            if !known[k].0 {
                continue;
            }
            let rx = sx as i32 + tx as i32 - off;
            let ry = sy as i32 + ty as i32 - off;
            if rx < 0 || ry < 0 || rx >= w as i32 || ry >= h as i32 {
                continue;
            }
            let rx = rx as usize;
            let ry = ry as usize;
            if tight[ry * w + rx] {
                continue;
            }
            s += rgb_ssd(rgb_at(image, rx, ry), known[k].1);
            c += 1.0;
        }
    }
    if c > 0.0 {
        s / c
    } else {
        f32::MAX
    }
}

fn walk_to_film(
    image: &Array3<f32>,
    tight: &[bool],
    mut sx: i32,
    mut sy: i32,
    ox: i32,
    oy: i32,
    w: usize,
    h: usize,
    rim: (f32, f32, f32),
    gate: f32,
) -> Option<(u16, u16)> {
    let step_x = ox.signum();
    let step_y = oy.signum();
    for _ in 0..48 {
        if let Some(src) = source_ok(image, tight, sx, sy, w, h, rim, gate) {
            return Some(src);
        }
        if step_x == 0 && step_y == 0 {
            break;
        }
        sx += step_x;
        sy += step_y;
    }
    None
}

fn try_propagate(
    image: &Array3<f32>,
    tight: &[bool],
    sources: &[Option<(u16, u16)>],
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    rim: (f32, f32, f32),
    gate: f32,
) -> Option<(u16, u16)> {
    for (dx, dy) in DIRS {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
            continue;
        }
        let Some((sx, sy)) = sources[ny as usize * w + nx as usize] else {
            continue;
        };
        let ox = sx as i32 - nx;
        let oy = sy as i32 - ny;
        let tx = x as i32 + ox;
        let ty = y as i32 + oy;
        if let Some(src) = walk_to_film(image, tight, tx, ty, ox, oy, w, h, rim, gate) {
            return Some(src);
        }
    }
    None
}

fn search_source(
    image: &Array3<f32>,
    tight: &[bool],
    candidates: &[(u16, u16)],
    known: &[(bool, (f32, f32, f32))],
    x: usize,
    y: usize,
    n: usize,
    w: usize,
    h: usize,
) -> Option<(u16, u16)> {
    if candidates.is_empty() || known.iter().all(|k| !k.0) {
        return None;
    }
    let mut best = f32::MAX;
    let mut pick = None;
    for &(sx, sy) in candidates {
        let s = overlap_ssd(image, tight, known, sx as usize, sy as usize, n, w, h);
        if s >= f32::MAX / 2.0 {
            continue;
        }
        let dist = (sx as i32 - x as i32)
            .unsigned_abs()
            .max((sy as i32 - y as i32).unsigned_abs()) as f32;
        let s = s - 0.002 * dist.min(12.0);
        if s < best {
            best = s;
            pick = Some((sx, sy));
        }
    }
    pick
}

fn hv_fallback(
    image: &Array3<f32>,
    tight: &[bool],
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) -> (f32, f32, f32) {
    structure_hv(image, tight, x, y, w, h).unwrap_or_else(|| {
        let mut acc = (0.0f32, 0.0f32, 0.0f32);
        let mut n = 0f32;
        for (dx, dy) in DIRS {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                continue;
            }
            let nx = nx as usize;
            let ny = ny as usize;
            if tight[ny * w + nx] {
                continue;
            }
            let c = rgb_at(image, nx, ny);
            acc.0 += c.0;
            acc.1 += c.1;
            acc.2 += c.2;
            n += 1.0;
        }
        if n > 0.0 {
            (acc.0 / n, acc.1 / n, acc.2 / n)
        } else {
            (0.4, 0.4, 0.4)
        }
    })
}

pub(crate) fn hole_pass_count(component: &[usize], w: usize, n: usize) -> u32 {
    if component.is_empty() {
        return 0;
    }
    let mut x0 = usize::MAX;
    let mut y0 = usize::MAX;
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
    let span = (x1 - x0).max(y1 - y0);
    let step = (n / 2).max(1);
    ((span / step) as u32 + 3).clamp(3, MAX_PASSES)
}

pub(crate) fn exemplar_fill(
    image: &Array3<f32>,
    tight: &[bool],
    component: &[usize],
    candidates: &[(u16, u16)],
    color_gate: f32,
    rim_mean: (f32, f32, f32),
    n: usize,
    w: usize,
    h: usize,
) -> (Vec<(f32, f32, f32)>, Vec<Option<(u16, u16)>>) {
    let cells = component.len();
    let mut colors = vec![(0.0, 0.0, 0.0); cells];
    let mut srcs = vec![None; cells];
    if candidates.is_empty() {
        for (k, &i) in component.iter().enumerate() {
            colors[k] = hv_fallback(image, tight, i % w, i / w, w, h);
        }
        return (colors, srcs);
    }

    let mut index_of = vec![usize::MAX; w * h];
    for (k, &i) in component.iter().enumerate() {
        index_of[i] = k;
    }
    let mut placed = vec![None; w * h];
    let mut sources = vec![None; w * h];
    let passes = hole_pass_count(component, w, n);
    for _ in 0..passes {
        let prev_placed = placed.clone();
        let prev_src = sources.clone();
        let updates: Vec<(usize, (u16, u16))> = component
            .par_iter()
            .filter_map(|&i| {
                if prev_src[i].is_some() {
                    return None;
                }
                let x = i % w;
                let y = i / w;
                if let Some(src) =
                    try_propagate(image, tight, &prev_src, x, y, w, h, rim_mean, color_gate)
                {
                    return Some((i, src));
                }
                let known = sample_known(x, y, n, image, &prev_placed, tight, w, h);
                let src = search_source(image, tight, candidates, &known, x, y, n, w, h)?;
                Some((i, src))
            })
            .collect();
        for (i, src) in updates {
            let c = rgb_at(image, src.0 as usize, src.1 as usize);
            placed[i] = Some(c);
            sources[i] = Some(src);
            let k = index_of[i];
            colors[k] = c;
            srcs[k] = Some(src);
        }
    }

    for (k, &i) in component.iter().enumerate() {
        if placed[i].is_none() {
            colors[k] = hv_fallback(image, tight, i % w, i / w, w, h);
        }
    }
    (colors, srcs)
}

fn source_offset(i: usize, src: Option<(u16, u16)>, w: usize) -> Option<(i32, i32)> {
    let (sx, sy) = src?;
    Some((sx as i32 - (i % w) as i32, sy as i32 - (i / w) as i32))
}

fn offset_chebyshev(a: (i32, i32), b: (i32, i32)) -> u32 {
    a.0.abs_diff(b.0).max(a.1.abs_diff(b.1))
}

/// Cross-dissolve where neighboring hole pixels copied from disagreeing offsets.
/// Skips when every source agrees (one rigid patch, no X).
pub(crate) fn blend_offset_seams(
    image: &Array3<f32>,
    tight: &[bool],
    component: &[usize],
    colors: &mut [(f32, f32, f32)],
    srcs: &[Option<(u16, u16)>],
    color_gate: f32,
    rim_mean: (f32, f32, f32),
    w: usize,
    h: usize,
) {
    if component.is_empty() || srcs.iter().all(Option::is_none) {
        return;
    }

    let mut index_of = vec![usize::MAX; w * h];
    let mut hole = vec![false; w * h];
    for (k, &i) in component.iter().enumerate() {
        index_of[i] = k;
        hole[i] = true;
    }

    let mut marked = vec![false; w * h];
    let mut any = false;
    for (k, &i) in component.iter().enumerate() {
        let Some(off_p) = source_offset(i, srcs[k], w) else {
            continue;
        };
        let x = (i % w) as i32;
        let y = (i / w) as i32;
        for (dx, dy) in DIRS {
            let nx = x + dx;
            let ny = y + dy;
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                continue;
            }
            let ni = ny as usize * w + nx as usize;
            if !hole[ni] {
                continue;
            }
            let nk = index_of[ni];
            let disagree = match source_offset(ni, srcs[nk], w) {
                None => true,
                Some(off_n) => offset_chebyshev(off_p, off_n) > 1,
            };
            if disagree {
                marked[i] = true;
                any = true;
                break;
            }
        }
    }
    if !any {
        return;
    }

    let dilated = dilate(&marked, w, h, 1);
    for (k, &i) in component.iter().enumerate() {
        if !dilated[i] {
            continue;
        }
        let x = (i % w) as i32;
        let y = (i / w) as i32;
        let mut acc = (0.0f32, 0.0f32, 0.0f32);
        let mut n = 0.0f32;
        let mut seen = [(i32::MAX, i32::MAX); 5];
        let mut seen_n = 0usize;
        let mut consider = |off: (i32, i32)| {
            if seen[..seen_n].contains(&off) {
                return;
            }
            seen[seen_n] = off;
            seen_n += 1;
            let Some(src) = source_ok(image, tight, x + off.0, y + off.1, w, h, rim_mean, color_gate)
            else {
                return;
            };
            let c = rgb_at(image, src.0 as usize, src.1 as usize);
            acc.0 += c.0;
            acc.1 += c.1;
            acc.2 += c.2;
            n += 1.0;
        };
        if let Some(off) = source_offset(i, srcs[k], w) {
            consider(off);
        }
        for (dx, dy) in DIRS {
            let nx = x + dx;
            let ny = y + dy;
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                continue;
            }
            let ni = ny as usize * w + nx as usize;
            if !hole[ni] {
                continue;
            }
            if let Some(off) = source_offset(ni, srcs[index_of[ni]], w) {
                consider(off);
            }
        }
        if n > 0.0 {
            colors[k] = (acc.0 / n, acc.1 / n, acc.2 / n);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn luma(c: (f32, f32, f32)) -> f32 {
        0.2126 * c.0 + 0.7152 * c.1 + 0.0722 * c.2
    }

    fn grainy_field(w: usize, h: usize) -> Array3<f32> {
        let mut img = Array3::<f32>::from_elem((h, w, 3), 0.42);
        for y in 0..h {
            for x in 0..w {
                let hsh = pixel_hash(x, y);
                let n = hash_unit(hsh) * 0.18 - 0.09;
                let n2 = hash_unit(hsh.wrapping_mul(0x9E37_79B9)) * 0.08 - 0.04;
                img[(y, x, 0)] = (0.44 + n + n2).clamp(0.0, 1.0);
                img[(y, x, 1)] = (0.41 + n * 0.85).clamp(0.0, 1.0);
                img[(y, x, 2)] = (0.39 + n * 0.65 - n2 * 0.3).clamp(0.0, 1.0);
            }
        }
        img
    }

    fn hole_square(w: usize, x0: usize, y0: usize, s: usize) -> (Vec<bool>, Vec<usize>) {
        let mut tight = vec![false; w * w];
        let mut component = Vec::new();
        for y in y0..y0 + s {
            for x in x0..x0 + s {
                tight[y * w + x] = true;
                component.push(y * w + x);
            }
        }
        (tight, component)
    }

    fn luma_stats(img: &Array3<f32>, mask: &[bool], w: usize, invert: bool) -> (f32, f32) {
        let mut lo = 1.0f32;
        let mut hi = 0.0f32;
        let mut sum = 0.0f32;
        let mut n = 0.0f32;
        for (i, &on) in mask.iter().enumerate() {
            if on == invert {
                continue;
            }
            let c = rgb_at(img, i % w, i / w);
            let y = luma(c);
            lo = lo.min(y);
            hi = hi.max(y);
            sum += y;
            n += 1.0;
        }
        (if n > 0.0 { sum / n } else { 0.0 }, hi - lo)
    }

    fn run_fill(
        img: &Array3<f32>,
        tight: &[bool],
        component: &[usize],
        loosen: f32,
        n: usize,
        w: usize,
        h: usize,
    ) -> (Vec<(f32, f32, f32)>, Vec<Option<(u16, u16)>>) {
        let mut hole = vec![false; w * h];
        for &i in component {
            hole[i] = true;
        }
        let (cands, gate, rim) = build_candidates(img, tight, &hole, loosen, w, h);
        let (mut colors, srcs) = exemplar_fill(img, tight, component, &cands, gate, rim, n, w, h);
        blend_offset_seams(img, tight, component, &mut colors, &srcs, gate, rim, w, h);
        (colors, srcs)
    }

    fn max_hole_luma_jump(
        colors: &[(f32, f32, f32)],
        component: &[usize],
        w: usize,
    ) -> f32 {
        let mut index_of = vec![usize::MAX; w * w];
        for (k, &i) in component.iter().enumerate() {
            index_of[i] = k;
        }
        let mut max_j = 0.0f32;
        for (k, &i) in component.iter().enumerate() {
            let y0 = luma(colors[k]);
            let x = (i % w) as i32;
            let y = (i / w) as i32;
            for (dx, dy) in DIRS {
                let nx = x + dx;
                let ny = y + dy;
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= w as i32 {
                    continue;
                }
                let ni = ny as usize * w + nx as usize;
                let nk = index_of[ni];
                if nk == usize::MAX {
                    continue;
                }
                max_j = max_j.max((y0 - luma(colors[nk])).abs());
            }
        }
        max_j
    }

    #[test]
    fn empty_library_falls_back_without_panic() {
        let img = Array3::<f32>::from_elem((8, 8, 3), 0.4);
        let tight = vec![false; 64];
        let component = vec![27usize];
        let (colors, _) = exemplar_fill(&img, &tight, &component, &[], 0.01, (0.4, 0.4, 0.4), 3, 8, 8);
        assert_eq!(colors.len(), 1);
    }

    #[test]
    fn grainy_hole_uses_library_not_only_hv() {
        let img = grainy_field(32, 32);
        let (tight, component) = hole_square(32, 14, 14, 4);
        let (wfc, srcs) = run_fill(&img, &tight, &component, 2.5, 3, 32, 32);
        assert!(srcs.iter().any(|s| s.is_some()), "must copy film sources");
        let mut diff = 0.0f32;
        for (k, &i) in component.iter().enumerate() {
            let hv = hv_fallback(&img, &tight, i % 32, i / 32, 32, 32);
            diff += (wfc[k].0 - hv.0).abs() + (wfc[k].1 - hv.1).abs() + (wfc[k].2 - hv.2).abs();
        }
        assert!(
            diff > 0.02,
            "exemplar must copy film, not only H+V (diff={diff})"
        );
    }

    #[test]
    fn grainy_hole_keeps_collar_variance_and_mean() {
        let img = grainy_field(48, 48);
        let (tight, component) = hole_square(48, 16, 16, 5);
        let (colors, _) = run_fill(&img, &tight, &component, 2.5, 3, 48, 48);

        let mut filled = img.clone();
        for (&i, c) in component.iter().zip(&colors) {
            filled[(i / 48, i % 48, 0)] = c.0;
            filled[(i / 48, i % 48, 1)] = c.1;
            filled[(i / 48, i % 48, 2)] = c.2;
        }

        let collar = {
            let outer = dilate(&tight, 48, 48, 8);
            outer
                .iter()
                .zip(tight.iter())
                .map(|(&o, &t)| o && !t)
                .collect::<Vec<_>>()
        };
        let (collar_mean, collar_spread) = luma_stats(&img, &collar, 48, false);
        let (hole_mean, hole_spread) = luma_stats(&filled, &tight, 48, false);

        assert!(
            hole_spread >= collar_spread * 0.40,
            "hole luma spread must stay near the collar (hole={hole_spread}, collar={collar_spread})"
        );
        assert!(
            (hole_mean - collar_mean).abs() < 0.08,
            "hole mean must stay near the collar (hole={hole_mean}, collar={collar_mean})"
        );
    }

    #[test]
    fn sky_hole_ignores_distant_yellow() {
        let mut img = Array3::<f32>::from_elem((48, 48, 3), 0.0);
        for y in 0..48 {
            for x in 0..48 {
                img[(y, x, 0)] = 0.16;
                img[(y, x, 1)] = 0.38;
                img[(y, x, 2)] = 0.78;
            }
        }
        for y in 2..10 {
            for x in 20..28 {
                img[(y, x, 0)] = 0.86;
                img[(y, x, 1)] = 0.74;
                img[(y, x, 2)] = 0.12;
            }
        }
        let (tight, component) = hole_square(48, 22, 22, 5);
        let (colors, _) = run_fill(&img, &tight, &component, 3.0, 3, 48, 48);
        let mut mr = 0.0;
        let mut mb = 0.0;
        for c in &colors {
            mr += c.0;
            mb += c.2;
        }
        mr /= colors.len() as f32;
        mb /= colors.len() as f32;
        assert!(
            mb > 0.55 && mr < 0.40,
            "blue-sky hole must stay blue, not yellow (r={mr}, b={mb})"
        );
    }

    #[test]
    fn hole_copies_coherent_offsets() {
        let img = grainy_field(48, 48);
        let (tight, component) = hole_square(48, 16, 16, 5);
        assert!(component.len() >= 16);
        let (colors, srcs) = run_fill(&img, &tight, &component, 2.5, 3, 48, 48);
        let mut offsets: HashMap<(i32, i32), usize> = HashMap::new();
        for (k, &i) in component.iter().enumerate() {
            let Some((sx, sy)) = srcs[k] else {
                continue;
            };
            let x = (i % 48) as i32;
            let y = (i / 48) as i32;
            *offsets
                .entry((sx as i32 - x, sy as i32 - y))
                .or_insert(0) += 1;
        }
        let max = offsets.values().copied().max().unwrap_or(0);
        assert!(
            max >= 2,
            "at least two hole pixels should share a source offset (max={max})"
        );
        assert!(
            offsets.len() < component.len(),
            "must not pick a unique source offset per pixel"
        );

        let mut lo = 1.0f32;
        let mut hi = 0.0f32;
        for c in &colors {
            let y = luma(*c);
            lo = lo.min(y);
            hi = hi.max(y);
        }
        assert!(
            hi - lo > 0.02,
            "coherent copy must still have grain (spread={})",
            hi - lo
        );
    }

    #[test]
    fn seam_blend_softens_offset_jump() {
        let mut img = Array3::<f32>::from_elem((24, 24, 3), 0.0);
        for y in 0..24 {
            for x in 0..24 {
                let v = if x < 12 { 0.18 } else { 0.82 };
                img[(y, x, 0)] = v;
                img[(y, x, 1)] = v;
                img[(y, x, 2)] = v;
            }
        }
        let (tight, component) = hole_square(24, 8, 8, 8);
        let mut colors = vec![(0.0, 0.0, 0.0); component.len()];
        let mut srcs = vec![None; component.len()];
        for (k, &i) in component.iter().enumerate() {
            let x = i % 24;
            let y = i / 24;
            let (sx, sy, v) = if x < 12 {
                (2u16, y as u16, 0.18)
            } else {
                (20u16, y as u16, 0.82)
            };
            srcs[k] = Some((sx, sy));
            colors[k] = (v, v, v);
        }
        let before = max_hole_luma_jump(&colors, &component, 24);
        assert!(
            before > 0.50,
            "fixture must have a hard mid-hole jump (got {before})"
        );
        blend_offset_seams(
            &img,
            &tight,
            &component,
            &mut colors,
            &srcs,
            1.0,
            (0.5, 0.5, 0.5),
            24,
            24,
        );
        let after = max_hole_luma_jump(&colors, &component, 24);
        assert!(
            after < before * 0.75,
            "seam blend must cut the offset jump (before={before}, after={after})"
        );
        let mut lo = 1.0f32;
        let mut hi = 0.0f32;
        for c in &colors {
            let y = luma(*c);
            lo = lo.min(y);
            hi = hi.max(y);
        }
        assert!(
            hi - lo > 0.20,
            "blend must not flatten the hole to one slab (spread={})",
            hi - lo
        );
    }
}
