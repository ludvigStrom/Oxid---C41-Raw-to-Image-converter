//! PatchMatch infill for dust holes.
//!
//! Local, color-gated Barnes NN-field: 7×7 SSD, propagate + random search,
//! coarse-to-fine pyramid, patch splat. Grain is WFC statistical (hole + feather).

use image::{imageops, Rgb, Rgb32FImage};
use ndarray::Array3;
use rayon::prelude::*;

use crate::dust::{connected_components, dilate, pixel_hash, rgb_at, structure_hv};
use crate::dust_grain::{apply_statistical_grain, component_bbox, nlm_bbox, MARGIN};

const PATCH: i32 = 3;
const ITERS: u32 = 5;
const RIM_R: i32 = 8;
const MAX_LEVELS: usize = 4;

fn rgb_ssd(a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
    let d0 = a.0 - b.0;
    let d1 = a.1 - b.1;
    let d2 = a.2 - b.2;
    (d0 * d0 + d1 * d1 + d2 * d2) / 3.0
}

#[cfg(test)]
fn luma(c: (f32, f32, f32)) -> f32 {
    0.2126 * c.0 + 0.7152 * c.1 + 0.0722 * c.2
}

fn median_f32(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.01;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

fn search_radius(loosen: f32) -> i32 {
    (10.0 + 8.0 * loosen.clamp(1.0, 4.0)).round() as i32
}

/// PatchMatch on `tight` using only nearby film, then alpha composite.
pub(crate) fn heal_patchmatch(
    image: &mut Array3<f32>,
    tight: &[bool],
    dilated: &[bool],
    alpha: &[f32],
    match_loosen: f32,
    grain_amount: f32,
    w: usize,
    h: usize,
) {
    let loosen = match_loosen.clamp(1.0, 4.0);
    let grain = grain_amount.clamp(0.0, 3.0);
    let n_pix = w * h;
    let mut fill = vec![None; n_pix];
    let components = connected_components(dilated, w, h);
    let results: Vec<(Vec<usize>, Vec<(f32, f32, f32)>)> = components
        .par_iter()
        .map(|component| {
            let colors = fill_component(image, tight, component, loosen, w, h);
            (component.clone(), colors)
        })
        .collect();
    for (component, colors) in results {
        for (&i, c) in component.iter().zip(colors) {
            fill[i] = Some(c);
        }
    }
    if grain > 1.0e-5 {
        grain_fill(image, tight, dilated, &components, &mut fill, grain, loosen, w, h);
    }
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
}

fn fill_component(
    image: &Array3<f32>,
    tight: &[bool],
    component: &[usize],
    loosen: f32,
    w: usize,
    h: usize,
) -> Vec<(f32, f32, f32)> {
    let mut colors = vec![(0.4, 0.4, 0.4); component.len()];
    if component.is_empty() {
        return colors;
    }
    let mut hole = vec![false; w * h];
    let mut tight_in = Vec::new();
    for &i in component {
        if tight[i] {
            hole[i] = true;
            tight_in.push(i);
        }
    }
    let (rim_mean, _gate, sources) = collect_sources(image, tight, &hole, loosen, w, h);
    let mut color_at = vec![None; w * h];
    if tight_in.is_empty() {
        return colors;
    }
    if sources.is_empty() {
        for &i in &tight_in {
            color_at[i] = Some(
                structure_hv(image, &hole, i % w, i / w, w, h).unwrap_or(rim_mean),
            );
        }
    } else {
        let (x0, y0, x1, y1) = bbox_of(&sources, &tight_in, w);
        let bw = x1 - x0 + 1;
        let bh = y1 - y0 + 1;
        let mut rgb = vec![0.0f32; bw * bh * 3];
        let mut hole_c = vec![false; bw * bh];
        let mut src_c = vec![false; bw * bh];
        for y in 0..bh {
            for x in 0..bw {
                let gx = x0 + x;
                let gy = y0 + y;
                let i = y * bw + x;
                let c = rgb_at(image, gx, gy);
                rgb[i * 3] = c.0;
                rgb[i * 3 + 1] = c.1;
                rgb[i * 3 + 2] = c.2;
                hole_c[i] = hole[gy * w + gx];
            }
        }
        for &si in &sources {
            let gx = si % w;
            let gy = si / w;
            if gx >= x0 && gx <= x1 && gy >= y0 && gy <= y1 {
                src_c[(gy - y0) * bw + (gx - x0)] = true;
            }
        }

        let mut levels = build_pyramid(rgb, hole_c, src_c, bw, bh, search_radius(loosen));
        let last = levels.len() - 1;
        init_nn(&mut levels[last]);
        for _ in 0..ITERS {
            patchmatch_iters(&mut levels[last]);
        }
        for li in (0..last).rev() {
            let (fine, rest) = levels.split_at_mut(li + 1);
            upsample_nn(&rest[0], &mut fine[li]);
            for _ in 0..ITERS {
                patchmatch_iters(&mut fine[li]);
            }
        }

        let fine = &levels[0];
        splat_patches(fine, &tight_in, component, x0, y0, w, h, &mut color_at);
        for &i in &tight_in {
            if color_at[i].is_some() {
                continue;
            }
            let x = (i % w) - x0;
            let y = (i / w) - y0;
            color_at[i] = Some(vote(fine, x, y).unwrap_or_else(|| {
                structure_hv(image, &hole, i % w, i / w, w, h).unwrap_or(rim_mean)
            }));
        }
    }
    for (k, &i) in component.iter().enumerate() {
        colors[k] = color_at[i].unwrap_or(rim_mean);
    }
    colors
}

/// WFC statistical grain on the painted hole and feather ring.
fn grain_fill(
    image: &Array3<f32>,
    tight: &[bool],
    dilated: &[bool],
    components: &[Vec<usize>],
    fill: &mut [Option<(f32, f32, f32)>],
    grain: f32,
    loosen: f32,
    w: usize,
    h: usize,
) {
    let mut den = image.clone();
    for component in components {
        if component.is_empty() {
            continue;
        }
        let mut hole = vec![false; w * h];
        for &i in component {
            if tight[i] {
                hole[i] = true;
            }
        }
        let (rim_mean, gate, _) = collect_sources(image, tight, &hole, loosen, w, h);
        let (x0, y0, x1, y1) = component_bbox(component, w, h, MARGIN);
        nlm_bbox(&mut den, image, tight, x0, y0, x1, y1, w, h);
        apply_statistical_grain(
            image, &den, dilated, component, fill, grain, rim_mean, gate, loosen, w, h,
        );
    }
}

/// Copy each matched source patch onto the dilated component (hole + overlap).
fn splat_patches(
    fine: &Level,
    tight_in: &[usize],
    component: &[usize],
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    color_at: &mut [Option<(f32, f32, f32)>],
) {
    let n = w * h;
    let mut in_comp = vec![false; n];
    for &i in component {
        in_comp[i] = true;
    }
    let mut acc = vec![(0.0f32, 0.0f32, 0.0f32); n];
    let mut wt = vec![0.0f32; n];
    let r = RIM_R;
    let r2 = r * r;
    for &i in tight_in {
        let x = (i % w).saturating_sub(x0);
        let y = (i / w).saturating_sub(y0);
        if x >= fine.w || y >= fine.h {
            continue;
        }
        let Some((sx, sy)) = fine.nn[y * fine.w + x] else {
            continue;
        };
        let gx = (i % w) as i32;
        let gy = (i / w) as i32;
        let sx = sx as i32;
        let sy = sy as i32;
        for dy in -r..=r {
            for dx in -r..=r {
                let d2 = dx * dx + dy * dy;
                if d2 > r2 {
                    continue;
                }
                let dx_dest = gx + dx;
                let dy_dest = gy + dy;
                if dx_dest < 0 || dy_dest < 0 || dx_dest >= w as i32 || dy_dest >= h as i32 {
                    continue;
                }
                let di = dy_dest as usize * w + dx_dest as usize;
                if !in_comp[di] {
                    continue;
                }
                let qx = sx + dx;
                let qy = sy + dy;
                if qx < 0 || qy < 0 || qx >= fine.w as i32 || qy >= fine.h as i32 {
                    continue;
                }
                let qi = qy as usize * fine.w + qx as usize;
                if fine.hole[qi] {
                    continue;
                }
                let c = rgb_at_level(fine, qx as usize, qy as usize);
                let dist = (d2 as f32).sqrt();
                let wgt = (r as f32 + 1.0 - dist).max(0.0);
                acc[di].0 += c.0 * wgt;
                acc[di].1 += c.1 * wgt;
                acc[di].2 += c.2 * wgt;
                wt[di] += wgt;
            }
        }
    }
    for &i in component {
        if wt[i] > 1.0e-8 {
            color_at[i] = Some((acc[i].0 / wt[i], acc[i].1 / wt[i], acc[i].2 / wt[i]));
        }
    }
}

fn collect_sources(
    image: &Array3<f32>,
    tight: &[bool],
    hole: &[bool],
    loosen: f32,
    w: usize,
    h: usize,
) -> ((f32, f32, f32), f32, Vec<usize>) {
    let rim = dilate(hole, w, h, RIM_R);
    let search = dilate(hole, w, h, search_radius(loosen));
    let mut acc = (0.0f32, 0.0f32, 0.0f32);
    let mut n = 0.0f32;
    let mut colors = Vec::new();
    for i in 0..w * h {
        if !rim[i] || tight[i] || hole[i] {
            continue;
        }
        let c = rgb_at(image, i % w, i / w);
        acc.0 += c.0;
        acc.1 += c.1;
        acc.2 += c.2;
        n += 1.0;
        colors.push(c);
    }
    let rim_mean = if n > 0.0 {
        (acc.0 / n, acc.1 / n, acc.2 / n)
    } else {
        (0.4, 0.4, 0.4)
    };
    let scale = if colors.is_empty() {
        0.02
    } else {
        median_f32(colors.iter().map(|&c| rgb_ssd(c, rim_mean)).collect())
    };
    let gate = scale.max(1.0e-4) * loosen;
    let mut sources = Vec::new();
    for i in 0..w * h {
        if !search[i] || tight[i] || hole[i] {
            continue;
        }
        let c = rgb_at(image, i % w, i / w);
        if rgb_ssd(c, rim_mean) <= gate {
            sources.push(i);
        }
    }
    (rim_mean, gate, sources)
}

fn bbox_of(sources: &[usize], component: &[usize], w: usize) -> (usize, usize, usize, usize) {
    let mut x0 = usize::MAX;
    let mut y0 = usize::MAX;
    let mut x1 = 0usize;
    let mut y1 = 0usize;
    for &i in component.iter().chain(sources.iter()) {
        let x = i % w;
        let y = i / w;
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    (x0, y0, x1, y1)
}

struct Level {
    w: usize,
    h: usize,
    rgb: Vec<f32>,
    hole: Vec<bool>,
    source: Vec<bool>,
    nn: Vec<Option<(u16, u16)>>,
    src_rad: i32,
}

fn source_near(x: usize, y: usize, sx: usize, sy: usize, rad: i32) -> bool {
    let dx = sx as i32 - x as i32;
    let dy = sy as i32 - y as i32;
    dx * dx + dy * dy <= rad * rad
}

fn build_pyramid(
    rgb: Vec<f32>,
    hole: Vec<bool>,
    source: Vec<bool>,
    w: usize,
    h: usize,
    src_rad: i32,
) -> Vec<Level> {
    let mut rad = src_rad.max(2);
    let mut levels = vec![Level {
        w,
        h,
        rgb,
        hole,
        source,
        nn: vec![None; w * h],
        src_rad: rad,
    }];
    while levels.len() < MAX_LEVELS {
        let prev = levels.last().unwrap();
        if prev.w < 10 || prev.h < 10 {
            break;
        }
        let hole_n = prev.hole.iter().filter(|&&v| v).count();
        if hole_n <= 4 {
            break;
        }
        let (rgb, nw, nh) = downsample_rgb(&prev.rgb, prev.w, prev.h);
        let hole = downsample_or(&prev.hole, prev.w, prev.h, nw, nh);
        let mut source = downsample_or(&prev.source, prev.w, prev.h, nw, nh);
        for i in 0..nw * nh {
            if hole[i] {
                source[i] = false;
            }
        }
        if !source.iter().any(|&s| s) {
            break;
        }
        rad = (rad / 2).max(2);
        let n = nw * nh;
        levels.push(Level {
            w: nw,
            h: nh,
            rgb,
            hole,
            source,
            nn: vec![None; n],
            src_rad: rad,
        });
    }
    levels
}

fn downsample_rgb(rgb: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let mut im = Rgb32FImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            im.put_pixel(x as u32, y as u32, Rgb([rgb[i], rgb[i + 1], rgb[i + 2]]));
        }
    }
    let nw = (w / 2).max(1);
    let nh = (h / 2).max(1);
    let small = imageops::resize(&im, nw as u32, nh as u32, imageops::FilterType::Triangle);
    let mut out = vec![0.0f32; nw * nh * 3];
    for y in 0..nh {
        for x in 0..nw {
            let p = small.get_pixel(x as u32, y as u32).0;
            let i = (y * nw + x) * 3;
            out[i] = p[0];
            out[i + 1] = p[1];
            out[i + 2] = p[2];
        }
    }
    (out, nw, nh)
}

fn downsample_or(on: &[bool], w: usize, h: usize, nw: usize, nh: usize) -> Vec<bool> {
    let mut out = vec![false; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let x0 = x * w / nw;
            let y0 = y * h / nh;
            let x1 = ((x + 1) * w / nw).max(x0 + 1).min(w);
            let y1 = ((y + 1) * h / nh).max(y0 + 1).min(h);
            let mut any = false;
            'c: for yy in y0..y1 {
                for xx in x0..x1 {
                    if on[yy * w + xx] {
                        any = true;
                        break 'c;
                    }
                }
            }
            out[y * nw + x] = any;
        }
    }
    out
}

fn source_list(level: &Level) -> Vec<(u16, u16)> {
    let mut s = Vec::new();
    for y in 0..level.h {
        for x in 0..level.w {
            if level.source[y * level.w + x] {
                s.push((x as u16, y as u16));
            }
        }
    }
    s
}

fn init_nn(level: &mut Level) {
    let srcs = source_list(level);
    if srcs.is_empty() {
        return;
    }
    let rad = level.src_rad;
    for y in 0..level.h {
        for x in 0..level.w {
            let i = y * level.w + x;
            if !level.hole[i] {
                continue;
            }
            let local: Vec<(u16, u16)> = srcs
                .iter()
                .copied()
                .filter(|&(sx, sy)| source_near(x, y, sx as usize, sy as usize, rad))
                .collect();
            let pick = if local.is_empty() { &srcs } else { &local };
            let hsh = pixel_hash(x, y) as usize;
            level.nn[i] = Some(pick[hsh % pick.len()]);
        }
    }
}

fn upsample_nn(coarse: &Level, fine: &mut Level) {
    let srcs = source_list(fine);
    for y in 0..fine.h {
        for x in 0..fine.w {
            let i = y * fine.w + x;
            if !fine.hole[i] {
                continue;
            }
            let cx = (x / 2).min(coarse.w.saturating_sub(1));
            let cy = (y / 2).min(coarse.h.saturating_sub(1));
            if let Some((sx, sy)) = coarse.nn[cy * coarse.w + cx] {
                let fx = ((sx as usize) * 2 + x % 2).min(fine.w.saturating_sub(1));
                let fy = ((sy as usize) * 2 + y % 2).min(fine.h.saturating_sub(1));
                if fine.source[fy * fine.w + fx]
                    && source_near(x, y, fx, fy, fine.src_rad)
                {
                    fine.nn[i] = Some((fx as u16, fy as u16));
                    continue;
                }
                if let Some(near) = nearest_source(fine, fx, fy) {
                    if source_near(x, y, near.0 as usize, near.1 as usize, fine.src_rad) {
                        fine.nn[i] = Some(near);
                        continue;
                    }
                }
            }
            let local: Vec<(u16, u16)> = srcs
                .iter()
                .copied()
                .filter(|&(sx, sy)| source_near(x, y, sx as usize, sy as usize, fine.src_rad))
                .collect();
            let pick = if local.is_empty() { &srcs } else { &local };
            if !pick.is_empty() {
                fine.nn[i] = Some(pick[pixel_hash(x, y) as usize % pick.len()]);
            }
        }
    }
}

fn nearest_source(level: &Level, x: usize, y: usize) -> Option<(u16, u16)> {
    let mut best = (i32::MAX, (0u16, 0u16));
    let mut any = false;
    for sy in 0..level.h {
        for sx in 0..level.w {
            if !level.source[sy * level.w + sx] {
                continue;
            }
            let d = (sx as i32 - x as i32).pow(2) + (sy as i32 - y as i32).pow(2);
            if d < best.0 {
                best = (d, (sx as u16, sy as u16));
                any = true;
            }
        }
    }
    if any {
        Some(best.1)
    } else {
        None
    }
}

fn rgb_at_level(level: &Level, x: usize, y: usize) -> (f32, f32, f32) {
    let i = (y * level.w + x) * 3;
    (level.rgb[i], level.rgb[i + 1], level.rgb[i + 2])
}

fn patch_ssd(level: &Level, x: usize, y: usize, sx: usize, sy: usize, best: f32) -> f32 {
    let mut s = 0.0f32;
    let mut n = 0.0f32;
    for dy in -PATCH..=PATCH {
        for dx in -PATCH..=PATCH {
            let px = x as i32 + dx;
            let py = y as i32 + dy;
            let qx = sx as i32 + dx;
            let qy = sy as i32 + dy;
            if px < 0 || py < 0 || qx < 0 || qy < 0 {
                continue;
            }
            if px >= level.w as i32
                || py >= level.h as i32
                || qx >= level.w as i32
                || qy >= level.h as i32
            {
                continue;
            }
            let pi = py as usize * level.w + px as usize;
            let qi = qy as usize * level.w + qx as usize;
            if level.hole[pi] || level.hole[qi] {
                continue;
            }
            s += rgb_ssd(rgb_at_level(level, px as usize, py as usize), rgb_at_level(level, qx as usize, qy as usize));
            n += 1.0;
            if n > 0.0 && s / n >= best {
                return s / n;
            }
        }
    }
    if n < 4.0 {
        f32::MAX
    } else {
        s / n
    }
}

fn try_assign(level: &mut Level, x: usize, y: usize, sx: i32, sy: i32, best: &mut f32) {
    if sx < 0 || sy < 0 || sx >= level.w as i32 || sy >= level.h as i32 {
        return;
    }
    let sx = sx as usize;
    let sy = sy as usize;
    if !level.source[sy * level.w + sx] {
        return;
    }
    if !source_near(x, y, sx, sy, level.src_rad) {
        return;
    }
    let cost = patch_ssd(level, x, y, sx, sy, *best);
    if cost < *best {
        *best = cost;
        level.nn[y * level.w + x] = Some((sx as u16, sy as u16));
    }
}

fn current_best(level: &Level, x: usize, y: usize) -> f32 {
    match level.nn[y * level.w + x] {
        Some((sx, sy)) => patch_ssd(level, x, y, sx as usize, sy as usize, f32::MAX),
        None => f32::MAX,
    }
}

fn patchmatch_iters(level: &mut Level) {
    propagate(level, true);
    random_search(level);
    propagate(level, false);
    random_search(level);
}

fn propagate(level: &mut Level, forward: bool) {
    let ys: Vec<usize> = if forward {
        (0..level.h).collect()
    } else {
        (0..level.h).rev().collect()
    };
    let xs: Vec<usize> = if forward {
        (0..level.w).collect()
    } else {
        (0..level.w).rev().collect()
    };
    let nd = if forward {
        [(-1i32, 0i32), (0, -1)]
    } else {
        [(1, 0), (0, 1)]
    };
    for y in ys {
        for x in xs.iter().copied() {
            if !level.hole[y * level.w + x] {
                continue;
            }
            let mut best = current_best(level, x, y);
            for (dx, dy) in nd {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= level.w as i32 || ny >= level.h as i32 {
                    continue;
                }
                let Some((sx, sy)) = level.nn[ny as usize * level.w + nx as usize] else {
                    continue;
                };
                try_assign(
                    level,
                    x,
                    y,
                    sx as i32 - dx,
                    sy as i32 - dy,
                    &mut best,
                );
            }
        }
    }
}

fn random_search(level: &mut Level) {
    let span = level.w.max(level.h) as i32;
    for y in 0..level.h {
        for x in 0..level.w {
            if !level.hole[y * level.w + x] {
                continue;
            }
            let Some((sx0, sy0)) = level.nn[y * level.w + x] else {
                continue;
            };
            let mut best = current_best(level, x, y);
            let mut rad = span;
            let mut k = 1u32;
            while rad >= 1 {
                let hx = pixel_hash(x.wrapping_add(k as usize), y);
                let hy = pixel_hash(x, y.wrapping_add(k as usize * 3));
                let range = (2 * rad + 1) as u32;
                let jx = (hx % range) as i32 - rad;
                let jy = (hy % range) as i32 - rad;
                try_assign(
                    level,
                    x,
                    y,
                    sx0 as i32 + jx,
                    sy0 as i32 + jy,
                    &mut best,
                );
                rad = (rad / 2).max(0);
                if rad == 0 {
                    break;
                }
                k += 1;
            }
        }
    }
}

fn vote(level: &Level, x: usize, y: usize) -> Option<(f32, f32, f32)> {
    let Some((sx, sy)) = level.nn[y * level.w + x] else {
        return None;
    };
    if !level.source[sy as usize * level.w + sx as usize] {
        return None;
    }
    let c0 = rgb_at_level(level, sx as usize, sy as usize);
    let mut acc = (c0.0 * 4.0, c0.1 * 4.0, c0.2 * 4.0);
    let mut wt = 4.0f32;
    for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        if nx < 0 || ny < 0 || nx >= level.w as i32 || ny >= level.h as i32 {
            continue;
        }
        let ni = ny as usize * level.w + nx as usize;
        if !level.hole[ni] {
            let c = rgb_at_level(level, nx as usize, ny as usize);
            acc.0 += c.0 * 0.45;
            acc.1 += c.1 * 0.45;
            acc.2 += c.2 * 0.45;
            wt += 0.45;
            continue;
        }
        let Some((msx, msy)) = level.nn[ni] else {
            continue;
        };
        let tx = msx as i32 - dx;
        let ty = msy as i32 - dy;
        if tx < 0 || ty < 0 || tx >= level.w as i32 || ty >= level.h as i32 {
            continue;
        }
        let ti = ty as usize * level.w + tx as usize;
        if !level.source[ti] {
            continue;
        }
        let c = rgb_at_level(level, tx as usize, ty as usize);
        acc.0 += c.0 * 0.45;
        acc.1 += c.1 * 0.45;
        acc.2 += c.2 * 0.45;
        wt += 0.45;
    }
    if wt < 1.0e-6 {
        None
    } else {
        Some((acc.0 / wt, acc.1 / wt, acc.2 / wt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dust::pixel_hash;

    fn hash_unit(h: u32) -> f32 {
        (h as f32) / (u32::MAX as f32)
    }

    fn grainy_field(w: usize, h: usize) -> Array3<f32> {
        let mut img = Array3::<f32>::from_elem((h, w, 3), 0.42);
        for y in 0..h {
            for x in 0..w {
                let hsh = pixel_hash(x, y);
                let n = hash_unit(hsh) * 0.18 - 0.09;
                img[(y, x, 0)] = (0.44 + n).clamp(0.0, 1.0);
                img[(y, x, 1)] = (0.41 + n * 0.85).clamp(0.0, 1.0);
                img[(y, x, 2)] = (0.39 + n * 0.65).clamp(0.0, 1.0);
            }
        }
        img
    }

    fn hole_square(w: usize, x0: usize, y0: usize, s: usize) -> Vec<bool> {
        let mut tight = vec![false; w * w];
        for y in y0..y0 + s {
            for x in x0..x0 + s {
                tight[y * w + x] = true;
            }
        }
        tight
    }

    fn run_fill(img: &Array3<f32>, tight: &[bool], loosen: f32, w: usize, h: usize) -> Array3<f32> {
        let mut out = img.clone();
        let alpha: Vec<f32> = tight.iter().map(|&t| if t { 1.0 } else { 0.0 }).collect();
        heal_patchmatch(&mut out, tight, tight, &alpha, loosen, 0.0, w, h);
        out
    }

    fn luma_stats(img: &Array3<f32>, mask: &[bool], w: usize) -> (f32, f32) {
        let mut lo = 1.0f32;
        let mut hi = 0.0f32;
        let mut sum = 0.0f32;
        let mut n = 0.0f32;
        for (i, &on) in mask.iter().enumerate() {
            if !on {
                continue;
            }
            let y = luma(rgb_at(img, i % w, i / w));
            lo = lo.min(y);
            hi = hi.max(y);
            sum += y;
            n += 1.0;
        }
        (if n > 0.0 { sum / n } else { 0.0 }, hi - lo)
    }

    #[test]
    fn empty_hole_does_not_panic() {
        let mut img = Array3::<f32>::from_elem((8, 8, 3), 0.4);
        let tight = vec![false; 64];
        let alpha = vec![0.0f32; 64];
        heal_patchmatch(&mut img, &tight, &tight, &alpha, 2.0, 0.0, 8, 8);
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
        let tight = hole_square(48, 22, 22, 5);
        let filled = run_fill(&img, &tight, 2.0, 48, 48);
        let mut mr = 0.0;
        let mut mb = 0.0;
        let mut n = 0.0;
        for i in 0..48 * 48 {
            if !tight[i] {
                continue;
            }
            mr += filled[(i / 48, i % 48, 0)];
            mb += filled[(i / 48, i % 48, 2)];
            n += 1.0;
        }
        mr /= n;
        mb /= n;
        assert!(
            mb > 0.55 && mr < 0.40,
            "blue-sky hole must stay blue, not yellow (r={mr}, b={mb})"
        );
    }

    #[test]
    fn grainy_hole_keeps_collar_variance() {
        let img = grainy_field(48, 48);
        let tight = hole_square(48, 16, 16, 5);
        let filled = run_fill(&img, &tight, 2.0, 48, 48);
        let collar = {
            let outer = dilate(&tight, 48, 48, 8);
            outer
                .iter()
                .zip(tight.iter())
                .map(|(&o, &t)| o && !t)
                .collect::<Vec<_>>()
        };
        let (collar_mean, collar_spread) = luma_stats(&img, &collar, 48);
        let (hole_mean, hole_spread) = luma_stats(&filled, &tight, 48);
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
    fn feather_ring_blends_and_leaves_outside() {
        let mut img = Array3::<f32>::from_elem((48, 48, 3), 0.25);
        let tight = hole_square(48, 22, 22, 5);
        let dilated = dilate(&tight, 48, 48, 3);
        let mut ring_i = None;
        let mut outside_i = None;
        for i in 0..48 * 48 {
            if dilated[i] && !tight[i] && ring_i.is_none() {
                ring_i = Some(i);
            }
            if !dilated[i] && outside_i.is_none() {
                outside_i = Some(i);
            }
        }
        let ring_i = ring_i.expect("feather ring");
        let outside_i = outside_i.expect("outside dilated");
        img[(ring_i / 48, ring_i % 48, 0)] = 0.95;
        img[(ring_i / 48, ring_i % 48, 1)] = 0.95;
        img[(ring_i / 48, ring_i % 48, 2)] = 0.95;
        img[(outside_i / 48, outside_i % 48, 0)] = 0.95;
        img[(outside_i / 48, outside_i % 48, 1)] = 0.95;
        img[(outside_i / 48, outside_i % 48, 2)] = 0.95;

        let mut alpha = vec![0.0f32; 48 * 48];
        for i in 0..48 * 48 {
            if tight[i] {
                alpha[i] = 1.0;
            } else if dilated[i] {
                alpha[i] = 0.5;
            }
        }
        let mut out = img.clone();
        heal_patchmatch(&mut out, &tight, &dilated, &alpha, 2.0, 0.0, 48, 48);

        let ring = out[(ring_i / 48, ring_i % 48, 0)];
        assert!(
            ring < 0.85,
            "feather ring must blend toward the fill (got {ring})"
        );
        assert!(
            (out[(outside_i / 48, outside_i % 48, 0)] - 0.95).abs() < 1e-5,
            "pixels outside dilated must stay"
        );
        let core = out[(24, 24, 0)];
        assert!(
            core < 0.50,
            "tight hole must still be replaced (got {core})"
        );
    }

    #[test]
    fn contrast_edge_ring_blends() {
        let mut img = Array3::<f32>::from_elem((48, 48, 3), 0.15);
        for y in 0..48 {
            for x in 24..48 {
                img[(y, x, 0)] = 0.85;
                img[(y, x, 1)] = 0.85;
                img[(y, x, 2)] = 0.85;
            }
        }
        let tight = hole_square(48, 21, 22, 6);
        let dilated = dilate(&tight, 48, 48, 3);
        let mut ring_i = None;
        let mut outside_i = None;
        for y in 0..48 {
            for x in 0..48 {
                let i = y * 48 + x;
                if dilated[i] && !tight[i] && x >= 24 && ring_i.is_none() {
                    ring_i = Some(i);
                }
                if !dilated[i] && x >= 30 && outside_i.is_none() {
                    outside_i = Some(i);
                }
            }
        }
        let ring_i = ring_i.expect("light-side feather ring");
        let outside_i = outside_i.expect("outside dilated");
        let orig_out = img[(outside_i / 48, outside_i % 48, 0)];
        img[(ring_i / 48, ring_i % 48, 0)] = 0.0;
        img[(ring_i / 48, ring_i % 48, 1)] = 0.0;
        img[(ring_i / 48, ring_i % 48, 2)] = 0.0;

        let mut alpha = vec![0.0f32; 48 * 48];
        for i in 0..48 * 48 {
            if tight[i] {
                alpha[i] = 1.0;
            } else if dilated[i] {
                alpha[i] = 0.5;
            }
        }
        let mut out = img.clone();
        heal_patchmatch(&mut out, &tight, &dilated, &alpha, 2.0, 0.0, 48, 48);

        let ring = out[(ring_i / 48, ring_i % 48, 0)];
        assert!(
            ring > 0.12,
            "ring must receive fill, not stay the marker (got {ring})"
        );
        assert!(
            ring < 0.70,
            "ring must blend with original, not full-copy (got {ring})"
        );
        assert!(
            (out[(outside_i / 48, outside_i % 48, 0)] - orig_out).abs() < 1e-5,
            "pixels outside dilated must stay"
        );
    }

    #[test]
    fn patch_overlap_fills_ring_from_splat() {
        let w = 32usize;
        let h = 32usize;
        let mut img = Array3::<f32>::from_elem((h, w, 3), 0.20);
        for y in 10..22 {
            for x in 6..12 {
                img[(y, x, 0)] = 0.90;
                img[(y, x, 1)] = 0.15;
                img[(y, x, 2)] = 0.15;
            }
        }
        let tight = hole_square(w, 12, 12, 5);
        let dilated = dilate(&tight, w, h, 3);
        let mut ring_left = None;
        let mut outside_i = None;
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if dilated[i] && !tight[i] && x < 12 && ring_left.is_none() {
                    ring_left = Some(i);
                }
                if !dilated[i] && x > 26 && outside_i.is_none() {
                    outside_i = Some(i);
                }
            }
        }
        let ring_i = ring_left.expect("left feather ring");
        let outside_i = outside_i.expect("outside dilated");
        let orig_out = img[(outside_i / w, outside_i % w, 0)];
        img[(ring_i / w, ring_i % w, 0)] = 0.0;
        img[(ring_i / w, ring_i % w, 1)] = 0.0;
        img[(ring_i / w, ring_i % w, 2)] = 0.0;

        let mut alpha = vec![0.0f32; w * h];
        for i in 0..w * h {
            if tight[i] {
                alpha[i] = 1.0;
            } else if dilated[i] {
                alpha[i] = 1.0;
            }
        }
        let mut out = img.clone();
        heal_patchmatch(&mut out, &tight, &dilated, &alpha, 2.0, 0.0, w, h);

        let ring_r = out[(ring_i / w, ring_i % w, 0)];
        assert!(
            ring_r > 0.05,
            "dilated ring must receive splat fill, not stay the marker (got {ring_r})"
        );
        assert!(
            (out[(outside_i / w, outside_i % w, 0)] - orig_out).abs() < 1e-5,
            "pixels outside dilated must stay"
        );
    }

    #[test]
    fn pm_ring_grain_raises_spread() {
        let w = 48usize;
        let h = 48usize;
        let img = grainy_field(w, h);
        let tight = hole_square(w, 16, 16, 5);
        let dilated = dilate(&tight, w, h, 3);
        let ring: Vec<bool> = dilated
            .iter()
            .zip(tight.iter())
            .map(|(&d, &t)| d && !t)
            .collect();
        let mut alpha = vec![0.0f32; w * h];
        for i in 0..w * h {
            if tight[i] {
                alpha[i] = 1.0;
            } else if dilated[i] {
                alpha[i] = 0.5;
            }
        }
        let mut none = img.clone();
        heal_patchmatch(&mut none, &tight, &dilated, &alpha, 2.0, 0.0, w, h);
        let mut grained = img.clone();
        heal_patchmatch(&mut grained, &tight, &dilated, &alpha, 2.0, 1.5, w, h);

        let (_, ring_none) = luma_stats(&none, &ring, w);
        let (_, ring_grain) = luma_stats(&grained, &ring, w);
        assert!(
            ring_grain > ring_none + 0.002,
            "feather grain must raise ring spread (none={ring_none}, grain={ring_grain})"
        );

        let (_, core_none) = luma_stats(&none, &tight, w);
        let (_, core_grain) = luma_stats(&grained, &tight, w);
        assert!(
            core_grain > core_none + 0.002,
            "core grain must raise hole spread (none={core_none}, grain={core_grain})"
        );
        let mut core_diff = 0.0f32;
        for i in 0..w * h {
            if !tight[i] {
                continue;
            }
            let y = i / w;
            let x = i % w;
            core_diff += (grained[(y, x, 0)] - none[(y, x, 0)]).abs();
        }
        assert!(
            core_diff > 0.02,
            "tight pixels must receive statistical grain (diff={core_diff})"
        );
    }

    #[test]
    fn long_stroke_does_not_carry_dark_along_sky() {
        let mut img = Array3::<f32>::from_elem((24, 64, 3), 0.78);
        for y in 0..24 {
            for x in 0..14 {
                img[(y, x, 0)] = 0.06;
                img[(y, x, 1)] = 0.06;
                img[(y, x, 2)] = 0.06;
            }
        }
        let mut tight = vec![false; 64 * 24];
        for x in 8..52 {
            tight[11 * 64 + x] = true;
            tight[12 * 64 + x] = true;
            tight[13 * 64 + x] = true;
        }
        let filled = run_fill(&img, &tight, 2.0, 64, 24);
        let mut sky = 0.0;
        let mut n = 0.0;
        for x in 32..48 {
            sky += luma(rgb_at(&filled, x, 12));
            n += 1.0;
        }
        sky /= n;
        assert!(
            sky > 0.55,
            "sky part of a long stroke must not copy the dark end (luma={sky})"
        );
    }
}
