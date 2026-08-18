//! Nearby overlapping-tile wave-function collapse for dust holes.

use std::collections::HashMap;

use ndarray::Array3;

use crate::dust::{connected_components, dilate, pixel_hash, rgb_at, structure_hv};

/// How far out from the stroke we harvest tiles.
const COLLAR_R: i32 = 40;
const Q_LEVELS: f32 = 31.0;
const DOMAIN_KEEP: usize = 48;
const MAX_ATTEMPTS: u32 = 6;
const GRAIN_AMP: f32 = 0.035;
const MONOTONE_MIN_CELLS: usize = 12;
const MONOTONE_SHARE: f32 = 0.25;
const REUSE_SSD: f32 = 0.004;

const DIR_RIGHT: usize = 0;
const DIR_LEFT: usize = 1;
const DIR_DOWN: usize = 2;
const DIR_UP: usize = 3;
const DIRS: [(usize, i32, i32); 4] = [
    (DIR_RIGHT, 1, 0),
    (DIR_LEFT, -1, 0),
    (DIR_DOWN, 0, 1),
    (DIR_UP, 0, -1),
];

#[derive(Clone)]
struct Tile {
    rgb: Vec<(f32, f32, f32)>,
    freq: u32,
}

struct Compat {
    n_tiles: usize,
    words: usize,
    bits: Vec<u64>,
}

impl Compat {
    fn new(n_tiles: usize) -> Self {
        let words = n_tiles.div_ceil(64).max(1);
        Self {
            n_tiles,
            words,
            bits: vec![0; n_tiles * 4 * words],
        }
    }

    fn set(&mut self, from: usize, dir: usize, to: usize) {
        let i = (from * 4 + dir) * self.words + to / 64;
        self.bits[i] |= 1u64 << (to % 64);
    }

    fn test(&self, from: usize, dir: usize, to: usize) -> bool {
        if from >= self.n_tiles || to >= self.n_tiles {
            return false;
        }
        let i = (from * 4 + dir) * self.words + to / 64;
        self.bits[i] & (1u64 << (to % 64)) != 0
    }
}

fn tile_n(tile: u8) -> usize {
    (tile as usize).clamp(2, 5)
}

fn max_tiles(n: usize) -> usize {
    if n >= 4 {
        1024
    } else {
        1536
    }
}

fn center_idx(n: usize) -> usize {
    let o = n / 2;
    o * n + o
}

fn quantize(c: f32) -> u8 {
    (c.clamp(0.0, 1.0) * Q_LEVELS).round() as u8
}

fn quantize_rgb(rgb: &[(f32, f32, f32)]) -> Vec<u8> {
    let mut q = Vec::with_capacity(rgb.len() * 3);
    for c in rgb {
        q.push(quantize(c.0));
        q.push(quantize(c.1));
        q.push(quantize(c.2));
    }
    q
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

fn remap_tile(
    rgb: &[(f32, f32, f32)],
    n: usize,
    map: impl Fn(usize, usize) -> (usize, usize),
) -> Vec<(f32, f32, f32)> {
    let mut out = vec![(0.0, 0.0, 0.0); n * n];
    for y in 0..n {
        for x in 0..n {
            let (sx, sy) = map(x, y);
            out[y * n + x] = rgb[sy * n + sx];
        }
    }
    out
}

fn flip_h(rgb: &[(f32, f32, f32)], n: usize) -> Vec<(f32, f32, f32)> {
    remap_tile(rgb, n, |x, y| (n - 1 - x, y))
}

fn flip_v(rgb: &[(f32, f32, f32)], n: usize) -> Vec<(f32, f32, f32)> {
    remap_tile(rgb, n, |x, y| (x, n - 1 - y))
}

fn flip_180(rgb: &[(f32, f32, f32)], n: usize) -> Vec<(f32, f32, f32)> {
    remap_tile(rgb, n, |x, y| (n - 1 - x, n - 1 - y))
}

fn median_f32(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.01;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// Sharp WFC fill on `tight`, composite with `alpha`, then synthetic hash grain.
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
    let (tiles, tau_base) = build_library(image, tight, &hole, n, w, h);
    let tau = (tau_base * loosen).max(1.0e-6);
    let (colors, _) = collapse_or_fallback(image, tight, component, &tiles, tau, n, w, h);
    for (&i, color) in component.iter().zip(colors) {
        fill[i] = Some(color);
    }
}

fn extract_rgb(
    image: &Array3<f32>,
    tight: &[bool],
    x: usize,
    y: usize,
    n: usize,
    w: usize,
    h: usize,
) -> Option<Vec<(f32, f32, f32)>> {
    let off = n / 2;
    if x < off || y < off || x + (n - off) > w || y + (n - off) > h {
        return None;
    }
    let x0 = x - off;
    let y0 = y - off;
    let mut rgb = vec![(0.0, 0.0, 0.0); n * n];
    for ty in 0..n {
        for tx in 0..n {
            let px = x0 + tx;
            let py = y0 + ty;
            if tight[py * w + px] {
                return None;
            }
            rgb[ty * n + tx] = rgb_at(image, px, py);
        }
    }
    Some(rgb)
}

fn insert_tile(
    tiles: &mut Vec<Tile>,
    seen: &mut HashMap<Vec<u8>, usize>,
    rgb: Vec<(f32, f32, f32)>,
    cap: usize,
) {
    let q = quantize_rgb(&rgb);
    if let Some(&idx) = seen.get(&q) {
        tiles[idx].freq = tiles[idx].freq.saturating_add(1);
    } else if tiles.len() < cap {
        seen.insert(q, tiles.len());
        tiles.push(Tile { rgb, freq: 1 });
    }
}

fn insert_with_flips(
    tiles: &mut Vec<Tile>,
    seen: &mut HashMap<Vec<u8>, usize>,
    rgb: Vec<(f32, f32, f32)>,
    n: usize,
    cap: usize,
) {
    let h = flip_h(&rgb, n);
    let v = flip_v(&rgb, n);
    let r = flip_180(&rgb, n);
    insert_tile(tiles, seen, rgb, cap);
    insert_tile(tiles, seen, h, cap);
    insert_tile(tiles, seen, v, cap);
    insert_tile(tiles, seen, r, cap);
}

fn collect_collar_ssds(
    image: &Array3<f32>,
    tight: &[bool],
    hole: &[bool],
    w: usize,
    h: usize,
) -> Vec<f32> {
    let collar = dilate(hole, w, h, COLLAR_R);
    let mut ssds = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !collar[i] || tight[i] || hole[i] {
                continue;
            }
            let a = rgb_at(image, x, y);
            if x + 1 < w {
                let j = i + 1;
                if collar[j] && !tight[j] && !hole[j] {
                    ssds.push(rgb_ssd(a, rgb_at(image, x + 1, y)));
                }
            }
            if y + 1 < h {
                let j = i + w;
                if collar[j] && !tight[j] && !hole[j] {
                    ssds.push(rgb_ssd(a, rgb_at(image, x, y + 1)));
                }
            }
        }
    }
    ssds
}

fn build_library(
    image: &Array3<f32>,
    tight: &[bool],
    hole: &[bool],
    n: usize,
    w: usize,
    h: usize,
) -> (Vec<Tile>, f32) {
    let tau_base = median_f32(collect_collar_ssds(image, tight, hole, w, h));
    let cap = max_tiles(n);
    let mut tiles: Vec<Tile> = Vec::new();
    let mut seen: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut prev = hole.to_vec();
    let margin = n / 2;
    let end_sub = n.saturating_sub(margin + 1);
    for r in 1..=COLLAR_R {
        let ring = dilate(hole, w, h, r);
        for y in margin..h.saturating_sub(end_sub) {
            for x in margin..w.saturating_sub(end_sub) {
                let i = y * w + x;
                if !ring[i] || prev[i] || tight[i] {
                    continue;
                }
                let Some(rgb) = extract_rgb(image, tight, x, y, n, w, h) else {
                    continue;
                };
                insert_with_flips(&mut tiles, &mut seen, rgb, n, cap);
            }
        }
        prev = ring;
        if tiles.len() >= cap {
            break;
        }
    }
    (tiles, tau_base)
}

fn overlap_h_ssd(a: &Tile, b: &Tile, n: usize) -> f32 {
    let mut s = 0.0f32;
    let mut c = 0.0f32;
    for row in 0..n {
        for col in 0..(n - 1) {
            s += rgb_ssd(a.rgb[row * n + col + 1], b.rgb[row * n + col]);
            c += 1.0;
        }
    }
    if c > 0.0 {
        s / c
    } else {
        0.0
    }
}

fn overlap_v_ssd(a: &Tile, b: &Tile, n: usize) -> f32 {
    let mut s = 0.0f32;
    let mut c = 0.0f32;
    for row in 0..(n - 1) {
        for col in 0..n {
            s += rgb_ssd(a.rgb[(row + 1) * n + col], b.rgb[row * n + col]);
            c += 1.0;
        }
    }
    if c > 0.0 {
        s / c
    } else {
        0.0
    }
}

fn build_compat(tiles: &[Tile], n: usize, tau: f32) -> Compat {
    let n_tiles = tiles.len();
    let mut compat = Compat::new(n_tiles);
    for i in 0..n_tiles {
        for j in 0..n_tiles {
            if overlap_h_ssd(&tiles[i], &tiles[j], n) < tau {
                compat.set(i, DIR_RIGHT, j);
                compat.set(j, DIR_LEFT, i);
            }
            if overlap_v_ssd(&tiles[i], &tiles[j], n) < tau {
                compat.set(i, DIR_DOWN, j);
                compat.set(j, DIR_UP, i);
            }
        }
    }
    compat
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

fn tile_known_ssd(tile: &Tile, known: &[(bool, (f32, f32, f32))]) -> f32 {
    let mut s = 0.0f32;
    let mut c = 0.0f32;
    for (k, &(ok, b)) in known.iter().enumerate() {
        if !ok {
            continue;
        }
        s += rgb_ssd(tile.rgb[k], b);
        c += 1.0;
    }
    if c > 0.0 {
        s / c
    } else {
        0.0
    }
}

fn reuse_count(t: usize, tiles: &[Tile], n: usize, placed: &[Option<u16>]) -> f32 {
    let tc = tiles[t].rgb[center_idx(n)];
    let mut reuse = 0.0f32;
    for &p in placed.iter().flatten() {
        let p = p as usize;
        if p == t {
            reuse += 1.0;
        } else if rgb_ssd(tc, tiles[p].rgb[center_idx(n)]) < REUSE_SSD {
            reuse += 0.6;
        }
    }
    reuse
}

fn neighbor_reuse(t: usize, k: usize, component: &[usize], placed: &[Option<u16>], w: usize) -> f32 {
    let i = component[k];
    let x = (i % w) as i32;
    let y = (i / w) as i32;
    let mut extra = 0.0f32;
    for &(_, dx, dy) in &DIRS {
        let nx = x + dx;
        let ny = y + dy;
        if nx < 0 || ny < 0 {
            continue;
        }
        let ni = ny as usize * w + nx as usize;
        let Some(pk) = component.iter().position(|&ci| ci == ni) else {
            continue;
        };
        if placed.get(pk).and_then(|p| *p) == Some(t as u16) {
            extra += 1.0;
        }
    }
    extra
}

fn best_tiles(
    tiles: &[Tile],
    known: &[(bool, (f32, f32, f32))],
    keep: usize,
) -> Vec<u16> {
    let mut scored: Vec<(f32, u16)> = tiles
        .iter()
        .enumerate()
        .map(|(i, t)| (tile_known_ssd(t, known), i as u16))
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(keep.min(tiles.len()).max(1));
    scored.into_iter().map(|(_, i)| i).collect()
}

fn pick_greedy_tile(
    tiles: &[Tile],
    known: &[(bool, (f32, f32, f32))],
    n: usize,
    placed_ids: &[Option<u16>],
    neighbor_same: &[u16],
) -> Option<u16> {
    if tiles.is_empty() {
        return None;
    }
    let mut best = f32::MAX;
    let mut best_i = 0u16;
    for (i, tile) in tiles.iter().enumerate() {
        let reuse = reuse_count(i, tiles, n, placed_ids);
        let neigh = neighbor_same.iter().filter(|&&t| t == i as u16).count() as f32;
        let score = tile_known_ssd(tile, known) + 0.02 * reuse + 0.05 * neigh;
        if score < best {
            best = score;
            best_i = i as u16;
        }
    }
    Some(best_i)
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
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
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

fn collapse_or_fallback(
    image: &Array3<f32>,
    tight: &[bool],
    component: &[usize],
    tiles: &[Tile],
    tau: f32,
    n: usize,
    w: usize,
    h: usize,
) -> (Vec<(f32, f32, f32)>, Vec<Option<u16>>) {
    let cells = component.len();
    let mut colors = vec![(0.0, 0.0, 0.0); cells];
    let mut ids = vec![None; cells];
    if tiles.is_empty() {
        for (k, &i) in component.iter().enumerate() {
            colors[k] = hv_fallback(image, tight, i % w, i / w, w, h);
        }
        return (colors, ids);
    }

    let mut cell_of = vec![-1i32; w * h];
    for (k, &i) in component.iter().enumerate() {
        cell_of[i] = k as i32;
    }

    let compat = build_compat(tiles, n, tau);
    let placed_empty = vec![None; w * h];
    let skip = vec![false; cells];

    let mut initial: Vec<Vec<u16>> = Vec::with_capacity(cells);
    for &i in component {
        let x = i % w;
        let y = i / w;
        let known = sample_known(x, y, n, image, &placed_empty, tight, w, h);
        initial.push(best_tiles(tiles, &known, DOMAIN_KEEP));
    }

    for attempt in 0..MAX_ATTEMPTS {
        if let Some(picked) =
            try_collapse(component, tiles, &compat, &cell_of, &initial, &skip, n, w, attempt)
        {
            if !is_monotone(&picked) {
                for (k, id) in picked.into_iter().enumerate() {
                    ids[k] = id;
                    if let Some(t) = id {
                        colors[k] = tiles[t as usize].rgb[center_idx(n)];
                    } else {
                        let (c, tid) =
                            greedy_pixel(image, tight, tiles, component[k], &placed_empty, &ids, n, w, h);
                        colors[k] = c;
                        ids[k] = tid;
                    }
                }
                return (colors, ids);
            }
        }
    }

    greedy_fill(image, tight, component, tiles, n, w, h)
}

fn is_monotone(picked: &[Option<u16>]) -> bool {
    let cells = picked.len();
    if cells < MONOTONE_MIN_CELLS {
        return false;
    }
    let mut counts: HashMap<u16, usize> = HashMap::new();
    for &id in picked {
        if let Some(t) = id {
            *counts.entry(t).or_insert(0) += 1;
        }
    }
    counts
        .values()
        .copied()
        .max()
        .is_some_and(|m| m as f32 > MONOTONE_SHARE * cells as f32)
}

fn greedy_pixel(
    image: &Array3<f32>,
    tight: &[bool],
    tiles: &[Tile],
    i: usize,
    placed: &[Option<(f32, f32, f32)>],
    placed_ids: &[Option<u16>],
    n: usize,
    w: usize,
    h: usize,
) -> ((f32, f32, f32), Option<u16>) {
    let x = i % w;
    let y = i / w;
    let known = sample_known(x, y, n, image, placed, tight, w, h);
    let mut neighbor_same = Vec::new();
    for &(_, dx, dy) in &DIRS {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
            continue;
        }
        let ni = ny as usize * w + nx as usize;
        if let Some(c) = placed.get(ni).and_then(|c| *c) {
            for (ti, tile) in tiles.iter().enumerate() {
                if rgb_ssd(tile.rgb[center_idx(n)], c) < REUSE_SSD {
                    neighbor_same.push(ti as u16);
                }
            }
        }
    }
    if let Some(ti) = pick_greedy_tile(tiles, &known, n, placed_ids, &neighbor_same) {
        (tiles[ti as usize].rgb[center_idx(n)], Some(ti))
    } else {
        (hv_fallback(image, tight, x, y, w, h), None)
    }
}

fn greedy_fill(
    image: &Array3<f32>,
    tight: &[bool],
    component: &[usize],
    tiles: &[Tile],
    n: usize,
    w: usize,
    h: usize,
) -> (Vec<(f32, f32, f32)>, Vec<Option<u16>>) {
    let mut placed = vec![None; w * h];
    let mut remaining = component.to_vec();
    let mut colors = vec![(0.0, 0.0, 0.0); component.len()];
    let mut ids = vec![None; component.len()];
    let mut index_of = vec![usize::MAX; w * h];
    for (k, &i) in component.iter().enumerate() {
        index_of[i] = k;
    }
    while !remaining.is_empty() {
        let mut best_k = 0usize;
        let mut best_n = -1i32;
        for (ri, &i) in remaining.iter().enumerate() {
            let known = sample_known(i % w, i / w, n, image, &placed, tight, w, h);
            let kn = known.iter().filter(|k| k.0).count() as i32;
            if kn > best_n {
                best_n = kn;
                best_k = ri;
            }
        }
        let i = remaining.swap_remove(best_k);
        let k = index_of[i];
        let (c, tid) = greedy_pixel(image, tight, tiles, i, &placed, &ids, n, w, h);
        placed[i] = Some(c);
        colors[k] = c;
        ids[k] = tid;
    }
    (colors, ids)
}

fn try_collapse(
    component: &[usize],
    tiles: &[Tile],
    compat: &Compat,
    cell_of: &[i32],
    initial: &[Vec<u16>],
    skip: &[bool],
    n: usize,
    w: usize,
    attempt: u32,
) -> Option<Vec<Option<u16>>> {
    let cells = component.len();
    let mut possible = initial.to_vec();
    let mut stack = Vec::new();
    for k in 0..cells {
        if !skip[k] && possible[k].len() < tiles.len() {
            stack.push(k);
        }
    }
    if !propagate(&mut possible, skip, component, cell_of, compat, w, &mut stack) {
        return None;
    }

    let mut picked = vec![None; cells];
    loop {
        let mut best: Option<(usize, usize, u32)> = None;
        for k in 0..cells {
            if skip[k] || picked[k].is_some() {
                continue;
            }
            let pn = possible[k].len();
            if pn == 0 {
                return None;
            }
            if pn == 1 {
                picked[k] = Some(possible[k][0]);
                stack.clear();
                stack.push(k);
                if !propagate(&mut possible, skip, component, cell_of, compat, w, &mut stack) {
                    return None;
                }
                best = None;
                break;
            }
            let i = component[k];
            let tie = pixel_hash(i % w, i / w) ^ attempt.wrapping_mul(0x9E37_79B9);
            match best {
                None => best = Some((pn, k, tie)),
                Some((bn, _, bt)) if pn < bn || (pn == bn && tie < bt) => {
                    best = Some((pn, k, tie));
                }
                _ => {}
            }
        }
        let Some((_, k, _)) = best else {
            break;
        };
        if possible[k].len() <= 1 {
            continue;
        }
        let choice = weighted_pick(&possible[k], tiles, &picked, component, n, k, w, attempt);
        possible[k] = vec![choice];
        picked[k] = Some(choice);
        stack.clear();
        stack.push(k);
        if !propagate(&mut possible, skip, component, cell_of, compat, w, &mut stack) {
            return None;
        }
    }

    Some(picked)
}

fn weighted_pick(
    domain: &[u16],
    tiles: &[Tile],
    placed: &[Option<u16>],
    component: &[usize],
    n: usize,
    k: usize,
    w: usize,
    attempt: u32,
) -> u16 {
    let mut weights = Vec::with_capacity(domain.len());
    let mut total = 0.0f32;
    for &t in domain {
        let reuse = reuse_count(t as usize, tiles, n, placed);
        let neigh = neighbor_reuse(t as usize, k, component, placed, w);
        let wgt = tiles[t as usize].freq.max(1) as f32 / (1.0 + 4.0 * reuse + 3.0 * neigh);
        weights.push(wgt);
        total += wgt;
    }
    if total <= 1.0e-8 {
        return domain[0];
    }
    let h = pixel_hash(component[k] % w, component[k] / w) ^ attempt.wrapping_mul(0x85EB_CA6B);
    let mut r = hash_unit(h) * total;
    for (&t, &wgt) in domain.iter().zip(&weights) {
        if r < wgt {
            return t;
        }
        r -= wgt;
    }
    *domain.last().unwrap_or(&0)
}

fn propagate(
    possible: &mut [Vec<u16>],
    skip: &[bool],
    component: &[usize],
    cell_of: &[i32],
    compat: &Compat,
    w: usize,
    stack: &mut Vec<usize>,
) -> bool {
    while let Some(ci) = stack.pop() {
        if skip[ci] {
            continue;
        }
        let i = component[ci];
        let x = (i % w) as i32;
        let y = (i / w) as i32;
        for &(dir, dx, dy) in &DIRS {
            let nx = x + dx;
            let ny = y + dy;
            if nx < 0 || ny < 0 {
                continue;
            }
            let ni = ny as usize * w + nx as usize;
            if ni >= cell_of.len() {
                continue;
            }
            let nj = cell_of[ni];
            if nj < 0 {
                continue;
            }
            let nj = nj as usize;
            if skip[nj] {
                continue;
            }
            let before = possible[nj].len();
            let src = possible[ci].clone();
            possible[nj].retain(|&tb| {
                src.iter()
                    .any(|&ta| compat.test(ta as usize, dir, tb as usize))
            });
            if possible[nj].is_empty() {
                return false;
            }
            if possible[nj].len() < before {
                stack.push(nj);
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn luma_stats(
        img: &Array3<f32>,
        mask: &[bool],
        w: usize,
        invert: bool,
    ) -> (f32, f32) {
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

    #[test]
    fn empty_library_falls_back_without_panic() {
        let img = Array3::<f32>::from_elem((8, 8, 3), 0.4);
        let tight = vec![false; 64];
        let component = vec![27usize];
        let (colors, _) = collapse_or_fallback(&img, &tight, &component, &[], 0.01, 3, 8, 8);
        assert_eq!(colors.len(), 1);
    }

    #[test]
    fn grainy_hole_uses_library_not_only_hv() {
        let img = grainy_field(32, 32);
        let (tight, component) = hole_square(32, 14, 14, 4);
        let (tiles, tau_base) = build_library(&img, &tight, &tight, 3, 32, 32);
        assert!(!tiles.is_empty(), "collar must yield tiles");
        let (wfc, _) = collapse_or_fallback(&img, &tight, &component, &tiles, tau_base * 2.5, 3, 32, 32);
        let mut diff = 0.0f32;
        for (k, &i) in component.iter().enumerate() {
            let hv = hv_fallback(&img, &tight, i % 32, i / 32, 32, 32);
            diff += (wfc[k].0 - hv.0).abs() + (wfc[k].1 - hv.1).abs() + (wfc[k].2 - hv.2).abs();
        }
        assert!(
            diff > 0.02,
            "WFC must place library tiles, not only H+V (diff={diff})"
        );
    }

    #[test]
    fn harvest_includes_flips() {
        let mut img = Array3::<f32>::from_elem((24, 24, 3), 0.4);
        // Asymmetric 3×3 so H/V/180 are distinct after quantize.
        for y in 8..11 {
            for x in 8..11 {
                img[(y, x, 0)] = 0.2 + x as f32 * 0.08;
                img[(y, x, 1)] = 0.3 + y as f32 * 0.04;
                img[(y, x, 2)] = 0.15 + (x + y) as f32 * 0.02;
            }
        }
        let mut hole = vec![false; 24 * 24];
        hole[12 * 24 + 12] = true;
        let tight = hole.clone();
        let (tiles, _) = build_library(&img, &tight, &hole, 3, 24, 24);
        let Some(src) = extract_rgb(&img, &tight, 9, 9, 3, 24, 24) else {
            panic!("expected harvestable tile");
        };
        let variants = [src.clone(), flip_h(&src, 3), flip_v(&src, 3), flip_180(&src, 3)];
        for (i, rgb) in variants.iter().enumerate() {
            let q = quantize_rgb(rgb);
            assert!(
                tiles.iter().any(|t| quantize_rgb(&t.rgb) == q),
                "library must contain flip variant {i}"
            );
        }
    }

    #[test]
    fn grainy_hole_is_not_one_tile() {
        let img = grainy_field(48, 48);
        let (tight, component) = hole_square(48, 16, 16, 5);
        assert!(component.len() >= 16);
        let (tiles, tau_base) = build_library(&img, &tight, &tight, 3, 48, 48);
        assert!(tiles.len() > 8, "grainy collar should yield many tiles");
        let (_, ids) = collapse_or_fallback(&img, &tight, &component, &tiles, tau_base * 2.5, 3, 48, 48);
        let mut counts: HashMap<u16, usize> = HashMap::new();
        for id in ids.iter().flatten() {
            *counts.entry(*id).or_insert(0) += 1;
        }
        let max = counts.values().copied().max().unwrap_or(0);
        let share = max as f32 / component.len() as f32;
        assert!(
            share <= 0.25,
            "no tile may cover more than 25% of a 16+ px hole (share={share}, max={max})"
        );
    }

    #[test]
    fn grainy_hole_keeps_collar_variance_and_mean() {
        let img = grainy_field(48, 48);
        let (tight, component) = hole_square(48, 16, 16, 5);
        let (tiles, tau_base) = build_library(&img, &tight, &tight, 3, 48, 48);
        let (colors, _) =
            collapse_or_fallback(&img, &tight, &component, &tiles, tau_base * 2.5, 3, 48, 48);

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
}
