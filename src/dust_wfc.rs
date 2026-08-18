//! Nearby overlapping-tile wave-function collapse for dust holes.

use ndarray::Array3;

use crate::dust::{connected_components, dilate, pixel_hash, rgb_at, structure_hv};

const TILE: usize = 3;
const TILE_PIX: usize = 9;
const COLLAR_R: i32 = 12;
const MAX_TILES: usize = 256;
const Q_LEVELS: f32 = 15.0;
const OVERLAP_TOL: u8 = 1;
const DOMAIN_KEEP: usize = 24;
const MAX_ATTEMPTS: u32 = 6;
const GRAIN_AMP: f32 = 0.035;

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
    q: [u8; 27],
    rgb: [(f32, f32, f32); TILE_PIX],
    freq: u32,
}

fn quantize(c: f32) -> u8 {
    (c.clamp(0.0, 1.0) * Q_LEVELS).round() as u8
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

/// Sharp WFC fill on `tight`, composite with `alpha`, then synthetic hash grain.
pub(crate) fn heal_wfc(
    image: &mut Array3<f32>,
    tight: &[bool],
    dilated: &[bool],
    alpha: &[f32],
    grain_amount: f32,
    w: usize,
    h: usize,
) {
    let n = w * h;
    let mut fill = vec![None; n];
    for component in connected_components(tight, w, h) {
        fill_component(image, tight, &component, &mut fill, w, h);
    }

    for i in 0..n {
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
    for i in 0..n {
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
    let tiles = build_library(image, tight, &hole, w, h);
    let colors = collapse_or_fallback(image, tight, component, &tiles, w, h);
    for (&i, color) in component.iter().zip(colors) {
        fill[i] = Some(color);
    }
}

fn build_library(
    image: &Array3<f32>,
    tight: &[bool],
    hole: &[bool],
    w: usize,
    h: usize,
) -> Vec<Tile> {
    let outer = dilate(hole, w, h, COLLAR_R);
    let mut tiles: Vec<Tile> = Vec::new();
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let i = y * w + x;
            if !outer[i] || tight[i] {
                continue;
            }
            let mut ok = true;
            let mut q = [0u8; 27];
            let mut rgb = [(0.0f32, 0.0f32, 0.0f32); TILE_PIX];
            for ty in 0..TILE {
                for tx in 0..TILE {
                    let px = x + tx - 1;
                    let py = y + ty - 1;
                    let pi = py * w + px;
                    if tight[pi] {
                        ok = false;
                        break;
                    }
                    let c = rgb_at(image, px, py);
                    let k = ty * TILE + tx;
                    rgb[k] = c;
                    let qi = k * 3;
                    q[qi] = quantize(c.0);
                    q[qi + 1] = quantize(c.1);
                    q[qi + 2] = quantize(c.2);
                }
                if !ok {
                    break;
                }
            }
            if !ok {
                continue;
            }
            if let Some(t) = tiles.iter_mut().find(|t| t.q == q) {
                t.freq = t.freq.saturating_add(1);
            } else {
                tiles.push(Tile { q, rgb, freq: 1 });
            }
        }
    }
    if tiles.len() > MAX_TILES {
        tiles.sort_by(|a, b| b.freq.cmp(&a.freq));
        tiles.truncate(MAX_TILES);
    }
    tiles
}

fn q_close(a: u8, b: u8) -> bool {
    a.abs_diff(b) <= OVERLAP_TOL
}

fn overlap_h(a: &Tile, b: &Tile) -> bool {
    for row in 0..TILE {
        for col in 0..2 {
            let ia = (row * TILE + col + 1) * 3;
            let ib = (row * TILE + col) * 3;
            if !q_close(a.q[ia], b.q[ib])
                || !q_close(a.q[ia + 1], b.q[ib + 1])
                || !q_close(a.q[ia + 2], b.q[ib + 2])
            {
                return false;
            }
        }
    }
    true
}

fn overlap_v(a: &Tile, b: &Tile) -> bool {
    for row in 0..2 {
        for col in 0..TILE {
            let ia = ((row + 1) * TILE + col) * 3;
            let ib = (row * TILE + col) * 3;
            if !q_close(a.q[ia], b.q[ib])
                || !q_close(a.q[ia + 1], b.q[ib + 1])
                || !q_close(a.q[ia + 2], b.q[ib + 2])
            {
                return false;
            }
        }
    }
    true
}

fn build_compat(tiles: &[Tile]) -> Vec<[[bool; MAX_TILES]; 4]> {
    let n = tiles.len().min(MAX_TILES);
    let mut compat = vec![[[false; MAX_TILES]; 4]; n];
    for i in 0..n {
        for j in 0..n {
            if overlap_h(&tiles[i], &tiles[j]) {
                compat[i][DIR_RIGHT][j] = true;
                compat[j][DIR_LEFT][i] = true;
            }
            if overlap_v(&tiles[i], &tiles[j]) {
                compat[i][DIR_DOWN][j] = true;
                compat[j][DIR_UP][i] = true;
            }
        }
    }
    compat
}

fn sample_known(
    x: usize,
    y: usize,
    image: &Array3<f32>,
    placed: &[Option<(f32, f32, f32)>],
    tight: &[bool],
    w: usize,
    h: usize,
) -> [(bool, (f32, f32, f32)); TILE_PIX] {
    let mut out = [(false, (0.0, 0.0, 0.0)); TILE_PIX];
    for ty in 0..TILE {
        for tx in 0..TILE {
            let px = x as i32 + tx as i32 - 1;
            let py = y as i32 + ty as i32 - 1;
            if px < 0 || py < 0 || px >= w as i32 || py >= h as i32 {
                continue;
            }
            let px = px as usize;
            let py = py as usize;
            let i = py * w + px;
            let k = ty * TILE + tx;
            if let Some(c) = placed.get(i).and_then(|c| *c) {
                out[k] = (true, c);
            } else if !tight[i] {
                out[k] = (true, rgb_at(image, px, py));
            }
        }
    }
    out
}

fn tile_known_ssd(tile: &Tile, known: &[(bool, (f32, f32, f32)); TILE_PIX]) -> f32 {
    let mut s = 0.0f32;
    let mut n = 0f32;
    for k in 0..TILE_PIX {
        if !known[k].0 {
            continue;
        }
        let a = tile.rgb[k];
        let b = known[k].1;
        let d0 = a.0 - b.0;
        let d1 = a.1 - b.1;
        let d2 = a.2 - b.2;
        s += d0 * d0 + d1 * d1 + d2 * d2;
        n += 1.0;
    }
    if n > 0.0 {
        s / n
    } else {
        0.0
    }
}

fn best_tiles(
    tiles: &[Tile],
    known: &[(bool, (f32, f32, f32)); TILE_PIX],
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
    w: usize,
    h: usize,
) -> Vec<(f32, f32, f32)> {
    let cells = component.len();
    let mut colors = vec![(0.0, 0.0, 0.0); cells];
    if tiles.is_empty() {
        for (k, &i) in component.iter().enumerate() {
            colors[k] = hv_fallback(image, tight, i % w, i / w, w, h);
        }
        return colors;
    }

    let mut cell_of = vec![-1i32; w * h];
    for (k, &i) in component.iter().enumerate() {
        cell_of[i] = k as i32;
    }

    let n_tiles = tiles.len().min(MAX_TILES);
    let tiles = &tiles[..n_tiles];
    let compat = build_compat(tiles);
    let placed_empty = vec![None; w * h];
    let skip = vec![false; cells];

    let mut initial: Vec<Vec<u16>> = Vec::with_capacity(cells);
    for &i in component {
        let x = i % w;
        let y = i / w;
        let known = sample_known(x, y, image, &placed_empty, tight, w, h);
        initial.push(best_tiles(tiles, &known, DOMAIN_KEEP));
    }

    for attempt in 0..MAX_ATTEMPTS {
        match try_collapse(
            component,
            tiles,
            &compat,
            &cell_of,
            &initial,
            &skip,
            w,
            attempt,
        ) {
            Some(picked) => {
                for (k, _) in component.iter().enumerate() {
                    if let Some(t) = picked[k] {
                        colors[k] = tiles[t as usize].rgb[4];
                    } else {
                        colors[k] = greedy_pixel(image, tight, tiles, component[k], &placed_empty, w, h);
                    }
                }
                return colors;
            }
            None => continue,
        }
    }

    greedy_fill(image, tight, component, tiles, w, h)
}

fn greedy_pixel(
    image: &Array3<f32>,
    tight: &[bool],
    tiles: &[Tile],
    i: usize,
    placed: &[Option<(f32, f32, f32)>],
    w: usize,
    h: usize,
) -> (f32, f32, f32) {
    let known = sample_known(i % w, i / w, image, placed, tight, w, h);
    if let Some(&ti) = best_tiles(tiles, &known, 1).first() {
        tiles[ti as usize].rgb[4]
    } else {
        hv_fallback(image, tight, i % w, i / w, w, h)
    }
}

fn greedy_fill(
    image: &Array3<f32>,
    tight: &[bool],
    component: &[usize],
    tiles: &[Tile],
    w: usize,
    h: usize,
) -> Vec<(f32, f32, f32)> {
    let mut placed = vec![None; w * h];
    let mut remaining = component.to_vec();
    let mut colors = vec![(0.0, 0.0, 0.0); component.len()];
    let mut index_of = vec![usize::MAX; w * h];
    for (k, &i) in component.iter().enumerate() {
        index_of[i] = k;
    }
    while !remaining.is_empty() {
        let mut best_k = 0usize;
        let mut best_n = -1i32;
        for (ri, &i) in remaining.iter().enumerate() {
            let x = i % w;
            let y = i / w;
            let known = sample_known(x, y, image, &placed, tight, w, h);
            let n = known.iter().filter(|k| k.0).count() as i32;
            if n > best_n {
                best_n = n;
                best_k = ri;
            }
        }
        let i = remaining.swap_remove(best_k);
        let c = greedy_pixel(image, tight, tiles, i, &placed, w, h);
        placed[i] = Some(c);
        colors[index_of[i]] = c;
    }
    colors
}

fn try_collapse(
    component: &[usize],
    tiles: &[Tile],
    compat: &[[[bool; MAX_TILES]; 4]],
    cell_of: &[i32],
    initial: &[Vec<u16>],
    skip: &[bool],
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
            let n = possible[k].len();
            if n == 0 {
                return None;
            }
            if n == 1 {
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
                None => best = Some((n, k, tie)),
                Some((bn, _, bt)) if n < bn || (n == bn && tie < bt) => {
                    best = Some((n, k, tie));
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
        let choice = weighted_pick(&possible[k], tiles, component[k], w, attempt);
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

fn weighted_pick(domain: &[u16], tiles: &[Tile], i: usize, w: usize, attempt: u32) -> u16 {
    let total: u32 = domain.iter().map(|&t| tiles[t as usize].freq.max(1)).sum();
    if total == 0 {
        return domain[0];
    }
    let h = pixel_hash(i % w, i / w) ^ attempt.wrapping_mul(0x85EB_CA6B);
    let mut r = h % total;
    for &t in domain {
        let f = tiles[t as usize].freq.max(1);
        if r < f {
            return t;
        }
        r -= f;
    }
    *domain.last().unwrap_or(&0)
}

fn propagate(
    possible: &mut [Vec<u16>],
    skip: &[bool],
    component: &[usize],
    cell_of: &[i32],
    compat: &[[[bool; MAX_TILES]; 4]],
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
                    .any(|&ta| compat[ta as usize][dir][tb as usize])
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

    #[test]
    fn empty_library_falls_back_without_panic() {
        let img = Array3::<f32>::from_elem((8, 8, 3), 0.4);
        let tight = vec![false; 64];
        let component = vec![27usize];
        let colors = collapse_or_fallback(&img, &tight, &component, &[], 8, 8);
        assert_eq!(colors.len(), 1);
    }

    #[test]
    fn grainy_hole_uses_library_not_only_hv() {
        let mut img = Array3::<f32>::from_elem((32, 32, 3), 0.4);
        for y in 0..32 {
            for x in 0..32 {
                let n = ((x * 13 + y * 29) % 9) as f32 * 0.04 - 0.16;
                img[(y, x, 0)] = (0.4 + n).clamp(0.0, 1.0);
                img[(y, x, 1)] = (0.4 + n * 0.8).clamp(0.0, 1.0);
                img[(y, x, 2)] = (0.4 + n * 0.6).clamp(0.0, 1.0);
            }
        }
        let mut tight = vec![false; 32 * 32];
        let mut component = Vec::new();
        for y in 14..=17 {
            for x in 14..=17 {
                tight[y * 32 + x] = true;
                component.push(y * 32 + x);
            }
        }
        let tiles = build_library(&img, &tight, &tight, 32, 32);
        assert!(!tiles.is_empty(), "collar must yield tiles");
        let wfc = collapse_or_fallback(&img, &tight, &component, &tiles, 32, 32);
        let mut diff = 0.0f32;
        for (k, &i) in component.iter().enumerate() {
            let hv = hv_fallback(&img, &tight, i % 32, i / 32, 32, 32);
            diff += (wfc[k].0 - hv.0).abs()
                + (wfc[k].1 - hv.1).abs()
                + (wfc[k].2 - hv.2).abs();
        }
        assert!(
            diff > 0.02,
            "WFC must place library tiles, not only H+V (diff={diff})"
        );
    }
}
