//! Structure-tensor bridge + extracted film grain for dust holes.
//!
//! Low-pass is smeared along the collar’s direction (or H+V when isotropic).
//! High-pass residual is copied from film along that flow. No patch RGB,
//! so mid-hole X seams cannot form.

use ndarray::Array3;

use crate::dust::{
    blur_rgb_masked, connected_components, dilate, estimate_grain_sigma, rgb_at, structure_hv,
};

const RIM_R: i32 = 8;
const DIRS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

fn tile_n(tile: u8) -> i32 {
    (tile as i32).clamp(2, 5)
}

fn rgb_ssd(a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
    let d0 = a.0 - b.0;
    let d1 = a.1 - b.1;
    let d2 = a.2 - b.2;
    (d0 * d0 + d1 * d1 + d2 * d2) / 3.0
}

fn luma(c: (f32, f32, f32)) -> f32 {
    0.2126 * c.0 + 0.7152 * c.1 + 0.0722 * c.2
}

fn lerp3(a: (f32, f32, f32), b: (f32, f32, f32), t: f32) -> (f32, f32, f32) {
    (
        a.0 + (b.0 - a.0) * t,
        a.1 + (b.1 - a.1) * t,
        a.2 + (b.2 - a.2) * t,
    )
}

fn median_f32(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.01;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

fn add3(a: (f32, f32, f32), b: (f32, f32, f32), s: f32) -> (f32, f32, f32) {
    (
        (a.0 + b.0 * s).clamp(0.0, 1.0),
        (a.1 + b.1 * s).clamp(0.0, 1.0),
        (a.2 + b.2 * s).clamp(0.0, 1.0),
    )
}

/// Structure-flow fill on `tight`, then alpha composite. Grain scales residual.
pub(crate) fn heal_wfc(
    image: &mut Array3<f32>,
    tight: &[bool],
    _dilated: &[bool],
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
    let grain = grain_amount.clamp(0.0, 3.0);
    let sigma = estimate_grain_sigma(image, tight, w, h);
    let low = blur_rgb_masked(image, tight, sigma);
    let mut fill = vec![None; n_pix];
    for component in connected_components(tight, w, h) {
        fill_component(
            image, &low, tight, &component, &mut fill, n, loosen, grain, w, h,
        );
    }
    composite_alpha(image, &fill, alpha, w, h);
}

fn composite_alpha(
    image: &mut Array3<f32>,
    fill: &[Option<(f32, f32, f32)>],
    alpha: &[f32],
    w: usize,
    h: usize,
) {
    let _ = (w, h);
    for i in 0..fill.len() {
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
    low: &Array3<f32>,
    tight: &[bool],
    component: &[usize],
    fill: &mut [Option<(f32, f32, f32)>],
    tile: i32,
    loosen: f32,
    grain: f32,
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
    let (rim_mean, color_gate, ring) = rim_stats(image, tight, &hole, w, h);
    let (flow, coherence) = collar_tensor(low, tight, &hole, w, h);
    let aniso = (coherence * loosen / 2.5).clamp(0.0, 1.0);
    let use_flow = aniso > 0.30;
    let offset = coherent_offset(tight, component, flow, use_flow, w, h);

    for &i in component {
        let x = i % w;
        let y = i / w;
        let bridged = bridge_low(
            low, tight, x, y, flow, aniso, rim_mean, w, h,
        );
        let residual = if grain <= 1.0e-5 {
            (0.0, 0.0, 0.0)
        } else {
            residual_along(
                image, low, tight, x, y, offset, flow, tile, rim_mean, color_gate, &ring, w, h,
            )
        };
        fill[i] = Some(add3(bridged, residual, grain));
    }
}

fn rim_stats(
    image: &Array3<f32>,
    tight: &[bool],
    hole: &[bool],
    w: usize,
    h: usize,
) -> ((f32, f32, f32), f32, Vec<usize>) {
    let rim = dilate(hole, w, h, RIM_R);
    let mut acc = (0.0f32, 0.0f32, 0.0f32);
    let mut n = 0.0f32;
    let mut colors = Vec::new();
    let mut ring = Vec::new();
    for i in 0..w * h {
        if !rim[i] || tight[i] {
            continue;
        }
        let c = rgb_at(image, i % w, i / w);
        acc.0 += c.0;
        acc.1 += c.1;
        acc.2 += c.2;
        n += 1.0;
        colors.push(c);
        ring.push(i);
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
    (rim_mean, scale.max(1.0e-4) * 2.5, ring)
}

fn luma_at(low: &Array3<f32>, x: usize, y: usize) -> f32 {
    luma(rgb_at(low, x, y))
}

fn grad_luma(
    low: &Array3<f32>,
    tight: &[bool],
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) -> Option<(f32, f32)> {
    let ok = |xx: usize, yy: usize| !tight[yy * w + xx];
    let ix = if x > 0 && x + 1 < w && ok(x - 1, y) && ok(x + 1, y) {
        (luma_at(low, x + 1, y) - luma_at(low, x - 1, y)) * 0.5
    } else if x + 1 < w && ok(x + 1, y) && ok(x, y) {
        luma_at(low, x + 1, y) - luma_at(low, x, y)
    } else if x > 0 && ok(x - 1, y) && ok(x, y) {
        luma_at(low, x, y) - luma_at(low, x - 1, y)
    } else {
        return None;
    };
    let iy = if y > 0 && y + 1 < h && ok(x, y - 1) && ok(x, y + 1) {
        (luma_at(low, x, y + 1) - luma_at(low, x, y - 1)) * 0.5
    } else if y + 1 < h && ok(x, y + 1) && ok(x, y) {
        luma_at(low, x, y + 1) - luma_at(low, x, y)
    } else if y > 0 && ok(x, y - 1) && ok(x, y) {
        luma_at(low, x, y) - luma_at(low, x, y - 1)
    } else {
        return None;
    };
    Some((ix, iy))
}

/// Flow (edge tangent) and coherence on the tight collar.
fn collar_tensor(
    low: &Array3<f32>,
    tight: &[bool],
    hole: &[bool],
    w: usize,
    h: usize,
) -> ((f32, f32), f32) {
    let rim = dilate(hole, w, h, RIM_R);
    let mut e = 0.0f32;
    let mut f = 0.0f32;
    let mut g = 0.0f32;
    let mut n = 0.0f32;
    for i in 0..w * h {
        if !rim[i] || tight[i] {
            continue;
        }
        let Some((ix, iy)) = grad_luma(low, tight, i % w, i / w, w, h) else {
            continue;
        };
        e += ix * ix;
        f += ix * iy;
        g += iy * iy;
        n += 1.0;
    }
    if n < 4.0 {
        return ((1.0, 0.0), 0.0);
    }
    e /= n;
    f /= n;
    g /= n;
    let trace = e + g;
    let det = e * g - f * f;
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    let l1 = 0.5 * (trace + disc);
    let l2 = 0.5 * (trace - disc);
    let coherence = ((l1 - l2) / (l1 + l2 + 1.0e-8)).clamp(0.0, 1.0);
    let mut vx = f;
    let mut vy = l2 - e;
    let mut len = (vx * vx + vy * vy).sqrt();
    if len < 1.0e-8 {
        vx = l2 - g;
        vy = f;
        len = (vx * vx + vy * vy).sqrt();
    }
    let flow = if len < 1.0e-8 {
        (1.0, 0.0)
    } else {
        (vx / len, vy / len)
    };
    (flow, coherence)
}

fn walk_to_known(
    tight: &[bool],
    mut x: f32,
    mut y: f32,
    dx: f32,
    dy: f32,
    w: usize,
    h: usize,
) -> Option<(usize, usize, f32)> {
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1.0e-6 {
        return None;
    }
    let dx = dx / len;
    let dy = dy / len;
    x += 0.5;
    y += 0.5;
    for step in 1..=48 {
        x += dx;
        y += dy;
        let ix = x.floor() as i32;
        let iy = y.floor() as i32;
        if ix < 0 || iy < 0 || ix >= w as i32 || iy >= h as i32 {
            return None;
        }
        let ix = ix as usize;
        let iy = iy as usize;
        if !tight[iy * w + ix] {
            return Some((ix, iy, step as f32));
        }
    }
    None
}

fn bridge_low(
    low: &Array3<f32>,
    tight: &[bool],
    x: usize,
    y: usize,
    flow: (f32, f32),
    aniso: f32,
    rim_mean: (f32, f32, f32),
    w: usize,
    h: usize,
) -> (f32, f32, f32) {
    let hv = structure_hv(low, tight, x, y, w, h);
    let flowed = if aniso > 0.05 {
        let a = walk_to_known(tight, x as f32, y as f32, flow.0, flow.1, w, h);
        let b = walk_to_known(tight, x as f32, y as f32, -flow.0, -flow.1, w, h);
        match (a, b) {
            (Some((ax, ay, da)), Some((bx, by, db))) => {
                let t = da / (da + db).max(1.0e-6);
                Some(lerp3(rgb_at(low, ax, ay), rgb_at(low, bx, by), t))
            }
            (Some((ax, ay, _)), None) => Some(rgb_at(low, ax, ay)),
            (None, Some((bx, by, _))) => Some(rgb_at(low, bx, by)),
            (None, None) => None,
        }
    } else {
        None
    };
    match (flowed, hv) {
        (Some(a), Some(b)) => lerp3(b, a, aniso),
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => rim_mean,
    }
}

fn coherent_offset(
    tight: &[bool],
    component: &[usize],
    flow: (f32, f32),
    use_flow: bool,
    w: usize,
    h: usize,
) -> (i32, i32) {
    let mut cx = 0i32;
    let mut cy = 0i32;
    for &i in component {
        cx += (i % w) as i32;
        cy += (i / w) as i32;
    }
    let n = component.len().max(1) as i32;
    let cx = cx / n;
    let cy = cy / n;
    if use_flow {
        let a = walk_to_known(tight, cx as f32, cy as f32, flow.0, flow.1, w, h);
        let b = walk_to_known(tight, cx as f32, cy as f32, -flow.0, -flow.1, w, h);
        if let Some((sx, sy, _)) = match (a, b) {
            (Some(a), Some(b)) => Some(if a.2 <= b.2 { a } else { b }),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        } {
            return (sx as i32 - cx, sy as i32 - cy);
        }
    }
    nearest_known_offset(tight, cx, cy, w, h)
}

fn nearest_known_offset(tight: &[bool], cx: i32, cy: i32, w: usize, h: usize) -> (i32, i32) {
    for r in 1..=24 {
        for (dx, dy) in DIRS {
            let x = cx + dx * r;
            let y = cy + dy * r;
            if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
                continue;
            }
            if !tight[y as usize * w + x as usize] {
                return (dx * r, dy * r);
            }
        }
    }
    (1, 0)
}

fn residual_at(
    image: &Array3<f32>,
    low: &Array3<f32>,
    tight: &[bool],
    x: i32,
    y: i32,
    rim: (f32, f32, f32),
    gate: f32,
    w: usize,
    h: usize,
) -> Option<(f32, f32, f32)> {
    if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
        return None;
    }
    let x = x as usize;
    let y = y as usize;
    if tight[y * w + x] {
        return None;
    }
    let c = rgb_at(image, x, y);
    if rgb_ssd(c, rim) > gate {
        return None;
    }
    let l = rgb_at(low, x, y);
    Some((c.0 - l.0, c.1 - l.1, c.2 - l.2))
}

fn residual_along(
    image: &Array3<f32>,
    low: &Array3<f32>,
    tight: &[bool],
    x: usize,
    y: usize,
    offset: (i32, i32),
    flow: (f32, f32),
    tile: i32,
    rim: (f32, f32, f32),
    gate: f32,
    ring: &[usize],
    w: usize,
    h: usize,
) -> (f32, f32, f32) {
    let (ox, oy) = offset;
    let mut sx = x as i32 + ox;
    let mut sy = y as i32 + oy;
    let step_x = ox.signum();
    let step_y = oy.signum();
    for _ in 0..48 {
        if let Some(r) = residual_at(image, low, tight, sx, sy, rim, gate, w, h) {
            return r;
        }
        if step_x == 0 && step_y == 0 {
            break;
        }
        if sx >= 0 && sy >= 0 && sx < w as i32 && sy < h as i32 && !tight[sy as usize * w + sx as usize]
        {
            break;
        }
        sx += step_x;
        sy += step_y;
    }

    for k in 1..=tile {
        for sign in [1, -1] {
            let tx = x as i32 + ox + ((sign as f32) * (k as f32) * flow.0).round() as i32;
            let ty = y as i32 + oy + ((sign as f32) * (k as f32) * flow.1).round() as i32;
            if let Some(r) = residual_at(image, low, tight, tx, ty, rim, gate, w, h) {
                return r;
            }
        }
    }

    nearest_ring_residual(image, low, x as i32, y as i32, rim, gate, ring, w)
}

fn nearest_ring_residual(
    image: &Array3<f32>,
    low: &Array3<f32>,
    x: i32,
    y: i32,
    rim: (f32, f32, f32),
    gate: f32,
    ring: &[usize],
    w: usize,
) -> (f32, f32, f32) {
    let mut best = (i32::MAX, (0.0, 0.0, 0.0));
    let mut any = false;
    for &i in ring {
        let rx = (i % w) as i32;
        let ry = (i / w) as i32;
        let c = rgb_at(image, rx as usize, ry as usize);
        if rgb_ssd(c, rim) > gate {
            continue;
        }
        let d = (rx - x) * (rx - x) + (ry - y) * (ry - y);
        if d < best.0 {
            let l = rgb_at(low, rx as usize, ry as usize);
            best = (d, (c.0 - l.0, c.1 - l.1, c.2 - l.2));
            any = true;
        }
    }
    if any {
        best.1
    } else {
        (0.0, 0.0, 0.0)
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
            let y = luma(rgb_at(img, i % w, i / w));
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
        grain: f32,
        tile: u8,
        loosen: f32,
        w: usize,
        h: usize,
    ) -> Array3<f32> {
        let mut out = img.clone();
        let alpha: Vec<f32> = tight.iter().map(|&t| if t { 1.0 } else { 0.0 }).collect();
        heal_wfc(&mut out, tight, tight, &alpha, grain, tile, loosen, w, h);
        out
    }

    fn max_hole_luma_jump(img: &Array3<f32>, tight: &[bool], w: usize) -> f32 {
        let mut max_j = 0.0f32;
        for (i, &on) in tight.iter().enumerate() {
            if !on {
                continue;
            }
            let y0 = luma(rgb_at(img, i % w, i / w));
            let x = (i % w) as i32;
            let y = (i / w) as i32;
            for (dx, dy) in DIRS {
                let nx = x + dx;
                let ny = y + dy;
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= w as i32 {
                    continue;
                }
                let ni = ny as usize * w + nx as usize;
                if !tight[ni] {
                    continue;
                }
                max_j = max_j.max((y0 - luma(rgb_at(img, nx as usize, ny as usize))).abs());
            }
        }
        max_j
    }

    #[test]
    fn empty_hole_does_not_panic() {
        let mut img = Array3::<f32>::from_elem((8, 8, 3), 0.4);
        let tight = vec![false; 64];
        let alpha = vec![0.0f32; 64];
        heal_wfc(&mut img, &tight, &tight, &alpha, 1.0, 3, 2.5, 8, 8);
    }

    #[test]
    fn grainy_hole_uses_residual_not_only_hv() {
        let img = grainy_field(32, 32);
        let (tight, _) = hole_square(32, 14, 14, 4);
        let smooth = run_fill(&img, &tight, 0.0, 3, 2.5, 32, 32);
        let grained = run_fill(&img, &tight, 1.0, 3, 2.5, 32, 32);
        let mut diff = 0.0f32;
        for i in 0..32 * 32 {
            if !tight[i] {
                continue;
            }
            let x = i % 32;
            let y = i / 32;
            diff += (grained[(y, x, 0)] - smooth[(y, x, 0)]).abs();
        }
        assert!(
            diff > 0.02,
            "grain>0 must add extracted residual (diff={diff})"
        );
    }

    #[test]
    fn grainy_hole_keeps_collar_variance_and_mean() {
        let img = grainy_field(48, 48);
        let (tight, _) = hole_square(48, 16, 16, 5);
        let filled = run_fill(&img, &tight, 1.0, 3, 2.5, 48, 48);
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
        let (tight, _) = hole_square(48, 22, 22, 5);
        let filled = run_fill(&img, &tight, 1.0, 5, 4.0, 48, 48);
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
    fn grainy_hole_has_no_diagonal_ridge() {
        let img = grainy_field(48, 48);
        let (tight, _) = hole_square(48, 16, 16, 8);
        let filled = run_fill(&img, &tight, 1.0, 5, 4.0, 48, 48);
        let hole_jump = max_hole_luma_jump(&filled, &tight, 48);
        let collar = {
            let outer = dilate(&tight, 48, 48, 6);
            outer
                .iter()
                .zip(tight.iter())
                .map(|(&o, &t)| o && !t)
                .collect::<Vec<_>>()
        };
        let collar_jump = max_hole_luma_jump(&img, &collar, 48);
        assert!(
            hole_jump <= collar_jump * 1.8 + 0.02,
            "hole neighbor jumps must stay near film grain, not an X ridge (hole={hole_jump}, collar={collar_jump})"
        );
    }

    #[test]
    fn two_tone_keeps_vertical_edge() {
        let mut img = Array3::<f32>::from_elem((24, 24, 3), 0.0);
        for y in 0..24 {
            for x in 0..24 {
                let v = if x < 12 { 0.18 } else { 0.82 };
                img[(y, x, 0)] = v;
                img[(y, x, 1)] = v;
                img[(y, x, 2)] = v;
            }
        }
        let (tight, _) = hole_square(24, 8, 8, 8);
        let filled = run_fill(&img, &tight, 1.0, 5, 4.0, 24, 24);
        let mut col_spread = 0.0f32;
        for x in 8..16 {
            let mut lo = 1.0f32;
            let mut hi = 0.0f32;
            for y in 8..16 {
                let v = luma(rgb_at(&filled, x, y));
                lo = lo.min(v);
                hi = hi.max(v);
            }
            col_spread = col_spread.max(hi - lo);
        }
        assert!(
            col_spread < 0.20,
            "flow along the edge must not paint a diagonal X (column spread={col_spread})"
        );
        let left = luma(rgb_at(&filled, 9, 12));
        let right = luma(rgb_at(&filled, 14, 12));
        assert!(
            (right - left).abs() > 0.30,
            "the real vertical edge must remain (left={left}, right={right})"
        );
    }
}
