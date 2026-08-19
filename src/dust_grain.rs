//! Statistical film grain for Wave-function dust holes.
//!
//! Edge-preserving NLM → residual → NLF + PSD from flat, on-color film.
//! Synthesize shaped noise into the hole only.

use std::f32::consts::PI;
use std::sync::Arc;

use ndarray::Array3;
use rayon::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::dust::{pixel_hash, rgb_at};

pub const MARGIN: i32 = 48;
pub const PSD_N: usize = 16;
const SEARCH: i32 = 3;
const PATCH: i32 = 1;
const NLM_H: f32 = 0.06;
const NLF_BINS: usize = 32;
const MIN_BIN: usize = 16;
const MIN_PATCHES: usize = 4;

fn luma(c: (f32, f32, f32)) -> f32 {
    0.2126 * c.0 + 0.7152 * c.1 + 0.0722 * c.2
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

fn gauss(x: usize, y: usize, salt: u32) -> f32 {
    let u1 = hash_unit(pixel_hash(x, y) ^ salt).max(1.0e-6);
    let u2 = hash_unit(pixel_hash(x.wrapping_add(17), y.wrapping_add(31)) ^ salt.rotate_left(7));
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

fn median_f32(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

fn mad_sigma(mut v: Vec<f32>) -> f32 {
    if v.len() < 4 {
        return 0.0;
    }
    let med = median_f32(v.clone());
    for x in &mut v {
        *x = (*x - med).abs();
    }
    median_f32(v) * 1.4826
}

pub(crate) fn component_bbox(
    component: &[usize],
    w: usize,
    h: usize,
    margin: i32,
) -> (usize, usize, usize, usize) {
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
    let x0 = (x0 as i32 - margin).max(0) as usize;
    let y0 = (y0 as i32 - margin).max(0) as usize;
    let x1 = ((x1 as i32 + margin).min(w as i32 - 1)) as usize;
    let y1 = ((y1 as i32 + margin).min(h as i32 - 1)) as usize;
    (x0, y0, x1, y1)
}

/// NLM on the bbox; hole pixels are never read. Writes known pixels of `den`.
pub(crate) fn nlm_bbox(
    den: &mut Array3<f32>,
    image: &Array3<f32>,
    hole: &[bool],
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    w: usize,
    h: usize,
) {
    let h2 = (NLM_H * NLM_H).max(1.0e-8);
    let rows: Vec<(usize, Vec<(f32, f32, f32)>)> = (y0..=y1)
        .into_par_iter()
        .map(|y| {
            let mut row = Vec::with_capacity(x1 - x0 + 1);
            for x in x0..=x1 {
                if hole[y * w + x] {
                    row.push(rgb_at(image, x, y));
                    continue;
                }
                let mut acc = (0.0f32, 0.0f32, 0.0f32);
                let mut wsum = 0.0f32;
                for dy in -SEARCH..=SEARCH {
                    for dx in -SEARCH..=SEARCH {
                        let qx = x as i32 + dx;
                        let qy = y as i32 + dy;
                        if qx < 0 || qy < 0 || qx >= w as i32 || qy >= h as i32 {
                            continue;
                        }
                        let qx = qx as usize;
                        let qy = qy as usize;
                        if hole[qy * w + qx] {
                            continue;
                        }
                        let Some(ssd) = patch_ssd(image, hole, x, y, qx, qy, w, h) else {
                            continue;
                        };
                        let wt = (-ssd / h2).exp();
                        let c = rgb_at(image, qx, qy);
                        acc.0 += c.0 * wt;
                        acc.1 += c.1 * wt;
                        acc.2 += c.2 * wt;
                        wsum += wt;
                    }
                }
                if wsum > 1.0e-8 {
                    row.push((acc.0 / wsum, acc.1 / wsum, acc.2 / wsum));
                } else {
                    row.push(rgb_at(image, x, y));
                }
            }
            (y, row)
        })
        .collect();

    for (y, row) in rows {
        for (k, c) in row.into_iter().enumerate() {
            let x = x0 + k;
            den[(y, x, 0)] = c.0;
            den[(y, x, 1)] = c.1;
            den[(y, x, 2)] = c.2;
        }
    }
}

fn patch_ssd(
    image: &Array3<f32>,
    hole: &[bool],
    px: usize,
    py: usize,
    qx: usize,
    qy: usize,
    w: usize,
    h: usize,
) -> Option<f32> {
    let mut s = 0.0f32;
    let mut n = 0.0f32;
    for dy in -PATCH..=PATCH {
        for dx in -PATCH..=PATCH {
            let ax = px as i32 + dx;
            let ay = py as i32 + dy;
            let bx = qx as i32 + dx;
            let by = qy as i32 + dy;
            if ax < 0 || ay < 0 || bx < 0 || by < 0 {
                continue;
            }
            if ax >= w as i32 || ay >= h as i32 || bx >= w as i32 || by >= h as i32 {
                continue;
            }
            let ax = ax as usize;
            let ay = ay as usize;
            let bx = bx as usize;
            let by = by as usize;
            if hole[ay * w + ax] || hole[by * w + bx] {
                continue;
            }
            s += rgb_ssd(rgb_at(image, ax, ay), rgb_at(image, bx, by));
            n += 1.0;
        }
    }
    if n < 5.0 {
        None
    } else {
        Some(s / n)
    }
}

fn local_var_and_lap(
    den: &Array3<f32>,
    hole: &[bool],
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    w: usize,
    h: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = w * h;
    let mut var = vec![f32::MAX; n];
    let mut lap = vec![f32::MAX; n];
    for y in y0..=y1 {
        for x in x0..=x1 {
            let i = y * w + x;
            if hole[i] {
                continue;
            }
            let mut s = 0.0f32;
            let mut s2 = 0.0f32;
            let mut c = 0.0f32;
            for dy in -2i32..=2 {
                for dx in -2i32..=2 {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let nx = nx as usize;
                    let ny = ny as usize;
                    if hole[ny * w + nx] {
                        continue;
                    }
                    let v = luma(rgb_at(den, nx, ny));
                    s += v;
                    s2 += v * v;
                    c += 1.0;
                }
            }
            if c >= 8.0 {
                let m = s / c;
                var[i] = (s2 / c - m * m).max(0.0);
            }
            let mut acc = 0.0f32;
            let mut ln = 0.0f32;
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    continue;
                }
                let nx = nx as usize;
                let ny = ny as usize;
                if hole[ny * w + nx] {
                    continue;
                }
                acc += luma(rgb_at(den, nx, ny));
                ln += 1.0;
            }
            if ln >= 3.0 {
                lap[i] = (acc - ln * luma(rgb_at(den, x, y))).abs();
            }
        }
    }
    (var, lap)
}

struct GrainModel {
    nlf: [Vec<(f32, f32)>; 3],
    psd_sqrt: Vec<f32>,
    patch: usize,
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
}

fn interp_nlf(nlf: &[(f32, f32)], luma: f32) -> f32 {
    if nlf.is_empty() {
        return 0.02;
    }
    if nlf.len() == 1 || luma <= nlf[0].0 {
        return nlf[0].1;
    }
    let last = nlf[nlf.len() - 1];
    if luma >= last.0 {
        return last.1;
    }
    for w in nlf.windows(2) {
        if luma <= w[1].0 {
            let t = (luma - w[0].0) / (w[1].0 - w[0].0).max(1.0e-6);
            return w[0].1 + (w[1].1 - w[0].1) * t;
        }
    }
    last.1
}

fn estimate(
    image: &Array3<f32>,
    den: &Array3<f32>,
    hole: &[bool],
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    rim: (f32, f32, f32),
    gate: f32,
    loosen: f32,
    w: usize,
    h: usize,
) -> GrainModel {
    let patch = if (x1 - x0).min(y1 - y0) + 1 >= 40 {
        PSD_N
    } else {
        8
    };
    let (var, lap) = local_var_and_lap(den, hole, x0, y0, x1, y1, w, h);
    let mut vars = Vec::new();
    for y in y0..=y1 {
        for x in x0..=x1 {
            let i = y * w + x;
            if hole[i] || var[i] >= f32::MAX / 2.0 {
                continue;
            }
            vars.push(var[i]);
        }
    }
    let med_v = median_f32(vars);
    let v_th = (med_v * 2.0 * loosen).max(1.0e-5);
    let lap_th = 0.025 * loosen;

    let mut flat = vec![false; w * h];
    for y in y0..=y1 {
        for x in x0..=x1 {
            let i = y * w + x;
            if hole[i] {
                continue;
            }
            if rgb_ssd(rgb_at(image, x, y), rim) > gate {
                continue;
            }
            if var[i] <= v_th && lap[i] <= lap_th {
                flat[i] = true;
            }
        }
    }

    let mut bins: [Vec<Vec<f32>>; 3] = [
        vec![Vec::new(); NLF_BINS],
        vec![Vec::new(); NLF_BINS],
        vec![Vec::new(); NLF_BINS],
    ];
    for y in y0..=y1 {
        for x in x0..=x1 {
            let i = y * w + x;
            if !flat[i] {
                continue;
            }
            let d = rgb_at(den, x, y);
            let r = rgb_at(image, x, y);
            let b = ((luma(d).clamp(0.0, 0.999)) * NLF_BINS as f32) as usize;
            bins[0][b].push(r.0 - d.0);
            bins[1][b].push(r.1 - d.1);
            bins[2][b].push(r.2 - d.2);
        }
    }
    let mut nlf = [Vec::new(), Vec::new(), Vec::new()];
    for ch in 0..3 {
        for (i, bin) in bins[ch].iter().enumerate() {
            if bin.len() < MIN_BIN {
                continue;
            }
            let center = (i as f32 + 0.5) / NLF_BINS as f32;
            nlf[ch].push((center, mad_sigma(bin.clone()).max(1.0e-5)));
        }
    }
    if nlf.iter().any(|c| c.is_empty()) {
        let mut all = [Vec::new(), Vec::new(), Vec::new()];
        for y in y0..=y1 {
            for x in x0..=x1 {
                let i = y * w + x;
                if hole[i] || rgb_ssd(rgb_at(image, x, y), rim) > gate {
                    continue;
                }
                let d = rgb_at(den, x, y);
                let r = rgb_at(image, x, y);
                all[0].push(r.0 - d.0);
                all[1].push(r.1 - d.1);
                all[2].push(r.2 - d.2);
            }
        }
        for ch in 0..3 {
            if nlf[ch].is_empty() {
                nlf[ch].push((0.5, mad_sigma(all[ch].clone()).max(0.015)));
            }
        }
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(patch);
    let ifft = planner.plan_fft_inverse(patch);
    let mut psd = vec![0.0f32; patch * patch];
    let mut n_psd = 0u32;
    let half = patch as i32 / 2;
    let step = (patch / 2).max(1);
    let mut y = y0 as i32 + half;
    while y + half <= y1 as i32 {
        let mut x = x0 as i32 + half;
        while x + half <= x1 as i32 {
            if patch_is_flat(&flat, x, y, patch, w, h) {
                let spec = fft2_residual(image, den, &nlf, x as usize, y as usize, patch, &fft, w);
                for (p, c) in psd.iter_mut().zip(spec) {
                    *p += c.re * c.re + c.im * c.im;
                }
                n_psd += 1;
            }
            x += step as i32;
        }
        y += step as i32;
    }
    if n_psd >= MIN_PATCHES as u32 {
        let inv = 1.0 / n_psd as f32;
        for p in &mut psd {
            *p = (*p * inv).sqrt();
        }
    } else {
        psd.fill(1.0);
    }
    psd[0] = 0.0;

    GrainModel {
        nlf,
        psd_sqrt: psd,
        patch,
        fft,
        ifft,
    }
}

fn patch_is_flat(flat: &[bool], cx: i32, cy: i32, patch: usize, w: usize, h: usize) -> bool {
    let half = patch as i32 / 2;
    for dy in 0..patch as i32 {
        for dx in 0..patch as i32 {
            let x = cx + dx - half;
            let y = cy + dy - half;
            if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
                return false;
            }
            if !flat[y as usize * w + x as usize] {
                return false;
            }
        }
    }
    true
}

fn fft2_residual(
    image: &Array3<f32>,
    den: &Array3<f32>,
    nlf: &[Vec<(f32, f32)>; 3],
    cx: usize,
    cy: usize,
    patch: usize,
    fft: &Arc<dyn Fft<f32>>,
    _w: usize,
) -> Vec<Complex<f32>> {
    let half = patch / 2;
    let mut buf = vec![Complex::new(0.0, 0.0); patch * patch];
    for dy in 0..patch {
        for dx in 0..patch {
            let x = cx + dx - half;
            let y = cy + dy - half;
            let d = rgb_at(den, x, y);
            let r = rgb_at(image, x, y);
            let sig = (interp_nlf(&nlf[0], luma(d))
                + interp_nlf(&nlf[1], luma(d))
                + interp_nlf(&nlf[2], luma(d)))
                / 3.0;
            let res = (luma(r) - luma(d)) / sig.max(1.0e-5);
            buf[dy * patch + dx] = Complex::new(res, 0.0);
        }
    }
    fft2(&mut buf, patch, fft);
    buf
}

fn fft2(buf: &mut [Complex<f32>], n: usize, fft: &Arc<dyn Fft<f32>>) {
    for y in 0..n {
        fft.process(&mut buf[y * n..(y + 1) * n]);
    }
    let mut col = vec![Complex::new(0.0, 0.0); n];
    for x in 0..n {
        for y in 0..n {
            col[y] = buf[y * n + x];
        }
        fft.process(&mut col);
        for y in 0..n {
            buf[y * n + x] = col[y];
        }
    }
}

fn filter_spectrum(
    noise: &[f32],
    psd_sqrt: &[f32],
    n: usize,
    fft: &Arc<dyn Fft<f32>>,
    ifft: &Arc<dyn Fft<f32>>,
) -> Vec<f32> {
    let mut buf: Vec<Complex<f32>> = noise.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft2(&mut buf, n, fft);
    for (c, &s) in buf.iter_mut().zip(psd_sqrt) {
        *c *= s;
    }
    fft2(&mut buf, n, ifft);
    let scale = 1.0 / (n * n) as f32;
    let mut out: Vec<f32> = buf.iter().map(|c| c.re * scale).collect();
    let mut mean = 0.0f32;
    for &v in &out {
        mean += v;
    }
    mean /= out.len() as f32;
    let mut var = 0.0f32;
    for v in &mut out {
        *v -= mean;
        var += *v * *v;
    }
    let s = (var / out.len() as f32).sqrt().max(1.0e-6);
    for v in &mut out {
        *v /= s;
    }
    out
}

fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            if n <= 1 {
                1.0
            } else {
                0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos())
            }
        })
        .collect()
}

/// Add NLF×PSD grain onto structure fill, hole pixels only.
pub(crate) fn apply_statistical_grain(
    image: &Array3<f32>,
    den: &Array3<f32>,
    hole: &[bool],
    component: &[usize],
    fill: &mut [Option<(f32, f32, f32)>],
    grain: f32,
    rim: (f32, f32, f32),
    gate: f32,
    loosen: f32,
    w: usize,
    h: usize,
) {
    if component.is_empty() || grain <= 1.0e-5 {
        return;
    }
    let (x0, y0, x1, y1) = component_bbox(component, w, h, MARGIN);
    let model = estimate(image, den, hole, x0, y0, x1, y1, rim, gate, loosen, w, h);
    let n = model.patch;
    let step = (n / 2).max(1);
    let win = hann(n);
    let mut acc = vec![(0.0f32, 0.0f32, 0.0f32); w * h];
    let mut wsum = vec![0.0f32; w * h];

    let mut ty = y0 as i32 - n as i32 / 2;
    let y_end = y1 as i32 + n as i32 / 2;
    let x_end = x1 as i32 + n as i32 / 2;
    while ty <= y_end {
        let mut tx = x0 as i32 - n as i32 / 2;
        while tx <= x_end {
            for ch in 0..3 {
                let salt =
                    0xA5A5_A5A5u32.wrapping_mul(ch as u32 + 1) ^ ((tx as u32) << 8) ^ (ty as u32);
                let mut noise = vec![0.0f32; n * n];
                for py in 0..n {
                    for px in 0..n {
                        noise[py * n + px] = gauss(
                            (tx + px as i32).max(0) as usize,
                            (ty + py as i32).max(0) as usize,
                            salt,
                        );
                    }
                }
                let shaped = filter_spectrum(&noise, &model.psd_sqrt, n, &model.fft, &model.ifft);
                for py in 0..n {
                    for px in 0..n {
                        let hx = tx + px as i32;
                        let hy = ty + py as i32;
                        if hx < 0 || hy < 0 || hx >= w as i32 || hy >= h as i32 {
                            continue;
                        }
                        let hi = hy as usize * w + hx as usize;
                        if !hole[hi] {
                            continue;
                        }
                        let wt = win[px] * win[py];
                        let g = shaped[py * n + px] * wt;
                        match ch {
                            0 => acc[hi].0 += g,
                            1 => acc[hi].1 += g,
                            _ => acc[hi].2 += g,
                        }
                        if ch == 0 {
                            wsum[hi] += wt;
                        }
                    }
                }
            }
            tx += step as i32;
        }
        ty += step as i32;
    }

    for &i in component {
        let Some(base) = fill[i] else {
            continue;
        };
        let wt = wsum[i].max(1.0e-6);
        let y = luma(base);
        let g0 = acc[i].0 / wt * interp_nlf(&model.nlf[0], y) * grain;
        let g1 = acc[i].1 / wt * interp_nlf(&model.nlf[1], y) * grain;
        let g2 = acc[i].2 / wt * interp_nlf(&model.nlf[2], y) * grain;
        fill[i] = Some((
            (base.0 + g0).clamp(0.0, 1.0),
            (base.1 + g1).clamp(0.0, 1.0),
            (base.2 + g2).clamp(0.0, 1.0),
        ));
    }
}
