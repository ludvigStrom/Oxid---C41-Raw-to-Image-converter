//! D-min neutralization: sample unexposed film border and divide image by median R, G, B.
//!
//! Purely linear: each channel is divided by its median in the crop region.

use anyhow::{bail, Result};

/// Compute median of a slice of f32. Sorts a copy; returns 0.0 if empty.
fn median_f32(slice: &[f32]) -> f32 {
    if slice.is_empty() {
        return 0.0;
    }
    let mut v = slice.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    if v.len() % 2 == 1 {
        v[mid]
    } else {
        (v[mid - 1] + v[mid]) / 2.0
    }
}

/// Neutralize D-min in-place: sample the crop region (x, y, width, height), compute median R/G/B, divide the whole image by those values.
///
/// If any median is 0, that channel is left unchanged (divide by 1.0) to avoid NaNs.
/// The rect is clamped to the image bounds; at least one pixel must remain in the crop.
pub fn neutralize(
    image: &mut ndarray::Array3<f32>,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<()> {
    let (h, w, _c) = image.dim();
    if _c != 3 {
        bail!("D-min expects RGB image (3 channels), got {}", _c);
    }

    let x = x as usize;
    let y = y as usize;
    let rw = width as usize;
    let rh = height as usize;

    // Clamp to image bounds
    let x_end = (x + rw).min(w);
    let y_end = (y + rh).min(h);
    let x_start = x.min(w.saturating_sub(1));
    let y_start = y.min(h.saturating_sub(1));

    if x_start >= x_end || y_start >= y_end {
        bail!(
            "D-min rect [{}, {}] + {}x{} is outside or zero-size for image {}x{}",
            x,
            y,
            rw,
            rh,
            w,
            h
        );
    }

    let region = image.slice(ndarray::s![y_start..y_end, x_start..x_end, ..]);
    let n = (y_end - y_start) * (x_end - x_start);

    let mut r_vals = Vec::with_capacity(n);
    let mut g_vals = Vec::with_capacity(n);
    let mut b_vals = Vec::with_capacity(n);

    for row in region.axis_iter(ndarray::Axis(0)) {
        for pixel in row.axis_iter(ndarray::Axis(0)) {
            r_vals.push(pixel[0]);
            g_vals.push(pixel[1]);
            b_vals.push(pixel[2]);
        }
    }

    let med_r = median_f32(&r_vals);
    let med_g = median_f32(&g_vals);
    let med_b = median_f32(&b_vals);

    // Print D-min medians so they can be measured once and hard-coded later if desired.
    println!(
        "D-min medians (linear [0,1]): R={:.6} G={:.6} B={:.6}",
        med_r, med_g, med_b
    );

    neutralize_with_medians(image, med_r, med_g, med_b)
}

/// Neutralize D-min using fixed medians (e.g. previously measured once).
pub fn neutralize_with_medians(
    image: &mut ndarray::Array3<f32>,
    med_r: f32,
    med_g: f32,
    med_b: f32,
) -> Result<()> {
    let (_h, _w, c) = image.dim();
    if c != 3 {
        bail!("D-min expects RGB image (3 channels), got {}", c);
    }

    let div_r = if med_r > 0.0 { med_r } else { 1.0 };
    let div_g = if med_g > 0.0 { med_g } else { 1.0 };
    let div_b = if med_b > 0.0 { med_b } else { 1.0 };

    let mut im = image.view_mut();
    im.slice_mut(ndarray::s![.., .., 0]).mapv_inplace(|v| v / div_r);
    im.slice_mut(ndarray::s![.., .., 1]).mapv_inplace(|v| v / div_g);
    im.slice_mut(ndarray::s![.., .., 2]).mapv_inplace(|v| v / div_b);

    Ok(())
}
