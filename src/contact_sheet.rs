//! Contact-sheet JPEG: grid of already-encoded RGB thumbnails.

use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::jpeg_export;

/// Layout for a single JPEG contact sheet.
#[derive(Debug, Clone)]
pub struct ContactSheetLayout {
    pub columns: u32,
    pub cell_long: u32,
    pub gap: u32,
    pub caption_height: u32,
    pub background: [u8; 3],
}

impl Default for ContactSheetLayout {
    fn default() -> Self {
        Self {
            columns: 4,
            cell_long: 400,
            gap: 16,
            caption_height: 18,
            background: [18, 18, 20],
        }
    }
}

#[derive(Debug, Clone)]
pub struct ContactSheetCell {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
    pub caption: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContactSheetSize {
    pub width: u32,
    pub height: u32,
    pub rows: u32,
}

pub fn sheet_size(n: usize, layout: &ContactSheetLayout) -> Result<ContactSheetSize> {
    if n == 0 {
        bail!("Contact sheet needs at least one image");
    }
    let cols = layout.columns.max(1);
    let rows = (n as u32).div_ceil(cols);
    let cell_w = layout.cell_long;
    let cell_h = layout.cell_long + layout.caption_height;
    let width = cols * cell_w + (cols + 1) * layout.gap;
    let height = rows * cell_h + (rows + 1) * layout.gap;
    if width == 0 || height == 0 {
        bail!("Contact sheet size is zero");
    }
    Ok(ContactSheetSize {
        width,
        height,
        rows,
    })
}

/// Fit `src_w × src_h` inside a `box_w × box_h` rectangle, keeping aspect.
pub fn fit_size(src_w: u32, src_h: u32, box_w: u32, box_h: u32) -> (u32, u32) {
    let src_w = src_w.max(1) as f32;
    let src_h = src_h.max(1) as f32;
    let scale = (box_w as f32 / src_w).min(box_h as f32 / src_h).min(1.0);
    (
        (src_w * scale).round().max(1.0) as u32,
        (src_h * scale).round().max(1.0) as u32,
    )
}

pub fn compose(cells: &[ContactSheetCell], layout: &ContactSheetLayout) -> Result<(u32, u32, Vec<u8>)> {
    let size = sheet_size(cells.len(), layout)?;
    let mut rgb = vec![0u8; size.width as usize * size.height as usize * 3];
    for px in rgb.chunks_exact_mut(3) {
        px.copy_from_slice(&layout.background);
    }

    let cols = layout.columns.max(1);
    let cell_w = layout.cell_long;
    let cell_h = layout.cell_long + layout.caption_height;
    for (i, cell) in cells.iter().enumerate() {
        let col = i as u32 % cols;
        let row = i as u32 / cols;
        let x0 = layout.gap + col * (cell_w + layout.gap);
        let y0 = layout.gap + row * (cell_h + layout.gap);
        blit_fit(
            &mut rgb,
            size.width,
            size.height,
            x0,
            y0,
            cell_w,
            layout.cell_long,
            cell,
        );
        draw_caption(
            &mut rgb,
            size.width,
            size.height,
            x0,
            y0 + layout.cell_long,
            cell_w,
            layout.caption_height,
            &cell.caption,
        );
    }
    Ok((size.width, size.height, rgb))
}

fn blit_fit(
    dest: &mut [u8],
    dest_w: u32,
    dest_h: u32,
    x0: u32,
    y0: u32,
    box_w: u32,
    box_h: u32,
    cell: &ContactSheetCell,
) {
    if cell.width == 0 || cell.height == 0 || cell.rgb.len() < (cell.width * cell.height * 3) as usize
    {
        return;
    }
    let (fw, fh) = fit_size(cell.width, cell.height, box_w, box_h);
    let ox = x0 + (box_w.saturating_sub(fw)) / 2;
    let oy = y0 + (box_h.saturating_sub(fh)) / 2;
    for y in 0..fh {
        let sy = (y as u64 * cell.height as u64 / fh as u64) as u32;
        for x in 0..fw {
            let sx = (x as u64 * cell.width as u64 / fw as u64) as u32;
            let di = (((oy + y) as usize * dest_w as usize) + (ox + x) as usize) * 3;
            let si = ((sy as usize * cell.width as usize) + sx as usize) * 3;
            if di + 2 < dest.len()
                && si + 2 < cell.rgb.len()
                && oy + y < dest_h
                && ox + x < dest_w
            {
                dest[di..di + 3].copy_from_slice(&cell.rgb[si..si + 3]);
            }
        }
    }
}

/// Tiny 5×7 ASCII font for captions.
fn draw_caption(
    dest: &mut [u8],
    dest_w: u32,
    dest_h: u32,
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
    text: &str,
) {
    let max_chars = (width.saturating_sub(2) / 6) as usize;
    let shown: String = text.chars().take(max_chars).collect();
    let text_w = (shown.chars().count() as u32) * 6;
    let mut x = x0 + width.saturating_sub(text_w) / 2;
    let y = y0 + height.saturating_sub(7) / 2;
    for ch in shown.chars() {
        blit_glyph(dest, dest_w, dest_h, x, y, ch);
        x += 6;
    }
}

fn blit_glyph(dest: &mut [u8], dest_w: u32, dest_h: u32, x: u32, y: u32, ch: char) {
    let bits = glyph_bits(ch);
    for row in 0..7u32 {
        let line = (bits >> ((6 - row) * 5)) & 0b11111;
        for col in 0..5u32 {
            if (line & (1 << (4 - col))) == 0 {
                continue;
            }
            let px = x + col;
            let py = y + row;
            if px >= dest_w || py >= dest_h {
                continue;
            }
            let i = ((py as usize * dest_w as usize) + px as usize) * 3;
            dest[i] = 210;
            dest[i + 1] = 210;
            dest[i + 2] = 214;
        }
    }
}

fn glyph_bits(ch: char) -> u64 {
    match ch {
        '0' => 0b01110_10001_10011_10101_11001_10001_01110,
        '1' => 0b00100_01100_00100_00100_00100_00100_01110,
        '2' => 0b01110_10001_00001_00010_00100_01000_11111,
        '3' => 0b01110_10001_00001_00110_00001_10001_01110,
        '4' => 0b00010_00110_01010_10010_11111_00010_00010,
        '5' => 0b11111_10000_11110_00001_00001_10001_01110,
        '6' => 0b00110_01000_10000_11110_10001_10001_01110,
        '7' => 0b11111_00001_00010_00100_01000_01000_01000,
        '8' => 0b01110_10001_10001_01110_10001_10001_01110,
        '9' => 0b01110_10001_10001_01111_00001_00010_01100,
        'A' | 'a' => 0b01110_10001_10001_11111_10001_10001_10001,
        'B' | 'b' => 0b11110_10001_10001_11110_10001_10001_11110,
        'C' | 'c' => 0b01110_10001_10000_10000_10000_10001_01110,
        'D' | 'd' => 0b11110_10001_10001_10001_10001_10001_11110,
        'E' | 'e' => 0b11111_10000_10000_11110_10000_10000_11111,
        'F' | 'f' => 0b11111_10000_10000_11110_10000_10000_10000,
        'G' | 'g' => 0b01110_10001_10000_10111_10001_10001_01110,
        'H' | 'h' => 0b10001_10001_10001_11111_10001_10001_10001,
        'I' | 'i' => 0b01110_00100_00100_00100_00100_00100_01110,
        'J' | 'j' => 0b00111_00010_00010_00010_00010_10010_01100,
        'K' | 'k' => 0b10001_10010_10100_11000_10100_10010_10001,
        'L' | 'l' => 0b10000_10000_10000_10000_10000_10000_11111,
        'M' | 'm' => 0b10001_11011_10101_10001_10001_10001_10001,
        'N' | 'n' => 0b10001_11001_10101_10011_10001_10001_10001,
        'O' | 'o' => 0b01110_10001_10001_10001_10001_10001_01110,
        'P' | 'p' => 0b11110_10001_10001_11110_10000_10000_10000,
        'R' | 'r' => 0b11110_10001_10001_11110_10100_10010_10001,
        'S' | 's' => 0b01110_10001_10000_01110_00001_10001_01110,
        'T' | 't' => 0b11111_00100_00100_00100_00100_00100_00100,
        'U' | 'u' => 0b10001_10001_10001_10001_10001_10001_01110,
        'V' | 'v' => 0b10001_10001_10001_10001_10001_01010_00100,
        'W' | 'w' => 0b10001_10001_10001_10001_10101_11011_10001,
        'X' | 'x' => 0b10001_10001_01010_00100_01010_10001_10001,
        'Y' | 'y' => 0b10001_10001_01010_00100_00100_00100_00100,
        'Z' | 'z' => 0b11111_00001_00010_00100_01000_10000_11111,
        '_' => 0b00000_00000_00000_00000_00000_00000_11111,
        '-' => 0b00000_00000_00000_11111_00000_00000_00000,
        '.' => 0b00000_00000_00000_00000_00000_00100_00100,
        _ => 0b00000_00100_00000_00100_00000_00100_00000,
    }
}

pub fn write_jpeg(
    path: &Path,
    width: u32,
    height: u32,
    rgb: &[u8],
    icc: &[u8],
    quality: u8,
) -> Result<()> {
    jpeg_export::write_jpeg_with_icc(path, width, height, rgb, icc, quality)
        .with_context(|| format!("Failed to write contact sheet {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_math() {
        let layout = ContactSheetLayout {
            columns: 4,
            cell_long: 100,
            gap: 10,
            caption_height: 10,
            background: [0, 0, 0],
        };
        let size = sheet_size(10, &layout).unwrap();
        assert_eq!(size.rows, 3);
        assert_eq!(size.width, 4 * 100 + 5 * 10);
        assert_eq!(size.height, 3 * 110 + 4 * 10);
    }

    #[test]
    fn compose_fills_background() {
        let cell = ContactSheetCell {
            width: 2,
            height: 2,
            rgb: vec![255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0],
            caption: "A".to_string(),
        };
        let layout = ContactSheetLayout {
            columns: 1,
            cell_long: 4,
            gap: 1,
            caption_height: 8,
            background: [10, 20, 30],
        };
        let (w, h, rgb) = compose(&[cell], &layout).unwrap();
        assert_eq!(w, 6);
        assert_eq!(rgb.len(), (w * h * 3) as usize);
        assert_eq!(&rgb[0..3], &[10, 20, 30]);
    }
}
