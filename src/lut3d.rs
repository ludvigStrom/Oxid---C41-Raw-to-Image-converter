//! 3D LUT in the density domain: generate from 3×3 matrix, .cube read/write, tetrahedral interpolation.
//!
//! The LUT is defined on normalized density [0, 1]³ (interpreted as density [0, d_max] when applying).
//! Pipeline slot: after T→D, before D→RA-4 (same as the density matrix).

use std::path::Path;

use ndarray::Array3;

/// 3D LUT for density-domain color correction.
/// Grid has `size`³ vertices; input/output are normalized [0, 1] (scale by `d_max` for density).
#[derive(Debug, Clone)]
pub struct Lut3d {
    pub size: usize,
    /// Density range used when applying: normalized value v corresponds to density v * d_max.
    pub d_max: f32,
    /// Output RGB at each grid vertex, normalized [0, 1]. Index: r + g*size + b*size² (red major).
    data: Vec<[f32; 3]>,
}

impl Lut3d {
    /// Create a LUT by evaluating the 3×3 matrix at each grid vertex.
    /// Input density at vertex (i,j,k) is (i,j,k)/(size-1) * d_max; output = matrix * input, normalized to [0,1].
    pub fn generate_from_matrix(matrix: &[[f32; 3]; 3], size: usize, d_max: f32) -> Self {
        let n = size;
        let mut data = Vec::with_capacity(n * n * n);
        let scale = if n > 1 { (n - 1) as f32 } else { 1.0 };

        for b in 0..n {
            for g in 0..n {
                for r in 0..n {
                    let nr = r as f32 / scale;
                    let ng = g as f32 / scale;
                    let nb = b as f32 / scale;
                    let d_in = [nr * d_max, ng * d_max, nb * d_max];
                    let d_out = [
                        matrix[0][0] * d_in[0] + matrix[0][1] * d_in[1] + matrix[0][2] * d_in[2],
                        matrix[1][0] * d_in[0] + matrix[1][1] * d_in[1] + matrix[1][2] * d_in[2],
                        matrix[2][0] * d_in[0] + matrix[2][1] * d_in[1] + matrix[2][2] * d_in[2],
                    ];
                    let out_norm = [
                        (d_out[0] / d_max).clamp(0.0, 1.0),
                        (d_out[1] / d_max).clamp(0.0, 1.0),
                        (d_out[2] / d_max).clamp(0.0, 1.0),
                    ];
                    data.push(out_norm);
                }
            }
        }

        Self {
            size: n,
            d_max,
            data,
        }
    }

    #[inline]
    fn index(&self, r: usize, g: usize, b: usize) -> [f32; 3] {
        self.data[r + g * self.size + b * self.size * self.size]
    }

    /// Read-only access to the raw LUT data for GPU upload.
    pub fn data_slice(&self) -> &[[f32; 3]] {
        &self.data
    }

    /// Sample the LUT at normalized coordinates (r, g, b) in [0, 1]³ using tetrahedral interpolation.
    /// Returns normalized output [0, 1]³; multiply by d_max to get density.
    #[inline]
    pub fn sample_normalized(&self, r: f32, g: f32, b: f32) -> [f32; 3] {
        let n = self.size as f32;
        let r = r.clamp(0.0, 1.0) * (n - 1.0);
        let g = g.clamp(0.0, 1.0) * (n - 1.0);
        let b = b.clamp(0.0, 1.0) * (n - 1.0);

        let r0 = r.floor() as usize;
        let g0 = g.floor() as usize;
        let b0 = b.floor() as usize;
        let r1 = (r0 + 1).min(self.size - 1);
        let g1 = (g0 + 1).min(self.size - 1);
        let b1 = (b0 + 1).min(self.size - 1);

        let fr = r - r0 as f32;
        let fg = g - g0 as f32;
        let fb = b - b0 as f32;

        // Tetrahedral interpolation: choose one of 6 tetrahedra based on order of fr, fg, fb.
        let (v0, v1, v2, v3, d1, d2, d3) = if fr >= fg && fg >= fb {
            (
                self.index(r0, g0, b0),
                self.index(r1, g0, b0),
                self.index(r1, g1, b0),
                self.index(r1, g1, b1),
                fr - fg,
                fg - fb,
                fb,
            )
        } else if fr >= fb && fb >= fg {
            (
                self.index(r0, g0, b0),
                self.index(r1, g0, b0),
                self.index(r1, g0, b1),
                self.index(r1, g1, b1),
                fr - fb,
                fb - fg,
                fg,
            )
        } else if fg >= fr && fr >= fb {
            (
                self.index(r0, g0, b0),
                self.index(r0, g1, b0),
                self.index(r1, g1, b0),
                self.index(r1, g1, b1),
                fg - fr,
                fr - fb,
                fb,
            )
        } else if fg >= fb && fb >= fr {
            (
                self.index(r0, g0, b0),
                self.index(r0, g1, b0),
                self.index(r0, g1, b1),
                self.index(r1, g1, b1),
                fg - fb,
                fb - fr,
                fr,
            )
        } else if fb >= fr && fr >= fg {
            (
                self.index(r0, g0, b0),
                self.index(r0, g0, b1),
                self.index(r1, g0, b1),
                self.index(r1, g1, b1),
                fb - fr,
                fr - fg,
                fg,
            )
        } else {
            // fb >= fg && fg >= fr
            (
                self.index(r0, g0, b0),
                self.index(r0, g0, b1),
                self.index(r0, g1, b1),
                self.index(r1, g1, b1),
                fb - fg,
                fg - fr,
                fr,
            )
        };

        [
            v0[0] + d1 * (v1[0] - v0[0]) + d2 * (v2[0] - v0[0]) + d3 * (v3[0] - v0[0]),
            v0[1] + d1 * (v1[1] - v0[1]) + d2 * (v2[1] - v0[1]) + d3 * (v3[1] - v0[1]),
            v0[2] + d1 * (v1[2] - v0[2]) + d2 * (v2[2] - v0[2]) + d3 * (v3[2] - v0[2]),
        ]
    }

    /// Apply LUT to a density triplet: input density (d_r, d_g, d_b) -> output density.
    /// Input is clamped to [0, d_max] before lookup.
    #[inline]
    pub fn sample_density(&self, d_r: f32, d_g: f32, d_b: f32) -> [f32; 3] {
        let norm_r = (d_r / self.d_max).clamp(0.0, 1.0);
        let norm_g = (d_g / self.d_max).clamp(0.0, 1.0);
        let norm_b = (d_b / self.d_max).clamp(0.0, 1.0);
        let out = self.sample_normalized(norm_r, norm_g, norm_b);
        [
            out[0] * self.d_max,
            out[1] * self.d_max,
            out[2] * self.d_max,
        ]
    }
}

/// Return the .cube file content as a string (for embedding in .oxid zip).
pub fn cube_to_string(lut: &Lut3d) -> String {
    let mut out = String::new();
    out.push_str("# Oxid 3D LUT (density domain, normalized 0..1)\n");
    out.push_str(&format!("LUT_3D_SIZE {}\n", lut.size));
    for tri in &lut.data {
        out.push_str(&format!("{} {} {}\n", tri[0], tri[1], tri[2]));
    }
    out
}

/// Write a 3D LUT to a .cube file (Adobe CUBE format).
/// Grid is normalized [0, 1] input and output; one line per vertex, red-major order.
pub fn write_cube(lut: &Lut3d, path: &Path) -> std::io::Result<()> {
    std::fs::write(path, cube_to_string(lut))
}

/// Read a .cube file (Adobe/Resolve CUBE format).
/// Supports headers: `TITLE`, `LUT_3D_SIZE`, `DOMAIN_MIN`, `DOMAIN_MAX`, `LUT_1D_SIZE`,
/// and comment lines starting with `#`. Unknown keyword lines are skipped.
/// Data must be red-major (R fastest).
pub fn read_cube(path: &Path) -> anyhow::Result<Lut3d> {
    let text = std::fs::read_to_string(path)?;
    let mut size: Option<usize> = None;
    let mut d_max = 4.0_f32; // default density range for our pipeline
    let mut data = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        // Known header keywords — consume and continue.
        match parts[0] {
            "LUT_3D_SIZE" => {
                if parts.len() >= 2 {
                    size = parts[1].parse().ok();
                }
                continue;
            }
            "DOMAIN_MAX" if parts.len() >= 4 => {
                let r: f32 = parts[1].parse().unwrap_or(1.0);
                let g: f32 = parts[2].parse().unwrap_or(1.0);
                let b: f32 = parts[3].parse().unwrap_or(1.0);
                d_max = r.max(g).max(b).max(1e-6);
                continue;
            }
            "TITLE" | "DOMAIN_MIN" | "LUT_1D_SIZE" | "LUT_1D_INPUT_RANGE"
            | "LUT_3D_INPUT_RANGE" => {
                continue;
            }
            _ => {}
        }

        // Skip any line whose first token is not a number (unknown keyword).
        if parts[0].parse::<f32>().is_err() {
            continue;
        }

        // Data line: three floats.
        if parts.len() >= 3 {
            let r: f32 = parts[0]
                .parse()
                .map_err(|_| anyhow::anyhow!("Invalid number in .cube data"))?;
            let g: f32 = parts[1]
                .parse()
                .map_err(|_| anyhow::anyhow!("Invalid number in .cube data"))?;
            let b: f32 = parts[2]
                .parse()
                .map_err(|_| anyhow::anyhow!("Invalid number in .cube data"))?;
            data.push([r, g, b]);
        }
    }

    let size = size.ok_or_else(|| anyhow::anyhow!("LUT_3D_SIZE not found in .cube file"))?;
    let expected = size * size * size;
    if data.len() != expected {
        anyhow::bail!(
            ".cube has {} entries, expected {} for size {}",
            data.len(),
            expected,
            size
        );
    }

    Ok(Lut3d { size, d_max, data })
}

/// Apply a 3D LUT to an image in density domain (in place).
/// Image values are interpreted as density; they are scaled by d_max for lookup then written back as density.
pub fn apply_lut3d_to_image(image: &mut Array3<f32>, lut: &Lut3d) {
    let (h, w, _) = image.dim();
    for y in 0..h {
        for x in 0..w {
            let d_r = image[[y, x, 0]];
            let d_g = image[[y, x, 1]];
            let d_b = image[[y, x, 2]];
            let out = lut.sample_density(d_r, d_g, d_b);
            image[[y, x, 0]] = out[0];
            image[[y, x, 1]] = out[1];
            image[[y, x, 2]] = out[2];
        }
    }
}
