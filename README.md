# c41-raw-tool

A high-performance, command-line RAW image processor for **C-41 color negative film** scanned with a **custom narrowband RGB light source**. The pipeline uses physically accurate log-density math: no auto white balance, no hidden base curves, and no complex color science -- only explicit mathematical steps suitable for scientific and repeatable workflows.

**Target camera:** Sony a7R II (42MP uncompressed `.arw`). You can also **ingest PNG** (any size) for development or testing; it skips raw/demosaic and runs the same D-min / curve / export pipeline.

---

## Why log-density, not linear inversion?

Film dye density is logarithmic. A simple `1.0 - input` inversion in linear space produces flat results with color cast. Instead, this tool converts transmittance to optical density (`D = -log10(T)`), inverts in the density domain, and applies an RA-4 paper S-curve (Michaelis-Menten). This models a physical darkroom enlarger and produces accurate tonality.

---

## Prerequisites

- **Rust** (2021 edition; install via [rustup](https://rustup.rs/))
- **LibRaw** (used for raw decoding)
  - **macOS:** `brew install libraw`
  - **Debian/Ubuntu:** `sudo apt-get install libraw-dev`

---

## Build and run

```bash
# From the project root (directory containing Cargo.toml)
cargo build --release
```

Run with required input and output directories:

```bash
cargo run --release -- --input-dir /path/to/arw/folder --output-dir /path/to/output
```

Full example with D-min, white balance, and curve tuning:

```bash
cargo run --release -- \
  -i "test files/png" \
  -o "test files/png/output" \
  --dmin-rect 35,15,20,20 \
  --wb-r 1.15 --wb-g 0.88 \
  --curve-offset 0.0 \
  --curve-gamma 2.5 \
  --curve-pivot 3.0 \
  --curve-white 0.745
```

`--curve-white 0.745` (190/255) pulls the white point in so highlights don’t blow; a 256-bin histogram summary is printed after the curve for inspection.

---

## Output format: keeping as much data as possible

Output is always **uncompressed** TIFF. You choose the sample format (applies when `--no-curve`):

| Format | Flag | What it does | Use when |
|--------|------|----------------|----------|
| **32-bit float** | `--format 32f` (default) | Writes f32 directly. No clamping, no quantization. | Archival, further linear processing. |
| **16-bit integer** | `--format 16` | Clamps to [0, 1], scales to 0-65535. | Viewing, printing, compatibility. |

When the print curve is active (default), output is always 16-bit (the LUT produces u16).

---

## CLI reference

| Option | Short | Description |
|--------|-------|-------------|
| `--input-dir` | `-i` | Directory containing `.arw` (RAW) and/or `.png` files. Other extensions are ignored. |
| `--output-dir` | `-o` | Directory for TIFF output. Created if missing. |
| `--dmin-rect` | -- | D-min crop as `X,Y,WIDTH,HEIGHT` (pixels). Optional. Example: `35,15,20,20`. |
| `--format` | -- | `32f` (float, default) or `16` (integer). Only used with `--no-curve`. |
| `--no-invert` | -- | Skip linear `1-x` inversion (only applies with `--no-curve`; the print curve inverts in log domain). |
| `--no-curve` | -- | Skip physical print curve; output stays as linear transmittance. |
| `--wb-r` | -- | Red channel gain (after D-min). Default 1.0. Compensates narrowband LED imbalance. |
| `--wb-g` | -- | Green channel gain (after D-min). Default 1.0. |
| `--wb-b` | -- | Blue channel gain (after D-min). Default 1.0. |
| `--curve-offset` | -- | Print exposure bias (log-domain shift). Default 0.0. Higher = brighter print. |
| `--curve-gamma` | -- | Paper grade / contrast. Default 2.5. Higher = harder paper. Range 0.5-5.0. |
| `--curve-pivot` | -- | Half-saturation exposure for RA-4 S-curve. Default 3.0. |
| `--curve-white` | -- | Normalized code that maps to display white (0–1). Default 1.0. Use e.g. 0.745 (190/255) to pull white in. |

When the print curve is used, a **histogram summary** (min, p50, p90, p99, max in 8-bit bins of the u16 output) is printed to the console for tuning.

Output filenames are derived from the input stem: e.g. `frame_001.arw` or `frame_001.png` -> `frame_001.tiff`.

---

## Processing pipeline

Order of operations:

1. **Linear extraction** (ARW only) -- LibRaw decodes the raw Bayer plane only (no gamma, no camera WB). Data is normalized to `[0, 1]` as f32.
2. **Demosaic** (ARW only) -- Bilinear interpolation from Bayer to RGB. Pattern: **RGGB** (Sony a7R II). Result: `(height, width, 3)` f32.
   *PNG input skips 1-2: loaded as RGB and normalized to [0, 1].*
3. **D-min neutralization** (optional) -- If `--dmin-rect` is set, the median R, G, B in that rectangle are computed. The entire image is divided by those values per channel. After this, data represents **linear transmittance**.
4. **White balance gains** (optional) -- Per-channel multipliers `--wb-r`, `--wb-g`, `--wb-b` (default 1.0). Applied after D-min; compensates narrowband LED intensity imbalance (e.g. increase red, decrease green). Same gains can be reused for a given light source.
5. **Physical print film curve** (default on) -- 65 536-entry 1D LUT:
   - `D = -log10(T)` -- optical density of the negative
   - `logE = D + offset` -- density as print exposure (inversion in log domain)
   - `E = 10^logE` -- back to linear exposure
   - `out = E^g / (E^g + pivot^g)` -- RA-4 paper S-curve (Michaelis-Menten)
   Parameters: `--curve-offset`, `--curve-gamma`, `--curve-pivot`. Then **white point**: if `--curve-white` &lt; 1, output is scaled so that value maps to display white (e.g. 0.745 for 190/255). A **histogram summary** (min, p50, p90, p99, max) is printed for the final u16 image.
6. **Fallback** (`--no-curve`) -- When the curve is off, optionally apply linear `1-x` inversion (`--no-invert` to skip). Export as 32-bit float (default) or 16-bit via `--format`.

---

## Project structure

| Path | Role |
|------|------|
| `src/main.rs` | CLI (clap), directory iteration, pipeline orchestration. |
| `src/raw_reader.rs` | Load `.arw` via LibRaw raw decode -> `Array3<f32>` (HxWx1). |
| `src/png_reader.rs` | Load `.png` (or other image crate formats) -> RGB `Array3<f32>` (HxWx3); any size. |
| `src/demosaic.rs` | Bayer->RGB bilinear demosaic; supports RGGB, Grbg, Gbrg, Bggr. |
| `src/dmin.rs` | D-min: sample rect, median R/G/B, divide image in-place. |
| `src/inversion.rs` | Simple linear inversion (`1-x`); used only with `--no-curve`. |
| `src/curve.rs` | Physical Cineon/RA-4 print emulation: log-density inversion + Michaelis-Menten S-curve, 65 536-entry LUT, parallel apply. |
| `src/tiff_export.rs` | Write uncompressed RGB TIFF: 32f/16 from f32, or u16 (after curve) via `write_tiff_u16`. |

Dependencies (see `Cargo.toml`): `libraw-rs`, `ndarray`, `rayon`, `clap`, `tiff`, `anyhow`, `image` (PNG/raster ingestion).

---

## License

See repository for license information. LibRaw is used under its own license (e.g. LGPL/CDDL).
