# c41-raw-tool

A high-performance, command-line RAW image processor for **C-41 color negative film** scanned with a **custom narrowband RGB light source**. The pipeline is strictly linear: no auto white balance, no hidden base curves, and no complex color science—only explicit mathematical steps suitable for scientific and repeatable workflows.

**Target camera:** Sony a7R II (42MP uncompressed `.arw`).

---

## Why linear-only?

With narrowband RGB illumination, the cyan, magenta, and yellow dye layers are physically separated with minimal crosstalk. The goal is to preserve that separation and work in sensor/linear space until you choose explicit steps (D-min, inversion, tone curve). This tool does not apply camera profiles, creative color grading, or “smart” corrections.

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

Optional D-min region (unexposed film border) and 16-bit output for display:

```bash
cargo run --release -- \
  --input-dir "test files/raw" \
  --output-dir "test files/raw/output" \
  --dmin-rect 0,0,200,200 \
  --format 16
```

---

## Output format: keeping as much data as possible

Output is always **uncompressed** TIFF. You choose the sample format:

| Format | Flag | What it does | Use when |
|--------|------|----------------|----------|
| **32-bit float** | `--format 32f` (default) | Writes f32 directly. No clamping, no quantization. Values &gt;1 (e.g. after D-min) are preserved. | Archival, further linear processing, or when you want to keep the full pipeline result. |
| **16-bit integer** | `--format 16` | Clamps to [0, 1], then scales to 0–65535. Values &gt;1 are clipped; precision in shadows is reduced. | Viewing, printing, or when you need maximum compatibility with other software. |

**Recommendation:** Use the default `32f` to preserve all data. Use `16` only when you need a smaller file or 16-bit-only workflows.

---

## CLI reference

| Option | Short | Description |
|--------|-------|-------------|
| `--input-dir` | `-i` | Directory containing Sony `.arw` files. Only `.arw` is processed. |
| `--output-dir` | `-o` | Directory for TIFF output. Created if missing. |
| `--dmin-rect` | — | D-min crop as `X,Y,WIDTH,HEIGHT` (pixels). Optional. Example: `50,50,200,200`. |
| `--format` | — | `32f` (float, default) or `16` (integer). See “Output format” above. |

Output filenames are derived from the input: e.g. `frame_001.arw` → `frame_001.tiff` in the output directory.

---

## Processing pipeline

Order of operations:

1. **Linear extraction** — LibRaw decodes the raw Bayer plane only (no gamma, no camera WB). Data is normalized to `[0, 1]` as f32.
2. **Demosaic** — Bilinear interpolation from Bayer to RGB. Pattern: **RGGB** (Sony a7R II). Result: `(height, width, 3)` f32.
3. **D-min neutralization** (optional) — If `--dmin-rect` is set, the median R, G, and B in that rectangle are computed. The entire image is then divided by these three values (per channel). Use a region on the unexposed film border.
4. **TIFF export** — Uncompressed RGB TIFF. By default **32-bit float** (no clamping/quantization). Option `--format 16` writes 16-bit integer (clamp to [0,1], scale to 0–65535).

Planned (not yet implemented):

- **Inversion** — `output = 1.0 - input` for negative→positive.
- **Universal tone curve** — 1D LUT or spline (e.g. RA-4 / Cineon-style) for viewable contrast.

---

## Project structure

| Path | Role |
|------|------|
| `src/main.rs` | CLI (clap), directory iteration, pipeline orchestration. |
| `src/raw_reader.rs` | Load `.arw` via LibRaw raw decode → `Array3<f32>` (H×W×1). |
| `src/demosaic.rs` | Bayer→RGB bilinear demosaic; supports RGGB, Grbg, Gbrg, Bggr. |
| `src/dmin.rs` | D-min: sample rect, median R/G/B, divide image in-place. |
| `src/tiff_export.rs` | Write uncompressed RGB TIFF: 32-bit float (default) or 16-bit integer from f32 image. |

Dependencies (see `Cargo.toml`): `libraw-rs`, `ndarray`, `rayon`, `clap`, `tiff`, `anyhow`.

---

## License

See repository for license information. LibRaw is used under its own license (e.g. LGPL/CDDL).
