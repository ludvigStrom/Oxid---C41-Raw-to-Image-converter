# c41-raw-tool

A high-performance, command-line and GUI RAW image processor for **C-41 color negative film** scanned with a **custom narrowband RGB light source**. The pipeline uses physically accurate log-density math: no auto white balance, no hidden base curves, and no complex color science -- only explicit mathematical steps suitable for scientific and repeatable workflows.

**Target cameras:** Any LibRaw-supported Bayer RAW (Sony, Nikon, Canon, etc.). Initially tuned for Sony a7R II (42MP uncompressed `.arw`). You can also **ingest PNG** (any size) for development or testing; it skips raw/demosaic and runs the same D-min / curve / export pipeline.

---

## Why log-density, not linear inversion?

Film dye density is logarithmic. A simple `1.0 - input` inversion in linear space produces flat results with color cast. Instead, this tool converts transmittance to optical density (`D = -log10(T)`), inverts in the density domain, and applies an RA-4 paper S-curve (Michaelis-Menten). This models a physical darkroom enlarger and produces accurate tonality.

---

## Prerequisites

- **Rust** (2021 edition; install via [rustup](https://rustup.rs/))
- **LibRaw** (used for raw decoding)
  - **macOS:** `brew install libraw`
  - **Debian/Ubuntu:** `sudo apt-get install libraw-dev`
  - **Windows** just give up... package managing sucks on windows.

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

### Minimal GUI

A desktop UI provides three tabs: **Process** (main development with per-step checkboxes for D-min, white balance, print curve, and color calibration profile), **Color calibration** (solve a 3×3 matrix from a ColorChecker and save/load profiles), and **Luminance calibration** (load/save flat-field reference frames). Pick files, set parameters, choose an output folder, and run Convert:

```bash
cargo run --release --bin c41-gui --features gui
```

You need to enable the `gui` feature (adds `eframe` and `rfd`). Same pipeline as the CLI; outputs go to the folder you select.

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
| `--input-dir` | `-i` | Directory containing RAW files (`.arw`, `.nef`, `.nrw`, `.cr2`, `.cr3`, `.crw`, `.dng`, `.raf`, `.orf`, `.rw2`) and/or `.png` files. Others are ignored. |
| `--output-dir` | `-o` | Directory for TIFF output. Created if missing. |
| `--dmin-rect` | -- | D-min crop as `X,Y,WIDTH,HEIGHT` (pixels). Optional. Example: `35,15,20,20`. |
| `--dmin-fixed` | -- | Fixed D-min medians `R,G,B` in linear [0,1]. Bypasses crop measurement. Example: `0.635294,0.635294,0.623529`. |
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
| `--write-exr` | -- | Also write an OpenEXR `.exr` alongside the TIFF (RGB, 32-bit float in [0,1]). |
| `--density-matrix` | -- | 3×3 density-domain calibration matrix in row-major order: `C00,C01,C02,C10,C11,C12,C20,C21,C22`. Defaults to identity (`1,0,0,0,1,0,0,0,1`). Used for **color calibration** (dye crosstalk correction) in the density domain before the RA-4 curve. |
| `--flat-field` | -- | Path to a RAW (or pre-blurred 32f TIFF) of an unexposed, developed frame for **luminance calibration**. When set, flat-field division replaces D-min neutralization. |

When the print curve is used, a **histogram summary** (min, p50, p90, p99, max in 8-bit bins of the u16 output) is printed to the console for tuning.

Output filenames are derived from the input stem: e.g. `frame_001.arw` or `frame_001.png` -> `frame_001.tiff`.

---

## Calibration: color and luminance

The tool supports two kinds of calibration, each with a clear role.

### Color calibration (density-domain matrix)

**What it does:** A 3×3 matrix is applied in the **optical density** domain (after log, before the RA-4 curve). It maps measured densities to reference densities, correcting dye crosstalk and aligning your film/light combination to a known target (e.g. a ColorChecker Classic).

**Philosophy:** C-41 dyes are not perfectly separated; red dye absorbs some green and blue, etc. Measuring 24 patches from a chart and solving OLS gives a matrix that best fits your specific film stock and light source. The correction is explicit, repeatable, and saved as a JSON profile (name, light source notes, matrix, optional D-min medians). In Process mode you can enable or disable this step and choose a profile; when disabled, the identity matrix is used so no color correction is applied.

**Workflow:** Use the **Color calibration** tab: load a RAW of a ColorChecker, align the 4 corner points to the chart, then “Solve 3×3 matrix from chart”. Save the profile to `profiles/` and in Process mode select it from the Color calibration profile dropdown.

### Luminance calibration (flat-field)

**What it does:** You provide a reference image of an **unexposed, developed** frame from the same roll (or same stock). It is linearized and heavily blurred to remove grain and dust, leaving only the low-frequency luminance pattern of your light source and lens. Each scan is then divided by this map pixel-by-pixel.

**Philosophy:** The “empty” frame is not truly empty: it records the orange mask and base density, which is uniform, plus the **illumination** (LED falloff, vignetting). Dividing by this map makes the film base resolve to ~1.0 transmittance everywhere, so exposure and color are no longer biased by where the pixel sits in the frame. This is classic flat-field correction. D-min (single scalar per channel) is a special case when the light is perfectly even; flat-field generalizes to real setups.

**Workflow:** Use the **Luminance calibration** tab: “Load Reference Frame…” (RAW of empty frame). The tool blurs it and keeps it in memory. You can “Save blurred flat-field as 32f TIFF…” to reuse the same camera/lens/light setup without re-blurring. In **Process** mode, under D-min, “Load flat-field map…” lets you choose either a RAW empty frame (linearized and blurred on the fly) or a saved 32f TIFF; when a flat-field is set, it overrides D-min for that session.

---

## Processing pipeline

Order of operations:

1. **Linear extraction** (RAW only) -- LibRaw decodes the raw Bayer plane only (no gamma, no camera WB). Data is normalized to `[0, 1]` as f32.
2. **Demosaic** (RAW only) -- Bilinear interpolation from Bayer to RGB. Pattern: **RGGB** (Sony a7R II). Result: `(height, width, 3)` f32.
   *PNG input skips 1-2: loaded as RGB and normalized to [0, 1].*
3. **D-min or flat-field** (optional, can be disabled) -- Either:
   - **Flat-field:** If `--flat-field` is set, load that reference (RAW → linearize → heavy blur), then divide the image by it pixel-by-pixel. This removes light falloff and vignetting; the film base normalizes to ~1.0 transmittance everywhere. **Or**
   - **D-min:** If `--dmin-fixed` is set, divide by those fixed R,G,B medians; if `--dmin-rect` is set, compute median R,G,B in that rectangle and divide.  
   After this step, data represents **linear transmittance**.
4. **White balance gains** (optional, can be disabled) -- Per-channel multipliers `--wb-r`, `--wb-g`, `--wb-b` (default 1.0). Applied after step 3; compensates narrowband LED intensity imbalance. Same gains can be reused for a given light source.
5. **Density-domain color calibration + physical print curve** (optional, default on) -- Implemented as high-resolution 1D LUTs plus a 3×3 matrix:
   - `D = -log10(T)` -- optical density of the negative (per channel).
   - **Color calibration matrix:** `D_out = M · D_in` where `M` is the 3×3 matrix (from a saved profile or `--density-matrix`). When color calibration is disabled, the identity matrix is used. This step removes dye crosstalk and aligns colors to a reference (e.g. ColorChecker).
   - `logE = D_out + offset` -- density as print exposure (inversion in log domain)
   - `E = 10^logE` -- back to linear exposure
   - `out = E^g / (E^g + pivot^g)` -- RA-4 paper S-curve (Michaelis-Menten), implemented as a dedicated `D → RA-4` LUT over a fixed density range.
   Parameters: `--curve-offset`, `--curve-gamma`, `--curve-pivot`, `--density-matrix`. Then **white point**: if `--curve-white` < 1, output is scaled so that value maps to display white (e.g. 0.745 for 190/255). A **histogram summary** (min, p50, p90, p99, max) is printed for the final u16 image.
6. **Fallback** (`--no-curve`) -- When the curve is off, optionally apply linear `1-x` inversion (`--no-invert` to skip). Export as 32-bit float (default) or 16-bit via `--format`.

In the GUI, **D-min**, **White balance**, **Print curve**, and **Color calibration profile** each have a checkbox: when unchecked, that step is skipped (identity or no-op), so you can isolate the effect of each stage.

---

## Project structure

| Path | Role |
|------|------|
| `src/lib.rs` | Shared pipeline: `PipelineOptions`, `process_files()`. Used by CLI and GUI. |
| `src/main.rs` | CLI (clap), directory iteration, calls lib. |
| `src/bin/c41_gui.rs` | GUI (egui/eframe): Process / Color calibration / Luminance calibration tabs, per-step checkboxes, profile and flat-field load/save, Convert. Requires `--features gui`. |
| `src/raw_reader.rs` | Load RAW via LibRaw (`.arw`, `.nef`, `.nrw`, `.cr2`, `.cr3`, `.crw`, `.dng`, `.raf`, `.orf`, `.rw2`) -> `Array3<f32>` (HxWx1) Bayer. |
| `src/png_reader.rs` | Load `.png` (or other image crate formats) -> RGB `Array3<f32>` (HxWx3); any size. |
| `src/demosaic.rs` | Bayer->RGB bilinear demosaic; supports RGGB, Grbg, Gbrg, Bggr. |
| `src/dmin.rs` | D-min: sample rect, median R/G/B, divide image in-place; supports fixed medians via `--dmin-fixed`. |
| `src/inversion.rs` | Simple linear inversion (`1-x`); used only with `--no-curve`. |
| `src/curve.rs` | Physical Cineon/RA-4 print emulation: multi-stage pipeline (T → density → 3×3 density matrix → RA-4 S-curve) using high-resolution LUTs and rayon-parallel apply. |
| `src/calibration.rs` | Color calibration: ColorChecker reference densities, OLS solver for 3×3 matrix, JSON profile load/save. |
| `src/tiff_export.rs` | Write uncompressed RGB TIFF: 32f/16 from f32, or u16 (after curve) via `write_tiff_u16`. |
| `src/exr_export.rs` | Write RGB OpenEXR: f32 or normalized u16 to EXR via `--write-exr`. |

Dependencies (see `Cargo.toml`): `libraw-rs`, `ndarray`, `rayon`, `clap`, `tiff`, `anyhow`, `image` (PNG/raster ingestion), `exr` (OpenEXR export), `nalgebra` (calibration OLS), `serde`/`serde_json` (profiles).

---

## License

See repository for license information. LibRaw is used under its own license (e.g. LGPL/CDDL).
﻿
---

## Future work (ideas)

- **Color space / ICC profiles**: Today the tool operates in an implicit RGB working space and writes untagged TIFF/EXR. A possible extension is to define an explicit working space (e.g. sRGB, ProPhoto) and embed ICC profiles into TIFFs for stricter color management across viewers.

- **Highlight handling**: `--curve-white` currently acts as a scalar white-point control. A future enhancement could add an optional soft‑clip or smooth highlight roll‑off just before 16‑bit quantization for even more graceful specular handling.