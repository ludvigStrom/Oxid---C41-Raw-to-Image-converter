# Oxid

**Oxid** is a GPU-accelerated application for converting RAW captures of **C-41 color-negative film**, written in Rust. It takes RAW camera captures and turns them into photographs by working in optical density, the space the dyes occupy, so color is treated as the film intended, not flipped as an RGB negative. White balance and film gamma are applied there. An RA-4 paper curve (Hill / Michaelis-Menten) forms the image. No hidden tone curves or auto-adjustments run unless you enable them.

[![Oxid: import, develop, dust removal, and export of a 42 MB Sony ARW C-41 scan](https://img.youtube.com/vi/zQfDPatA1Rs/maxresdefault.jpg)](https://youtu.be/zQfDPatA1Rs)

Import, develop, dust removal, and export of a 42 MB Sony ARW scan of a C-41 negative. [Watch on YouTube](https://youtu.be/zQfDPatA1Rs).

![Oxid GUI: import, develop, dust, and export of a C-41 RAW scan](img/screen%20shot.png)

**Supported cameras:** any `rawloader`-supported Bayer RAW (Sony `.arw`, Nikon `.nef`/`.nrw`, Canon `.cr2`/`.cr3`/`.crw`, Adobe `.dng`, Fujifilm `.raf`, Olympus `.orf`, Panasonic `.rw2`). PNG, JPEG (`.jpg`/`.jpeg`), and TIFF (`.tiff`/`.tif`) input are also accepted and run the same D-min / curve / export pipeline (skips raw decode and demosaic).

Although supported in theory, it is only tested with Sony and Fujifilm cameras. This is a side project with limited resources.

**File types**

| Extension | What it is |
|-----------|------------|
| `.oxidProj` | Multi-image project (image list + per-image develop settings) |
| `.oxid` | Color profile (ZIP of `profile.json` + `lut.cube`) |

Older `.c41proj` / `.c41` files still open. New saves use the Oxid extensions.

---

## Why the pipeline is in density space

Film dye accumulates logarithmically: `D = -log₁₀(T)`. The pipeline converts transmittance to density immediately after D-min, applies WB and film gamma as multiplicative density scales, and inverts in the density domain by passing D directly as log-exposure into the RA-4 curve. The Michaelis-Menten S-curve then models paper exposure:

```
density = -log10(T)
logE    = density + offset
E       = 10^logE
out     = E^gamma / (E^gamma + pivot^gamma)
```

This is the exact function in [`src/curve.rs`](src/curve.rs), computed once into a 65 536-entry f32 LUT (`build_density_to_ra4_lut`). A simple `1 - T` inversion is therefore never needed when the curve is active.

---

## Prerequisites

- **Rust** 2021 edition. Install via [rustup](https://rustup.rs/).
- No system libraries; `rawloader` is pure Rust.
- **Windows:** use the MSVC toolchain, not GNU. GNU builds call `dlltool.exe` (MinGW) and fail if it is not on `PATH`. One-time setup:

```powershell
rustup toolchain install stable-x86_64-pc-windows-msvc
rustup override set stable-x86_64-pc-windows-msvc
```

You also need the MSVC linker from [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (Desktop development with C++).

---

## Build and run

Launch Oxid (GPU-accelerated GUI):

```bash
cargo guigpu
```

This is an alias for `cargo run --release --bin Oxid --features gui,gpu` (see [`.cargo/config.toml`](.cargo/config.toml)). On Windows, if you have not set the directory override above, use `cargo +stable-x86_64-pc-windows-msvc guigpu`.

**Windows installer** (requires [Inno Setup 6](https://jrsoftware.org/isinfo.php); uses the MSVC Rust toolchain so the package does not need MinGW DLLs):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/release_windows_installer.ps1
```

The setup exe is written to `build/dist/Oxid-<version>-setup.exe`. It installs Oxid, adds a Start Menu shortcut, optionally associates `.oxidProj` files, and can create a desktop icon.

On macOS you can also build a signed installer with `scripts/release_pkg_notarize.sh` (see `.env.example`).

**CLI (convert subcommand):**

```bash
cargo run --release -- convert \
  --input-dir  /path/to/scans \
  --output-dir /path/to/output
```

Full example with D-min rect, manual WB, and curve tuning:

```bash
cargo run --release -- convert \
  -i "test files/png" \
  -o "test files/png/output" \
  --dmin-rect 35,15,20,20 \
  --wb-r 1.15 --wb-g 0.88 \
  --curve-offset 0.0 \
  --curve-gamma 2.5 \
  --curve-pivot 3.0 \
  --curve-white 0.745
```

`--curve-white 0.745` maps normalized code 190/255 to display white, pulling in the white point so bright highlights don't clip before the shoulder.

**Validate a single RAW file:**

```bash
cargo run --release -- debug-raw /path/to/file.arw
```

Prints rawloader metadata and sample pixel values without running the full pipeline.

**GUI (CPU only):**

```bash
cargo run --release --bin Oxid --features gui
```

Requires the `gui` feature (adds `eframe`, `rfd`, `arboard`). Prefer `cargo guigpu` for the GPU build. Oxid has three tabs: **Process** (main development with per-step checkboxes, plus **Dust** heal), **Color calibration** (solve a 3×3 density matrix from a ColorChecker), and **Luminance calibration** (load/save flat-field reference frames).

**GPU-accelerated build:**

```bash
cargo run --release --features gpu -- convert \
  --input-dir /path/to/scans --output-dir /path/to/output
```

The `gpu` feature adds `wgpu`, `pollster`, and `bytemuck`. When enabled, pipeline steps 4-6 (T→D / WB / shadow cast, density matrix / 3D LUT / saturation / zones, and the full output stage with post-curve ops) run on the GPU via WGSL compute shaders. A unified pipeline uploads the image once, runs all three steps as consecutive compute dispatches in a single command encoder submission, and reads back the final result once, minimizing PCIe/bus overhead.

The GPU path produces results virtually identical to the CPU reference:

| Step | Max diff (f32) | Max diff (u16) | Notes |
|------|---------------|----------------|-------|
| Step 4 (T→D, WB, shadow cast) | 2.4×10⁻⁷ | - | Hardware `log2` vs software `log10` |
| Step 5 (matrix, LUT, saturation) | 1.2×10⁻⁶ | - | |
| Step 6 (RA-4, FilmPrint, Lut2383) | <1×10⁻⁶ | 0-1 LSB (no Lab) | With Lab: ≤10 LSB due to `pow` precision |
| Unified 4→5→6 end-to-end | - | ≤7 LSB | Compound precision; 0.01% |

If no GPU adapter is available, the pipeline falls back to CPU automatically.

`cargo guigpu` initializes the GPU at startup and adds a **GPU acceleration** checkbox in the Debug tab. Toggling it switches between GPU and CPU paths instantly. The step cache (steps 1-3) remains valid across GPU/CPU switches.

To run the CPU-vs-GPU comparison tests:

```bash
cargo test --features gpu -- --nocapture
```

This runs 25 tests across 4 test suites (`gpu_step4`, `gpu_step5`, `gpu_step6`, `gpu_unified`).

---

## Shortcuts

⌘ is Command on macOS and Ctrl on Windows/Linux. **Ctrl+Shift+D** and **Ctrl+drag** always use the Control key, including on macOS.

### Project and edit

| Shortcut | Action |
|----------|--------|
| ⌘S | Save project |
| ⌘⇧S | Save project as |
| ⌘O | Load project |
| ⌘Z | Undo |
| ⌘⇧Z | Redo |
| Ctrl+Shift+D | Toggle Debug mode |

### Preview

| Shortcut | Action |
|----------|--------|
| Scroll | Zoom toward the pointer (1×-16×) |
| Left-drag | Pan (not while painting dust, or dragging a crop / D-min handle) |
| Middle-drag | Pan |
| Space + left-drag | Pan |

### Dust tab

Only while **Process → Dust** is open.

| Shortcut | Action |
|----------|--------|
| P | Pen |
| E | Eraser |
| Esc | Deselect pen and eraser |
| [ / ] | Smaller / larger brush |
| Ctrl+drag | Change pen / eraser size |
| ⌘D | Disable view (no heal) |
| ⌘E | Edit view (mask overlay) |
| ⌘P | Process view (healed preview) |
| Space | Pan (hand cursor; does not paint) |

---

## Pipeline: exact order of operations

The steps below match the literal execution order in `process_files` and `process_one_to_preview` in `src/lib.rs`. `debug_pipeline_step` (default 6) can stop the pipeline early at any step for inspection.

### Step 1: Load and demosaic

RAW files: `rawloader` decodes the file → single-channel Bayer or X-Trans CFA `Array3<f32>` → `demosaic_quality` → linear RGB.

`demosaic_quality` (in `src/demosaic.rs`) uses:
1. Edge-aware green interpolation (Hamilton-Adams gradient).
2. R and B recovered via **color-difference** (R−G, B−G) interpolation. Interpolating differences rather than raw channel values eliminates false color at edges.

Bayer preview downsampling preserves 2×2 super-pixels so the pattern stays intact after subsampling. X-Trans downsampling uses 6×6 super-pixels for the same reason.

PNG files: loaded as linear RGB, bypassing raw decode and demosaic.

Optional rotation (0°, 90°, 180°, 270°) is applied immediately after demosaic.

### Dust removal (after step 1)

Optional. Runs after load, demosaic, rotation, and IDT, before D-min. Paint on **Process → Dust**; the pen is the hole. Size the brush to the speck. Feather (default 6 px) fades the heal outside the stroke so the rim does not look cut out. Grain (default 1.0) puts film texture back so the fill does not look airbrushed.

![Dust speck marked with the pen tool and healed with PatchMatch plus statistical grain](img/Dust%20Speck.png)

Three infill algorithms (`DustInfill` in [`src/dust.rs`](src/dust.rs)). **PatchMatch with statistical grain is the default and the best result.**

| Infill | What it does |
|--------|--------------|
| `PatchMatch` (default) | Barnes nearest-neighbour field: 7×7 SSD, colour-gated search of nearby film, propagate + random search, coarse-to-fine pyramid, then patch splat. Grain is synthesized (NLF × clump PSD) on the hole and the feather. Match (1-4) is how far and how loosely it may search. |
| `WaveFunction` | Structure-tensor flow: smears the low-pass along the collar direction (or H+V when isotropic). Grain is the same statistical model, hole only. Match controls how strongly local direction is used. GPU path available. |
| `Telea` | Fast-marching Telea on a blurred low-pass, then a high-pass copy of grain from film next to the stroke. Faster, less structure-aware. Grain 1.0 is a 1:1 residual copy. |

Statistical grain ([`src/dust_grain.rs`](src/dust_grain.rs)): edge-preserving NLM on nearby flat, on-colour film → residual → noise-level function (σ vs luma) and a 16×16 power spectrum. Shaped noise is synthesized into the hole. Grain 1.0 matches the measured σ.

Views: **Disable** (no heal), **Edit** (mask overlay), **Process** (healed preview). Implementation: [`src/dust.rs`](src/dust.rs) (Telea + prep), [`src/dust_pm.rs`](src/dust_pm.rs), [`src/dust_wfc.rs`](src/dust_wfc.rs).

### Step 3: D-min / flat-field normalization

Four modes (`DminMode` in `src/lib.rs`):

| Mode | What it does |
|------|--------------|
| `AutoPercentile` (default) | Works in log-density space: finds the 0.5th-percentile density per channel in the inner 80% of the frame (border excluded), converts back to linear, divides image by that value. |
| `SampleRegion` | Samples a user-defined `X,Y,WIDTH,HEIGHT` rect; divides image by per-channel median. |
| `Fixed` | Divides by manually-supplied R,G,B values. |
| `Off` | Skipped entirely. |

When `neutral_only = true`, all three channels are divided by the geometric mean of the medians (`(med_r * med_g * med_b)^(1/3)`), removing density without shifting the orange mask color.

Flat-field override: if `--flat-field` is provided, D-min is replaced by pixel-by-pixel division against a blurred reference frame. RAW flat-field inputs are demosaiced then blurred with a separable f32 Gaussian (σ = 60 px) to remove grain and dust, leaving only low-frequency illumination falloff. Pre-blurred 32f TIFFs are used directly.

### Step 4: T → D → WB → film gamma

All in one pass per pixel after D-min:

```
4a:  D = -log10(T)       clamped T ≥ 1e-10
4b:  auto WB:   D *= mean_D / ch_median_D   (equalize channel medians)
4c:  manual WB: D *= slider                 (default 1.0 per channel)
4d:  film γ:    D *= 1 / film_gamma         (default γ = 0.65 for C-41)
```

Auto WB is multiplicative in density space (not additive), so it preserves `D = 0` as the black point in all channels. No black-point shift occurs. Manual WB and film gamma are folded into the same per-channel scalar for a single pass over the data.

Optional colour temperature (`--temp-k`, in Kelvin): a Kelvin-to-RGB model converts the temperature to additive density offsets that are added per channel.

Optional shadow cast correction (`shadow_cast_strength`): detects per-channel colour imbalance in pixels with mean density below 1.2 and applies a correction weighted by `t^1.5` (strongest at D = 0, zero by threshold).

### Step 5: Density calibration

Either a 3×3 matrix or a `.cube` 3D LUT is applied in the density domain. Both live at the same pipeline slot (after T→D, before the output stage). If the matrix is identity, the pixel loop is skipped.

3×3 density matrix (per pixel):

```
D'_r = m[0][0]*D_r + m[0][1]*D_g + m[0][2]*D_b
D'_g = m[1][0]*D_r + m[1][1]*D_g + m[1][2]*D_b
D'_b = m[2][0]*D_r + m[2][1]*D_g + m[2][2]*D_b
```

After the matrix, `limit_highlight_density_spread` compresses speckle pixels where one channel is an extreme outlier while the other two are close (ratio-based detection, blended 85% toward the mean density).

Optional density-domain saturation boost (`saturation`, default 1.0):

```
D_mean  = (D_r + D_g + D_b) / 3
D'_ch   = D_mean + saturation * (D_ch - D_mean)
```

Optional Gaussian-masked zone adjustments: shadow (`zone_shadows`) and highlight (`zone_highlights`) density offsets. The shadow mask is a Gaussian centred at D = 0.4 (σ² = 0.25); the highlight mask at D = 2.2 (σ² = 0.50). They are additive and channel-neutral: all three channels shift by the same amount.

### Step 6: Output stage

Four output stages selectable via `output_stage`:

| Stage | What it does |
|-------|--------------|
| `Ra4` (default) | 65 536-entry `density → u16` LUT (`build_density_to_ra4_lut`), applied in parallel over rows via rayon. Includes a soft shoulder starting at 0.93: `t_shaped = 1 - (1-t)^1.5` above the knee. |
| `FilmPrint` | Per-channel RA-4 curves with independent offset and gamma. Color bleed mixes adjacent-channel density before the curve. Post-curve luminance-aware vibrance boosts muted colors: `boost = 1 + strength * (1 - chroma)`. |
| `Lut2383` | Density → code value with selectable encoding (Cineon log D/2.046, Rec.709, or linear D/2.5) → user-supplied `.cube` 3D LUT → display-space output. |
| `None` | Direct density display: `D / 2.5` clamped to [0, 1]. |

Post-curve operations applied after `Ra4` and `FilmPrint` (all operate on the u16 output):

- **Toe/shoulder shaping**: smoothstep-masked additive offset; toe mask centered on [0.07, 0.60], shoulder mask on [0.45, 0.95].
- **Soft clip**: exponential highlight roll-off: `v + (1 - exp(-(v-s)/(1-s))) * (1-s)` above knee `s` (default 0.93).
- **Lab separation**: converts sRGB → XYZ → Lab, scales the a/b chroma deviation by a bell-shaped function `1 + strength * c_norm * (1 - c_norm) * 2`. Near-neutral pixels (chroma < 1e-4) are not touched.
- **Highlight warmth**: Noritsu/Frontier-style golden tint on neutral highlights: `+0.035 R, +0.015 G, −0.055 B`, weighted by `smoothstep(0.35, 0.85, luma) * (1 − smoothstep(0.04, 0.18, chroma))`. Saturated colors receive no warmth.

### De-Bujack (after step 6)

Optional,  This is an experimental novel feature that is **off by default**. Runs after the output transform and display-space looks, before grain / sharpen / encode. Skipped when the pipeline stopped before step 6, or when the buffer is still density (`output_stage = None`).

Bujack et al. showed that perceived color difference is not a Riemannian metric: large differences compress (diminishing returns). A pointwise grade cannot undo that. Any pointwise map of a Riemannian metric is still Riemannian, so this pass is spatial.

It works in **OkLab** on linear Rec.709-like RGB (RA-4 / FilmPrint u16 print RGB; Lut2383 Rec.709 is decoded to linear for the pass). Each pixel’s difference from an edge-aware local mean (bilateral) is pushed through the inverse of a saturating response `f(d) = k·d/(k+d)`, stretching large differences while leaving small ones alone. Out-of-gamut results are pulled toward their own luminance.

The paper proves the effect exists, not its numbers, so the knobs are taste. In the GUI they live under **Develop → De-Bujack**:

| Knob | Default | What it does |
|------|---------|--------------|
| Knee L (`bujack_k_l`) | 0.25 | Where lightness differences start to flatten. Smaller = more aggressive. |
| Knee C (`bujack_k_c`) | 0.30 | Same knee on the (a, b) chroma vector. |
| Strength | 0.2 | Dry/wet mix. 1.0 = full inverse-response; above 1.0 over-corrects. |
| Radius | 16 px | Bilateral radius in pixels of the **current** buffer. Preview is smaller than export, so the same number covers more of the frame in preview. |
| Edge preserve | 0.25 | Bilateral range σ. Low keeps edges out of the base (less halo); 1.0 ≈ Gaussian. |

Implementation: [`src/bujack.rs`](src/bujack.rs), called from `pipeline::apply_bujack`.

---

## Output formats

| Format | Flag | Notes |
|--------|------|-------|
| 16-bit TIFF (default with curve) | - | Uncompressed, u16 per channel via `write_tiff_u16`. Embeds an sRGB ICC profile. |
| 32-bit float TIFF | `--format 32f` | Uncompressed, f32 per channel. Only with `--no-curve`. Untagged linear (no ICC). |
| 16-bit integer TIFF | `--format 16` | Clamped and scaled. Only with `--no-curve`. Embeds an sRGB ICC profile. |
| OpenEXR | `--write-exr` | f32 or normalized u16, written via the `exr` crate. |
| JPEG | `--write-jpeg` | 8-bit sRGB OETF via `write_jpeg_srgb`, with an embedded sRGB ICC profile. |
| ACES2065-1 EXR | `--export-aces-exr` | Linear AP0-primaries EXR via `linear_acescg_to_aces2065_1`. |

On macOS the GUI window and the OpenGL backing layer are tagged sRGB when no monitor ICC is available, so the preview matches color-managed viewers of sRGB-tagged files. When a display profile is detected, preview pixels are converted to that profile and the layer is left in the display space. Do not treat the OpenGL framebuffer as an sRGB texture — egui already writes encoded pixels.

### Color management (ICC)

ICC is a **post-step-6 encode**, separate from `.oxid` density-matrix “color profiles”.

| Piece | Where | What it does |
|-------|--------|----------------|
| Output ICC | Export tab / `--output-icc` | Convert linear Rec.709 print RGB to sRGB (default), Display P3, Adobe RGB, or a custom RGB `.icc`, then embed that profile in 16-bit TIFF and JPEG. |
| Monitor ICC | Auto-detected (macOS / Windows) | Preview hops working RGB → display. Falls back to sRGB encode + sRGB window tag. |
| Soft proof | Export tab + preview **Proof** toggle | Simulate a paper/printer RGB ICC on the detected monitor (intent, paper white, optional gamut warning). Export stays RGB — not CMYK. |
| Filename template | Export tab / `--filename-template` | Tokens `{stem}` `{index}` `{index:03}` `{date}` `{time}` `{preset}` `{profile}` `{w}` `{h}`. Default `{stem}` keeps `frame_001.arw` → `frame_001.tiff`. |
| Export preset | Export tab Save/Load | Format, JPEG sidecar/quality, output ICC, intent, filename. Not a Develop look preset. |
| Contact sheet | Export tab | JPEG grid of selected/all frames, same output ICC. |

Default output is still the compact IEC 61966-2.1 sRGB profile, so existing sRGB files stay bit-identical.

Output filenames are derived from the template (default `{stem}`): `frame_001.arw` → `frame_001.tiff`.

---

## Calibration

### Color calibration (density matrix)

Solve a 3×3 matrix from a scan of a **ColorChecker Classic** (24 patches, reference values baked into `src/calibration.rs` as manufacturer sRGB values). OLS is solved via `nalgebra`. The result is stored in a `.oxid` profile (a ZIP containing `profile.json` + `lut.cube`):

```json
{
  "name": "Kodak Gold 200",
  "light_source": "narrowband RGB LED",
  "matrix": [[...], [...], [...]],
  "dmin_medians": [0.635, 0.635, 0.624]
}
```

Profiles are loaded in Process mode; when `apply_color_profile = false` the identity matrix is used (no color correction).

**3D LUT alternative:** the GUI can generate a `.cube` from the current 3×3 matrix using `Lut3d::generate_from_matrix` (default 33³ grid). The 3D LUT sits in the same pipeline slot as the matrix and is applied with tetrahedral interpolation. You can also load an external `.cube` file directly via `--lut3d-path`.

### Luminance calibration (flat-field)

Provide a RAW scan of an **unexposed, developed frame** from the same roll. The pipeline:
1. Demosaics to linear RGB.
2. Applies a separable f32 Gaussian blur at σ = 60 px to remove grain and dust.
3. Divides every pixel of the source image by the blurred map: `T_out = T_in / T_flat`.

The blurred map can be saved as a 32f TIFF and reloaded to skip re-blurring. When a flat-field is active it fully replaces D-min for that session.

---

## CLI reference

Oxid’s command-line tool (crate binary `c41-raw-tool`):

```
c41-raw-tool convert [OPTIONS] --input-dir <PATH> --output-dir <PATH>
```

| Option | Description |
|--------|-------------|
| `-i`, `--input-dir` | Directory of RAW / PNG files. |
| `-o`, `--output-dir` | Output directory (created if missing). |
| `--dmin-rect X,Y,W,H` | Sample D-min from this pixel rectangle. |
| `--dmin-fixed R,G,B` | Fixed D-min medians in linear [0, 1]; bypass measurement. |
| `--format 32f\|16` | TIFF sample format (only with `--no-curve`). |
| `--write-exr` | Also write OpenEXR alongside TIFF. |
| `--no-curve` | Skip RA-4 curve; output is density or linear transmittance. |
| `--no-invert` | Skip `1-x` inversion (only relevant with `--no-curve`). |
| `--wb-r/g/b` | Per-channel density scale factors (default 1.0). |
| `--curve-offset` | Print exposure bias (log-domain shift). Default 0.0. |
| `--curve-gamma` | Paper grade / contrast (0.5-5.0). Default 2.5. |
| `--curve-pivot` | Half-saturation exposure for Michaelis-Menten. Default 3.0. |
| `--curve-white` | Code value [0-1] that maps to display white. Default 1.0. |
| `--density-matrix C00,...,C22` | 3×3 density calibration matrix, row-major (9 values). |
| `--flat-field PATH` | RAW or 32f TIFF flat-field for luminance calibration. |
| `--export-aces-exr` | Write linear ACES2065-1 EXR alongside display output. |
| `--idt-matrix M00,...,M22` | 3×3 IDT matrix (camera linear → working space), row-major. |
| `--output-icc srgb\|display-p3\|adobe-rgb\|PATH` | Output RGB ICC for 16-bit TIFF / JPEG. Default `srgb`. |
| `--export-intent perceptual\|relative\|absolute` | Output ICC rendering intent. Default `relative`. |
| `--no-export-bpc` | Disable black-point compensation on the output hop. |
| `--filename-template STR` | Output name without extension. Default `{stem}`. |

---

## Project structure

| Path | Role |
|------|------|
| `src/lib.rs` | `PipelineOptions`, `process_files`, `process_one_to_preview`. Shared by CLI and GUI. Contains all pipeline logic. |
| `src/main.rs` | CLI (`clap`): `convert` and `debug-raw` subcommands. |
| `src/bin/c41_gui.rs` | Oxid GUI (`eframe`/`egui`): Process / Color calibration / Luminance calibration tabs. Requires `--features gui`. |
| `src/raw_reader.rs` | `rawloader` wrapper → single-channel CFA `Array3<f32>` + `CfaPattern`. |
| `src/png_reader.rs` | `image` crate PNG loader → linear RGB `Array3<f32>`. |
| `src/demosaic.rs` | Bayer and X-Trans demosaic: bilinear (fallback), edge-aware green, and quality (color-difference R/B). |
| `src/dmin.rs` | D-min: rect sampling, fixed medians, auto-percentile, flat-field division. |
| `src/dust.rs` | Dust mask, heal prep, Telea infill, soft-alpha composite. |
| `src/dust_pm.rs` | PatchMatch infill (default): Barnes NN-field + statistical grain. |
| `src/dust_wfc.rs` | Wave-function infill: structure-tensor flow + statistical grain. |
| `src/dust_grain.rs` | Statistical film grain: NLM residual, NLF, 16×16 PSD, shaped noise. |
| `src/curve.rs` | RA-4 Michaelis-Menten curve: 65 536-entry LUT generation, rayon-parallel apply, Film Print variant with per-channel curves and color bleed. |
| `src/calibration.rs` | ColorChecker reference data, OLS 3×3 solver via `nalgebra`, `.oxid` profile load/save (zip). |
| `src/lut3d.rs` | Density-domain 3D LUT: generate from 3×3 matrix, `.cube` file I/O, tetrahedral interpolation. |
| `src/aces.rs` | ACEScg IDT and `linear_acescg_to_aces2065_1` matrix. |
| `src/tiff_export.rs` | Uncompressed TIFF writer: `write_tiff_u16` (u16), `write_tiff` (f32 or u16). 16-bit paths embed sRGB ICC. |
| `src/jpeg_export.rs` | 8-bit JPEG writer with embedded ICC (`write_jpeg_srgb` / `write_jpeg_with_icc`). |
| `src/cms.rs` | LittleCMS post-step-6 ICC: output encode, monitor display, paper soft proof. |
| `src/monitor.rs` | Detect display ICC (macOS / Windows). |
| `src/filename.rs` | Export filename templates + collision suffixes. |
| `src/export_preset.rs` | Export-tab JSON presets (format, ICC, intent, template). |
| `src/contact_sheet.rs` | Contact-sheet grid layout + JPEG write. |
| `src/exr_export.rs` | OpenEXR writer: f32, u16, and ACES2065-1 paths. |
| `src/pipeline.rs` | Shared pipeline steps 3-6 used by both `process_files` and `process_one_to_preview`. |
| `src/bujack.rs` | De-Bujack: non-local OkLab difference stretch after step 6 (optional, off by default). |
| `src/pipeline_cache.rs` | Step-level cache for preview: reuse earlier stages when only later options change. |
| `src/gpu/mod.rs` | wgpu initialization and `GpuContext` (optional, `--features gpu`). |
| `src/gpu/demosaic.rs` | GPU demosaic for RGGB Bayer (edge-aware G + color-diff R/B); X-Trans and non-RGGB fall back to CPU. |
| `src/gpu/demosaic.wgsl` | WGSL 2-pass compute: green interpolation, then R/B from (R-G), (B-G). |
| `src/gpu/flat_field.rs` | GPU flat-field divide: image /= flat. CPU does resize. |
| `src/gpu/flat_field.wgsl` | WGSL: per-pixel divide + clamp. |
| `src/gpu/step3_dmin.rs` | GPU D-min divide: image /= (div_r, div_g, div_b). CPU does rect/percentile. |
| `src/gpu/step3_dmin.wgsl` | WGSL: per-channel divide + clamp. |
| `src/gpu/unified.rs` | Unified GPU pipeline: steps 3-6; Step3Gpu holds flat_field + step3_dmin. |
| `src/gpu/dust_wfc.rs` | GPU wave-function dust heal (structure-flow + residual). |
| `src/gpu/dust_wfc.wgsl` | WGSL coherent exemplar copy for wave-function fill. |
| `src/gpu/step4.rs` | GPU dispatch for step 4: T→D, WB, shadow cast. CPU precomputes auto-WB medians and shadow analysis. |
| `src/gpu/step4.wgsl` | WGSL compute shader: T→D via hardware `log2`, per-channel scale+offset, shadow cast correction. |
| `src/gpu/step5.rs` | GPU dispatch for step 5: density matrix, 3D LUT, saturation, zones. |
| `src/gpu/step5.wgsl` | WGSL compute shader: matrix multiply, tetrahedral 3D LUT interpolation, highlight spread, saturation, zones. |
| `src/gpu/step6.rs` | GPU dispatch for step 6: all output stages (RA-4, FilmPrint, Lut2383, None) with post-curve ops. |
| `src/gpu/step6.wgsl` | WGSL compute shader: 1D/3D LUT lookup, toe/shoulder, soft knee, Lab separation, highlight warmth. |
| `src/inversion.rs` | Simple `1-x` linear inversion used only with `--no-curve`. |

---

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `rawloader` | 0.37 | Pure-Rust RAW decode (Bayer + X-Trans) |
| `ndarray` | 0.17 | `Array3<f32>` image volumes; `rayon` feature |
| `rayon` | 1.10 | Row-parallel curve and demosaic passes |
| `clap` | 4.5 | CLI argument parsing (derive) |
| `tiff` | 0.11 | Uncompressed TIFF output |
| `image` | 0.25 | PNG ingestion + resize |
| `exr` | 1.74 | OpenEXR output |
| `nalgebra` | 0.34 | OLS solver for 3×3 calibration matrix |
| `serde` / `serde_json` | 1.0 | Calibration profile JSON |
| `zip` | 2.2 | `.oxid` profile format (zip of JSON + LUT) |
| `lcms2` | 6 | ICC CMM (output profiles, soft proof, embed) |
| `eframe` | 0.29 | GUI (optional, `--features gui`) |
| `rfd` | 0.15 | Native file dialogs (optional) |
| `arboard` | 3 | Clipboard (optional) |
| `wgpu` | 24 | GPU compute pipeline (optional, `--features gpu`) |
| `pollster` | 0.4 | Blocking async executor for wgpu init (optional) |
| `bytemuck` | 1 | Safe byte casting for GPU buffer upload (optional) |

---

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. See [LICENSE](LICENSE).

## Credits

Oxid borrows some parts from:

- **[Negadoctor](https://github.com/darktable-org/darktable/blob/master/src/iop/negadoctor.c)**: darktable’s film-negative inversion module, © darktable developers, GPL-3.0-or-later.

The auto crop function is heavily inspired by:
- **[NegPy](https://github.com/marcinz606/NegPy)** by Marcin Zawalski, GPL-3.0.
