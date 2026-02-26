# c41-raw-tool

A high-performance, command-line RAW image processor for **C-41 color negative film** scanned with a **custom narrowband RGB light source**. The pipeline uses physically accurate log-density math: no auto white balance, no hidden base curves, and no complex color science -- only explicit mathematical steps suitable for scientific and repeatable workflows.

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

A small desktop UI lets you pick files, set all parameters with sliders and checkboxes, choose an output folder, and run Convert:

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
| `--density-matrix` | -- | 3×3 density-domain calibration matrix in row-major order: `C00,C01,C02,C10,C11,C12,C20,C21,C22`. Defaults to identity (`1,0,0,0,1,0,0,0,1`). Used to remove dye crosstalk in the **density** domain before the RA-4 curve. |

When the print curve is used, a **histogram summary** (min, p50, p90, p99, max in 8-bit bins of the u16 output) is printed to the console for tuning.

Output filenames are derived from the input stem: e.g. `frame_001.arw` or `frame_001.png` -> `frame_001.tiff`.

---

## Processing pipeline

Order of operations:

1. **Linear extraction** (RAW only) -- LibRaw decodes the raw Bayer plane only (no gamma, no camera WB). Data is normalized to `[0, 1]` as f32.
2. **Demosaic** (RAW only) -- Bilinear interpolation from Bayer to RGB. Pattern: **RGGB** (Sony a7R II). Result: `(height, width, 3)` f32.
   *PNG input skips 1-2: loaded as RGB and normalized to [0, 1].*
3. **D-min neutralization** (optional) -- Either:
   - If `--dmin-fixed` is set, use those fixed medians R,G,B (previously measured once) and divide the entire image per channel, **or**
   - If `--dmin-rect` is set, compute median R,G,B in that rectangle and divide the entire image per channel.\n   After this, data represents **linear transmittance**.
4. **White balance gains** (optional) -- Per-channel multipliers `--wb-r`, `--wb-g`, `--wb-b` (default 1.0). Applied after D-min; compensates narrowband LED intensity imbalance (e.g. increase red, decrease green). Same gains can be reused for a given light source.
5. **Density-domain calibration + physical print film curve** (default on) -- implemented as high-resolution 1D LUTs plus a 3×3 matrix:
   - `D = -log10(T)` -- optical density of the negative (per channel), computed either directly or via a `T → D` LUT.
   - **Density matrix:** `D_out = M · D_in` where `M` is a 3×3 matrix supplied via `--density-matrix`. This operates strictly in the density domain to remove dye crosstalk.
   - `logE = D_out + offset` -- density as print exposure (inversion in log domain)
   - `E = 10^logE` -- back to linear exposure
   - `out = E^g / (E^g + pivot^g)` -- RA-4 paper S-curve (Michaelis-Menten), implemented as a dedicated `D → RA-4` LUT over a fixed density range.
   Parameters: `--curve-offset`, `--curve-gamma`, `--curve-pivot`, `--density-matrix`. Then **white point**: if `--curve-white` < 1, output is scaled so that value maps to display white (e.g. 0.745 for 190/255). A **histogram summary** (min, p50, p90, p99, max) is printed for the final u16 image.
6. **Fallback** (`--no-curve`) -- When the curve is off, optionally apply linear `1-x` inversion (`--no-invert` to skip). Export as 32-bit float (default) or 16-bit via `--format`.

---

## Project structure

| Path | Role |
|------|------|
| `src/lib.rs` | Shared pipeline: `PipelineOptions`, `process_files()`. Used by CLI and GUI. |
| `src/main.rs` | CLI (clap), directory iteration, calls lib. |
| `src/bin/c41_gui.rs` | Minimal GUI (egui/eframe): file picker, sliders/checkboxes, Convert. Requires `--features gui`. |
| `src/raw_reader.rs` | Load RAW via LibRaw (`.arw`, `.nef`, `.nrw`, `.cr2`, `.cr3`, `.crw`, `.dng`, `.raf`, `.orf`, `.rw2`) -> `Array3<f32>` (HxWx1) Bayer. |
| `src/png_reader.rs` | Load `.png` (or other image crate formats) -> RGB `Array3<f32>` (HxWx3); any size. |
| `src/demosaic.rs` | Bayer->RGB bilinear demosaic; supports RGGB, Grbg, Gbrg, Bggr. |
| `src/dmin.rs` | D-min: sample rect, median R/G/B, divide image in-place; supports fixed medians via `--dmin-fixed`. |
| `src/inversion.rs` | Simple linear inversion (`1-x`); used only with `--no-curve`. |
| `src/curve.rs` | Physical Cineon/RA-4 print emulation: multi-stage pipeline (T → density → 3×3 density matrix → RA-4 S-curve) using high-resolution LUTs and rayon-parallel apply. |
| `src/tiff_export.rs` | Write uncompressed RGB TIFF: 32f/16 from f32, or u16 (after curve) via `write_tiff_u16`. |
| `src/exr_export.rs` | Write RGB OpenEXR: f32 or normalized u16 to EXR via `--write-exr`. |

Dependencies (see `Cargo.toml`): `libraw-rs`, `ndarray`, `rayon`, `clap`, `tiff`, `anyhow`, `image` (PNG/raster ingestion), `exr` (OpenEXR export).

---

## License

See repository for license information. LibRaw is used under its own license (e.g. LGPL/CDDL).


##TODO 

TODO calibration 

Phase 1: GUI Updates (The Calibration Tab)You need a dedicated workspace in your GUI so the user isn't accidentally trying to process standard photos with calibration tools.

TODO 1.1: Mode Toggle. Add a segmented button or tab system at the top of the UI to switch between [ Process Mode ] and [ Calibrate Mode ].

TODO 1.2: Reference Data Setup. Hardcode the 24 linear RGB (or CIELAB converted to linear RGB) reference values of the ColorChecker Classic into your codebase as a constant $24 \times 3$ array.

TODO 1.3: Image Canvas. In Calibrate Mode, render the loaded RAW image to an egui texture so the user can see what they are clicking on. (You can use a fast, low-res demosaic for the preview to keep the UI snappy).

Phase 2: The Interactive 24-Patch GridDrawing 24 individual draggable squares is a UI nightmare. The standard way to do this is to draw a grid controlled by 4 corner points.

TODO 2.1: 4-Point Draggable Overlay. Create 4 draggable anchor points in egui corresponding to the 4 corners of the ColorChecker.

TODO 2.2: Grid Interpolation. Write a function that takes those 4 points and mathematically interpolates the center coordinates of the 24 patches (a $6 \times 4$ grid).

TODO 2.3: Patch Bounding Boxes. Calculate a small, fixed-size bounding box (e.g., $10 \times 10$ pixels) around each of the 24 interpolated center points. Draw these boxes on the UI overlay so the user can verify they are only sampling pure color, not the black borders of the chart.

Phase 3: Data Extraction & Pipeline TapYou need to run the image through your pipeline, but stop halfway. The math must be done on the neutralized data.
TODO 3.1: D-min & Linearize. Run the RAW image through 
Step 1 (LibRaw), 
Step 2 (Demosaic), and 
Step 3 (D-min neutralization). 

You now have pure Linear Transmittance.

TODO 3.2: Sample the Patches. For each of the 24 bounding boxes, calculate the median RGB values from the Linear Transmittance array.

TODO 3.3: Convert to Density. Convert these 24 median RGB values into Optical Density: $D = -\log_{10}(T)$. Store this as your Measured_X array ($24 \times 3$). Convert your hardcoded reference values into density to create your Reference_Y array.

Phase 4: The Least Squares Solver 
TODO 4.1: Include a Linear Algebra Crate. Add ndarray-linalg or nalgebra to your Cargo.toml.

TODO 4.2: Implement OLS. Write a function that computes the Ordinary Least Squares equation:$$M = (X^T X)^{-1} X^T Y$$(This calculates the $3 \times 3$ matrix $M$ that best maps the measured density $X$ to the reference density $Y$).

TODO 4.3: Calculate Error. Compute the Mean Squared Error (MSE) of the result to display in the UI. If the MSE is huge, the user probably put the grid on upside down!

Phase 5: JSON Export & Profile Management
TODO 5.1: Define the Schema. Create a Rust struct that derives serde::Serialize and serde::Deserialize. It should include:Profile Name / Film Stock (String)Light Source Notes (String)Matrix (9 floats)Optional: The D-min RGB medians used during calibration (useful for consistency).

TODO 5.2: Save to Disk. Add a "Save Calibration Profile" button in the UI that writes this struct to a .json file in a dedicated profiles/ folder.

Phase 6: Integrating with Process Mode

TODO 6.1: Profile Dropdown. In your main [ Process Mode ] UI, add a dropdown that reads the profiles/ directory and lets the user select a saved JSON calibration.

TODO 6.2: Pipeline Update. Update src/curve.rs (as discussed previously) to split the 1D LUT. Apply the log-conversion, multiply the array by the loaded $3 \times 3$ matrix, and then apply the RA-4 curve.

Flat field (Empty frame calibration)
Phase 1: Ingesting the Flat-Field (The Empty Frame)You need to allow the user to load a master reference frame that represents the specific light source and film stock's base.TODO 1.1: CLI / GUI Input. Add a --flat-field argument to your CLI and a "Load Reference Frame" file picker in your UI. This should accept a RAW file of an unexposed, developed frame from the same roll of film (or at least the same film stock).TODO 1.2: Initial Linearization. When a flat-field RAW is loaded, run it through Step 1 (LibRaw extraction) and Step 2 (Demosaic) so you have an Array3<f32> representing linear transmittance.Phase 2: The Grain-Busting Blur (Crucial)To isolate the luminance falloff of the "Big scanlight" without capturing the microscopic film grain, you must aggressively low-pass filter (blur) the flat-field image.TODO 2.1: Implement a Fast Blur. Write or import a function to apply a heavy Gaussian blur to the flat-field Array3<f32>. Since ndarray doesn't have built-in blurring, you can temporarily convert it to an image::Rgb32FImage, use the image crate's image::imageops::blur, and convert it back.TODO 2.2: Extreme Radius. The blur radius needs to be massive (e.g., 50+ pixels) to completely obliterate film grain and dust, leaving only the smooth, low-frequency gradients of the LED light falloff and lens vignetting.Phase 3: Pixel-by-Pixel DivisionThis replaces your current Step 3 (D-min neutralization) when a flat-field is provided.TODO 3.1: The Flat-Field Division. Instead of dividing by a single scalar value, divide your actual image array by your blurred flat-field array, pixel-by-pixel, channel-by-channel:$$T_{out}(x, y) = \frac{T_{in}(x, y)}{T_{flat\_blurred}(x, y)}$$TODO 3.2: Safe Division Check. Ensure you handle cases where $T_{flat\_blurred} \le 0$ to avoid divide-by-zero errors or NaNs (though a properly exposed light source should never be zero). Clamp the denominator to a very small positive number if necessary.TODO 3.3: Normalization Verification. After this division, the darkest parts of the film (the orange mask/film base) will mathematically resolve to exactly 1.0 transmittance across the entire frame, completely erasing the light source's luminance variations.Phase 4: Workflow and Batch OptimizationTODO 4.1: Caching the Flat-Field. Blurring a 42MP image is computationally expensive. When the user loads a flat-field frame, process and blur it once, keep the Array3<f32> in memory, and reuse it for every frame in the batch (or every frame exported from the GUI).TODO 4.2: Save as Profile (Optional). Allow the user to save the processed, blurred flat-field as a raw binary file or 32-bit float TIFF so they don't have to re-process the empty frame every time they use that specific camera/lens/light setup.