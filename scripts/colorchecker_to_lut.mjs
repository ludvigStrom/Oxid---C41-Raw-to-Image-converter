#!/usr/bin/env node
/**
 * Generate a 3D LUT (.cube) from ColorChecker scan data.
 *
 * Two modes:
 *   correction (default): measured → reference. Fix scans to match reference.
 *   look (--look):        reference → measured. Apply at end to mimic scanner look (e.g. Noritsu).
 *
 * Usage:
 *   node scripts/colorchecker_to_lut.mjs profiles/colorchecker_scan.json -o scan_correction.cube
 *   node scripts/colorchecker_to_lut.mjs profiles/colorchecker_scan.json --look -o noritsu_look.cube
 */

import fs from "fs";
import path from "path";

function hexToSrgb(hexStr) {
  const h = hexStr.replace(/^#/, "");
  if (h.length !== 6) throw new Error(`Invalid hex: ${hexStr}`);
  return [
    parseInt(h.slice(0, 2), 16) / 255,
    parseInt(h.slice(2, 4), 16) / 255,
    parseInt(h.slice(4, 6), 16) / 255,
  ];
}

function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function hexToLinear(hexStr) {
  const [r, g, b] = hexToSrgb(hexStr);
  return [srgbToLinear(r), srgbToLinear(g), srgbToLinear(b)];
}

/** Multiply 3×3 matrix A by 3×1 vector v. Returns [3]. */
function matVec3(A, v) {
  return [
    A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2],
    A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2],
    A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2],
  ];
}

/** 3×3 matrix inverse (for small matrices). */
function inv3(M) {
  const [[a, b, c], [d, e, f], [g, h, i]] = M;
  const det =
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (Math.abs(det) < 1e-12) throw new Error("Singular matrix in OLS");
  return [
    [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
    [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
    [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
  ];
}

/** 3×3 × 3×3 matrix multiply */
function matMul3(A, B) {
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < 3; k++) C[i][j] += A[i][k] * B[k][j];
    }
  }
  return C;
}

/** OLS: solve Y ≈ X @ B for 3×3 B. X = [n][3], Y = [n][3]. Returns B^T as 3×3 M. */
function solveOls(X, Y) {
  const n = X.length;
  let XtX = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  let XtY = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < 3; p++) {
      for (let q = 0; q < 3; q++) XtX[p][q] += X[i][p] * X[i][q];
      for (let q = 0; q < 3; q++) XtY[p][q] += X[i][p] * Y[i][q];
    }
  }
  const B = matMul3(inv3(XtX), XtY);
  // M = B^T so output = M @ input
  return [
    [B[0][0], B[1][0], B[2][0]],
    [B[0][1], B[1][1], B[2][1]],
    [B[0][2], B[1][2], B[2][2]],
  ];
}

function parseArgs() {
  const args = process.argv.slice(2);
  let jsonPath = null;
  let output = null;
  let size = 17;
  let look = false;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "-o" || args[i] === "--output") {
      output = args[++i];
    } else if (args[i] === "-s" || args[i] === "--size") {
      size = parseInt(args[++i], 10);
    } else if (args[i] === "--look" || args[i] === "-l") {
      look = true;
    } else if (!args[i].startsWith("-")) {
      jsonPath = args[i];
    }
  }
  if (!jsonPath) {
    console.error("Usage: node colorchecker_to_lut.mjs <json_path> [-o output.cube] [-s size] [--look]");
    process.exit(1);
  }
  return { jsonPath, output, size, look };
}

function main() {
  const { jsonPath, output, size, look } = parseArgs();

  const data = JSON.parse(fs.readFileSync(jsonPath, "utf8"));
  const patches = data.patches || [];
  if (patches.length < 6) {
    console.error("Error: need at least 6 patches for a stable 3×3 fit");
    process.exit(1);
  }

  const measured = patches.map((p) => hexToLinear(p.measured_hex));
  const reference = patches.map((p) => hexToLinear(p.reference_hex));

  // look: input=reference → output=measured. correction: input=measured → output=reference.
  // Use --look flag or type:"look" in JSON.
  const useLook = look || data.type === "look";
  const [X, Y] = useLook ? [reference, measured] : [measured, reference];
  const M = solveOls(X, Y);

  let mse = 0;
  for (let i = 0; i < X.length; i++) {
    const pred = matVec3(M, X[i]);
    for (let c = 0; c < 3; c++) mse += (pred[c] - Y[i][c]) ** 2;
  }
  mse /= X.length * 3;
  const mode = useLook ? "look (reference → measured)" : "correction (measured → reference)";
  console.log(`Solved 3×3 matrix (OLS), mode: ${mode}, MSE = ${mse.toFixed(6)}`);

  const scale = size > 1 ? size - 1 : 1;
  const title = useLook ? "Scanner look LUT (Noritsu-style)" : "ColorChecker correction LUT";
  const lines = [
    `# ${title} (display linear RGB 0-1)`,
    `# Generated from ${path.basename(jsonPath)}${useLook ? " [look mode]" : ""}`,
    `LUT_3D_SIZE ${size}`,
  ];
  for (let b = 0; b < size; b++) {
    for (let g = 0; g < size; g++) {
      for (let r = 0; r < size; r++) {
        const ir = r / scale;
        const ig = g / scale;
        const ib = b / scale;
        let out = matVec3(M, [ir, ig, ib]);
        out = out.map((v) => Math.max(0, Math.min(1, v)));
        lines.push(`${out[0].toFixed(6)} ${out[1].toFixed(6)} ${out[2].toFixed(6)}`);
      }
    }
  }

  const outPath = output || jsonPath.replace(/\.json$/i, ".cube");
  fs.writeFileSync(outPath, lines.join("\n") + "\n");
  console.log(`Wrote ${outPath} (${size}³ = ${size ** 3} entries)`);
}

main();
