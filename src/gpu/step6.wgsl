// Step 6 compute shader: all output stages (Ra4, FilmPrint, None, Lut2383)
// with post-curve operations. Matches CPU pipeline exactly.

struct Params {
    width: u32,
    height: u32,
    mode: u32,               // 0=Ra4, 1=FilmPrint, 2=None, 3=Lut2383
    d_max: f32,
    lut_in_black: f32,
    lut_in_white: f32,
    lut_in_mid: f32,
    levels_active: u32,
    white_point: f32,
    toe_strength: f32,
    shoulder_strength: f32,
    soft_clip: f32,
    highlight_warmth: f32,
    apply_lab: u32,
    lab_separation: f32,
    no_invert: u32,
    color_bleed: f32,
    vibrance: f32,
    output_lut_encoding: u32, // 0=CineonLog, 1=Rec709, 2=LinearDensity
    output_lut_size: u32,
    use_output_lut: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_image: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_image: array<f32>;
// 1D curve LUTs: Ra4 = 65536 entries, FilmPrint = 3*65536 entries (R,G,B sequential)
@group(0) @binding(3) var<storage, read> curve_lut: array<f32>;
// 3D output LUT for Lut2383: N^3 entries as vec4 (rgb + pad). Same format as step5.
@group(0) @binding(4) var<storage, read> output_lut_3d: array<vec4<f32>>;

const WG_X_STRIDE: u32 = 65535u * 256u;

// ─── Helper functions ───

fn smoothstep_fn(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn soft_knee_scalar(x: f32, s: f32) -> f32 {
    let sc = clamp(s, 0.0, 0.9999);
    if x <= sc {
        return x;
    }
    let one_minus_s = 1.0 - sc;
    let t = -(x - sc) / one_minus_s;
    return sc + (1.0 - exp(t)) * one_minus_s;
}

fn linear_to_srgb(v: f32) -> f32 {
    let x = clamp(v, 0.0, 1.0);
    if x <= 0.0031308 {
        return 12.92 * x;
    }
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

fn srgb_to_linear(v: f32) -> f32 {
    let x = clamp(v, 0.0, 1.0);
    if x <= 0.04045 {
        return x / 12.92;
    }
    return pow((x + 0.055) / 1.055, 2.4);
}

fn rgb_to_xyz(r: f32, g: f32, b: f32) -> vec3<f32> {
    return vec3<f32>(
        0.4124564 * r + 0.3575761 * g + 0.1804375 * b,
        0.2126729 * r + 0.7151522 * g + 0.0721750 * b,
        0.0193339 * r + 0.1191920 * g + 0.9503041 * b,
    );
}

fn xyz_to_rgb(x: f32, y: f32, z: f32) -> vec3<f32> {
    return vec3<f32>(
         3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
        -0.9692660 * x + 1.8760108 * y + 0.0415560 * z,
         0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
    );
}

fn lab_f(t: f32) -> f32 {
    // 216/24389 ≈ 0.008856, 24389/27 ≈ 903.3
    if t > 0.008856 {
        return pow(t, 1.0 / 3.0);
    }
    return (903.3 * t + 16.0) / 116.0;
}

fn lab_f_inv(t: f32) -> f32 {
    let t3 = t * t * t;
    if t3 > 0.008856 {
        return t3;
    }
    return (116.0 * t - 16.0) / 903.3;
}

fn xyz_to_lab(x: f32, y: f32, z: f32) -> vec3<f32> {
    let xr = x / 0.95047;
    let yr = y / 1.0;
    let zr = z / 1.08883;
    let fx = lab_f(xr);
    let fy = lab_f(yr);
    let fz = lab_f(zr);
    return vec3<f32>(116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz));
}

fn lab_to_xyz(l: f32, a: f32, b: f32) -> vec3<f32> {
    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;
    return vec3<f32>(lab_f_inv(fx) * 0.95047, lab_f_inv(fy) * 1.0, lab_f_inv(fz) * 1.08883);
}

// ─── Post-curve operations ───

fn apply_lab_separation(r_in: f32, g_in: f32, b_in: f32, strength: f32) -> vec3<f32> {
    let sr = clamp(r_in, 0.0, 1.0);
    let sg = clamp(g_in, 0.0, 1.0);
    let sb = clamp(b_in, 0.0, 1.0);

    let r_lin = srgb_to_linear(sr);
    let g_lin = srgb_to_linear(sg);
    let b_lin = srgb_to_linear(sb);

    let xyz = rgb_to_xyz(r_lin, g_lin, b_lin);
    let lab = xyz_to_lab(xyz.x, xyz.y, xyz.z);
    let l = lab.x;
    let a = lab.y;
    let b_l = lab.z;

    let c_ab = sqrt(a * a + b_l * b_l);
    if c_ab < 1e-4 {
        return vec3<f32>(sr, sg, sb);
    }

    let s = clamp(strength, -2.0, 2.0);
    let c_norm = clamp(c_ab / 100.0, 0.0, 1.0);
    let mid_boost = 1.0 + s * (c_norm * (1.0 - c_norm)) * 2.0;
    let edge_soften = 1.0 + 0.2 * s * (1.0 - c_norm);
    let gain = max(mid_boost * edge_soften, 0.0);

    let a2 = a * gain;
    let b2 = b_l * gain;

    let xyz2 = lab_to_xyz(l, a2, b2);
    let rgb2 = xyz_to_rgb(xyz2.x, xyz2.y, xyz2.z);

    return vec3<f32>(
        clamp(linear_to_srgb(rgb2.x), 0.0, 1.0),
        clamp(linear_to_srgb(rgb2.y), 0.0, 1.0),
        clamp(linear_to_srgb(rgb2.z), 0.0, 1.0),
    );
}

fn apply_highlight_warmth(r_in: f32, g_in: f32, b_in: f32, warmth: f32) -> vec3<f32> {
    var r = r_in;
    var g = g_in;
    var b = b_in;

    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let chroma = max(r, max(g, b)) - min(r, min(g, b));

    let highlight_ramp = smoothstep_fn(0.35, 0.85, luma);
    let neutrality = 1.0 - smoothstep_fn(0.04, 0.18, chroma);
    let strength = highlight_ramp * neutrality * warmth;

    r = clamp(r + strength * 0.035, 0.0, 1.0);
    g = clamp(g + strength * 0.015, 0.0, 1.0);
    b = clamp(b - strength * 0.055, 0.0, 1.0);

    let luma2 = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let chroma2 = max(r, max(g, b)) - min(r, min(g, b));
    if luma2 > 0.96 && chroma2 > 0.10 {
        let t = smoothstep_fn(0.96, 1.0, luma2);
        let max_chroma = 0.10;
        let reduce = clamp((chroma2 - max_chroma) / chroma2, 0.0, 1.0) * t;
        r = r + (luma2 - r) * reduce;
        g = g + (luma2 - g) * reduce;
        b = b + (luma2 - b) * reduce;
    }

    return vec3<f32>(r, g, b);
}

fn apply_toe_shoulder(v: f32, toe: f32, shoulder: f32) -> f32 {
    let toe_mask = 1.0 - smoothstep_fn(0.07, 0.60, v);
    let shoulder_mask = smoothstep_fn(0.45, 0.95, v);
    let toe_offset = toe * toe_mask * (0.5 - v) * 0.60;
    let shoulder_offset = shoulder * shoulder_mask * (0.5 - v) * 0.90;
    return clamp(v + toe_offset + shoulder_offset, 0.0, 1.0);
}

fn apply_vibrance_pixel(r: f32, g: f32, b: f32, strength: f32) -> vec3<f32> {
    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let chroma = max(r, max(g, b)) - min(r, min(g, b));
    let boost = 1.0 + strength * (1.0 - clamp(chroma, 0.0, 1.0));
    return vec3<f32>(
        clamp(luma + (r - luma) * boost, 0.0, 1.0),
        clamp(luma + (g - luma) * boost, 0.0, 1.0),
        clamp(luma + (b - luma) * boost, 0.0, 1.0),
    );
}

// ─── Density levels ───

fn apply_density_levels_pixel(v: f32, d_max: f32, in_black: f32, in_white: f32, in_mid: f32) -> f32 {
    let range = max(in_white - in_black, 1e-6);
    let inv_mid = 1.0 / clamp(in_mid, 0.01, 10.0);
    var nv = clamp(v / d_max, 0.0, 1.0);
    nv = clamp((nv - in_black) / range, 0.0, 1.0);
    if abs(in_mid - 1.0) > 1e-6 {
        nv = pow(nv, inv_mid);
    }
    return nv;
}

fn density_to_rec709_pixel(v: f32, in_black: f32, in_white: f32, in_mid: f32) -> f32 {
    let norm = clamp(v / 2.5, 0.0, 1.0);
    let gamma = linear_to_srgb(norm);
    let range = max(in_white - in_black, 1e-6);
    var nv = clamp((gamma - in_black) / range, 0.0, 1.0);
    if abs(in_mid - 1.0) > 1e-6 {
        let inv_mid = 1.0 / clamp(in_mid, 0.01, 10.0);
        nv = pow(nv, inv_mid);
    }
    return nv;
}

// ─── 3D LUT sampling (tetrahedral, same as step5) ───

fn lut3d_index(r: u32, g: u32, b: u32, n: u32) -> vec3<f32> {
    let idx = r + g * n + b * n * n;
    return output_lut_3d[idx].xyz;
}

fn sample_output_lut(nr: f32, ng: f32, nb: f32) -> vec3<f32> {
    let n = f32(params.output_lut_size);
    let rc = clamp(nr, 0.0, 1.0) * (n - 1.0);
    let gc = clamp(ng, 0.0, 1.0) * (n - 1.0);
    let bc = clamp(nb, 0.0, 1.0) * (n - 1.0);

    let r0 = u32(floor(rc));
    let g0 = u32(floor(gc));
    let b0 = u32(floor(bc));
    let r1 = min(r0 + 1u, params.output_lut_size - 1u);
    let g1 = min(g0 + 1u, params.output_lut_size - 1u);
    let b1 = min(b0 + 1u, params.output_lut_size - 1u);

    let fr = rc - f32(r0);
    let fg = gc - f32(g0);
    let fb = bc - f32(b0);

    var v0: vec3<f32>; var v1: vec3<f32>; var v2: vec3<f32>; var v3: vec3<f32>;
    var d1: f32; var d2: f32; var d3: f32;

    if fr >= fg && fg >= fb {
        v0 = lut3d_index(r0,g0,b0,params.output_lut_size); v1 = lut3d_index(r1,g0,b0,params.output_lut_size);
        v2 = lut3d_index(r1,g1,b0,params.output_lut_size); v3 = lut3d_index(r1,g1,b1,params.output_lut_size);
        d1 = fr-fg; d2 = fg-fb; d3 = fb;
    } else if fr >= fb && fb >= fg {
        v0 = lut3d_index(r0,g0,b0,params.output_lut_size); v1 = lut3d_index(r1,g0,b0,params.output_lut_size);
        v2 = lut3d_index(r1,g0,b1,params.output_lut_size); v3 = lut3d_index(r1,g1,b1,params.output_lut_size);
        d1 = fr-fb; d2 = fb-fg; d3 = fg;
    } else if fg >= fr && fr >= fb {
        v0 = lut3d_index(r0,g0,b0,params.output_lut_size); v1 = lut3d_index(r0,g1,b0,params.output_lut_size);
        v2 = lut3d_index(r1,g1,b0,params.output_lut_size); v3 = lut3d_index(r1,g1,b1,params.output_lut_size);
        d1 = fg-fr; d2 = fr-fb; d3 = fb;
    } else if fg >= fb && fb >= fr {
        v0 = lut3d_index(r0,g0,b0,params.output_lut_size); v1 = lut3d_index(r0,g1,b0,params.output_lut_size);
        v2 = lut3d_index(r0,g1,b1,params.output_lut_size); v3 = lut3d_index(r1,g1,b1,params.output_lut_size);
        d1 = fg-fb; d2 = fb-fr; d3 = fr;
    } else if fb >= fr && fr >= fg {
        v0 = lut3d_index(r0,g0,b0,params.output_lut_size); v1 = lut3d_index(r0,g0,b1,params.output_lut_size);
        v2 = lut3d_index(r1,g0,b1,params.output_lut_size); v3 = lut3d_index(r1,g1,b1,params.output_lut_size);
        d1 = fb-fr; d2 = fr-fg; d3 = fg;
    } else {
        v0 = lut3d_index(r0,g0,b0,params.output_lut_size); v1 = lut3d_index(r0,g0,b1,params.output_lut_size);
        v2 = lut3d_index(r0,g1,b1,params.output_lut_size); v3 = lut3d_index(r1,g1,b1,params.output_lut_size);
        d1 = fb-fg; d2 = fg-fr; d3 = fr;
    }

    return v0 + d1*(v1-v0) + d2*(v2-v0) + d3*(v3-v0);
}

// ─── Main ───

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel_idx = gid.y * WG_X_STRIDE + gid.x;
    let total = params.width * params.height;
    if pixel_idx >= total {
        return;
    }

    let base = pixel_idx * 3u;
    let dr = input_image[base + 0u];
    let dg = input_image[base + 1u];
    let db = input_image[base + 2u];

    var r: f32 = 0.0;
    var g: f32 = 0.0;
    var b: f32 = 0.0;

    if params.mode == 0u {
        // ── Ra4 ──
        var lr = dr;
        var lg = dg;
        var lb = db;
        if params.levels_active == 1u {
            lr = apply_density_levels_pixel(lr, params.d_max, params.lut_in_black, params.lut_in_white, params.lut_in_mid) * params.d_max;
            lg = apply_density_levels_pixel(lg, params.d_max, params.lut_in_black, params.lut_in_white, params.lut_in_mid) * params.d_max;
            lb = apply_density_levels_pixel(lb, params.d_max, params.lut_in_black, params.lut_in_white, params.lut_in_mid) * params.d_max;
        }
        let frac_r = clamp(lr / params.d_max, 0.0, 1.0);
        let frac_g = clamp(lg / params.d_max, 0.0, 1.0);
        let frac_b = clamp(lb / params.d_max, 0.0, 1.0);
        let idx_r = min(u32(round(frac_r * 65535.0)), 65535u);
        let idx_g = min(u32(round(frac_g * 65535.0)), 65535u);
        let idx_b = min(u32(round(frac_b * 65535.0)), 65535u);
        r = curve_lut[idx_r];
        g = curve_lut[idx_g];
        b = curve_lut[idx_b];

        // White point
        let wp = clamp(params.white_point, 1e-6, 10.0);
        if abs(wp - 1.0) > 1e-7 {
            let inv_wp = 1.0 / wp;
            r = clamp(r * inv_wp, 0.0, 1.0);
            g = clamp(g * inv_wp, 0.0, 1.0);
            b = clamp(b * inv_wp, 0.0, 1.0);
        }

        // Toe/shoulder
        let toe = clamp(params.toe_strength, -1.0, 1.0);
        let shoulder = clamp(params.shoulder_strength, -1.0, 1.0);
        if abs(toe) > 1e-6 || abs(shoulder) > 1e-6 {
            r = apply_toe_shoulder(r, toe, shoulder);
            g = apply_toe_shoulder(g, toe, shoulder);
            b = apply_toe_shoulder(b, toe, shoulder);
        }

        // Soft knee
        if params.soft_clip >= 0.0 && params.soft_clip < 0.999 {
            r = soft_knee_scalar(r, params.soft_clip);
            g = soft_knee_scalar(g, params.soft_clip);
            b = soft_knee_scalar(b, params.soft_clip);
        }

        // Lab separation
        if params.apply_lab == 1u && abs(params.lab_separation) > 1e-6 {
            let lab_result = apply_lab_separation(r, g, b, params.lab_separation);
            r = lab_result.x; g = lab_result.y; b = lab_result.z;
        }

        // Highlight warmth
        if abs(params.highlight_warmth) > 1e-6 {
            let hw = apply_highlight_warmth(r, g, b, params.highlight_warmth);
            r = hw.x; g = hw.y; b = hw.z;
        }

    } else if params.mode == 1u {
        // ── FilmPrint ──
        var lr = dr;
        var lg = dg;
        var lb = db;
        if params.levels_active == 1u {
            lr = apply_density_levels_pixel(lr, params.d_max, params.lut_in_black, params.lut_in_white, params.lut_in_mid) * params.d_max;
            lg = apply_density_levels_pixel(lg, params.d_max, params.lut_in_black, params.lut_in_white, params.lut_in_mid) * params.d_max;
            lb = apply_density_levels_pixel(lb, params.d_max, params.lut_in_black, params.lut_in_white, params.lut_in_mid) * params.d_max;
        }

        // Color bleed
        let bleed = params.color_bleed;
        if bleed > 0.0 {
            let keep = 1.0 - bleed;
            let half_bleed = bleed * 0.5;
            let br = lr * keep + lg * half_bleed + lb * half_bleed;
            let bg = lg * keep + lr * half_bleed + lb * half_bleed;
            let bb = lb * keep + lr * half_bleed + lg * half_bleed;
            lr = br; lg = bg; lb = bb;
        }

        // Per-channel LUT lookup (3 LUTs stored sequentially: [R*65536, G*65536, B*65536])
        let frac_r = clamp(lr / params.d_max, 0.0, 1.0);
        let frac_g = clamp(lg / params.d_max, 0.0, 1.0);
        let frac_b = clamp(lb / params.d_max, 0.0, 1.0);
        let idx_r = min(u32(round(frac_r * 65535.0)), 65535u);
        let idx_g = min(u32(round(frac_g * 65535.0)), 65535u);
        let idx_b = min(u32(round(frac_b * 65535.0)), 65535u);
        r = curve_lut[idx_r];
        g = curve_lut[65536u + idx_g];
        b = curve_lut[131072u + idx_b];

        // White point
        let wp = clamp(params.white_point, 1e-6, 10.0);
        if abs(wp - 1.0) > 1e-7 {
            let inv_wp = 1.0 / wp;
            r = clamp(r * inv_wp, 0.0, 1.0);
            g = clamp(g * inv_wp, 0.0, 1.0);
            b = clamp(b * inv_wp, 0.0, 1.0);
        }

        // Vibrance
        if abs(params.vibrance) > 1e-6 {
            let vib = apply_vibrance_pixel(r, g, b, params.vibrance);
            r = vib.x; g = vib.y; b = vib.z;
        }

        // Toe/shoulder
        let toe = clamp(params.toe_strength, -1.0, 1.0);
        let shoulder = clamp(params.shoulder_strength, -1.0, 1.0);
        if abs(toe) > 1e-6 || abs(shoulder) > 1e-6 {
            r = apply_toe_shoulder(r, toe, shoulder);
            g = apply_toe_shoulder(g, toe, shoulder);
            b = apply_toe_shoulder(b, toe, shoulder);
        }

        // Soft knee
        if params.soft_clip >= 0.0 && params.soft_clip < 0.999 {
            r = soft_knee_scalar(r, params.soft_clip);
            g = soft_knee_scalar(g, params.soft_clip);
            b = soft_knee_scalar(b, params.soft_clip);
        }

        // Lab separation
        if params.apply_lab == 1u && abs(params.lab_separation) > 1e-6 {
            let lab_result = apply_lab_separation(r, g, b, params.lab_separation);
            r = lab_result.x; g = lab_result.y; b = lab_result.z;
        }

        // Highlight warmth
        if abs(params.highlight_warmth) > 1e-6 {
            let hw = apply_highlight_warmth(r, g, b, params.highlight_warmth);
            r = hw.x; g = hw.y; b = hw.z;
        }

    } else if params.mode == 2u {
        // ── None ──
        if params.no_invert == 0u {
            r = clamp(dr / 2.5, 0.0, 1.0);
            g = clamp(dg / 2.5, 0.0, 1.0);
            b = clamp(db / 2.5, 0.0, 1.0);
        } else {
            r = dr; g = dg; b = db;
        }

    } else {
        // ── Lut2383 ──
        if params.output_lut_encoding == 1u {
            // Rec709
            r = density_to_rec709_pixel(dr, params.lut_in_black, params.lut_in_white, params.lut_in_mid);
            g = density_to_rec709_pixel(dg, params.lut_in_black, params.lut_in_white, params.lut_in_mid);
            b = density_to_rec709_pixel(db, params.lut_in_black, params.lut_in_white, params.lut_in_mid);
        } else {
            var d_enc_max: f32;
            if params.output_lut_encoding == 0u {
                d_enc_max = 2.046;
            } else {
                d_enc_max = 2.5;
            }
            r = apply_density_levels_pixel(dr, d_enc_max, params.lut_in_black, params.lut_in_white, params.lut_in_mid);
            g = apply_density_levels_pixel(dg, d_enc_max, params.lut_in_black, params.lut_in_white, params.lut_in_mid);
            b = apply_density_levels_pixel(db, d_enc_max, params.lut_in_black, params.lut_in_white, params.lut_in_mid);
        }

        // 3D output LUT
        if params.use_output_lut == 1u {
            let lut_out = sample_output_lut(clamp(r,0.0,1.0), clamp(g,0.0,1.0), clamp(b,0.0,1.0));
            r = lut_out.x; g = lut_out.y; b = lut_out.z;
        }

        // Lab separation
        if params.apply_lab == 1u && abs(params.lab_separation) > 1e-6 {
            let lab_result = apply_lab_separation(r, g, b, params.lab_separation);
            r = lab_result.x; g = lab_result.y; b = lab_result.z;
        }

        // Soft knee
        if params.soft_clip >= 0.0 && params.soft_clip < 0.999 {
            r = soft_knee_scalar(clamp(r,0.0,1.0), params.soft_clip);
            g = soft_knee_scalar(clamp(g,0.0,1.0), params.soft_clip);
            b = soft_knee_scalar(clamp(b,0.0,1.0), params.soft_clip);
        }

        // Highlight warmth
        if abs(params.highlight_warmth) > 1e-6 {
            let hw = apply_highlight_warmth(clamp(r,0.0,1.0), clamp(g,0.0,1.0), clamp(b,0.0,1.0), params.highlight_warmth);
            r = hw.x; g = hw.y; b = hw.z;
        }
    }

    output_image[base + 0u] = r;
    output_image[base + 1u] = g;
    output_image[base + 2u] = b;
}
