// Step 5 compute shader: density matrix / 3D LUT, highlight spread,
// saturation, zone adjustments. Matches CPU pipeline exactly.

struct Params {
    width: u32,
    height: u32,
    use_lut: u32,       // 1 = use 3D LUT, 0 = use matrix
    lut_size: u32,      // grid dimension N of the LUT (e.g. 33)
    lut_d_max: f32,
    saturation: f32,
    zone_shadows: f32,
    zone_highlights: f32,
    color_shadows_r: f32,
    color_shadows_g: f32,
    color_shadows_b: f32,
    color_mids_r: f32,
    color_mids_g: f32,
    color_mids_b: f32,
    color_highlights_r: f32,
    color_highlights_g: f32,
    color_highlights_b: f32,
    zone_shadow_gain: f32,
    zone_mid_gain: f32,
    zone_highlight_gain: f32,
    color_shadow_gain_r: f32,
    color_shadow_gain_g: f32,
    color_shadow_gain_b: f32,
    color_mid_gain_r: f32,
    color_mid_gain_g: f32,
    color_mid_gain_b: f32,
    color_highlight_gain_r: f32,
    color_highlight_gain_g: f32,
    color_highlight_gain_b: f32,
    _pad_gain0: f32,
    _pad_gain1: f32,
    _pad_gain2: f32,
    // 3x3 density matrix (row-major)
    mat_r0: vec3<f32>,
    _pad0: f32,
    mat_r1: vec3<f32>,
    _pad1: f32,
    mat_r2: vec3<f32>,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Image buffer: packed as [r, g, b, r, g, b, ...] with width*height*3 f32 values.
@group(0) @binding(1) var<storage, read_write> image: array<f32>;
// 3D LUT data: N^3 entries, each 4 floats (rgb + padding). Index: r + g*N + b*N*N.
@group(0) @binding(2) var<storage, read> lut_data: array<vec4<f32>>;

fn lut_index(r: u32, g: u32, b: u32) -> vec3<f32> {
    let n = params.lut_size;
    let idx = r + g * n + b * n * n;
    return lut_data[idx].xyz;
}

fn sample_lut_tetrahedral(nr: f32, ng: f32, nb: f32) -> vec3<f32> {
    let n = f32(params.lut_size);
    let rc = clamp(nr, 0.0, 1.0) * (n - 1.0);
    let gc = clamp(ng, 0.0, 1.0) * (n - 1.0);
    let bc = clamp(nb, 0.0, 1.0) * (n - 1.0);

    let r0 = u32(floor(rc));
    let g0 = u32(floor(gc));
    let b0 = u32(floor(bc));
    let r1 = min(r0 + 1u, params.lut_size - 1u);
    let g1 = min(g0 + 1u, params.lut_size - 1u);
    let b1 = min(b0 + 1u, params.lut_size - 1u);

    let fr = rc - f32(r0);
    let fg = gc - f32(g0);
    let fb = bc - f32(b0);

    // Tetrahedral interpolation: 6 cases by order of fr, fg, fb.
    // Matches lut3d.rs exactly.
    var v0: vec3<f32>;
    var v1: vec3<f32>;
    var v2: vec3<f32>;
    var v3: vec3<f32>;
    var d1: f32;
    var d2: f32;
    var d3: f32;

    if fr >= fg && fg >= fb {
        v0 = lut_index(r0, g0, b0);
        v1 = lut_index(r1, g0, b0);
        v2 = lut_index(r1, g1, b0);
        v3 = lut_index(r1, g1, b1);
        d1 = fr - fg; d2 = fg - fb; d3 = fb;
    } else if fr >= fb && fb >= fg {
        v0 = lut_index(r0, g0, b0);
        v1 = lut_index(r1, g0, b0);
        v2 = lut_index(r1, g0, b1);
        v3 = lut_index(r1, g1, b1);
        d1 = fr - fb; d2 = fb - fg; d3 = fg;
    } else if fg >= fr && fr >= fb {
        v0 = lut_index(r0, g0, b0);
        v1 = lut_index(r0, g1, b0);
        v2 = lut_index(r1, g1, b0);
        v3 = lut_index(r1, g1, b1);
        d1 = fg - fr; d2 = fr - fb; d3 = fb;
    } else if fg >= fb && fb >= fr {
        v0 = lut_index(r0, g0, b0);
        v1 = lut_index(r0, g1, b0);
        v2 = lut_index(r0, g1, b1);
        v3 = lut_index(r1, g1, b1);
        d1 = fg - fb; d2 = fb - fr; d3 = fr;
    } else if fb >= fr && fr >= fg {
        v0 = lut_index(r0, g0, b0);
        v1 = lut_index(r0, g0, b1);
        v2 = lut_index(r1, g0, b1);
        v3 = lut_index(r1, g1, b1);
        d1 = fb - fr; d2 = fr - fg; d3 = fg;
    } else {
        // fb >= fg && fg >= fr
        v0 = lut_index(r0, g0, b0);
        v1 = lut_index(r0, g0, b1);
        v2 = lut_index(r0, g1, b1);
        v3 = lut_index(r1, g1, b1);
        d1 = fb - fg; d2 = fg - fr; d3 = fr;
    }

    return v0 + d1 * (v1 - v0) + d2 * (v2 - v0) + d3 * (v3 - v0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel_idx = gid.x;
    let total = params.width * params.height;
    if pixel_idx >= total {
        return;
    }

    let base = pixel_idx * 3u;
    var r = image[base + 0u];
    var g = image[base + 1u];
    var b = image[base + 2u];

    // --- Matrix or 3D LUT ---
    if params.use_lut == 1u {
        let d_max = params.lut_d_max;
        let inv_d_max = 1.0 / max(d_max, 1e-10);
        let out = sample_lut_tetrahedral(r * inv_d_max, g * inv_d_max, b * inv_d_max);
        r = out.x * d_max;
        g = out.y * d_max;
        b = out.z * d_max;
    } else {
        let din = vec3<f32>(r, g, b);
        r = dot(params.mat_r0, din);
        g = dot(params.mat_r1, din);
        b = dot(params.mat_r2, din);
    }

    // Clamp to >= 0
    r = max(r, 0.0);
    g = max(g, 0.0);
    b = max(b, 0.0);

    // --- Highlight density spread limiter ---
    // Sort to (lo, mid, hi).
    var lo = r;
    var mid_v = g;
    var hi = b;
    if lo > mid_v { let t = lo; lo = mid_v; mid_v = t; }
    if mid_v > hi { let t = mid_v; mid_v = hi; hi = t; }
    if lo > mid_v { let t = lo; lo = mid_v; mid_v = t; }

    let range = hi - lo;
    if range >= 0.02 {
        let mid_pos = (mid_v - lo) / range;
        let outlier = abs(0.5 - mid_pos) * 2.0;
        if outlier >= 0.5 {
            let excess = (outlier - 0.5) / 0.5;
            let blend = excess * 0.85;
            let mean_hl = (r + g + b) * (1.0 / 3.0);
            r = r + (mean_hl - r) * blend;
            g = g + (mean_hl - g) * blend;
            b = b + (mean_hl - b) * blend;
        }
    }

    // --- Saturation ---
    let sat = params.saturation;
    if abs(sat - 1.0) > 1e-6 {
        let d_mean = (r + g + b) * (1.0 / 3.0);
        r = max(d_mean + sat * (r - d_mean), 0.0);
        g = max(d_mean + sat * (g - d_mean), 0.0);
        b = max(d_mean + sat * (b - d_mean), 0.0);
    }

    // --- Zone density adjustments: gain (mult) then offset ---
    let zone_s = params.zone_shadows;
    let zone_h = params.zone_highlights;
    let cs = vec3<f32>(params.color_shadows_r, params.color_shadows_g, params.color_shadows_b);
    let cm = vec3<f32>(params.color_mids_r, params.color_mids_g, params.color_mids_b);
    let ch = vec3<f32>(params.color_highlights_r, params.color_highlights_g, params.color_highlights_b);
    let g_s = params.zone_shadow_gain;
    let g_m = params.zone_mid_gain;
    let g_h = params.zone_highlight_gain;
    let cgs = vec3<f32>(params.color_shadow_gain_r, params.color_shadow_gain_g, params.color_shadow_gain_b);
    let cgm = vec3<f32>(params.color_mid_gain_r, params.color_mid_gain_g, params.color_mid_gain_b);
    let cgh = vec3<f32>(params.color_highlight_gain_r, params.color_highlight_gain_g, params.color_highlight_gain_b);

    let has_zones = abs(zone_s) > 1e-6 || abs(zone_h) > 1e-6
        || abs(cs.x) > 1e-6 || abs(cs.y) > 1e-6 || abs(cs.z) > 1e-6
        || abs(cm.x) > 1e-6 || abs(cm.y) > 1e-6 || abs(cm.z) > 1e-6
        || abs(ch.x) > 1e-6 || abs(ch.y) > 1e-6 || abs(ch.z) > 1e-6
        || abs(g_s) > 1e-6 || abs(g_m) > 1e-6 || abs(g_h) > 1e-6
        || abs(cgs.x) > 1e-6 || abs(cgs.y) > 1e-6 || abs(cgs.z) > 1e-6
        || abs(cgm.x) > 1e-6 || abs(cgm.y) > 1e-6 || abs(cgm.z) > 1e-6
        || abs(cgh.x) > 1e-6 || abs(cgh.y) > 1e-6 || abs(cgh.z) > 1e-6;

    if has_zones {
        let d_mean_z = (r + g + b) * (1.0 / 3.0);

        let s_diff = d_mean_z - 0.4;
        let s_mask = exp(-s_diff * s_diff * 4.0);
        let m_diff = d_mean_z - 1.3;
        let m_mask = exp(-m_diff * m_diff * 5.0);
        let h_diff = d_mean_z - 2.2;
        let h_mask = exp(-h_diff * h_diff * 2.0);

        // Per-channel gain: (1 + global*mask) * (1 + color*mask) per zone
        let mult_r = (1.0 + g_s * s_mask) * (1.0 + g_m * m_mask) * (1.0 + g_h * h_mask)
            * (1.0 + cgs.x * s_mask) * (1.0 + cgm.x * m_mask) * (1.0 + cgh.x * h_mask);
        let mult_g = (1.0 + g_s * s_mask) * (1.0 + g_m * m_mask) * (1.0 + g_h * h_mask)
            * (1.0 + cgs.y * s_mask) * (1.0 + cgm.y * m_mask) * (1.0 + cgh.y * h_mask);
        let mult_b = (1.0 + g_s * s_mask) * (1.0 + g_m * m_mask) * (1.0 + g_h * h_mask)
            * (1.0 + cgs.z * s_mask) * (1.0 + cgm.z * m_mask) * (1.0 + cgh.z * h_mask);

        r = r * mult_r;
        g = g * mult_g;
        b = b * mult_b;

        let scale = 2.0;
        let global_offset = zone_s * scale * s_mask + zone_h * scale * h_mask;
        let color_scale = -scale;

        let offset_r = global_offset + (cs.x * s_mask + cm.x * m_mask + ch.x * h_mask) * color_scale;
        let offset_g = global_offset + (cs.y * s_mask + cm.y * m_mask + ch.y * h_mask) * color_scale;
        let offset_b = global_offset + (cs.z * s_mask + cm.z * m_mask + ch.z * h_mask) * color_scale;

        r = max(r + offset_r, 0.0);
        g = max(g + offset_g, 0.0);
        b = max(b + offset_b, 0.0);
    }

    image[base + 0u] = r;
    image[base + 1u] = g;
    image[base + 2u] = b;
}
