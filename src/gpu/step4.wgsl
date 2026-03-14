// Step 4 compute shader: T→D, per-channel WB scale, temp_k offset, shadow cast correction.
// All reduction-based analysis (auto WB medians, shadow cast analysis) is precomputed
// on CPU and passed as uniforms. This shader does only per-pixel transforms.

struct Params {
    width: u32,
    height: u32,
    s_r: f32,
    s_g: f32,
    s_b: f32,
    off_r: f32,
    off_g: f32,
    off_b: f32,
    shadow_cast_active: u32,
    cr: f32,
    cg: f32,
    cb: f32,
    shadow_cast_strength: f32,
    inv_threshold: f32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> image: array<f32>;

const LOG2_10_INV: f32 = 0.30102999566;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel_idx = gid.x;
    let total = params.width * params.height;
    if pixel_idx >= total {
        return;
    }

    let base = pixel_idx * 3u;

    // T → D: density = -log10(max(T, 1e-10)), clamped to >= 0
    let tr = max(image[base + 0u], 1e-10);
    let tg = max(image[base + 1u], 1e-10);
    let tb = max(image[base + 2u], 1e-10);

    var dr = max(-log2(tr) * LOG2_10_INV, 0.0);
    var dg = max(-log2(tg) * LOG2_10_INV, 0.0);
    var db = max(-log2(tb) * LOG2_10_INV, 0.0);

    // Per-channel scale (auto_wb * manual_wb * inv_gamma) + temp_k offset
    dr = dr * params.s_r + params.off_r;
    dg = dg * params.s_g + params.off_g;
    db = db * params.s_b + params.off_b;

    // Shadow cast correction
    if params.shadow_cast_active == 1u {
        let d_mean = (dr + dg + db) * (1.0 / 3.0);
        let t = max(1.0 - d_mean * params.inv_threshold, 0.0);
        let weight = t * sqrt(t) * params.shadow_cast_strength;
        dr = max(dr + params.cr * weight, 0.0);
        dg = max(dg + params.cg * weight, 0.0);
        db = max(db + params.cb * weight, 0.0);
    }

    image[base + 0u] = dr;
    image[base + 1u] = dg;
    image[base + 2u] = db;
}
