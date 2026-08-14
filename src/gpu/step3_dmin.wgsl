// D-min divide: image[ch] /= div_ch (per-channel scalars from CPU rect/percentile).
// Matches CPU dmin::neutralize_with_medians exactly.

struct Params {
    width: u32,
    height: u32,
    div_r: f32,
    div_g: f32,
    div_b: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> image: array<f32>;

const WG_X_STRIDE: u32 = 65535u * 256u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel_idx = gid.y * WG_X_STRIDE + gid.x;
    let total = params.width * params.height;
    if pixel_idx >= total {
        return;
    }

    let base = pixel_idx * 3u;
    image[base + 0u] = max(image[base + 0u] / params.div_r, 0.0);
    image[base + 1u] = max(image[base + 1u] / params.div_g, 0.0);
    image[base + 2u] = max(image[base + 2u] / params.div_b, 0.0);
}
