// Flat-field division: out = image / max(flat, eps), then clamp >= 0.
// Matches CPU flat_field::apply_flat_field_division exactly.

struct Params {
    width: u32,
    height: u32,
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> flat: array<f32>;
@group(0) @binding(2) var<storage, read_write> image: array<f32>;

const WG_X_STRIDE: u32 = 65535u * 256u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel_idx = gid.y * WG_X_STRIDE + gid.x;
    let total = params.width * params.height;
    if pixel_idx >= total {
        return;
    }

    let base = pixel_idx * 3u;
    let denom_r = max(flat[base + 0u], params.eps);
    let denom_g = max(flat[base + 1u], params.eps);
    let denom_b = max(flat[base + 2u], params.eps);
    image[base + 0u] = max(image[base + 0u] / denom_r, 0.0);
    image[base + 1u] = max(image[base + 1u] / denom_g, 0.0);
    image[base + 2u] = max(image[base + 2u] / denom_b, 0.0);
}
