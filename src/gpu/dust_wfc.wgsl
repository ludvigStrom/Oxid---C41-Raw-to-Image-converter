// Parallel Jacobi tile pick. Same score/pick as CPU dust_wfc::pick_tile.

struct Params {
    width: u32,
    height: u32,
    n: u32,
    n_tiles: u32,
    tau: f32,
    mean_w: f32,
    tau_penalty: f32,
    score_band: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> image: array<f32>;
@group(0) @binding(2) var<storage, read> tight: array<u32>;
@group(0) @binding(3) var<storage, read> component: array<u32>;
@group(0) @binding(4) var<storage, read> prev_fill: array<f32>;
@group(0) @binding(5) var<storage, read_write> next_fill: array<f32>;
@group(0) @binding(6) var<storage, read> tiles: array<f32>;

const WG_X_STRIDE: u32 = 65535u * 256u;

fn pixel_hash(x: u32, y: u32) -> u32 {
    var n = x * 0x9E3779B9u ^ y * 0x85EBCA6Bu;
    n = n ^ (n >> 16u);
    n = n * 0x7FEB352Du;
    n = n ^ (n >> 15u);
    return n;
}

fn rgb_ssd(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let d = a - b;
    return dot(d, d) / 3.0;
}

fn img_rgb(x: u32, y: u32) -> vec3<f32> {
    let i = (y * params.width + x) * 3u;
    return vec3<f32>(image[i], image[i + 1u], image[i + 2u]);
}

fn tile_rgb(t: u32, k: u32) -> vec3<f32> {
    let i = (t * params.n * params.n + k) * 3u;
    return vec3<f32>(tiles[i], tiles[i + 1u], tiles[i + 2u]);
}

fn center_idx() -> u32 {
    let o = params.n / 2u;
    return o * params.n + o;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.y * WG_X_STRIDE + gid.x;
    let total = params.width * params.height;
    if idx >= total {
        return;
    }
    let base = idx * 4u;
    if component[idx] == 0u {
        next_fill[base] = prev_fill[base];
        next_fill[base + 1u] = prev_fill[base + 1u];
        next_fill[base + 2u] = prev_fill[base + 2u];
        next_fill[base + 3u] = prev_fill[base + 3u];
        return;
    }

    let x = idx % params.width;
    let y = idx / params.width;
    let n = params.n;
    let off = i32(n / 2u);

    var known_r = array<f32, 25>();
    var known_g = array<f32, 25>();
    var known_b = array<f32, 25>();
    var known_ok = array<u32, 25>();
    var known_n = 0.0;
    var mean = vec3<f32>(0.0);

    for (var ty = 0u; ty < n; ty++) {
        for (var tx = 0u; tx < n; tx++) {
            let k = ty * n + tx;
            let px = i32(x) + i32(tx) - off;
            let py = i32(y) + i32(ty) - off;
            known_ok[k] = 0u;
            if px < 0 || py < 0 || px >= i32(params.width) || py >= i32(params.height) {
                continue;
            }
            let ni = u32(py) * params.width + u32(px);
            let fb = ni * 4u;
            var c = vec3<f32>(0.0);
            if prev_fill[fb + 3u] > 0.5 {
                c = vec3<f32>(prev_fill[fb], prev_fill[fb + 1u], prev_fill[fb + 2u]);
            } else if tight[ni] == 0u {
                c = img_rgb(u32(px), u32(py));
            } else {
                continue;
            }
            known_r[k] = c.x;
            known_g[k] = c.y;
            known_b[k] = c.z;
            known_ok[k] = 1u;
            mean += c;
            known_n += 1.0;
        }
    }

    if known_n < 0.5 || params.n_tiles == 0u {
        next_fill[base] = prev_fill[base];
        next_fill[base + 1u] = prev_fill[base + 1u];
        next_fill[base + 2u] = prev_fill[base + 2u];
        next_fill[base + 3u] = prev_fill[base + 3u];
        return;
    }
    mean = mean / known_n;

    var best = 1.0e30;
    var pick = 0u;
    for (var t = 0u; t < params.n_tiles; t++) {
        var ssd = 0.0;
        var c = 0.0;
        let np = n * n;
        for (var k = 0u; k < np; k++) {
            if known_ok[k] == 0u {
                continue;
            }
            ssd += rgb_ssd(tile_rgb(t, k), vec3<f32>(known_r[k], known_g[k], known_b[k]));
            c += 1.0;
        }
        if c > 0.0 {
            ssd = ssd / c;
        }
        let mean_term = rgb_ssd(tile_rgb(t, center_idx()), mean);
        let over = max(ssd - params.tau, 0.0);
        let score = ssd + params.mean_w * mean_term + params.tau_penalty * over;
        if score < best {
            best = score;
            pick = t;
        }
    }

    let ctr = tile_rgb(pick, center_idx());
    next_fill[base] = ctr.x;
    next_fill[base + 1u] = ctr.y;
    next_fill[base + 2u] = ctr.z;
    next_fill[base + 3u] = 1.0;
}
