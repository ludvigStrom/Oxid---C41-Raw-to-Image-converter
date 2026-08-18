// Coherent exemplar copy. Same kernel as CPU dust_wfc::exemplar_fill.

struct Params {
    width: u32,
    height: u32,
    n: u32,
    n_cand: u32,
    color_gate: f32,
    rim_r: f32,
    rim_g: f32,
    rim_b: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> image: array<f32>;
@group(0) @binding(2) var<storage, read> tight: array<u32>;
@group(0) @binding(3) var<storage, read> hole_mask: array<u32>;
@group(0) @binding(4) var<storage, read> prev_fill: array<f32>;
@group(0) @binding(5) var<storage, read_write> next_fill: array<f32>;
@group(0) @binding(6) var<storage, read> prev_src: array<u32>;
@group(0) @binding(7) var<storage, read_write> next_src: array<u32>;
@group(0) @binding(8) var<storage, read> candidates: array<u32>;

const WG_X_STRIDE: u32 = 65535u * 256u;

fn rgb_ssd(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let d = a - b;
    return dot(d, d) / 3.0;
}

fn img_rgb(x: u32, y: u32) -> vec3<f32> {
    let i = (y * params.width + x) * 3u;
    return vec3<f32>(image[i], image[i + 1u], image[i + 2u]);
}

fn pack_src(sx: u32, sy: u32) -> u32 {
    return ((sx + 1u) << 16u) | (sy + 1u);
}

fn src_valid(p: u32) -> bool {
    return p != 0u;
}

fn unpack_sx(p: u32) -> u32 {
    return (p >> 16u) - 1u;
}

fn unpack_sy(p: u32) -> u32 {
    return (p & 0xFFFFu) - 1u;
}

fn color_legal(c: vec3<f32>) -> bool {
    let rim = vec3<f32>(params.rim_r, params.rim_g, params.rim_b);
    return rgb_ssd(c, rim) <= params.color_gate;
}

fn isign(v: i32) -> i32 {
    if v > 0 {
        return 1;
    }
    if v < 0 {
        return -1;
    }
    return 0;
}

fn source_ok(sx: i32, sy: i32) -> bool {
    if sx < 0 || sy < 0 || sx >= i32(params.width) || sy >= i32(params.height) {
        return false;
    }
    let i = u32(sy) * params.width + u32(sx);
    if tight[i] != 0u {
        return false;
    }
    return color_legal(img_rgb(u32(sx), u32(sy)));
}

fn copy_prev(base: u32, idx: u32) {
    next_fill[base] = prev_fill[base];
    next_fill[base + 1u] = prev_fill[base + 1u];
    next_fill[base + 2u] = prev_fill[base + 2u];
    next_fill[base + 3u] = prev_fill[base + 3u];
    next_src[idx] = prev_src[idx];
}

fn write_src(base: u32, idx: u32, sx: u32, sy: u32) {
    let c = img_rgb(sx, sy);
    next_fill[base] = c.x;
    next_fill[base + 1u] = c.y;
    next_fill[base + 2u] = c.z;
    next_fill[base + 3u] = 1.0;
    next_src[idx] = pack_src(sx, sy);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.y * WG_X_STRIDE + gid.x;
    let total = params.width * params.height;
    if idx >= total {
        return;
    }
    let base = idx * 4u;
    if hole_mask[idx] == 0u {
        copy_prev(base, idx);
        return;
    }
    if prev_fill[base + 3u] > 0.5 {
        copy_prev(base, idx);
        return;
    }

    let x = idx % params.width;
    let y = idx / params.width;

    let dir_x = array<i32, 4>(-1, 1, 0, 0);
    let dir_y = array<i32, 4>(0, 0, -1, 1);
    for (var d = 0u; d < 4u; d++) {
        let nx = i32(x) + dir_x[d];
        let ny = i32(y) + dir_y[d];
        if nx < 0 || ny < 0 || nx >= i32(params.width) || ny >= i32(params.height) {
            continue;
        }
        let ni = u32(ny) * params.width + u32(nx);
        let packed = prev_src[ni];
        if !src_valid(packed) {
            continue;
        }
        let ox = i32(unpack_sx(packed)) - nx;
        let oy = i32(unpack_sy(packed)) - ny;
        var sx = i32(x) + ox;
        var sy = i32(y) + oy;
        let step_x = isign(ox);
        let step_y = isign(oy);
        var found_prop = 0u;
        for (var step = 0u; step < 48u; step++) {
            if source_ok(sx, sy) {
                write_src(base, idx, u32(sx), u32(sy));
                found_prop = 1u;
                break;
            }
            if step_x == 0 && step_y == 0 {
                break;
            }
            sx += step_x;
            sy += step_y;
        }
        if found_prop == 1u {
            return;
        }
    }

    let n = params.n;
    let off = i32(n / 2u);
    var known_r = array<f32, 25>();
    var known_g = array<f32, 25>();
    var known_b = array<f32, 25>();
    var known_ok = array<u32, 25>();
    var known_n = 0.0;

    for (var ty = 0u; ty < n; ty++) {
        for (var tx = 0u; tx < n; tx++) {
            let k = ty * n + tx;
            known_ok[k] = 0u;
            let px = i32(x) + i32(tx) - off;
            let py = i32(y) + i32(ty) - off;
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
            known_n += 1.0;
        }
    }

    if known_n < 0.5 || params.n_cand == 0u {
        copy_prev(base, idx);
        return;
    }

    var best = 1.0e30;
    var pick_sx = 0u;
    var pick_sy = 0u;
    var found = 0u;
    for (var t = 0u; t < params.n_cand; t++) {
        let sx = candidates[t * 2u];
        let sy = candidates[t * 2u + 1u];
        var ssd = 0.0;
        var c = 0.0;
        for (var ty = 0u; ty < n; ty++) {
            for (var tx = 0u; tx < n; tx++) {
                let k = ty * n + tx;
                if known_ok[k] == 0u {
                    continue;
                }
                let rx = i32(sx) + i32(tx) - off;
                let ry = i32(sy) + i32(ty) - off;
                if rx < 0 || ry < 0 || rx >= i32(params.width) || ry >= i32(params.height) {
                    continue;
                }
                let ri = u32(ry) * params.width + u32(rx);
                if tight[ri] != 0u {
                    continue;
                }
                let src = img_rgb(u32(rx), u32(ry));
                ssd += rgb_ssd(src, vec3<f32>(known_r[k], known_g[k], known_b[k]));
                c += 1.0;
            }
        }
        if c < 0.5 {
            continue;
        }
        ssd = ssd / c;
        let dist = max(abs(i32(sx) - i32(x)), abs(i32(sy) - i32(y)));
        ssd = ssd - 0.002 * min(f32(dist), 12.0);
        if ssd < best {
            best = ssd;
            pick_sx = sx;
            pick_sy = sy;
            found = 1u;
        }
    }

    if found == 0u {
        copy_prev(base, idx);
        return;
    }
    write_src(base, idx, pick_sx, pick_sy);
}
