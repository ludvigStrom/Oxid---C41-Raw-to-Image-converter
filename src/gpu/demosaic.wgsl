// RGGB quality demosaic: 2-pass. Matches CPU demosaic_quality_bayer exactly.
// Pass 1: edge-aware green interpolation at every pixel.
// Pass 2: (R-G) and (B-G) interpolation, then R=(R-G)+G, B=(B-G)+G.

struct Params {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> bayer: array<f32>;
@group(0) @binding(2) var<storage, read_write> g_plane: array<f32>;

fn clamp_coord(y: i32, x: i32) -> vec2<i32> {
    let yy = clamp(y, 0, i32(params.height) - 1);
    let xx = clamp(x, 0, i32(params.width) - 1);
    return vec2<i32>(yy, xx);
}

fn sample_bayer(y: i32, x: i32) -> f32 {
    let c = clamp_coord(y, x);
    let idx = c.x * i32(params.width) + c.y;
    return bayer[idx];
}

// Pass 1: Edge-aware green for RGGB. At G sites (y+x)%2==1, native. At R/B sites, interpolate.
const WG_X_STRIDE_P1: u32 = 65535u * 256u;

@compute @workgroup_size(256)
fn pass1_green(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel_idx = gid.y * WG_X_STRIDE_P1 + gid.x;
    let total = params.width * params.height;
    if pixel_idx >= total {
        return;
    }
    let y = i32(pixel_idx / params.width);
    let x = i32(pixel_idx % params.width);
    let yu = u32(y);
    let xu = u32(x);

    // G at (even, odd) and (odd, even): (y+x)%2 == 1
    if ((yu + xu) % 2u) == 1u {
        g_plane[pixel_idx] = bayer[pixel_idx];
        return;
    }

    // R/B site: edge-aware interpolation
    let gh = abs(sample_bayer(y, x - 1) - sample_bayer(y, x + 1));
    let gv = abs(sample_bayer(y - 1, x) - sample_bayer(y + 1, x));
    let lh = abs(2.0 * sample_bayer(y, x) - sample_bayer(y, x - 2) - sample_bayer(y, x + 2));
    let lv = abs(2.0 * sample_bayer(y, x) - sample_bayer(y - 2, x) - sample_bayer(y + 2, x));
    let dh = gh + lh;
    let dv = gv + lv;

    var g: f32;
    if dh < dv {
        g = 0.5 * (sample_bayer(y, x - 1) + sample_bayer(y, x + 1));
    } else if dv < dh {
        g = 0.5 * (sample_bayer(y - 1, x) + sample_bayer(y + 1, x));
    } else {
        g = 0.25 * (sample_bayer(y, x - 1) + sample_bayer(y, x + 1) + sample_bayer(y - 1, x) + sample_bayer(y + 1, x));
    }
    g_plane[pixel_idx] = g;
}

// Pass 2: R-G and B-G interpolation
@group(0) @binding(0) var<uniform> params2: Params;
@group(0) @binding(1) var<storage, read> bayer2: array<f32>;
@group(0) @binding(2) var<storage, read> g_plane2: array<f32>;
@group(0) @binding(3) var<storage, read_write> rgb: array<f32>;

fn sample_bayer2(y: i32, x: i32) -> f32 {
    let yy = clamp(y, 0, i32(params2.height) - 1);
    let xx = clamp(x, 0, i32(params2.width) - 1);
    return bayer2[yy * i32(params2.width) + xx];
}

fn sample_g(y: i32, x: i32) -> f32 {
    let yy = clamp(y, 0, i32(params2.height) - 1);
    let xx = clamp(x, 0, i32(params2.width) - 1);
    return g_plane2[yy * i32(params2.width) + xx];
}

// Average (R-G) at 4 nearest R sites: (yt,xt), (yt,xt+2), (yt+2,xt), (yt+2,xt+2). RGGB: R at (even,even).
fn interp_r_minus_g(uy: u32, ux: u32) -> f32 {
    let y = i32(uy);
    let x = i32(ux);
    let yt = (y >> 1) << 1;
    let xt = (x >> 1) << 1;
    let h = i32(params2.height);
    let w = i32(params2.width);
    let yy0 = clamp(yt + 0, 0, h - 1);
    let yy2 = clamp(yt + 2, 0, h - 1);
    let xx0 = clamp(xt + 0, 0, w - 1);
    let xx2 = clamp(xt + 2, 0, w - 1);
    var sum = sample_bayer2(yy0, xx0) - sample_g(yy0, xx0);
    sum += sample_bayer2(yy0, xx2) - sample_g(yy0, xx2);
    sum += sample_bayer2(yy2, xx0) - sample_g(yy2, xx0);
    sum += sample_bayer2(yy2, xx2) - sample_g(yy2, xx2);
    return sum / 4.0;
}

// Average (B-G) at 4 nearest B sites: (base_y+1,base_x+1), etc. RGGB: B at (odd,odd).
fn interp_b_minus_g(uy: u32, ux: u32) -> f32 {
    let y = i32(uy);
    let x = i32(ux);
    let by = (y >> 1) << 1;
    let bx = (x >> 1) << 1;
    let h = i32(params2.height);
    let w = i32(params2.width);
    let yy1 = clamp(by + 1, 0, h - 1);
    let yy3 = clamp(by + 3, 0, h - 1);
    let xx1 = clamp(bx + 1, 0, w - 1);
    let xx3 = clamp(bx + 3, 0, w - 1);
    var sum = sample_bayer2(yy1, xx1) - sample_g(yy1, xx1);
    sum += sample_bayer2(yy1, xx3) - sample_g(yy1, xx3);
    sum += sample_bayer2(yy3, xx1) - sample_g(yy3, xx1);
    sum += sample_bayer2(yy3, xx3) - sample_g(yy3, xx3);
    return sum / 4.0;
}

const WG_X_STRIDE: u32 = 65535u * 256u;

@compute @workgroup_size(256)
fn pass2_rgb(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel_idx = gid.y * WG_X_STRIDE + gid.x;
    let total = params2.width * params2.height;
    if pixel_idx >= total {
        return;
    }
    let y = pixel_idx / params2.width;
    let x = pixel_idx % params2.width;
    let g = g_plane2[pixel_idx];
    let r_minus_g = interp_r_minus_g(y, x);
    let b_minus_g = interp_b_minus_g(y, x);
    let r = max(r_minus_g + g, 0.0);
    let b = max(b_minus_g + g, 0.0);
    rgb[pixel_idx * 3u + 0u] = r;
    rgb[pixel_idx * 3u + 1u] = g;
    rgb[pixel_idx * 3u + 2u] = b;
}
