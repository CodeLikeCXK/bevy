// GPU Compute Shader for offline Surface SpatialHash Virtual Lightmap baking.
// Mirrors CPU vlm_baker.rs math bit-for-bit:
//   - bake_direct_main: directional sun soft shadow (A) + static local lights direct diffuse + shadows
//   - bake_bounce_main: cosine-weighted hemisphere path tracing using the SpatialHash itself as a Radiance Cache

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;
const BVH_LEAF_BIT: u32 = 0x80000000u;
const BVH_COUNT_MASK: u32 = 0x7FFFFFFFu;
const BVH_STACK_SIZE: u32 = 64u;
const VLM_TILE_NONE: u32 = 0xFFFFFFFFu;
const TRIS_PER_BUFFER: u32 = 6291456u; // 6M tris * 64B = 384 MB (< 512 MB DX12 limit)
const LOCAL_LIGHT_SHADOW_RAYS: u32 = 6u;

struct BakeParams {
    // Dispatch chunking & sample batching (prevents Windows 2s TDR on low-end GPUs)
    active_offset: u32,
    active_count: u32,
    sample_start: u32,
    sample_batch_count: u32,

    total_samples: u32,
    bounce_salt: u32,
    sun_shadow_rays: u32,
    local_light_count: u32,

    // SpatialHash lookup parameters
    table_mask: u32,
    search_count: u32,
    level_count: u32,
    sky_width: u32,

    sky_height: u32,
    node_count: u32,
    local_light_clearance: f32,
    local_light_radius: f32,

    // Sun & lighting parameters
    sun_dir_and_cos_cone: vec4<f32>,
    sun_color_and_illuminance: vec4<f32>,

    // Scalar floats
    base_texel_size: f32,
    gi_scale: f32,
    sky_yaw_rad: f32,
    sky_scale: f32,
};

struct GpuBvhNode {
    aabb_min: vec3<f32>,
    left_or_first: u32,
    aabb_max: vec3<f32>,
    right_or_count: u32,
};

struct GpuTri {
    p0: vec3<f32>,
    n0_oct: u32,
    e1: vec3<f32>,
    n1_oct: u32,
    e2: vec3<f32>,
    n2_oct: u32,
    // x = oct geom_normal, y = rgba8unorm(albedo.rgb, double_sided), z = f16x2(emissive.rg), w = f16x2(emissive.b, 0)
    tri_meta: vec4<u32>,
};

struct GpuLocalLight {
    pos_and_range: vec4<f32>,
    forward_and_inv_range_sq: vec4<f32>,
    color_and_candela: vec4<f32>,
    // x = is_spot (>0.5), y = cos_inner, z = cos_outer, w = 0
    spot_params: vec4<f32>,
};

struct GpuActiveTexel {
    pos: vec3<f32>,
    tile_texel_idx: u32,
    normal: vec3<f32>,
    level: u32,
    // xyz = sun_direct_over_pi, w = sun_vis
    sun_direct_and_vis: vec4<f32>,
    // xyz = local_direct_over_pi, w = 0
    local_direct_and_pad: vec4<f32>,
    // xyz = indirect_over_pi (or running sum across sample batches), w = 0
    indirect_and_pad: vec4<f32>,
};

struct RayHit {
    hit: bool,
    t: f32,
    pos: vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    emissive: vec3<f32>,
};

@group(0) @binding(0) var<uniform> params: BakeParams;
@group(0) @binding(1) var<storage, read> bvh_nodes: array<GpuBvhNode>;
@group(0) @binding(2) var<storage, read> tris_0: array<GpuTri>;
@group(0) @binding(3) var<storage, read_write> active_texels: array<GpuActiveTexel>;
@group(0) @binding(4) var<storage, read_write> tile_cache_write: array<vec4<f32>>;

// Pass 1 only
@group(0) @binding(5) var<storage, read> lights: array<GpuLocalLight>;

// Pass 2 only
@group(0) @binding(6) var<storage, read> sky_pixels: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> tile_table: array<vec2<u32>>;
@group(0) @binding(8) var<storage, read> tile_cache_read: array<vec4<f32>>;

// Both Pass 1 & Pass 2 (overflow slice beyond TRIS_PER_BUFFER)
@group(0) @binding(9) var<storage, read> tris_1: array<GpuTri>;

fn load_tri(tri_idx: u32) -> GpuTri {
    if (tri_idx < TRIS_PER_BUFFER) {
        return tris_0[tri_idx];
    }
    return tris_1[tri_idx - TRIS_PER_BUFFER];
}

// ============================================================================
// Hash & SpatialHash helpers (identical to vlm_baker.rs & virtual_lightmap.wesl)
// ============================================================================

fn vlm_hash_u32(x: u32) -> u32 {
    var a = x;
    a = (a ^ 61u) ^ (a >> 16u);
    a = a + (a << 3u);
    a = a ^ (a >> 4u);
    a = a * 0x27d4eb2du;
    a = a ^ (a >> 15u);
    return a;
}

fn vlm_pcg_u32(x: u32) -> u32 {
    let state = x * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn vlm_get_tile_key(tile_u: i32, tile_v: i32, w_slice: i32, dir_id: u32, level: u32) -> vec2<u32> {
    let u = bitcast<u32>(tile_u);
    let v = bitcast<u32>(tile_v);
    let w = bitcast<u32>(w_slice);
    let tile_meta = (dir_id & 7u) | ((level & 7u) << 3u);
    let h = vlm_hash_u32(u ^ vlm_hash_u32(v ^ vlm_hash_u32(w ^ vlm_hash_u32(tile_meta))));
    let key = max(h, 1u);
    let slot = vlm_pcg_u32(u + vlm_pcg_u32(v + vlm_pcg_u32(w + vlm_pcg_u32(tile_meta)))) & params.table_mask;
    return vec2<u32>(key, slot);
}

fn vlm_find_tile(tile_u: i32, tile_v: i32, w_slice: i32, dir_id: u32, level: u32) -> u32 {
    let key_slot = vlm_get_tile_key(tile_u, tile_v, w_slice, dir_id, level);
    let key = key_slot.x;
    var index = key_slot.y;
    for (var i = 0u; i < params.search_count; i = i + 1u) {
        let entry = tile_table[index];
        if (entry.x == key) {
            return entry.y;
        }
        if (entry.x == 0u) {
            break;
        }
        index = (index + 1u) & params.table_mask;
    }
    return VLM_TILE_NONE;
}

fn dir_id_from_normal(n: vec3<f32>) -> u32 {
    let a = abs(n);
    if (a.x >= a.y && a.x >= a.z) {
        return select(1u, 0u, n.x >= 0.0);
    }
    if (a.y >= a.z) {
        return select(3u, 2u, n.y >= 0.0);
    }
    return select(5u, 4u, n.z >= 0.0);
}

fn dir_vec_from_id(dir_id: u32) -> vec3<f32> {
    switch dir_id {
        case 0u: { return vec3<f32>(1.0, 0.0, 0.0); }
        case 1u: { return vec3<f32>(-1.0, 0.0, 0.0); }
        case 2u: { return vec3<f32>(0.0, 1.0, 0.0); }
        case 3u: { return vec3<f32>(0.0, -1.0, 0.0); }
        case 4u: { return vec3<f32>(0.0, 0.0, 1.0); }
        default: { return vec3<f32>(0.0, 0.0, -1.0); }
    }
}

fn lookup_cached_irradiance(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let dir_id = dir_id_from_normal(n);
    let dir_vec = dir_vec_from_id(dir_id);

    // Search from coarsest level down to finest (smoother radiance cache, higher hit rate)
    var lvl = i32(params.level_count) - 1;
    loop {
        if (lvl < 0) {
            break;
        }
        let level = u32(lvl);
        let texel_size = params.base_texel_size * f32(1u << level);
        let shifted = p + dir_vec * (texel_size * 0.15);

        var uv_world: vec2<f32>;
        var w_world: f32;
        if (dir_id < 2u) {
            uv_world = shifted.yz;
            w_world = shifted.x;
        } else if (dir_id < 4u) {
            uv_world = shifted.xz;
            w_world = shifted.y;
        } else {
            uv_world = shifted.xy;
            w_world = shifted.z;
        }

        let w_step = texel_size * 2.0;
        let w_scaled = w_world / w_step;
        let w_slice = i32(floor(w_scaled));
        let tile_world_size = texel_size * 8.0;
        let tile_coord_f = floor(uv_world / tile_world_size);
        let tile_u = i32(tile_coord_f.x);
        let tile_v = i32(tile_coord_f.y);
        let tx = u32(clamp(floor((uv_world.x - tile_coord_f.x * tile_world_size) / texel_size), 0.0, 7.0));
        let ty = u32(clamp(floor((uv_world.y - tile_coord_f.y * tile_world_size) / texel_size), 0.0, 7.0));

        let delta = select(-1, 1, fract(w_scaled) >= 0.5);
        for (var attempt = 0u; attempt < 2u; attempt = attempt + 1u) {
            let ws = select(w_slice, w_slice + delta, attempt == 1u);
            let tid = vlm_find_tile(tile_u, tile_v, ws, dir_id, level);
            if (tid != VLM_TILE_NONE) {
                let cached = tile_cache_read[tid * 64u + ty * 8u + tx];
                if (cached.w > 0.5) {
                    return cached.xyz;
                }
            }
        }
        lvl = lvl - 1;
    }
    return vec3<f32>(0.0);
}

// ============================================================================
// Sampling & Geometry Unpacking
// ============================================================================

fn unpack_oct_normal(packed: u32) -> vec3<f32> {
    let e = unpack2x16snorm(packed);
    var n = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        let sx = select(-1.0, 1.0, n.x >= 0.0);
        let sy = select(-1.0, 1.0, n.y >= 0.0);
        n = vec3<f32>((1.0 - abs(n.y)) * sx, (1.0 - abs(n.x)) * sy, n.z);
    }
    return normalize(n);
}

fn radical_inverse_base2(bits_in: u32) -> f32 {
    return f32(reverseBits(bits_in)) * 2.3283064e-10;
}

fn hash_pos_seed(p: vec3<f32>, salt: u32) -> vec2<f32> {
    let qx = bitcast<u32>(i32(round(p.x * 1000.0)));
    let qy = bitcast<u32>(i32(round(p.y * 1000.0)));
    let qz = bitcast<u32>(i32(round(p.z * 1000.0)));
    let h1 = vlm_hash_u32(qx ^ vlm_hash_u32(qy ^ vlm_hash_u32(qz ^ salt)));
    let h2 = vlm_pcg_u32(h1 ^ 0xA5A55A5Au);
    return vec2<f32>(
        f32(h1 & 0x00FFFFFFu) / 16777216.0,
        f32(h2 & 0x00FFFFFFu) / 16777216.0,
    );
}

fn build_orthonormal_basis(n: vec3<f32>) -> mat2x3<f32> {
    if (n.z < -0.9999999) {
        return mat2x3<f32>(vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(-1.0, 0.0, 0.0));
    }
    let a = 1.0 / (1.0 + n.z);
    let b = -n.x * n.y * a;
    let t = vec3<f32>(1.0 - n.x * n.x * a, b, -n.x);
    let bt = vec3<f32>(b, 1.0 - n.y * n.y * a, -n.y);
    return mat2x3<f32>(t, bt);
}

fn sample_cone_dir(axis: vec3<f32>, cos_max_angle: f32, u1: f32, u2: f32) -> vec3<f32> {
    let cos_theta = 1.0 - u1 * (1.0 - cos_max_angle);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let phi = TAU * u2;
    let basis = build_orthonormal_basis(axis);
    return normalize(basis[0] * (cos(phi) * sin_theta) + basis[1] * (sin(phi) * sin_theta) + axis * cos_theta);
}

fn sample_sky(dir: vec3<f32>) -> vec3<f32> {
    if (params.sky_width == 0u || params.sky_height == 0u) {
        let up = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
        let ground = vec3<f32>(0.14, 0.13, 0.12);
        let horizon = vec3<f32>(0.62, 0.72, 0.86);
        let zenith = vec3<f32>(0.38, 0.58, 0.95);
        let c = select(mix(horizon, zenith, (up - 0.5) * 2.0), mix(ground, horizon, up * 2.0), up < 0.5);
        return c * params.sky_scale;
    }
    let sin_y = sin(-params.sky_yaw_rad);
    let cos_y = cos(-params.sky_yaw_rad);
    let rx = dir.x * cos_y + dir.z * sin_y;
    let rz = -dir.x * sin_y + dir.z * cos_y;
    let ry = clamp(dir.y, -1.0, 1.0);

    var u = atan2(rz, rx) / TAU + 0.5;
    u = u - floor(u);
    let v = clamp(acos(ry) / PI, 0.0, 1.0);
    let ix = min(u32(u * f32(params.sky_width)), params.sky_width - 1u);
    let iy = min(u32(v * f32(params.sky_height)), params.sky_height - 1u);
    return sky_pixels[iy * params.sky_width + ix].xyz * params.sky_scale;
}

// ============================================================================
// BVH Ray Tracing (is_occluded & trace_closest)
// ============================================================================

fn safe_inv_dir(dir: vec3<f32>) -> vec3<f32> {
    let sx = select(-1.0, 1.0, dir.x >= 0.0);
    let sy = select(-1.0, 1.0, dir.y >= 0.0);
    let sz = select(-1.0, 1.0, dir.z >= 0.0);
    return vec3<f32>(sx, sy, sz) / max(abs(dir), vec3<f32>(1e-8));
}

fn intersect_aabb(ro_inv_rd: vec3<f32>, inv_dir: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>, t_min: f32, t_max: f32) -> f32 {
    let t0 = fma(bmin, inv_dir, -ro_inv_rd);
    let t1 = fma(bmax, inv_dir, -ro_inv_rd);
    let t_near = min(t0, t1);
    let t_far = max(t0, t1);
    let enter = max(max(max(t_near.x, t_near.y), t_near.z), t_min);
    let exit = min(min(min(t_far.x, t_far.y), t_far.z), t_max);
    return select(-1.0, enter, enter <= exit);
}

fn intersect_tri(
    origin: vec3<f32>,
    dir: vec3<f32>,
    tri_idx: u32,
    t_min: f32,
    t_max: f32,
    backface_guard_dist: f32,
    ignore_emissive: bool,
) -> vec3<f32> {
    let tri = load_tri(tri_idx);
    if (ignore_emissive) {
        let em_rg = unpack2x16float(tri.tri_meta.z);
        let em_b0 = unpack2x16float(tri.tri_meta.w);
        if (em_rg.x + em_rg.y + em_b0.x > 0.05) {
            return vec3<f32>(-1.0, 0.0, 0.0);
        }
    }
    let p0 = tri.p0;
    let e1 = tri.e1;
    let e2 = tri.e2;

    let h = cross(dir, e2);
    let a = dot(e1, h);
    if (abs(a) < 1e-8) {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }
    let f = 1.0 / a;
    let s = origin - p0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }
    let q = cross(s, e1);
    let v = f * dot(dir, q);
    if (v < 0.0 || u + v > 1.0) {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }
    let t = f * dot(e2, q);
    if (t < t_min || t >= t_max) {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }
    if (t < backface_guard_dist) {
        let gn = unpack_oct_normal(tri.tri_meta.x);
        if (dot(dir, gn) > 0.0) {
            return vec3<f32>(-1.0, 0.0, 0.0);
        }
    }
    return vec3<f32>(t, u, v);
}

fn is_occluded(
    origin: vec3<f32>,
    dir: vec3<f32>,
    t_min: f32,
    max_dist: f32,
    backface_guard_dist: f32,
    ignore_emissive: bool,
) -> bool {
    if (params.node_count == 0u) {
        return false;
    }
    let inv_dir = safe_inv_dir(dir);
    let ro_inv_rd = origin * inv_dir;

    let root = bvh_nodes[0u];
    if (intersect_aabb(ro_inv_rd, inv_dir, root.aabb_min, root.aabb_max, t_min, max_dist) < 0.0) {
        return false;
    }

    var stack: array<u32, BVH_STACK_SIZE>;
    var sp = 1u;
    stack[0u] = 0u;

    while (sp > 0u) {
        sp = sp - 1u;
        let node = bvh_nodes[stack[sp]];
        if (intersect_aabb(ro_inv_rd, inv_dir, node.aabb_min, node.aabb_max, t_min, max_dist) < 0.0) {
            continue;
        }
        if ((node.right_or_count & BVH_LEAF_BIT) != 0u) {
            let tri_count = node.right_or_count & BVH_COUNT_MASK;
            let first = node.left_or_first;
            for (var i = 0u; i < tri_count; i = i + 1u) {
                if (intersect_tri(origin, dir, first + i, t_min, max_dist, backface_guard_dist, ignore_emissive).x >= 0.0) {
                    return true;
                }
            }
        } else {
            if (sp + 2u <= BVH_STACK_SIZE) {
                stack[sp] = node.right_or_count;
                stack[sp + 1u] = node.left_or_first;
                sp = sp + 2u;
            }
        }
    }
    return false;
}

fn trace_closest(origin: vec3<f32>, dir: vec3<f32>, t_min: f32, max_dist: f32, backface_guard_dist: f32) -> RayHit {
    var result: RayHit;
    result.hit = false;
    result.t = max_dist;
    if (params.node_count == 0u) {
        return result;
    }

    let inv_dir = safe_inv_dir(dir);
    let ro_inv_rd = origin * inv_dir;

    let root = bvh_nodes[0u];
    if (intersect_aabb(ro_inv_rd, inv_dir, root.aabb_min, root.aabb_max, t_min, max_dist) < 0.0) {
        return result;
    }

    var best_t = max_dist;
    var best_uv = vec2<f32>(0.0);
    var best_tri = 0xFFFFFFFFu;
    var stack: array<u32, BVH_STACK_SIZE>;
    var sp = 1u;
    stack[0u] = 0u;

    while (sp > 0u) {
        sp = sp - 1u;
        let node = bvh_nodes[stack[sp]];
        if (intersect_aabb(ro_inv_rd, inv_dir, node.aabb_min, node.aabb_max, t_min, best_t) < 0.0) {
            continue;
        }
        if ((node.right_or_count & BVH_LEAF_BIT) != 0u) {
            let tri_count = node.right_or_count & BVH_COUNT_MASK;
            let first = node.left_or_first;
            for (var i = 0u; i < tri_count; i = i + 1u) {
                let hit = intersect_tri(origin, dir, first + i, t_min, best_t, backface_guard_dist, false);
                if (hit.x >= 0.0) {
                    best_t = hit.x;
                    best_uv = hit.yz;
                    best_tri = first + i;
                }
            }
        } else {
            let l_idx = node.left_or_first;
            let r_idx = node.right_or_count;
            let l_node = bvh_nodes[l_idx];
            let r_node = bvh_nodes[r_idx];
            let tl = intersect_aabb(ro_inv_rd, inv_dir, l_node.aabb_min, l_node.aabb_max, t_min, best_t);
            let tr = intersect_aabb(ro_inv_rd, inv_dir, r_node.aabb_min, r_node.aabb_max, t_min, best_t);
            if (tl >= 0.0 && tr >= 0.0) {
                if (sp + 2u <= BVH_STACK_SIZE) {
                    if (tl <= tr) {
                        stack[sp] = r_idx;
                        stack[sp + 1u] = l_idx;
                    } else {
                        stack[sp] = l_idx;
                        stack[sp + 1u] = r_idx;
                    }
                    sp = sp + 2u;
                }
            } else if (tl >= 0.0) {
                if (sp < BVH_STACK_SIZE) {
                    stack[sp] = l_idx;
                    sp = sp + 1u;
                }
            } else if (tr >= 0.0) {
                if (sp < BVH_STACK_SIZE) {
                    stack[sp] = r_idx;
                    sp = sp + 1u;
                }
            }
        }
    }

    if (best_tri == 0xFFFFFFFFu) {
        return result;
    }

    let tri = load_tri(best_tri);
    let n0 = unpack_oct_normal(tri.n0_oct);
    let n1 = unpack_oct_normal(tri.n1_oct);
    let n2 = unpack_oct_normal(tri.n2_oct);
    let w = 1.0 - best_uv.x - best_uv.y;
    let raw_n = n0 * w + n1 * best_uv.x + n2 * best_uv.y;
    let raw_n_len_sq = dot(raw_n, raw_n);
    var n = select(unpack_oct_normal(tri.tri_meta.x), raw_n * inverseSqrt(raw_n_len_sq), raw_n_len_sq > 1e-10);

    let alb_ds = unpack4x8unorm(tri.tri_meta.y);
    if (alb_ds.w > 0.5 && dot(n, dir) > 0.0) {
        n = -n;
    }
    let em_rg = unpack2x16float(tri.tri_meta.z);
    let em_b0 = unpack2x16float(tri.tri_meta.w);

    result.hit = true;
    result.t = best_t;
    result.pos = origin + dir * best_t;
    result.normal = n;
    result.albedo = alb_ds.xyz;
    result.emissive = vec3<f32>(em_rg.x, em_rg.y, em_b0.x);
    return result;
}

// ============================================================================
// Pass 1: Direct Sun Soft Shadows + Static Local Lights
// ============================================================================

@compute @workgroup_size(64, 1, 1)
fn bake_direct_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_i = gid.x;
    if (local_i >= params.active_count) {
        return;
    }
    let idx = params.active_offset + local_i;
    let texel = active_texels[idx];

    let p = texel.pos;
    let tile_texel_idx = texel.tile_texel_idx;
    let n = texel.normal;
    let level = texel.level;

    let texel_size = params.base_texel_size * f32(1u << level);
    let bias = clamp(texel_size * 0.2, 0.004, 0.03);
    let t_min = max(texel_size * 0.35, 0.008);
    let backface_guard = texel_size * 4.0;
    let origin = p + n * bias;

    // 1a. Primary directional sun soft shadow visibility
    let sun_dir = params.sun_dir_and_cos_cone.xyz;
    let cos_sun_cone = params.sun_dir_and_cos_cone.w;
    let sun_color = params.sun_color_and_illuminance.xyz;
    let sun_illuminance = params.sun_color_and_illuminance.w;
    let ndotl_sun = dot(n, sun_dir);

    var sun_vis = 0.0;
    if (ndotl_sun > 0.0) {
        let seed = hash_pos_seed(p, 0x13579BDFu);
        var unoccluded = 0u;
        let n_rays = max(params.sun_shadow_rays, 1u);
        for (var s = 0u; s < n_rays; s = s + 1u) {
            var ray_dir = sun_dir;
            if (n_rays > 1u) {
                let u1 = fract((f32(s) + 0.5) / f32(n_rays) + seed.x);
                let u2 = fract(radical_inverse_base2(s) + seed.y);
                ray_dir = sample_cone_dir(sun_dir, cos_sun_cone, u1, u2);
            }
            if (!is_occluded(origin, ray_dir, t_min, 1e5, backface_guard, false)) {
                unoccluded = unoccluded + 1u;
            }
        }
        sun_vis = f32(unoccluded) / f32(n_rays);
    }
    let sun_direct_over_pi = sun_color * (sun_illuminance * max(ndotl_sun, 0.0) * sun_vis / PI);

    // 1b. Static Point / Spot lights with spherical soft shadow + fixture clearance + emissive pass-through
    let local_clearance = max(params.local_light_clearance, 0.05);
    let local_radius = max(params.local_light_radius, 0.0);
    var local_direct_over_pi = vec3<f32>(0.0);
    for (var li = 0u; li < params.local_light_count; li = li + 1u) {
        let light = lights[li];
        let to_light = light.pos_and_range.xyz - p;
        let dist_sq = dot(to_light, to_light);
        let range = light.pos_and_range.w;
        if (dist_sq >= range * range || dist_sq < 1e-6) {
            continue;
        }
        let dist = sqrt(dist_sq);
        let l_dir = to_light / dist;
        let ndotl = dot(n, l_dir);
        if (ndotl <= 0.0) {
            continue;
        }

        var spot_att = 1.0;
        if (light.spot_params.x > 0.5) {
            let cos_a = dot(-l_dir, light.forward_and_inv_range_sq.xyz);
            let cos_outer = light.spot_params.z;
            if (cos_a <= cos_outer) {
                continue;
            }
            let denom = max(light.spot_params.y - cos_outer, 1e-4);
            let t = clamp((cos_a - cos_outer) / denom, 0.0, 1.0);
            spot_att = t * t;
        }

        var vis = 1.0;
        if (dist > local_clearance) {
            let eff_radius = min(local_radius, (dist - local_clearance) * 0.45);
            let l_basis = build_orthonormal_basis(l_dir);
            let l_seed = hash_pos_seed(p, 0x517CC1B7u ^ (li * 0x9E3779B9u));
            var unoccluded = 0u;
            for (var s = 0u; s < LOCAL_LIGHT_SHADOW_RAYS; s = s + 1u) {
                let u1 = fract((f32(s) + 0.5) / f32(LOCAL_LIGHT_SHADOW_RAYS) + l_seed.x);
                let u2 = fract(radical_inverse_base2(s) + l_seed.y);
                let r = sqrt(u2) * eff_radius;
                let phi = TAU * u1;
                let sample_pos = light.pos_and_range.xyz + (l_basis[0] * cos(phi) + l_basis[1] * sin(phi)) * r;
                let ray_vec = sample_pos - origin;
                let ray_dist = length(ray_vec);
                let max_t = max(ray_dist - local_clearance, t_min);
                if (max_t <= t_min || !is_occluded(origin, ray_vec / max(ray_dist, 1e-6), t_min, max_t, backface_guard, true)) {
                    unoccluded = unoccluded + 1u;
                }
            }
            vis = f32(unoccluded) / f32(LOCAL_LIGHT_SHADOW_RAYS);
        }
        if (vis <= 0.0) {
            continue;
        }

        let factor = dist_sq * light.forward_and_inv_range_sq.w;
        let smooth_f = clamp(1.0 - factor * factor, 0.0, 1.0);
        let range_att = (smooth_f * smooth_f) / max(dist_sq, 0.04);
        local_direct_over_pi = local_direct_over_pi + light.color_and_candela.xyz * (light.color_and_candela.w * range_att * spot_att * ndotl * vis / PI);
    }

    active_texels[idx].sun_direct_and_vis = vec4<f32>(sun_direct_over_pi, sun_vis);
    active_texels[idx].local_direct_and_pad = vec4<f32>(local_direct_over_pi, 0.0);
    active_texels[idx].indirect_and_pad = vec4<f32>(0.0);
    tile_cache_write[tile_texel_idx] = vec4<f32>(sun_direct_over_pi + local_direct_over_pi, 1.0);
}

// ============================================================================
// Pass 2: Multi-bounce Indirect Diffuse GI (chunked by texel & sample batch)
// ============================================================================

@compute @workgroup_size(64, 1, 1)
fn bake_bounce_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_i = gid.x;
    if (local_i >= params.active_count) {
        return;
    }
    let idx = params.active_offset + local_i;
    let texel = active_texels[idx];

    let p = texel.pos;
    let tile_texel_idx = texel.tile_texel_idx;
    let n = texel.normal;
    let level = texel.level;

    let texel_size = params.base_texel_size * f32(1u << level);
    let bias = clamp(texel_size * 0.2, 0.004, 0.03);
    let t_min = max(texel_size * 0.35, 0.008);
    let backface_guard = texel_size * 4.0;
    let origin = p + n * bias;

    let basis = build_orthonormal_basis(n);
    let seed = hash_pos_seed(p, params.bounce_salt);
    let inv_total = 1.0 / f32(max(params.total_samples, 1u));

    var batch_sum = vec3<f32>(0.0);
    let s_end = min(params.sample_start + params.sample_batch_count, params.total_samples);
    for (var s = params.sample_start; s < s_end; s = s + 1u) {
        let u1 = fract((f32(s) + 0.5) * inv_total + seed.x);
        let u2 = fract(radical_inverse_base2(s) + seed.y) * 0.992;
        let phi = TAU * u1;
        let r = sqrt(u2);
        let z = sqrt(max(1.0 - u2, 0.0));
        let rd = normalize(basis[0] * (r * cos(phi)) + basis[1] * (r * sin(phi)) + n * z);

        let hit = trace_closest(origin, rd, t_min, 1e5, backface_guard);
        if (hit.hit) {
            let cached_irr = lookup_cached_irradiance(hit.pos, hit.normal);
            batch_sum = batch_sum + cached_irr * hit.albedo + hit.emissive;
        } else {
            batch_sum = batch_sum + sample_sky(rd);
        }
    }

    var accum = batch_sum;
    if (params.sample_start > 0u) {
        accum = accum + texel.indirect_and_pad.xyz;
    }

    if (s_end >= params.total_samples) {
        let irr_over_pi = clamp(accum * inv_total * params.gi_scale, vec3<f32>(0.0), vec3<f32>(60000.0));
        active_texels[idx].indirect_and_pad = vec4<f32>(irr_over_pi, 0.0);
        let total_irr = texel.sun_direct_and_vis.xyz + texel.local_direct_and_pad.xyz + irr_over_pi;
        tile_cache_write[tile_texel_idx] = vec4<f32>(total_irr, 1.0);
    } else {
        active_texels[idx].indirect_and_pad = vec4<f32>(accum, 0.0);
    }
}
