import package::render::mesh_view_bindings::{
    vlm_params,
    vlm_tile_buffer,
    vlm_atlas_texture,
    vlm_atlas_sampler,
};

@if(VIRTUAL_LIGHTMAP)
const VLM_TILE_NONE: u32 = 0xFFFFFFFFu;

@if(VIRTUAL_LIGHTMAP)
struct VlmSample {
    irradiance: vec3<f32>,
    sun_shadow: f32,
    coverage: f32,
}

@if(VIRTUAL_LIGHTMAP)
fn vlm_hash_u32(x: u32) -> u32 {
    var a = x;
    a = (a ^ 61u) ^ (a >> 16u);
    a = a + (a << 3u);
    a = a ^ (a >> 4u);
    a = a * 0x27d4eb2du;
    a = a ^ (a >> 15u);
    return a;
}

@if(VIRTUAL_LIGHTMAP)
fn vlm_pcg_u32(x: u32) -> u32 {
    let state = x * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

@if(VIRTUAL_LIGHTMAP)
fn vlm_get_tile_key(tile_u: i32, tile_v: i32, w_slice: i32, dir_id: u32, level: u32) -> vec2<u32> {
    let u = bitcast<u32>(tile_u);
    let v = bitcast<u32>(tile_v);
    let w = bitcast<u32>(w_slice);
    let tile_meta = (dir_id & 7u) | ((level & 7u) << 3u);
    let h = vlm_hash_u32(u ^ vlm_hash_u32(v ^ vlm_hash_u32(w ^ vlm_hash_u32(tile_meta))));
    let key = max(h, 1u);
    let slot = vlm_pcg_u32(u + vlm_pcg_u32(v + vlm_pcg_u32(w + vlm_pcg_u32(tile_meta)))) & vlm_params.table_mask;
    return vec2<u32>(key, slot);
}

@if(VIRTUAL_LIGHTMAP)
fn vlm_find_tile(tile_u: i32, tile_v: i32, w_slice: i32, dir_id: u32, level: u32) -> u32 {
    let key_slot = vlm_get_tile_key(tile_u, tile_v, w_slice, dir_id, level);
    let key = key_slot.x;
    var index = key_slot.y;
    for (var i = 0u; i < vlm_params.search_count; i = i + 1u) {
        let entry = vlm_tile_buffer[index];
        if (entry.x == key) {
            return entry.y;
        }
        if (entry.x == 0u) {
            break;
        }
        index = (index + 1u) & vlm_params.table_mask;
    }
    return VLM_TILE_NONE;
}

@if(VIRTUAL_LIGHTMAP)
fn vlm_dir_id_from_normal(n: vec3<f32>) -> u32 {
    let a = abs(n);
    if (a.x >= a.y && a.x >= a.z) {
        return select(1u, 0u, n.x >= 0.0);
    }
    if (a.y >= a.z) {
        return select(3u, 2u, n.y >= 0.0);
    }
    return select(5u, 4u, n.z >= 0.0);
}

@if(VIRTUAL_LIGHTMAP)
fn vlm_dir_vec(dir_id: u32) -> vec3<f32> {
    switch dir_id {
        case 0u: { return vec3<f32>(1.0, 0.0, 0.0); }
        case 1u: { return vec3<f32>(-1.0, 0.0, 0.0); }
        case 2u: { return vec3<f32>(0.0, 1.0, 0.0); }
        case 3u: { return vec3<f32>(0.0, -1.0, 0.0); }
        case 4u: { return vec3<f32>(0.0, 0.0, 1.0); }
        default: { return vec3<f32>(0.0, 0.0, -1.0); }
    }
}

@if(VIRTUAL_LIGHTMAP)
fn vlm_sample_dir_level(world_pos: vec3<f32>, dir_id: u32, level: u32) -> vec4<f32> {
    let texel_size = vlm_params.texel_size_min * f32(1u << level);
    let dir_vec = vlm_dir_vec(dir_id);
    let p = world_pos + dir_vec * (texel_size * 0.15);

    var uv_world: vec2<f32>;
    var w_world: f32;
    if (dir_id < 2u) {
        uv_world = p.yz;
        w_world = p.x;
    } else if (dir_id < 4u) {
        uv_world = p.xz;
        w_world = p.y;
    } else {
        uv_world = p.xy;
        w_world = p.z;
    }

    let w_step = texel_size * 2.0;
    let w_scaled = w_world / w_step;
    let w_slice = i32(floor(w_scaled));
    let tile_world_size = texel_size * 8.0;
    let tile_coord_f = floor(uv_world / tile_world_size);
    let tile_u = i32(tile_coord_f.x);
    let tile_v = i32(tile_coord_f.y);

    var tile_index = vlm_find_tile(tile_u, tile_v, w_slice, dir_id, level);
    if (tile_index == VLM_TILE_NONE) {
        // Prefer outward depth slice along the surface normal (+1 for +X/+Y/+Z, -1 for -X/-Y/-Z)
        // so 3D relief meshes never fall back into inner wall cavities.
        let is_positive_axis = (dir_id & 1u) == 0u;
        let first_delta = select(
            select(1, -1, fract(w_scaled) <= 0.65),
            select(-1, 1, fract(w_scaled) >= 0.35),
            is_positive_axis,
        );
        tile_index = vlm_find_tile(tile_u, tile_v, w_slice + first_delta, dir_id, level);
        if (tile_index == VLM_TILE_NONE) {
            tile_index = vlm_find_tile(tile_u, tile_v, w_slice - first_delta, dir_id, level);
        }
    }
    if (tile_index == VLM_TILE_NONE) {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }

    // Physical tile in atlas is 10x10 (8x8 core + 1px neighbor border on each side).
    let local_uv = clamp((uv_world - tile_coord_f * tile_world_size) / texel_size, vec2<f32>(0.0), vec2<f32>(8.0));
    let tile_col = tile_index % vlm_params.tiles_per_row;
    let tile_row = tile_index / vlm_params.tiles_per_row;
    let atlas_px = vec2<f32>(f32(tile_col * 10u + 1u), f32(tile_row * 10u + 1u)) + local_uv;
    let atlas_uv = atlas_px * vlm_params.inv_atlas_size;

    let sample = textureSampleLevel(vlm_atlas_texture, vlm_atlas_sampler, atlas_uv, 0.0);
    // Guard against unpopulated/truncated all-zero texels [0, 0, 0, 0] so they fall back to
    // coarser VLM levels or runtime IBL/IrradianceVolume instead of rendering black holes.
    if (max(max(sample.r, sample.g), max(sample.b, sample.a)) <= 1e-5) {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }
    return sample;
}

@if(VIRTUAL_LIGHTMAP)
fn vlm_sample_direction(world_pos: vec3<f32>, dir_id: u32, preferred_level: u32) -> vec4<f32> {
    var val = vlm_sample_dir_level(world_pos, dir_id, preferred_level);
    if (val.a >= 0.0) {
        return val;
    }
    if (preferred_level + 1u < vlm_params.level_count) {
        val = vlm_sample_dir_level(world_pos, dir_id, preferred_level + 1u);
        if (val.a >= 0.0) {
            return val;
        }
    }
    if (preferred_level > 0u) {
        val = vlm_sample_dir_level(world_pos, dir_id, preferred_level - 1u);
    }
    return val;
}

@if(VIRTUAL_LIGHTMAP)
fn sample_virtual_lightmap(world_pos: vec3<f32>, N: vec3<f32>, view_dist: f32) -> VlmSample {
    let preferred_level = select(
        0u,
        min(1u, vlm_params.level_count - 1u),
        vlm_params.level_count > 1u && view_dist > vlm_params.level0_max_distance,
    );

    let dir0 = vlm_dir_id_from_normal(N);
    let s0 = vlm_sample_direction(world_pos, dir0, preferred_level);

    // Optional smooth dual-axis blend on ~45-degree slopes and curved surfaces.
    let dir0_vec = vlm_dir_vec(dir0);
    let rem_n = N * (vec3<f32>(1.0) - abs(dir0_vec));
    let dir1 = vlm_dir_id_from_normal(rem_n);
    let dir1_vec = vlm_dir_vec(dir1);
    let primary_c = max(dot(abs(N), abs(dir0_vec)), 1e-4);
    let secondary_c = dot(abs(N), abs(dir1_vec));
    let ratio = secondary_c / primary_c;
    let w = 0.5 * smoothstep(vlm_params.direction_blend, 1.0, ratio);

    var final_val = s0;
    if (w > 0.001) {
        let s1 = vlm_sample_direction(world_pos, dir1, preferred_level);
        if (s1.a >= 0.0) {
            if (s0.a >= 0.0) {
                final_val = mix(s0, s1, w);
            } else {
                final_val = s1;
            }
        }
    }

    if (final_val.a < 0.0) {
        return VlmSample(vec3<f32>(0.0), 1.0, 0.0);
    }

    let sun_vis = mix(1.0, saturate(final_val.a), vlm_params.sun_shadow_strength);
    return VlmSample(max(final_val.rgb, vec3<f32>(0.0)) * vlm_params.intensity, sun_vis, 1.0);
}

@if(VIRTUAL_LIGHTMAP && AO_DEBUG_VLM_TILES)
fn virtual_lightmap_debug_tile_color(world_pos: vec3<f32>, N: vec3<f32>, view_dist: f32) -> vec3<f32> {
    var level = select(
        0u,
        min(1u, vlm_params.level_count - 1u),
        vlm_params.level_count > 1u && view_dist > vlm_params.level0_max_distance,
    );
    let dir_id = vlm_dir_id_from_normal(N);
    let dir_vec = vlm_dir_vec(dir_id);

    var found_tile = VLM_TILE_NONE;
    var found_texel = vec2<i32>(0);
    var found_key = 0u;

    for (var attempt = 0u; attempt < 2u; attempt = attempt + 1u) {
        let cur_level = select(level, min(level + 1u, vlm_params.level_count - 1u), attempt == 1u);
        let texel_size = vlm_params.texel_size_min * f32(1u << cur_level);
        let p = world_pos + dir_vec * (texel_size * 0.15);
        var uv_world: vec2<f32>;
        var w_world: f32;
        if (dir_id < 2u) {
            uv_world = p.yz;
            w_world = p.x;
        } else if (dir_id < 4u) {
            uv_world = p.xz;
            w_world = p.y;
        } else {
            uv_world = p.xy;
            w_world = p.z;
        }
        let w_step = texel_size * 2.0;
        let w_scaled = w_world / w_step;
        let w_slice = i32(floor(w_scaled));
        let tile_world_size = texel_size * 8.0;
        let tile_coord_f = floor(uv_world / tile_world_size);
        let tile_u = i32(tile_coord_f.x);
        let tile_v = i32(tile_coord_f.y);

        var tid = vlm_find_tile(tile_u, tile_v, w_slice, dir_id, cur_level);
        var used_w = w_slice;
        if (tid == VLM_TILE_NONE) {
            let is_positive_axis = (dir_id & 1u) == 0u;
            let delta = select(
                select(1, -1, fract(w_scaled) <= 0.65),
                select(-1, 1, fract(w_scaled) >= 0.35),
                is_positive_axis,
            );
            used_w = w_slice + delta;
            tid = vlm_find_tile(tile_u, tile_v, used_w, dir_id, cur_level);
            if (tid == VLM_TILE_NONE) {
                used_w = w_slice - delta;
                tid = vlm_find_tile(tile_u, tile_v, used_w, dir_id, cur_level);
            }
        }
        if (tid != VLM_TILE_NONE) {
            found_tile = tid;
            found_texel = vec2<i32>(floor((uv_world - tile_coord_f * tile_world_size) / texel_size));
            found_key = vlm_get_tile_key(tile_u, tile_v, used_w, dir_id, cur_level).x;
            level = cur_level;
            break;
        }
        if (vlm_params.level_count <= 1u) {
            break;
        }
    }

    if (found_tile == VLM_TILE_NONE) {
        // Highlight uncovered hole pixels in bright magenta.
        return vec3<f32>(1.0, 0.0, 0.75);
    }

    let r = f32((found_key >> 0u) & 255u) / 255.0;
    let g = f32((found_key >> 8u) & 255u) / 255.0;
    let b = f32((found_key >> 16u) & 255u) / 255.0;
    let tile_color = vec3<f32>(0.22) + vec3<f32>(r, g, b) * 0.78;
    let checker = select(0.72, 1.0, ((found_texel.x + found_texel.y) & 1) == 0);
    // Subtle dark border on level 1 to visually distinguish LOD levels.
    let level_tint = select(vec3<f32>(1.0), vec3<f32>(0.75, 0.9, 1.0), level > 0u);
    return tile_color * checker * level_tint;
}
