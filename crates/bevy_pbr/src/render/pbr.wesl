import package::{
    render::{
        pbr_types,
        pbr_functions::alpha_discard,
        pbr_fragment::pbr_input_from_standard_material,
    },
    decal::clustered::apply_decals,
};

@if(PREPASS_PIPELINE)
import package::{
    prepass::io::{VertexOutput, FragmentOutput},
    deferred::functions::deferred_output,
};
@else
import package::render::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
};

@if(VISIBILITY_RANGE_DITHER)
import package::render::pbr_functions::visibility_range_dither;

@if(MESHLET_MESH_MATERIAL_PASS)
import package::meshlet::visibility_buffer_resolve::resolve_vertex_output;

@if(MESHLET_MESH_MATERIAL_PASS && AO_DEBUG_MESHLET_CLUSTERS)
import package::render::utils::rand_f;

@if(!PREPASS_PIPELINE && VIRTUAL_LIGHTMAP && AO_DEBUG_VLM_TILES)
import package::{
    light_probe::virtual_lightmap,
    render::mesh_view_bindings::view,
};

@if(MATERIAL_OIT_ENABLED)
import bevy_core_pipeline::oit::draw::oit_draw;

@if(FORWARD_DECAL)
import package::decal::forward::get_forward_decal_info;

@fragment
fn fragment(
@if(MESHLET_MESH_MATERIAL_PASS)
    @builtin(position) frag_coord: vec4<f32>,
@if(!MESHLET_MESH_MATERIAL_PASS)
    vertex_output: VertexOutput,
@if(!MESHLET_MESH_MATERIAL_PASS)
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
@if(MESHLET_MESH_MATERIAL_PASS)
    let vertex_output = resolve_vertex_output(frag_coord);
@if(MESHLET_MESH_MATERIAL_PASS)
    let is_front = true;

    var in = vertex_output;

    // If we're in the crossfade section of a visibility range, conditionally
    // discard the fragment according to the visibility pattern.
@if(VISIBILITY_RANGE_DITHER)
    visibility_range_dither(in.position, in.visibility_range_dither);

@if(FORWARD_DECAL)
    let forward_decal_info = get_forward_decal_info(in);
@if(FORWARD_DECAL)
    in.world_position = forward_decal_info.world_position;
@if(FORWARD_DECAL)
    in.uv = forward_decal_info.uv;

    // generate a PbrInput struct from the StandardMaterial bindings
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // alpha discard
    pbr_input.material.base_color = alpha_discard(
        pbr_input.material.flags,
        pbr_input.material.alpha_cutoff,
        pbr_input.material.base_color
    );

    // clustered decals
    apply_decals(&pbr_input);

    // write the gbuffer, lighting pass id, and optionally normal and motion_vector textures
@if(PREPASS_PIPELINE)
    let out = deferred_output(in, pbr_input);
    // in forward mode, we calculate the lit color immediately, and then apply some post-lighting effects here.
    // in deferred mode the lit color and these effects will be calculated in the deferred lighting shader
@if(!PREPASS_PIPELINE)
    var out: FragmentOutput;
@if(!PREPASS_PIPELINE) {
    @if(AO_DEBUG_VERTEX_AO) {
        out.color = vec4<f32>(vec3<f32>(pbr_input.vertex_ao), pbr_input.material.base_color.a);
    } @elif(AO_DEBUG_DIFFUSE_OCCLUSION) {
        out.color = vec4<f32>(pbr_input.diffuse_occlusion, pbr_input.material.base_color.a);
    } @elif(AO_DEBUG_SPECULAR_OCCLUSION) {
        out.color = vec4<f32>(vec3<f32>(pbr_input.specular_occlusion), pbr_input.material.base_color.a);
    } @elif(AO_DEBUG_MESHLET_CLUSTERS) {
        @if(MESHLET_MESH_MATERIAL_PASS) {
            var rng = in.cluster_id;
            let cluster_color = vec3<f32>(rand_f(&rng), rand_f(&rng), rand_f(&rng));
            out.color = vec4<f32>(cluster_color, 1.0);
        } @else {
            let shade = 0.75 + 0.25 * saturate(dot(pbr_input.N, normalize(vec3<f32>(0.4, 1.0, 0.3))));
            out.color = vec4<f32>(vec3<f32>(0.08, 0.09, 0.12) * shade, pbr_input.material.base_color.a);
        }
    } @elif(AO_DEBUG_MESHLET_TRIANGLE_DENSITY) {
        @if(MESHLET_MESH_MATERIAL_PASS) {
            let color = meshlet_triangle_density_color(in.triangle_area_px, in.triangle_edge_px, pbr_input.N);
            out.color = vec4<f32>(color, 1.0);
        } @else {
            let shade = 0.75 + 0.25 * saturate(dot(pbr_input.N, normalize(vec3<f32>(0.4, 1.0, 0.3))));
            out.color = vec4<f32>(vec3<f32>(0.08, 0.09, 0.12) * shade, pbr_input.material.base_color.a);
        }
    } @elif(AO_DEBUG_VLM_TILES) {
        @if(VIRTUAL_LIGHTMAP) {
            let dist = length(view.world_position.xyz - pbr_input.world_position.xyz);
            let tile_col = virtual_lightmap::virtual_lightmap_debug_tile_color(
                pbr_input.world_position.xyz,
                normalize(pbr_input.world_normal),
                dist,
            );
            out.color = vec4<f32>(tile_col, pbr_input.material.base_color.a);
        } @else {
            out.color = vec4<f32>(vec3<f32>(0.08, 0.09, 0.12), pbr_input.material.base_color.a);
        }
    } @else {
        if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
            out.color = apply_pbr_lighting(pbr_input);
        } else {
            out.color = pbr_input.material.base_color;
        }

        // apply in-shader post processing (fog, alpha-premultiply, and also tonemapping, debanding if the camera is non-hdr)
        // note this does not include fullscreen postprocessing effects like bloom.
        out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    }
}

@if(MATERIAL_OIT_ENABLED) {
    let alpha_mode = pbr_input.material.flags & pbr_types::STANDARD_MATERIAL_FLAGS_ALPHA_MODE_RESERVED_BITS;
    if alpha_mode == pbr_types::STANDARD_MATERIAL_FLAGS_ALPHA_MODE_BLEND {
        // The fragments will only be drawn during the oit resolve pass.
        oit_draw(in.position, vec4(out.color.rgb * out.color.a, out.color.a));
        discard;
    }
    // Both `Premultiplied` and `Add` colors are premultiplied in `premultiply_alpha()`
    if alpha_mode == pbr_types::STANDARD_MATERIAL_FLAGS_ALPHA_MODE_PREMULTIPLIED
        || alpha_mode == pbr_types::STANDARD_MATERIAL_FLAGS_ALPHA_MODE_ADD {
        // The fragments will only be drawn during the oit resolve pass.
        oit_draw(in.position, out.color);
        discard;
    }
}

@if(FORWARD_DECAL)
        out.color.a = min(forward_decal_info.alpha, out.color.a);

        return out;
}

@if(!PREPASS_PIPELINE && MESHLET_MESH_MATERIAL_PASS && AO_DEBUG_MESHLET_TRIANGLE_DENSITY)
fn meshlet_triangle_density_color(area_px: f32, edge_px: f32, world_normal: vec3<f32>) -> vec3<f32> {
    let log_area = log2(max(area_px, 1e-4));

    let c_magenta = vec3<f32>(1.0, 0.02, 0.70); // <= 0.25 px^2 (extreme micro-poly)
    let c_red     = vec3<f32>(0.95, 0.03, 0.02); // ~1 px^2 (sub-pixel threshold)
    let c_yellow  = vec3<f32>(0.95, 0.65, 0.02); // ~4 px^2 (dense)
    let c_green   = vec3<f32>(0.04, 0.65, 0.08); // ~16 px^2 (optimal)
    let c_cyan    = vec3<f32>(0.02, 0.45, 0.78); // ~64 px^2 (coarse)
    let c_blue    = vec3<f32>(0.02, 0.06, 0.55); // >= 256 px^2 (very large)

    var color: vec3<f32>;
    if log_area < 0.0 {
        color = mix(c_magenta, c_red, saturate((log_area + 2.0) * 0.5));
    } else if log_area < 2.0 {
        color = mix(c_red, c_yellow, saturate(log_area * 0.5));
    } else if log_area < 4.0 {
        color = mix(c_yellow, c_green, saturate((log_area - 2.0) * 0.5));
    } else if log_area < 6.0 {
        color = mix(c_green, c_cyan, saturate((log_area - 4.0) * 0.5));
    } else {
        color = mix(c_cyan, c_blue, saturate((log_area - 6.0) * 0.5));
    }

    // Fade in subtle analytical triangle wireframe only on larger triangles (> 4 px^2)
    let wire_fade = saturate((area_px - 4.0) / 12.0);
    let edge_factor = mix(1.0, smoothstep(0.0, 1.1, edge_px), wire_fade * 0.55);
    let shade = 0.82 + 0.18 * saturate(dot(world_normal, normalize(vec3<f32>(0.4, 1.0, 0.3))));
    return color * edge_factor * shade;
}
