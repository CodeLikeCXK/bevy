//! Vertex/fragment shader for rasterizing large clusters into a visibility buffer.

import package::{
    meshlet::bindings::{
        meshlet_cluster_meshlet_ids,
        meshlets,
        meshlet_cluster_instance_ids,
        meshlet_instance_uniforms,
        meshlet_raster_clusters,
        meshlet_previous_raster_counts,
        meshlet_depth_buffer,
        meshlet_visibility_buffer,
        view,
        get_meshlet_triangle_count,
        get_meshlet_vertex_id,
        get_meshlet_vertex_position,
    },
    render::mesh_functions::mesh_position_local_to_world,
};
import bevy_render::maths::affine3_to_square;
struct HardwareRasterImmediates {
    meshlet_raster_cluster_rightmost_slot: u32,
}
var<immediate> immediates: HardwareRasterImmediates;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @if(MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT)
    @location(0) @interpolate(flat) packed_ids: u32,
}

@vertex
fn vertex(@builtin(instance_index) instance_index: u32, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let cluster_in_draw = meshlet_previous_raster_counts[1] + instance_index;
    let cluster_id = immediates.meshlet_raster_cluster_rightmost_slot - cluster_in_draw;
    let instanced_offset = meshlet_raster_clusters[cluster_id];
    var meshlet = meshlets[instanced_offset.offset];

    let triangle_id = vertex_index / 3u;
    if triangle_id >= get_meshlet_triangle_count(&meshlet) { return dummy_vertex(); }
    let index_id = vertex_index;
    let vertex_id = get_meshlet_vertex_id(meshlet.start_index_id + index_id);

    let instance_uniform = meshlet_instance_uniforms[instanced_offset.instance_id];

    let vertex_position = get_meshlet_vertex_position(&meshlet, vertex_id);
    let world_from_local = affine3_to_square(instance_uniform.world_from_local);
    let world_position = mesh_position_local_to_world(world_from_local, vec4(vertex_position, 1.0));
    let clip_position = view.clip_from_world * vec4(world_position.xyz, 1.0);

    @if(MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT)
    return VertexOutput(
        clip_position,
        (cluster_id << 7u) | triangle_id,
    );
    @else
    return VertexOutput(
        clip_position,
    );
}

@fragment
fn fragment(vertex_output: VertexOutput) {
    let depth = bitcast<u32>(vertex_output.position.z);
    let frag_coord_1d = u32(vertex_output.position.y) * u32(view.viewport.z) + u32(vertex_output.position.x);
    @if(MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT) {
        let prev = atomicMax(&meshlet_depth_buffer[frag_coord_1d], depth);
        if depth >= prev {
            atomicExchange(&meshlet_visibility_buffer[frag_coord_1d], vertex_output.packed_ids);
        }
    }
    @else {
        atomicMax(&meshlet_depth_buffer[frag_coord_1d], depth);
    }
}

fn dummy_vertex() -> VertexOutput {
    @if(MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT)
    return VertexOutput(
        vec4(divide(0.0, 0.0)), // NaN vertex position
        0u,
    );
    @else
    return VertexOutput(
        vec4(divide(0.0, 0.0)),
    );
}

/// Naga doesn't allow divide by zero literals, but this lets us work around it
fn divide(a: f32, b: f32) -> f32 {
    return a / b;
}
