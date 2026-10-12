import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;
import package::meshlet::bindings::InstancedOffset;

struct ResolveImmediates {
    view_width: u32,
}
var<immediate> immediates: ResolveImmediates;

@group(0) @binding(0) var<storage, read> meshlet_depth_buffer: array<u32>;
@if(MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT) {
@group(0) @binding(1) var<storage, read> meshlet_raster_clusters: array<InstancedOffset>;  // Per cluster
@group(0) @binding(2) var<storage, read> meshlet_instance_material_ids: array<u32>; // Per entity instance
@group(0) @binding(3) var<storage, read> meshlet_visibility_buffer: array<u32>;
}

/// This pass writes out the depth texture.
@fragment
fn resolve_depth(in: FullscreenVertexOutput) -> @builtin(frag_depth) f32 {
    let frag_coord_1d = u32(in.position.y) * immediates.view_width + u32(in.position.x);
    let depth = meshlet_depth_buffer[frag_coord_1d];
    if depth == 0u { discard; }

    return bitcast<f32>(depth);
}

/// This pass writes out the material depth texture.
@if(MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT)
@fragment
fn resolve_material_depth(in: FullscreenVertexOutput) -> @builtin(frag_depth) f32 {
    let frag_coord_1d = u32(in.position.y) * immediates.view_width + u32(in.position.x);
    let depth = meshlet_depth_buffer[frag_coord_1d];
    if depth == 0u { discard; }

    let packed_ids = meshlet_visibility_buffer[frag_coord_1d];
    if packed_ids == 0xFFFFFFFFu { discard; }

    let cluster_id = packed_ids >> 7u;
    let instance_id = meshlet_raster_clusters[cluster_id].instance_id;
    let material_id = meshlet_instance_material_ids[instance_id];
    return f32(material_id) / 65535.0;
}
