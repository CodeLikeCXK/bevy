@group(0) @binding(0) var<storage, read_write> meshlet_depth_buffer: array<u32>;
@if(MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT)
@group(0) @binding(1) var<storage, read_write> meshlet_visibility_buffer: array<u32>;
struct ClearVisibilityBufferImmediates {
    view_size: vec2<u32>,
}
var<immediate> immediates: ClearVisibilityBufferImmediates;

@compute
@workgroup_size(16, 16, 1)
fn clear_visibility_buffer(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= immediates.view_size) { return; }

    let idx = global_id.y * immediates.view_size.x + global_id.x;
    meshlet_depth_buffer[idx] = 0u;
    @if(MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT)
    meshlet_visibility_buffer[idx] = 0xFFFFFFFFu;
}
