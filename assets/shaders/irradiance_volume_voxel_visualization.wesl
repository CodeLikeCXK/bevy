import bevy_pbr::render::forward_io::VertexOutput;
@if(IRRADIANCE_VOLUME) import bevy_pbr::light_probe::irradiance_volume;
import bevy_pbr::render::mesh_view_bindings;
import bevy_pbr::render::clustered_forward;

struct VoxelVisualizationIrradianceVolumeInfo {
    world_from_voxel: mat4x4<f32>,
    voxel_from_world: mat4x4<f32>,
    resolution: vec3<u32>,
    /// A scale factor that's applied to the diffuse and specular light from the
    /// light probe. This is in units of cd/m² (candela per square meter).
    intensity: f32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(100)
var<uniform> irradiance_volume_info: VoxelVisualizationIrradianceVolumeInfo;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    @if(IRRADIANCE_VOLUME) {
        // Look up the irradiance volume range in the cluster list.
        let view_z = dot(vec4<f32>(
            mesh_view_bindings::view.view_from_world[0].z,
            mesh_view_bindings::view.view_from_world[1].z,
            mesh_view_bindings::view.view_from_world[2].z,
            mesh_view_bindings::view.view_from_world[3].z
        ), mesh.world_position);
        let cluster_index =
            clustered_forward::view_fragment_cluster_index(mesh.position.xy, view_z, false);
        var clusterable_object_index_ranges =
            clustered_forward::unpack_clusterable_object_index_ranges(cluster_index);

        // `irradiance_volume_light()` multiplies by intensity, so cancel it out.
        // If we take intensity into account, the cubes will be way too bright.
        let rgb = irradiance_volume::irradiance_volume_light(
            mesh.world_position.xyz,
            mesh.world_normal,
            &clusterable_object_index_ranges,
        ) / irradiance_volume_info.intensity;

        return vec4<f32>(rgb, 1.0f);
    } @else {
        // No irradiance volume is present in the view yet.
        return vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
    }
}
