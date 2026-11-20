//! Surface tangent-plane spatial-hash Virtual Lightmap (VLM) runtime support.
//!
//! Stores offline-baked static surface irradiance (`RGB`, pre-divided by `PI`) and
//! stationary directional sun soft-shadow visibility (`A`, in `0..=1`) inside a compact
//! open-addressing hash table (`[key, compact_tile_idx]`) pointing into a hardware-filterable
//! `Rgba16Float` 2D atlas of `10x10` physical tiles (`8x8` logical texels + `1px` cross-tile
//! neighbor border for seamless single-tap hardware bilinear filtering).

use alloc::sync::Arc;
use bevy_app::{App, Plugin};
use bevy_asset::{AssetId, Handle};
use bevy_ecs::{
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{Res, ResMut},
};
use bevy_image::Image;
use bevy_math::Vec2;
use bevy_render::{
    render_asset::{prepare_assets, RenderAssets},
    render_resource::{
        binding_types::{
            sampler, storage_buffer_read_only, storage_buffer_read_only_sized, texture_2d,
        },
        BindGroupLayoutEntryBuilder, BindingResource, Buffer, BufferDescriptor, BufferUsages,
        Sampler, SamplerBindingType, ShaderType, StorageBuffer, TextureSampleType, TextureView,
    },
    renderer::{RenderDevice, RenderQueue},
    texture::GpuImage,
    Extract, ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderSystems,
};

use crate::check_views_need_specialization;

/// Main-world resource controlling the active surface spatial-hash Virtual Lightmap.
#[derive(Resource, Clone)]
pub struct VirtualLightmap {
    /// Master toggle for runtime A/B comparison without unloading GPU data.
    pub enabled: bool,
    /// Multiplier applied to baked diffuse irradiance (`RGB`).
    pub intensity: f32,
    /// Blend strength of baked directional sun shadow mask (`A`, `0.0..=1.0`).
    pub sun_shadow_strength: f32,
    /// Normal secondary/primary axis ratio threshold where dual-axis blending begins (`0.0..=1.0`).
    pub direction_blend: f32,
    /// World distance (meters) from camera beyond which sampling switches from Level 0 to Level 1.
    pub level0_max_distance: f32,
    /// Physical size (meters) of a Level 0 texel.
    pub texel_size_min: f32,
    /// Number of baked spatial levels (`1..=3`).
    pub level_count: u32,
    /// Hash table slot count (always a power of two).
    pub table_capacity: u32,
    /// Number of occupied tiles stored in the compact atlas.
    pub used_tile_count: u32,
    /// Maximum linear probing steps required for this hash table.
    pub search_count: u32,
    /// Number of `10x10` tiles packed along each row of `atlas_image`.
    pub tiles_per_row: u32,
    /// Atlas width in pixels (`tiles_per_row * 10`).
    pub atlas_width: u32,
    /// Atlas height in pixels (`tiles_per_col * 10`).
    pub atlas_height: u32,
    /// Open-addressing hash table entries `[checksum_key, compact_tile_index]`.
    pub tile_table: Arc<[[u32; 2]]>,
    /// Hardware-filterable `Rgba16Float` 2D tile atlas.
    pub atlas_image: Handle<Image>,
    /// Monotonic revision bumped whenever `tile_table` changes.
    pub revision: u64,
}

impl VirtualLightmap {
    pub fn is_valid(&self) -> bool {
        self.enabled
            && self.used_tile_count > 0
            && self.table_capacity > 0
            && self.table_capacity.is_power_of_two()
            && self.tile_table.len() == self.table_capacity as usize
            && self.tiles_per_row > 0
            && self.atlas_width > 0
            && self.atlas_height > 0
    }
}

/// GPU uniform parameters matching `VirtualLightmapParams` in `mesh_view_types.wesl`.
#[derive(Clone, Copy, ShaderType, Default, Debug)]
pub struct GpuVirtualLightmapParams {
    pub texel_size_min: f32,
    pub level0_max_distance: f32,
    pub intensity: f32,
    pub sun_shadow_strength: f32,
    pub table_mask: u32,
    pub search_count: u32,
    pub tiles_per_row: u32,
    pub level_count: u32,
    pub inv_atlas_size: Vec2,
    pub direction_blend: f32,
    pub _padding: f32,
}

#[derive(Resource, Default)]
pub(crate) struct ExtractedVirtualLightmap {
    pub valid: bool,
    pub params: GpuVirtualLightmapParams,
    pub tile_table: Option<Arc<[[u32; 2]]>>,
    pub atlas_image_id: Option<AssetId<Image>>,
    pub revision: u64,
}

/// Render-world GPU resources for the active [`VirtualLightmap`].
#[derive(Resource, Default)]
pub struct GpuVirtualLightmap {
    params_buffer: StorageBuffer<GpuVirtualLightmapParams>,
    tile_buffer: Option<Buffer>,
    uploaded_revision: u64,
    atlas_image_id: Option<AssetId<Image>>,
    ready: bool,
}

impl GpuVirtualLightmap {
    /// Returns `true` when the VLM uniform, hash table SSBO, and atlas GPU texture are all ready
    /// to be bound in `@group(1)` for the current frame.
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    pub(crate) fn bindings<'a>(
        &'a self,
        images: &'a RenderAssets<GpuImage>,
    ) -> Option<(
        BindingResource<'a>,
        BindingResource<'a>,
        &'a TextureView,
        &'a Sampler,
    )> {
        if !self.ready {
            return None;
        }
        let params_binding = self.params_buffer.binding()?;
        let tile_buffer = self.tile_buffer.as_ref()?;
        let atlas_id = self.atlas_image_id?;
        let gpu_image = images.get(atlas_id)?;
        Some((
            params_binding,
            tile_buffer.as_entire_binding(),
            &gpu_image.texture_view,
            &gpu_image.sampler,
        ))
    }
}

pub(crate) fn get_bind_group_layout_entries() -> [BindGroupLayoutEntryBuilder; 4] {
    [
        storage_buffer_read_only::<GpuVirtualLightmapParams>(false),
        storage_buffer_read_only_sized(false, None),
        texture_2d(TextureSampleType::Float { filterable: true }),
        sampler(SamplerBindingType::Filtering),
    ]
}

pub struct VirtualLightmapPlugin;

impl Plugin for VirtualLightmapPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<ExtractedVirtualLightmap>()
            .init_gpu_resource::<GpuVirtualLightmap>()
            .add_systems(ExtractSchedule, extract_virtual_lightmap)
            .add_systems(
                Render,
                prepare_virtual_lightmap
                    .in_set(RenderSystems::PrepareAssets)
                    .after(prepare_assets::<GpuImage>)
                    .before(check_views_need_specialization),
            );
    }
}

fn extract_virtual_lightmap(
    vlm: Extract<Option<Res<VirtualLightmap>>>,
    mut extracted: ResMut<ExtractedVirtualLightmap>,
) {
    let Some(vlm) = vlm.as_deref() else {
        extracted.valid = false;
        extracted.tile_table = None;
        extracted.atlas_image_id = None;
        return;
    };

    if !vlm.is_valid() {
        extracted.valid = false;
        extracted.atlas_image_id = None;
        return;
    }

    extracted.valid = true;
    extracted.params = GpuVirtualLightmapParams {
        texel_size_min: vlm.texel_size_min.max(1e-4),
        level0_max_distance: vlm.level0_max_distance.max(1.0),
        intensity: vlm.intensity.max(0.0),
        sun_shadow_strength: vlm.sun_shadow_strength.clamp(0.0, 1.0),
        table_mask: vlm.table_capacity - 1,
        search_count: vlm.search_count.clamp(1, 256),
        tiles_per_row: vlm.tiles_per_row,
        level_count: vlm.level_count.clamp(1, 4),
        inv_atlas_size: Vec2::new(1.0 / vlm.atlas_width as f32, 1.0 / vlm.atlas_height as f32),
        direction_blend: vlm.direction_blend.clamp(0.0, 0.99),
        _padding: 0.0,
    };
    extracted.atlas_image_id = Some(vlm.atlas_image.id());
    if extracted.revision != vlm.revision || extracted.tile_table.is_none() {
        extracted.tile_table = Some(Arc::clone(&vlm.tile_table));
        extracted.revision = vlm.revision;
    }
}

fn prepare_virtual_lightmap(
    extracted: Res<ExtractedVirtualLightmap>,
    images: Res<RenderAssets<GpuImage>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut gpu_vlm: ResMut<GpuVirtualLightmap>,
) {
    if !extracted.valid {
        gpu_vlm.ready = false;
        gpu_vlm.atlas_image_id = None;
        return;
    }

    let Some(atlas_id) = extracted.atlas_image_id else {
        gpu_vlm.ready = false;
        gpu_vlm.atlas_image_id = None;
        return;
    };

    if images.get(atlas_id).is_none() {
        gpu_vlm.ready = false;
        gpu_vlm.atlas_image_id = Some(atlas_id);
        return;
    }

    gpu_vlm.params_buffer.set(extracted.params);
    gpu_vlm
        .params_buffer
        .write_buffer(&render_device, &render_queue);

    if (gpu_vlm.tile_buffer.is_none() || gpu_vlm.uploaded_revision != extracted.revision)
        && let Some(ref table) = extracted.tile_table
    {
        let byte_data: &[u8] = bytemuck::cast_slice(table);
        let buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("virtual_lightmap_tile_table"),
            size: byte_data.len().max(8) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        render_queue.write_buffer(&buffer, 0, byte_data);
        gpu_vlm.tile_buffer = Some(buffer);
        gpu_vlm.uploaded_revision = extracted.revision;
    }

    gpu_vlm.atlas_image_id = Some(atlas_id);
    gpu_vlm.ready = gpu_vlm.params_buffer.binding().is_some() && gpu_vlm.tile_buffer.is_some();
}
