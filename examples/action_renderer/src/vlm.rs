//! 静态表面虚拟光照贴图（Virtual Lightmap SpatialHash Tile `.vlm.bin`）加载与运行时控制模块
//!
//! 职责：
//! 1. 解析 `makemeshlet` 离线烘焙的 `.vlm.bin`（`BVVLM001` 格式，LZ4 压缩的紧凑哈希页表 + `10x10` 带边框 `Rgba16Float` 2D 图集）；
//! 2. 异步加载完成后自动写入 `bevy::pbr::VirtualLightmap` 资源，供 Meshlet 与标准 Mesh3d 渲染管线统一采样；
//! 3. 提供运行时快捷键：
//!    - `[V]`：一键开关 Virtual Lightmap（实时对比画质与帧率）
//!    - `[;]` / `[']`：实时微调 VLM 静态漫反射强度

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bevy::{
    asset::{io::Reader, AssetLoader, LoadContext, RenderAssetUsages},
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    pbr::VirtualLightmap,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use lz4_flex::frame::FrameDecoder;
use thiserror::Error;

const VLM_FILE_MAGIC: &[u8; 8] = b"BVVLM001";
const VLM_FILE_VERSION: u32 = 1;
const VLM_HEADER_BYTES: usize = 48;

#[derive(Asset, TypePath, Clone)]
pub struct VirtualLightmapAsset {
    pub texel_size_min: f32,
    pub level_count: u32,
    pub table_capacity: u32,
    pub used_tile_count: u32,
    pub search_count: u32,
    pub tiles_per_row: u32,
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub tile_table: Arc<[[u32; 2]]>,
    pub atlas_image: Handle<Image>,
}

#[derive(Default, TypePath)]
pub struct VirtualLightmapLoader;

#[derive(Debug, Error)]
pub enum VirtualLightmapLoaderError {
    #[error("读取 .vlm.bin 文件失败: {0}")]
    Io(#[from] std::io::Error),
    #[error("无效的 .vlm.bin 文件魔数（非 BVVLM001）")]
    InvalidMagic,
    #[error("不支持的 .vlm.bin 版本: {0}")]
    UnsupportedVersion(u32),
    #[error(".vlm.bin 数据损坏或长度不足")]
    CorruptedData,
}

impl AssetLoader for VirtualLightmapLoader {
    type Asset = VirtualLightmapAsset;
    type Settings = ();
    type Error = VirtualLightmapLoaderError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        if bytes.len() < VLM_HEADER_BYTES {
            return Err(VirtualLightmapLoaderError::CorruptedData);
        }
        if &bytes[0..8] != VLM_FILE_MAGIC {
            return Err(VirtualLightmapLoaderError::InvalidMagic);
        }

        let read_u32 = |off: usize| -> u32 {
            u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
        };
        let read_f32 = |off: usize| -> f32 {
            f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
        };

        let version = read_u32(8);
        if version != VLM_FILE_VERSION {
            return Err(VirtualLightmapLoaderError::UnsupportedVersion(version));
        }
        let _tile_size = read_u32(12);
        let table_capacity = read_u32(16);
        let used_tile_count = read_u32(20);
        let search_count = read_u32(24);
        let tiles_per_row = read_u32(28);
        let atlas_width = read_u32(32);
        let atlas_height = read_u32(36);
        let level_count = read_u32(40);
        let texel_size_min = read_f32(44);

        if table_capacity == 0
            || !table_capacity.is_power_of_two()
            || atlas_width == 0
            || atlas_height == 0
        {
            return Err(VirtualLightmapLoaderError::CorruptedData);
        }

        let mut decoder = FrameDecoder::new(&bytes[VLM_HEADER_BYTES..]);
        let mut tile_table_vec = vec![[0u32; 2]; table_capacity as usize];
        decoder.read_exact(bytemuck::cast_slice_mut(&mut tile_table_vec))?;

        let atlas_byte_len = (atlas_width as usize) * (atlas_height as usize) * 8;
        let mut atlas_bytes = vec![0u8; atlas_byte_len];
        decoder.read_exact(&mut atlas_bytes)?;

        // 显式使用 ClampToEdge + Linear 硬件双线性采样（配合 10x10 物理砖块 1px 邻居边框实现单次无缝采样）
        let mut sampler_desc = ImageSamplerDescriptor::linear();
        sampler_desc.set_address_mode(ImageAddressMode::ClampToEdge);
        sampler_desc.set_filter(ImageFilterMode::Linear);
        sampler_desc.anisotropy_clamp = 1;

        let mut atlas_img = Image::new(
            Extent3d {
                width: atlas_width,
                height: atlas_height,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            atlas_bytes,
            TextureFormat::Rgba16Float,
            RenderAssetUsages::RENDER_WORLD,
        );
        atlas_img.sampler = ImageSampler::Descriptor(sampler_desc);
        let atlas_handle = load_context.add_labeled_asset("atlas".to_string(), atlas_img);

        Ok(VirtualLightmapAsset {
            texel_size_min,
            level_count,
            table_capacity,
            used_tile_count,
            search_count,
            tiles_per_row,
            atlas_width,
            atlas_height,
            tile_table: tile_table_vec.into(),
            atlas_image: atlas_handle,
        })
    }

    fn extensions(&self) -> &[&str] {
        &["vlm.bin"]
    }
}

/// 扫描场景目录自动发现 `.vlm.bin`（无需额外命令行参数）
pub fn find_vlm_file_in_dir(scene_dir: &Path) -> Option<String> {
    let entries = std::fs::read_dir(scene_dir).ok()?;
    for entry in entries.flatten() {
        let path: PathBuf = entry.path();
        if let Some(fname) = path.file_name().and_then(|s| s.to_str())
            && fname.ends_with(".vlm.bin")
        {
            return Some(fname.to_string());
        }
    }
    None
}

#[derive(Resource, Clone)]
pub struct VlmConfig {
    pub asset_path: Option<String>,
    pub enabled: bool,
    pub intensity: f32,
    pub sun_shadow_strength: f32,
    pub direction_blend: f32,
    pub level0_max_distance: f32,
}

impl Default for VlmConfig {
    fn default() -> Self {
        Self {
            asset_path: None,
            enabled: true,
            intensity: 1.0,
            sun_shadow_strength: 1.0,
            direction_blend: 0.55,
            level0_max_distance: 18.0,
        }
    }
}

#[derive(Resource, Clone, Default)]
pub struct VlmStatus {
    pub loaded: bool,
    pub enabled: bool,
    pub used_tiles: u32,
    pub atlas_size: UVec2,
    pub texel_size_cm: f32,
    pub intensity: f32,
}

#[derive(Resource, Default)]
struct VlmRuntime {
    handle: Option<Handle<VirtualLightmapAsset>>,
    applied: bool,
    revision: u64,
}

pub struct VlmScenePlugin;

impl Plugin for VlmScenePlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<VirtualLightmapAsset>()
            .init_asset_loader::<VirtualLightmapLoader>()
            .init_resource::<VlmStatus>()
            .init_resource::<VlmRuntime>()
            .add_systems(Startup, load_vlm_asset)
            .add_systems(
                Update,
                (activate_vlm_when_ready, toggle_vlm, adjust_vlm_intensity),
            );
    }
}

fn load_vlm_asset(
    asset_server: Res<AssetServer>,
    config: Res<VlmConfig>,
    mut runtime: ResMut<VlmRuntime>,
    mut status: ResMut<VlmStatus>,
) {
    status.enabled = config.enabled && config.asset_path.is_some();
    status.intensity = config.intensity;
    if let Some(ref path) = config.asset_path {
        info!("正在异步加载预构建 Virtual Lightmap: {path}");
        runtime.handle = Some(asset_server.load(path));
    }
}

fn activate_vlm_when_ready(
    mut commands: Commands,
    config: Res<VlmConfig>,
    mut runtime: ResMut<VlmRuntime>,
    mut status: ResMut<VlmStatus>,
    vlm_assets: Res<Assets<VirtualLightmapAsset>>,
) {
    if runtime.applied {
        return;
    }
    let Some(ref handle) = runtime.handle else {
        return;
    };
    let Some(asset) = vlm_assets.get(handle) else {
        return;
    };

    runtime.revision += 1;
    let resource = VirtualLightmap {
        enabled: config.enabled,
        intensity: config.intensity,
        sun_shadow_strength: config.sun_shadow_strength,
        direction_blend: config.direction_blend,
        level0_max_distance: config.level0_max_distance,
        texel_size_min: asset.texel_size_min,
        level_count: asset.level_count,
        table_capacity: asset.table_capacity,
        used_tile_count: asset.used_tile_count,
        search_count: asset.search_count,
        tiles_per_row: asset.tiles_per_row,
        atlas_width: asset.atlas_width,
        atlas_height: asset.atlas_height,
        tile_table: Arc::clone(&asset.tile_table),
        atlas_image: asset.atlas_image.clone(),
        revision: runtime.revision,
    };

    info!(
        "Virtual Lightmap 就绪: {} 砖块 (表长={}, max_probe={}), Atlas {}x{}, Level0={:.1}cm, levels={}",
        asset.used_tile_count,
        asset.table_capacity,
        asset.search_count,
        asset.atlas_width,
        asset.atlas_height,
        asset.texel_size_min * 100.0,
        asset.level_count
    );

    status.loaded = true;
    status.enabled = config.enabled;
    status.used_tiles = asset.used_tile_count;
    status.atlas_size = UVec2::new(asset.atlas_width, asset.atlas_height);
    status.texel_size_cm = asset.texel_size_min * 100.0;
    status.intensity = config.intensity;

    commands.insert_resource(resource);
    runtime.applied = true;
}

fn toggle_vlm(
    keys: Res<ButtonInput<KeyCode>>,
    mut config: ResMut<VlmConfig>,
    mut status: ResMut<VlmStatus>,
    mut vlm: Option<ResMut<VirtualLightmap>>,
) {
    if !keys.just_pressed(KeyCode::KeyV) {
        return;
    }
    let Some(ref mut vlm_res) = vlm else {
        return;
    };
    config.enabled = !config.enabled;
    vlm_res.enabled = config.enabled;
    status.enabled = config.enabled;
    info!(
        "Virtual Lightmap: {}",
        if config.enabled { "ON" } else { "OFF" }
    );
}

fn adjust_vlm_intensity(
    keys: Res<ButtonInput<KeyCode>>,
    mut config: ResMut<VlmConfig>,
    mut status: ResMut<VlmStatus>,
    mut vlm: Option<ResMut<VirtualLightmap>>,
) {
    let down = keys.just_pressed(KeyCode::Semicolon);
    let up = keys.just_pressed(KeyCode::Quote);
    if down == up {
        return;
    }
    let Some(ref mut vlm_res) = vlm else {
        return;
    };
    let next = if up {
        (config.intensity + 0.1).min(5.0)
    } else {
        (config.intensity - 0.1).max(0.0)
    };
    config.intensity = (next * 20.0).round() / 20.0;
    vlm_res.intensity = config.intensity;
    status.intensity = config.intensity;
    info!("Virtual Lightmap 强度调整为: {:.2}x", config.intensity);
}
