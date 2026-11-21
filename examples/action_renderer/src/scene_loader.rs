//! 场景清单加载与 Chunk 区块装配模块
//!
//! 兼容 `makemeshlet --scene` 导出的 v1 / v2 `.meshlet_scene.json`：
//! 1. 创建去重后的 PBR `StandardMaterial`（支持 Opaque / AlphaMode::Mask / AlphaMode::Blend）。
//! 2. 按 `chunk` 名称聚合同区块内的 `Meshlet` 与 `Mesh3d_LOD` 实例，计算区块合并 AABB，生成 `SceneChunk` 交由 `StreamingPlugin` 调度。
//! 3. 提取并生成场景灯光与 `GameplayMarker` 标记点。

use std::collections::HashMap;
use std::f32::consts::PI;
use std::path::{Path, PathBuf};

use bevy::{
    asset::RenderAssetUsages,
    image::{
        ImageAddressMode, ImageFilterMode, ImageLoaderSettings, ImageSampler,
        ImageSamplerDescriptor,
    },
    light::CascadeShadowConfigBuilder,
    math::Affine2,
    prelude::*,
    render::render_resource::{TextureDimension, TextureFormat},
};
use serde::Deserialize;

use crate::streaming::{ChunkMesh3dItem, ChunkMeshletItem, SceneChunk};

#[derive(Debug, Clone, Deserialize)]
pub struct MeshletSceneManifest {
    #[serde(default)]
    pub version: u32,
    #[serde(default)]
    pub source_file: String,
    pub materials: Vec<SceneMaterial>,
    #[serde(default)]
    pub meshlet_instances: Vec<SceneMeshletInstance>,
    #[serde(default)]
    pub mesh3d_instances: Vec<SceneMesh3dInstance>,
    #[serde(default)]
    pub lights: Vec<SceneLight>,
    #[serde(default)]
    pub gameplay_markers: Vec<SceneGameplayMarker>,
    #[serde(default)]
    pub skipped_non_opaque: Vec<serde_json::Value>,
    #[serde(default)]
    pub world_hdri: Option<SceneWorldHdri>,
    #[serde(default)]
    pub vlm_file: Option<String>,
}

fn default_hdri_strength() -> f32 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct SceneWorldHdri {
    pub image: String,
    #[serde(default = "default_hdri_strength")]
    pub strength: f32,
    #[serde(default)]
    pub yaw_degrees: f32,
    #[serde(default)]
    pub projection: String,
}

fn default_alpha_mode() -> String {
    "Opaque".to_string()
}

fn default_alpha_cutoff() -> f32 {
    0.5
}

fn default_chunk() -> String {
    "main".to_string()
}

fn default_mesh3d_tag() -> String {
    "MESH3D_LOD".to_string()
}

fn default_wrap_pair() -> [String; 2] {
    ["Repeat".to_string(), "Repeat".to_string()]
}

fn default_true() -> bool {
    true
}

fn default_ior() -> f32 {
    1.5
}

fn default_reflectance() -> f32 {
    0.5
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct SceneUvTransform {
    pub offset: [f32; 2],
    pub scale: [f32; 2],
    pub rotation: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SceneMaterial {
    pub id: u32,
    pub name: String,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    #[serde(default = "default_alpha_mode")]
    pub alpha_mode: String,
    #[serde(default = "default_alpha_cutoff")]
    pub alpha_cutoff: f32,
    pub base_color_texture: Option<String>,
    pub normal_texture: Option<String>,
    pub metallic_roughness_texture: Option<String>,
    pub occlusion_texture: Option<String>,
    pub emissive_texture: Option<String>,
    pub double_sided: bool,
    #[serde(default = "default_wrap_pair")]
    pub wrap_mode: [String; 2],
    #[serde(default)]
    pub unlit: bool,
    #[serde(default = "default_ior")]
    pub ior: f32,
    #[serde(default = "default_reflectance")]
    pub reflectance: f32,
    #[serde(default)]
    pub flip_normal_map_y: bool,
    #[serde(default)]
    pub uv_transform: Option<SceneUvTransform>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SceneMeshletInstance {
    pub name: String,
    #[serde(default = "default_chunk")]
    pub chunk: String,
    pub mesh: String,
    pub material_id: u32,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    #[serde(default)]
    pub dynamic: bool,
    #[serde(default = "default_true")]
    pub cast_shadows: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SceneMesh3dInstance {
    pub name: String,
    #[serde(default = "default_chunk")]
    pub chunk: String,
    #[serde(default = "default_mesh3d_tag")]
    pub tag: String,
    pub mesh: String,
    pub material_id: u32,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    #[serde(default)]
    pub lod_distances: Vec<f32>,
    #[serde(default = "default_true")]
    pub cast_shadows: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SceneLight {
    pub name: String,
    pub kind: String,
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: Option<f32>,
    pub inner_cone_angle: Option<f32>,
    pub outer_cone_angle: Option<f32>,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
}

#[derive(Debug, Clone, Deserialize)]
pub struct SceneGameplayMarker {
    pub name: String,
    pub tag: String,
    pub spawn_group: String,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

/// 挂载在世界中的 Gameplay 标记点组件（供后续编辑器与游戏逻辑读取）
#[derive(Component, Debug, Clone)]
pub struct GameplayMarker {
    pub tag: String,
    pub spawn_group: String,
}

#[derive(Resource, Clone)]
pub struct MeshletSceneConfig {
    /// 磁盘上的场景目录（包含 meshes/ 与可选的 .meshlet_scene.json）
    pub scene_dir: PathBuf,
    /// 可选的清单文件名（未指定时自动扫描 scene_dir 下的 *.meshlet_scene.json）
    pub manifest_file: Option<String>,
    /// 注册到 AssetServer 的资产源名称，例如 "meshlet_scene"
    pub asset_source: String,
    /// 场景最大实时局部灯光激活数量上限（优先激活距相机最近的灯，0 表示不限制）
    pub max_lights: usize,
    /// 是否开启主平行光阴影投射
    pub shadows_enabled: bool,
    /// 平行光阴影贴图尺寸（像素），用于按纹素物理尺寸自动缩放 normal_bias
    pub shadow_map_size: usize,
    /// 平行光级联数（默认 2 级联兼顾近景精度与远距显示）
    pub shadow_cascades: usize,
    /// 第 1 级联近景覆盖半径（米）
    pub first_cascade_distance: f32,
    /// 平行光阴影覆盖最远半径（米）
    pub shadow_distance: f32,
    /// 可选显式覆盖阴影法线偏移
    pub shadow_normal_bias: Option<f32>,
    /// 可选显式覆盖阴影深度偏移
    pub shadow_depth_bias: Option<f32>,
    /// 是否允许植物/镂空（AlphaMask/Blend）Mesh3d 投射阴影
    pub foliage_shadows: bool,
    /// 灯光统一强度倍率
    pub light_intensity_scale: f32,
    /// Sun/Directional 导出强度 <= 0.01 时的兜底照度 (lux)
    pub default_sun_illuminance: f32,
    /// Point/Spot 导出强度 <= 0.01 时的兜底流明 (lumens)
    pub default_point_lumens: f32,
    /// 是否检测到并启用了预构建 Virtual Lightmap（自动开启静态场景实时阴影减负与 Stationary 局部光模式）
    pub vlm_active: bool,
    /// 即使启用了 VLM 也强制静态物体继续投射实时 CSM 阴影（默认 false 以最大化 60fps 性能）
    pub static_realtime_shadows: bool,
}

/// 挂在场景局部光源（Point/Spot）上，供运行时按相机距离平滑渐入/渐出激活，避免硬切换跳变
#[derive(Component, Debug, Clone, Copy)]
struct ManagedLocalLight {
    base_intensity: f32,
    current_intensity: f32,
    active: bool,
}

/// 场景贴图运行时 Mipmap 生成语义：不同语义采用不同滤波规则（sRGB 线性滤波 / Normal 重归一化 / Linear 直接平均）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextureMipRole {
    SrgbColor,
    NormalMap,
    LinearData,
}

#[derive(Resource, Default)]
struct PendingSceneMipmaps {
    /// 待补全 Mipmap 链的场景贴图（按 AssetId 去重，ORM 共用句柄只生成一次）
    pending: HashMap<AssetId<Image>, TextureMipRole>,
    generated_count: usize,
}

pub struct MeshletScenePlugin;

impl Plugin for MeshletScenePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PendingSceneMipmaps>()
            .add_systems(Startup, setup_meshlet_scene)
            .add_systems(
                Update,
                (generate_scene_texture_mipmaps, update_active_local_lights),
            );
    }
}

pub fn find_manifest_path(scene_dir: &Path, manifest_file: Option<&str>) -> Option<PathBuf> {
    if let Some(name) = manifest_file {
        let p = scene_dir.join(name);
        if p.exists() {
            return Some(p);
        }
    }
    let entries = std::fs::read_dir(scene_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(fname) = path.file_name().and_then(|s| s.to_str())
            && fname.ends_with(".meshlet_scene.json")
        {
            return Some(path);
        }
    }
    None
}

fn parse_wrap_mode(mode: &str) -> ImageAddressMode {
    match mode {
        "ClampToEdge" => ImageAddressMode::ClampToEdge,
        "MirroredRepeat" | "MirrorRepeat" => ImageAddressMode::MirrorRepeat,
        _ => ImageAddressMode::Repeat,
    }
}

/// 加载场景贴图：配置色彩空间（sRGB vs Linear）、三线性+各向异性采样器，并登记至运行时 Mipmap 生成队列
fn load_scene_texture(
    asset_server: &AssetServer,
    pending_mips: &mut PendingSceneMipmaps,
    asset_uri: &str,
    role: TextureMipRole,
    wrap_mode: &[String; 2],
) -> Handle<Image> {
    let is_srgb = matches!(role, TextureMipRole::SrgbColor);
    let wrap_u = parse_wrap_mode(&wrap_mode[0]);
    let wrap_v = parse_wrap_mode(&wrap_mode[1]);
    let handle: Handle<Image> = asset_server
        .load_builder()
        .with_settings(move |s: &mut ImageLoaderSettings| {
            s.is_srgb = is_srgb;
            s.asset_usage = RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD;
            let mut desc = ImageSamplerDescriptor::linear();
            desc.address_mode_u = wrap_u;
            desc.address_mode_v = wrap_v;
            desc.address_mode_w = wrap_u;
            desc.set_filter(ImageFilterMode::Linear);
            desc.mipmap_filter = ImageFilterMode::Linear;
            desc.anisotropy_clamp = 4;
            s.sampler = ImageSampler::Descriptor(desc);
        })
        .load(asset_uri.to_string());
    pending_mips.pending.entry(handle.id()).or_insert(role);
    handle
}

/// 将局部空间 AABB 的 8 个角点变换到世界空间，求紧致世界 AABB
fn transform_aabb(tf: &Transform, local_min: Vec3, local_max: Vec3) -> (Vec3, Vec3) {
    let mut world_min = Vec3::splat(f32::INFINITY);
    let mut world_max = Vec3::splat(f32::NEG_INFINITY);
    for x in [local_min.x, local_max.x] {
        for y in [local_min.y, local_max.y] {
            for z in [local_min.z, local_max.z] {
                let p = tf.transform_point(Vec3::new(x, y, z));
                world_min = world_min.min(p);
                world_max = world_max.max(p);
            }
        }
    }
    (world_min, world_max)
}

#[derive(Default)]
struct ChunkAccumulator {
    aabb_min: Vec3,
    aabb_max: Vec3,
    has_bounds: bool,
    meshlets: Vec<ChunkMeshletItem>,
    mesh3ds: Vec<ChunkMesh3dItem>,
}

impl ChunkAccumulator {
    fn expand_bounds(&mut self, min: Vec3, max: Vec3) {
        if !self.has_bounds {
            self.aabb_min = min;
            self.aabb_max = max;
            self.has_bounds = true;
        } else {
            self.aabb_min = self.aabb_min.min(min);
            self.aabb_max = self.aabb_max.max(max);
        }
    }
}

fn setup_meshlet_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
    mut pending_mips: ResMut<PendingSceneMipmaps>,
    config: Res<MeshletSceneConfig>,
) {
    let prefix = format!("{}://", config.asset_source);

    if let Some(manifest_path) =
        find_manifest_path(&config.scene_dir, config.manifest_file.as_deref())
    {
        info!("加载场景清单: {}", manifest_path.display());
        let json_bytes = match std::fs::read(&manifest_path) {
            Ok(b) => b,
            Err(e) => {
                error!("读取清单文件失败: {e}");
                return;
            }
        };
        let manifest: MeshletSceneManifest = match serde_json::from_slice(&json_bytes) {
            Ok(m) => m,
            Err(e) => {
                error!("解析清单 JSON 失败: {e}");
                return;
            }
        };

        // 1. 构建去重 PBR 材质表（支持 Opaque / Mask / Blend、Albedo+Normal+ORM 三贴图与 UV 变换）
        let mut material_handles = Vec::with_capacity(manifest.materials.len());
        for mat in &manifest.materials {
            let base_color_texture = mat.base_color_texture.as_ref().map(|p| {
                load_scene_texture(
                    &asset_server,
                    &mut pending_mips,
                    &format!("{prefix}{p}"),
                    TextureMipRole::SrgbColor,
                    &mat.wrap_mode,
                )
            });
            let normal_map_texture = mat.normal_texture.as_ref().map(|p| {
                load_scene_texture(
                    &asset_server,
                    &mut pending_mips,
                    &format!("{prefix}{p}"),
                    TextureMipRole::NormalMap,
                    &mat.wrap_mode,
                )
            });
            let metallic_roughness_texture = mat.metallic_roughness_texture.as_ref().map(|p| {
                load_scene_texture(
                    &asset_server,
                    &mut pending_mips,
                    &format!("{prefix}{p}"),
                    TextureMipRole::LinearData,
                    &mat.wrap_mode,
                )
            });
            let occlusion_texture = mat.occlusion_texture.as_ref().map(|p| {
                load_scene_texture(
                    &asset_server,
                    &mut pending_mips,
                    &format!("{prefix}{p}"),
                    TextureMipRole::LinearData,
                    &mat.wrap_mode,
                )
            });
            let emissive_texture = mat.emissive_texture.as_ref().map(|p| {
                load_scene_texture(
                    &asset_server,
                    &mut pending_mips,
                    &format!("{prefix}{p}"),
                    TextureMipRole::SrgbColor,
                    &mat.wrap_mode,
                )
            });

            let alpha_mode = match mat.alpha_mode.as_str() {
                "Mask" => AlphaMode::Mask(mat.alpha_cutoff),
                "Blend" => AlphaMode::Blend,
                _ => AlphaMode::Opaque,
            };
            let uv_transform = mat.uv_transform.map_or(Affine2::IDENTITY, |t| {
                Affine2::from_scale_angle_translation(
                    t.scale.into(),
                    -t.rotation,
                    t.offset.into(),
                )
            });

            let handle = standard_materials.add(StandardMaterial {
                base_color: Color::linear_rgba(
                    mat.base_color[0],
                    mat.base_color[1],
                    mat.base_color[2],
                    mat.base_color[3],
                ),
                metallic: mat.metallic,
                perceptual_roughness: mat.roughness,
                emissive: LinearRgba::rgb(mat.emissive[0], mat.emissive[1], mat.emissive[2]),
                alpha_mode,
                base_color_texture,
                normal_map_texture,
                metallic_roughness_texture,
                occlusion_texture,
                emissive_texture,
                double_sided: mat.double_sided,
                cull_mode: if mat.double_sided {
                    None
                } else {
                    Some(bevy::render::render_resource::Face::Back)
                },
                unlit: mat.unlit,
                ior: mat.ior,
                reflectance: mat.reflectance,
                flip_normal_map_y: mat.flip_normal_map_y,
                uv_transform,
                ..default()
            });
            material_handles.push(handle);
        }

        let fallback_mat = standard_materials.add(StandardMaterial::default());
        let mut chunk_map: HashMap<String, ChunkAccumulator> = HashMap::new();

        // 2. 将 Meshlet 实例按 chunk 聚合并累计世界空间包围盒
        for inst in &manifest.meshlet_instances {
            let mat_handle = material_handles
                .get(inst.material_id as usize)
                .cloned()
                .unwrap_or_else(|| fallback_mat.clone());
            let tf = Transform {
                translation: Vec3::from_array(inst.translation),
                rotation: Quat::from_array(inst.rotation),
                scale: Vec3::from_array(inst.scale),
            };
            let (wmin, wmax) = transform_aabb(
                &tf,
                Vec3::from_array(inst.aabb_min),
                Vec3::from_array(inst.aabb_max),
            );

            let static_casts_realtime_shadow = !config.vlm_active || config.static_realtime_shadows;
            let cast_shadows = config.shadows_enabled
                && inst.cast_shadows
                && (inst.dynamic || static_casts_realtime_shadow);

            let acc = chunk_map.entry(inst.chunk.clone()).or_default();
            acc.expand_bounds(wmin, wmax);
            acc.meshlets.push(ChunkMeshletItem {
                asset_path: format!("{prefix}{}", inst.mesh),
                material: mat_handle,
                transform: tf,
                cast_shadows,
            });
        }

        // 3. 将 Mesh3d LOD / 半透明实例按 chunk 聚合
        for inst in &manifest.mesh3d_instances {
            let mat_handle = material_handles
                .get(inst.material_id as usize)
                .cloned()
                .unwrap_or_else(|| fallback_mat.clone());
            let tf = Transform {
                translation: Vec3::from_array(inst.translation),
                rotation: Quat::from_array(inst.rotation),
                scale: Vec3::from_array(inst.scale),
            };
            let (wmin, wmax) = transform_aabb(
                &tf,
                Vec3::from_array(inst.aabb_min),
                Vec3::from_array(inst.aabb_max),
            );

            let is_transparent = inst.tag == "MESH3D_TRANS";
            let mat_is_non_opaque = manifest
                .materials
                .get(inst.material_id as usize)
                .is_some_and(|m| m.alpha_mode != "Opaque");
            let static_casts_realtime_shadow = !config.vlm_active || config.static_realtime_shadows;
            let cast_shadows = config.shadows_enabled
                && inst.cast_shadows
                && static_casts_realtime_shadow
                && (config.foliage_shadows || (!is_transparent && !mat_is_non_opaque));
            let acc = chunk_map.entry(inst.chunk.clone()).or_default();
            acc.expand_bounds(wmin, wmax);
            acc.mesh3ds.push(ChunkMesh3dItem {
                name: inst.name.clone(),
                asset_path: format!("{prefix}{}", inst.mesh),
                material: mat_handle,
                transform: tf,
                lod_distances: inst.lod_distances.clone(),
                is_transparent,
                cast_shadows,
                lod_asset: None,
                lods_spawned: false,
            });
        }

        // 4. 生成所有 SceneChunk 实体交由 StreamingPlugin 统一流送调度
        let chunk_count = chunk_map.len();
        for (chunk_name, acc) in chunk_map {
            commands.spawn((
                Name::new(format!("Chunk:{chunk_name}")),
                SceneChunk {
                    chunk_name,
                    aabb_min: if acc.has_bounds { acc.aabb_min } else { Vec3::ZERO },
                    aabb_max: if acc.has_bounds { acc.aabb_max } else { Vec3::ZERO },
                    meshlets: acc.meshlets,
                    mesh3ds: acc.mesh3ds,
                    meshlet_handles: HashMap::new(),
                    spawned_entities: Vec::new(),
                    loaded: false,
                    pending_meshlet_cursor: 0,
                    out_of_range_timer: 0.0,
                },
            ));
        }

        // 5. 生成清单中的 Gameplay 标记点
        for marker in &manifest.gameplay_markers {
            commands.spawn((
                Name::new(marker.name.clone()),
                Transform {
                    translation: Vec3::from_array(marker.translation),
                    rotation: Quat::from_array(marker.rotation),
                    scale: Vec3::from_array(marker.scale),
                },
                GameplayMarker {
                    tag: marker.tag.clone(),
                    spawn_group: marker.spawn_group.clone(),
                },
            ));
        }

        // 6. 生成清单中的灯光：
        //    - Directional (Sun) 常驻，若导出强度 <= 0.01（如 FBX 导入 Blender 未设置 Strength）则应用兜底照度
        //    - Point / Spot 进行 glTF candela -> Bevy lumens (* 4*PI) 换算，全部生成并交由运行期按相机距离动态激活 Top-N
        let mut dir_count = 0usize;
        let mut local_count = 0usize;
        let mut zero_fallback_count = 0usize;
        let mut skipped_hdri_helper = 0usize;
        let mut dir_shadow_assigned = false;

        if manifest.lights.is_empty() {
            warn!("场景清单未包含灯光，生成默认 DirectionalLight");
            spawn_default_light(&mut commands, config.shadows_enabled, &config);
            dir_count = 1;
        } else {
            for (idx, light) in manifest.lights.iter().enumerate() {
                let name_lower = light.name.to_lowercase();
                if light.kind != "Directional"
                    && name_lower.contains("hdri_sky")
                    && light.intensity <= 0.01
                {
                    skipped_hdri_helper += 1;
                    continue;
                }

                let color = Color::linear_rgb(light.color[0], light.color[1], light.color[2]);
                let mut rotation = Quat::from_array(light.rotation);
                if rotation.length_squared() > 1e-6 {
                    rotation = rotation.normalize();
                } else {
                    rotation = Quat::IDENTITY;
                }
                let tf = Transform {
                    translation: Vec3::from_array(light.translation),
                    rotation,
                    scale: Vec3::ONE,
                };

                match light.kind.as_str() {
                    "Directional" => {
                        let raw = light.intensity;
                        let illuminance = if raw <= 0.01 && config.default_sun_illuminance > 0.0 {
                            zero_fallback_count += 1;
                            config.default_sun_illuminance * config.light_intensity_scale
                        } else {
                            raw * config.light_intensity_scale
                        };
                        let enable_shadow = config.shadows_enabled && !dir_shadow_assigned;
                        dir_shadow_assigned = true;
                        dir_count += 1;

                        let (shadow_depth_bias, shadow_normal_bias, est_texel) =
                            resolve_shadow_biases(&config);
                        // 当 VLM 静态软阴影掩码激活且静态物体免投实时阴影时，实时 CSM 仅服务动态角色，自动收缩为 1 级联近景高精阴影
                        let char_only_csm = config.vlm_active && !config.static_realtime_shadows;
                        let num_cascades = if char_only_csm {
                            1
                        } else {
                            config.shadow_cascades.clamp(1, 4)
                        };
                        let max_dist = if char_only_csm {
                            config.first_cascade_distance.max(14.0).min(config.shadow_distance.max(14.0))
                        } else {
                            config.shadow_distance.max(6.0)
                        };
                        let first_bound = config.first_cascade_distance.clamp(3.0, max_dist * 0.8);
                        if enable_shadow {
                            info!(
                                "  [阴影配置] {}px × {} casc (近距 {:.1}m / 最远 {:.1}m, 首级纹素约 {:.3}m{}): depth_bias={:.3}, normal_bias={:.3}",
                                config.shadow_map_size,
                                num_cascades,
                                first_bound,
                                max_dist,
                                est_texel,
                                if char_only_csm { ", VLM 静态阴影减负: 仅动态角色投射实时阴影" } else { "" },
                                shadow_depth_bias,
                                shadow_normal_bias
                            );
                            if est_texel > 0.15 {
                                warn!(
                                    "首级阴影单纹素覆盖高达 {:.2}m（建议调小 --first-cascade-distance 或提高 --shadow-size）",
                                    est_texel
                                );
                            }
                        }

                        if idx < 8 || zero_fallback_count <= 2 {
                            info!(
                                "  [灯光] Directional '{}': raw={:.3} -> {:.1} lux, shadow={}, pos={:?}",
                                light.name, raw, illuminance, enable_shadow, light.translation
                            );
                        }

                        let mut entity_cmds = commands.spawn((
                            Name::new(light.name.clone()),
                            DirectionalLight {
                                color,
                                illuminance,
                                shadow_maps_enabled: enable_shadow,
                                shadow_depth_bias,
                                shadow_normal_bias,
                                ..default()
                            },
                            tf,
                        ));
                        if enable_shadow {
                            entity_cmds.insert(
                                CascadeShadowConfigBuilder {
                                    num_cascades,
                                    first_cascade_far_bound: first_bound,
                                    maximum_distance: max_dist,
                                    ..default()
                                }
                                .build(),
                            );
                        }
                    }
                    "Spot" => {
                        let raw = light.intensity;
                        let lumens = if raw <= 0.01 && config.default_point_lumens > 0.0 {
                            zero_fallback_count += 1;
                            config.default_point_lumens * config.light_intensity_scale
                        } else {
                            // glTF KHR_lights_punctual 单位为 candela，Bevy 需要 lumens (4 * pi * cd)
                            raw * 4.0 * PI * config.light_intensity_scale
                        };
                        let range = match light.range {
                            Some(r) if r > 0.05 => r,
                            _ => 10.0,
                        };
                        let inner_angle = light.inner_cone_angle.unwrap_or(0.0).clamp(0.0, PI * 0.45);
                        let outer_angle = light
                            .outer_cone_angle
                            .unwrap_or(0.785)
                            .max(inner_angle + 0.05)
                            .clamp(0.05, PI * 0.49);
                        local_count += 1;

                        if local_count <= 4 {
                            info!(
                                "  [灯光] Spot '{}': raw={:.3} cd -> {:.1} lm, range={:.1}m, pos={:?}",
                                light.name, raw, lumens, range, light.translation
                            );
                        }

                        let init_intensity = if config.max_lights == 0 { lumens } else { 0.0 };
                        commands.spawn((
                            Name::new(light.name.clone()),
                            SpotLight {
                                color,
                                intensity: init_intensity,
                                range,
                                inner_angle,
                                outer_angle,
                                shadow_maps_enabled: false,
                                affects_lightmapped_mesh_diffuse: !config.vlm_active,
                                ..default()
                            },
                            ManagedLocalLight {
                                base_intensity: lumens,
                                current_intensity: init_intensity,
                                active: config.max_lights == 0,
                            },
                            tf,
                        ));
                    }
                    _ => {
                        let raw = light.intensity;
                        let lumens = if raw <= 0.01 && config.default_point_lumens > 0.0 {
                            zero_fallback_count += 1;
                            config.default_point_lumens * config.light_intensity_scale
                        } else {
                            // glTF KHR_lights_punctual 单位为 candela，Bevy 需要 lumens (4 * pi * cd)
                            raw * 4.0 * PI * config.light_intensity_scale
                        };
                        let range = match light.range {
                            Some(r) if r > 0.05 => r,
                            _ => 10.0,
                        };
                        local_count += 1;

                        if local_count <= 4 {
                            info!(
                                "  [灯光] Point '{}': raw={:.3} cd -> {:.1} lm, range={:.1}m, pos={:?}",
                                light.name, raw, lumens, range, light.translation
                            );
                        }

                        let init_intensity = if config.max_lights == 0 { lumens } else { 0.0 };
                        commands.spawn((
                            Name::new(light.name.clone()),
                            PointLight {
                                color,
                                intensity: init_intensity,
                                range,
                                shadow_maps_enabled: false,
                                affects_lightmapped_mesh_diffuse: !config.vlm_active,
                                ..default()
                            },
                            ManagedLocalLight {
                                base_intensity: lumens,
                                current_intensity: init_intensity,
                                active: config.max_lights == 0,
                            },
                            tf,
                        ));
                    }
                }
            }

            if dir_count == 0 {
                warn!("场景清单中未发现 Directional 光源，补充默认弱日光用于基础轮廓");
                spawn_default_light(&mut commands, config.shadows_enabled, &config);
            }
            if zero_fallback_count > 0 {
                info!(
                    "检测到 {} 个导出强度为 0 的灯光（常见于 FBX 导入 Blender），已应用默认兜底光强（可用 --default-sun-illuminance / --default-point-lumens 调整）",
                    zero_fallback_count
                );
            }
            if skipped_hdri_helper > 0 {
                info!("已过滤 {} 个零强度 HDRI_SKY 辅助节点", skipped_hdri_helper);
            }
        }

        info!(
            "场景初始化完成 (v{}): {} 个区块, {} 个材质, {} 个 Meshlet 实例, {} 个 Mesh3d LOD 实例, 灯光={} 平行光 + {} 局部光(同时激活上限={}), {} 个 Gameplay 标记",
            manifest.version,
            chunk_count,
            manifest.materials.len(),
            manifest.meshlet_instances.len(),
            manifest.mesh3d_instances.len(),
            dir_count,
            local_count,
            if config.max_lights == 0 {
                "不限".to_string()
            } else {
                config.max_lights.to_string()
            },
            manifest.gameplay_markers.len()
        );
    } else {
        // 回退模式：未找到 .meshlet_scene.json 时，扫描 meshes/*.meshlet_mesh 放入单个默认区块
        let meshes_dir = config.scene_dir.join("meshes");
        warn!(
            "未在 {} 找到 .meshlet_scene.json，回退至目录扫描: {}",
            config.scene_dir.display(),
            meshes_dir.display()
        );

        let default_mat = standard_materials.add(StandardMaterial {
            base_color: Color::srgb(0.72, 0.74, 0.78),
            perceptual_roughness: 0.45,
            metallic: 0.2,
            ..default()
        });

        let mut items = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&meshes_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("meshlet_mesh")
                    && let Some(fname) = path.file_name().and_then(|s| s.to_str())
                {
                    items.push(ChunkMeshletItem {
                        asset_path: format!("{prefix}meshes/{fname}"),
                        material: default_mat.clone(),
                        transform: Transform::default(),
                        cast_shadows: config.shadows_enabled,
                    });
                }
            }
        }

        let count = items.len();
        commands.spawn((
            Name::new("Chunk:fallback"),
            SceneChunk {
                chunk_name: "fallback".to_string(),
                aabb_min: Vec3::splat(-10.0),
                aabb_max: Vec3::splat(10.0),
                meshlets: items,
                mesh3ds: Vec::new(),
                meshlet_handles: HashMap::new(),
                spawned_entities: Vec::new(),
                loaded: false,
                pending_meshlet_cursor: 0,
                out_of_range_timer: 0.0,
            },
        ));
        spawn_default_light(&mut commands, config.shadows_enabled, &config);
        info!("回退模式已注册 {} 个 .meshlet_mesh 文件到 Chunk 'fallback'", count);
    }
}

/// 根据首级阴影覆盖半径与贴图尺寸，将 `normal_bias` 的物理世界偏移封顶在 ~2.5cm 左右，
/// 避免低分辨率阴影图（如 512px）因默认 1.8 texel 偏移过大而在薄窗帘/墙面产生黑斑与阴影脱落。
fn resolve_shadow_biases(config: &MeshletSceneConfig) -> (f32, f32, f32) {
    let max_dist = config.shadow_distance.max(6.0);
    let near_radius = if config.shadow_cascades > 1 {
        config.first_cascade_distance.clamp(3.0, max_dist * 0.8)
    } else {
        max_dist
    };
    let est_texel = (near_radius * 2.0) / (config.shadow_map_size.max(64) as f32);
    let auto_normal = (0.025 / est_texel).clamp(0.12, 1.8);
    let auto_depth = if est_texel > 0.08 { 0.05 } else { 0.025 };
    (
        config.shadow_depth_bias.unwrap_or(auto_depth),
        config.shadow_normal_bias.unwrap_or(auto_normal),
        est_texel,
    )
}

/// 运行期按「相机前向视锥加权距离」带迟滞地选择最值得点亮的 `max_lights` 个局部光，
/// 保证正前方远端走廊吊灯优先亮起，不被相机身后的灯光抢占名额，并以指数平滑淡入/淡出消除跳变。
fn update_active_local_lights(
    time: Res<Time>,
    config: Res<MeshletSceneConfig>,
    camera_query: Query<&GlobalTransform, With<Camera3d>>,
    mut local_lights: Query<(
        Entity,
        &GlobalTransform,
        &mut ManagedLocalLight,
        Option<&mut PointLight>,
        Option<&mut SpotLight>,
    )>,
) {
    if config.max_lights == 0 {
        return;
    }
    let Some(cam_tf) = camera_query.iter().next() else {
        return;
    };
    let cam_pos = cam_tf.translation();
    let cam_forward: Vec3 = cam_tf.forward().into();

    let mut ranked: Vec<(Entity, f32)> = local_lights
        .iter()
        .map(|(entity, tf, _, _, _)| {
            let delta = tf.translation() - cam_pos;
            let dist = delta.length();
            let dir_dot = if dist > 1e-3 {
                (delta / dist).dot(cam_forward)
            } else {
                1.0
            };
            // 视锥前向权重：正前方视野内保持真实距离；侧面适度加罚；相机背后额外增加 20~40m 惩罚
            let score = if dir_dot >= 0.25 {
                dist
            } else if dir_dot >= -0.2 {
                dist + (0.25 - dir_dot) * 24.0
            } else {
                dist + 20.0 + (-dir_dot) * 20.0
            };
            (entity, score)
        })
        .collect();
    if ranked.is_empty() {
        return;
    }
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));

    let active_limit = config.max_lights.min(ranked.len());
    // 迟滞阈值：已激活灯光只要得分未超过第 max_lights 名的 1.15 倍 + 2m 就暂保持点亮，防止边界抖动
    let cutoff_score_hysteresis = ranked[active_limit - 1].1 * 1.15 + 2.0;
    let fade_alpha = (time.delta_secs() * 6.0).clamp(0.05, 1.0);

    for (rank, (entity, score)) in ranked.into_iter().enumerate() {
        if let Ok((_, _, mut managed, point_opt, spot_opt)) = local_lights.get_mut(entity) {
            let should_enable =
                rank < active_limit || (managed.active && score <= cutoff_score_hysteresis);
            managed.active = should_enable;

            let target = if should_enable {
                managed.base_intensity
            } else {
                0.0
            };
            managed.current_intensity += (target - managed.current_intensity) * fade_alpha;
            if (target - managed.current_intensity).abs() < 1.0 {
                managed.current_intensity = target;
            }

            let val = managed.current_intensity;
            if let Some(mut pt) = point_opt
                && (pt.intensity - val).abs() > 0.01
            {
                pt.intensity = val;
            }
            if let Some(mut sp) = spot_opt
                && (sp.intensity - val).abs() > 0.01
            {
                sp.intensity = val;
            }
        }
    }
}

fn spawn_default_light(commands: &mut Commands, shadows_enabled: bool, config: &MeshletSceneConfig) {
    let (shadow_depth_bias, shadow_normal_bias, _) = resolve_shadow_biases(config);
    let num_cascades = config.shadow_cascades.clamp(1, 4);
    let max_dist = config.shadow_distance.max(6.0);
    let first_bound = config.first_cascade_distance.clamp(3.0, max_dist * 0.8);
    let mut entity_cmds = commands.spawn((
        Name::new("DefaultDirectionalLight"),
        DirectionalLight {
            illuminance: light_consts::lux::OVERCAST_DAY,
            shadow_maps_enabled: shadows_enabled,
            shadow_depth_bias,
            shadow_normal_bias,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(
            EulerRot::ZYX,
            0.0,
            -std::f32::consts::FRAC_PI_4,
            -std::f32::consts::FRAC_PI_4,
        )),
    ));
    if shadows_enabled {
        entity_cmds.insert(
            CascadeShadowConfigBuilder {
                num_cascades,
                first_cascade_far_bound: first_bound,
                maximum_distance: max_dist,
                ..default()
            }
            .build(),
        );
    }
}

/// 为刚加载完成的场景 PBR PNG 贴图按帧批量补全完整 Mipmap 链（让 Meshlet 的 `textureSampleGrad` 与各向异性过滤真正生效），
/// 并在生成完毕后将 `asset_usage` 切为 `RENDER_WORLD` 以自动释放 CPU 副本内存。
fn generate_scene_texture_mipmaps(
    mut pending: ResMut<PendingSceneMipmaps>,
    mut images: ResMut<Assets<Image>>,
) {
    if pending.pending.is_empty() {
        return;
    }

    const MAX_PER_FRAME: usize = 8;
    let ready: Vec<(AssetId<Image>, TextureMipRole)> = pending
        .pending
        .iter()
        .filter(|(id, _)| images.contains(**id))
        .take(MAX_PER_FRAME)
        .map(|(id, role)| (*id, *role))
        .collect();

    if ready.is_empty() {
        return;
    }

    for (id, role) in ready {
        pending.pending.remove(&id);
        let Some(mut image) = images.get_mut(id) else {
            continue;
        };
        if build_rgba8_mip_chain(&mut image, role) {
            pending.generated_count += 1;
        }
    }

    if pending.pending.is_empty() && pending.generated_count > 0 {
        info!(
            "场景 PBR 贴图运行时 Mipmap 链生成完成: 共 {} 张（sRGB/Normal 重归一化/ORM 线性滤波）",
            pending.generated_count
        );
    }
}

#[inline]
fn srgb_u8_to_linear(v: u8) -> f32 {
    let c = v as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn linear_to_srgb_u8(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// 为单层 2D `Rgba8Unorm`/`Rgba8UnormSrgb` 图像补全完整 Mipmap 金字塔（向下至 1×1）。
fn build_rgba8_mip_chain(image: &mut Image, role: TextureMipRole) -> bool {
    let desc = &image.texture_descriptor;
    if desc.dimension != TextureDimension::D2
        || desc.mip_level_count > 1
        || desc.size.depth_or_array_layers != 1
        || !matches!(
            desc.format,
            TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb
        )
    {
        return false;
    }

    let mut cur_w = desc.size.width;
    let mut cur_h = desc.size.height;
    if cur_w <= 1 && cur_h <= 1 {
        return false;
    }

    let expected_len = (cur_w as usize) * (cur_h as usize) * 4;
    let Some(ref mut data) = image.data else {
        return false;
    };
    if data.len() != expected_len {
        return false;
    }

    let mip_levels = (cur_w.max(cur_h)).ilog2() + 1;
    let mut total_bytes = expected_len;
    let (mut tw, mut th) = (cur_w, cur_h);
    for _ in 1..mip_levels {
        tw = (tw / 2).max(1);
        th = (th / 2).max(1);
        total_bytes += (tw as usize) * (th as usize) * 4;
    }
    data.reserve(total_bytes - expected_len);

    let mut prev_offset = 0usize;
    for _ in 1..mip_levels {
        let next_w = (cur_w / 2).max(1);
        let next_h = (cur_h / 2).max(1);
        let prev_len = (cur_w as usize) * (cur_h as usize) * 4;
        let next_offset = data.len();
        data.resize(next_offset + (next_w as usize) * (next_h as usize) * 4, 0);

        let (head, dst) = data.split_at_mut(next_offset);
        let src = &head[prev_offset..prev_offset + prev_len];

        for ny in 0..next_h {
            let sy0 = (ny * 2).min(cur_h - 1);
            let sy1 = (sy0 + 1).min(cur_h - 1);
            // 奇数高度时在最后一行吸收第 3 行防止边缘偏移
            let sy2 = if cur_h > 1 && (cur_h & 1) == 1 && ny == next_h - 1 {
                Some(cur_h - 1)
            } else {
                None
            };
            for nx in 0..next_w {
                let sx0 = (nx * 2).min(cur_w - 1);
                let sx1 = (sx0 + 1).min(cur_w - 1);
                let sx2 = if cur_w > 1 && (cur_w & 1) == 1 && nx == next_w - 1 {
                    Some(cur_w - 1)
                } else {
                    None
                };

                let mut xs = [sx0, sx1, sx0];
                let x_count = if let Some(x2) = sx2 {
                    xs[2] = x2;
                    3usize
                } else if cur_w > 1 {
                    2usize
                } else {
                    1usize
                };
                let mut ys = [sy0, sy1, sy0];
                let y_count = if let Some(y2) = sy2 {
                    ys[2] = y2;
                    3usize
                } else if cur_h > 1 {
                    2usize
                } else {
                    1usize
                };

                let out_idx = ((ny as usize) * (next_w as usize) + (nx as usize)) * 4;
                let inv_n = 1.0 / (x_count * y_count) as f32;

                match role {
                    TextureMipRole::SrgbColor => {
                        let (mut r, mut g, mut b, mut a) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                        for &sy in &ys[..y_count] {
                            let row = (sy as usize) * (cur_w as usize);
                            for &sx in &xs[..x_count] {
                                let i = (row + sx as usize) * 4;
                                r += srgb_u8_to_linear(src[i]);
                                g += srgb_u8_to_linear(src[i + 1]);
                                b += srgb_u8_to_linear(src[i + 2]);
                                a += src[i + 3] as f32;
                            }
                        }
                        dst[out_idx] = linear_to_srgb_u8(r * inv_n);
                        dst[out_idx + 1] = linear_to_srgb_u8(g * inv_n);
                        dst[out_idx + 2] = linear_to_srgb_u8(b * inv_n);
                        dst[out_idx + 3] = (a * inv_n).round().clamp(0.0, 255.0) as u8;
                    }
                    TextureMipRole::NormalMap => {
                        let (mut nx_sum, mut ny_sum, mut nz_sum) = (0.0f32, 0.0f32, 0.0f32);
                        for &sy in &ys[..y_count] {
                            let row = (sy as usize) * (cur_w as usize);
                            for &sx in &xs[..x_count] {
                                let i = (row + sx as usize) * 4;
                                nx_sum += src[i] as f32 * (2.0 / 255.0) - 1.0;
                                ny_sum += src[i + 1] as f32 * (2.0 / 255.0) - 1.0;
                                nz_sum += src[i + 2] as f32 * (2.0 / 255.0) - 1.0;
                            }
                        }
                        let len_sq = nx_sum * nx_sum + ny_sum * ny_sum + nz_sum * nz_sum;
                        let (ux, uy, uz) = if len_sq > 1e-8 {
                            let inv = len_sq.sqrt().recip();
                            (nx_sum * inv, ny_sum * inv, nz_sum * inv)
                        } else {
                            (0.0, 0.0, 1.0)
                        };
                        dst[out_idx] = ((ux * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
                        dst[out_idx + 1] = ((uy * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
                        dst[out_idx + 2] = ((uz * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
                        dst[out_idx + 3] = 255;
                    }
                    TextureMipRole::LinearData => {
                        let (mut r, mut g, mut b, mut a) = (0u32, 0u32, 0u32, 0u32);
                        for &sy in &ys[..y_count] {
                            let row = (sy as usize) * (cur_w as usize);
                            for &sx in &xs[..x_count] {
                                let i = (row + sx as usize) * 4;
                                r += src[i] as u32;
                                g += src[i + 1] as u32;
                                b += src[i + 2] as u32;
                                a += src[i + 3] as u32;
                            }
                        }
                        let cnt = (x_count * y_count) as u32;
                        let half = cnt / 2;
                        dst[out_idx] = ((r + half) / cnt) as u8;
                        dst[out_idx + 1] = ((g + half) / cnt) as u8;
                        dst[out_idx + 2] = ((b + half) / cnt) as u8;
                        dst[out_idx + 3] = ((a + half) / cnt) as u8;
                    }
                }
            }
        }

        prev_offset = next_offset;
        cur_w = next_w;
        cur_h = next_h;
    }

    image.texture_descriptor.mip_level_count = mip_levels;
    image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::render::render_resource::Extent3d;

    #[test]
    fn test_runtime_mip_chain_srgb_normal_and_orm() {
        // 1. 4x4 Normal map (2x2 alternating +X/-X tilted normals -> mip1/mip2 renormalize to (0, 0, 1))
        let mut normal_bytes = Vec::with_capacity(4 * 4 * 4);
        for y in 0..4 {
            for x in 0..4 {
                let nx = if (x + y) % 2 == 0 { 204u8 } else { 51u8 }; // ~ +0.6 / -0.6
                normal_bytes.extend_from_slice(&[nx, 128, 230, 255]);
            }
        }
        let mut normal_img = Image::new(
            Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            normal_bytes,
            TextureFormat::Rgba8Unorm,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        );
        assert!(build_rgba8_mip_chain(&mut normal_img, TextureMipRole::NormalMap));
        assert_eq!(normal_img.texture_descriptor.mip_level_count, 3);
        assert_eq!(normal_img.asset_usage, RenderAssetUsages::RENDER_WORLD);
        let data = normal_img.data.as_ref().unwrap();
        // Mip0: 16*4=64, Mip1: 4*4=16, Mip2: 1*4=4 => total 84 bytes
        assert_eq!(data.len(), 84);
        // Mip2 (last 4 bytes) should be renormalized to ~(128, 128, 255, 255)
        let last = &data[80..84];
        assert!((last[0] as i32 - 128).abs() <= 1);
        assert!((last[1] as i32 - 128).abs() <= 1);
        assert_eq!(last[2], 255);
        assert_eq!(last[3], 255);

        // 2. 2x2 sRGB checkerboard -> mip1 linear average ~188
        let srgb_bytes = vec![
            0, 0, 0, 255, 255, 255, 255, 255, //
            255, 255, 255, 255, 0, 0, 0, 255,
        ];
        let mut srgb_img = Image::new(
            Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            srgb_bytes,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        );
        assert!(build_rgba8_mip_chain(&mut srgb_img, TextureMipRole::SrgbColor));
        assert_eq!(srgb_img.texture_descriptor.mip_level_count, 2);
        let srgb_mip1 = &srgb_img.data.as_ref().unwrap()[16..20];
        assert!((srgb_mip1[0] as i32 - 188).abs() <= 2);

        // 3. 3x3 NPOT ORM linear data -> 3 mips (3x3, 1x1) => ilog2(3)+1 = 2 mips
        let orm_bytes = vec![100, 200, 50, 255].repeat(9);
        let mut orm_img = Image::new(
            Extent3d {
                width: 3,
                height: 3,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            orm_bytes,
            TextureFormat::Rgba8Unorm,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        );
        assert!(build_rgba8_mip_chain(&mut orm_img, TextureMipRole::LinearData));
        assert_eq!(orm_img.texture_descriptor.mip_level_count, 2);
        let orm_mip1 = &orm_img.data.as_ref().unwrap()[36..40];
        assert_eq!(orm_mip1, &[100, 200, 50, 255]);
    }
}
