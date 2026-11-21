//! 场景级离线烘焙模块：
//! 1. 递归遍历 glTF 场景树，继承并解析 Blender 插件写入节点 `extras` 中的 Bevy 角标：
//!    - `MESHLET_STATIC`: 按 (chunk, material_id) 合并静态高模并构建 `.meshlet_mesh`
//!    - `MESHLET_DYN`:    保留局部坐标与独立节点，去重构建 `.meshlet_mesh`
//!    - `MESH3D_LOD`:     使用 `meshopt` 生成多级 LOD 并保存为 `.mesh3d_lod`
//!    - `MESH3D_TRANS`:   半透明件（玻璃/酒瓶），保存为 `.mesh3d_lod` 并附带剔除距离
//!    - `GAME_SPAWN`:     仅记录 Gameplay 标记点坐标与分组，不导出渲染网格
//!    - `GAME_COLLIDER`:  跳过渲染网格导出
//! 2. 自动兜底规则：若材质为 `Mask` / `Blend`（非 `Opaque`），即使未打标也会自动分流至 `Mesh3d` 分支。
//! 3. 提取 PBR 材质（含 `alpha_mode` / `alpha_cutoff`）、贴图与 `KHR_lights_punctual` 灯光，
//!    输出统一的 `<stem>.meshlet_scene.json` 清单。

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use bevy_asset::saver::{AssetSaver, SavedAsset};
use bevy_asset::{AssetPath, RenderAssetUsages};
use bevy_math::{Mat3, Mat4, Vec3};
use bevy_mesh::{Indices, Mesh, PrimitiveTopology};
use bevy_pbr::experimental::meshlet::{MeshletMesh, MeshletMeshSaver};
use serde::{Deserialize, Serialize};

use crate::ao_baker::{bake_vertex_ao, AoBakeConfig, AoBvh, AoOccluderCollector};
use crate::lod_mesh::{
    build_mesh3d_lod_asset, parse_lod_distances, parse_lod_ratios, save_mesh3d_lod_asset,
};
use crate::texture_export::SemanticTextureExporter;
use crate::vlm_baker::{
    bake_and_save_vlm, estimate_gltf_image_avg_linear_rgb, VlmBakeConfig, VlmMaterialInfo,
    VlmSceneCollector,
};

/// 内存异步写入适配器，实现 futures_io::AsyncWrite（供 Bevy AssetSaver 使用）
pub struct MemoryWriter(pub Vec<u8>);

impl futures_io::AsyncWrite for MemoryWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        self.0.extend_from_slice(buf);
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

/// 导出给渲染器的完整场景清单（`.meshlet_scene.json`）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshletSceneManifest {
    pub version: u32,
    pub source_file: String,
    pub materials: Vec<SceneMaterial>,
    pub meshlet_instances: Vec<SceneMeshletInstance>,
    pub mesh3d_instances: Vec<SceneMesh3dInstance>,
    pub lights: Vec<SceneLight>,
    pub gameplay_markers: Vec<SceneGameplayMarker>,
    pub skipped_non_opaque: Vec<SkippedPrimitiveInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub world_hdri: Option<SceneWorldHdri>,
    /// 离线烘焙的静态表面虚拟光照贴图文件（相对输出目录，如 `<stem>.vlm.bin`）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vlm_file: Option<String>,
}

fn default_hdri_strength() -> f32 {
    1.0
}

fn default_hdri_projection() -> String {
    "equirectangular".to_string()
}

/// 从 Blender World 提取的 HDRI 环境贴图配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneWorldHdri {
    pub image: String,
    #[serde(default = "default_hdri_strength")]
    pub strength: f32,
    #[serde(default)]
    pub yaw_degrees: f32,
    #[serde(default = "default_hdri_projection")]
    pub projection: String,
}

fn default_wrap_pair() -> [String; 2] {
    ["Repeat".to_string(), "Repeat".to_string()]
}

fn default_ior() -> f32 {
    1.5
}

fn default_reflectance() -> f32 {
    0.5
}

fn is_default_ior(v: &f32) -> bool {
    (*v - 1.5).abs() < 1e-4
}

fn is_default_reflectance(v: &f32) -> bool {
    (*v - 0.5).abs() < 1e-4
}

fn is_false(v: &bool) -> bool {
    !*v
}

/// glTF `KHR_texture_transform` 导出的 2D UV 仿射变换参数
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SceneUvTransform {
    pub offset: [f32; 2],
    pub scale: [f32; 2],
    pub rotation: f32,
}

/// PBR 材质描述（Meshlet 与 Mesh3d 实例共用此材质表）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneMaterial {
    pub id: u32,
    pub name: String,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub alpha_mode: String,
    pub alpha_cutoff: f32,
    pub base_color_texture: Option<String>,
    pub normal_texture: Option<String>,
    pub metallic_roughness_texture: Option<String>,
    pub occlusion_texture: Option<String>,
    pub emissive_texture: Option<String>,
    pub double_sided: bool,
    /// 贴图 UV 寻址模式 [wrap_u, wrap_v]（"Repeat" / "MirroredRepeat" / "ClampToEdge"）
    #[serde(default = "default_wrap_pair")]
    pub wrap_mode: [String; 2],
    #[serde(default, skip_serializing_if = "is_false")]
    pub unlit: bool,
    #[serde(default = "default_ior", skip_serializing_if = "is_default_ior")]
    pub ior: f32,
    #[serde(default = "default_reflectance", skip_serializing_if = "is_default_reflectance")]
    pub reflectance: f32,
    #[serde(default, skip_serializing_if = "is_false")]
    pub flip_normal_map_y: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uv_transform: Option<SceneUvTransform>,
}

fn default_true() -> bool {
    true
}

/// Meshlet 实例描述
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneMeshletInstance {
    pub name: String,
    pub chunk: String,
    pub mesh: String,
    pub material_id: u32,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub dynamic: bool,
    #[serde(default = "default_true")]
    pub cast_shadows: bool,
}

/// 标准 `Mesh3d` 多级 LOD 实例描述（植物/玻璃/半透明/剔除件）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneMesh3dInstance {
    pub name: String,
    pub chunk: String,
    pub tag: String,
    pub mesh: String,
    pub material_id: u32,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    /// 各级 LOD 切换/剔除距离上限（米），例如 [18.0, 45.0, 120.0]
    pub lod_distances: Vec<f32>,
    #[serde(default = "default_true")]
    pub cast_shadows: bool,
}

/// 从 glTF `KHR_lights_punctual` 提取的灯光描述
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Gameplay 标记点（如出生点、刷怪点）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneGameplayMarker {
    pub name: String,
    pub tag: String,
    pub spawn_group: String,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

/// 记录因拓扑异常等原因跳过的图元信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkippedPrimitiveInfo {
    pub node_name: String,
    pub mesh_name: String,
    pub alpha_mode: String,
    pub triangle_count: usize,
}

/// 从 glTF 节点 `extras` JSON 解析出的 Blender 角标（支持父节点向下继承）
#[derive(Debug, Clone)]
struct InheritedNodeTags {
    bevy_tag: String,
    chunk: String,
    lod_ratios: Option<String>,
    lod_dists: Option<String>,
    spawn_group: String,
    cast_shadows: bool,
}

impl Default for InheritedNodeTags {
    fn default() -> Self {
        Self {
            bevy_tag: "MESHLET_STATIC".to_string(),
            chunk: "main".to_string(),
            lod_ratios: None,
            lod_dists: None,
            spawn_group: "default".to_string(),
            cast_shadows: true,
        }
    }
}

/// 材质去重哈希键
#[derive(Hash, Eq, PartialEq, Clone)]
struct MaterialKey {
    base_color_bits: [u32; 4],
    metallic_bits: u32,
    roughness_bits: u32,
    emissive_bits: [u32; 3],
    alpha_mode: u8,
    alpha_cutoff_bits: u32,
    base_color_tex: Option<usize>,
    normal_tex: Option<usize>,
    mr_tex: Option<usize>,
    occlusion_tex: Option<usize>,
    emissive_tex: Option<usize>,
    double_sided: bool,
    wrap_mode_codes: [u8; 2],
    unlit: bool,
    ior_bits: u32,
    reflectance_bits: u32,
    uv_transform_bits: Option<[u32; 5]>,
}

/// 静态合并几何体缓冲（按 `(chunk, material_id)` 聚合）
struct StaticBatch {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
}

impl Default for StaticBatch {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            aabb_min: [f32::INFINITY; 3],
            aabb_max: [f32::NEG_INFINITY; 3],
        }
    }
}

pub struct SceneExportOptions<'a> {
    pub input: &'a Path,
    pub out_dir: &'a Path,
    pub input_stem: &'a str,
    pub quantization: u8,
    pub merge_by_material: bool,
    pub dynamic_prefixes: &'a [String],
    pub export_textures: bool,
    pub max_texture_size: u32,
    /// 是否将分离的 Occlusion 与 MetallicRoughness 自动打包为单张标准 ORM 贴图
    pub pack_orm: bool,
    /// 空间自动分块网格尺寸（米，0 表示不启用空间切分）
    pub grid_size: f32,
    /// 单个静态 Meshlet 批次最大三角形数（0 表示不限）
    pub max_batch_tris: usize,
    /// 同一图元出现 >= N 次时自动走局部实例化复用（0 表示禁用）
    pub auto_instance_min: usize,
    /// 静态逐顶点 AO 烘焙参数
    pub ao: AoBakeConfig,
    /// 静态表面虚拟光照贴图（VLM）离线烘焙参数
    pub vlm: VlmBakeConfig,
    /// 允许显式 MESH3D_LOD 保留 glTF AlphaMode::Blend（默认自动转为 Mask）
    pub lod_keep_blend: bool,
    /// 仅离线烘焙静态表面虚拟光照贴图 (.vlm.bin)，跳过贴图导出、逐顶点 AO 与 Meshlet/LOD 网格构建
    pub vlm_only: bool,
}

struct SceneContext<'a> {
    buffers: &'a [gltf::buffer::Data],
    tex_exporter: SemanticTextureExporter<'a>,
    image_avg_colors: &'a [[f32; 3]],
    options: &'a SceneExportOptions<'a>,
    prim_usage_counts: HashMap<(usize, usize), usize>,
    material_map: HashMap<MaterialKey, u32>,
    materials: Vec<SceneMaterial>,
    vlm_materials: Vec<VlmMaterialInfo>,
    /// 当前活跃静态批次序号：Key = (chunk_id, material_id, cast_shadows)
    active_batch_seq: HashMap<(String, u32, bool), usize>,
    /// 静态 Meshlet 合并池：Key = (chunk_id, material_id, cast_shadows, batch_seq)
    static_batches: HashMap<(String, u32, bool, usize), StaticBatch>,
    /// 除 static_batches 外参与全局静态遮蔽的几何（如自动实例化静态件与不透明 Mesh3d_LOD）
    ao_occluders: AoOccluderCollector,
    /// 参与静态 VLM 烘焙的全场景静态三角形（带反照率/自发光）
    vlm_collector: VlmSceneCollector,
    /// 独立/动态 Meshlet 资产去重缓存：Key = (mesh_index, prim_index)
    dynamic_assets_cache: HashMap<(usize, usize), (String, [f32; 3], [f32; 3])>,
    dynamic_instances: Vec<SceneMeshletInstance>,
    /// 标准 Mesh3d LOD 资产去重缓存：Key = (mesh_index, prim_index, ratios_signature)
    mesh3d_assets_cache: HashMap<(usize, usize, String), (String, [f32; 3], [f32; 3])>,
    mesh3d_instances: Vec<SceneMesh3dInstance>,
    lights: Vec<SceneLight>,
    gameplay_markers: Vec<SceneGameplayMarker>,
    skipped_non_opaque: Vec<SkippedPrimitiveInfo>,
    total_nodes_visited: usize,
    total_primitives_merged: usize,
    auto_instanced_count: usize,
    collider_skipped_count: usize,
    degenerate_tris_removed: usize,
    degenerate_prims_skipped: usize,
}

pub fn run_scene_export(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
    options: &SceneExportOptions<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(options.out_dir)?;
    if !options.vlm_only {
        let meshes_dir = options.out_dir.join("meshes");
        std::fs::create_dir_all(&meshes_dir)?;
    }

    // 1. 准备语义感知贴图导出器（按材质引用延迟导出并去重：sRGB 线性缩放 / Normal 重归一化 / ORM 自动打包）
    let export_textures_active = options.export_textures && !options.vlm_only && !images.is_empty();
    let tex_exporter = SemanticTextureExporter::new(
        images,
        options.out_dir,
        options.input_stem,
        options.max_texture_size,
        options.pack_orm,
        export_textures_active,
    );
    if export_textures_active {
        println!(
            "\n[2/5] 已启用语义感知贴图导出 (源贴图={}, max_texture_size={}, pack_orm={})...",
            images.len(),
            if options.max_texture_size == 0 {
                "unlimited".to_string()
            } else {
                options.max_texture_size.to_string()
            },
            options.pack_orm
        );
    }

    // 1.5 预扫描图元引用次数，供 --auto-instance-min 自动实例化复用检测
    let mut prim_usage_counts: HashMap<(usize, usize), usize> = HashMap::new();
    if options.auto_instance_min > 0 {
        let default_tags = InheritedNodeTags::default();
        if let Some(default_scene) = document.default_scene() {
            for node in default_scene.nodes() {
                count_node_prims(&node, Mat4::IDENTITY, &default_tags, &mut prim_usage_counts);
            }
        } else if let Some(first_scene) = document.scenes().next() {
            for node in first_scene.nodes() {
                count_node_prims(&node, Mat4::IDENTITY, &default_tags, &mut prim_usage_counts);
            }
        }
    }

    let image_avg_colors: Vec<[f32; 3]> = if options.vlm.enabled {
        images
            .iter()
            .map(estimate_gltf_image_avg_linear_rgb)
            .collect()
    } else {
        Vec::new()
    };

    let mut ctx = SceneContext {
        buffers,
        tex_exporter,
        image_avg_colors: &image_avg_colors,
        options,
        prim_usage_counts,
        material_map: HashMap::new(),
        materials: Vec::new(),
        vlm_materials: Vec::new(),
        active_batch_seq: HashMap::new(),
        static_batches: HashMap::new(),
        ao_occluders: AoOccluderCollector::default(),
        vlm_collector: VlmSceneCollector::default(),
        dynamic_assets_cache: HashMap::new(),
        dynamic_instances: Vec::new(),
        mesh3d_assets_cache: HashMap::new(),
        mesh3d_instances: Vec::new(),
        lights: Vec::new(),
        gameplay_markers: Vec::new(),
        skipped_non_opaque: Vec::new(),
        total_nodes_visited: 0,
        total_primitives_merged: 0,
        auto_instanced_count: 0,
        collider_skipped_count: 0,
        degenerate_tris_removed: 0,
        degenerate_prims_skipped: 0,
    };

    // 2. 遍历场景树：继承角标 Flag、按角标与材质分流几何体、提取灯光与 Gameplay 点
    println!(
        "\n[3/5] 正在遍历场景节点树、清洗退化几何并分流 (grid_size={}, max_batch_tris={}, auto_instance_min={})...",
        options.grid_size, options.max_batch_tris, options.auto_instance_min
    );
    let default_tags = InheritedNodeTags::default();
    if let Some(default_scene) = document.default_scene() {
        for node in default_scene.nodes() {
            visit_node(&node, Mat4::IDENTITY, false, &default_tags, &mut ctx)?;
        }
    } else if let Some(first_scene) = document.scenes().next() {
        for node in first_scene.nodes() {
            visit_node(&node, Mat4::IDENTITY, false, &default_tags, &mut ctx)?;
        }
    }

    println!(
        "      节点: {} | 材质: {} | 合并静态图元: {} | 独立/实例化Meshlet: {} (其中自动实例化 {}) | Mesh3d_LOD: {} | 灯光: {} | Gameplay: {} | 跳过碰撞: {} | 清洗退化三角面: {} (跳过空图元: {})",
        ctx.total_nodes_visited,
        ctx.materials.len(),
        ctx.total_primitives_merged,
        ctx.dynamic_instances.len(),
        ctx.auto_instanced_count,
        ctx.mesh3d_instances.len(),
        ctx.lights.len(),
        ctx.gameplay_markers.len(),
        ctx.collider_skipped_count,
        ctx.degenerate_tris_removed,
        ctx.degenerate_prims_skipped
    );
    if export_textures_active {
        println!(
            "      语义贴图导出: 共写出 {} 张 PNG (降采样 {} 张, 自动打包 ORM {} 组)",
            ctx.tex_exporter.exported_count,
            ctx.tex_exporter.resized_count,
            ctx.tex_exporter.orm_packed_count
        );
    }

    // 3. 构建全局静态遮蔽 BVH（并同步收集静态批次到 VLM 烘焙收集器）
    let mut final_meshlet_instances = Vec::new();
    let mut sorted_keys: Vec<(String, u32, bool, usize)> = ctx.static_batches.keys().cloned().collect();
    sorted_keys.sort();

    if options.vlm.enabled {
        for key in &sorted_keys {
            if let Some(batch) = ctx.static_batches.get(key) {
                let mat_info = ctx
                    .vlm_materials
                    .get(key.1 as usize)
                    .copied()
                    .unwrap_or(VlmMaterialInfo {
                        diffuse_albedo: [0.5, 0.5, 0.5],
                        emissive: [0.0, 0.0, 0.0],
                        double_sided: false,
                    });
                ctx.vlm_collector.push_indexed_mesh(
                    &batch.positions,
                    &batch.normals,
                    &batch.indices,
                    None,
                    mat_info,
                );
            }
        }
    }

    if !options.vlm_only {
        let global_ao_bvh: Option<AoBvh> = if options.ao.is_active() && !sorted_keys.is_empty() {
            let t_bvh = Instant::now();
            let mut collector = std::mem::take(&mut ctx.ao_occluders);
            for key in &sorted_keys {
                if let Some(batch) = ctx.static_batches.get(key) {
                    collector.push_indexed_mesh(&batch.positions, &batch.indices, None);
                }
            }
            let tri_total = collector.triangle_count();
            let bvh = collector.build();
            println!(
                "\n[4/5] 已构建全局静态 AO BVH（{} 个遮蔽三角面，耗时 {:.2}s），正在并行烘焙逐顶点 AO 并构建 MeshletMesh...",
                tri_total,
                t_bvh.elapsed().as_secs_f64()
            );
            Some(bvh)
        } else {
            println!("\n[4/5] 正在构建并导出 MeshletMesh 静态分块资产...");
            None
        };

        for (idx, key) in sorted_keys.iter().enumerate() {
            let mut batch = ctx.static_batches.remove(key).unwrap();
            let (chunk, mat_id, cast_shadows, batch_seq) = key;
            let tri_count = batch.indices.len() / 3;
            if tri_count == 0 {
                continue;
            }
            println!(
                "  -> Meshlet 批次 [{}/{}] chunk='{}' mat_id={} shadow={} b={} (顶点: {}, 三角形: {})",
                idx + 1,
                sorted_keys.len(),
                chunk,
                mat_id,
                cast_shadows,
                batch_seq,
                batch.positions.len(),
                tri_count
            );

            // 先在世界空间（与 global_ao_bvh 同坐标系）烘焙静态逐顶点 AO
            let vertex_ao = global_ao_bvh.as_ref().map(|bvh| {
                let t_ao = Instant::now();
                let ao = bake_vertex_ao(bvh, &batch.positions, &batch.normals, options.ao);
                let min_ao = ao.iter().copied().fold(1.0f32, f32::min);
                let avg_ao = ao.iter().sum::<f32>() / ao.len().max(1) as f32;
                println!(
                    "     [AO] 耗时 {:.2}s (avg={:.3}, min={:.3})",
                    t_ao.elapsed().as_secs_f64(),
                    avg_ao,
                    min_ao
                );
                ao
            });

            // 将世界合并坐标平移至批次自身 AABB 中心附近，减小定点量化误差并保证紧凑局部包围盒
            let batch_center =
                (Vec3::from_array(batch.aabb_min) + Vec3::from_array(batch.aabb_max)) * 0.5;
            for p in &mut batch.positions {
                p[0] -= batch_center.x;
                p[1] -= batch_center.y;
                p[2] -= batch_center.z;
            }
            let local_aabb_min = [
                batch.aabb_min[0] - batch_center.x,
                batch.aabb_min[1] - batch_center.y,
                batch.aabb_min[2] - batch_center.z,
            ];
            let local_aabb_max = [
                batch.aabb_max[0] - batch_center.x,
                batch.aabb_max[1] - batch_center.y,
                batch.aabb_max[2] - batch_center.z,
            ];

            let mut bevy_mesh =
                Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
            bevy_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, batch.positions);
            bevy_mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, batch.normals);
            bevy_mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, batch.uvs);
            bevy_mesh.insert_indices(Indices::U32(batch.indices));

            let t0 = Instant::now();
            let meshlet_mesh = MeshletMesh::from_mesh_with_ao(
                &bevy_mesh,
                options.quantization,
                vertex_ao.as_deref(),
            )
            .map_err(|e| {
                format!(
                    "静态批次 chunk='{}' mat_id={} b={} 构建 Meshlet 失败: {:?}",
                    chunk, mat_id, batch_seq, e
                )
            })?;

            let safe_chunk = sanitize_filename(chunk);
            let shadow_tag = if *cast_shadows { "" } else { "_noshadow" };
            let rel_mesh_path = format!(
                "meshes/{}_chunk_{}_mat_{}{}_b{}.meshlet_mesh",
                options.input_stem, safe_chunk, mat_id, shadow_tag, batch_seq
            );
            let abs_mesh_path = options.out_dir.join(&rel_mesh_path);
            save_meshlet_mesh(&meshlet_mesh, &abs_mesh_path)?;
            println!(
                "     完成，耗时 {:.2}s -> {}",
                t0.elapsed().as_secs_f64(),
                rel_mesh_path
            );

            final_meshlet_instances.push(SceneMeshletInstance {
                name: format!("static_{}_mat_{}{}_b{}", safe_chunk, mat_id, shadow_tag, batch_seq),
                chunk: chunk.clone(),
                mesh: rel_mesh_path,
                material_id: *mat_id,
                translation: batch_center.to_array(),
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
                aabb_min: local_aabb_min,
                aabb_max: local_aabb_max,
                dynamic: false,
                cast_shadows: *cast_shadows,
            });
        }

        final_meshlet_instances.extend(ctx.dynamic_instances);
    } else {
        ctx.static_batches.clear();
        println!("\n[4/5] 仅烘焙 VLM 模式 (--vlm-only)，已跳过 MeshletMesh/LOD 与顶点 AO 构建。");
    }

    // 4. 提取场景 World HDRI 配置（优先读取 glTF scene.extras，其次读旁路 .bevy_hdri.json）并随场景打包
    let world_hdri = extract_and_bundle_world_hdri(&document, options);

    // 4.5 离线烘焙静态表面虚拟光照贴图（VLM SpatialHash Tile -> <stem>.vlm.bin）
    let vlm_file = if options.vlm.enabled && ctx.vlm_collector.triangle_count() > 0 {
        println!("\n[4.5/5] 正在离线路径追踪烘焙静态表面 Virtual Lightmap (.vlm.bin)...");
        let rel_vlm = format!("{}.vlm.bin", options.input_stem);
        let abs_vlm = options.out_dir.join(&rel_vlm);
        match bake_and_save_vlm(
            ctx.vlm_collector,
            &ctx.lights,
            world_hdri.as_ref(),
            options.out_dir,
            &abs_vlm,
            options.vlm,
        ) {
            Ok(_report) => Some(rel_vlm),
            Err(err) => {
                eprintln!("  [警告] VLM 烘焙失败（已跳过，不影响 Meshlet 场景导出）: {}", err);
                None
            }
        }
    } else {
        None
    };

    if options.vlm_only {
        let manifest_path = options
            .out_dir
            .join(format!("{}.meshlet_scene.json", options.input_stem));
        if manifest_path.is_file()
            && let Ok(existing_bytes) = std::fs::read(&manifest_path)
            && let Ok(mut existing_manifest) =
                serde_json::from_slice::<MeshletSceneManifest>(&existing_bytes)
        {
            existing_manifest.vlm_file = vlm_file.clone();
            if existing_manifest.world_hdri.is_none() {
                existing_manifest.world_hdri = world_hdri.clone();
            }
            let json_str = serde_json::to_string_pretty(&existing_manifest)?;
            std::fs::write(&manifest_path, json_str)?;
            println!(
                "\n[5/5] 已无损更新现有场景清单的 vlm_file 字段: {}",
                manifest_path.display()
            );
        } else {
            println!(
                "\n[5/5] 输出目录未找到现有清单 {}，已仅生成 .vlm.bin。",
                manifest_path.display()
            );
        }

        println!("\n=======================================================");
        println!("VLM 独立烘焙完成！(--vlm-only)");
        if let Some(hdri) = &world_hdri {
            println!(
                "  World HDRI: {} (strength={:.2}, yaw={:.1}°)",
                hdri.image, hdri.strength, hdri.yaw_degrees
            );
        }
        if let Some(vlm) = &vlm_file {
            println!("  Virtual Lightmap: {}", options.out_dir.join(vlm).display());
        } else {
            println!("  Virtual Lightmap: 未生成（场景无可用静态几何或烘焙失败）");
        }
        println!("=======================================================");

        return Ok(());
    }

    // 5. 写入场景清单 JSON
    println!("\n[5/5] 正在生成场景清单 (.meshlet_scene.json)...");
    let manifest = MeshletSceneManifest {
        version: 2,
        source_file: options
            .input
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown.glb")
            .to_string(),
        materials: ctx.materials,
        meshlet_instances: final_meshlet_instances,
        mesh3d_instances: ctx.mesh3d_instances,
        lights: ctx.lights,
        gameplay_markers: ctx.gameplay_markers,
        skipped_non_opaque: ctx.skipped_non_opaque,
        world_hdri,
        vlm_file,
    };

    let manifest_path = options
        .out_dir
        .join(format!("{}.meshlet_scene.json", options.input_stem));
    let json_str = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, json_str)?;

    println!("\n=======================================================");
    println!("场景导出完成！");
    println!("  清单文件: {}", manifest_path.display());
    println!("  Meshlet 实例总数: {}", manifest.meshlet_instances.len());
    println!("  Mesh3d LOD 实例总数: {}", manifest.mesh3d_instances.len());
    println!("  去重材质总数: {}", manifest.materials.len());
    println!("  提取灯光总数: {}", manifest.lights.len());
    println!("  Gameplay 标记点数: {}", manifest.gameplay_markers.len());
    if let Some(hdri) = &manifest.world_hdri {
        println!(
            "  World HDRI: {} (strength={:.2}, yaw={:.1}°)",
            hdri.image, hdri.strength, hdri.yaw_degrees
        );
    }
    if let Some(vlm) = &manifest.vlm_file {
        println!("  Virtual Lightmap: {}", vlm);
    }
    println!("=======================================================");

    Ok(())
}

/// 从节点 `extras` JSON 解析 Bevy 自定义角标，并继承父级未显式覆盖的字段
fn resolve_node_tags(node: &gltf::Node, parent: &InheritedNodeTags) -> InheritedNodeTags {
    let mut out = parent.clone();
    let Some(extras) = node.extras() else {
        return out;
    };
    let Ok(val) = serde_json::from_str::<serde_json::Value>(extras.get()) else {
        return out;
    };

    if let Some(tag) = val.get("bevy_tag").and_then(|v| v.as_str()) {
        out.bevy_tag = tag.to_string();
    }
    if let Some(chunk) = val.get("bevy_chunk").and_then(|v| v.as_str())
        && !chunk.trim().is_empty()
    {
        out.chunk = chunk.trim().to_string();
    }
    if let Some(ratios) = val.get("bevy_lod_ratios").and_then(|v| v.as_str()) {
        out.lod_ratios = Some(ratios.to_string());
    }
    if let Some(dists) = val.get("bevy_lod_dists").and_then(|v| v.as_str()) {
        out.lod_dists = Some(dists.to_string());
    }
    if let Some(group) = val.get("bevy_spawn_group").and_then(|v| v.as_str()) {
        out.spawn_group = group.to_string();
    }
    if let Some(cast_shadow) = val.get("bevy_cast_shadow").and_then(|v| v.as_bool()) {
        out.cast_shadows = cast_shadow;
    }
    out
}

/// 预扫描统计每个 `(mesh_index, prim_index)` 在场景中（非镜像且可渲染节点）的引用次数，用于自动实例化检测
fn count_node_prims(
    node: &gltf::Node,
    parent_world: Mat4,
    parent_tags: &InheritedNodeTags,
    counts: &mut HashMap<(usize, usize), usize>,
) {
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent_world * local;
    let tags = resolve_node_tags(node, parent_tags);

    if tags.bevy_tag != "GAME_COLLIDER"
        && tags.bevy_tag != "GAME_SPAWN"
        && world.determinant() > 1e-12
        && let Some(mesh) = node.mesh()
    {
        for (prim_idx, prim) in mesh.primitives().enumerate() {
            if prim.mode() == gltf::mesh::Mode::Triangles {
                *counts.entry((mesh.index(), prim_idx)).or_insert(0) += 1;
            }
        }
    }

    for child in node.children() {
        count_node_prims(&child, world, &tags, counts);
    }
}

/// 清洗单个图元中的退化几何：
/// 1. 过滤越界、重复索引、含 NaN/Inf 顶点与零面积退化三角形（防止 Meshlet BVH/球合并异常）
/// 2. 紧缩重映射顶点数组，丢弃未被引用的游离顶点
fn sanitize_primitive_geometry(
    positions: Vec<[f32; 3]>,
    normals: Option<Vec<[f32; 3]>>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
    world: &Mat4,
) -> (Vec<[f32; 3]>, Option<Vec<[f32; 3]>>, Vec<[f32; 2]>, Vec<u32>, usize) {
    let linear = Mat3::from_mat4(*world);
    let max_scale = linear
        .x_axis
        .length()
        .max(linear.y_axis.length())
        .max(linear.z_axis.length())
        .max(1e-6);
    let area_sq_eps = 1e-12 / (max_scale * max_scale * max_scale * max_scale).max(1e-12);

    let mut clean_indices_raw = Vec::with_capacity(indices.len());
    let mut removed = 0usize;
    let vlen = positions.len();

    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if i0 >= vlen || i1 >= vlen || i2 >= vlen || i0 == i1 || i1 == i2 || i0 == i2 {
            removed += 1;
            continue;
        }
        let p0 = Vec3::from_array(positions[i0]);
        let p1 = Vec3::from_array(positions[i1]);
        let p2 = Vec3::from_array(positions[i2]);
        if !p0.is_finite() || !p1.is_finite() || !p2.is_finite() {
            removed += 1;
            continue;
        }
        if (p1 - p0).cross(p2 - p0).length_squared() <= area_sq_eps {
            removed += 1;
            continue;
        }
        clean_indices_raw.push(tri[0]);
        clean_indices_raw.push(tri[1]);
        clean_indices_raw.push(tri[2]);
    }

    if clean_indices_raw.is_empty() {
        return (Vec::new(), None, Vec::new(), Vec::new(), removed);
    }

    // 重映射仅保留被有效三角形引用的顶点
    let mut remap = vec![u32::MAX; vlen];
    let mut new_positions = Vec::new();
    let mut new_normals = normals.as_ref().map(|_| Vec::new());
    let mut new_uvs = Vec::new();
    let mut new_indices = Vec::with_capacity(clean_indices_raw.len());

    for &old_idx in &clean_indices_raw {
        let oi = old_idx as usize;
        let ni = if remap[oi] != u32::MAX {
            remap[oi]
        } else {
            let assigned = new_positions.len() as u32;
            remap[oi] = assigned;
            new_positions.push(positions[oi]);
            if let (Some(src_n), Some(dst_n)) = (&normals, &mut new_normals) {
                let n = Vec3::from_array(src_n[oi]);
                let clean_n = if n.is_finite() && n.length_squared() > 1e-12 {
                    n.normalize().to_array()
                } else {
                    [0.0, 1.0, 0.0]
                };
                dst_n.push(clean_n);
            }
            let uv = uvs[oi];
            new_uvs.push(if uv[0].is_finite() && uv[1].is_finite() {
                uv
            } else {
                [0.0, 0.0]
            });
            assigned
        };
        new_indices.push(ni);
    }

    (new_positions, new_normals, new_uvs, new_indices, removed)
}

fn sanitize_filename(raw: &str) -> String {
    raw.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn visit_node(
    node: &gltf::Node,
    parent_world: Mat4,
    parent_is_dynamic_prefix: bool,
    parent_tags: &InheritedNodeTags,
    ctx: &mut SceneContext<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    ctx.total_nodes_visited += 1;
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent_world * local;

    let node_name = node.name().unwrap_or("unnamed_node");
    let tags = resolve_node_tags(node, parent_tags);

    let is_dynamic_by_prefix = parent_is_dynamic_prefix
        || ctx
            .options
            .dynamic_prefixes
            .iter()
            .any(|p| node_name.starts_with(p));

    // 若节点被标记为 Gameplay 标记点，记录其世界位姿后直接跳过网格构建
    if tags.bevy_tag == "GAME_SPAWN" {
        let (scale, rotation, translation) = world.to_scale_rotation_translation();
        ctx.gameplay_markers.push(SceneGameplayMarker {
            name: node_name.to_string(),
            tag: tags.bevy_tag.clone(),
            spawn_group: tags.spawn_group.clone(),
            translation: translation.to_array(),
            rotation: rotation.to_array(),
            scale: scale.to_array(),
        });
    }

    // 1. 提取 KHR_lights_punctual 灯光
    if let Some(light) = node.light() {
        let (_scale, rotation, translation) = world.to_scale_rotation_translation();
        let (kind, inner, outer) = match light.kind() {
            gltf::khr_lights_punctual::Kind::Directional => ("Directional".to_string(), None, None),
            gltf::khr_lights_punctual::Kind::Point => ("Point".to_string(), None, None),
            gltf::khr_lights_punctual::Kind::Spot {
                inner_cone_angle,
                outer_cone_angle,
            } => (
                "Spot".to_string(),
                Some(inner_cone_angle),
                Some(outer_cone_angle),
            ),
        };
        ctx.lights.push(SceneLight {
            name: light.name().unwrap_or(node_name).to_string(),
            kind,
            color: light.color(),
            intensity: light.intensity(),
            range: light.range(),
            inner_cone_angle: inner,
            outer_cone_angle: outer,
            translation: translation.to_array(),
            rotation: rotation.normalize().to_array(),
        });
    }

    // 2. 处理节点上的网格图元（仅在非标记点、非碰撞体时处理）
    if tags.bevy_tag == "GAME_COLLIDER" {
        if node.mesh().is_some() {
            ctx.collider_skipped_count += 1;
        }
    } else if tags.bevy_tag != "GAME_SPAWN"
        && let Some(mesh) = node.mesh()
    {
        let mesh_name = mesh.name().unwrap_or("mesh");
        for (prim_idx, primitive) in mesh.primitives().enumerate() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }

            let gltf_mat = primitive.material();
            let alpha_mode = gltf_mat.alpha_mode();
            let reader = primitive.reader(|b| Some(&ctx.buffers[b.index()]));

            let Some(positions_iter) = reader.read_positions() else {
                continue;
            };
            let positions: Vec<[f32; 3]> = positions_iter.collect();
            let vert_count = positions.len();
            let indices: Vec<u32> = reader
                .read_indices()
                .map(|i| i.into_u32().collect())
                .unwrap_or_else(|| (0..vert_count as u32).collect());
            if indices.len() < 3 {
                continue;
            }

            let force_lod_alpha_mask = tags.bevy_tag == "MESH3D_LOD"
                && alpha_mode == gltf::material::AlphaMode::Blend
                && !ctx.options.lod_keep_blend;
            if force_lod_alpha_mask {
                println!(
                    "  [优化] MESH3D_LOD 节点 '{}' 使用的 AlphaBlend 材质 '{}' 已自动转为 AlphaMode::Mask（避免 Transparent Pass 过绘）",
                    node_name,
                    gltf_mat.name().unwrap_or("unnamed")
                );
            }
            let mat_id = register_or_get_material(&gltf_mat, force_lod_alpha_mask, ctx);
            let raw_normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|i| i.collect());
            let raw_uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|i| i.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; vert_count]);

            // 清洗退化零面积三角形、非法索引及 NaN，防止 Meshlet BVH 构建崩溃
            let (positions, normals, uvs, indices, removed_deg) =
                sanitize_primitive_geometry(positions, raw_normals, raw_uvs, indices, &world);
            ctx.degenerate_tris_removed += removed_deg;
            if indices.len() < 3 {
                ctx.degenerate_prims_skipped += 1;
                continue;
            }

            // 计算该图元世界包围盒中心，用于空间网格自动分块（grid_size > 0）
            let (local_min, local_max) = compute_local_aabb(&positions);
            let local_center = (Vec3::from_array(local_min) + Vec3::from_array(local_max)) * 0.5;
            let world_center = world.transform_point3(local_center);
            let effective_chunk = if ctx.options.grid_size > 0.0 && tags.chunk == "main" {
                let gx = (world_center.x / ctx.options.grid_size).floor() as i32;
                let gy = (world_center.y / ctx.options.grid_size).floor() as i32;
                let gz = (world_center.z / ctx.options.grid_size).floor() as i32;
                format!("{}_g{}_{}_{}", tags.chunk, gx, gy, gz)
            } else {
                tags.chunk.clone()
            };

            let prim_usage = ctx
                .prim_usage_counts
                .get(&(mesh.index(), prim_idx))
                .copied()
                .unwrap_or(1);
            let world_det = world.determinant();
            let auto_instanced = ctx.options.auto_instance_min > 0
                && prim_usage >= ctx.options.auto_instance_min
                && world_det > 1e-12;

            // 分流决策：
            // - 若显式标为 MESH3D_LOD / MESH3D_TRANS，或材质本身为 Mask/Blend（非 Opaque 兜底），走 Mesh3d LOD 分支
            // - 若显式标为 MESHLET_DYN、匹配 --dynamic-prefix、被 --auto-instance-min 命中或禁用合并，走独立/实例化 Meshlet 分支
            // - 其余 Opaque 物体走按 (spatial_chunk, material_id, batch_seq) 合并与面数封顶的静态 Meshlet 分支
            let route_to_mesh3d = tags.bevy_tag == "MESH3D_LOD"
                || tags.bevy_tag == "MESH3D_TRANS"
                || alpha_mode != gltf::material::AlphaMode::Opaque;

            if route_to_mesh3d {
                let effective_tag = if tags.bevy_tag == "MESH3D_TRANS"
                    || (tags.bevy_tag != "MESH3D_LOD"
                        && alpha_mode == gltf::material::AlphaMode::Blend)
                {
                    "MESH3D_TRANS"
                } else {
                    "MESH3D_LOD"
                };

                let default_ratios: &[f32] = if effective_tag == "MESH3D_TRANS" {
                    &[1.0]
                } else {
                    &[1.0, 0.4, 0.15]
                };
                let default_dists: &[f32] = if effective_tag == "MESH3D_TRANS" {
                    &[18.0, 30.0]
                } else {
                    &[18.0, 45.0, 120.0]
                };

                let ratios = parse_lod_ratios(tags.lod_ratios.as_deref(), default_ratios);
                let lod_distances = parse_lod_distances(tags.lod_dists.as_deref(), default_dists);
                let ratios_sig = ratios
                    .iter()
                    .map(|r| format!("{:.3}", r))
                    .collect::<Vec<_>>()
                    .join("_");

                if (ctx.options.ao.is_active() || ctx.options.vlm.enabled)
                    && alpha_mode == gltf::material::AlphaMode::Opaque
                    && !is_dynamic_by_prefix
                    && tags.bevy_tag != "MESHLET_DYN"
                {
                    if ctx.options.ao.is_active() {
                        ctx.ao_occluders
                            .push_indexed_mesh(&positions, &indices, Some(&world));
                    }
                    if ctx.options.vlm.enabled {
                        let local_n = ensure_local_normals(&positions, normals.as_deref(), &indices);
                        let mat_info = ctx
                            .vlm_materials
                            .get(mat_id as usize)
                            .copied()
                            .unwrap_or(VlmMaterialInfo {
                                diffuse_albedo: [0.5, 0.5, 0.5],
                                emissive: [0.0, 0.0, 0.0],
                                double_sided: false,
                            });
                        ctx.vlm_collector.push_indexed_mesh(
                            &positions,
                            &local_n,
                            &indices,
                            Some(&world),
                            mat_info,
                        );
                    }
                }

                if ctx.options.vlm_only {
                    continue;
                }

                let cache_key = (mesh.index(), prim_idx, ratios_sig.clone());
                let (rel_mesh_path, aabb_min, aabb_max) =
                    if let Some(cached) = ctx.mesh3d_assets_cache.get(&cache_key) {
                        cached.clone()
                    } else {
                        let (aabb_min, aabb_max) = compute_local_aabb(&positions);
                        let local_normals =
                            ensure_local_normals(&positions, normals.as_deref(), &indices);

                        let lod_asset = build_mesh3d_lod_asset(
                            positions,
                            local_normals,
                            uvs,
                            &indices,
                            &ratios,
                        )
                        .map_err(|e| {
                            format!(
                                "Mesh3d LOD 构建失败 ({} prim #{}): {}",
                                mesh_name, prim_idx, e
                            )
                        })?;

                        let rel_path = format!(
                            "meshes/{}_m3d_{}_{}_{}.mesh3d_lod",
                            ctx.options.input_stem,
                            mesh.index(),
                            prim_idx,
                            ratios_sig
                        );
                        let abs_path = ctx.options.out_dir.join(&rel_path);
                        save_mesh3d_lod_asset(&lod_asset, &abs_path)?;

                        let entry = (rel_path, aabb_min, aabb_max);
                        ctx.mesh3d_assets_cache.insert(cache_key, entry.clone());
                        entry
                    };

                let (scale, rotation, translation) = world.to_scale_rotation_translation();
                ctx.mesh3d_instances.push(SceneMesh3dInstance {
                    name: node_name.to_string(),
                    chunk: effective_chunk,
                    tag: effective_tag.to_string(),
                    mesh: rel_mesh_path,
                    material_id: mat_id,
                    translation: translation.to_array(),
                    rotation: rotation.to_array(),
                    scale: scale.to_array(),
                    aabb_min,
                    aabb_max,
                    lod_distances,
                    cast_shadows: tags.cast_shadows,
                });
            } else if ctx.options.merge_by_material
                && !is_dynamic_by_prefix
                && tags.bevy_tag != "MESHLET_DYN"
                && !auto_instanced
            {
                // 合并到按 (effective_chunk, material_id, cast_shadows, batch_seq) 分组并控制面数上限的静态 Meshlet 批次中
                let prim_tris = indices.len() / 3;
                let seq_entry = ctx
                    .active_batch_seq
                    .entry((effective_chunk.clone(), mat_id, tags.cast_shadows))
                    .or_insert(0);
                if ctx.options.max_batch_tris > 0 {
                    let cur_key = (effective_chunk.clone(), mat_id, tags.cast_shadows, *seq_entry);
                    if let Some(existing) = ctx.static_batches.get(&cur_key) {
                        let existing_tris = existing.indices.len() / 3;
                        if existing_tris > 0 && existing_tris + prim_tris > ctx.options.max_batch_tris
                        {
                            *seq_entry += 1;
                        }
                    }
                }
                let batch_key = (effective_chunk, mat_id, tags.cast_shadows, *seq_entry);
                append_to_static_batch(
                    ctx.static_batches.entry(batch_key).or_default(),
                    world,
                    &positions,
                    normals.as_deref(),
                    &uvs,
                    &indices,
                );
                ctx.total_primitives_merged += 1;
            } else {
                // 独立/动态/自动实例化 Meshlet 节点：按 (mesh.index(), prim_idx) 去重构建局部空间 MeshletMesh
                if auto_instanced {
                    ctx.auto_instanced_count += 1;
                }
                let is_dynamic_instance = is_dynamic_by_prefix || tags.bevy_tag == "MESHLET_DYN";
                if !is_dynamic_instance {
                    if ctx.options.ao.is_active() {
                        ctx.ao_occluders
                            .push_indexed_mesh(&positions, &indices, Some(&world));
                    }
                    if ctx.options.vlm.enabled {
                        let local_n = ensure_local_normals(&positions, normals.as_deref(), &indices);
                        let mat_info = ctx
                            .vlm_materials
                            .get(mat_id as usize)
                            .copied()
                            .unwrap_or(VlmMaterialInfo {
                                diffuse_albedo: [0.5, 0.5, 0.5],
                                emissive: [0.0, 0.0, 0.0],
                                double_sided: false,
                            });
                        ctx.vlm_collector.push_indexed_mesh(
                            &positions,
                            &local_n,
                            &indices,
                            Some(&world),
                            mat_info,
                        );
                    }
                }

                if ctx.options.vlm_only {
                    continue;
                }

                let key = (mesh.index(), prim_idx);
                let (rel_mesh_path, aabb_min, aabb_max) =
                    if let Some(cached) = ctx.dynamic_assets_cache.get(&key) {
                        cached.clone()
                    } else {
                        let (aabb_min, aabb_max) = compute_local_aabb(&positions);
                        let local_normals =
                            ensure_local_normals(&positions, normals.as_deref(), &indices);

                        // 独立/动态/实例化资产被多个位置或动态节点复用，因此仅烘焙局部自身遮蔽 AO
                        let local_ao = if ctx.options.ao.is_active() {
                            let local_bvh = AoBvh::from_indexed_mesh(&positions, &indices);
                            Some(bake_vertex_ao(
                                &local_bvh,
                                &positions,
                                &local_normals,
                                ctx.options.ao,
                            ))
                        } else {
                            None
                        };

                        let mut bevy_mesh =
                            Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
                        bevy_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
                        bevy_mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, local_normals);
                        bevy_mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
                        bevy_mesh.insert_indices(Indices::U32(indices));

                        let meshlet_mesh = MeshletMesh::from_mesh_with_ao(
                            &bevy_mesh,
                            ctx.options.quantization,
                            local_ao.as_deref(),
                        )
                        .map_err(|e| {
                            format!(
                                "独立图元 {}_{} 构建 Meshlet 失败: {:?}",
                                mesh_name, prim_idx, e
                            )
                        })?;

                        let rel_path = format!(
                            "meshes/{}_mesh_{}_prim_{}.meshlet_mesh",
                            ctx.options.input_stem,
                            mesh.index(),
                            prim_idx
                        );
                        let abs_path = ctx.options.out_dir.join(&rel_path);
                        save_meshlet_mesh(&meshlet_mesh, &abs_path)?;

                        let entry = (rel_path, aabb_min, aabb_max);
                        ctx.dynamic_assets_cache.insert(key, entry.clone());
                        entry
                    };

                let (scale, rotation, translation) = world.to_scale_rotation_translation();
                ctx.dynamic_instances.push(SceneMeshletInstance {
                    name: node_name.to_string(),
                    chunk: effective_chunk,
                    mesh: rel_mesh_path,
                    material_id: mat_id,
                    translation: translation.to_array(),
                    rotation: rotation.to_array(),
                    scale: scale.to_array(),
                    aabb_min,
                    aabb_max,
                    dynamic: is_dynamic_instance,
                    cast_shadows: tags.cast_shadows,
                });
            }
        }
    }

    for child in node.children() {
        visit_node(&child, world, is_dynamic_by_prefix, &tags, ctx)?;
    }

    Ok(())
}

fn compute_local_aabb(positions: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let (mut aabb_min, mut aabb_max) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for p in positions {
        for i in 0..3 {
            aabb_min[i] = aabb_min[i].min(p[i]);
            aabb_max[i] = aabb_max[i].max(p[i]);
        }
    }
    (aabb_min, aabb_max)
}

fn ensure_local_normals(
    positions: &[[f32; 3]],
    normals: Option<&[[f32; 3]]>,
    indices: &[u32],
) -> Vec<[f32; 3]> {
    if let Some(n) = normals {
        return n.to_vec();
    }
    let mut temp = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    temp.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions.to_vec());
    temp.insert_indices(Indices::U32(indices.to_vec()));
    temp.compute_smooth_normals();
    temp.attribute(Mesh::ATTRIBUTE_NORMAL)
        .and_then(|a| a.as_float3())
        .map(|s| s.to_vec())
        .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()])
}

fn append_to_static_batch(
    batch: &mut StaticBatch,
    world: Mat4,
    positions: &[[f32; 3]],
    normals: Option<&[[f32; 3]]>,
    uvs: &[[f32; 2]],
    indices: &[u32],
) {
    let base_vertex = batch.positions.len() as u32;
    let det = world.determinant();
    let flip_winding = det < 0.0;

    // 计算世界空间位置并更新包围盒
    for &p in positions {
        let wp = world.transform_point3(Vec3::from_array(p)).to_array();
        for i in 0..3 {
            batch.aabb_min[i] = batch.aabb_min[i].min(wp[i]);
            batch.aabb_max[i] = batch.aabb_max[i].max(wp[i]);
        }
        batch.positions.push(wp);
    }

    // 计算世界空间法线；若缺失则先在局部生成平滑法线
    let local_normals = ensure_local_normals(positions, normals, indices);
    let linear_mat3 = Mat3::from_mat4(world);
    let normal_mat = if linear_mat3.determinant().abs() > 1e-12 {
        linear_mat3.inverse().transpose()
    } else {
        Mat3::IDENTITY
    };
    for n in local_normals {
        let wn = (normal_mat * Vec3::from_array(n)).normalize_or(Vec3::Y);
        batch.normals.push(wn.to_array());
    }

    batch.uvs.extend_from_slice(uvs);

    // 若存在负缩放镜像，翻转三角形绕序
    for tri in indices.chunks_exact(3) {
        if flip_winding {
            batch.indices.push(base_vertex + tri[0]);
            batch.indices.push(base_vertex + tri[2]);
            batch.indices.push(base_vertex + tri[1]);
        } else {
            batch.indices.push(base_vertex + tri[0]);
            batch.indices.push(base_vertex + tri[1]);
            batch.indices.push(base_vertex + tri[2]);
        }
    }
}

fn gltf_wrap_to_code_and_str(mode: gltf::texture::WrappingMode) -> (u8, String) {
    match mode {
        gltf::texture::WrappingMode::ClampToEdge => (0, "ClampToEdge".to_string()),
        gltf::texture::WrappingMode::MirroredRepeat => (1, "MirroredRepeat".to_string()),
        gltf::texture::WrappingMode::Repeat => (2, "Repeat".to_string()),
    }
}

fn extract_uv_transform(transform: gltf::texture::TextureTransform<'_>) -> Option<SceneUvTransform> {
    let offset = transform.offset();
    let scale = transform.scale();
    let rotation = transform.rotation();
    let is_identity = offset[0].abs() < 1e-6
        && offset[1].abs() < 1e-6
        && (scale[0] - 1.0).abs() < 1e-6
        && (scale[1] - 1.0).abs() < 1e-6
        && rotation.abs() < 1e-6;
    if is_identity {
        None
    } else {
        Some(SceneUvTransform {
            offset,
            scale,
            rotation,
        })
    }
}

fn register_or_get_material(
    gltf_mat: &gltf::Material,
    force_alpha_mask: bool,
    ctx: &mut SceneContext<'_>,
) -> u32 {
    let pbr = gltf_mat.pbr_metallic_roughness();
    let base_color = pbr.base_color_factor();
    let metallic = pbr.metallic_factor();
    let roughness = pbr.roughness_factor();
    let emissive_strength = gltf_mat.emissive_strength().unwrap_or(1.0);
    let raw_emissive = gltf_mat.emissive_factor();
    let emissive = [
        raw_emissive[0] * emissive_strength,
        raw_emissive[1] * emissive_strength,
        raw_emissive[2] * emissive_strength,
    ];
    let unlit = gltf_mat.unlit();
    let ior = gltf_mat.ior().unwrap_or(1.5);
    let reflectance = gltf_mat
        .specular()
        .map(|s| 0.5 * s.specular_factor())
        .unwrap_or(0.5);

    let (alpha_mode_code, alpha_mode_str) = match gltf_mat.alpha_mode() {
        gltf::material::AlphaMode::Opaque => (0u8, "Opaque".to_string()),
        gltf::material::AlphaMode::Mask => (1u8, "Mask".to_string()),
        gltf::material::AlphaMode::Blend => {
            if force_alpha_mask {
                (1u8, "Mask".to_string())
            } else {
                (2u8, "Blend".to_string())
            }
        }
    };
    let alpha_cutoff = gltf_mat.alpha_cutoff().unwrap_or(0.5);

    let base_color_info = pbr.base_color_texture();
    let normal_info = gltf_mat.normal_texture();
    let mr_info = pbr.metallic_roughness_texture();
    let occlusion_info = gltf_mat.occlusion_texture();
    let emissive_info = gltf_mat.emissive_texture();

    let uv_transform = base_color_info
        .as_ref()
        .and_then(|t| t.texture_transform().and_then(extract_uv_transform))
        .or_else(|| {
            mr_info
                .as_ref()
                .and_then(|t| t.texture_transform().and_then(extract_uv_transform))
        })
        .or_else(|| {
            emissive_info
                .as_ref()
                .and_then(|t| t.texture_transform().and_then(extract_uv_transform))
        });

    let primary_sampler = base_color_info
        .as_ref()
        .map(|t| t.texture().sampler())
        .or_else(|| normal_info.as_ref().map(|t| t.texture().sampler()))
        .or_else(|| mr_info.as_ref().map(|t| t.texture().sampler()))
        .or_else(|| occlusion_info.as_ref().map(|t| t.texture().sampler()))
        .or_else(|| emissive_info.as_ref().map(|t| t.texture().sampler()));

    let (wrap_u_code, wrap_u_str) = primary_sampler
        .as_ref()
        .map(|s| gltf_wrap_to_code_and_str(s.wrap_s()))
        .unwrap_or_else(|| (2, "Repeat".to_string()));
    let (wrap_v_code, wrap_v_str) = primary_sampler
        .as_ref()
        .map(|s| gltf_wrap_to_code_and_str(s.wrap_t()))
        .unwrap_or_else(|| (2, "Repeat".to_string()));

    let base_color_tex = base_color_info.map(|t| t.texture().source().index());
    let normal_tex = normal_info.map(|t| t.texture().source().index());
    let mr_tex = mr_info.map(|t| t.texture().source().index());
    let occlusion_tex = occlusion_info.map(|t| t.texture().source().index());
    let emissive_tex = emissive_info.map(|t| t.texture().source().index());
    let double_sided = gltf_mat.double_sided();

    let uv_transform_bits = uv_transform.map(|t| {
        [
            t.offset[0].to_bits(),
            t.offset[1].to_bits(),
            t.scale[0].to_bits(),
            t.scale[1].to_bits(),
            t.rotation.to_bits(),
        ]
    });

    let key = MaterialKey {
        base_color_bits: base_color.map(f32::to_bits),
        metallic_bits: metallic.to_bits(),
        roughness_bits: roughness.to_bits(),
        emissive_bits: emissive.map(f32::to_bits),
        alpha_mode: alpha_mode_code,
        alpha_cutoff_bits: alpha_cutoff.to_bits(),
        base_color_tex,
        normal_tex,
        mr_tex,
        occlusion_tex,
        emissive_tex,
        double_sided,
        wrap_mode_codes: [wrap_u_code, wrap_v_code],
        unlit,
        ior_bits: ior.to_bits(),
        reflectance_bits: reflectance.to_bits(),
        uv_transform_bits,
    };

    if let Some(&id) = ctx.material_map.get(&key) {
        return id;
    }

    let id = ctx.materials.len() as u32;
    let base_color_texture = ctx.tex_exporter.export_srgb(base_color_tex);
    let normal_texture = ctx.tex_exporter.export_normal(normal_tex);
    let (occlusion_texture, metallic_roughness_texture) =
        ctx.tex_exporter.export_orm_pair(occlusion_tex, mr_tex);
    let emissive_texture = ctx.tex_exporter.export_srgb(emissive_tex);

    let scene_mat = SceneMaterial {
        id,
        name: gltf_mat
            .name()
            .map(ToString::to_string)
            .unwrap_or_else(|| format!("material_{}", id)),
        base_color,
        metallic,
        roughness,
        emissive,
        alpha_mode: alpha_mode_str,
        alpha_cutoff,
        base_color_texture,
        normal_texture,
        metallic_roughness_texture,
        occlusion_texture,
        emissive_texture,
        double_sided,
        wrap_mode: [wrap_u_str, wrap_v_str],
        unlit,
        ior,
        reflectance,
        flip_normal_map_y: false,
        uv_transform,
    };
    let tex_avg = base_color_tex.and_then(|i| ctx.image_avg_colors.get(i).copied());
    ctx.vlm_materials
        .push(VlmMaterialInfo::from_scene_material(&scene_mat, tex_avg));
    ctx.materials.push(scene_mat);
    ctx.material_map.insert(key, id);
    id
}

fn save_meshlet_mesh(
    meshlet_mesh: &MeshletMesh,
    out_file_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let saver = MeshletMeshSaver;
    let mut memory_writer = MemoryWriter(Vec::new());
    let saved_asset = SavedAsset::from_asset(meshlet_mesh);
    bevy_tasks::block_on(saver.save(
        &mut memory_writer,
        saved_asset,
        &(),
        AssetPath::default(),
    ))?;
    std::fs::write(out_file_path, &memory_writer.0)?;
    Ok(())
}

fn parse_world_hdri_from_scene_extras(scene: &gltf::Scene) -> Option<SceneWorldHdri> {
    let extras = scene.extras().as_ref()?;
    let val: serde_json::Value = serde_json::from_str(extras.get()).ok()?;
    let hdri_val = val.get("bevy_world_hdri")?;
    if let Some(raw_json) = hdri_val.as_str() {
        serde_json::from_str::<SceneWorldHdri>(raw_json).ok()
    } else {
        serde_json::from_value::<SceneWorldHdri>(hdri_val.clone()).ok()
    }
}

fn extract_and_bundle_world_hdri(
    document: &gltf::Document,
    options: &SceneExportOptions,
) -> Option<SceneWorldHdri> {
    let mut cfg = document
        .default_scene()
        .and_then(|s| parse_world_hdri_from_scene_extras(&s))
        .or_else(|| {
            document
                .scenes()
                .find_map(|s| parse_world_hdri_from_scene_extras(&s))
        });

    if cfg.is_none() {
        let sidecar = options
            .input
            .with_file_name(format!("{}.bevy_hdri.json", options.input_stem));
        if sidecar.is_file()
            && let Ok(bytes) = std::fs::read(&sidecar)
        {
            cfg = serde_json::from_slice::<SceneWorldHdri>(&bytes).ok();
        }
    }

    let mut cfg = cfg?;
    let src_path = PathBuf::from(&cfg.image);
    let resolved_src = if src_path.is_file() {
        Some(src_path)
    } else if let Some(parent) = options.input.parent() {
        let rel = parent.join(&cfg.image);
        rel.is_file().then_some(rel)
    } else {
        None
    };

    if let Some(src) = resolved_src
        && let Some(file_name) = src.file_name().and_then(|n| n.to_str())
    {
        let env_dir = options.out_dir.join("environment");
        if std::fs::create_dir_all(&env_dir).is_ok() {
            let safe_name = sanitize_filename(file_name);
            let dst = env_dir.join(&safe_name);
            if std::fs::copy(&src, &dst).is_ok() {
                cfg.image = format!("environment/{safe_name}");
                println!("  -> 已打包 World HDRI: {}", cfg.image);
                return Some(cfg);
            }
        }
    }

    println!("  -> 记录 World HDRI 路径: {}", cfg.image);
    Some(cfg)
}
