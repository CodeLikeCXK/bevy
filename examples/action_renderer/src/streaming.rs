//! Chunk 区块级资源流送与 `Mesh3d` 自动 LOD 挂载调度模块
//!
//! 职责：
//! 1. 按 Blender 中设置的 `bevy_chunk`（如 `exterior_main`、`interior_wine`）组织场景单元 `SceneChunk`；
//! 2. 根据相机到区块包围盒（AABB）的距离，异步加载或滞后卸载整个区块的 `.meshlet_mesh` 与 `.mesh3d_lod` 资产；
//! 3. 当 `.mesh3d_lod` 异步加载就绪后，自动为每个 LOD 级别生成 `Mesh3d` 实体并挂载 `VisibilityRange` 组件实现距离切换与远距剔除。

use std::collections::HashMap;

use bevy::{
    camera::visibility::VisibilityRange,
    light::NotShadowCaster,
    pbr::experimental::meshlet::{MeshletMesh, MeshletMesh3d},
    prelude::*,
};

use crate::lod_loader::Mesh3dLodAsset;

/// 流送调度配置
#[derive(Resource, Clone)]
pub struct StreamingConfig {
    /// 是否启用按距离动态加载与显存回收（开启 --no-streaming 时设为 false，全部常驻）
    pub enabled: bool,
    /// 进入该距离（米）时开始异步加载区块
    pub load_distance: f32,
    /// 离开超过该距离（米）并持续超过滞后时间后卸载区块并释放句柄
    pub unload_distance: f32,
    /// 卸载滞后防抖时间（秒），防止相机在边界来回移动导致频繁加载/卸载
    pub unload_delay_secs: f32,
    /// 区块加载时每帧最多生成的 Meshlet 实体数量（0 表示不限制；分帧生成可避免 GPU PersistentBuffer 暴涨触发 TDR）
    pub max_spawns_per_frame: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            load_distance: 80.0,
            unload_distance: 110.0,
            unload_delay_secs: 5.0,
            max_spawns_per_frame: 2,
        }
    }
}

/// 区块内的 Meshlet 渲染项
pub struct ChunkMeshletItem {
    pub asset_path: String,
    pub material: Handle<StandardMaterial>,
    pub transform: Transform,
    pub cast_shadows: bool,
}

/// 挂载在已展开的 Mesh3d LOD 实体上的三角面数量元数据（供 HUD 统计，不访问已提取到 RenderWorld 的 Mesh 缓冲）
#[derive(Component, Debug, Clone, Copy)]
pub struct Mesh3dLodTris(pub usize);

/// 区块内的标准 Mesh3d 多级 LOD 渲染项
pub struct ChunkMesh3dItem {
    pub name: String,
    pub asset_path: String,
    pub material: Handle<StandardMaterial>,
    pub transform: Transform,
    pub lod_distances: Vec<f32>,
    /// 半透明件（`MESH3D_TRANS`）禁用 Dither Crossfade，使用硬切换/硬剔除
    pub is_transparent: bool,
    /// 是否向实时光阴影贴图投射阴影（植物/半透明默认关闭以提升集显帧率）
    pub cast_shadows: bool,
    pub lod_asset: Option<Handle<Mesh3dLodAsset>>,
    pub lods_spawned: bool,
}

/// 挂载在每个区块实体上的流送状态组件
#[derive(Component)]
pub struct SceneChunk {
    pub chunk_name: String,
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
    pub meshlets: Vec<ChunkMeshletItem>,
    pub mesh3ds: Vec<ChunkMesh3dItem>,
    pub meshlet_handles: HashMap<String, Handle<MeshletMesh>>,
    pub spawned_entities: Vec<Entity>,
    pub loaded: bool,
    /// 分帧生成 Meshlet 实体的当前游标
    pub pending_meshlet_cursor: usize,
    pub out_of_range_timer: f32,
}

pub struct StreamingPlugin;

impl Plugin for StreamingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<StreamingConfig>().add_systems(
            Update,
            (update_chunk_streaming, spawn_ready_mesh3d_lods).chain(),
        );
    }
}

fn distance_to_aabb(point: Vec3, aabb_min: Vec3, aabb_max: Vec3) -> f32 {
    let clamped = point.clamp(aabb_min, aabb_max);
    point.distance(clamped)
}

/// 阶段 1：按相机与区块 AABB 的距离，驱动 Chunk 的加载与滞后卸载
fn update_chunk_streaming(
    mut commands: Commands,
    time: Res<Time>,
    config: Res<StreamingConfig>,
    asset_server: Res<AssetServer>,
    camera_query: Query<&GlobalTransform, With<Camera3d>>,
    mut chunks: Query<&mut SceneChunk>,
) {
    let Some(cam_tf) = camera_query.iter().next() else {
        return;
    };
    let cam_pos = cam_tf.translation();
    let dt = time.delta_secs();
    let mut spawns_remaining = if config.max_spawns_per_frame == 0 {
        usize::MAX
    } else {
        config.max_spawns_per_frame
    };

    for mut chunk in &mut chunks {
        let dist = if config.enabled {
            distance_to_aabb(cam_pos, chunk.aabb_min, chunk.aabb_max)
        } else {
            0.0
        };

        if !chunk.loaded && (dist <= config.load_distance || !config.enabled) {
            chunk.out_of_range_timer = 0.0;
            chunk.loaded = true;
            chunk.pending_meshlet_cursor = 0;

            // 去重发起当前区块的 .mesh3d_lod 异步加载（待资产就绪后由下一个系统展开 LOD 实体）
            let mut mesh3d_cache: HashMap<String, Handle<Mesh3dLodAsset>> = HashMap::new();
            for item in &mut chunk.mesh3ds {
                let handle = mesh3d_cache
                    .entry(item.asset_path.clone())
                    .or_insert_with(|| asset_server.load(&item.asset_path))
                    .clone();
                item.lod_asset = Some(handle);
                item.lods_spawned = false;
            }

            info!(
                "激活区块 '{}': {} 个 Meshlet 实例 (每帧上限={}), {} 个 Mesh3d LOD 实例",
                chunk.chunk_name,
                chunk.meshlets.len(),
                if config.max_spawns_per_frame == 0 {
                    "unlimited".to_string()
                } else {
                    config.max_spawns_per_frame.to_string()
                },
                chunk.mesh3ds.len()
            );
        }

        // 若当前区块已激活但距离超限且达到滞后时间，则卸载释放；否则按每帧配额分批生成 Meshlet 实体
        if chunk.loaded && config.enabled && dist >= config.unload_distance {
            chunk.out_of_range_timer += dt;
            if chunk.out_of_range_timer >= config.unload_delay_secs {
                // 销毁该区块下全部已生成实体并清空句柄，释放 GPU PersistentBuffer 与 Mesh 内存
                for entity in chunk.spawned_entities.drain(..) {
                    commands.entity(entity).despawn();
                }
                chunk.meshlet_handles.clear();
                for item in &mut chunk.mesh3ds {
                    item.lod_asset = None;
                    item.lods_spawned = false;
                }
                chunk.loaded = false;
                chunk.pending_meshlet_cursor = 0;
                chunk.out_of_range_timer = 0.0;

                info!("已卸载远距区块 '{}' 并释放资源句柄", chunk.chunk_name);
                continue;
            }
        } else {
            chunk.out_of_range_timer = 0.0;
        }

        // 分帧生成已激活区块内尚未生成的 Meshlet 实体，平滑 GPU PersistentBuffer 扩容与上传峰值
        if chunk.loaded && chunk.pending_meshlet_cursor < chunk.meshlets.len() && spawns_remaining > 0
        {
            let start = chunk.pending_meshlet_cursor;
            let batch_count = (chunk.meshlets.len() - start).min(spawns_remaining);
            let end = start + batch_count;

            let chunk_ref = &mut *chunk;
            for idx in start..end {
                let item = &chunk_ref.meshlets[idx];
                let handle = chunk_ref
                    .meshlet_handles
                    .entry(item.asset_path.clone())
                    .or_insert_with(|| asset_server.load(&item.asset_path))
                    .clone();

                let mut entity_cmds = commands.spawn((
                    MeshletMesh3d(handle),
                    MeshMaterial3d(item.material.clone()),
                    item.transform,
                ));
                if !item.cast_shadows {
                    entity_cmds.insert(NotShadowCaster);
                }
                chunk_ref.spawned_entities.push(entity_cmds.id());
            }

            chunk_ref.pending_meshlet_cursor = end;
            spawns_remaining -= batch_count;

            if end == chunk_ref.meshlets.len() {
                info!(
                    "区块 '{}' 全部 {} 个 Meshlet 实体已分帧生成完毕",
                    chunk_ref.chunk_name,
                    chunk_ref.meshlets.len()
                );
            }
        }
    }
}

/// 阶段 2：当 `.mesh3d_lod` 异步加载完成后，自动生成各级 LOD 实体并挂载 `VisibilityRange`
fn spawn_ready_mesh3d_lods(
    mut commands: Commands,
    lod_assets: Res<Assets<Mesh3dLodAsset>>,
    mut chunks: Query<&mut SceneChunk>,
) {
    for mut chunk in &mut chunks {
        if !chunk.loaded {
            continue;
        }

        let mut newly_spawned = Vec::new();
        for item in &mut chunk.mesh3ds {
            if item.lods_spawned {
                continue;
            }
            let Some(ref handle) = item.lod_asset else {
                continue;
            };
            let Some(asset) = lod_assets.get(handle) else {
                continue;
            };

            let lod_count = asset.lod_meshes.len();
            for (lod_idx, mesh_handle) in asset.lod_meshes.iter().enumerate() {
                let vis_range = build_visibility_range(
                    lod_idx,
                    lod_count,
                    &item.lod_distances,
                    item.is_transparent,
                );

                let tri_count = asset.lod_tri_counts.get(lod_idx).copied().unwrap_or(0);
                let mut entity_cmds = commands.spawn((
                    Name::new(format!("{}_LOD{}", item.name, lod_idx)),
                    Mesh3d(mesh_handle.clone()),
                    Mesh3dLodTris(tri_count),
                    MeshMaterial3d(item.material.clone()),
                    item.transform,
                    vis_range,
                ));
                if !item.cast_shadows {
                    entity_cmds.insert(NotShadowCaster);
                }
                newly_spawned.push(entity_cmds.id());
            }
            item.lods_spawned = true;
        }

        chunk.spawned_entities.extend(newly_spawned);
    }
}

/// 根据给定的距离数组为第 `lod_idx` 级网格构造 Bevy `VisibilityRange`
fn build_visibility_range(
    lod_idx: usize,
    lod_count: usize,
    lod_distances: &[f32],
    is_transparent: bool,
) -> VisibilityRange {
    // 辅助函数：读取第 idx 级距离上限；若配置数量少于 LOD 级数则按 2.5 倍递推
    let dist_at = |idx: usize| -> f32 {
        if let Some(&d) = lod_distances.get(idx) {
            return d.max(0.5);
        }
        let last = lod_distances.last().copied().unwrap_or(20.0);
        let extra = (idx + 1 - lod_distances.len()) as i32;
        last * 2.5f32.powi(extra)
    };

    // 半透明件或单级 LOD 使用无 Dither 的硬切换/剔除，避免透明物体出现 Dither 噪点
    if is_transparent || lod_count <= 1 {
        let start = if lod_idx == 0 { 0.0 } else { dist_at(lod_idx - 1) };
        let end = dist_at(lod_idx).max(start);
        return VisibilityRange {
            start_margin: start..start,
            end_margin: end..end,
            use_aabb: true,
        };
    }

    // 多级不透明/AlphaMask 物体使用平滑 Crossfade 过渡带（要求相邻 LOD 边缘区间严格对齐）
    let start_margin = if lod_idx == 0 {
        0.0..0.0
    } else {
        let prev_d = dist_at(lod_idx - 1);
        let half_w = (prev_d * 0.08).clamp(0.5, 2.0);
        (prev_d - half_w).max(0.0)..(prev_d + half_w)
    };

    let cur_d = dist_at(lod_idx);
    let half_w_end = (cur_d * 0.08).clamp(0.5, 2.0);
    let end_start = (cur_d - half_w_end).max(start_margin.end);
    let end_end = (cur_d + half_w_end).max(end_start);

    VisibilityRange {
        start_margin,
        end_margin: end_start..end_end,
        use_aabb: true,
    }
}
