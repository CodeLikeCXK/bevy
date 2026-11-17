//! 右上角实时性能与场景诊断 HUD 面板
//!
//! 显示内容：
//! 1. 实时与平均帧率（FPS / ms）
//! 2. 当前实时阴影状态（开关、贴图尺寸、级联距离、植物投影开关）
//! 3. 当前画面与场景灯光数量（平行光 / 激活与总局部光 / IrradianceVolume 状态）
//! 4. 当前已流送区块、Meshlet 实例与 Cluster 簇/多边形数量、Mesh3d LOD 可见多边形数量
//!
//! 快捷键：按 `F3` 切换显示/隐藏 HUD 面板。

use std::collections::HashMap;

use bevy::{
    camera::visibility::VisibilityRange,
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    ecs::system::SystemParam,
    light::{DirectionalLightShadowMap, IrradianceVolume},
    pbr::{
        experimental::meshlet::{MeshletMesh, MeshletMesh3d},
        AoDebugView,
    },
    prelude::*,
};

use crate::{
    irradiance::IrradianceConfig,
    scene_loader::MeshletSceneConfig,
    skybox::{SkyboxConfig, SkyboxStatus},
    streaming::{Mesh3dLodTris, SceneChunk},
};

#[derive(Component)]
struct HudRoot;

#[derive(Component)]
struct HudText;

#[derive(Resource)]
struct HudRefreshTimer(Timer);

/// 在 MeshletMesh 被 `MeshletMeshManager::queue_upload_if_needed` 从 `Assets<MeshletMesh>` 移走前，
/// 提前缓存每份资产的 `(meshlet_count, total_triangle_count)`，供 HUD 准确统计。
#[derive(Resource, Default)]
struct MeshletAssetStatsCache(HashMap<AssetId<MeshletMesh>, (usize, usize)>);

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MeshletAssetStatsCache>()
            .insert_resource(HudRefreshTimer(Timer::from_seconds(
                0.2,
                TimerMode::Repeating,
            )))
            .add_systems(Startup, spawn_hud)
            .add_systems(
                Update,
                (
                    cache_meshlet_asset_stats,
                    toggle_hud_visibility,
                    cycle_ao_debug_view,
                    update_hud_text,
                )
                    .chain(),
            );
    }
}

fn cache_meshlet_asset_stats(
    mut events: MessageReader<AssetEvent<MeshletMesh>>,
    meshlet_assets: Res<Assets<MeshletMesh>>,
    meshlet_entities: Query<&MeshletMesh3d>,
    mut cache: ResMut<MeshletAssetStatsCache>,
) {
    for event in events.read() {
        match event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                if let Some(asset) = meshlet_assets.get(*id) {
                    cache
                        .0
                        .insert(*id, (asset.meshlet_count(), asset.total_triangle_count()));
                }
            }
            AssetEvent::Removed { id } => {
                // 注意：MeshletMeshManager 会调用 remove_untracked（不触发 AssetEvent::Removed）；
                // 仅当资源被真正销毁卸载时才清除缓存
                cache.0.remove(id);
            }
            _ => {}
        }
    }

    // 兜底：若部分资源在首帧事件之外仍停留在 Assets 中，顺带补齐缓存
    for m3d in &meshlet_entities {
        let id = m3d.0.id();
        if !cache.0.contains_key(&id)
            && let Some(asset) = meshlet_assets.get(&m3d.0)
        {
            cache
                .0
                .insert(id, (asset.meshlet_count(), asset.total_triangle_count()));
        }
    }
}

fn spawn_hud(mut commands: Commands) {
    commands
        .spawn((
            HudRoot,
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(10.0),
                right: Val::Px(10.0),
                padding: UiRect::axes(Val::Px(36.0), Val::Px(24.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.05, 0.06, 0.09, 0.78)),
            Visibility::Inherited,
        ))
        .with_children(|parent| {
            parent.spawn((
                HudText,
                Text::new("Initializing stats..."),
                TextFont::from_font_size(28.0),
                TextColor(Color::srgb(0.92, 0.96, 1.0)),
            ));
        });
}

fn toggle_hud_visibility(
    keys: Res<ButtonInput<KeyCode>>,
    mut root_query: Query<&mut Visibility, With<HudRoot>>,
) {
    if !keys.just_pressed(KeyCode::F3) {
        return;
    }
    for mut vis in &mut root_query {
        *vis = if *vis == Visibility::Hidden {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
}

fn cycle_ao_debug_view(
    keys: Res<ButtonInput<KeyCode>>,
    mut cameras: Query<&mut AoDebugView, With<Camera3d>>,
) {
    if !keys.just_pressed(KeyCode::F4) {
        return;
    }
    for mut mode in &mut cameras {
        *mode = mode.next();
        info!("调试视图切换为: {}", mode.label());
    }
}

fn format_compact_count(count: usize) -> String {
    if count >= 1_000_000 {
        format!("{:.2}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

#[derive(SystemParam)]
struct HudLightQueries<'w, 's> {
    dir_lights: Query<'w, 's, &'static DirectionalLight>,
    point_lights: Query<'w, 's, &'static PointLight>,
    spot_lights: Query<'w, 's, &'static SpotLight>,
    gi_volumes: Query<'w, 's, (), With<IrradianceVolume>>,
}

#[derive(SystemParam)]
struct HudGeometryQueries<'w, 's> {
    camera_query: Query<'w, 's, (&'static GlobalTransform, Option<&'static AoDebugView>), With<Camera3d>>,
    chunks: Query<'w, 's, &'static SceneChunk>,
    meshlet_entities: Query<'w, 's, &'static MeshletMesh3d>,
    mesh3d_entities: Query<
        'w,
        's,
        (
            &'static GlobalTransform,
            Option<&'static Mesh3dLodTris>,
            Option<&'static VisibilityRange>,
            Option<&'static InheritedVisibility>,
        ),
        With<Mesh3d>,
    >,
}

fn update_hud_text(
    time: Res<Time>,
    mut timer: ResMut<HudRefreshTimer>,
    diagnostics: Res<DiagnosticsStore>,
    scene_config: Res<MeshletSceneConfig>,
    shadow_map: Res<DirectionalLightShadowMap>,
    irradiance_config: Res<IrradianceConfig>,
    skybox_config: Res<SkyboxConfig>,
    skybox_status: Res<SkyboxStatus>,
    stats_cache: Res<MeshletAssetStatsCache>,
    lights: HudLightQueries,
    geom: HudGeometryQueries,
    root_query: Query<&Visibility, With<HudRoot>>,
    mut hud_text: Query<&mut Text, With<HudText>>,
) {
    // 面板隐藏时跳过统计计算，零额外开销
    if root_query
        .iter()
        .next()
        .is_some_and(|v| *v == Visibility::Hidden)
    {
        return;
    }

    if !timer.0.tick(time.delta()).just_finished() {
        return;
    }

    let Some(mut text) = hud_text.iter_mut().next() else {
        return;
    };

    // 1. FPS 与帧耗时
    let fps_diag = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS);
    let fps_cur = fps_diag.and_then(|d| d.value()).unwrap_or(0.0);
    let fps_avg = fps_diag.and_then(|d| d.smoothed()).unwrap_or(fps_cur);
    let frame_ms = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|d| d.smoothed().or_else(|| d.value()))
        .unwrap_or(0.0);

    // 2. 阴影状态统计
    let dir_total = lights.dir_lights.iter().count();
    let dir_shadow_count = lights
        .dir_lights
        .iter()
        .filter(|d| d.shadow_maps_enabled)
        .count();
    let shadow_line = if scene_config.shadows_enabled && dir_shadow_count > 0 {
        format!(
            "Shadow: ON {}px @ {:.0}m ({} casc) | Foliage: {}",
            shadow_map.size,
            scene_config.shadow_distance,
            scene_config.shadow_cascades,
            if scene_config.foliage_shadows {
                "ON"
            } else {
                "OFF"
            }
        )
    } else {
        "Shadow: OFF (Irradiance GI only)".to_string()
    };

    // 3. 光源状态统计（实时局部光激活数 vs 场景总数）
    let mut local_total = 0usize;
    let mut local_active = 0usize;
    for pt in &lights.point_lights {
        local_total += 1;
        if pt.intensity > 0.01 {
            local_active += 1;
        }
    }
    for sp in &lights.spot_lights {
        local_total += 1;
        if sp.intensity > 0.01 {
            local_active += 1;
        }
    }
    let gi_state = if lights.gi_volumes.is_empty() {
        format!("OFF ({:.0})", irradiance_config.intensity)
    } else {
        format!("ON ({:.0})", irradiance_config.intensity)
    };

    // 4. Chunk 流送与 Meshlet 统计
    let mut chunk_total = 0usize;
    let mut chunk_loaded = 0usize;
    for ch in &geom.chunks {
        chunk_total += 1;
        if ch.loaded {
            chunk_loaded += 1;
        }
    }

    let mut meshlet_inst_count = 0usize;
    let mut total_clusters = 0usize;
    let mut total_meshlet_pool_tris = 0usize;
    for m3d in &geom.meshlet_entities {
        meshlet_inst_count += 1;
        if let Some(&(clusters, tris)) = stats_cache.0.get(&m3d.0.id()) {
            total_clusters += clusters;
            total_meshlet_pool_tris += tris;
        }
    }

    // 5. Mesh3d LOD 可见多边形统计（结合相机与 VisibilityRange 估算当前激活 LOD 面数）
    let mut cam_pos = None;
    let mut debug_view_label = AoDebugView::Lit.label();
    if let Some((tf, dbg_opt)) = geom.camera_query.iter().next() {
        cam_pos = Some(tf.translation());
        if let Some(dbg) = dbg_opt {
            debug_view_label = dbg.label();
        }
    }
    let mut mesh3d_total_entities = 0usize;
    let mut mesh3d_visible_entities = 0usize;
    let mut mesh3d_visible_tris = 0usize;
    for (tf, lod_tris_opt, vis_range_opt, inherited_vis_opt) in &geom.mesh3d_entities {
        mesh3d_total_entities += 1;
        if inherited_vis_opt.is_some_and(|v| !v.get()) {
            continue;
        }
        let in_range = match (cam_pos, vis_range_opt) {
            (Some(cp), Some(vr)) => {
                let d = cp.distance(tf.translation());
                d >= vr.start_margin.start && d <= vr.end_margin.end
            }
            _ => true,
        };
        if !in_range {
            continue;
        }
        mesh3d_visible_entities += 1;
        if let Some(lod_tris) = lod_tris_opt {
            mesh3d_visible_tris += lod_tris.0;
        }
    }

    let total_poly_estimate = total_meshlet_pool_tris + mesh3d_visible_tris;

    let sky_state = if skybox_status.visible {
        format!(
            "ON (env {:.0} | {})",
            skybox_config.env_intensity, skybox_status.mode_label
        )
    } else {
        format!(
            "OFF (env {:.0} | {})",
            skybox_config.env_intensity, skybox_status.mode_label
        )
    };

    text.0 = format!(
        "FPS: {:.1} (avg {:.1}) | Frame: {:.1} ms | View: {}\n\
         {}\n\
         Lights: Dir {} (Shadow {}) | Local {}/{} | GI: {} | Sky: {}\n\
         Chunks: {}/{} | Meshlet: {} inst, {} clusters ({} pool tris)\n\
         Mesh3d LOD: {}/{} visible ({} tris) | Total Pool Tris: {}\n\
         [F3] HUD  [F4] View  [Space] GI  [[ / ]] GI±  [B] Sky  [- / =] Env±",
        fps_cur,
        fps_avg,
        frame_ms,
        debug_view_label,
        shadow_line,
        dir_total,
        dir_shadow_count,
        local_active,
        local_total,
        gi_state,
        sky_state,
        chunk_loaded,
        chunk_total,
        meshlet_inst_count,
        format_compact_count(total_clusters),
        format_compact_count(total_meshlet_pool_tris),
        mesh3d_visible_entities,
        mesh3d_total_entities,
        format_compact_count(mesh3d_visible_tris),
        format_compact_count(total_poly_estimate),
    );
}
