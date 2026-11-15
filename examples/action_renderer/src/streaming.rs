use bevy::{
    pbr::experimental::meshlet::{MeshletMesh, MeshletMesh3d},
    prelude::*,
};

/// 区块/实例级资源流送配置。
#[derive(Resource, Clone)]
pub struct StreamingConfig {
    pub enabled: bool,
    pub load_distance: f32,
    pub unload_distance: f32,
    pub unload_delay_secs: f32,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            load_distance: 80.0,
            unload_distance: 110.0,
            unload_delay_secs: 5.0,
        }
    }
}

/// 挂在需要按距离流送的 Meshlet 节点上的调度组件。
#[derive(Component)]
pub struct StreamedMeshletNode {
    pub asset_path: String,
    pub material: Handle<StandardMaterial>,
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
    pub loaded_handle: Option<Handle<MeshletMesh>>,
    pub spawned_entity: Option<Entity>,
    pub out_of_range_timer: f32,
}

pub struct StreamingPlugin;

impl Plugin for StreamingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<StreamingConfig>()
            .add_systems(Update, update_meshlet_streaming);
    }
}

fn distance_to_aabb(point: Vec3, aabb_min: Vec3, aabb_max: Vec3) -> f32 {
    let clamped = point.clamp(aabb_min, aabb_max);
    point.distance(clamped)
}

fn update_meshlet_streaming(
    mut commands: Commands,
    time: Res<Time>,
    config: Res<StreamingConfig>,
    asset_server: Res<AssetServer>,
    camera_query: Query<&GlobalTransform, With<Camera3d>>,
    mut nodes: Query<(&mut StreamedMeshletNode, &Transform)>,
) {
    let Some(cam_tf) = camera_query.iter().next() else {
        return;
    };
    let cam_pos = cam_tf.translation();
    let dt = time.delta_secs();

    for (mut node, node_tf) in &mut nodes {
        let dist = if config.enabled {
            // 将局部 AABB 近似变换到世界空间计算距离
            let world_min = node_tf.transform_point(node.aabb_min);
            let world_max = node_tf.transform_point(node.aabb_max);
            let min = world_min.min(world_max);
            let max = world_min.max(world_max);
            distance_to_aabb(cam_pos, min, max)
        } else {
            0.0
        };

        if dist <= config.load_distance || !config.enabled {
            node.out_of_range_timer = 0.0;
            if node.loaded_handle.is_none() {
                let handle: Handle<MeshletMesh> = asset_server.load(&node.asset_path);
                let spawned = commands
                    .spawn((
                        MeshletMesh3d(handle.clone()),
                        MeshMaterial3d(node.material.clone()),
                        *node_tf,
                    ))
                    .id();
                node.loaded_handle = Some(handle);
                node.spawned_entity = Some(spawned);
            }
        } else if dist >= config.unload_distance && node.loaded_handle.is_some() {
            node.out_of_range_timer += dt;
            if node.out_of_range_timer >= config.unload_delay_secs {
                if let Some(entity) = node.spawned_entity.take() {
                    commands.entity(entity).despawn();
                }
                // 释放 Handle 后，Bevy 资产系统与 MeshletMeshManager::remove 会回收对应显存切片
                node.loaded_handle = None;
                node.out_of_range_timer = 0.0;
            }
        } else {
            node.out_of_range_timer = 0.0;
        }
    }
}
