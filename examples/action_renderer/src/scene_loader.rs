use std::path::{Path, PathBuf};

use bevy::{
    image::ImageLoaderSettings,
    light::CascadeShadowConfigBuilder,
    prelude::*,
};
use serde::Deserialize;

use crate::streaming::StreamedMeshletNode;

#[derive(Debug, Clone, Deserialize)]
pub struct MeshletSceneManifest {
    pub version: u32,
    pub source_file: String,
    pub materials: Vec<SceneMaterial>,
    pub meshlet_instances: Vec<SceneMeshletInstance>,
    pub lights: Vec<SceneLight>,
    #[serde(default)]
    pub skipped_non_opaque: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SceneMaterial {
    pub id: u32,
    pub name: String,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub base_color_texture: Option<String>,
    pub normal_texture: Option<String>,
    pub metallic_roughness_texture: Option<String>,
    pub occlusion_texture: Option<String>,
    pub emissive_texture: Option<String>,
    pub double_sided: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SceneMeshletInstance {
    pub name: String,
    pub mesh: String,
    pub material_id: u32,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub dynamic: bool,
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

#[derive(Resource, Clone)]
pub struct MeshletSceneConfig {
    /// 磁盘上的场景目录（包含 meshes/ 和可选的 .meshlet_scene.json）
    pub scene_dir: PathBuf,
    /// 可选的清单文件名（未指定时自动扫描 scene_dir 下的 *.meshlet_scene.json）
    pub manifest_file: Option<String>,
    /// 注册到 AssetServer 的资产源名称，例如 "meshlet_scene"
    pub asset_source: String,
}

pub struct MeshletScenePlugin;

impl Plugin for MeshletScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_meshlet_scene);
    }
}

fn find_manifest_path(scene_dir: &Path, manifest_file: Option<&str>) -> Option<PathBuf> {
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

fn load_linear_texture(asset_server: &AssetServer, asset_uri: &str) -> Handle<Image> {
    asset_server
        .load_builder()
        .with_settings(|s: &mut ImageLoaderSettings| {
            s.is_srgb = false;
        })
        .load(asset_uri.to_string())
}

fn setup_meshlet_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
    config: Res<MeshletSceneConfig>,
) {
    let prefix = format!("{}://", config.asset_source);

    if let Some(manifest_path) =
        find_manifest_path(&config.scene_dir, config.manifest_file.as_deref())
    {
        info!("加载 Meshlet 场景清单: {}", manifest_path.display());
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

        // 1. 构建去重材质表
        let mut material_handles = Vec::with_capacity(manifest.materials.len());
        for mat in &manifest.materials {
            let base_color_texture = mat
                .base_color_texture
                .as_ref()
                .map(|p| asset_server.load(format!("{prefix}{p}")));
            let normal_map_texture = mat
                .normal_texture
                .as_ref()
                .map(|p| load_linear_texture(&asset_server, &format!("{prefix}{p}")));
            let metallic_roughness_texture = mat
                .metallic_roughness_texture
                .as_ref()
                .map(|p| load_linear_texture(&asset_server, &format!("{prefix}{p}")));
            let occlusion_texture = mat
                .occlusion_texture
                .as_ref()
                .map(|p| load_linear_texture(&asset_server, &format!("{prefix}{p}")));
            let emissive_texture = mat
                .emissive_texture
                .as_ref()
                .map(|p| asset_server.load(format!("{prefix}{p}")));

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
                ..default()
            });
            material_handles.push(handle);
        }

        let fallback_mat = standard_materials.add(StandardMaterial::default());

        // 2. 注册所有 Meshlet 实例到流送调度器
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
            commands.spawn((
                Name::new(inst.name.clone()),
                tf,
                StreamedMeshletNode {
                    asset_path: format!("{prefix}{}", inst.mesh),
                    material: mat_handle,
                    aabb_min: Vec3::from_array(inst.aabb_min),
                    aabb_max: Vec3::from_array(inst.aabb_max),
                    loaded_handle: None,
                    spawned_entity: None,
                    out_of_range_timer: 0.0,
                },
            ));
        }

        // 3. 生成清单中的灯光；若 GLB 未包含灯光则补一个默认日光
        if manifest.lights.is_empty() {
            warn!("场景清单未包含灯光，生成默认 DirectionalLight");
            spawn_default_light(&mut commands);
        } else {
            for light in &manifest.lights {
                let color = Color::linear_rgb(light.color[0], light.color[1], light.color[2]);
                let tf = Transform {
                    translation: Vec3::from_array(light.translation),
                    rotation: Quat::from_array(light.rotation),
                    scale: Vec3::ONE,
                };
                match light.kind.as_str() {
                    "Directional" => {
                        commands.spawn((
                            DirectionalLight {
                                color,
                                illuminance: light.intensity.max(1000.0),
                                shadow_maps_enabled: true,
                                ..default()
                            },
                            CascadeShadowConfigBuilder {
                                num_cascades: 2,
                                maximum_distance: 25.0,
                                ..default()
                            }
                            .build(),
                            tf,
                        ));
                    }
                    "Spot" => {
                        commands.spawn((
                            SpotLight {
                                color,
                                intensity: light.intensity,
                                range: light.range.unwrap_or(20.0),
                                inner_angle: light.inner_cone_angle.unwrap_or(0.0),
                                outer_angle: light.outer_cone_angle.unwrap_or(0.785),
                                shadow_maps_enabled: true,
                                ..default()
                            },
                            tf,
                        ));
                    }
                    _ => {
                        commands.spawn((
                            PointLight {
                                color,
                                intensity: light.intensity,
                                range: light.range.unwrap_or(20.0),
                                shadow_maps_enabled: true,
                                ..default()
                            },
                            tf,
                        ));
                    }
                }
            }
        }

        info!(
            "Meshlet 场景初始化完成: {} 个材质, {} 个实例, {} 个灯光",
            manifest.materials.len(),
            manifest.meshlet_instances.len(),
            manifest.lights.len()
        );
    } else {
        // 回退模式：未找到 .meshlet_scene.json 时，直接扫描 meshes/*.meshlet_mesh
        let meshes_dir = config.scene_dir.join("meshes");
        warn!(
            "未在 {} 找到 .meshlet_scene.json，回退至目录扫描: {} (建议使用 makemeshlet --scene 重新导出完整材质与坐标)",
            config.scene_dir.display(),
            meshes_dir.display()
        );

        let default_mat = standard_materials.add(StandardMaterial {
            base_color: Color::srgb(0.72, 0.74, 0.78),
            perceptual_roughness: 0.45,
            metallic: 0.2,
            ..default()
        });

        let mut count = 0usize;
        if let Ok(entries) = std::fs::read_dir(&meshes_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("meshlet_mesh")
                    && let Some(fname) = path.file_name().and_then(|s| s.to_str())
                {
                    commands.spawn((
                        Name::new(fname.to_string()),
                        Transform::default(),
                        StreamedMeshletNode {
                            asset_path: format!("{prefix}meshes/{fname}"),
                            material: default_mat.clone(),
                            aabb_min: Vec3::splat(-5.0),
                            aabb_max: Vec3::splat(5.0),
                            loaded_handle: None,
                            spawned_entity: None,
                            out_of_range_timer: 0.0,
                        },
                    ));
                    count += 1;
                }
            }
        }
        spawn_default_light(&mut commands);
        info!("回退模式已注册 {} 个 .meshlet_mesh 文件", count);
    }
}

fn spawn_default_light(commands: &mut Commands) {
    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::OVERCAST_DAY,
            shadow_maps_enabled: true,
            ..default()
        },
        CascadeShadowConfigBuilder {
            num_cascades: 2,
            maximum_distance: 25.0,
            ..default()
        }
        .build(),
        Transform::from_rotation(Quat::from_euler(
            EulerRot::ZYX,
            0.0,
            -std::f32::consts::FRAC_PI_4,
            -std::f32::consts::FRAC_PI_4,
        )),
    ));
}
