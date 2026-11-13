//! Meshlet rendering for dense high-poly scenes (experimental).

// Note: This example showcases the meshlet API, but is not the type of scene that would benefit from using meshlets.

use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    light::{CascadeShadowConfigBuilder, DirectionalLightShadowMap},
    pbr::experimental::meshlet::{MeshletMesh3d, MeshletPlugin},
    prelude::*,
    render::{
        render_resource::AsBindGroup,
        settings::{Backends, Dx12Compiler, WgpuSettings},
        RenderPlugin,
    },
};
use std::f32::consts::PI;



const ASSET_URL: &str =
    "models/Military_Trenches_Tunnel.meshlet_mesh";
fn main() {
    App::new()
        .insert_resource(DirectionalLightShadowMap { size: 1024 })
        .add_plugins((
            DefaultPlugins.set(RenderPlugin {
                render_creation: WgpuSettings {
                    backends: Some(Backends::DX12),
                    dx12_shader_compiler: Dx12Compiler::DynamicDxc {
                        dxc_path: "dxcompiler.dll".to_string(),
                    },
                    ..default()
                }
                .into(),
                ..default()
            }),
            MeshletPlugin {
                cluster_buffer_slots: 1 << 18,
            },
            MaterialPlugin::<MeshletDebugMaterial>::default(),
            FreeCameraPlugin,
            FrameTimeDiagnosticsPlugin::default(),
            LogDiagnosticsPlugin::default(),
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, bunny_wiggler)
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
    mut debug_materials: ResMut<Assets<MeshletDebugMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    //add texture loading
use bevy::image::ImageLoaderSettings;

// 1. 加载固有色贴图 (sRGB)
let albedo_texture = asset_server.load("textures/bunker/albedo.jpeg");

// 2. 加载法线贴图 (必须为线性颜色空间 Linear, is_srgb = false)
let normal_texture = asset_server
    .load_builder()
    .with_settings(|settings: &mut ImageLoaderSettings| {
        settings.is_srgb = false;
    })
    .load("textures/bunker/normal.jpeg");

// 3. 加载 ORM 贴图 (线性空间 Linear, is_srgb = false)
let orm_texture = asset_server
    .load_builder()
    .with_settings(|settings: &mut ImageLoaderSettings| {
        settings.is_srgb = false;
    })
    .load("textures/bunker/orm.jpeg");

// 4. 构建 Bunker PBR 材质
let bunker_material = standard_materials.add(StandardMaterial {
    base_color: Color::WHITE,
    base_color_texture: Some(albedo_texture),

    normal_map_texture: Some(normal_texture),

    // ORM 贴图同时作为金属/粗糙度贴图与环境遮蔽贴图
    metallic_roughness_texture: Some(orm_texture.clone()),
    occlusion_texture: Some(orm_texture),

    // 关键：将因子设为 1.0，完全由贴图像素决定金属度与粗糙度
    metallic: 1.0,
    perceptual_roughness: 1.0,

    // 隧道为内凹中空结构，开启双面渲染可以看到内外两侧钢板
    double_sided: true,
    cull_mode: None,

    ..default()
});
 //texture loading end

    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(Vec3::new(3.5, 2.5, 4.0)).looking_at(Vec3::ZERO, Vec3::Y),
        Msaa::Off,
        EnvironmentMapLight {
            diffuse_map: asset_server.load("environment_maps/pisa_diffuse_rgb9e5_zstd.ktx2"),
            specular_map: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
            intensity: 150.0,
            ..default()
        },
        FreeCamera::default(),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::FULL_DAYLIGHT,
            shadow_maps_enabled: true,
            ..default()
        },
        CascadeShadowConfigBuilder {
            num_cascades: 1,
            maximum_distance: 15.0,
            ..default()
        }
        .build(),
        Transform::from_rotation(Quat::from_euler(EulerRot::ZYX, 0.0, PI * -0.15, PI * -0.15)),
    ));

    // A custom file format storing a [`bevy_mesh::Mesh`]
    // that has been converted to a [`bevy_pbr::meshlet::MeshletMesh`]
    // using [`bevy_pbr::meshlet::MeshletMesh::from_mesh`], which is
    // a function only available when the `meshlet_processor` cargo feature is enabled.
    let meshlet_mesh_handle = asset_server.load(ASSET_URL);
    let debug_material = debug_materials.add(MeshletDebugMaterial::default());

    // 2D 网格生成：奇数行使用 Debug 材质，偶数行使用色相渐变材质
    let grid_x = 4; // 每排 9 只 (-4..=4)
    let grid_z = 4; // 共 9 排 (-4..=4)，总计 81 只高精度兔子
    let spacing_x = 4.0;
    let spacing_z = 4.0;

    for z in -grid_z..=grid_z {
        let is_debug_row = (z + grid_z) % 2 == 1; // 一排渐变色，一排 debug 材质交替
        for x in -grid_x..=grid_x {
            let translation = Vec3::new(x as f32 * spacing_x, 0.0, z as f32 * spacing_z);

            let mut entity_cmds = if is_debug_row {
                // Debug 材质排（面朝反方向展示不同角度特征）
                commands.spawn((
                    MeshletMesh3d(meshlet_mesh_handle.clone()),
                    MeshMaterial3d(debug_material.clone()),
                    Transform::default()
                        .with_scale(Vec3::splat(0.01))
                        .with_rotation(Quat::from_rotation_y(PI))
                        .with_translation(translation),
                ))
            } else {
                // 色相渐变排（使用色相环计算平滑彩虹过渡颜色）
                // 尝试改用标准材质
                let norm_x = (x + grid_x) as f32 / (2 * grid_x) as f32;
                let hue = norm_x * 360.0;
                //let color = Hsla::hsl(hue, 0.85, 0.55).into();

                commands.spawn((
                    MeshletMesh3d(meshlet_mesh_handle.clone()),
                    MeshMaterial3d(bunker_material.clone()),
                    Transform::default()
                        .with_scale(Vec3::splat(0.01))
                        .with_translation(translation),
                ))
            };

            // 给中心的一只兔子添加摇摆动作
            if x == 0 && z == 0 {
                entity_cmds.insert(BunnyWiggler);
            }
        }
    }

    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(20.0, 20.0))),
        MeshMaterial3d(standard_materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 1.0,
            ..default()
        })),
    ));
}

#[derive(Component)]
struct BunnyWiggler;

fn bunny_wiggler(mut bunny: Query<&mut Transform, With<BunnyWiggler>>, time: Res<Time>) {
    bunny.single_mut().as_deref_mut().unwrap().translation.z +=
        ops::cos(time.elapsed_secs() * 10.0) * 0.003;
}

#[derive(Asset, TypePath, AsBindGroup, Clone, Default)]
struct MeshletDebugMaterial {
    _dummy: (),
}

impl Material for MeshletDebugMaterial {}
