mod character_stub;
mod irradiance;
mod scene_loader;
mod streaming;

use std::path::PathBuf;

use bevy::{
    anti_alias::fxaa::Fxaa,
    asset::io::AssetSourceBuilder,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    pbr::experimental::meshlet::MeshletPlugin,
    post_process::bloom::Bloom,
    prelude::*,
    render::{
        settings::{Backends, Dx12Compiler, WgpuSettings},
        RenderPlugin,
    },
};
use clap::Parser;

use character_stub::CharacterActionStubPlugin;
use irradiance::{IrradianceConfig, IrradianceScenePlugin};
use scene_loader::{MeshletSceneConfig, MeshletScenePlugin};
use streaming::{StreamingConfig, StreamingPlugin};

#[derive(Parser, Debug)]
#[command(
    name = "action_renderer",
    about = "Standalone hybrid Meshlet + Irradiance Volume renderer for character action games"
)]
struct Cli {
    /// Meshlet 场景目录（包含 meshes/ 与 .meshlet_scene.json）
    #[arg(
        long,
        default_value = r"C:\Users\12410\Downloads\lego_technic_test\out_meshlet"
    )]
    scene_dir: PathBuf,

    /// 可选的 .meshlet_scene.json 文件名（未指定时自动扫描 scene_dir）
    #[arg(long)]
    manifest: Option<String>,

    /// Irradiance Volume .ktx2 路径（相对于 Bevy assets 目录）
    #[arg(
        long,
        default_value = "irradiance_volumes/lego_technic_bugatti_chiron_lighted_Irradiance_Info.ktx2"
    )]
    irradiance_ktx2: String,

    /// Irradiance Volume .meta.json 路径（相对于 Bevy assets 目录）
    #[arg(
        long,
        default_value = "irradiance_volumes/lego_technic_bugatti_chiron_lighted_Irradiance_Info.meta.json"
    )]
    irradiance_meta: String,

    /// Irradiance Volume 间接光强度
    #[arg(long, default_value_t = 1200.0)]
    irradiance_intensity: f32,

    /// 禁用按距离流送（全部实例常驻）
    #[arg(long, default_value_t = false)]
    no_streaming: bool,
}

fn resolve_bevy_assets_dir() -> String {
    let candidates = [
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets"),
        PathBuf::from("assets"),
        PathBuf::from("../../assets"),
        PathBuf::from(r"C:\bevy\assets"),
    ];
    for candidate in candidates {
        if candidate.is_dir()
            && let Ok(abs) = std::path::absolute(&candidate)
        {
            return abs.to_string_lossy().to_string();
        }
    }
    r"C:\bevy\assets".to_string()
}

fn main() {
    let cli = Cli::parse();
    let bevy_assets_dir = resolve_bevy_assets_dir();
    let scene_dir_str = cli.scene_dir.to_string_lossy().to_string();

    App::new()
        .register_asset_source(
            "meshlet_scene",
            AssetSourceBuilder::platform_default(&scene_dir_str, None),
        )
        .insert_resource(GlobalAmbientLight {
            color: Color::WHITE,
            brightness: 0.02,
            ..default()
        })
        .insert_resource(MeshletSceneConfig {
            scene_dir: cli.scene_dir,
            manifest_file: cli.manifest,
            asset_source: "meshlet_scene".to_string(),
        })
        .insert_resource(IrradianceConfig {
            ktx2_asset_path: cli.irradiance_ktx2,
            meta_asset_path: cli.irradiance_meta,
            intensity: cli.irradiance_intensity,
        })
        .insert_resource(StreamingConfig {
            enabled: !cli.no_streaming,
            ..default()
        })
        .add_plugins((
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: bevy_assets_dir,
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Action Renderer (Meshlet + Irradiance Volume + Hybrid)".into(),
                        ..default()
                    }),
                    ..default()
                })
                .set(RenderPlugin {
                    render_creation: WgpuSettings {
                        backends: Some(Backends::DX12 | Backends::VULKAN),
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
            FreeCameraPlugin,
            FrameTimeDiagnosticsPlugin::default(),
            LogDiagnosticsPlugin::default(),
            MeshletScenePlugin,
            IrradianceScenePlugin,
            StreamingPlugin,
            CharacterActionStubPlugin,
        ))
        .add_systems(Startup, setup_camera)
        .run();
}

fn setup_camera(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(4.5, 2.2, 5.5).looking_at(Vec3::new(0.0, 0.6, 0.8), Vec3::Y),
        // MeshletPlugin 强制要求关闭 MSAA；使用 FXAA 避免触发 Bevy 主干 meshlet_prepass 的 bind_group 槽位 bug
        Msaa::Off,
        Fxaa::default(),
        Bloom::NATURAL,
        EnvironmentMapLight {
            diffuse_map: asset_server.load("environment_maps/pisa_diffuse_rgb9e5_zstd.ktx2"),
            specular_map: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
            intensity: 150.0,
            ..default()
        },
        FreeCamera::default(),
    ));
}
