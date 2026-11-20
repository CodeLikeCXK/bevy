mod character_stub;
mod hud;
mod irradiance;
mod lod_loader;
mod scene_loader;
mod skybox;
mod streaming;
mod vlm;

use std::path::{Path, PathBuf};

use bevy::{
    anti_alias::fxaa::Fxaa,
    asset::io::AssetSourceBuilder,
    camera::RenderTarget,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    image::{ImageAddressMode, ImagePlugin, ImageSampler, ImageSamplerDescriptor},
    light::DirectionalLightShadowMap,
    pbr::{experimental::meshlet::MeshletPlugin, AoDebugView},
    post_process::bloom::Bloom,
    prelude::*,
    render::{
        diagnostic::RenderDiagnosticsPlugin,
        render_resource::{Extent3d, TextureFormat},
        settings::{Backends, Dx12Compiler, WgpuSettings},
        RenderPlugin,
    },
    window::{PresentMode, PrimaryWindow, WindowResized, WindowResolution},
};
use clap::Parser;

use character_stub::CharacterActionStubPlugin;
use hud::HudPlugin;
use irradiance::{IrradianceConfig, IrradianceScenePlugin};
use lod_loader::Mesh3dLodPlugin;
use scene_loader::{find_manifest_path, MeshletSceneConfig, MeshletSceneManifest, MeshletScenePlugin, SceneWorldHdri};
use skybox::{SkyboxConfig, SkyboxScenePlugin};
use streaming::{StreamingConfig, StreamingPlugin};
use vlm::{find_vlm_file_in_dir, VlmConfig, VlmScenePlugin};

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

    /// 场景最大实时局部光激活上限（依靠视锥前向优先 + GPU Clustered Lighting；0 表示不限制）
    #[arg(long, default_value_t = 16)]
    max_lights: usize,

    /// 区块加载时每帧最多生成的 Meshlet 实体数，避免 GPU staging/PersistentBuffer 同帧暴涨触发 TDR（0 表示不限制）
    #[arg(long, default_value_t = 2)]
    max_spawns_per_frame: usize,

    /// 区块开始异步加载的相机距离（米，匹配 --grid-size 10 空间切分）
    #[arg(long, default_value_t = 32.0)]
    load_distance: f32,

    /// 区块滞后卸载的相机距离（米）
    #[arg(long, default_value_t = 45.0)]
    unload_distance: f32,

    /// 窗口渲染宽度（像素，默认 960 适配 8cx Gen 3 等集显以兼顾阴影与 30+ FPS；可指定 1280）
    #[arg(long, default_value_t = 960)]
    width: u32,

    /// 窗口渲染高度（像素，默认 540；可指定 720）
    #[arg(long, default_value_t = 540)]
    height: u32,

    /// 强制 DPI 缩放倍率（默认 1.0，避免 Windows 高分屏把物理像素放大 2x~4x 导致集显掉帧）
    #[arg(long, default_value_t = 1.0)]
    scale_factor: f32,

    /// 平行光阴影贴图尺寸（2 的幂，默认 512 适配集显；0 表示关闭实时阴影投射以大幅提帧）
    #[arg(long, default_value_t = 512)]
    shadow_size: usize,

    /// 平行光阴影覆盖最远半径（米）；配合 2 级联可在 512px 下同时兼顾近处锐度与远处阴影显示距离
    #[arg(long, default_value_t = 48.0)]
    shadow_distance: f32,

    /// 平行光阴影级联数（1~4，默认 2 级联 @ 512px：纹素面积仅为单级 1024px 的一半，远景不断、近景更清）
    #[arg(long, default_value_t = 2)]
    shadow_cascades: usize,

    /// 第 1 级联（高精度近景阴影）覆盖半径（米，仅当级联数 > 1 时生效）
    #[arg(long, default_value_t = 12.0)]
    first_cascade_distance: f32,

    /// 覆盖自动计算的阴影法线偏移（texels）；未指定时按阴影纹素物理尺寸自动限制，防止 512px 阴影穿透薄窗帘产生黑斑
    #[arg(long)]
    shadow_normal_bias: Option<f32>,

    /// 覆盖自动计算的阴影深度偏移
    #[arg(long)]
    shadow_depth_bias: Option<f32>,

    /// 关闭垂直同步（避免 60Hz VSync 在帧耗时刚过 16.6ms/33.3ms 时断崖掉档至 30/15 FPS）
    #[arg(long, default_value_t = false)]
    no_vsync: bool,

    /// 开启 GPU Pass 级耗时诊断日志（RenderDiagnosticsPlugin）
    #[arg(long, default_value_t = false)]
    gpu_profile: bool,

    /// 开启 Bloom 泛光后处理（默认关闭以节省集显显存带宽）
    #[arg(long, default_value_t = false)]
    bloom: bool,

    /// 允许植物/半透明 Mesh3d LOD 物体投射实时阴影（默认关闭以减少 AlphaMask 阴影 Overdraw）
    #[arg(long, default_value_t = false)]
    foliage_shadows: bool,

    /// 清单灯光统一强度倍率（可配合快速调亮/调暗场景直光）
    #[arg(long, default_value_t = 1.0)]
    light_intensity_scale: f32,

    /// 当场景清单中的 Sun/DirectionalLight 导出强度 <= 0.01（常见于 FBX 导入 Blender 未设 Strength）时的兜底照度 (lux)
    #[arg(long, default_value_t = 22000.0)]
    default_sun_illuminance: f32,

    /// 当场景清单中的 Point/Spot 灯光导出强度 <= 0.01（如 Power=0.000）时的兜底流明 (lumens)
    #[arg(long, default_value_t = 1600.0)]
    default_point_lumens: f32,

    /// 天空盒 / HDRI 贴图路径（支持 Cubemap .ktx2 / 6层拼接图 / 2:1 等距柱状 .hdr/.exr；未指定时自动读取场景导出的 World HDRI）
    #[arg(long, visible_alias = "hdri")]
    skybox: Option<String>,

    /// 可选预烘培 Diffuse 环境光 Cubemap (.ktx2)；配合 --env-specular 可跳过运行时 IBL 卷积
    #[arg(long)]
    env_diffuse: Option<String>,

    /// 可选预烘培 Specular 环境光 Cubemap (.ktx2)；配合 --env-diffuse 可跳过运行时 IBL 卷积
    #[arg(long)]
    env_specular: Option<String>,

    /// 当输入为 2:1 等距柱状 HDRI 时展开的 Cubemap 单面尺寸（2 的幂，默认 512）
    #[arg(long, default_value_t = 512)]
    skybox_cubemap_size: u32,

    /// 天空盒背景亮度 (cd/m^2；未指定时按场景 HDRI strength 自动缩放，默认约 1500)
    #[arg(long)]
    skybox_brightness: Option<f32>,

    /// HDRI 环境反射/漫反射强度 (cd/m^2；未指定时默认 150~250)
    #[arg(long)]
    env_intensity: Option<f32>,

    /// 天空盒与环境光绕世界 Y 轴旋转角度（度；未指定时读取场景清单或 0）
    #[arg(long)]
    skybox_yaw: Option<f32>,

    /// 禁用天空盒显示与场景清单自动加载 HDRI（仅保留默认占位环境光）
    #[arg(long, default_value_t = false)]
    no_skybox: bool,

    /// 3D 场景内部分辨率缩放比例 (0.25..=1.0)，低于 1.0 时先渲染到缩放 RT 再全屏双线性放大（HUD 保持原生清晰度）
    #[arg(long, default_value_t = 1.0)]
    render_scale: f32,

    /// 禁用自动加载场景附带的预构建 Virtual Lightmap (.vlm.bin)
    #[arg(long, default_value_t = false)]
    no_vlm: bool,

    /// 可选显式覆盖 Virtual Lightmap (.vlm.bin) 路径（未指定时自动随场景清单或场景目录加载，无需额外参数）
    #[arg(long)]
    vlm: Option<String>,

    /// Virtual Lightmap 烘焙漫反射强度倍率（默认 1.0，运行时可按 ; / ' 微调）
    #[arg(long, default_value_t = 1.0)]
    vlm_intensity: f32,

    /// 即使启用了 VLM 也强制静态场景继续投射实时 CSM 阴影（默认关闭以省去静态高模阴影光栅化开销，仅动态角色投射实时阴影）
    #[arg(long, default_value_t = false)]
    static_realtime_shadows: bool,
}

#[derive(Resource, Clone)]
struct RenderOptions {
    enable_bloom: bool,
}

#[derive(Resource, Clone, Copy, Debug)]
pub struct RenderResolutionConfig {
    pub scale: f32,
    pub internal_size: UVec2,
    pub output_size: UVec2,
}

#[derive(Resource)]
struct ScaledRenderTarget {
    image: Handle<Image>,
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

/// 若传入的是绝对路径或磁盘上直接存在的外部文件，则自动为其父目录注册独立的 Bevy AssetSource，
/// 并改写为 `<source_name>://<filename>`，避免触发 `UnapprovedPathMode::Forbid` 报错；
/// 若为相对 `assets/` 目录的资源路径则原样返回。
fn resolve_external_asset_source(app: &mut App, raw_path: &str, source_name: &str) -> String {
    let path = Path::new(raw_path);
    if path.is_absolute() || path.exists() {
        let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
            return raw_path.to_string();
        };
        let parent_dir = path
            .parent()
            .filter(|d| !d.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let dir_str = parent_dir.to_string_lossy().to_string();
        app.register_asset_source(
            source_name.to_string(),
            AssetSourceBuilder::platform_default(&dir_str, None),
        );
        return format!("{source_name}://{file_name}");
    }
    raw_path.to_string()
}

fn peek_manifest_extras(
    scene_dir: &Path,
    manifest_file: Option<&str>,
) -> (Option<SceneWorldHdri>, Option<String>) {
    let Some(manifest_path) = find_manifest_path(scene_dir, manifest_file) else {
        return (None, None);
    };
    let Ok(bytes) = std::fs::read(manifest_path) else {
        return (None, None);
    };
    let Ok(manifest) = serde_json::from_slice::<MeshletSceneManifest>(&bytes) else {
        return (None, None);
    };
    (manifest.world_hdri, manifest.vlm_file)
}

/// 优先解析为 `meshlet_scene://` 相对路径，其次通过外部文件源或内置 `assets/` 加载
fn resolve_scene_or_external_asset(
    app: &mut App,
    scene_dir: &Path,
    raw_path: &str,
    source_name: &str,
) -> String {
    let norm = raw_path.replace('\\', "/");
    if !Path::new(raw_path).is_absolute() && scene_dir.join(&norm).is_file() {
        return format!("meshlet_scene://{norm}");
    }
    resolve_external_asset_source(app, raw_path, source_name)
}

fn main() {
    let cli = Cli::parse();
    let bevy_assets_dir = resolve_bevy_assets_dir();
    let scene_dir_str = cli.scene_dir.to_string_lossy().to_string();

    let mut app = App::new();
    app.register_asset_source(
        "meshlet_scene",
        AssetSourceBuilder::platform_default(&scene_dir_str, None),
    );

    let ktx2_asset_path =
        resolve_external_asset_source(&mut app, &cli.irradiance_ktx2, "irradiance_ktx2_src");
    let meta_asset_path =
        resolve_external_asset_source(&mut app, &cli.irradiance_meta, "irradiance_meta_src");

    let (manifest_hdri_raw, manifest_vlm_file) =
        peek_manifest_extras(&cli.scene_dir, cli.manifest.as_deref());
    let manifest_hdri = if cli.no_skybox {
        None
    } else {
        manifest_hdri_raw
    };

    // 自动随场景发现并读取预构建 Virtual Lightmap（无需额外命令行参数）：
    // 优先 --vlm 覆盖 -> 清单中的 vlm_file -> 自动扫描 scene_dir 下的 *.vlm.bin
    let raw_vlm_path = if cli.no_vlm {
        None
    } else {
        cli.vlm
            .clone()
            .or(manifest_vlm_file)
            .or_else(|| find_vlm_file_in_dir(&cli.scene_dir))
    };
    let vlm_asset_path = raw_vlm_path.as_deref().map(|p| {
        resolve_scene_or_external_asset(&mut app, &cli.scene_dir, p, "vlm_external_src")
    });
    let vlm_active = vlm_asset_path.is_some();
    let vlm_config = VlmConfig {
        asset_path: vlm_asset_path,
        enabled: vlm_active,
        intensity: cli.vlm_intensity,
        ..default()
    };
    let raw_skybox_path = if cli.no_skybox {
        None
    } else {
        cli.skybox
            .clone()
            .or_else(|| manifest_hdri.as_ref().map(|h| h.image.clone()))
    };
    let skybox_asset = raw_skybox_path.as_deref().map(|p| {
        resolve_scene_or_external_asset(&mut app, &cli.scene_dir, p, "skybox_hdri_src")
    });
    let env_diffuse_asset = cli.env_diffuse.as_deref().map(|p| {
        resolve_scene_or_external_asset(&mut app, &cli.scene_dir, p, "env_diffuse_src")
    });
    let env_specular_asset = cli.env_specular.as_deref().map(|p| {
        resolve_scene_or_external_asset(&mut app, &cli.scene_dir, p, "env_specular_src")
    });
    let manifest_strength = manifest_hdri.as_ref().map_or(1.0, |h| h.strength);
    let skybox_brightness = cli
        .skybox_brightness
        .unwrap_or(1500.0 * manifest_strength);
    let env_intensity = cli.env_intensity.unwrap_or(
        if skybox_asset.is_some() || (env_diffuse_asset.is_some() && env_specular_asset.is_some())
        {
            250.0 * manifest_strength
        } else {
            150.0
        },
    );
    let yaw_degrees = cli
        .skybox_yaw
        .unwrap_or_else(|| manifest_hdri.as_ref().map_or(0.0, |h| h.yaw_degrees));
    let skybox_config = SkyboxConfig {
        skybox_asset,
        env_diffuse_asset,
        env_specular_asset,
        cubemap_size: cli.skybox_cubemap_size,
        skybox_brightness,
        env_intensity,
        yaw_degrees,
        enabled: !cli.no_skybox,
    };

    let shadows_enabled = cli.shadow_size > 0;
    let shadow_map_size = if shadows_enabled {
        cli.shadow_size.next_power_of_two().max(256)
    } else {
        256
    };

    // 将默认贴图寻址模式设为 Repeat + 各向异性过滤，作为平铺 UV（Sponza/Bistro 墙砖/地面）的全局兜底
    let mut default_sampler = ImageSamplerDescriptor::linear();
    default_sampler.set_address_mode(ImageAddressMode::Repeat);
    default_sampler.anisotropy_clamp = 4;

    let render_scale = cli.render_scale.clamp(0.25, 1.0);
    let initial_output_size = UVec2::new(
        ((cli.width as f32) * cli.scale_factor).round().max(1.0) as u32,
        ((cli.height as f32) * cli.scale_factor).round().max(1.0) as u32,
    );
    let initial_internal_size = compute_scaled_size(initial_output_size, render_scale);

    app.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 0.02,
        ..default()
    })
    .insert_resource(DirectionalLightShadowMap {
        size: shadow_map_size,
    })
    .insert_resource(RenderOptions {
        enable_bloom: cli.bloom,
    })
    .insert_resource(RenderResolutionConfig {
        scale: render_scale,
        internal_size: initial_internal_size,
        output_size: initial_output_size,
    })
    .insert_resource(MeshletSceneConfig {
        scene_dir: cli.scene_dir,
        manifest_file: cli.manifest,
        asset_source: "meshlet_scene".to_string(),
        max_lights: cli.max_lights,
        shadows_enabled,
        shadow_map_size,
        shadow_cascades: cli.shadow_cascades.clamp(1, 4),
        first_cascade_distance: cli.first_cascade_distance.max(2.0),
        shadow_distance: cli.shadow_distance,
        shadow_normal_bias: cli.shadow_normal_bias,
        shadow_depth_bias: cli.shadow_depth_bias,
        foliage_shadows: cli.foliage_shadows,
        light_intensity_scale: cli.light_intensity_scale,
        default_sun_illuminance: cli.default_sun_illuminance,
        default_point_lumens: cli.default_point_lumens,
        vlm_active,
        static_realtime_shadows: cli.static_realtime_shadows,
    })
    .insert_resource(IrradianceConfig {
        ktx2_asset_path,
        meta_asset_path,
        intensity: cli.irradiance_intensity,
    })
    .insert_resource(vlm_config)
    .insert_resource(skybox_config)
    .insert_resource(StreamingConfig {
        enabled: !cli.no_streaming,
        load_distance: cli.load_distance,
        unload_distance: cli.unload_distance.max(cli.load_distance + 8.0),
        max_spawns_per_frame: cli.max_spawns_per_frame,
        ..default()
    })
        .add_plugins((
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: bevy_assets_dir,
                    ..default()
                })
                .set(ImagePlugin::default().with_default_sampler(default_sampler))
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Action Renderer (Meshlet + Irradiance Volume + Hybrid)".into(),
                        resolution: WindowResolution::new(cli.width, cli.height)
                            .with_scale_factor_override(cli.scale_factor),
                        present_mode: if cli.no_vsync {
                            PresentMode::AutoNoVsync
                        } else {
                            PresentMode::AutoVsync
                        },
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
                // 提升至 104 万槽位 (~4MB VRAM)，防止高面数场景+阴影视图同时光栅化时 Cluster 溢出闪烁
                cluster_buffer_slots: 1 << 20,
            },
            FreeCameraPlugin,
            FrameTimeDiagnosticsPlugin::default(),
            LogDiagnosticsPlugin::default(),
            MeshletScenePlugin,
            Mesh3dLodPlugin,
            IrradianceScenePlugin,
            VlmScenePlugin,
            SkyboxScenePlugin,
            StreamingPlugin,
            CharacterActionStubPlugin,
            HudPlugin,
        ));

    if cli.gpu_profile {
        app.add_plugins(RenderDiagnosticsPlugin);
    }

    app.add_systems(Startup, setup_camera)
        .add_systems(Update, sync_scaled_render_target_on_resize)
        .run();
}

fn compute_scaled_size(output_size: UVec2, scale: f32) -> UVec2 {
    let clamped_out = output_size.max(UVec2::ONE);
    if scale >= 0.999 {
        return clamped_out;
    }
    let w = ((clamped_out.x as f32) * scale).round().max(64.0) as u32;
    let h = ((clamped_out.y as f32) * scale).round().max(64.0) as u32;
    UVec2::new(w.min(clamped_out.x.max(64)), h.min(clamped_out.y.max(64)))
}

fn setup_camera(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    render_opts: Res<RenderOptions>,
    mut resolution_cfg: ResMut<RenderResolutionConfig>,
    skybox_config: Res<SkyboxConfig>,
    mut images: ResMut<Assets<Image>>,
    primary_window: Option<Single<&Window, With<PrimaryWindow>>>,
) {
    if let Some(window) = primary_window {
        let phys = UVec2::new(window.physical_width(), window.physical_height());
        if phys.x > 0 && phys.y > 0 {
            resolution_cfg.output_size = phys;
            resolution_cfg.internal_size = compute_scaled_size(phys, resolution_cfg.scale);
        }
    }

    let mut cam = commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(4.5, 2.2, 5.5).looking_at(Vec3::new(0.0, 0.6, 0.8), Vec3::Y),
        // MeshletPlugin 强制要求关闭 MSAA；保留 FXAA 避免触发 Bevy 主干 meshlet_prepass 的 bind_group 槽位 bug
        Msaa::Off,
        Fxaa::default(),
        AoDebugView::default(),
        EnvironmentMapLight {
            diffuse_map: asset_server.load("environment_maps/pisa_diffuse_rgb9e5_zstd.ktx2"),
            specular_map: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
            intensity: skybox_config.env_intensity,
            ..default()
        },
        FreeCamera::default(),
    ));
    if render_opts.enable_bloom {
        cam.insert(Bloom::NATURAL);
    }

    if resolution_cfg.scale < 0.999 {
        let internal = resolution_cfg.internal_size;
        let mut target_image = Image::new_target_texture(
            internal.x,
            internal.y,
            TextureFormat::Rgba8Unorm,
            Some(TextureFormat::Rgba8UnormSrgb),
        );
        // 显式使用 ClampToEdge + 双线性过滤，避免全局默认 Repeat 寻址在全屏放大边缘采到对面像素
        target_image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor::linear());
        let target_handle = images.add(target_image);

        cam.insert((
            Camera {
                order: -1,
                ..default()
            },
            RenderTarget::Image(target_handle.clone().into()),
        ));

        commands.insert_resource(ScaledRenderTarget {
            image: target_handle.clone(),
        });

        // 原生分辨率 UI 合成相机：先画全屏缩放后的 3D 画面，再由上层 HudRoot (GlobalZIndex 0) 渲染原生清晰度文字
        commands.spawn((
            Camera2d,
            Camera {
                order: 1,
                clear_color: ClearColorConfig::Custom(Color::BLACK),
                ..default()
            },
            Msaa::Off,
            IsDefaultUiCamera,
        ));

        commands.spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                top: Val::Px(0.0),
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                ..default()
            },
            ImageNode::new(target_handle),
            GlobalZIndex(-100),
        ));

        info!(
            "启用 3D 内部分辨率缩放: scale={:.2}x, internal={}x{}, output={}x{}",
            resolution_cfg.scale,
            resolution_cfg.internal_size.x,
            resolution_cfg.internal_size.y,
            resolution_cfg.output_size.x,
            resolution_cfg.output_size.y,
        );
    }
}

fn sync_scaled_render_target_on_resize(
    mut resize_events: MessageReader<WindowResized>,
    primary_window: Option<Single<&Window, With<PrimaryWindow>>>,
    mut resolution_cfg: ResMut<RenderResolutionConfig>,
    scaled_target: Option<Res<ScaledRenderTarget>>,
    mut images: ResMut<Assets<Image>>,
) {
    let Some(window) = primary_window else {
        resize_events.clear();
        return;
    };
    let mut has_primary_resize = false;
    for _ev in resize_events.read() {
        has_primary_resize = true;
    }
    if !has_primary_resize {
        return;
    }

    let phys = UVec2::new(window.physical_width(), window.physical_height());
    if phys.x == 0 || phys.y == 0 || phys == resolution_cfg.output_size {
        return;
    }

    let new_internal = compute_scaled_size(phys, resolution_cfg.scale);
    resolution_cfg.output_size = phys;
    resolution_cfg.internal_size = new_internal;

    if let Some(target) = scaled_target
        && let Some(mut image) = images.get_mut(&target.image)
    {
        image.resize(Extent3d {
            width: new_internal.x,
            height: new_internal.y,
            depth_or_array_layers: 1,
        });
    }
}
