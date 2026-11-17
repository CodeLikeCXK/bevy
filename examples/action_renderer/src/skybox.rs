//! 天空盒（Skybox）与 HDRI 环境光照（IBL）模块
//!
//! 同时支持两条输入路径：
//! 1. **预烘培 Cubemap（推荐运行时/低算力平台）**：直接加载 6 面 `.ktx2`（或 6 层纵向拼接图）作为 `Skybox`，
//!    配合可选的 `--env-diffuse` / `--env-specular` 直接驱动 `EnvironmentMapLight`（零运行时卷积开销）。
//! 2. **2:1 等距柱状 HDRI 全景图（`.hdr` / `.exr` / `.png`）**：加载完成后在 `ComputeTaskPool` 并行展开为
//!    6 面 `Rgba16Float` Cubemap，自动挂载 `Skybox` 与 `GeneratedEnvironmentMapLight` 实时生成 IBL。
//!
//! 快捷键：按 `B` 切换天空盒背景显示/隐藏（保留环境光反射）。

use std::{
    f32::consts::{PI, TAU},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use bevy::{
    asset::RenderAssetUsages,
    image::TextureFormatPixelInfo,
    light::{EnvironmentMapLight, GeneratedEnvironmentMapLight, Skybox},
    pbr::generate::{filtering_system, GeneratorBindGroups, RenderEnvironmentMap},
    prelude::*,
    render::{
        render_resource::{
            Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
        },
        Render, RenderApp, RenderSystems,
    },
    tasks::ComputeTaskPool,
};

#[derive(Resource, Clone)]
pub struct SkyboxConfig {
    /// 天空盒或 HDRI 贴图 Asset URI（支持 Cubemap .ktx2 / 6层图 / 2:1 Equirectangular .hdr/.exr）
    pub skybox_asset: Option<String>,
    /// 可选预烘培 Diffuse Cubemap (.ktx2)
    pub env_diffuse_asset: Option<String>,
    /// 可选预烘培 Specular Cubemap (.ktx2)
    pub env_specular_asset: Option<String>,
    /// 2:1 全景图展开为 Cubemap 时的单面边长（需为 2 的幂，默认 512）
    pub cubemap_size: u32,
    /// 天空盒背景亮度 (cd/m^2)
    pub skybox_brightness: f32,
    /// 环境光反射/漫反射强度 (cd/m^2)
    pub env_intensity: f32,
    /// 绕世界 Y 轴旋转角度（度）
    pub yaw_degrees: f32,
    /// 是否显示天空盒背景
    pub enabled: bool,
}

impl Default for SkyboxConfig {
    fn default() -> Self {
        Self {
            skybox_asset: None,
            env_diffuse_asset: None,
            env_specular_asset: None,
            cubemap_size: 512,
            skybox_brightness: 1500.0,
            env_intensity: 150.0,
            yaw_degrees: 0.0,
            enabled: true,
        }
    }
}

/// 供 HUD 展示当前天空盒与 HDRI 状态
#[derive(Resource, Clone)]
pub struct SkyboxStatus {
    pub visible: bool,
    pub mode_label: String,
}

impl Default for SkyboxStatus {
    fn default() -> Self {
        Self {
            visible: false,
            mode_label: "Fallback EnvMap".to_string(),
        }
    }
}

#[derive(Resource, Clone)]
struct IblFilterCompletionFlag(Arc<AtomicBool>);

#[derive(Resource, Default)]
struct SkyboxRuntime {
    source_image: Option<Handle<Image>>,
    diffuse_image: Option<Handle<Image>>,
    specular_image: Option<Handle<Image>>,
    ready_cubemap: Option<Handle<Image>>,
    ibl_owned_cubemap: Option<Handle<Image>>,
    source_processed: bool,
    prefiltered_applied: bool,
    pending_ibl_finalize: bool,
    ibl_ready_frames: u32,
}

pub struct SkyboxScenePlugin;

impl Plugin for SkyboxScenePlugin {
    fn build(&self, app: &mut App) {
        let completion_flag = IblFilterCompletionFlag(Arc::new(AtomicBool::new(false)));

        app.insert_resource(completion_flag.clone())
            .init_resource::<SkyboxStatus>()
            .init_resource::<SkyboxRuntime>()
            .add_systems(Startup, load_skybox_assets)
            .add_systems(
                Update,
                (
                    process_skybox_and_hdri,
                    finalize_generated_ibl,
                    toggle_skybox_visibility,
                    adjust_env_intensity,
                )
                    .chain(),
            );

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.insert_resource(completion_flag).add_systems(
                Render,
                mark_generated_ibl_ready
                    .after(filtering_system)
                    .before(RenderSystems::Render),
            );
        }
    }
}

fn mark_generated_ibl_ready(
    flag: Res<IblFilterCompletionFlag>,
    query: Query<(), (With<GeneratorBindGroups>, With<RenderEnvironmentMap>)>,
) {
    if !query.is_empty() {
        flag.0.store(true, Ordering::Release);
    }
}

fn load_skybox_assets(
    asset_server: Res<AssetServer>,
    config: Res<SkyboxConfig>,
    mut runtime: ResMut<SkyboxRuntime>,
    mut status: ResMut<SkyboxStatus>,
) {
    runtime.source_image = config.skybox_asset.as_ref().map(|p| asset_server.load(p));
    runtime.diffuse_image = config
        .env_diffuse_asset
        .as_ref()
        .map(|p| asset_server.load(p));
    runtime.specular_image = config
        .env_specular_asset
        .as_ref()
        .map(|p| asset_server.load(p));

    if config.skybox_asset.is_some() {
        status.visible = config.enabled;
        status.mode_label = "Loading...".to_string();
    } else if runtime.diffuse_image.is_some() && runtime.specular_image.is_some() {
        status.visible = false;
        status.mode_label = "Custom KTX2 IBL".to_string();
    } else {
        status.visible = false;
        status.mode_label = "Fallback EnvMap".to_string();
    }
}

fn process_skybox_and_hdri(
    mut commands: Commands,
    config: Res<SkyboxConfig>,
    mut runtime: ResMut<SkyboxRuntime>,
    mut status: ResMut<SkyboxStatus>,
    completion_flag: Res<IblFilterCompletionFlag>,
    mut images: ResMut<Assets<Image>>,
    cameras: Query<Entity, With<Camera3d>>,
) {
    if runtime.source_processed {
        return;
    }

    let Ok(camera_entity) = cameras.single() else {
        return;
    };

    let rotation = Quat::from_rotation_y(config.yaw_degrees.to_radians());
    let has_prefiltered_pair =
        runtime.diffuse_image.is_some() && runtime.specular_image.is_some();

    // 1. 若只传了预烘培 diffuse/specular KTX2（未传 skybox 贴图），直接替换相机 EnvironmentMapLight
    if runtime.source_image.is_none() {
        if has_prefiltered_pair && !runtime.prefiltered_applied {
            commands.entity(camera_entity).insert(EnvironmentMapLight {
                diffuse_map: runtime.diffuse_image.clone().unwrap(),
                specular_map: runtime.specular_image.clone().unwrap(),
                intensity: config.env_intensity,
                rotation,
                affects_lightmapped_mesh_diffuse: true,
            });
            runtime.prefiltered_applied = true;
            runtime.source_processed = true;
            info!(
                "已应用预烘培 HDRI EnvironmentMapLight (intensity={:.1}, yaw={:.1}°)",
                config.env_intensity, config.yaw_degrees
            );
        }
        return;
    }

    let Some(src_handle) = runtime.source_image.clone() else {
        return;
    };
    let Some(src_image) = images.get(&src_handle) else {
        return;
    };

    let width = src_image.width();
    let height = src_image.height();
    let layers = src_image.texture_descriptor.array_layer_count();

    let mut ibl_input_handle: Option<Handle<Image>> = None;
    let (cubemap_handle, mode_desc) = if layers == 6 {
        // 路径 1a：标准 6 面 Cubemap（如 .ktx2）
        if src_image
            .texture_view_descriptor
            .as_ref()
            .and_then(|v| v.dimension)
            != Some(TextureViewDimension::Cube)
            && let Some(mut img_mut) = images.get_mut(&src_handle)
        {
            img_mut.texture_view_descriptor = Some(TextureViewDescriptor {
                dimension: Some(TextureViewDimension::Cube),
                ..default()
            });
        }
        (src_handle.clone(), format!("Cubemap {}px", width))
    } else if layers == 1 && width.is_power_of_two() && height == width * 6 {
        // 路径 1b：纵向 6 连拼 2D Cubemap（如 PNG/HDR 纵向 6 面图）
        if let Some(mut img_mut) = images.get_mut(&src_handle) {
            if let Err(err) = img_mut.reinterpret_stacked_2d_as_array(6) {
                warn!("纵向 6 面图转 Cubemap 失败: {err}");
                runtime.source_processed = true;
                return;
            }
            img_mut.texture_view_descriptor = Some(TextureViewDescriptor {
                dimension: Some(TextureViewDimension::Cube),
                ..default()
            });
        }
        (src_handle.clone(), format!("Stacked Cube {}px", width))
    } else {
        // 路径 2：2:1 等距柱状全景 HDRI (.hdr / .png) -> 展开为 6 面 Rgba16Float Cubemap
        let target_size = config
            .cubemap_size
            .clamp(64, 2048)
            .next_power_of_two()
            .min(2048);
        let t0 = std::time::Instant::now();
        let Some(cubemap_img) = equirectangular_to_cubemap(src_image, target_size) else {
            warn!(
                "无法将环境贴图 ({}x{}, {:?}) 转换为 Cubemap，保留默认环境光",
                width, height, src_image.texture_descriptor.format
            );
            runtime.source_processed = true;
            runtime.source_image = None;
            images.remove(&src_handle);
            status.mode_label = "Convert Failed".to_string();
            return;
        };

        // 为 IBL 生成独立的 256px 轻量源图，降低首次 GPU 卷积与 PBR specular_map 采样带宽
        let ibl_size = target_size.min(256);
        let ibl_img = if !has_prefiltered_pair && ibl_size < target_size {
            equirectangular_to_cubemap(src_image, ibl_size)
        } else {
            None
        };

        info!(
            "已将等距柱状 HDRI ({}x{}, {:?}) 展开为 {}px×6 Skybox + {}px×6 IBL，耗时 {:.1}ms（释放原图）",
            width,
            height,
            src_image.texture_descriptor.format,
            target_size,
            if ibl_img.is_some() { ibl_size } else { target_size },
            t0.elapsed().as_secs_f64() * 1000.0
        );

        let sky_handle = images.add(cubemap_img);
        if let Some(ibl_cube) = ibl_img {
            let h = images.add(ibl_cube);
            runtime.ibl_owned_cubemap = Some(h.clone());
            ibl_input_handle = Some(h);
        }
        // 立即释放 4K 原始 Rgba32Float 源图（约 128MB），避免常驻内存与冗余显存上传
        runtime.source_image = None;
        images.remove(&src_handle);
        (sky_handle, format!("HDRI->Cube {}px", target_size))
    };

    runtime.ready_cubemap = Some(cubemap_handle.clone());
    runtime.source_processed = true;

    // 挂载或保持 Skybox
    if config.enabled {
        commands.entity(camera_entity).insert(Skybox {
            image: Some(cubemap_handle.clone()),
            brightness: config.skybox_brightness,
            rotation,
        });
    } else {
        commands.entity(camera_entity).remove::<Skybox>();
    }

    // 挂载 IBL 环境光：若提供了预烘培 diffuse/specular 则直用（零卷积）；否则由 GPU 一次性生成并在完成后卸载
    if has_prefiltered_pair {
        commands.entity(camera_entity).insert(EnvironmentMapLight {
            diffuse_map: runtime.diffuse_image.clone().unwrap(),
            specular_map: runtime.specular_image.clone().unwrap(),
            intensity: config.env_intensity,
            rotation,
            affects_lightmapped_mesh_diffuse: true,
        });
        status.mode_label = format!("{mode_desc} + KTX2 IBL");
    } else {
        let gen_input = ibl_input_handle.unwrap_or(cubemap_handle);
        let cube_w = images
            .get(&gen_input)
            .map(|i| i.width())
            .unwrap_or(config.cubemap_size);
        let cube_h = images
            .get(&gen_input)
            .map(|i| i.height())
            .unwrap_or(config.cubemap_size);
        if cube_w == cube_h && cube_w.is_power_of_two() && cube_w <= 8192 {
            completion_flag.0.store(false, Ordering::Release);
            runtime.pending_ibl_finalize = true;
            runtime.ibl_ready_frames = 0;
            // 必须先移除初始占位的 EnvironmentMapLight，GeneratedEnvironmentMapLight 才会触发 GPU 卷积
            commands
                .entity(camera_entity)
                .remove::<EnvironmentMapLight>()
                .insert(GeneratedEnvironmentMapLight {
                    environment_map: gen_input,
                    intensity: config.env_intensity,
                    rotation,
                    affects_lightmapped_mesh_diffuse: true,
                });
            status.mode_label = format!("{mode_desc} + GenIBL...");
        } else {
            warn!(
                "Cubemap 尺寸 ({}x{}) 非 2 的幂正方形，跳过 GeneratedEnvironmentMapLight，沿用默认 IBL",
                cube_w, cube_h
            );
            status.mode_label = format!("{mode_desc} + FallbackIBL");
        }
    }

    status.visible = config.enabled;
}

/// 等待 RenderWorld 至少完成一帧 GPU GGX/Irradiance 卷积后，立即移除 `GeneratedEnvironmentMapLight`，
/// 保留已生成的 `EnvironmentMapLight` 贴图，避免 Bevy 每帧对静态 HDRI 重复执行 ~76M 次采样计算。
fn finalize_generated_ibl(
    mut commands: Commands,
    mut runtime: ResMut<SkyboxRuntime>,
    mut status: ResMut<SkyboxStatus>,
    completion_flag: Res<IblFilterCompletionFlag>,
    mut images: ResMut<Assets<Image>>,
    cameras: Query<Entity, (With<Camera3d>, With<EnvironmentMapLight>, With<GeneratedEnvironmentMapLight>)>,
) {
    if !runtime.pending_ibl_finalize {
        return;
    }
    if !completion_flag.0.load(Ordering::Acquire) {
        return;
    }
    let Ok(camera_entity) = cameras.single() else {
        return;
    };

    runtime.ibl_ready_frames += 1;
    if runtime.ibl_ready_frames < 2 {
        return;
    }

    commands
        .entity(camera_entity)
        .remove::<GeneratedEnvironmentMapLight>();
    if let Some(ibl_handle) = runtime.ibl_owned_cubemap.take() {
        images.remove(&ibl_handle);
    }
    runtime.pending_ibl_finalize = false;
    status.mode_label = status.mode_label.replace("GenIBL...", "BakedIBL");
    info!("HDRI GPU 一次性卷积完成，已卸载 GeneratedEnvironmentMapLight（后续帧零卷积开销）");
}

fn toggle_skybox_visibility(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    mut config: ResMut<SkyboxConfig>,
    mut status: ResMut<SkyboxStatus>,
    runtime: Res<SkyboxRuntime>,
    cameras: Query<Entity, With<Camera3d>>,
) {
    if !keys.just_pressed(KeyCode::KeyB) {
        return;
    }
    let Some(cubemap_handle) = runtime.ready_cubemap.clone() else {
        return;
    };
    let Ok(camera_entity) = cameras.single() else {
        return;
    };

    config.enabled = !config.enabled;
    status.visible = config.enabled;
    if config.enabled {
        commands.entity(camera_entity).insert(Skybox {
            image: Some(cubemap_handle),
            brightness: config.skybox_brightness,
            rotation: Quat::from_rotation_y(config.yaw_degrees.to_radians()),
        });
    } else {
        commands.entity(camera_entity).remove::<Skybox>();
    }
}

/// 运行时按 `-` / `=` 实时微调天光环境光强度 (`EnvironmentMapLight.intensity`)。
fn adjust_env_intensity(
    keys: Res<ButtonInput<KeyCode>>,
    mut config: ResMut<SkyboxConfig>,
    mut env_lights: Query<
        (
            Option<&mut EnvironmentMapLight>,
            Option<&mut GeneratedEnvironmentMapLight>,
        ),
        With<Camera3d>,
    >,
) {
    let down = keys.just_pressed(KeyCode::Minus);
    let up = keys.just_pressed(KeyCode::Equal);
    if down == up {
        return;
    }

    let scaled = if up {
        (config.env_intensity / 0.85).max(config.env_intensity + 10.0)
    } else {
        (config.env_intensity * 0.85).min(config.env_intensity - 10.0)
    };
    let step = if scaled >= 500.0 { 25.0 } else { 10.0 };
    config.env_intensity = ((scaled / step).round() * step).clamp(10.0, 5000.0);

    for (env_opt, gen_opt) in &mut env_lights {
        if let Some(mut env) = env_opt {
            env.intensity = config.env_intensity;
        }
        if let Some(mut gen_env) = gen_opt {
            gen_env.intensity = config.env_intensity;
        }
    }
    info!("HDRI 天光强度调整为: {:.0}", config.env_intensity);
}

/// 将 2:1 等距柱状全景贴图并行重采样为 6 面 `Rgba16Float` Cubemap（匹配 Bevy 左手系采样反转 +Z 约定）。
fn equirectangular_to_cubemap(src: &Image, face_size: u32) -> Option<Image> {
    let src_w = src.width();
    let src_h = src.height();
    let data = src.data.as_ref()?;
    if src_w == 0 || src_h == 0 {
        return None;
    }

    let format = src.texture_descriptor.format;
    let is_rgba32f = format == TextureFormat::Rgba32Float
        && data.len() >= (src_w as usize) * (src_h as usize) * 16;

    // 若非 Rgba32Float（例如 Rgba8UnormSrgb / Rgba16Float），预先解包为线性 f32 RGBA 以统一并行双线性采样
    let decoded_rgba: Vec<[f32; 4]> = if is_rgba32f {
        Vec::new()
    } else {
        let mut buf = Vec::with_capacity((src_w as usize) * (src_h as usize));
        for y in 0..src_h {
            for x in 0..src_w {
                let c = src.get_color_at(x, y).ok()?;
                buf.push(LinearRgba::from(c).to_f32_array());
            }
        }
        buf
    };

    let bytes_per_pixel = TextureFormat::Rgba16Float.pixel_size().ok()?;
    let face_bytes = (face_size as usize) * (face_size as usize) * bytes_per_pixel;
    let mut out_bytes = vec![0u8; face_bytes * 6];
    let inv_size = 1.0 / (face_size as f32);

    let sample_rgba = |ix: u32, iy: u32| -> [f32; 4] {
        let idx = (iy as usize) * (src_w as usize) + (ix as usize);
        if is_rgba32f {
            let off = idx * 16;
            [
                f32::from_le_bytes(data[off..off + 4].try_into().unwrap()),
                f32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap()),
                f32::from_le_bytes(data[off + 8..off + 12].try_into().unwrap()),
                f32::from_le_bytes(data[off + 12..off + 16].try_into().unwrap()),
            ]
        } else {
            decoded_rgba[idx]
        }
    };

    let face_slices: Vec<&mut [u8]> = out_bytes.chunks_exact_mut(face_bytes).collect();
    let sample_rgba_ref = &sample_rgba;
    ComputeTaskPool::get().scope(|scope| {
        for (face_idx, face_buf) in face_slices.into_iter().enumerate() {
            scope.spawn(async move {
                for py in 0..face_size {
                    let v = 2.0 * ((py as f32 + 0.5) * inv_size) - 1.0;
                    for px in 0..face_size {
                        let u = 2.0 * ((px as f32 + 0.5) * inv_size) - 1.0;

                        // GPU Cubemap 采样方向；Bevy skybox/environment_map 采样时会做 world_dir * (1, 1, -1)
                        let sample_dir = match face_idx {
                            0 => Vec3::new(1.0, -v, -u),  // +X
                            1 => Vec3::new(-1.0, -v, u),  // -X
                            2 => Vec3::new(u, 1.0, v),    // +Y
                            3 => Vec3::new(u, -1.0, -v),  // -Y
                            4 => Vec3::new(u, -v, 1.0),   // +Z
                            _ => Vec3::new(-u, -v, -1.0), // -Z
                        };
                        let world_dir = Vec3::new(sample_dir.x, sample_dir.y, -sample_dir.z)
                            .normalize_or_zero();

                        // 与 Blender ShaderNodeTexEnvironment (EQUIRECTANGULAR) 对齐的球面映射
                        let u_eq = (world_dir.z.atan2(world_dir.x) / TAU + 0.5).rem_euclid(1.0);
                        let v_eq = (world_dir.y.clamp(-1.0, 1.0).acos() / PI).clamp(0.0, 1.0);

                        let fx = u_eq * (src_w as f32) - 0.5;
                        let fy = v_eq * (src_h as f32 - 1.0);

                        let x0 = fx.floor() as i32;
                        let y0 = fy.floor() as i32;
                        let tx = fx - (x0 as f32);
                        let ty = fy - (y0 as f32);

                        let ix0 = x0.rem_euclid(src_w as i32) as u32;
                        let ix1 = (x0 + 1).rem_euclid(src_w as i32) as u32;
                        let iy0 = y0.clamp(0, src_h as i32 - 1) as u32;
                        let iy1 = (y0 + 1).clamp(0, src_h as i32 - 1) as u32;

                        let c00 = sample_rgba_ref(ix0, iy0);
                        let c10 = sample_rgba_ref(ix1, iy0);
                        let c01 = sample_rgba_ref(ix0, iy1);
                        let c11 = sample_rgba_ref(ix1, iy1);

                        let mut rgba = [0.0f32; 4];
                        for ch in 0..4 {
                            let top = c00[ch] * (1.0 - tx) + c10[ch] * tx;
                            let bot = c01[ch] * (1.0 - tx) + c11[ch] * tx;
                            rgba[ch] = top * (1.0 - ty) + bot * ty;
                        }

                        let out_off = ((py as usize) * (face_size as usize) + (px as usize)) * 8;
                        face_buf[out_off..out_off + 2]
                            .copy_from_slice(&f32_to_f16_bits(rgba[0].max(0.0)).to_le_bytes());
                        face_buf[out_off + 2..out_off + 4]
                            .copy_from_slice(&f32_to_f16_bits(rgba[1].max(0.0)).to_le_bytes());
                        face_buf[out_off + 4..out_off + 6]
                            .copy_from_slice(&f32_to_f16_bits(rgba[2].max(0.0)).to_le_bytes());
                        face_buf[out_off + 6..out_off + 8]
                            .copy_from_slice(&f32_to_f16_bits(1.0).to_le_bytes());
                    }
                }
            });
        }
    });

    let mut cube = Image::new(
        Extent3d {
            width: face_size,
            height: face_size,
            depth_or_array_layers: 6,
        },
        TextureDimension::D2,
        out_bytes,
        TextureFormat::Rgba16Float,
        RenderAssetUsages::all(),
    );
    cube.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });
    Some(cube)
}

/// 纯 Rust IEEE-754 float32 -> float16 位转换（钳位至有限半精度最大值 65504，防止 NaN/Inf 污染 IBL）。
#[inline]
fn f32_to_f16_bits(val: f32) -> u16 {
    let clamped = if val.is_nan() {
        0.0
    } else {
        val.clamp(-65504.0, 65504.0)
    };
    let bits = clamped.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x007f_ffff;

    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (mantissa | 0x0080_0000) >> (1 - exp);
        return sign | ((m >> 13) as u16);
    }
    if exp >= 31 {
        return sign | 0x7bff;
    }
    sign | ((exp as u16) << 10) | ((mantissa >> 13) as u16)
}
