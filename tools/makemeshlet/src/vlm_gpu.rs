//! 离线 Virtual Lightmap 的 headless `wgpu` Compute Shader GPU 加速后端
//!
//! 仅将计算密集的 Pass 1（直接光 + 太阳软阴影）与 Pass 2（多跳间接反弹路径追踪）卸载到 GPU Compute，
//! BVH 构建、表面体素化、3×3 双边滤波、10×10 硬件边框打包与 LZ4 压缩保持在 CPU 端执行，
//! 保证与 CPU 后端输出格式 100% 一致；若环境无硬件 GPU 或超出显存/Buffer 限制，则优雅返回 `Err` 自动回退到 CPU 后端。

use std::sync::mpsc;
use std::time::Instant;

use bevy_math::{Vec2, Vec3};
use bytemuck::{Pod, Zeroable};

use crate::vlm_baker::{
    BakedLocalLight, BakedSunLight, SkySampler, TileData, VlmBakeConfig, VlmBvh, f32_to_f16_bits,
    texel_surface_pos_and_normal,
};

const BVH_LEAF_BIT: u32 = 0x8000_0000;
const WORKGROUP_SIZE: u32 = 64;
const DIRECT_CHUNK_TEXELS: u32 = 8192;
const BOUNCE_CHUNK_TEXELS: u32 = 4096;
const BOUNCE_SAMPLE_BATCH: u32 = 16;
/// DX12 单 Storage Buffer (`ByteAddressBuffer`) 真实硬件描述符上限:
/// `D3D12_REQ_BUFFER_RESOURCE_TEXEL_COUNT_2_TO_EXP = 27` -> 2^27 × 4B (`R32_TYPELESS`) = 512 MB
const DX12_SAFE_MAX_STORAGE_BUFFER_BYTES: u64 = 512 * 1024 * 1024;
/// 单个三角形 Storage Buffer 容量上限 (6,291,456 tris × 64B = 384 MB < 512 MB)，超出部分放入 tris_1 双缓冲
const TRIS_PER_BUFFER: usize = 6_291_456;
/// 单次常驻显存的有效纹素窗口上限 (104.8 万纹素 × 80B = 80 MB)
const MAX_ACTIVE_SLICE_TEXELS: usize = 1_048_576;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuBakeParams {
    active_offset: u32,
    active_count: u32,
    sample_start: u32,
    sample_batch_count: u32,

    total_samples: u32,
    bounce_salt: u32,
    sun_shadow_rays: u32,
    local_light_count: u32,

    table_mask: u32,
    search_count: u32,
    level_count: u32,
    sky_width: u32,

    sky_height: u32,
    node_count: u32,
    local_light_clearance: f32,
    local_light_radius: f32,

    sun_dir_and_cos_cone: [f32; 4],
    sun_color_and_illuminance: [f32; 4],

    base_texel_size: f32,
    gi_scale: f32,
    sky_yaw_rad: f32,
    sky_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuBvhNode {
    aabb_min: [f32; 3],
    left_or_first: u32,
    aabb_max: [f32; 3],
    right_or_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTri {
    p0: [f32; 3],
    n0_oct: u32,
    e1: [f32; 3],
    n1_oct: u32,
    e2: [f32; 3],
    n2_oct: u32,
    /// x = oct geom_normal, y = rgba8unorm(albedo, double_sided), z = f16x2(emissive.rg), w = f16x2(emissive.b, 0)
    meta: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLocalLight {
    pos_and_range: [f32; 4],
    forward_and_inv_range_sq: [f32; 4],
    color_and_candela: [f32; 4],
    spot_params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuActiveTexel {
    pos: [f32; 3],
    tile_texel_idx: u32,
    normal: [f32; 3],
    level: u32,
    sun_direct_and_vis: [f32; 4],
    local_direct_and_pad: [f32; 4],
    indirect_and_pad: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<GpuBakeParams>() == 112);
const _: () = assert!(std::mem::size_of::<GpuBvhNode>() == 32);
const _: () = assert!(std::mem::size_of::<GpuTri>() == 64);
const _: () = assert!(std::mem::size_of::<GpuLocalLight>() == 64);
const _: () = assert!(std::mem::size_of::<GpuActiveTexel>() == 80);

#[inline]
fn pack_oct_normal(n_arr: [f32; 3]) -> u32 {
    let n = Vec3::from_array(n_arr).normalize_or(Vec3::Y);
    let sum = n.x.abs() + n.y.abs() + n.z.abs();
    let mut oct = if sum > 1e-8 {
        Vec2::new(n.x / sum, n.y / sum)
    } else {
        Vec2::ZERO
    };
    if n.z < 0.0 {
        let ox = (1.0 - oct.y.abs()) * if oct.x >= 0.0 { 1.0 } else { -1.0 };
        let oy = (1.0 - oct.x.abs()) * if oct.y >= 0.0 { 1.0 } else { -1.0 };
        oct = Vec2::new(ox, oy);
    }
    let ix = (oct.x.clamp(-1.0, 1.0) * 32767.0).round() as i16 as u16 as u32;
    let iy = (oct.y.clamp(-1.0, 1.0) * 32767.0).round() as i16 as u16 as u32;
    ix | (iy << 16)
}

#[inline]
fn pack_unorm8x4(r: f32, g: f32, b: f32, a: f32) -> u32 {
    let c = |v: f32| -> u32 { (v.clamp(0.0, 1.0) * 255.0).round() as u32 };
    c(r) | (c(g) << 8) | (c(b) << 16) | (c(a) << 24)
}

#[inline]
fn pack_f16x2(a: f32, b: f32) -> u32 {
    (f32_to_f16_bits(a) as u32) | ((f32_to_f16_bits(b) as u32) << 16)
}

fn check_buffer_limits(
    label: &str,
    size: u64,
    is_storage: bool,
    limits: &wgpu::Limits,
) -> Result<(), String> {
    if size > limits.max_buffer_size {
        return Err(format!(
            "Buffer `{label}` 大小 ({:.1} MB) 超出 GPU max_buffer_size ({:.1} MB)",
            size as f64 / (1024.0 * 1024.0),
            limits.max_buffer_size as f64 / (1024.0 * 1024.0)
        ));
    }
    let safe_max_storage =
        (limits.max_storage_buffer_binding_size as u64).min(DX12_SAFE_MAX_STORAGE_BUFFER_BYTES);
    if is_storage && size > safe_max_storage {
        return Err(format!(
            "Storage Buffer `{label}` 大小 ({:.1} MB) 超出硬件安全上限 ({:.1} MB)",
            size as f64 / (1024.0 * 1024.0),
            safe_max_storage as f64 / (1024.0 * 1024.0)
        ));
    }
    Ok(())
}

fn create_buffer_with_bytes(
    device: &wgpu::Device,
    label: &str,
    bytes: &[u8],
    usage: wgpu::BufferUsages,
    is_storage: bool,
    limits: &wgpu::Limits,
) -> Result<wgpu::Buffer, String> {
    let size = bytes.len() as u64;
    check_buffer_limits(label, size, is_storage, limits)?;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: true,
    });
    buf.slice(..)
        .get_mapped_range_mut()
        .map_err(|e| format!("映射初始 Buffer `{label}` 失败: {e}"))?
        .copy_from_slice(bytes);
    buf.unmap();
    Ok(buf)
}

fn create_zeroed_buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
    is_storage: bool,
    limits: &wgpu::Limits,
) -> Result<wgpu::Buffer, String> {
    check_buffer_limits(label, size, is_storage, limits)?;
    Ok(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    }))
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn pop_error_scopes(
    validation_scope: wgpu::ErrorScopeGuard,
    oom_scope: wgpu::ErrorScopeGuard,
    stage: &str,
) -> Result<(), String> {
    if let Some(err) = bevy_tasks::block_on(oom_scope.pop()) {
        let _ = bevy_tasks::block_on(validation_scope.pop());
        return Err(format!("GPU 显存不足 ({stage}): {err}"));
    }
    if let Some(err) = bevy_tasks::block_on(validation_scope.pop()) {
        return Err(format!("GPU 验证错误 ({stage}): {err}"));
    }
    Ok(())
}

/// 使用 headless `wgpu` Compute Shader 执行 VLM Pass 1（直接光+太阳软阴影）与 Pass 2（多跳间接反弹）。
/// 成功时就地更新 `tile_list` 并返回 GPU 适配器名称；若无可用硬件 GPU 或运行出错则返回 `Err(原因)` 以回退 CPU。
#[allow(clippy::too_many_arguments)]
pub(crate) fn bake_direct_and_bounces_gpu(
    bvh: &VlmBvh,
    sun: &BakedSunLight,
    local_lights: &[BakedLocalLight],
    sky: &SkySampler,
    tile_table: &[[u32; 2]],
    table_mask: u32,
    search_count: u32,
    tile_list: &mut [TileData],
    effective_texel_size: f32,
    levels: u32,
    sun_shadow_rays: u32,
    cos_sun_cone: f32,
    config: &VlmBakeConfig,
) -> Result<String, String> {
    let t_init = Instant::now();

    // 1. 初始化 headless wgpu 设备（优先独显/高性能核显，排除纯软件模拟适配器）
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::DX12 | wgpu::Backends::VULKAN | wgpu::Backends::METAL,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
        backend_options: Default::default(),
    });

    let adapter = bevy_tasks::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        ..Default::default()
    }))
    .or_else(|_| {
        bevy_tasks::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
    })
    .map_err(|e| format!("未找到可用的 GPU 适配器: {e}"))?;

    let adapter_info = adapter.get_info();
    if adapter_info.device_type == wgpu::DeviceType::Cpu {
        return Err(format!(
            "仅检测到软件渲染适配器 `{}` ({:?})，回退至原生多线程 CPU 烘焙",
            adapter_info.name, adapter_info.backend
        ));
    }

    let adapter_limits = adapter.limits();
    if adapter_limits.max_storage_buffers_per_shader_stage < 8 {
        return Err(format!(
            "GPU `{}` 的 max_storage_buffers_per_shader_stage={} < 8，回退 CPU",
            adapter_info.name, adapter_limits.max_storage_buffers_per_shader_stage
        ));
    }

    let (device, queue) = bevy_tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("makemeshlet_vlm_gpu"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter_limits.clone(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
    }))
    .map_err(|e| format!("无法创建 GPU 设备 `{}`: {e}", adapter_info.name))?;

    let limits = device.limits();
    let adapter_label = format!("{} ({:?})", adapter_info.name, adapter_info.backend);

    // 2. 打包有效纹素、BVH、三角形、灯光与天空盒为 GPU 紧凑布局
    let mut active_texels: Vec<GpuActiveTexel> = Vec::new();
    for (ti, tile) in tile_list.iter().enumerate() {
        let level = tile.coord.level as u32;
        let dir_id = tile.coord.dir_id as u32;
        let texel_size = effective_texel_size * (1u32 << level) as f32;
        for (k, tx) in tile.texels.iter().enumerate() {
            if tx.weight <= 0.0 {
                continue;
            }
            let (p, n) = texel_surface_pos_and_normal(tx, dir_id, texel_size);
            active_texels.push(GpuActiveTexel {
                pos: p.to_array(),
                tile_texel_idx: (ti as u32) * 64 + (k as u32),
                normal: n.to_array(),
                level,
                sun_direct_and_vis: [0.0; 4],
                local_direct_and_pad: [0.0; 4],
                indirect_and_pad: [0.0; 4],
            });
        }
    }

    if active_texels.is_empty() {
        return Ok(adapter_label);
    }

    let gpu_nodes: Vec<GpuBvhNode> = if bvh.nodes.is_empty() {
        vec![GpuBvhNode::zeroed()]
    } else {
        bvh.nodes
            .iter()
            .map(|n| GpuBvhNode {
                aabb_min: n.aabb_min,
                left_or_first: n.left_or_first,
                aabb_max: n.aabb_max,
                right_or_count: if n.tri_count > 0 {
                    (n.tri_count & 0x7FFF_FFFF) | BVH_LEAF_BIT
                } else {
                    n.right_child
                },
            })
            .collect()
    };

    let gpu_tris: Vec<GpuTri> = if bvh.tris.is_empty() {
        vec![GpuTri::zeroed()]
    } else {
        bvh.tris
            .iter()
            .map(|t| GpuTri {
                p0: t.p0,
                n0_oct: pack_oct_normal(t.n0),
                e1: t.e1,
                n1_oct: pack_oct_normal(t.n1),
                e2: t.e2,
                n2_oct: pack_oct_normal(t.n2),
                meta: [
                    pack_oct_normal(t.geom_normal),
                    pack_unorm8x4(
                        t.albedo[0],
                        t.albedo[1],
                        t.albedo[2],
                        if t.double_sided { 1.0 } else { 0.0 },
                    ),
                    pack_f16x2(t.emissive[0], t.emissive[1]),
                    pack_f16x2(t.emissive[2], 0.0),
                ],
            })
            .collect()
    };

    let gpu_lights: Vec<GpuLocalLight> = if local_lights.is_empty() {
        vec![GpuLocalLight::zeroed()]
    } else {
        local_lights
            .iter()
            .map(|l| GpuLocalLight {
                pos_and_range: [l.position.x, l.position.y, l.position.z, l.range],
                forward_and_inv_range_sq: [l.forward.x, l.forward.y, l.forward.z, l.inv_range_sq],
                color_and_candela: [l.color.x, l.color.y, l.color.z, l.candela],
                spot_params: [
                    if l.is_spot { 1.0 } else { 0.0 },
                    l.cos_inner,
                    l.cos_outer,
                    0.0,
                ],
            })
            .collect()
    };

    let gpu_sky: Vec<[f32; 4]> = if sky.pixels.is_empty() {
        vec![[0.0; 4]]
    } else {
        sky.pixels.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect()
    };

    let total_tile_texels = (tile_list.len() as u64) * 64;
    let tile_cache_bytes = (total_tile_texels.max(1)) * 16;

    // 3. 创建 GPU Buffers 与 Compute Pipelines（带错误作用域捕获，绝不 panic）
    let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let oom_scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);

    let params_buf = create_zeroed_buffer(
        &device,
        "vlm_params",
        std::mem::size_of::<GpuBakeParams>() as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        false,
        &limits,
    )?;
    let bvh_buf = create_buffer_with_bytes(
        &device,
        "vlm_bvh_nodes",
        bytemuck::cast_slice(&gpu_nodes),
        wgpu::BufferUsages::STORAGE,
        true,
        &limits,
    )?;
    let tris_0_slice = &gpu_tris[..gpu_tris.len().min(TRIS_PER_BUFFER)];
    let dummy_tri = [GpuTri::zeroed()];
    let tris_1_slice = if gpu_tris.len() > TRIS_PER_BUFFER {
        &gpu_tris[TRIS_PER_BUFFER..]
    } else {
        &dummy_tri
    };
    let tris_0_buf = create_buffer_with_bytes(
        &device,
        "vlm_tris_0",
        bytemuck::cast_slice(tris_0_slice),
        wgpu::BufferUsages::STORAGE,
        true,
        &limits,
    )?;
    let tris_1_buf = create_buffer_with_bytes(
        &device,
        "vlm_tris_1",
        bytemuck::cast_slice(tris_1_slice),
        wgpu::BufferUsages::STORAGE,
        true,
        &limits,
    )?;
    let max_slice_texels = active_texels.len().min(MAX_ACTIVE_SLICE_TEXELS);
    let max_slice_bytes = (max_slice_texels * std::mem::size_of::<GpuActiveTexel>()) as u64;
    let active_buf = create_zeroed_buffer(
        &device,
        "vlm_active_texels",
        max_slice_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        true,
        &limits,
    )?;
    let readback_buf = create_zeroed_buffer(
        &device,
        "vlm_active_readback",
        max_slice_bytes,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        false,
        &limits,
    )?;
    let tile_cache_a = create_zeroed_buffer(
        &device,
        "vlm_tile_cache_a",
        tile_cache_bytes,
        wgpu::BufferUsages::STORAGE,
        true,
        &limits,
    )?;
    let tile_cache_b = create_zeroed_buffer(
        &device,
        "vlm_tile_cache_b",
        tile_cache_bytes,
        wgpu::BufferUsages::STORAGE,
        true,
        &limits,
    )?;
    let lights_buf = create_buffer_with_bytes(
        &device,
        "vlm_lights",
        bytemuck::cast_slice(&gpu_lights),
        wgpu::BufferUsages::STORAGE,
        true,
        &limits,
    )?;
    let sky_buf = create_buffer_with_bytes(
        &device,
        "vlm_sky_pixels",
        bytemuck::cast_slice(&gpu_sky),
        wgpu::BufferUsages::STORAGE,
        true,
        &limits,
    )?;
    let table_buf = create_buffer_with_bytes(
        &device,
        "vlm_tile_table",
        bytemuck::cast_slice(tile_table),
        wgpu::BufferUsages::STORAGE,
        true,
        &limits,
    )?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vlm_bake_wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("vlm_bake.wgsl").into()),
    });

    let uniform_entry = wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };

    // Pass 1 layout: bindings 0..=5 + 9 (6 storage buffers)
    let direct_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("vlm_direct_bgl"),
        entries: &[
            uniform_entry,
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            storage_entry(4, false),
            storage_entry(5, true),
            storage_entry(9, true),
        ],
    });
    let direct_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("vlm_direct_pl"),
        bind_group_layouts: &[Some(&direct_bgl)],
        immediate_size: 0,
    });
    let direct_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("vlm_direct_pipeline"),
        layout: Some(&direct_pl),
        module: &shader,
        entry_point: Some("bake_direct_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // Pass 2 layout: bindings 0..=4 + 6..=9 (8 storage buffers)
    let bounce_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("vlm_bounce_bgl"),
        entries: &[
            uniform_entry,
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            storage_entry(4, false),
            storage_entry(6, true),
            storage_entry(7, true),
            storage_entry(8, true),
            storage_entry(9, true),
        ],
    });
    let bounce_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("vlm_bounce_pl"),
        bind_group_layouts: &[Some(&bounce_bgl)],
        immediate_size: 0,
    });
    let bounce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("vlm_bounce_pipeline"),
        layout: Some(&bounce_pl),
        module: &shader,
        entry_point: Some("bake_bounce_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let direct_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vlm_direct_bg"),
        layout: &direct_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bvh_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: tris_0_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: active_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: tile_cache_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: lights_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: tris_1_buf.as_entire_binding(),
            },
        ],
    });

    let make_bounce_bg = |label: &str, read_buf: &wgpu::Buffer, write_buf: &wgpu::Buffer| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &bounce_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bvh_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tris_0_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: active_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: write_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: sky_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: read_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: tris_1_buf.as_entire_binding(),
                },
            ],
        })
    };
    let bounce_bg_a_to_b = make_bounce_bg("vlm_bounce_a_to_b", &tile_cache_a, &tile_cache_b);
    let bounce_bg_b_to_a = make_bounce_bg("vlm_bounce_b_to_a", &tile_cache_b, &tile_cache_a);

    pop_error_scopes(validation_scope, oom_scope, "资源与管线初始化")?;

    let vram_mb = (std::mem::size_of_val(gpu_nodes.as_slice())
        + std::mem::size_of_val(gpu_tris.as_slice())
        + (max_slice_bytes as usize)
        + (tile_cache_bytes as usize) * 2
        + std::mem::size_of_val(tile_table)) as f64
        / (1024.0 * 1024.0);
    println!(
        "  [VLM-GPU] 已启用硬件加速: {} | 显存占用 {:.1} MB ({} 有效纹素, {:.2}s)",
        adapter_label,
        vram_mb,
        active_texels.len(),
        t_init.elapsed().as_secs_f64()
    );

    let total_samples = config.samples.max(4);
    let mut base_params = GpuBakeParams {
        active_offset: 0,
        active_count: 0,
        sample_start: 0,
        sample_batch_count: 0,
        total_samples,
        bounce_salt: 0,
        sun_shadow_rays,
        local_light_count: local_lights.len() as u32,
        table_mask,
        search_count,
        level_count: levels,
        sky_width: sky.width,
        sky_height: sky.height,
        node_count: bvh.nodes.len() as u32,
        local_light_clearance: config.local_light_clearance.max(0.05),
        local_light_radius: config.local_light_radius.max(0.0),
        sun_dir_and_cos_cone: [
            sun.dir_to_light.x,
            sun.dir_to_light.y,
            sun.dir_to_light.z,
            cos_sun_cone,
        ],
        sun_color_and_illuminance: [sun.color.x, sun.color.y, sun.color.z, sun.illuminance],
        base_texel_size: effective_texel_size,
        gi_scale: config.gi_scale.max(0.0),
        sky_yaw_rad: sky.yaw_rad,
        sky_scale: sky.scale,
    };

    let readback_slice_into = |slice_dst: &mut [GpuActiveTexel]| -> Result<(), String> {
        let byte_len = std::mem::size_of_val(slice_dst) as u64;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vlm_readback_enc"),
        });
        encoder.copy_buffer_to_buffer(&active_buf, 0, &readback_buf, 0, byte_len);
        queue.submit(Some(encoder.finish()));

        let buf_slice = readback_buf.slice(0..byte_len);
        let (tx, rx) = mpsc::channel();
        buf_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("GPU 回读等待失败: {e:?}"))?;
        rx.recv()
            .map_err(|e| format!("GPU 回读通道关闭: {e}"))?
            .map_err(|e| format!("GPU 回读映射失败: {e}"))?;
        {
            let mapped = buf_slice
                .get_mapped_range()
                .map_err(|e| format!("获取 GPU 回读视图失败: {e}"))?;
            let results: &[GpuActiveTexel] = bytemuck::cast_slice(&mapped);
            slice_dst.copy_from_slice(results);
        }
        readback_buf.unmap();
        Ok(())
    };

    let multi_slice = active_texels.len() > MAX_ACTIVE_SLICE_TEXELS;

    // 4. Pass 1: 直接光与太阳软阴影（按窗口切片 + 8192 有效纹素分批提交，防 128MB 越界与 2s TDR）
    let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let oom_scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);

    let t_direct = Instant::now();
    for tex_slice in active_texels.chunks_mut(MAX_ACTIVE_SLICE_TEXELS) {
        queue.write_buffer(&active_buf, 0, bytemuck::cast_slice(tex_slice));
        let slice_len = tex_slice.len() as u32;
        let mut offset = 0u32;
        while offset < slice_len {
            let count = (slice_len - offset).min(DIRECT_CHUNK_TEXELS);
            base_params.active_offset = offset;
            base_params.active_count = count;
            queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&base_params));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vlm_direct_enc"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("vlm_direct_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&direct_pipeline);
                pass.set_bind_group(0, &direct_bg, &[]);
                pass.dispatch_workgroups(count.div_ceil(WORKGROUP_SIZE), 1, 1);
            }
            queue.submit(Some(encoder.finish()));
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| format!("GPU Pass 1 同步失败: {e:?}"))?;
            offset += count;
        }
        if multi_slice || config.bounces == 0 {
            readback_slice_into(tex_slice)?;
        }
    }
    println!(
        "  [VLM-GPU] 直接光与太阳软阴影烘焙完成 ({:.2}s)",
        t_direct.elapsed().as_secs_f64()
    );

    // 5. Pass 2: 多跳间接漫反射（纹素分块 × 采样分子批，乒乓缓冲 tile_cache_a <-> tile_cache_b）
    let mut read_is_a = true;
    for bounce in 0..config.bounces {
        let t_bounce = Instant::now();
        base_params.bounce_salt = 0x9E37_79B9u32.wrapping_mul(bounce + 1);
        let bounce_bg = if read_is_a {
            &bounce_bg_a_to_b
        } else {
            &bounce_bg_b_to_a
        };
        let is_last_bounce = bounce + 1 == config.bounces;

        for tex_slice in active_texels.chunks_mut(MAX_ACTIVE_SLICE_TEXELS) {
            if multi_slice {
                queue.write_buffer(&active_buf, 0, bytemuck::cast_slice(tex_slice));
            }
            let slice_len = tex_slice.len() as u32;
            let mut tex_offset = 0u32;
            while tex_offset < slice_len {
                let chunk_count = (slice_len - tex_offset).min(BOUNCE_CHUNK_TEXELS);
                base_params.active_offset = tex_offset;
                base_params.active_count = chunk_count;

                let mut sample_start = 0u32;
                while sample_start < total_samples {
                    let batch_count = (total_samples - sample_start).min(BOUNCE_SAMPLE_BATCH);
                    base_params.sample_start = sample_start;
                    base_params.sample_batch_count = batch_count;
                    queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&base_params));

                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("vlm_bounce_enc"),
                        });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("vlm_bounce_pass"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&bounce_pipeline);
                        pass.set_bind_group(0, bounce_bg, &[]);
                        pass.dispatch_workgroups(chunk_count.div_ceil(WORKGROUP_SIZE), 1, 1);
                    }
                    queue.submit(Some(encoder.finish()));
                    device
                        .poll(wgpu::PollType::wait_indefinitely())
                        .map_err(|e| {
                            format!("GPU Pass 2 Bounce {} 同步失败: {e:?}", bounce + 1)
                        })?;

                    sample_start += batch_count;
                }
                tex_offset += chunk_count;
            }
            if multi_slice || is_last_bounce {
                readback_slice_into(tex_slice)?;
            }
        }

        read_is_a = !read_is_a;
        println!(
            "  [VLM-GPU] 间接反弹 Bounce {}/{} 完成 ({} spp, {:.2}s)",
            bounce + 1,
            config.bounces,
            total_samples,
            t_bounce.elapsed().as_secs_f64()
        );
    }

    pop_error_scopes(validation_scope, oom_scope, "Compute 执行与回读")?;

    // 6. 将回读完成的烘焙结果写回 tile_list
    for item in &active_texels {
        let ti = (item.tile_texel_idx / 64) as usize;
        let k = (item.tile_texel_idx % 64) as usize;
        if let Some(tile) = tile_list.get_mut(ti) {
            let dst = &mut tile.texels[k];
            dst.sun_direct_over_pi = [
                item.sun_direct_and_vis[0],
                item.sun_direct_and_vis[1],
                item.sun_direct_and_vis[2],
            ];
            dst.sun_vis = item.sun_direct_and_vis[3];
            dst.local_direct_over_pi = [
                item.local_direct_and_pad[0],
                item.local_direct_and_pad[1],
                item.local_direct_and_pad[2],
            ];
            dst.indirect_over_pi = [
                item.indirect_and_pad[0],
                item.indirect_and_pad[1],
                item.indirect_and_pad[2],
            ];
        }
    }

    Ok(adapter_label)
}
