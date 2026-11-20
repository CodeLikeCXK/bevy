//! 离线静态表面虚拟光照贴图（Surface Tangent-Plane SpatialHash 8×8 Tile Virtual Lightmap）原生烘焙模块
//!
//! 高层架构思路启发自 `johanholwerda/virtualLightmaps`（空间哈希光照砖块与 Radiance Cache 自举多跳概念），
//! 面向 Bevy Meshlet 管线独立从零实现并针对低功耗 GPU 优化：
//! 1. 无需第二套 UV（UV2），直接按世界坐标 + 6 主法线方向 + 深度切片 `w_slice` 组织 8×8 表面砖块（Tile），与 Meshlet 多级 LOD 完全解耦；
//! 2. 离线计算：
//!    - `A` 通道：主平行光（Sun）软阴影可见度（Shadow Mask，0..=1），使运行时可关闭所有静态物体实时阴影投射，仅保留动态角色实时阴影；
//!    - `RGB` 通道：静态局部光（Point / Spot）直接阴影漫反射 + 太阳/局部光/HDRI 天光的多次反弹间接漫反射（GI，已预除 PI）；
//! 3. 切平面 3×3 边缘保持双边滤波 + 砖块内空纹素形态学膨胀 + 10×10 带 1px 跨邻块边界硬件 2D `Rgba16Float` 图集打包，
//!    保证运行时仅需 1 次 SSBO 哈希查找 + 1 次硬件 `textureSampleLevel` 即可完成无缝双线性插值。

use std::collections::HashMap;
use std::f32::consts::{FRAC_PI_4, PI, TAU};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use bevy_math::{EulerRot, Mat3, Mat4, Quat, Vec2, Vec3};
use bevy_tasks::{AsyncComputeTaskPool, ParallelSlice};
use lz4_flex::frame::FrameEncoder;

use crate::scene::{SceneLight, SceneMaterial, SceneWorldHdri};

pub const VLM_FILE_MAGIC: &[u8; 8] = b"BVVLM001";
pub const VLM_FILE_VERSION: u32 = 1;
pub const VLM_TILE_SIZE: u32 = 8;
pub const VLM_PHYSICAL_TILE_SIZE: u32 = 10; // 8x8 core + 1px border on each side
const VLM_W_STEP_FACTOR: f32 = 2.0;
const LOCAL_LIGHT_SHADOW_RAYS: u32 = 6;

/// VLM 离线路径追踪烘焙参数
#[derive(Debug, Clone, Copy)]
pub struct VlmBakeConfig {
    pub enabled: bool,
    /// Level 0 单纹素物理尺寸（米），默认 0.06m (6cm)
    pub texel_size: f32,
    /// 多级 LOD 层数（1..=3），默认 2（例如 6cm + 12cm）
    pub levels: u32,
    /// 每个有效纹素的半球路径追踪采样数（配合 3x3 切平面降噪）
    pub samples: u32,
    /// 多次反弹迭代轮数（基于哈希缓存自举，默认 2）
    pub bounces: u32,
    /// 主平行光软阴影视锥角半径（度），默认 0.8°
    pub sun_angle_deg: f32,
    /// 间接光反弹强度倍率（默认 1.0）
    pub gi_scale: f32,
    /// 导出强度 <= 0.01 的太阳光兜底照度 (lux)
    pub default_sun_illuminance: f32,
    /// 导出强度 <= 0.01 的局部光兜底流明 (lm)
    pub default_point_lumens: f32,
    /// 安全砖块总数上限（防止超大地形突破 8192px 纹理上限）
    pub max_tiles: usize,
    /// 局部点/聚光灯遮挡测试终点灯具避让半径（米，默认 0.35m，避免挂灯铁艺灯笼/灯罩自身挡光形成圆点走样）
    pub local_light_clearance: f32,
    /// 局部点/聚光灯软阴影球面采样半径（米，默认 0.15m，配合 6 spp 生成柔和半影）
    pub local_light_radius: f32,
    /// 优先使用 headless wgpu Compute Shader 进行 GPU 烘焙（失败自动回退多线程 CPU）
    pub use_gpu: bool,
}

impl Default for VlmBakeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            texel_size: 0.06,
            levels: 2,
            samples: 32,
            bounces: 2,
            sun_angle_deg: 0.8,
            gi_scale: 1.0,
            default_sun_illuminance: 22000.0,
            default_point_lumens: 1600.0,
            max_tiles: 240_000,
            local_light_clearance: 0.35,
            local_light_radius: 0.15,
            use_gpu: true,
        }
    }
}

/// 烘焙结果统计元数据
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct VlmBakeReport {
    pub effective_texel_size: f32,
    pub level_count: u32,
    pub used_tile_count: u32,
    pub table_capacity: u32,
    pub search_count: u32,
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub file_bytes: usize,
    pub used_gpu: bool,
}

/// 每个去重材质用于离线反弹计算的线性漫反射与自发光摘要
#[derive(Debug, Clone, Copy)]
pub struct VlmMaterialInfo {
    pub diffuse_albedo: [f32; 3],
    pub emissive: [f32; 3],
    pub double_sided: bool,
}

impl VlmMaterialInfo {
    pub fn from_scene_material(mat: &SceneMaterial, tex_avg_linear: Option<[f32; 3]>) -> Self {
        let non_metal = (1.0 - mat.metallic.clamp(0.0, 1.0)).max(0.05);
        let tex = tex_avg_linear.unwrap_or([1.0, 1.0, 1.0]);
        let diffuse_albedo = [
            (mat.base_color[0] * tex[0] * non_metal).clamp(0.02, 0.92),
            (mat.base_color[1] * tex[1] * non_metal).clamp(0.02, 0.92),
            (mat.base_color[2] * tex[2] * non_metal).clamp(0.02, 0.92),
        ];
        Self {
            diffuse_albedo,
            emissive: mat.emissive,
            double_sided: mat.double_sided,
        }
    }
}

/// 计算一张 glTF 解码贴图的平均线性 sRGB 颜色（快速跨步采样，供烘焙色溢估算）
pub fn estimate_gltf_image_avg_linear_rgb(data: &gltf::image::Data) -> [f32; 3] {
    use gltf::image::Format;
    let total_px = (data.width as usize) * (data.height as usize);
    if total_px == 0 {
        return [1.0, 1.0, 1.0];
    }
    let stride = (total_px / 4096).max(1);
    let mut sum = [0.0f64; 3];
    let mut count = 0usize;

    let srgb_to_lin = |c: f32| -> f64 {
        let v = c.clamp(0.0, 1.0);
        if v <= 0.04045 {
            (v / 12.92) as f64
        } else {
            (((v + 0.055) / 1.055).powf(2.4)) as f64
        }
    };

    match data.format {
        Format::R8 => {
            for i in (0..total_px).step_by(stride) {
                let l = srgb_to_lin(data.pixels[i] as f32 / 255.0);
                sum[0] += l;
                sum[1] += l;
                sum[2] += l;
                count += 1;
            }
        }
        Format::R8G8 => {
            for i in (0..total_px).step_by(stride) {
                let off = i * 2;
                sum[0] += srgb_to_lin(data.pixels[off] as f32 / 255.0);
                sum[1] += srgb_to_lin(data.pixels[off + 1] as f32 / 255.0);
                count += 1;
            }
        }
        Format::R8G8B8 => {
            for i in (0..total_px).step_by(stride) {
                let off = i * 3;
                sum[0] += srgb_to_lin(data.pixels[off] as f32 / 255.0);
                sum[1] += srgb_to_lin(data.pixels[off + 1] as f32 / 255.0);
                sum[2] += srgb_to_lin(data.pixels[off + 2] as f32 / 255.0);
                count += 1;
            }
        }
        Format::R8G8B8A8 => {
            for i in (0..total_px).step_by(stride) {
                let off = i * 4;
                if data.pixels[off + 3] < 16 {
                    continue;
                }
                sum[0] += srgb_to_lin(data.pixels[off] as f32 / 255.0);
                sum[1] += srgb_to_lin(data.pixels[off + 1] as f32 / 255.0);
                sum[2] += srgb_to_lin(data.pixels[off + 2] as f32 / 255.0);
                count += 1;
            }
        }
        Format::R16 | Format::R16G16 | Format::R16G16B16 | Format::R16G16B16A16 => {
            let ch = match data.format {
                Format::R16 => 1,
                Format::R16G16 => 2,
                Format::R16G16B16 => 3,
                _ => 4,
            };
            for i in (0..total_px).step_by(stride) {
                let off = i * ch * 2;
                let r = u16::from_le_bytes([data.pixels[off], data.pixels[off + 1]]) as f32 / 65535.0;
                let g = if ch >= 2 {
                    u16::from_le_bytes([data.pixels[off + 2], data.pixels[off + 3]]) as f32 / 65535.0
                } else {
                    r
                };
                let b = if ch >= 3 {
                    u16::from_le_bytes([data.pixels[off + 4], data.pixels[off + 5]]) as f32 / 65535.0
                } else {
                    r
                };
                sum[0] += srgb_to_lin(r);
                sum[1] += srgb_to_lin(g);
                sum[2] += srgb_to_lin(b);
                count += 1;
            }
        }
        Format::R32G32B32FLOAT | Format::R32G32B32A32FLOAT => {
            let ch = if matches!(data.format, Format::R32G32B32FLOAT) {
                3
            } else {
                4
            };
            for i in (0..total_px).step_by(stride) {
                let off = i * ch * 4;
                let r = f32::from_le_bytes(data.pixels[off..off + 4].try_into().unwrap());
                let g = f32::from_le_bytes(data.pixels[off + 4..off + 8].try_into().unwrap());
                let b = f32::from_le_bytes(data.pixels[off + 8..off + 12].try_into().unwrap());
                sum[0] += r.clamp(0.0, 1.0) as f64;
                sum[1] += g.clamp(0.0, 1.0) as f64;
                sum[2] += b.clamp(0.0, 1.0) as f64;
                count += 1;
            }
        }
    }

    if count == 0 {
        return [1.0, 1.0, 1.0];
    }
    let inv = 1.0 / count as f64;
    [
        (sum[0] * inv) as f32,
        (sum[1] * inv) as f32,
        (sum[2] * inv) as f32,
    ]
}

// ============================================================================
// 共享空间哈希函数（与 virtual_lightmap.wesl 保持位级完全一致）
// ============================================================================

#[inline]
pub fn vlm_hash_u32(mut a: u32) -> u32 {
    a = (a ^ 61) ^ (a >> 16);
    a = a.wrapping_add(a << 3);
    a = a ^ (a >> 4);
    a = a.wrapping_mul(0x27d4_eb2d);
    a = a ^ (a >> 15);
    a
}

#[inline]
pub fn vlm_pcg_u32(x: u32) -> u32 {
    let state = x.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

#[inline]
pub fn vlm_get_tile_key(
    tile_u: i32,
    tile_v: i32,
    w_slice: i32,
    dir_id: u32,
    level: u32,
    table_mask: u32,
) -> (u32, u32) {
    let u = tile_u as u32;
    let v = tile_v as u32;
    let w = w_slice as u32;
    let meta = (dir_id & 7) | ((level & 7) << 3);
    let h = vlm_hash_u32(u ^ vlm_hash_u32(v ^ vlm_hash_u32(w ^ vlm_hash_u32(meta))));
    let key = h.max(1);
    let slot = vlm_pcg_u32(
        u.wrapping_add(vlm_pcg_u32(
            v.wrapping_add(vlm_pcg_u32(w.wrapping_add(vlm_pcg_u32(meta)))),
        )),
    ) & table_mask;
    (key, slot)
}

#[inline]
fn dir_id_from_normal(n: Vec3) -> u32 {
    let a = n.abs();
    if a.x >= a.y && a.x >= a.z {
        if n.x >= 0.0 { 0 } else { 1 }
    } else if a.y >= a.z {
        if n.y >= 0.0 { 2 } else { 3 }
    } else if n.z >= 0.0 {
        4
    } else {
        5
    }
}

#[inline]
fn dir_vec_from_id(dir_id: u32) -> Vec3 {
    match dir_id {
        0 => Vec3::X,
        1 => Vec3::NEG_X,
        2 => Vec3::Y,
        3 => Vec3::NEG_Y,
        4 => Vec3::Z,
        _ => Vec3::NEG_Z,
    }
}

#[inline]
fn project_to_tangent_and_w(p: Vec3, dir_id: u32) -> (Vec2, f32) {
    if dir_id < 2 {
        (Vec2::new(p.y, p.z), p.x)
    } else if dir_id < 4 {
        (Vec2::new(p.x, p.z), p.y)
    } else {
        (Vec2::new(p.x, p.y), p.z)
    }
}

// ============================================================================
// 带材质属性的静态场景光线追踪 BVH
// ============================================================================

#[derive(Clone, Copy)]
pub(crate) struct VlmTri {
    pub(crate) p0: [f32; 3],
    pub(crate) e1: [f32; 3],
    pub(crate) e2: [f32; 3],
    pub(crate) n0: [f32; 3],
    pub(crate) n1: [f32; 3],
    pub(crate) n2: [f32; 3],
    pub(crate) geom_normal: [f32; 3],
    pub(crate) albedo: [f32; 3],
    pub(crate) emissive: [f32; 3],
    pub(crate) double_sided: bool,
}

#[derive(Clone, Copy)]
struct BuildMeta {
    tri: VlmTri,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    centroid: [f32; 3],
}

#[derive(Clone, Copy)]
pub(crate) struct BvhNode {
    pub(crate) aabb_min: [f32; 3],
    pub(crate) left_or_first: u32,
    pub(crate) aabb_max: [f32; 3],
    pub(crate) right_child: u32,
    pub(crate) tri_count: u32,
}

#[derive(Clone, Copy)]
struct RayHit {
    pos: Vec3,
    normal: Vec3,
    albedo: Vec3,
    emissive: Vec3,
}

#[derive(Default)]
pub struct VlmSceneCollector {
    metas: Vec<BuildMeta>,
    total_area: f64,
}

impl VlmSceneCollector {
    pub fn push_indexed_mesh(
        &mut self,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        indices: &[u32],
        world: Option<&Mat4>,
        mat: VlmMaterialInfo,
    ) {
        let normal_mat = world.map(|m| {
            let lin = Mat3::from_mat4(*m);
            if lin.determinant().abs() > 1e-12 {
                lin.inverse().transpose()
            } else {
                Mat3::IDENTITY
            }
        });
        let flip_winding = world.is_some_and(|m| m.determinant() < 0.0);

        self.metas.reserve(indices.len() / 3);
        for tri in indices.chunks_exact(3) {
            let (i0, i1, i2) = if flip_winding {
                (tri[0] as usize, tri[2] as usize, tri[1] as usize)
            } else {
                (tri[0] as usize, tri[1] as usize, tri[2] as usize)
            };
            if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
                continue;
            }
            let (p0, p1, p2) = if let Some(m) = world {
                (
                    m.transform_point3(Vec3::from_array(positions[i0])),
                    m.transform_point3(Vec3::from_array(positions[i1])),
                    m.transform_point3(Vec3::from_array(positions[i2])),
                )
            } else {
                (
                    Vec3::from_array(positions[i0]),
                    Vec3::from_array(positions[i1]),
                    Vec3::from_array(positions[i2]),
                )
            };
            if !p0.is_finite() || !p1.is_finite() || !p2.is_finite() {
                continue;
            }
            let e1 = p1 - p0;
            let e2 = p2 - p0;
            let cross = e1.cross(e2);
            let len_sq = cross.length_squared();
            if len_sq <= 1e-12 || !len_sq.is_finite() {
                continue;
            }
            let len = len_sq.sqrt();
            let geom_n = cross / len;
            let area = 0.5 * len;

            let xform_n = |idx: usize| -> Vec3 {
                let raw = normals
                    .get(idx)
                    .copied()
                    .map(Vec3::from_array)
                    .unwrap_or(geom_n);
                let wn = if let Some(nm) = normal_mat {
                    nm * raw
                } else {
                    raw
                };
                if wn.is_finite() && wn.length_squared() > 1e-10 {
                    wn.normalize()
                } else {
                    geom_n
                }
            };

            let n0 = xform_n(i0);
            let n1 = xform_n(i1);
            let n2 = xform_n(i2);

            self.total_area += area as f64;
            self.metas.push(BuildMeta {
                tri: VlmTri {
                    p0: p0.to_array(),
                    e1: e1.to_array(),
                    e2: e2.to_array(),
                    n0: n0.to_array(),
                    n1: n1.to_array(),
                    n2: n2.to_array(),
                    geom_normal: geom_n.to_array(),
                    albedo: mat.diffuse_albedo,
                    emissive: mat.emissive,
                    double_sided: mat.double_sided,
                },
                aabb_min: p0.min(p1).min(p2).to_array(),
                aabb_max: p0.max(p1).max(p2).to_array(),
                centroid: ((p0 + p1 + p2) * (1.0 / 3.0)).to_array(),
            });
        }
    }

    pub fn triangle_count(&self) -> usize {
        self.metas.len()
    }

    pub fn total_surface_area(&self) -> f64 {
        self.total_area
    }

    pub fn build(self) -> VlmBvh {
        VlmBvh::build_from_metas(self.metas)
    }
}

pub struct VlmBvh {
    pub(crate) nodes: Vec<BvhNode>,
    pub(crate) tris: Vec<VlmTri>,
}

impl VlmBvh {
    fn build_from_metas(mut metas: Vec<BuildMeta>) -> Self {
        if metas.is_empty() {
            return Self {
                nodes: Vec::new(),
                tris: Vec::new(),
            };
        }
        let count = metas.len();
        let mut nodes = Vec::with_capacity(count);
        let mut order: Vec<usize> = (0..count).collect();
        Self::build_node(&metas, &mut order, 0, count, 0, &mut nodes);
        let tris: Vec<VlmTri> = order.into_iter().map(|idx| metas[idx].tri).collect();
        metas.clear();
        Self { nodes, tris }
    }

    fn build_node(
        metas: &[BuildMeta],
        order: &mut [usize],
        start: usize,
        end: usize,
        depth: u32,
        nodes: &mut Vec<BvhNode>,
    ) -> u32 {
        let mut aabb_min = [f32::INFINITY; 3];
        let mut aabb_max = [f32::NEG_INFINITY; 3];
        let mut cent_min = [f32::INFINITY; 3];
        let mut cent_max = [f32::NEG_INFINITY; 3];

        for &idx in &order[start..end] {
            let m = &metas[idx];
            for k in 0..3 {
                aabb_min[k] = aabb_min[k].min(m.aabb_min[k]);
                aabb_max[k] = aabb_max[k].max(m.aabb_max[k]);
                cent_min[k] = cent_min[k].min(m.centroid[k]);
                cent_max[k] = cent_max[k].max(m.centroid[k]);
            }
        }

        let tri_count = end - start;
        let node_idx = nodes.len() as u32;
        nodes.push(BvhNode {
            aabb_min,
            left_or_first: start as u32,
            aabb_max,
            right_child: 0,
            tri_count: 0,
        });

        if tri_count <= 12 || depth >= 48 {
            nodes[node_idx as usize].tri_count = tri_count as u32;
            return node_idx;
        }

        let ext = [
            cent_max[0] - cent_min[0],
            cent_max[1] - cent_min[1],
            cent_max[2] - cent_min[2],
        ];
        let axis = if ext[0] >= ext[1] && ext[0] >= ext[2] {
            0
        } else if ext[1] >= ext[2] {
            1
        } else {
            2
        };

        if ext[axis] <= 1e-7 {
            nodes[node_idx as usize].tri_count = tri_count as u32;
            return node_idx;
        }

        let mid = start + tri_count / 2;
        let local_mid = mid - start;
        order[start..end].select_nth_unstable_by(local_mid, |&a, &b| {
            metas[a].centroid[axis].total_cmp(&metas[b].centroid[axis])
        });

        let left_child = Self::build_node(metas, order, start, mid, depth + 1, nodes);
        let right_child = Self::build_node(metas, order, mid, end, depth + 1, nodes);
        nodes[node_idx as usize].left_or_first = left_child;
        nodes[node_idx as usize].right_child = right_child;
        node_idx
    }

    #[inline]
    fn inv_dir(dir: Vec3) -> Vec3 {
        Vec3::new(
            if dir.x.abs() < 1e-8 {
                1e8_f32.copysign(dir.x)
            } else {
                1.0 / dir.x
            },
            if dir.y.abs() < 1e-8 {
                1e8_f32.copysign(dir.y)
            } else {
                1.0 / dir.y
            },
            if dir.z.abs() < 1e-8 {
                1e8_f32.copysign(dir.z)
            } else {
                1.0 / dir.z
            },
        )
    }

    #[inline]
    fn intersect_aabb(
        origin: Vec3,
        inv_dir: Vec3,
        bmin: [f32; 3],
        bmax: [f32; 3],
        t_min: f32,
        t_max: f32,
    ) -> Option<f32> {
        let t0x = (bmin[0] - origin.x) * inv_dir.x;
        let t1x = (bmax[0] - origin.x) * inv_dir.x;
        let t0y = (bmin[1] - origin.y) * inv_dir.y;
        let t1y = (bmax[1] - origin.y) * inv_dir.y;
        let t0z = (bmin[2] - origin.z) * inv_dir.z;
        let t1z = (bmax[2] - origin.z) * inv_dir.z;

        let enter = t0x.min(t1x).max(t0y.min(t1y)).max(t0z.min(t1z)).max(t_min);
        let exit = t0x.max(t1x).min(t0y.max(t1y)).min(t0z.max(t1z)).min(t_max);
        if enter <= exit { Some(enter) } else { None }
    }

    #[inline]
    fn intersect_tri(
        origin: Vec3,
        dir: Vec3,
        tri: &VlmTri,
        t_min: f32,
        t_max: f32,
        backface_guard_dist: f32,
        ignore_emissive: bool,
    ) -> Option<(f32, f32, f32)> {
        if ignore_emissive && (tri.emissive[0] + tri.emissive[1] + tri.emissive[2]) > 0.05 {
            return None;
        }
        let p0 = Vec3::from_array(tri.p0);
        let e1 = Vec3::from_array(tri.e1);
        let e2 = Vec3::from_array(tri.e2);
        let h = dir.cross(e2);
        let a = e1.dot(h);
        if a.abs() < 1e-8 {
            return None;
        }
        let f = 1.0 / a;
        let s = origin - p0;
        let u = f * s.dot(h);
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = s.cross(e1);
        let v = f * dir.dot(q);
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = f * e2.dot(q);
        if t < t_min || t >= t_max {
            return None;
        }
        if t < backface_guard_dist && dir.dot(Vec3::from_array(tri.geom_normal)) > 0.0 {
            return None;
        }
        Some((t, u, v))
    }

    /// 快速遮蔽测试（阴影射线，命中任意遮挡面即返回 true）
    fn is_occluded(
        &self,
        origin: Vec3,
        dir: Vec3,
        t_min: f32,
        max_dist: f32,
        backface_guard_dist: f32,
        ignore_emissive: bool,
    ) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let inv_dir = Self::inv_dir(dir);
        let root = &self.nodes[0];
        if Self::intersect_aabb(origin, inv_dir, root.aabb_min, root.aabb_max, t_min, max_dist)
            .is_none()
        {
            return false;
        }

        let mut stack = [0u32; 64];
        let mut sp = 1usize;
        stack[0] = 0;

        while sp > 0 {
            sp -= 1;
            let node = &self.nodes[stack[sp] as usize];
            if Self::intersect_aabb(origin, inv_dir, node.aabb_min, node.aabb_max, t_min, max_dist)
                .is_none()
            {
                continue;
            }
            if node.tri_count > 0 {
                let first = node.left_or_first as usize;
                let end = first + node.tri_count as usize;
                for tri in &self.tris[first..end] {
                    if Self::intersect_tri(
                        origin,
                        dir,
                        tri,
                        t_min,
                        max_dist,
                        backface_guard_dist,
                        ignore_emissive,
                    )
                    .is_some()
                    {
                        return true;
                    }
                }
            } else {
                if sp + 2 <= stack.len() {
                    stack[sp] = node.right_child;
                    stack[sp + 1] = node.left_or_first;
                    sp += 2;
                }
            }
        }
        false
    }

    /// 追踪最近表面交点（用于间接反弹查询）
    fn trace_closest(
        &self,
        origin: Vec3,
        dir: Vec3,
        t_min: f32,
        max_dist: f32,
        backface_guard_dist: f32,
    ) -> Option<RayHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let inv_dir = Self::inv_dir(dir);
        let root = &self.nodes[0];
        if Self::intersect_aabb(origin, inv_dir, root.aabb_min, root.aabb_max, t_min, max_dist)
            .is_none()
        {
            return None;
        }

        let mut best_t = max_dist;
        let mut best_uv = (0.0f32, 0.0f32);
        let mut best_tri_idx = usize::MAX;
        let mut stack = [0u32; 64];
        let mut sp = 1usize;
        stack[0] = 0;

        while sp > 0 {
            sp -= 1;
            let node = &self.nodes[stack[sp] as usize];
            if Self::intersect_aabb(origin, inv_dir, node.aabb_min, node.aabb_max, t_min, best_t)
                .is_none()
            {
                continue;
            }
            if node.tri_count > 0 {
                let first = node.left_or_first as usize;
                let end = first + node.tri_count as usize;
                for (ti, tri) in self.tris[first..end].iter().enumerate() {
                    if let Some((t, u, v)) = Self::intersect_tri(
                        origin,
                        dir,
                        tri,
                        t_min,
                        best_t,
                        backface_guard_dist,
                        false,
                    ) {
                        best_t = t;
                        best_uv = (u, v);
                        best_tri_idx = first + ti;
                    }
                }
            } else {
                let l_idx = node.left_or_first;
                let r_idx = node.right_child;
                let l_node = &self.nodes[l_idx as usize];
                let r_node = &self.nodes[r_idx as usize];
                let t_left = Self::intersect_aabb(
                    origin,
                    inv_dir,
                    l_node.aabb_min,
                    l_node.aabb_max,
                    t_min,
                    best_t,
                );
                let t_right = Self::intersect_aabb(
                    origin,
                    inv_dir,
                    r_node.aabb_min,
                    r_node.aabb_max,
                    t_min,
                    best_t,
                );
                match (t_left, t_right) {
                    (Some(tl), Some(tr)) if sp + 2 <= stack.len() => {
                        if tl <= tr {
                            stack[sp] = r_idx;
                            stack[sp + 1] = l_idx;
                        } else {
                            stack[sp] = l_idx;
                            stack[sp + 1] = r_idx;
                        }
                        sp += 2;
                    }
                    (Some(_), None) if sp < stack.len() => {
                        stack[sp] = l_idx;
                        sp += 1;
                    }
                    (None, Some(_)) if sp < stack.len() => {
                        stack[sp] = r_idx;
                        sp += 1;
                    }
                    _ => {}
                }
            }
        }

        if best_tri_idx == usize::MAX {
            return None;
        }
        let tri = &self.tris[best_tri_idx];
        let (u, v) = best_uv;
        let w = 1.0 - u - v;
        let mut n = (Vec3::from_array(tri.n0) * w
            + Vec3::from_array(tri.n1) * u
            + Vec3::from_array(tri.n2) * v)
            .normalize_or(Vec3::from_array(tri.geom_normal));
        if tri.double_sided && n.dot(dir) > 0.0 {
            n = -n;
        }
        Some(RayHit {
            pos: origin + dir * best_t,
            normal: n,
            albedo: Vec3::from_array(tri.albedo),
            emissive: Vec3::from_array(tri.emissive),
        })
    }
}

// ============================================================================
// 灯光与天空盒采样结构
// ============================================================================

#[derive(Clone, Copy)]
pub(crate) struct BakedSunLight {
    pub(crate) dir_to_light: Vec3,
    pub(crate) color: Vec3,
    pub(crate) illuminance: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct BakedLocalLight {
    pub(crate) is_spot: bool,
    pub(crate) position: Vec3,
    pub(crate) forward: Vec3,
    pub(crate) color: Vec3,
    pub(crate) candela: f32,
    pub(crate) range: f32,
    pub(crate) inv_range_sq: f32,
    pub(crate) cos_inner: f32,
    pub(crate) cos_outer: f32,
}

pub(crate) struct SkySampler {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pixels: Vec<[f32; 3]>,
    pub(crate) yaw_rad: f32,
    pub(crate) scale: f32,
}

impl SkySampler {
    fn from_world_hdri(hdri: Option<&SceneWorldHdri>, out_dir: &Path) -> Self {
        let Some(cfg) = hdri else {
            return Self::procedural_default(1.0);
        };
        let path = out_dir.join(&cfg.image);
        if path.is_file()
            && let Ok(img) = image::open(&path)
        {
            let rgb32 = img.to_rgb32f();
            let (w, h) = (rgb32.width(), rgb32.height());
            if w > 0 && h > 0 {
                // 降采样到最多 256x128 用于极速离线天光半球采样
                let dst_w = w.min(256);
                let dst_h = h.min(128);
                let mut pixels = Vec::with_capacity((dst_w * dst_h) as usize);
                for y in 0..dst_h {
                    let sy = ((y as f32 + 0.5) / dst_h as f32 * h as f32).min(h as f32 - 1.0) as u32;
                    for x in 0..dst_w {
                        let sx = ((x as f32 + 0.5) / dst_w as f32 * w as f32).min(w as f32 - 1.0) as u32;
                        let p = rgb32.get_pixel(sx, sy).0;
                        pixels.push([
                            p[0].clamp(0.0, 50.0),
                            p[1].clamp(0.0, 50.0),
                            p[2].clamp(0.0, 50.0),
                        ]);
                    }
                }
                return Self {
                    width: dst_w,
                    height: dst_h,
                    pixels,
                    yaw_rad: cfg.yaw_degrees.to_radians(),
                    scale: 250.0 * cfg.strength.max(0.05),
                };
            }
        }
        Self::procedural_default(cfg.strength.max(0.1))
    }

    fn procedural_default(strength: f32) -> Self {
        Self {
            width: 0,
            height: 0,
            pixels: Vec::new(),
            yaw_rad: 0.0,
            scale: 220.0 * strength,
        }
    }

    #[inline]
    fn sample(&self, dir: Vec3) -> Vec3 {
        if self.width == 0 || self.height == 0 {
            let up = (dir.y * 0.5 + 0.5).clamp(0.0, 1.0);
            let ground = Vec3::new(0.14, 0.13, 0.12);
            let horizon = Vec3::new(0.62, 0.72, 0.86);
            let zenith = Vec3::new(0.38, 0.58, 0.95);
            let c = if up < 0.5 {
                ground.lerp(horizon, up * 2.0)
            } else {
                horizon.lerp(zenith, (up - 0.5) * 2.0)
            };
            return c * self.scale;
        }
        // 与 action_renderer skybox.rs 对齐的球面映射
        let (sin_y, cos_y) = (-self.yaw_rad).sin_cos();
        let rx = dir.x * cos_y + dir.z * sin_y;
        let rz = -dir.x * sin_y + dir.z * cos_y;
        let ry = dir.y.clamp(-1.0, 1.0);

        let u = (rz.atan2(rx) / TAU + 0.5).rem_euclid(1.0);
        let v = (ry.acos() / PI).clamp(0.0, 1.0);
        let ix = ((u * self.width as f32) as u32).min(self.width - 1);
        let iy = ((v * self.height as f32) as u32).min(self.height - 1);
        let p = self.pixels[(iy * self.width + ix) as usize];
        Vec3::from_array(p) * self.scale
    }
}

fn prepare_lights(
    scene_lights: &[SceneLight],
    config: &VlmBakeConfig,
) -> (BakedSunLight, Vec<BakedLocalLight>) {
    let mut sun: Option<BakedSunLight> = None;
    let mut locals = Vec::new();

    for l in scene_lights {
        let name_lower = l.name.to_lowercase();
        if l.kind != "Directional" && name_lower.contains("hdri_sky") && l.intensity <= 0.01 {
            continue;
        }
        let color = Vec3::from_array(l.color).max(Vec3::ZERO);
        let mut rot = Quat::from_array(l.rotation);
        rot = if rot.length_squared() > 1e-6 {
            rot.normalize()
        } else {
            Quat::IDENTITY
        };

        match l.kind.as_str() {
            "Directional" => {
                if sun.is_none() {
                    let illuminance = if l.intensity <= 0.01 && config.default_sun_illuminance > 0.0 {
                        config.default_sun_illuminance
                    } else {
                        l.intensity
                    };
                    // Bevy DirectionalLight direction_to_light = transform.back() = rot * Vec3::Z
                    let dir_to_light = (rot * Vec3::Z).normalize_or(Vec3::Y);
                    sun = Some(BakedSunLight {
                        dir_to_light,
                        color,
                        illuminance,
                    });
                }
            }
            "Spot" => {
                let candela = if l.intensity <= 0.01 && config.default_point_lumens > 0.0 {
                    config.default_point_lumens / (4.0 * PI)
                } else {
                    l.intensity
                };
                let range = l.range.filter(|&r| r > 0.05).unwrap_or(10.0);
                let inner = l.inner_cone_angle.unwrap_or(0.0).clamp(0.0, PI * 0.45);
                let outer = l
                    .outer_cone_angle
                    .unwrap_or(0.785)
                    .max(inner + 0.05)
                    .clamp(0.05, PI * 0.49);
                locals.push(BakedLocalLight {
                    is_spot: true,
                    position: Vec3::from_array(l.translation),
                    forward: (rot * Vec3::NEG_Z).normalize_or(Vec3::NEG_Y),
                    color,
                    candela,
                    range,
                    inv_range_sq: 1.0 / (range * range),
                    cos_inner: inner.cos(),
                    cos_outer: outer.cos(),
                });
            }
            _ => {
                let candela = if l.intensity <= 0.01 && config.default_point_lumens > 0.0 {
                    config.default_point_lumens / (4.0 * PI)
                } else {
                    l.intensity
                };
                let range = l.range.filter(|&r| r > 0.05).unwrap_or(10.0);
                locals.push(BakedLocalLight {
                    is_spot: false,
                    position: Vec3::from_array(l.translation),
                    forward: Vec3::NEG_Y,
                    color,
                    candela,
                    range,
                    inv_range_sq: 1.0 / (range * range),
                    cos_inner: 1.0,
                    cos_outer: -1.0,
                });
            }
        }
    }

    let sun = sun.unwrap_or_else(|| {
        let default_rot = Quat::from_euler(EulerRot::ZYX, 0.0, -FRAC_PI_4, -FRAC_PI_4);
        BakedSunLight {
            dir_to_light: (default_rot * Vec3::Z).normalize(),
            color: Vec3::ONE,
            illuminance: 1000.0, // light_consts::lux::OVERCAST_DAY，匹配 scene_loader::spawn_default_light
        }
    });

    (sun, locals)
}

// ============================================================================
// 表面空间哈希砖块结构
// ============================================================================

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub(crate) struct TileCoord {
    pub(crate) tile_u: i32,
    pub(crate) tile_v: i32,
    pub(crate) w_slice: i32,
    pub(crate) dir_id: u8,
    pub(crate) level: u8,
}

#[derive(Clone, Copy)]
pub(crate) struct TexelSampleData {
    pub(crate) pos_sum: [f32; 3],
    pub(crate) norm_sum: [f32; 3],
    pub(crate) weight: f32,
    /// 该纹素内所有几何采样点沿砖块主方向的最大投影，防止曲面/拱门边缘质心内陷自遮挡
    pub(crate) max_dir_proj: f32,
    /// 直接太阳光辐照度 / PI（仅用于反弹计算，不写入最终 RGB 输出）
    pub(crate) sun_direct_over_pi: [f32; 3],
    /// 静态局部光直接漫反射辐照度 / PI
    pub(crate) local_direct_over_pi: [f32; 3],
    /// 多次反弹间接漫反射辐照度 / PI（含天光遮蔽）
    pub(crate) indirect_over_pi: [f32; 3],
    /// 太阳软阴影可见度 (0..=1)
    pub(crate) sun_vis: f32,
    /// 供硬件双线性边缘填充用的最终输出缓存 [r, g, b, sun_vis]
    pub(crate) final_rgba: [f32; 4],
    pub(crate) filled: bool,
}

impl Default for TexelSampleData {
    fn default() -> Self {
        Self {
            pos_sum: [0.0; 3],
            norm_sum: [0.0; 3],
            weight: 0.0,
            max_dir_proj: f32::NEG_INFINITY,
            sun_direct_over_pi: [0.0; 3],
            local_direct_over_pi: [0.0; 3],
            indirect_over_pi: [0.0; 3],
            sun_vis: 0.0,
            final_rgba: [0.0; 4],
            filled: false,
        }
    }
}

/// 根据纹素聚合数据计算最终射线起跳位置与法线（对弯曲曲面做外表面投影防内陷补偿）
#[inline]
pub(crate) fn texel_surface_pos_and_normal(
    tx: &TexelSampleData,
    dir_id: u32,
    texel_size: f32,
) -> (Vec3, Vec3) {
    let dir_vec = dir_vec_from_id(dir_id);
    let inv_w = 1.0 / tx.weight;
    let mut p = Vec3::from_array(tx.pos_sum) * inv_w;
    let n = Vec3::from_array(tx.norm_sum).normalize_or(dir_vec);
    if tx.max_dir_proj.is_finite() {
        let proj_avg = p.dot(dir_vec);
        let push = (tx.max_dir_proj - proj_avg).clamp(0.0, texel_size * 2.0);
        p += dir_vec * push;
    }
    (p, n)
}

pub(crate) struct TileData {
    pub(crate) coord: TileCoord,
    pub(crate) texels: Box<[TexelSampleData; 64]>,
}

impl TileData {
    fn new(coord: TileCoord) -> Self {
        Self {
            coord,
            texels: Box::new([TexelSampleData::default(); 64]),
        }
    }
}

#[inline]
fn rasterize_point_into_tile(
    tiles: &mut HashMap<TileCoord, usize>,
    tile_list: &mut Vec<TileData>,
    p: Vec3,
    n: Vec3,
    dir_id: u32,
    level: u32,
    texel_size: f32,
) {
    let dir_vec = dir_vec_from_id(dir_id);
    // 过滤高模 3D 砖缝中接近垂直于主墙面的陡峭侧壁（夹角 > ~63°），防止侧壁法线污染正面纹素
    if n.dot(dir_vec) < 0.45 {
        return;
    }
    let shifted = p + dir_vec * (texel_size * 0.15);
    let (uv_world, w_world) = project_to_tangent_and_w(shifted, dir_id);
    let w_slice = (w_world / (texel_size * VLM_W_STEP_FACTOR)).floor() as i32;
    let tile_world_size = texel_size * 8.0;
    let tile_u = (uv_world.x / tile_world_size).floor() as i32;
    let tile_v = (uv_world.y / tile_world_size).floor() as i32;
    let local_u = ((uv_world.x - tile_u as f32 * tile_world_size) / texel_size)
        .floor()
        .clamp(0.0, 7.0) as usize;
    let local_v = ((uv_world.y - tile_v as f32 * tile_world_size) / texel_size)
        .floor()
        .clamp(0.0, 7.0) as usize;

    let coord = TileCoord {
        tile_u,
        tile_v,
        w_slice,
        dir_id: dir_id as u8,
        level: level as u8,
    };
    let tile_idx = *tiles.entry(coord).or_insert_with(|| {
        let idx = tile_list.len();
        tile_list.push(TileData::new(coord));
        idx
    });
    let texel = &mut tile_list[tile_idx].texels[local_v * 8 + local_u];
    let proj = p.dot(dir_vec);
    let shell_band = texel_size * 0.25;
    if texel.weight > 0.0 {
        if proj > texel.max_dir_proj + shell_band {
            // 更外层的 3D 砖面覆盖底层底板/深砖缝：直接重置为最外层表皮
            texel.pos_sum = [p.x, p.y, p.z];
            texel.norm_sum = [n.x, n.y, n.z];
            texel.weight = 1.0;
            texel.max_dir_proj = proj;
            return;
        } else if proj < texel.max_dir_proj - shell_band {
            // 内层底板或深砖缝采样点：忽略，防止把射线起点拉入 3D 砖皮内部
            return;
        }
    }
    texel.pos_sum[0] += p.x;
    texel.pos_sum[1] += p.y;
    texel.pos_sum[2] += p.z;
    texel.norm_sum[0] += n.x;
    texel.norm_sum[1] += n.y;
    texel.norm_sum[2] += n.z;
    texel.weight += 1.0;
    if proj > texel.max_dir_proj {
        texel.max_dir_proj = proj;
    }
}

/// 将静态场景三角形按切平面步长光栅化到各层空间哈希砖块中（斜面上同时登记次主方向以保证平滑混合无空洞）
fn build_surface_tiles(
    tris: &[VlmTri],
    base_texel_size: f32,
    levels: u32,
) -> (HashMap<TileCoord, usize>, Vec<TileData>) {
    let mut tiles: HashMap<TileCoord, usize> = HashMap::new();
    let mut tile_list: Vec<TileData> = Vec::new();

    for level in 0..levels {
        let texel_size = base_texel_size * (1u32 << level) as f32;
        let sample_step = texel_size * 0.6;

        for tri in tris {
            let p0 = Vec3::from_array(tri.p0);
            let e1 = Vec3::from_array(tri.e1);
            let e2 = Vec3::from_array(tri.e2);
            let n0 = Vec3::from_array(tri.n0);
            let n1 = Vec3::from_array(tri.n1);
            let n2 = Vec3::from_array(tri.n2);
            let geom_n = Vec3::from_array(tri.geom_normal);

            let edge1_len = e1.length();
            let edge2_len = e2.length();
            let edge3_len = (e2 - e1).length();
            let max_edge = edge1_len.max(edge2_len).max(edge3_len);
            let steps = ((max_edge / sample_step).ceil() as u32).clamp(1, 2048);

            let mut emit_sample = |p: Vec3, n: Vec3| {
                let dir0 = dir_id_from_normal(n);
                rasterize_point_into_tile(
                    &mut tiles,
                    &mut tile_list,
                    p,
                    n,
                    dir0,
                    level,
                    texel_size,
                );
                // 45° 斜面或曲面同时登记次近轴向，确保 Shader direction_blend 过渡无缝
                let d0v = dir_vec_from_id(dir0);
                let rem = n * (Vec3::ONE - d0v.abs());
                let dir1 = dir_id_from_normal(rem);
                let d1v = dir_vec_from_id(dir1);
                let p_comp = n.abs().dot(d0v.abs()).max(1e-4);
                let s_comp = n.abs().dot(d1v.abs());
                if s_comp / p_comp >= 0.42 {
                    rasterize_point_into_tile(
                        &mut tiles,
                        &mut tile_list,
                        p,
                        n,
                        dir1,
                        level,
                        texel_size,
                    );
                }
            };

            if steps == 1 {
                // 小三角形直接采样质心（仅按真实几何正面法线登记，避免 double_sided 将实心砖块背面翻转为内部假表面）
                let center = p0 + (e1 + e2) * (1.0 / 3.0);
                let cn = (n0 + n1 + n2).normalize_or(geom_n);
                emit_sample(center, cn);
            } else {
                let inv_s = 1.0 / steps as f32;
                for iu in 0..=steps {
                    for iv in 0..=(steps - iu) {
                        let u = (iu as f32 + 0.333 * (iu < steps) as u32 as f32) * inv_s;
                        let v = (iv as f32 + 0.333 * (iu + iv < steps) as u32 as f32) * inv_s;
                        let (u, v) = if u + v > 1.0 {
                            (iu as f32 * inv_s, iv as f32 * inv_s)
                        } else {
                            (u, v)
                        };
                        let w = (1.0 - u - v).max(0.0);
                        let p = p0 + e1 * u + e2 * v;
                        let n = (n0 * w + n1 * u + n2 * v).normalize_or(geom_n);
                        emit_sample(p, n);
                    }
                }
            }
        }
    }

    // 跨深度切片最外层表皮提升与填补：当 3D 浮雕砖墙/石板地跨越 w_slice 与 w_slice + 1 边界时：
    // 1. 若内侧切片 (w_slice) 某纹素为空（例如凸起砖面全落在了 w_slice + 1，内侧切片仅剩稀疏灰缝），
    //    直接借用外侧切片 (w_slice + 1) 同位置纹素的外表皮几何，防止内侧切片被少量被挡灰缝纹素膨胀污染成整块黑砖；
    // 2. 若内侧切片某纹素有值，将其射线起跳深度提升至紧邻外侧切片 3×3 邻域的最外层表面。
    let mut texel_adoptions: Vec<(usize, usize, TexelSampleData)> = Vec::new();
    let mut proj_promotions: Vec<(usize, usize, f32)> = Vec::new();
    for (ti, tile) in tile_list.iter().enumerate() {
        let c = tile.coord;
        // project_to_tangent_and_w 直接取 p.x/p.y/p.z，因此正轴(+X/+Y/+Z, 偶数 dir_id)外侧为 +1，负轴(-X/-Y/-Z, 奇数 dir_id)外侧为 -1
        let outward_step: i32 = if (c.dir_id & 1) == 0 { 1 } else { -1 };
        let outer_coord = TileCoord {
            w_slice: c.w_slice + outward_step,
            ..c
        };
        let Some(&outer_ti) = tiles.get(&outer_coord) else {
            continue;
        };
        let texel_size = base_texel_size * (1u32 << (c.level as u32)) as f32;
        let max_relief_span = texel_size * VLM_W_STEP_FACTOR * 1.1;
        let outer_tile = &tile_list[outer_ti];

        for ty in 0..8i32 {
            for tx in 0..8i32 {
                let k = (ty * 8 + tx) as usize;
                let inner_tx = &tile.texels[k];
                let direct_outer_tx = &outer_tile.texels[k];
                if inner_tx.weight <= 0.0 || !inner_tx.max_dir_proj.is_finite() {
                    if direct_outer_tx.weight > 0.0 && direct_outer_tx.max_dir_proj.is_finite() {
                        texel_adoptions.push((ti, k, *direct_outer_tx));
                    }
                    continue;
                }
                let mut best_outer = inner_tx.max_dir_proj;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let ox = (tx + dx).clamp(0, 7) as usize;
                        let oy = (ty + dy).clamp(0, 7) as usize;
                        let otx = &outer_tile.texels[oy * 8 + ox];
                        if otx.weight > 0.0
                            && otx.max_dir_proj > best_outer
                            && (otx.max_dir_proj - inner_tx.max_dir_proj) <= max_relief_span
                        {
                            best_outer = otx.max_dir_proj;
                        }
                    }
                }
                if best_outer > inner_tx.max_dir_proj {
                    proj_promotions.push((ti, k, best_outer));
                }
            }
        }
    }
    for (ti, k, adopted_tx) in texel_adoptions {
        tile_list[ti].texels[k] = adopted_tx;
    }
    for (ti, k, promoted_proj) in proj_promotions {
        tile_list[ti].texels[k].max_dir_proj = promoted_proj;
    }

    (tiles, tile_list)
}

// ============================================================================
// 路径追踪采样数学工具
// ============================================================================

#[inline]
fn radical_inverse_base2(mut bits: u32) -> f32 {
    bits = bits.reverse_bits();
    bits as f32 * 2.3283064e-10
}

#[inline]
fn hash_pos_seed(p: Vec3, salt: u32) -> (f32, f32) {
    let qx = (p.x * 1000.0).round() as i32 as u32;
    let qy = (p.y * 1000.0).round() as i32 as u32;
    let qz = (p.z * 1000.0).round() as i32 as u32;
    let h1 = vlm_hash_u32(qx ^ vlm_hash_u32(qy ^ vlm_hash_u32(qz ^ salt)));
    let h2 = vlm_pcg_u32(h1 ^ 0xA5A5_5A5A);
    (
        (h1 & 0x00FF_FFFF) as f32 / 16_777_216.0,
        (h2 & 0x00FF_FFFF) as f32 / 16_777_216.0,
    )
}

#[inline]
fn build_orthonormal_basis(n: Vec3) -> (Vec3, Vec3) {
    if n.z < -0.9999999 {
        (Vec3::new(0.0, -1.0, 0.0), Vec3::new(-1.0, 0.0, 0.0))
    } else {
        let a = 1.0 / (1.0 + n.z);
        let b = -n.x * n.y * a;
        (
            Vec3::new(1.0 - n.x * n.x * a, b, -n.x),
            Vec3::new(b, 1.0 - n.y * n.y * a, -n.y),
        )
    }
}

#[inline]
fn sample_cone_dir(axis: Vec3, cos_max_angle: f32, u1: f32, u2: f32) -> Vec3 {
    let cos_theta = 1.0 - u1 * (1.0 - cos_max_angle);
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = TAU * u2;
    let (t, b) = build_orthonormal_basis(axis);
    (t * (phi.cos() * sin_theta) + b * (phi.sin() * sin_theta) + axis * cos_theta).normalize()
}

/// 在当前内存空间哈希中查询命中点的辐照度 / PI（Radiance Cache 自举）
#[inline]
fn lookup_cached_irradiance(
    tiles: &HashMap<TileCoord, usize>,
    tile_list: &[TileData],
    p: Vec3,
    n: Vec3,
    base_texel_size: f32,
    levels: u32,
) -> Option<Vec3> {
    let dir_id = dir_id_from_normal(n);
    let dir_vec = dir_vec_from_id(dir_id);
    // 优先从最粗层级向精细层级查询（辐射缓存更平滑、命中率更高）
    for level in (0..levels).rev() {
        let texel_size = base_texel_size * (1u32 << level) as f32;
        let shifted = p + dir_vec * (texel_size * 0.15);
        let (uv_world, w_world) = project_to_tangent_and_w(shifted, dir_id);
        let w_step = texel_size * VLM_W_STEP_FACTOR;
        let w_scaled = w_world / w_step;
        let w_slice = w_scaled.floor() as i32;
        let tile_world_size = texel_size * 8.0;
        let tile_u = (uv_world.x / tile_world_size).floor() as i32;
        let tile_v = (uv_world.y / tile_world_size).floor() as i32;
        let tx = ((uv_world.x - tile_u as f32 * tile_world_size) / texel_size)
            .floor()
            .clamp(0.0, 7.0) as usize;
        let ty = ((uv_world.y - tile_v as f32 * tile_world_size) / texel_size)
            .floor()
            .clamp(0.0, 7.0) as usize;

        for dw in [0, if (w_scaled - w_scaled.floor()) >= 0.5 { 1 } else { -1 }] {
            let coord = TileCoord {
                tile_u,
                tile_v,
                w_slice: w_slice + dw,
                dir_id: dir_id as u8,
                level: level as u8,
            };
            if let Some(&idx) = tiles.get(&coord) {
                let t = &tile_list[idx].texels[ty * 8 + tx];
                if t.weight > 0.0 {
                    return Some(
                        Vec3::from_array(t.sun_direct_over_pi)
                            + Vec3::from_array(t.local_direct_over_pi)
                            + Vec3::from_array(t.indirect_over_pi),
                    );
                }
            }
        }
    }
    None
}

// ============================================================================
// 滤波、空纹素膨胀与硬件 10×10 边框打包
// ============================================================================

#[inline]
fn fetch_texel_in_neighborhood<'a>(
    tiles: &HashMap<TileCoord, usize>,
    tile_list: &'a [TileData],
    base_coord: TileCoord,
    gx: i32,
    gy: i32,
) -> Option<&'a TexelSampleData> {
    let du = gx.div_euclid(8);
    let dv = gy.div_euclid(8);
    let lx = gx.rem_euclid(8) as usize;
    let ly = gy.rem_euclid(8) as usize;
    let coord = TileCoord {
        tile_u: base_coord.tile_u + du,
        tile_v: base_coord.tile_v + dv,
        w_slice: base_coord.w_slice,
        dir_id: base_coord.dir_id,
        level: base_coord.level,
    };
    let idx = *tiles.get(&coord)?;
    Some(&tile_list[idx].texels[ly * 8 + lx])
}

/// 切平面 3×3 边缘保持双边滤波（仅对间接光 RGB 平滑，太阳阴影 A 仅做极轻微平滑以保持锐利边缘）
fn bilateral_filter_and_dilate(tiles: &HashMap<TileCoord, usize>, tile_list: &mut [TileData]) {
    let n_tiles = tile_list.len();
    let pool = AsyncComputeTaskPool::get();
    let indices: Vec<usize> = (0..n_tiles).collect();

    // 1. 计算每个有效纹素滤波后的最终 RGBA
    let filtered_chunks: Vec<Vec<[Option<[f32; 4]>; 64]>> =
        indices.par_chunk_map(pool, 64, |_, chunk| {
            let mut results = Vec::with_capacity(chunk.len());
            for &ti in chunk {
                let coord = tile_list[ti].coord;
                let mut out_tile = [None; 64];
                for ty in 0..8usize {
                    for tx in 0..8usize {
                        let center = &tile_list[ti].texels[ty * 8 + tx];
                        if center.weight <= 0.0 {
                            continue;
                        }
                        let cn = Vec3::from_array(center.norm_sum).normalize_or(Vec3::Y);
                        let mut rgb_sum = Vec3::ZERO;
                        let mut w_rgb_sum = 0.0f32;

                        for dy in -1i32..=1 {
                            for dx in -1i32..=1 {
                                let Some(nb) = fetch_texel_in_neighborhood(
                                    tiles,
                                    tile_list,
                                    coord,
                                    tx as i32 + dx,
                                    ty as i32 + dy,
                                ) else {
                                    continue;
                                };
                                if nb.weight <= 0.0 {
                                    continue;
                                }
                                let nn = Vec3::from_array(nb.norm_sum).normalize_or(cn);
                                let n_dot = cn.dot(nn);
                                if n_dot < 0.7 {
                                    continue;
                                }
                                let spatial_w = match dx.abs() + dy.abs() {
                                    0 => 1.0,
                                    1 => 0.55,
                                    _ => 0.3,
                                };
                                let w = spatial_w * n_dot * n_dot;
                                let val = Vec3::from_array(nb.local_direct_over_pi)
                                    + Vec3::from_array(nb.indirect_over_pi);
                                rgb_sum += val * w;
                                w_rgb_sum += w;
                            }
                        }

                        let rgb = if w_rgb_sum > 0.0 {
                            (rgb_sum / w_rgb_sum).clamp(Vec3::ZERO, Vec3::splat(60000.0))
                        } else {
                            Vec3::ZERO
                        };
                        // 保证有效烘焙纹素至少有微小非零底数 (2e-4)，与未初始化的全零 [0,0,0,0] 空槽严格区分
                        out_tile[ty * 8 + tx] = Some([
                            rgb.x.max(2e-4),
                            rgb.y.max(2e-4),
                            rgb.z.max(2e-4),
                            center.sun_vis.clamp(0.0, 1.0),
                        ]);
                    }
                }
                results.push(out_tile);
            }
            results
        });

    let mut flat_idx = 0usize;
    for chunk in filtered_chunks {
        for out_tile in chunk {
            for k in 0..64 {
                if let Some(rgba) = out_tile[k] {
                    tile_list[flat_idx].texels[k].final_rgba = rgba;
                    tile_list[flat_idx].texels[k].filled = true;
                }
            }
            flat_idx += 1;
        }
    }

    // 2. 3 轮形态学膨胀：填充砖块内紧邻有效几何的空纹素，消除硬件双线性采样在边缘处的黑边
    for _pass in 0..3 {
        let dilated_chunks: Vec<Vec<Vec<(usize, [f32; 4])>>> =
            indices.par_chunk_map(pool, 64, |_, chunk| {
                let mut results = Vec::with_capacity(chunk.len());
                for &ti in chunk {
                    let coord = tile_list[ti].coord;
                    let mut updates = Vec::new();
                    for ty in 0..8usize {
                        for tx in 0..8usize {
                            if tile_list[ti].texels[ty * 8 + tx].filled {
                                continue;
                            }
                            let mut acc = [0.0f32; 4];
                            let mut cnt = 0.0f32;
                            for dy in -1i32..=1 {
                                for dx in -1i32..=1 {
                                    if dx == 0 && dy == 0 {
                                        continue;
                                    }
                                    if let Some(nb) = fetch_texel_in_neighborhood(
                                        tiles,
                                        tile_list,
                                        coord,
                                        tx as i32 + dx,
                                        ty as i32 + dy,
                                    ) && nb.filled
                                    {
                                        let w = if dx == 0 || dy == 0 { 1.0 } else { 0.7 };
                                        for c in 0..4 {
                                            acc[c] += nb.final_rgba[c] * w;
                                        }
                                        cnt += w;
                                    }
                                }
                            }
                            if cnt > 0.0 {
                                updates.push((
                                    ty * 8 + tx,
                                    [acc[0] / cnt, acc[1] / cnt, acc[2] / cnt, acc[3] / cnt],
                                ));
                            }
                        }
                    }
                    results.push(updates);
                }
                results
            });

        let mut flat_idx = 0usize;
        for chunk in dilated_chunks {
            for updates in chunk {
                for (k, rgba) in updates {
                    tile_list[flat_idx].texels[k].final_rgba = rgba;
                    tile_list[flat_idx].texels[k].filled = true;
                }
                flat_idx += 1;
            }
        }
        debug_assert_eq!(flat_idx, tile_list.len());
    }

    // 3. 砖块内 100% 填满兜底：若 3 轮膨胀后某砖块内仍有死角空纹素，使用同砖块最近有效纹素填充，
    //    确保已分配砖块的 64 个纹素全部有效，绝不在图集中留下 [0,0,0,0] 黑斑
    for tile in tile_list.iter_mut() {
        let mut any_unfilled = false;
        for k in 0..64 {
            if !tile.texels[k].filled {
                any_unfilled = true;
                break;
            }
        }
        if !any_unfilled {
            continue;
        }

        let snapshot = *tile.texels;
        for ty in 0..8i32 {
            for tx in 0..8i32 {
                let idx = (ty * 8 + tx) as usize;
                if snapshot[idx].filled {
                    continue;
                }
                let mut best_dist_sq = i32::MAX;
                let mut best_rgba = [2e-4, 2e-4, 2e-4, 0.0];
                for sy in 0..8i32 {
                    for sx in 0..8i32 {
                        let s_idx = (sy * 8 + sx) as usize;
                        if snapshot[s_idx].filled {
                            let d2 = (tx - sx) * (tx - sx) + (ty - sy) * (ty - sy);
                            if d2 < best_dist_sq {
                                best_dist_sq = d2;
                                best_rgba = snapshot[s_idx].final_rgba;
                            }
                        }
                    }
                }
                tile.texels[idx].final_rgba = best_rgba;
                tile.texels[idx].filled = true;
            }
        }
    }
}

#[inline]
pub(crate) fn f32_to_f16_bits(val: f32) -> u16 {
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

// ============================================================================
// 主入口：执行离线烘焙并导出 `.vlm.bin`
// ============================================================================

pub fn bake_and_save_vlm(
    collector: VlmSceneCollector,
    scene_lights: &[SceneLight],
    world_hdri: Option<&SceneWorldHdri>,
    out_dir: &Path,
    out_vlm_path: &Path,
    config: VlmBakeConfig,
) -> Result<VlmBakeReport, Box<dyn std::error::Error>> {
    let t_total = Instant::now();
    let tri_count = collector.triangle_count();
    let surface_area = collector.total_surface_area();

    if tri_count == 0 {
        return Err("没有可烘焙 VLM 的静态三角面".into());
    }

    let levels = config.levels.clamp(1, 3);
    let mut effective_texel_size = config.texel_size.max(0.01);

    // 预估 Level 0 + Level 1 砖块数量：若面积过大自动放大 texel_size，避免突破纹理尺寸或烘焙过慢
    let est_tile_area = (effective_texel_size * 8.0) as f64;
    let est_tiles = (surface_area / (est_tile_area * est_tile_area) * 2.2) as usize;
    if est_tiles > config.max_tiles {
        let scale = (est_tiles as f64 / config.max_tiles as f64).sqrt() as f32;
        let adjusted = (effective_texel_size * scale * 100.0).ceil() / 100.0;
        println!(
            "  [VLM] 静态总表面积 {:.1} m² 预计砖块数 ({}) 超出上限 {}，自动调整 texel_size: {:.3}m -> {:.3}m",
            surface_area, est_tiles, config.max_tiles, effective_texel_size, adjusted
        );
        effective_texel_size = adjusted;
    }

    let t_bvh = Instant::now();
    let bvh = collector.build();
    let (sun, local_lights) = prepare_lights(scene_lights, &config);
    let sky = SkySampler::from_world_hdri(world_hdri, out_dir);

    let t_rast = Instant::now();
    let (mut tiles_map, mut tile_list) =
        build_surface_tiles(&bvh.tris, effective_texel_size, levels);
    if tile_list.len() > config.max_tiles && config.max_tiles > 0 {
        let ratio = (tile_list.len() as f64 / config.max_tiles as f64).sqrt() as f32;
        let adjusted = ((effective_texel_size * ratio * 1.08) * 100.0).ceil() / 100.0;
        println!(
            "  [VLM] 实际生成 3D 表面砖块数 ({}) 超出硬件安全上限 {}，自动放大 texel_size: {:.3}m -> {:.3}m 并重新体素化...",
            tile_list.len(),
            config.max_tiles,
            effective_texel_size,
            adjusted
        );
        effective_texel_size = adjusted;
        (tiles_map, tile_list) = build_surface_tiles(&bvh.tris, effective_texel_size, levels);
    }
    let used_tile_count = tile_list.len();
    let active_texels: usize = tile_list
        .iter()
        .map(|t| t.texels.iter().filter(|x| x.weight > 0.0).count())
        .sum();
    println!(
        "  [VLM] BVH ({} 三角面, {:.2}s) | 表面体素化: {} 砖块, {} 有效纹素 ({:.2}s) | 灯光: 1 太阳({:.0} lux) + {} 局部光",
        tri_count,
        t_bvh.elapsed().as_secs_f64(),
        used_tile_count,
        active_texels,
        t_rast.elapsed().as_secs_f64(),
        sun.illuminance,
        local_lights.len()
    );

    if used_tile_count == 0 {
        return Err("VLM 表面光栅化未生成任何有效砖块".into());
    }

    // 提前构建开放寻址哈希表 [key, compact_tile_idx]，供 GPU Bounce 阶段与最终落盘共同复用
    let table_capacity = ((used_tile_count * 2).max(1024)).next_power_of_two() as u32;
    let table_mask = table_capacity - 1;
    let mut tile_table: Vec<[u32; 2]> = vec![[0, 0]; table_capacity as usize];
    let mut max_probe_len = 1u32;

    for (compact_idx, tile) in tile_list.iter().enumerate() {
        let c = tile.coord;
        let (key, start_slot) = vlm_get_tile_key(
            c.tile_u,
            c.tile_v,
            c.w_slice,
            c.dir_id as u32,
            c.level as u32,
            table_mask,
        );
        let mut slot = start_slot;
        let mut probes = 1u32;
        loop {
            if tile_table[slot as usize][0] == 0 {
                tile_table[slot as usize] = [key, compact_idx as u32];
                max_probe_len = max_probe_len.max(probes);
                break;
            }
            slot = (slot + 1) & table_mask;
            probes += 1;
            if probes > table_capacity {
                return Err("VLM 哈希表溢出".into());
            }
        }
    }
    let search_count = max_probe_len.clamp(4, 256);

    let sun_half_angle_rad = config.sun_angle_deg.clamp(0.0, 10.0).to_radians();
    let cos_sun_cone = sun_half_angle_rad.cos();
    let sun_shadow_rays = if sun_half_angle_rad > 1e-4 { 8u32 } else { 1u32 };

    // 优先使用 headless wgpu GPU Compute 执行 Pass 1 + Pass 2；不可用时无缝回退至多线程 CPU
    let mut used_gpu = false;
    if config.use_gpu {
        match crate::vlm_gpu::bake_direct_and_bounces_gpu(
            &bvh,
            &sun,
            &local_lights,
            &sky,
            &tile_table,
            table_mask,
            search_count,
            &mut tile_list,
            effective_texel_size,
            levels,
            sun_shadow_rays,
            cos_sun_cone,
            &config,
        ) {
            Ok(_adapter_name) => {
                used_gpu = true;
            }
            Err(reason) => {
                eprintln!(
                    "  [VLM] GPU 加速不可用（{}），自动回退至多线程 CPU 烘焙...",
                    reason
                );
            }
        }
    }

    if !used_gpu {
        let pool = AsyncComputeTaskPool::get();

        let local_clearance = config.local_light_clearance.max(0.05);
        let local_radius = config.local_light_radius.max(0.0);

        // ------------------------------------------------------------------------
        // Pass 1 (CPU): 直接光烘焙（主平行光软阴影 A + 静态局部光阴影漫反射 RGB）
        // ------------------------------------------------------------------------
        let t_direct = Instant::now();
        let tile_indices: Vec<usize> = (0..used_tile_count).collect();
        let direct_results: Vec<Vec<[Option<([f32; 3], [f32; 3], f32)>; 64]>> =
            tile_indices.par_chunk_map(pool, 32, |_, chunk| {
                let mut out_chunk = Vec::with_capacity(chunk.len());
                for &ti in chunk {
                    let level = tile_list[ti].coord.level as u32;
                    let dir_id = tile_list[ti].coord.dir_id as u32;
                    let texel_size = effective_texel_size * (1u32 << level) as f32;
                    let bias = (texel_size * 0.2).clamp(0.004, 0.03);
                    let t_min = (texel_size * 0.35).max(0.008);
                    let backface_guard = texel_size * 4.0;

                    let mut tile_out = [None; 64];
                    for k in 0..64 {
                        let tx = &tile_list[ti].texels[k];
                        if tx.weight <= 0.0 {
                            continue;
                        }
                        let (p, n) = texel_surface_pos_and_normal(tx, dir_id, texel_size);
                        let origin = p + n * bias;

                        // 1a. 主平行光软阴影可见度 + 直接太阳辐照度 (仅用于反弹)
                        let ndotl_sun = n.dot(sun.dir_to_light);
                        let sun_vis = if ndotl_sun > 0.0 {
                            let (u_off, v_off) = hash_pos_seed(p, 0x1357_9BDF);
                            let mut unoccluded = 0u32;
                            for s in 0..sun_shadow_rays {
                                let u1 = ((s as f32 + 0.5) / sun_shadow_rays as f32 + u_off).fract();
                                let u2 = (radical_inverse_base2(s) + v_off).fract();
                                let ray_dir = if sun_shadow_rays == 1 {
                                    sun.dir_to_light
                                } else {
                                    sample_cone_dir(sun.dir_to_light, cos_sun_cone, u1, u2)
                                };
                                if !bvh.is_occluded(
                                    origin,
                                    ray_dir,
                                    t_min,
                                    1e5,
                                    backface_guard,
                                    false,
                                ) {
                                    unoccluded += 1;
                                }
                            }
                            unoccluded as f32 / sun_shadow_rays as f32
                        } else {
                            0.0
                        };
                        let sun_direct_over_pi =
                            sun.color * (sun.illuminance * ndotl_sun.max(0.0) * sun_vis / PI);

                        // 1b. 静态局部光 (Point / Spot) 直接漫反射 + 球面软阴影遮挡（带灯具避让与自发光穿透）
                        let mut local_direct_over_pi = Vec3::ZERO;
                        for (li, light) in local_lights.iter().enumerate() {
                            let to_light = light.position - p;
                            let dist_sq = to_light.length_squared();
                            if dist_sq >= light.range * light.range || dist_sq < 1e-6 {
                                continue;
                            }
                            let dist = dist_sq.sqrt();
                            let l_dir = to_light / dist;
                            let ndotl = n.dot(l_dir);
                            if ndotl <= 0.0 {
                                continue;
                            }
                            let mut spot_att = 1.0f32;
                            if light.is_spot {
                                let cos_a = (-l_dir).dot(light.forward);
                                if cos_a <= light.cos_outer {
                                    continue;
                                }
                                let denom = (light.cos_inner - light.cos_outer).max(1e-4);
                                let t = ((cos_a - light.cos_outer) / denom).clamp(0.0, 1.0);
                                spot_att = t * t;
                            }

                            let vis = if dist <= local_clearance {
                                1.0
                            } else {
                                let eff_radius =
                                    local_radius.min((dist - local_clearance) * 0.45);
                                let (lt, lb) = build_orthonormal_basis(l_dir);
                                let (u_off, v_off) = hash_pos_seed(
                                    p,
                                    0x517C_C1B7u32 ^ (li as u32).wrapping_mul(0x9E37_79B9),
                                );
                                let mut unoccluded = 0u32;
                                for s in 0..LOCAL_LIGHT_SHADOW_RAYS {
                                    let u1 = ((s as f32 + 0.5) / LOCAL_LIGHT_SHADOW_RAYS as f32
                                        + u_off)
                                        .fract();
                                    let u2 = (radical_inverse_base2(s) + v_off).fract();
                                    let r = u2.sqrt() * eff_radius;
                                    let phi = TAU * u1;
                                    let target = light.position
                                        + lt * (phi.cos() * r)
                                        + lb * (phi.sin() * r);
                                    let ray_vec = target - origin;
                                    let ray_dist = ray_vec.length();
                                    let max_t = (ray_dist - local_clearance).max(t_min);
                                    if max_t <= t_min
                                        || !bvh.is_occluded(
                                            origin,
                                            ray_vec / ray_dist.max(1e-6),
                                            t_min,
                                            max_t,
                                            backface_guard,
                                            true,
                                        )
                                    {
                                        unoccluded += 1;
                                    }
                                }
                                unoccluded as f32 / LOCAL_LIGHT_SHADOW_RAYS as f32
                            };
                            if vis <= 0.0 {
                                continue;
                            }

                            let factor = dist_sq * light.inv_range_sq;
                            let smooth = (1.0 - factor * factor).clamp(0.0, 1.0);
                            let range_att = smooth * smooth / dist_sq.max(0.04);
                            let contrib = light.color
                                * (light.candela * range_att * spot_att * ndotl * vis / PI);
                            local_direct_over_pi += contrib;
                        }

                        tile_out[k] = Some((
                            sun_direct_over_pi.to_array(),
                            local_direct_over_pi.to_array(),
                            sun_vis,
                        ));
                    }
                    out_chunk.push(tile_out);
                }
                out_chunk
            });

        let mut flat_idx = 0usize;
        for chunk in direct_results {
            for tile_out in chunk {
                for k in 0..64 {
                    if let Some((sun_d, loc_d, sun_vis)) = tile_out[k] {
                        tile_list[flat_idx].texels[k].sun_direct_over_pi = sun_d;
                        tile_list[flat_idx].texels[k].local_direct_over_pi = loc_d;
                        tile_list[flat_idx].texels[k].sun_vis = sun_vis;
                    }
                }
                flat_idx += 1;
            }
        }
        println!(
            "  [VLM-CPU] 直接光与太阳软阴影烘焙完成 ({:.2}s)",
            t_direct.elapsed().as_secs_f64()
        );

        // ------------------------------------------------------------------------
        // Pass 2 (CPU): 多跳间接漫反射（基于 SpatialHash 自身 Radiance Cache 自举迭代）
        // ------------------------------------------------------------------------
        let samples = config.samples.max(4);
        let inv_samples = 1.0 / samples as f32;
        let gi_scale = config.gi_scale.max(0.0);

        for bounce in 0..config.bounces {
            let t_bounce = Instant::now();
            let bounce_salt = 0x9E37_79B9u32.wrapping_mul(bounce + 1);
            let bounce_results: Vec<Vec<[Option<[f32; 3]>; 64]>> =
                tile_indices.par_chunk_map(pool, 32, |_, chunk| {
                    let mut out_chunk = Vec::with_capacity(chunk.len());
                    for &ti in chunk {
                        let level = tile_list[ti].coord.level as u32;
                        let dir_id = tile_list[ti].coord.dir_id as u32;
                        let texel_size = effective_texel_size * (1u32 << level) as f32;
                        let bias = (texel_size * 0.2).clamp(0.004, 0.03);
                        let t_min = (texel_size * 0.35).max(0.008);
                        let backface_guard = texel_size * 4.0;

                        let mut tile_out = [None; 64];
                        for k in 0..64 {
                            let tx = &tile_list[ti].texels[k];
                            if tx.weight <= 0.0 {
                                continue;
                            }
                            let (p, n) = texel_surface_pos_and_normal(tx, dir_id, texel_size);
                            let origin = p + n * bias;
                            let (tangent, bitangent) = build_orthonormal_basis(n);
                            let (u_off, v_off) = hash_pos_seed(p, bounce_salt);

                            let mut sum = Vec3::ZERO;
                            for s in 0..samples {
                                let u1 = ((s as f32 + 0.5) * inv_samples + u_off).fract();
                                let u2 = (radical_inverse_base2(s) + v_off).fract() * 0.992;
                                let phi = TAU * u1;
                                let r = u2.sqrt();
                                let z = (1.0 - u2).sqrt();
                                let rd = (tangent * (r * phi.cos())
                                    + bitangent * (r * phi.sin())
                                    + n * z)
                                    .normalize();

                                if let Some(hit) =
                                    bvh.trace_closest(origin, rd, t_min, 1e5, backface_guard)
                                {
                                    let cached_irr = lookup_cached_irradiance(
                                        &tiles_map,
                                        &tile_list,
                                        hit.pos,
                                        hit.normal,
                                        effective_texel_size,
                                        levels,
                                    )
                                    .unwrap_or(Vec3::ZERO);
                                    // 离开命中表面的出射辐射度 L_out = cached_irradiance * albedo + emissive
                                    let leaving = cached_irr * hit.albedo + hit.emissive;
                                    sum += leaving;
                                } else {
                                    // 逃逸到天空盒
                                    sum += sky.sample(rd);
                                }
                            }

                            // 余弦加权重要性采样中 PDF = cos/PI 恰好与积分因子 cos/PI 抵消
                            let irr_over_pi = (sum * inv_samples * gi_scale)
                                .clamp(Vec3::ZERO, Vec3::splat(60000.0));
                            tile_out[k] = Some(irr_over_pi.to_array());
                        }
                        out_chunk.push(tile_out);
                    }
                    out_chunk
                });

            let mut flat_idx = 0usize;
            for chunk in bounce_results {
                for tile_out in chunk {
                    for k in 0..64 {
                        if let Some(irr) = tile_out[k] {
                            tile_list[flat_idx].texels[k].indirect_over_pi = irr;
                        }
                    }
                    flat_idx += 1;
                }
            }
            println!(
                "  [VLM-CPU] 间接反弹 Bounce {}/{} 完成 ({} spp, {:.2}s)",
                bounce + 1,
                config.bounces,
                samples,
                t_bounce.elapsed().as_secs_f64()
            );
        }
    }

    // ------------------------------------------------------------------------
    // Pass 3: 切平面双边降噪 + 空纹素形态学膨胀
    // ------------------------------------------------------------------------
    let t_filter = Instant::now();
    bilateral_filter_and_dilate(&tiles_map, &mut tile_list);

    // ------------------------------------------------------------------------
    // Pass 4: 打包 10×10 物理砖块到 Rgba16Float 2D Atlas（复用已构建的 tile_table）
    // ------------------------------------------------------------------------

    // 打包 10×10 物理砖块到 Rgba16Float 2D Atlas（含跨邻块 1px 无缝边界）
    let tiles_per_row = (used_tile_count as f32).sqrt().ceil().max(1.0) as u32;
    let tiles_per_col = (used_tile_count as u32).div_ceil(tiles_per_row).max(1);
    let atlas_width = tiles_per_row * VLM_PHYSICAL_TILE_SIZE;
    let atlas_height = tiles_per_col * VLM_PHYSICAL_TILE_SIZE;
    let mut atlas_bytes = vec![0u8; (atlas_width as usize) * (atlas_height as usize) * 8];

    for (compact_idx, tile) in tile_list.iter().enumerate() {
        let tile_col = (compact_idx as u32) % tiles_per_row;
        let tile_row = (compact_idx as u32) / tiles_per_row;
        let base_px_x = tile_col * VLM_PHYSICAL_TILE_SIZE;
        let base_px_y = tile_row * VLM_PHYSICAL_TILE_SIZE;
        let coord = tile.coord;

        for py in 0..10i32 {
            for px in 0..10i32 {
                // px, py in 0..10 -> gx, gy in -1..=8
                let gx = px - 1;
                let gy = py - 1;

                let rgba = if let Some(nb) =
                    fetch_texel_in_neighborhood(&tiles_map, &tile_list, coord, gx, gy)
                    && nb.filled
                {
                    nb.final_rgba
                } else {
                    // 无相邻砖块时安全 Clamp 到自身边缘纹素
                    let cx = gx.clamp(0, 7) as usize;
                    let cy = gy.clamp(0, 7) as usize;
                    tile.texels[cy * 8 + cx].final_rgba
                };

                let dst_x = (base_px_x as usize) + (px as usize);
                let dst_y = (base_px_y as usize) + (py as usize);
                let off = (dst_y * (atlas_width as usize) + dst_x) * 8;
                atlas_bytes[off..off + 2].copy_from_slice(&f32_to_f16_bits(rgba[0]).to_le_bytes());
                atlas_bytes[off + 2..off + 4]
                    .copy_from_slice(&f32_to_f16_bits(rgba[1]).to_le_bytes());
                atlas_bytes[off + 4..off + 6]
                    .copy_from_slice(&f32_to_f16_bits(rgba[2]).to_le_bytes());
                atlas_bytes[off + 6..off + 8]
                    .copy_from_slice(&f32_to_f16_bits(rgba[3]).to_le_bytes());
            }
        }
    }

    // ------------------------------------------------------------------------
    // Pass 5: 写入 LZ4 压缩 `.vlm.bin`
    // ------------------------------------------------------------------------
    let mut out_buf = Vec::with_capacity(64 + atlas_bytes.len() / 2);
    out_buf.extend_from_slice(VLM_FILE_MAGIC);
    out_buf.extend_from_slice(&VLM_FILE_VERSION.to_le_bytes());
    out_buf.extend_from_slice(&VLM_TILE_SIZE.to_le_bytes());
    out_buf.extend_from_slice(&table_capacity.to_le_bytes());
    out_buf.extend_from_slice(&(used_tile_count as u32).to_le_bytes());
    out_buf.extend_from_slice(&search_count.to_le_bytes());
    out_buf.extend_from_slice(&tiles_per_row.to_le_bytes());
    out_buf.extend_from_slice(&atlas_width.to_le_bytes());
    out_buf.extend_from_slice(&atlas_height.to_le_bytes());
    out_buf.extend_from_slice(&levels.to_le_bytes());
    out_buf.extend_from_slice(&effective_texel_size.to_le_bytes());

    {
        let mut encoder = FrameEncoder::new(&mut out_buf);
        encoder.write_all(bytemuck::cast_slice(&tile_table))?;
        encoder.write_all(&atlas_bytes)?;
        encoder.finish()?;
    }

    std::fs::write(out_vlm_path, &out_buf)?;
    let file_bytes = out_buf.len();

    println!(
        "  [VLM] 降噪打包与落盘完成 [{}] ({:.2}s, 总耗时 {:.2}s) -> {} ({:.2} MB, Atlas {}x{}, {} 砖块, max_probe={})",
        if used_gpu { "GPU" } else { "CPU" },
        t_filter.elapsed().as_secs_f64(),
        t_total.elapsed().as_secs_f64(),
        out_vlm_path.display(),
        file_bytes as f64 / (1024.0 * 1024.0),
        atlas_width,
        atlas_height,
        used_tile_count,
        search_count,
    );

    Ok(VlmBakeReport {
        effective_texel_size,
        level_count: levels,
        used_tile_count: used_tile_count as u32,
        table_capacity,
        search_count,
        atlas_width,
        atlas_height,
        file_bytes,
        used_gpu,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_tasks::{AsyncComputeTaskPool, TaskPool};

    fn build_canopy_test_scene() -> (VlmSceneCollector, Vec<SceneLight>) {
        // 构造一个 4m x 4m 地面 (y=0) + 悬空 2m x 2m 遮阳顶棚 (y=2, x in [-1, 1], z in [-1, 1])
        let positions = vec![
            // Floor
            [-2.0, 0.0, -2.0],
            [2.0, 0.0, -2.0],
            [2.0, 0.0, 2.0],
            [-2.0, 0.0, 2.0],
            // Roof
            [-1.0, 2.0, -1.0],
            [1.0, 2.0, -1.0],
            [1.0, 2.0, 1.0],
            [-1.0, 2.0, 1.0],
        ];
        let normals = vec![[0.0, 1.0, 0.0]; 8];
        let indices = vec![0, 2, 1, 0, 3, 2, 4, 6, 5, 4, 7, 6];

        let mut collector = VlmSceneCollector::default();
        collector.push_indexed_mesh(
            &positions,
            &normals,
            &indices,
            None,
            VlmMaterialInfo {
                diffuse_albedo: [0.6, 0.6, 0.6],
                emissive: [0.0, 0.0, 0.0],
                double_sided: true,
            },
        );

        // 垂直向下的正午太阳光 (+Y 方向来光，rotation 使 local +Z 指向 +Y)
        let sun_rot = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
        let lights = vec![
            SceneLight {
                name: "Sun".to_string(),
                kind: "Directional".to_string(),
                color: [1.0, 1.0, 1.0],
                intensity: 20000.0,
                range: None,
                inner_cone_angle: None,
                outer_cone_angle: None,
                translation: [0.0, 10.0, 0.0],
                rotation: sun_rot.to_array(),
            },
            SceneLight {
                name: "WarmLamp".to_string(),
                kind: "Point".to_string(),
                color: [1.0, 0.7, 0.4],
                intensity: 300.0,
                range: Some(6.0),
                inner_cone_angle: None,
                outer_cone_angle: None,
                translation: [0.0, 1.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
        ];
        (collector, lights)
    }

    fn read_vlm_sample(path: &Path, p: Vec3, n: Vec3) -> [f32; 4] {
        let raw = std::fs::read(path).expect("read .vlm.bin");
        assert_eq!(&raw[0..8], VLM_FILE_MAGIC);
        let table_cap = u32::from_le_bytes(raw[16..20].try_into().unwrap());
        let search_count = u32::from_le_bytes(raw[24..28].try_into().unwrap());
        let tiles_per_row = u32::from_le_bytes(raw[28..32].try_into().unwrap());
        let atlas_w = u32::from_le_bytes(raw[32..36].try_into().unwrap());
        let atlas_h = u32::from_le_bytes(raw[36..40].try_into().unwrap());
        let texel_size = f32::from_le_bytes(raw[44..48].try_into().unwrap());

        use std::io::Read;
        let mut dec = lz4_flex::frame::FrameDecoder::new(&raw[48..]);
        let mut table = vec![[0u32; 2]; table_cap as usize];
        dec.read_exact(bytemuck::cast_slice_mut(&mut table)).unwrap();
        let mut atlas = vec![0u8; (atlas_w * atlas_h * 8) as usize];
        dec.read_exact(&mut atlas).unwrap();

        let f16_to_f32 = |bits: u16| -> f32 {
            let s = ((bits >> 15) & 1) as u32;
            let e = ((bits >> 10) & 0x1f) as u32;
            let m = (bits & 0x3ff) as u32;
            if e == 0 {
                if m == 0 {
                    f32::from_bits(s << 31)
                } else {
                    (if s != 0 { -1.0 } else { 1.0 }) * (m as f32) * 2.0f32.powi(-24)
                }
            } else if e == 31 {
                f32::NAN
            } else {
                f32::from_bits((s << 31) | ((e + 112) << 23) | (m << 13))
            }
        };

        let dir_id = dir_id_from_normal(n);
        let shifted = p + dir_vec_from_id(dir_id) * (texel_size * 0.15);
        let (uv_w, w_w) = project_to_tangent_and_w(shifted, dir_id);
        let w_slice = (w_w / (texel_size * VLM_W_STEP_FACTOR)).floor() as i32;
        let tile_world = texel_size * 8.0;
        let tu = (uv_w.x / tile_world).floor() as i32;
        let tv = (uv_w.y / tile_world).floor() as i32;
        let mask = table_cap - 1;
        let (key, mut idx) = vlm_get_tile_key(tu, tv, w_slice, dir_id, 0, mask);
        let mut tid = None;
        for _ in 0..search_count {
            if table[idx as usize][0] == key {
                tid = Some(table[idx as usize][1]);
                break;
            }
            if table[idx as usize][0] == 0 {
                break;
            }
            idx = (idx + 1) & mask;
        }
        let tid = tid.expect("point should hit a baked VLM tile");
        let lx = ((uv_w.x - tu as f32 * tile_world) / texel_size)
            .floor()
            .clamp(0.0, 7.0) as u32;
        let ly = ((uv_w.y - tv as f32 * tile_world) / texel_size)
            .floor()
            .clamp(0.0, 7.0) as u32;
        let px = (tid % tiles_per_row) * 10 + 1 + lx;
        let py = (tid / tiles_per_row) * 10 + 1 + ly;
        let off = ((py * atlas_w + px) * 8) as usize;
        [
            f16_to_f32(u16::from_le_bytes([atlas[off], atlas[off + 1]])),
            f16_to_f32(u16::from_le_bytes([atlas[off + 2], atlas[off + 3]])),
            f16_to_f32(u16::from_le_bytes([atlas[off + 4], atlas[off + 5]])),
            f16_to_f32(u16::from_le_bytes([atlas[off + 6], atlas[off + 7]])),
        ]
    }

    #[test]
    fn test_vlm_sun_shadow_and_bounce_bake_cpu() {
        AsyncComputeTaskPool::get_or_init(TaskPool::default);
        let (collector, lights) = build_canopy_test_scene();

        let tmp = std::env::temp_dir().join("test_makemeshlet_vlm_cpu.vlm.bin");
        let report = bake_and_save_vlm(
            collector,
            &lights,
            None,
            std::env::temp_dir().as_path(),
            &tmp,
            VlmBakeConfig {
                enabled: true,
                texel_size: 0.25,
                levels: 1,
                samples: 16,
                bounces: 2,
                sun_angle_deg: 0.5,
                use_gpu: false,
                ..Default::default()
            },
        )
        .expect("CPU VLM bake should succeed");

        assert!(!report.used_gpu);
        assert!(report.used_tile_count > 0);
        assert!(report.atlas_width >= 10 && report.atlas_height >= 10);

        let shaded = read_vlm_sample(&tmp, Vec3::new(0.0, 0.0, 0.0), Vec3::Y);
        let sunlit = read_vlm_sample(&tmp, Vec3::new(1.6, 0.0, 0.0), Vec3::Y);
        assert!(
            shaded[3] < 0.05,
            "顶棚正下方应当处于太阳阴影中 (sun_vis < 0.05)，实际: {}",
            shaded[3]
        );
        assert!(
            sunlit[3] > 0.95,
            "顶棚外开阔地面应当被太阳直射 (sun_vis > 0.95)，实际: {}",
            sunlit[3]
        );
        assert!(
            shaded[0] > 1.0 && sunlit[0] > 1.0,
            "阴影区与直射区均应具有非零点光+天光/反弹 GI 辐照度: shaded={:?}, sunlit={:?}",
            shaded,
            sunlit
        );

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_vlm_gpu_matches_cpu() {
        AsyncComputeTaskPool::get_or_init(TaskPool::default);
        let (col_cpu, lights_cpu) = build_canopy_test_scene();
        let (col_gpu, lights_gpu) = build_canopy_test_scene();

        let tmp_cpu = std::env::temp_dir().join("test_makemeshlet_vlm_cmp_cpu.vlm.bin");
        let tmp_gpu = std::env::temp_dir().join("test_makemeshlet_vlm_cmp_gpu.vlm.bin");

        let base_cfg = VlmBakeConfig {
            enabled: true,
            texel_size: 0.25,
            levels: 1,
            samples: 32,
            bounces: 2,
            sun_angle_deg: 0.5,
            use_gpu: false,
            ..Default::default()
        };

        let rep_cpu = bake_and_save_vlm(
            col_cpu,
            &lights_cpu,
            None,
            std::env::temp_dir().as_path(),
            &tmp_cpu,
            base_cfg,
        )
        .expect("CPU bake should succeed");

        let rep_gpu = bake_and_save_vlm(
            col_gpu,
            &lights_gpu,
            None,
            std::env::temp_dir().as_path(),
            &tmp_gpu,
            VlmBakeConfig {
                use_gpu: true,
                ..base_cfg
            },
        )
        .expect("GPU (or fallback) bake should succeed");

        assert_eq!(rep_cpu.used_tile_count, rep_gpu.used_tile_count);
        println!(
            "test_vlm_gpu_matches_cpu: used_gpu={}, tiles={}",
            rep_gpu.used_gpu, rep_gpu.used_tile_count
        );

        let probes = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.6, 0.0, 0.0),
            Vec3::new(-1.5, 0.0, 1.5),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        for p in probes {
            let c = read_vlm_sample(&tmp_cpu, p, Vec3::Y);
            let g = read_vlm_sample(&tmp_gpu, p, Vec3::Y);
            for ch in 0..4 {
                let diff = (c[ch] - g[ch]).abs();
                let tol = c[ch].abs() * 0.03 + 0.05;
                assert!(
                    diff <= tol,
                    "GPU 与 CPU 烘焙结果差异过大 @ {:?} ch={}: cpu={:?}, gpu={:?} (diff={}, tol={})",
                    p,
                    ch,
                    c,
                    g,
                    diff,
                    tol
                );
            }
        }

        let _ = std::fs::remove_file(&tmp_cpu);
        let _ = std::fs::remove_file(&tmp_gpu);
    }

    #[test]
    fn test_dilation_multi_chunk_no_cross_tile_corruption() {
        AsyncComputeTaskPool::get_or_init(TaskPool::default);

        // 构造 150 个彼此空间隔离的砖块（跨越多个 par_chunk_map(64) 分块）：
        // - 砖块 0：室内暗区，仅填充 (0,0) 纹素，期望膨胀后全砖块保持 [0.1, 0.2, 0.3, 0.0]
        // - 砖块 1..150：室外强光区，仅填充 (7,7) 纹素，期望各自独立膨胀为 [10.0, 20.0, 30.0, 1.0]
        let mut tiles_map = HashMap::new();
        let mut tile_list = Vec::with_capacity(150);

        for i in 0..150i32 {
            let coord = TileCoord {
                tile_u: i * 10,
                tile_v: 0,
                w_slice: 0,
                dir_id: 2,
                level: 0,
            };
            let mut tile = TileData::new(coord);
            if i == 0 {
                let tx = &mut tile.texels[0];
                tx.weight = 1.0;
                tx.norm_sum = [0.0, 1.0, 0.0];
                tx.indirect_over_pi = [0.1, 0.2, 0.3];
                tx.sun_vis = 0.0;
            } else {
                let tx = &mut tile.texels[63];
                tx.weight = 1.0;
                tx.norm_sum = [0.0, 1.0, 0.0];
                tx.indirect_over_pi = [10.0, 20.0, 30.0];
                tx.sun_vis = 1.0;
            }
            tiles_map.insert(coord, tile_list.len());
            tile_list.push(tile);
        }

        bilateral_filter_and_dilate(&tiles_map, &mut tile_list);

        // 验证砖块 0 未被其他 63+ 个亮区砖块的膨胀数据串块污染
        for k in 0..64 {
            let rgba = tile_list[0].texels[k].final_rgba;
            assert!(
                (rgba[0] - 0.1).abs() < 1e-3
                    && (rgba[1] - 0.2).abs() < 1e-3
                    && (rgba[2] - 0.3).abs() < 1e-3
                    && rgba[3].abs() < 1e-3,
                "砖块 0 纹素 #{k} 被跨砖块污染: {rgba:?}"
            );
        }

        // 验证后续分块（>= 64）中的砖块均正确完成自身膨胀
        for ti in [1usize, 63, 64, 127, 149] {
            for k in 0..64 {
                let rgba = tile_list[ti].texels[k].final_rgba;
                assert!(
                    (rgba[0] - 10.0).abs() < 1e-2
                        && (rgba[1] - 20.0).abs() < 1e-2
                        && (rgba[2] - 30.0).abs() < 1e-2
                        && (rgba[3] - 1.0).abs() < 1e-3,
                    "砖块 #{ti} 纹素 #{k} 膨胀异常: {rgba:?}"
                );
            }
        }
    }
}
