//! 离线静态逐顶点环境光遮蔽（Per-Vertex Static AO）光追烘焙模块
//!
//! 为 `makemeshlet` 提供基于 CPU BVH + Quasi-Monte Carlo（Hammersley 余弦加权半球采样）的高速静态 AO 烘焙：
//! - `--scene` 模式下构建全局静态场景 BVH，让地面/墙体/家具之间产生跨批次、跨分块的接触遮蔽；
//! - 独立/自动实例化/单模型模式下基于局部几何 BVH 烘焙自身缝隙遮蔽；
//! - 输出 `[0.0, 1.0]` 逐顶点标量 AO，由 `MeshletMesh::from_mesh_with_ao` 紧打包进 `vertex_normals` 的高 8 位。

use std::f32::consts::TAU;

use bevy_math::{Mat4, Vec3};
use bevy_tasks::{AsyncComputeTaskPool, ParallelSlice};

/// 静态 AO 烘焙参数
#[derive(Debug, Clone, Copy)]
pub struct AoBakeConfig {
    pub enabled: bool,
    pub samples: u32,
    pub radius: f32,
    pub power: f32,
    pub bias: Option<f32>,
}

impl AoBakeConfig {
    pub fn effective_bias(&self) -> f32 {
        self.bias
            .unwrap_or_else(|| (self.radius * 0.0025).clamp(0.0005, 0.01))
    }

    pub fn is_active(&self) -> bool {
        self.enabled && self.samples > 0 && self.radius > 1e-4
    }
}

#[derive(Clone, Copy)]
struct AoTri {
    p0: [f32; 3],
    e1: [f32; 3],
    e2: [f32; 3],
    geom_normal: [f32; 3],
}

#[derive(Clone, Copy)]
struct BuildTriMeta {
    tri: AoTri,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    centroid: [f32; 3],
}

#[derive(Clone, Copy)]
struct AoBvhNode {
    aabb_min: [f32; 3],
    left_or_first: u32,
    aabb_max: [f32; 3],
    right_child: u32,
    tri_count: u32,
}

/// 面向静态 AO 射线查询的扁平 CPU BVH 加速结构
pub struct AoBvh {
    nodes: Vec<AoBvhNode>,
    tris: Vec<AoTri>,
}

/// 静态遮蔽三角形收集器
#[derive(Default)]
pub struct AoOccluderCollector {
    metas: Vec<BuildTriMeta>,
}

impl AoOccluderCollector {
    pub fn push_indexed_mesh(
        &mut self,
        positions: &[[f32; 3]],
        indices: &[u32],
        world: Option<&Mat4>,
    ) {
        self.metas.reserve(indices.len() / 3);
        for tri in indices.chunks_exact(3) {
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
                continue;
            }
            let (p0, p1, p2) = if let Some(mat) = world {
                (
                    mat.transform_point3(Vec3::from_array(positions[i0])),
                    mat.transform_point3(Vec3::from_array(positions[i1])),
                    mat.transform_point3(Vec3::from_array(positions[i2])),
                )
            } else {
                (
                    Vec3::from_array(positions[i0]),
                    Vec3::from_array(positions[i1]),
                    Vec3::from_array(positions[i2]),
                )
            };
            self.push_triangle(p0, p1, p2);
        }
    }

    pub fn push_triangle(&mut self, p0: Vec3, p1: Vec3, p2: Vec3) {
        if !p0.is_finite() || !p1.is_finite() || !p2.is_finite() {
            return;
        }
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let cross = e1.cross(e2);
        let len_sq = cross.length_squared();
        if len_sq <= 1e-14 || !len_sq.is_finite() {
            return;
        }
        let geom_normal = (cross / len_sq.sqrt()).to_array();
        let aabb_min = p0.min(p1).min(p2).to_array();
        let aabb_max = p0.max(p1).max(p2).to_array();
        let centroid = ((p0 + p1 + p2) * (1.0 / 3.0)).to_array();

        self.metas.push(BuildTriMeta {
            tri: AoTri {
                p0: p0.to_array(),
                e1: e1.to_array(),
                e2: e2.to_array(),
                geom_normal,
            },
            aabb_min,
            aabb_max,
            centroid,
        });
    }

    pub fn triangle_count(&self) -> usize {
        self.metas.len()
    }

    pub fn build(self) -> AoBvh {
        AoBvh::build_from_metas(self.metas)
    }
}

impl AoBvh {
    pub fn from_indexed_mesh(positions: &[[f32; 3]], indices: &[u32]) -> Self {
        let mut collector = AoOccluderCollector::default();
        collector.push_indexed_mesh(positions, indices, None);
        collector.build()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() || self.tris.is_empty()
    }

    fn build_from_metas(mut metas: Vec<BuildTriMeta>) -> Self {
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

        let tris: Vec<AoTri> = order.into_iter().map(|idx| metas[idx].tri).collect();
        metas.clear();

        Self { nodes, tris }
    }

    fn build_node(
        metas: &[BuildTriMeta],
        order: &mut [usize],
        start: usize,
        end: usize,
        depth: u32,
        nodes: &mut Vec<AoBvhNode>,
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
        nodes.push(AoBvhNode {
            aabb_min,
            left_or_first: start as u32,
            aabb_max,
            right_child: 0,
            tri_count: 0,
        });

        if tri_count <= 4 || depth >= 48 {
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
        let slice = &mut order[start..end];
        let local_mid = mid - start;
        slice.select_nth_unstable_by(local_mid, |&a, &b| {
            metas[a].centroid[axis].total_cmp(&metas[b].centroid[axis])
        });

        let left_child = Self::build_node(metas, order, start, mid, depth + 1, nodes);
        let right_child = Self::build_node(metas, order, mid, end, depth + 1, nodes);

        nodes[node_idx as usize].left_or_first = left_child;
        nodes[node_idx as usize].right_child = right_child;
        node_idx
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
        if enter <= exit {
            Some(enter)
        } else {
            None
        }
    }

    #[inline]
    fn intersect_tri(
        origin: Vec3,
        dir: Vec3,
        tri: &AoTri,
        t_min: f32,
        t_max: f32,
        backface_guard_dist: f32,
    ) -> Option<f32> {
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
        // 在极近距离内忽略背面自交（防止平滑法线在斜面上导致射线从背面击中自身邻接三角面），
        // 超过保护距离后按双面遮蔽处理，防止单面薄墙背面漏光。
        if t < backface_guard_dist && dir.dot(Vec3::from_array(tri.geom_normal)) > 0.0 {
            return None;
        }
        Some(t)
    }

    /// 追踪 `[t_min, max_dist)` 内的最近交点距离
    pub fn trace_closest(
        &self,
        origin: Vec3,
        dir: Vec3,
        t_min: f32,
        max_dist: f32,
        backface_guard_dist: f32,
    ) -> Option<f32> {
        if self.is_empty() {
            return None;
        }

        let inv_dir = Vec3::new(
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
        );

        let root = &self.nodes[0];
        if Self::intersect_aabb(origin, inv_dir, root.aabb_min, root.aabb_max, t_min, max_dist)
            .is_none()
        {
            return None;
        }

        let mut best_t = max_dist;
        let mut hit = false;
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
                for tri in &self.tris[first..end] {
                    if let Some(t) =
                        Self::intersect_tri(origin, dir, tri, t_min, best_t, backface_guard_dist)
                    {
                        best_t = t;
                        hit = true;
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
                    (Some(tl), Some(tr)) => {
                        if sp + 2 <= stack.len() {
                            if tl <= tr {
                                stack[sp] = r_idx;
                                stack[sp + 1] = l_idx;
                            } else {
                                stack[sp] = l_idx;
                                stack[sp + 1] = r_idx;
                            }
                            sp += 2;
                        }
                    }
                    (Some(_), None) => {
                        if sp < stack.len() {
                            stack[sp] = l_idx;
                            sp += 1;
                        }
                    }
                    (None, Some(_)) => {
                        if sp < stack.len() {
                            stack[sp] = r_idx;
                            sp += 1;
                        }
                    }
                    (None, None) => {}
                }
            }
        }

        if hit {
            Some(best_t)
        } else {
            None
        }
    }
}

#[inline]
fn radical_inverse_base2(mut bits: u32) -> f32 {
    bits = bits.reverse_bits();
    bits as f32 * 2.3283064e-10
}

#[inline]
fn hash_position_seed(p: Vec3) -> (f32, f32) {
    // 量化至毫米级网格作为旋转种子，保证 UV 接缝处重合顶点得到完全一致的采样方向，消除接缝色差
    let qx = (p.x * 1000.0).round() as i32 as u32;
    let qy = (p.y * 1000.0).round() as i32 as u32;
    let qz = (p.z * 1000.0).round() as i32 as u32;

    let mut h1 = qx
        .wrapping_mul(0x9E37_79B1)
        ^ qy.wrapping_mul(0x85EB_CA77)
        ^ qz.wrapping_mul(0xC2B2_AE3D);
    h1 ^= h1 >> 16;
    h1 = h1.wrapping_mul(0x7FEB_352D);
    h1 ^= h1 >> 15;

    let mut h2 = h1.wrapping_mul(0x846C_A68B) ^ 0xA5A5_5A5A;
    h2 ^= h2 >> 16;

    (
        (h1 & 0x00FF_FFFF) as f32 / 16_777_216.0,
        (h2 & 0x00FF_FFFF) as f32 / 16_777_216.0,
    )
}

#[inline]
fn build_orthonormal_basis(n: Vec3) -> (Vec3, Vec3) {
    // Frisvad 无分支正交基构建
    if n.z < -0.9999999 {
        (Vec3::new(0.0, -1.0, 0.0), Vec3::new(-1.0, 0.0, 0.0))
    } else {
        let a = 1.0 / (1.0 + n.z);
        let b = -n.x * n.y * a;
        let t = Vec3::new(1.0 - n.x * n.x * a, b, -n.x);
        let bit = Vec3::new(b, 1.0 - n.y * n.y * a, -n.y);
        (t, bit)
    }
}

/// 为一批顶点并行烘焙静态逐顶点 AO（返回与 `positions` 等长的 `[0.0, 1.0]` 数组）
pub fn bake_vertex_ao(
    bvh: &AoBvh,
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    config: AoBakeConfig,
) -> Vec<f32> {
    if !config.is_active() || bvh.is_empty() || positions.is_empty() {
        return vec![1.0; positions.len()];
    }

    let n_verts = positions.len().min(normals.len());
    let bias = config.effective_bias();
    let t_min = (bias * 0.05).max(1e-4);
    let backface_guard = bias * 3.0;
    let radius = config.radius;
    let inv_radius = 1.0 / radius;
    let samples = config.samples;
    let inv_samples = 1.0 / samples as f32;
    let power = config.power.max(0.01);

    let chunk_size = 1024usize;
    let indices: Vec<usize> = (0..n_verts).collect();
    let pool = AsyncComputeTaskPool::get();
    let chunk_results: Vec<Vec<f32>> = indices.par_chunk_map(pool, chunk_size, |_, chunk| {
        let mut out = Vec::with_capacity(chunk.len());
        for &vi in chunk {
            let p = Vec3::from_array(positions[vi]);
            let n = Vec3::from_array(normals[vi]).normalize_or(Vec3::Y);
            if !p.is_finite() || !n.is_finite() {
                out.push(1.0);
                continue;
            }

            let origin = p + n * bias;
            let (tangent, bitangent) = build_orthonormal_basis(n);
            let (u_off, v_off) = hash_position_seed(p);

            let mut occ_sum = 0.0f32;
            for s in 0..samples {
                let u1 = ((s as f32 + 0.5) * inv_samples + u_off).fract();
                // 限制 u2 上界在 0.992，避免极掠射角（< 5°）与邻接三角形产生自交噪点
                let u2 = (radical_inverse_base2(s) + v_off).fract() * 0.992;

                let phi = TAU * u1;
                let r = u2.sqrt();
                let z = (1.0 - u2).sqrt();
                let dir = (tangent * (r * phi.cos()) + bitangent * (r * phi.sin()) + n * z)
                    .normalize();

                if let Some(hit_t) = bvh.trace_closest(origin, dir, t_min, radius, backface_guard) {
                    let k = (1.0 - hit_t * inv_radius).clamp(0.0, 1.0);
                    occ_sum += k * k;
                }
            }

            let ao = (1.0 - occ_sum * inv_samples).clamp(0.0, 1.0).powf(power);
            out.push(ao);
        }
        out
    });

    let mut result = Vec::with_capacity(positions.len());
    for chunk in chunk_results {
        result.extend(chunk);
    }
    result.resize(positions.len(), 1.0);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_tasks::{AsyncComputeTaskPool, TaskPool};

    #[test]
    fn test_ao_corner_darkening() {
        AsyncComputeTaskPool::get_or_init(TaskPool::default);

        // 构造 L 形 90 度墙角：地面 y=0 (x in [0, 4], z in [-2, 2]) + 墙面 x=0 (y in [0, 4], z in [-2, 2])
        let positions = vec![
            [0.0, 0.0, -2.0],
            [4.0, 0.0, -2.0],
            [4.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, -2.0],
            [0.0, 4.0, -2.0],
            [0.0, 4.0, 2.0],
            [0.0, 0.0, 2.0],
        ];
        let indices = vec![0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7];

        // 在墙角内侧 (0.05, 0.0, 0.0) 与远离墙角处 (3.5, 0.0, 0.0) 采样对比
        let bvh = AoBvh::from_indexed_mesh(&positions, &indices);
        let test_pos = vec![[0.05, 0.0, 0.0], [3.5, 0.0, 0.0]];
        let test_norm = vec![[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let ao = bake_vertex_ao(
            &bvh,
            &test_pos,
            &test_norm,
            AoBakeConfig {
                enabled: true,
                samples: 64,
                radius: 1.5,
                power: 1.0,
                bias: None,
            },
        );

        assert!(
            ao[0] < 0.65,
            "墙角内侧 AO 应当明显受遮蔽 (< 0.65)，实际: {}",
            ao[0]
        );
        assert!(
            ao[1] > 0.98,
            "远离墙角的开阔地面 AO 应当接近 1.0 (> 0.98)，实际: {}",
            ao[1]
        );
    }
}
