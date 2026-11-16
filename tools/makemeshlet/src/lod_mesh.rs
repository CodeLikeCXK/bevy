//! 标准 `Mesh3d` 多级 LOD 生成与 `.mesh3d_lod` 二进制资产导出模块
//!
//! 用途：
//! 针对不适合走 Meshlet 管线的物体（如植物树叶 `AlphaMode::Mask`、玻璃/酒瓶 `AlphaMode::Blend`），
//! 在离线阶段使用 `meshopt` 对三角形索引执行边界锁定边折叠简化，生成共享同一套顶点数据的多级 LOD，
//! 并使用 LZ4 流式压缩保存为 `.mesh3d_lod` 文件，供渲染器端配合 `VisibilityRange` 加载。
//!
//! `.mesh3d_lod` 文件二进制布局（Little-Endian）：
//! 1. Magic (8 bytes): `b"BM3DLOD1"`
//! 2. Version (u32): `1`
//! 3. VertexCount (u32): 顶点数量 N
//! 4. LodCount (u32): LOD 级别数量 L
//! 5. LZ4 Frame 压缩体，解压后依次包含：
//!    - positions: N * [f32; 3]
//!    - normals:   N * [f32; 3]
//!    - uvs:       N * [f32; 2]
//!    - 重复 L 次：
//!      - index_count (u32): 当前 LOD 的索引数量 K_i
//!      - indices: K_i * u32

use std::io::Write;
use std::path::Path;

use lz4_flex::frame::FrameEncoder;
use meshopt::{simplify_with_attributes_and_locks, SimplifyOptions, VertexDataAdapter};

const MESH3D_LOD_MAGIC: &[u8; 8] = b"BM3DLOD1";
const MESH3D_LOD_VERSION: u32 = 1;

/// 一份完整的 Mesh3d 多级 LOD 资产数据
pub struct Mesh3dLodAsset {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    /// 各级 LOD 的三角形索引（按从精细到粗糙排序：LOD0, LOD1, ...）
    pub lod_indices: Vec<Vec<u32>>,
}

/// 解析逗号分隔的 LOD 简化比例字符串（例如 `"1.0,0.4,0.15"`）
pub fn parse_lod_ratios(raw: Option<&str>, fallback: &[f32]) -> Vec<f32> {
    let Some(s) = raw else {
        return fallback.to_vec();
    };
    let parsed: Vec<f32> = s
        .split(',')
        .filter_map(|part| part.trim().parse::<f32>().ok())
        .map(|v| v.clamp(0.02, 1.0))
        .collect();
    if parsed.is_empty() {
        fallback.to_vec()
    } else {
        parsed
    }
}

/// 解析逗号分隔的 LOD 切换/剔除距离字符串（例如 `"18,45,120"`）
pub fn parse_lod_distances(raw: Option<&str>, fallback: &[f32]) -> Vec<f32> {
    let Some(s) = raw else {
        return fallback.to_vec();
    };
    let parsed: Vec<f32> = s
        .split(',')
        .filter_map(|part| part.trim().parse::<f32>().ok())
        .filter(|v| *v > 0.0)
        .collect();
    if parsed.is_empty() {
        fallback.to_vec()
    } else {
        parsed
    }
}

/// 使用 `meshopt` 对输入网格按指定比例序列生成多级 LOD 索引缓冲
pub fn build_mesh3d_lod_asset(
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    base_indices: &[u32],
    ratios: &[f32],
) -> Result<Mesh3dLodAsset, String> {
    let vert_count = positions.len();
    if vert_count == 0 || base_indices.len() < 3 {
        return Err("网格顶点或三角形数量为空，无法生成 LOD".to_string());
    }

    // 构造 meshopt 位置适配器（stride = 12 bytes, offset = 0）
    let pos_bytes: &[u8] = bytemuck::cast_slice(&positions);
    let vertex_adapter = VertexDataAdapter::new(pos_bytes, size_of::<[f32; 3]>(), 0)
        .map_err(|e| format!("构造 VertexDataAdapter 失败: {e:?}"))?;

    // 将法线作为属性传入，避免简化过程破坏曲面明暗过渡
    let normal_floats: &[f32] = bytemuck::cast_slice(&normals);
    let normal_stride = size_of::<[f32; 3]>();
    let vertex_locks = vec![false; vert_count];

    let mut lod_indices = Vec::with_capacity(ratios.len());
    for &ratio in ratios {
        if ratio >= 0.999 {
            // LOD0 直接保留原始精细索引
            lod_indices.push(base_indices.to_vec());
            continue;
        }

        // 目标索引数必须是 3 的整数倍，且至少保留 1 个三角形
        let raw_target = ((base_indices.len() as f32) * ratio).round() as usize;
        let target_index_count = (raw_target / 3).max(1) * 3;

        // 锁定网格边界（LockBorder），避免植物叶片或开放边缘在简化后出现明显破洞
        let mut simplified = simplify_with_attributes_and_locks(
            base_indices,
            &vertex_adapter,
            normal_floats,
            &[0.5, 0.5, 0.5],
            normal_stride,
            &vertex_locks,
            target_index_count,
            f32::MAX,
            SimplifyOptions::LockBorder,
            None,
        );

        // 若极少数极端拓扑导致简化返回空，回退保留原始索引避免运行时缺面
        if simplified.len() < 3 {
            simplified = base_indices.to_vec();
        }

        lod_indices.push(simplified);
    }

    Ok(Mesh3dLodAsset {
        positions,
        normals,
        uvs,
        lod_indices,
    })
}

/// 将 `Mesh3dLodAsset` 写入 LZ4 压缩的 `.mesh3d_lod` 文件
pub fn save_mesh3d_lod_asset(
    asset: &Mesh3dLodAsset,
    out_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut raw_header_and_body = Vec::new();
    raw_header_and_body.extend_from_slice(MESH3D_LOD_MAGIC);
    raw_header_and_body.extend_from_slice(&MESH3D_LOD_VERSION.to_le_bytes());
    raw_header_and_body.extend_from_slice(&(asset.positions.len() as u32).to_le_bytes());
    raw_header_and_body.extend_from_slice(&(asset.lod_indices.len() as u32).to_le_bytes());

    // 压缩几何顶点与各级 LOD 索引数据
    let mut encoder = FrameEncoder::new(Vec::new());
    encoder.write_all(bytemuck::cast_slice(&asset.positions))?;
    encoder.write_all(bytemuck::cast_slice(&asset.normals))?;
    encoder.write_all(bytemuck::cast_slice(&asset.uvs))?;

    for indices in &asset.lod_indices {
        let count = indices.len() as u32;
        encoder.write_all(&count.to_le_bytes())?;
        encoder.write_all(bytemuck::cast_slice(indices))?;
    }

    let compressed = encoder.finish()?;
    raw_header_and_body.extend_from_slice(&compressed);
    std::fs::write(out_path, raw_header_and_body)?;
    Ok(())
}
