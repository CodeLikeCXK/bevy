//! `.mesh3d_lod` 二进制多级 LOD 资产加载器
//!
//! 对应 `tools/makemeshlet/src/lod_mesh.rs` 导出的资产格式：
//! 1. Magic: `b"BM3DLOD1"` (8 字节)
//! 2. Version: `u32` (1)
//! 3. VertexCount: `u32` (N)
//! 4. LodCount: `u32` (L)
//! 5. LZ4 Frame 压缩流：依次存放 positions / normals / uvs 以及各级 LOD 索引。
//!
//! 加载时通过 `load_context.add_labeled_asset("lod_0", ...)` 将每级 LOD 注册为独立的 Bevy `Mesh`，
//! 供 `StreamingPlugin` 配合 `VisibilityRange` 自动挂载与距离切换。

use std::io::Read;

use bevy::{
    asset::{io::Reader, AssetLoader, LoadContext, RenderAssetUsages},
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};
use lz4_flex::frame::FrameDecoder;
use thiserror::Error;

const MESH3D_LOD_MAGIC: &[u8; 8] = b"BM3DLOD1";
const MESH3D_LOD_VERSION: u32 = 1;

/// 一份包含多个简化级别（LOD0, LOD1, ...）的 Mesh 句柄集合资产
#[derive(Asset, TypePath, Debug, Clone)]
pub struct Mesh3dLodAsset {
    pub lod_meshes: Vec<Handle<Mesh>>,
    /// 各级 LOD 的三角形数量（在加载期缓存，避免 RenderWorld 提取后无法访问 Mesh CPU 缓冲区）
    pub lod_tri_counts: Vec<usize>,
}

#[derive(Default, TypePath)]
pub struct Mesh3dLodLoader;

#[derive(Debug, Error)]
pub enum Mesh3dLodLoaderError {
    #[error("读取 .mesh3d_lod 文件失败: {0}")]
    Io(#[from] std::io::Error),
    #[error("无效的 .mesh3d_lod 魔数（非 BM3DLOD1 格式）")]
    InvalidMagic,
    #[error("不支持的 .mesh3d_lod 版本: {0}")]
    UnsupportedVersion(u32),
    #[error("二进制数据损坏或长度不足")]
    CorruptedData,
}

impl AssetLoader for Mesh3dLodLoader {
    type Asset = Mesh3dLodAsset;
    type Settings = ();
    type Error = Mesh3dLodLoaderError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;

        // 1. 解析固定头部（20 字节）
        if bytes.len() < 20 {
            return Err(Mesh3dLodLoaderError::CorruptedData);
        }
        if &bytes[0..8] != MESH3D_LOD_MAGIC {
            return Err(Mesh3dLodLoaderError::InvalidMagic);
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != MESH3D_LOD_VERSION {
            return Err(Mesh3dLodLoaderError::UnsupportedVersion(version));
        }
        let vertex_count = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let lod_count = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;

        // 2. LZ4 解压共享顶点数据与各级 LOD 索引
        let mut decoder = FrameDecoder::new(&bytes[20..]);
        let mut positions = vec![[0.0f32; 3]; vertex_count];
        let mut normals = vec![[0.0f32; 3]; vertex_count];
        let mut uvs = vec![[0.0f32; 2]; vertex_count];

        decoder.read_exact(bytemuck::cast_slice_mut(&mut positions))?;
        decoder.read_exact(bytemuck::cast_slice_mut(&mut normals))?;
        decoder.read_exact(bytemuck::cast_slice_mut(&mut uvs))?;

        let mut lod_meshes = Vec::with_capacity(lod_count);
        let mut lod_tri_counts = Vec::with_capacity(lod_count);
        for lod_idx in 0..lod_count {
            let mut count_buf = [0u8; 4];
            decoder.read_exact(&mut count_buf)?;
            let index_count = u32::from_le_bytes(count_buf) as usize;
            lod_tri_counts.push(index_count / 3);

            let mut indices = vec![0u32; index_count];
            decoder.read_exact(bytemuck::cast_slice_mut(&mut indices))?;

            let mut mesh = Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::RENDER_WORLD,
            );
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions.clone());
            mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals.clone());
            mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs.clone());
            mesh.insert_indices(Indices::U32(indices));
            // 自动生成切线以支持法线贴图（若拓扑不满足条件则安全忽略）
            let _ = mesh.generate_tangents();

            let label = format!("lod_{lod_idx}");
            let handle = load_context.add_labeled_asset(label, mesh);
            lod_meshes.push(handle);
        }

        Ok(Mesh3dLodAsset {
            lod_meshes,
            lod_tri_counts,
        })
    }

    fn extensions(&self) -> &[&str] {
        &["mesh3d_lod"]
    }
}

pub struct Mesh3dLodPlugin;

impl Plugin for Mesh3dLodPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<Mesh3dLodAsset>()
            .init_asset_loader::<Mesh3dLodLoader>();
    }
}
