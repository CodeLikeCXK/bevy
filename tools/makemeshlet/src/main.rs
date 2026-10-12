use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use bevy_asset::saver::{AssetSaver, SavedAsset};
use bevy_asset::{AssetPath, RenderAssetUsages};
use bevy_mesh::{Indices, Mesh, PrimitiveTopology};
use bevy_pbr::experimental::meshlet::{
    MeshletMesh, MeshletMeshSaver, MESHLET_DEFAULT_VERTEX_POSITION_QUANTIZATION_FACTOR,
};
use clap::Parser;

/// 内存异步写入适配器，实现 futures_io::AsyncWrite
struct MemoryWriter(Vec<u8>);

impl futures_io::AsyncWrite for MemoryWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        self.0.extend_from_slice(buf);
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

/// 命令行参数
#[derive(Parser, Debug)]
#[command(
    name = "makemeshlet",
    about = "离线独立预处理工具：将 Blender/GLTF 网格转换为 Bevy MeshletMesh 资产 (.meshlet_mesh)"
)]
struct Args {
    /// 输入的 .gltf 或 .glb 文件路径
    #[arg(
        short,
        long,
        default_value = r"C:\Users\12410\Downloads\TestMesh\military_trenches_tunnel_metal_corrugated_01_yd0hehq_gltf_high\Military_Trenches_Tunnel_Metal_Corrugated_01_yd0hehq_High.gltf"
    )]
    input: PathBuf,

    /// 输出文件路径或目标目录（默认输出到与输入文件相同目录）
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// 顶点坐标定点数对齐量化因子 (x 表示 1/2^x 厘米，默认 4 = 1/16 厘米)
    #[arg(short, long, default_value_t = MESHLET_DEFAULT_VERTEX_POSITION_QUANTIZATION_FACTOR)]
    quantization: u8,
}

use bevy_tasks::{AsyncComputeTaskPool, TaskPool};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // 初始化多线程任务池（供 METIS 与 Meshlet 简化算法并行加速）
    AsyncComputeTaskPool::get_or_init(TaskPool::default);

    println!("=======================================================");
    println!("           Bevy Meshlet 离线预处理工具: makemeshlet     ");
    println!("=======================================================");
    println!("输入文件: {}", args.input.display());
    println!("量化因子: {} (精度: 1/2^{} cm)", args.quantization, args.quantization);

    if !args.input.exists() {
        return Err(format!("输入文件不存在: {}", args.input.display()).into());
    }

    // 1. 加载 glTF 文档与缓冲区
    println!("\n[1/4] 正在解析 GLTF 文件...");
    let start_load = Instant::now();
    let (document, buffers, _images) = gltf::import(&args.input)?;
    println!("      解析完成，耗时 {:.2}ms", start_load.elapsed().as_secs_f64() * 1000.0);

    // 计算总网格与图元数量
    let total_meshes = document.meshes().count();
    let mut total_primitives = 0;
    for mesh in document.meshes() {
        total_primitives += mesh.primitives().count();
    }
    println!("      场景包含 {} 个 Mesh, 共 {} 个 Primitive", total_meshes, total_primitives);

    // 确定输出目录或基础文件名
    let input_dir = args.input.parent().unwrap_or_else(|| Path::new("."));
    let input_stem = args
        .input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    let mut processed_count = 0;

    // 2. 遍历网格并预处理各个 Primitive
    for mesh in document.meshes() {
        let mesh_name = mesh.name().unwrap_or("mesh");
        for (prim_idx, primitive) in mesh.primitives().enumerate() {
            println!("\n-------------------------------------------------------");
            println!(
                "正在处理 Primitive [{}/{}]: {} (索引 #{})",
                processed_count + 1,
                total_primitives,
                mesh_name,
                prim_idx
            );

            // 检查拓扑模式
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                eprintln!(
                    "  [跳过] 图元拓扑不是 Triangles ({:?})，请在建模软件中三角化！",
                    primitive.mode()
                );
                continue;
            }

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            // 读取顶点坐标
            let positions: Vec<[f32; 3]> = match reader.read_positions() {
                Some(iter) => iter.collect(),
                None => {
                    eprintln!("  [跳过] 图元缺少 POSITION 坐标！");
                    continue;
                }
            };

            let vert_count = positions.len();

            // 计算包围盒尺寸
            let mut min = [f32::INFINITY; 3];
            let mut max = [f32::NEG_INFINITY; 3];
            for p in &positions {
                for i in 0..3 {
                    min[i] = min[i].min(p[i]);
                    max[i] = max[i].max(p[i]);
                }
            }
            let size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
            println!("  顶点数: {}", vert_count);
            println!(
                "  包围盒尺寸: [{:.2}, {:.2}, {:.2}] (Min: [{:.2}, {:.2}, {:.2}], Max: [{:.2}, {:.2}, {:.2}])",
                size[0], size[1], size[2], min[0], min[1], min[2], max[0], max[1], max[2]
            );

            // 读取法线
            let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|iter| iter.collect());

            // 读取 UV0（纹理贴图坐标）
            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|iter| iter.into_f32().collect())
                .unwrap_or_else(|| {
                    println!("  [警告] 图元未找到 TEXCOORD_0，自动填充默认 UV [0.0, 0.0]...");
                    vec![[0.0, 0.0]; vert_count]
                });

            // 读取三角形索引
            let indices: Vec<u32> = reader
                .read_indices()
                .map(|iter| iter.into_u32().collect())
                .unwrap_or_else(|| {
                    println!("  [警告] 图元未找到索引缓冲，自动生成顺序索引...");
                    (0..vert_count as u32).collect()
                });

            let tri_count = indices.len() / 3;
            println!("  三角形数: {}", tri_count);

            // 构造符合 Bevy 规范的纯净 Mesh (严格白名单：Position, Normal, UV_0)
            let mut bevy_mesh =
                Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
            bevy_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);

            if let Some(normals) = normals {
                bevy_mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
            } else {
                println!("  [提示] 自动计算平滑法线 (compute_smooth_normals)...");
                bevy_mesh.compute_smooth_normals();
            }

            bevy_mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
            bevy_mesh.insert_indices(Indices::U32(indices));

            // 3. 执行核心 Meshlet 预处理算法
            println!("\n[2/4] 正在执行 Meshlet 簇划分与分层 LOD 构建...");
            println!("      (METIS 图分割 + 边界锁定边折叠简化 + BVH8 树构建)");
            let start_meshlet = Instant::now();
            let meshlet_mesh = match MeshletMesh::from_mesh(&bevy_mesh, args.quantization) {
                Ok(meshlet) => meshlet,
                Err(err) => {
                    eprintln!("  [错误] 预处理失败: {:?}", err);
                    continue;
                }
            };
            let meshlet_duration = start_meshlet.elapsed();
            println!(
                "      Meshlet 构建完成！耗时: {:.2}s",
                meshlet_duration.as_secs_f64()
            );

            // 4. 序列化为 Bevy 原生 .meshlet_mesh 资产文件
            println!("\n[3/4] 正在序列化与位流打包 (LZ4 压缩)...");
            let saver = MeshletMeshSaver;
            let mut memory_writer = MemoryWriter(Vec::new());
            let saved_asset = SavedAsset::from_asset(&meshlet_mesh);

            bevy_tasks::block_on(saver.save(
                &mut memory_writer,
                saved_asset,
                &(),
                AssetPath::default(),
            ))?;

            // 确定目标输出路径
            let out_file_path = if let Some(ref out) = args.output {
                if out.is_dir() {
                    if total_primitives > 1 {
                        out.join(format!("{}_{}_{}.meshlet_mesh", input_stem, mesh_name, prim_idx))
                    } else {
                        out.join(format!("{}.meshlet_mesh", input_stem))
                    }
                } else {
                    out.clone()
                }
            } else if total_primitives > 1 {
                input_dir.join(format!("{}_{}_{}.meshlet_mesh", input_stem, mesh_name, prim_idx))
            } else {
                input_dir.join(format!("{}.meshlet_mesh", input_stem))
            };

            // 写入磁盘
            std::fs::write(&out_file_path, &memory_writer.0)?;
            let file_kb = memory_writer.0.len() as f64 / 1024.0;

            println!("\n[4/4] 导出成功！");
            println!("      输出文件: {}", out_file_path.display());
            println!("      文件大小: {:.2} KB ({} 字节)", file_kb, memory_writer.0.len());

            processed_count += 1;
        }
    }

    println!("\n=======================================================");
    println!("处理完成！共成功导出 {} 个 .meshlet_mesh 文件。", processed_count);
    println!("可在 Bevy 中直接使用:");
    println!(r#"  let handle = asset_server.load("path/to/{}.meshlet_mesh");"#, input_stem);
    println!("=======================================================");

    Ok(())
}
