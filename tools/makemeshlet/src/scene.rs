use std::collections::HashMap;
use std::path::Path;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use bevy_asset::saver::{AssetSaver, SavedAsset};
use bevy_asset::{AssetPath, RenderAssetUsages};
use bevy_math::{Mat3, Mat4, Vec3};
use bevy_mesh::{Indices, Mesh, PrimitiveTopology};
use bevy_pbr::experimental::meshlet::{MeshletMesh, MeshletMeshSaver};
use serde::{Deserialize, Serialize};

/// 内存异步写入适配器，实现 futures_io::AsyncWrite
pub struct MemoryWriter(pub Vec<u8>);

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshletSceneManifest {
    pub version: u32,
    pub source_file: String,
    pub materials: Vec<SceneMaterial>,
    pub meshlet_instances: Vec<SceneMeshletInstance>,
    pub lights: Vec<SceneLight>,
    pub skipped_non_opaque: Vec<SkippedPrimitiveInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneMaterial {
    pub id: u32,
    pub name: String,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub base_color_texture: Option<String>,
    pub normal_texture: Option<String>,
    pub metallic_roughness_texture: Option<String>,
    pub occlusion_texture: Option<String>,
    pub emissive_texture: Option<String>,
    pub double_sided: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneMeshletInstance {
    pub name: String,
    pub mesh: String,
    pub material_id: u32,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub dynamic: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneLight {
    pub name: String,
    pub kind: String,
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: Option<f32>,
    pub inner_cone_angle: Option<f32>,
    pub outer_cone_angle: Option<f32>,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkippedPrimitiveInfo {
    pub node_name: String,
    pub mesh_name: String,
    pub alpha_mode: String,
    pub triangle_count: usize,
}

#[derive(Hash, Eq, PartialEq, Clone)]
struct MaterialKey {
    base_color_bits: [u32; 4],
    metallic_bits: u32,
    roughness_bits: u32,
    emissive_bits: [u32; 3],
    base_color_tex: Option<usize>,
    normal_tex: Option<usize>,
    mr_tex: Option<usize>,
    occlusion_tex: Option<usize>,
    emissive_tex: Option<usize>,
    double_sided: bool,
}

struct StaticBatch {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
}

impl Default for StaticBatch {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            aabb_min: [f32::INFINITY; 3],
            aabb_max: [f32::NEG_INFINITY; 3],
        }
    }
}

pub struct SceneExportOptions<'a> {
    pub input: &'a Path,
    pub out_dir: &'a Path,
    pub input_stem: &'a str,
    pub quantization: u8,
    pub merge_by_material: bool,
    pub dynamic_prefixes: &'a [String],
    pub export_textures: bool,
}

struct SceneContext<'a> {
    buffers: &'a [gltf::buffer::Data],
    exported_tex_paths: &'a [Option<String>],
    options: &'a SceneExportOptions<'a>,
    material_map: HashMap<MaterialKey, u32>,
    materials: Vec<SceneMaterial>,
    static_batches: HashMap<u32, StaticBatch>,
    dynamic_assets_cache: HashMap<(usize, usize), (String, [f32; 3], [f32; 3])>,
    dynamic_instances: Vec<SceneMeshletInstance>,
    lights: Vec<SceneLight>,
    skipped_non_opaque: Vec<SkippedPrimitiveInfo>,
    total_nodes_visited: usize,
    total_primitives_merged: usize,
}

pub fn run_scene_export(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
    options: &SceneExportOptions<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(options.out_dir)?;
    let meshes_dir = options.out_dir.join("meshes");
    std::fs::create_dir_all(&meshes_dir)?;

    // 1. 导出贴图（若启用）
    let mut exported_tex_paths = vec![None; images.len()];
    if options.export_textures && !images.is_empty() {
        let textures_dir = options.out_dir.join("textures");
        std::fs::create_dir_all(&textures_dir)?;
        println!("\n[2/5] 正在导出 {} 张内嵌/引用贴图...", images.len());
        for (idx, img_data) in images.iter().enumerate() {
            let rel_path = format!("textures/{}_tex_{}.png", options.input_stem, idx);
            let abs_path = options.out_dir.join(&rel_path);
            if save_gltf_image_as_png(img_data, &abs_path).is_ok() {
                exported_tex_paths[idx] = Some(rel_path);
            } else {
                eprintln!("  [警告] 贴图 #{} 导出 PNG 失败，已跳过", idx);
            }
        }
    }

    let mut ctx = SceneContext {
        buffers,
        exported_tex_paths: &exported_tex_paths,
        options,
        material_map: HashMap::new(),
        materials: Vec::new(),
        static_batches: HashMap::new(),
        dynamic_assets_cache: HashMap::new(),
        dynamic_instances: Vec::new(),
        lights: Vec::new(),
        skipped_non_opaque: Vec::new(),
        total_nodes_visited: 0,
        total_primitives_merged: 0,
    };

    // 2. 遍历场景树，收集材质、合并静态几何体、提取动态件与灯光
    println!("\n[3/5] 正在遍历场景节点树并分组/变换几何体...");
    if let Some(default_scene) = document.default_scene() {
        for node in default_scene.nodes() {
            visit_node(&node, Mat4::IDENTITY, false, &mut ctx)?;
        }
    } else if let Some(first_scene) = document.scenes().next() {
        for node in first_scene.nodes() {
            visit_node(&node, Mat4::IDENTITY, false, &mut ctx)?;
        }
    }

    println!(
        "      遍历节点: {} | 去重材质: {} | 合并静态图元: {} | 独立/动态实例: {} | 灯光: {} | 跳过非Opaque图元: {}",
        ctx.total_nodes_visited,
        ctx.materials.len(),
        ctx.total_primitives_merged,
        ctx.dynamic_instances.len(),
        ctx.lights.len(),
        ctx.skipped_non_opaque.len()
    );

    // 3. 构建并保存按材质合并后的静态 MeshletMesh
    println!("\n[4/5] 正在构建并导出 MeshletMesh 资产...");
    let mut final_instances = Vec::new();
    let mut sorted_mat_ids: Vec<u32> = ctx.static_batches.keys().copied().collect();
    sorted_mat_ids.sort_unstable();

    for (idx, mat_id) in sorted_mat_ids.iter().enumerate() {
        let batch = ctx.static_batches.remove(mat_id).unwrap();
        let tri_count = batch.indices.len() / 3;
        println!(
            "  -> 静态材质批次 [{}/{}] mat_id={} (顶点: {}, 三角形: {})",
            idx + 1,
            sorted_mat_ids.len(),
            mat_id,
            batch.positions.len(),
            tri_count
        );

        let mut bevy_mesh =
            Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
        bevy_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, batch.positions);
        bevy_mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, batch.normals);
        bevy_mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, batch.uvs);
        bevy_mesh.insert_indices(Indices::U32(batch.indices));

        let t0 = Instant::now();
        let meshlet_mesh = MeshletMesh::from_mesh(&bevy_mesh, options.quantization)
            .map_err(|e| format!("静态材质批次 mat_id={} 构建 Meshlet 失败: {:?}", mat_id, e))?;

        let rel_mesh_path = format!("meshes/{}_mat_{}.meshlet_mesh", options.input_stem, mat_id);
        let abs_mesh_path = options.out_dir.join(&rel_mesh_path);
        save_meshlet_mesh(&meshlet_mesh, &abs_mesh_path)?;
        println!(
            "     完成，耗时 {:.2}s -> {}",
            t0.elapsed().as_secs_f64(),
            rel_mesh_path
        );

        final_instances.push(SceneMeshletInstance {
            name: format!("static_mat_{}", mat_id),
            mesh: rel_mesh_path,
            material_id: *mat_id,
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            aabb_min: batch.aabb_min,
            aabb_max: batch.aabb_max,
            dynamic: false,
        });
    }

    final_instances.extend(ctx.dynamic_instances);

    // 4. 写入场景清单 JSON
    println!("\n[5/5] 正在生成场景清单 (.meshlet_scene.json)...");
    let manifest = MeshletSceneManifest {
        version: 1,
        source_file: options
            .input
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown.glb")
            .to_string(),
        materials: ctx.materials,
        meshlet_instances: final_instances,
        lights: ctx.lights,
        skipped_non_opaque: ctx.skipped_non_opaque,
    };

    let manifest_path = options
        .out_dir
        .join(format!("{}.meshlet_scene.json", options.input_stem));
    let json_str = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, json_str)?;

    println!("\n=======================================================");
    println!("场景导出完成！");
    println!("  清单文件: {}", manifest_path.display());
    println!("  Meshlet 实例总数: {}", manifest.meshlet_instances.len());
    println!("  去重材质总数: {}", manifest.materials.len());
    println!("  提取灯光总数: {}", manifest.lights.len());
    if !manifest.skipped_non_opaque.is_empty() {
        println!(
            "  [提示] 有 {} 个非 Opaque 图元未生成 Meshlet（需在渲染器中走标准透明管线）",
            manifest.skipped_non_opaque.len()
        );
    }
    println!("=======================================================");

    Ok(())
}

fn visit_node(
    node: &gltf::Node,
    parent_world: Mat4,
    parent_is_dynamic: bool,
    ctx: &mut SceneContext<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    ctx.total_nodes_visited += 1;
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent_world * local;

    let node_name = node.name().unwrap_or("unnamed_node");
    let is_dynamic = parent_is_dynamic
        || ctx
            .options
            .dynamic_prefixes
            .iter()
            .any(|p| node_name.starts_with(p));

    // 提取灯光
    if let Some(light) = node.light() {
        let (_scale, rotation, translation) = world.to_scale_rotation_translation();
        let (kind, inner, outer) = match light.kind() {
            gltf::khr_lights_punctual::Kind::Directional => ("Directional".to_string(), None, None),
            gltf::khr_lights_punctual::Kind::Point => ("Point".to_string(), None, None),
            gltf::khr_lights_punctual::Kind::Spot {
                inner_cone_angle,
                outer_cone_angle,
            } => (
                "Spot".to_string(),
                Some(inner_cone_angle),
                Some(outer_cone_angle),
            ),
        };
        ctx.lights.push(SceneLight {
            name: light.name().unwrap_or(node_name).to_string(),
            kind,
            color: light.color(),
            intensity: light.intensity(),
            range: light.range(),
            inner_cone_angle: inner,
            outer_cone_angle: outer,
            translation: translation.to_array(),
            rotation: rotation.to_array(),
        });
    }

    // 处理 Mesh
    if let Some(mesh) = node.mesh() {
        let mesh_name = mesh.name().unwrap_or("mesh");
        for (prim_idx, primitive) in mesh.primitives().enumerate() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }

            let gltf_mat = primitive.material();
            let alpha_mode = gltf_mat.alpha_mode();
            let reader = primitive.reader(|b| Some(&ctx.buffers[b.index()]));

            let Some(positions_iter) = reader.read_positions() else {
                continue;
            };
            let positions: Vec<[f32; 3]> = positions_iter.collect();
            let vert_count = positions.len();
            let indices: Vec<u32> = reader
                .read_indices()
                .map(|i| i.into_u32().collect())
                .unwrap_or_else(|| (0..vert_count as u32).collect());

            if alpha_mode != gltf::material::AlphaMode::Opaque {
                ctx.skipped_non_opaque.push(SkippedPrimitiveInfo {
                    node_name: node_name.to_string(),
                    mesh_name: mesh_name.to_string(),
                    alpha_mode: format!("{:?}", alpha_mode),
                    triangle_count: indices.len() / 3,
                });
                continue;
            }

            let mat_id = register_or_get_material(&gltf_mat, ctx);
            let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|i| i.collect());
            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|i| i.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; vert_count]);

            if ctx.options.merge_by_material && !is_dynamic {
                // 合并到按材质分组的静态批次中
                append_to_static_batch(
                    ctx.static_batches.entry(mat_id).or_default(),
                    world,
                    &positions,
                    normals.as_deref(),
                    &uvs,
                    &indices,
                );
                ctx.total_primitives_merged += 1;
            } else {
                // 独立/动态节点：按 (mesh.index(), prim_idx) 去重构建局部空间 MeshletMesh
                let key = (mesh.index(), prim_idx);
                let (rel_mesh_path, aabb_min, aabb_max) =
                    if let Some(cached) = ctx.dynamic_assets_cache.get(&key) {
                        cached.clone()
                    } else {
                        let (mut aabb_min, mut aabb_max) =
                            ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
                        for p in &positions {
                            for i in 0..3 {
                                aabb_min[i] = aabb_min[i].min(p[i]);
                                aabb_max[i] = aabb_max[i].max(p[i]);
                            }
                        }

                        let mut bevy_mesh =
                            Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
                        bevy_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
                        if let Some(n) = normals {
                            bevy_mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, n);
                        } else {
                            bevy_mesh.compute_smooth_normals();
                        }
                        bevy_mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
                        bevy_mesh.insert_indices(Indices::U32(indices));

                        let meshlet_mesh =
                            MeshletMesh::from_mesh(&bevy_mesh, ctx.options.quantization)
                                .map_err(|e| {
                                    format!(
                                        "独立图元 {}_{} 构建 Meshlet 失败: {:?}",
                                        mesh_name, prim_idx, e
                                    )
                                })?;

                        let rel_path = format!(
                            "meshes/{}_mesh_{}_prim_{}.meshlet_mesh",
                            ctx.options.input_stem,
                            mesh.index(),
                            prim_idx
                        );
                        let abs_path = ctx.options.out_dir.join(&rel_path);
                        save_meshlet_mesh(&meshlet_mesh, &abs_path)?;

                        let entry = (rel_path, aabb_min, aabb_max);
                        ctx.dynamic_assets_cache.insert(key, entry.clone());
                        entry
                    };

                let (scale, rotation, translation) = world.to_scale_rotation_translation();
                ctx.dynamic_instances.push(SceneMeshletInstance {
                    name: node_name.to_string(),
                    mesh: rel_mesh_path,
                    material_id: mat_id,
                    translation: translation.to_array(),
                    rotation: rotation.to_array(),
                    scale: scale.to_array(),
                    aabb_min,
                    aabb_max,
                    dynamic: is_dynamic,
                });
            }
        }
    }

    for child in node.children() {
        visit_node(&child, world, is_dynamic, ctx)?;
    }

    Ok(())
}

fn append_to_static_batch(
    batch: &mut StaticBatch,
    world: Mat4,
    positions: &[[f32; 3]],
    normals: Option<&[[f32; 3]]>,
    uvs: &[[f32; 2]],
    indices: &[u32],
) {
    let base_vertex = batch.positions.len() as u32;
    let det = world.determinant();
    let flip_winding = det < 0.0;

    // 计算世界空间位置并更新包围盒
    for &p in positions {
        let wp = world.transform_point3(Vec3::from_array(p)).to_array();
        for i in 0..3 {
            batch.aabb_min[i] = batch.aabb_min[i].min(wp[i]);
            batch.aabb_max[i] = batch.aabb_max[i].max(wp[i]);
        }
        batch.positions.push(wp);
    }

    // 计算世界空间法线；若缺失则先在局部生成平滑法线
    let local_normals: Vec<[f32; 3]> = if let Some(n) = normals {
        n.to_vec()
    } else {
        let mut temp = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
        temp.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions.to_vec());
        temp.insert_indices(Indices::U32(indices.to_vec()));
        temp.compute_smooth_normals();
        temp.attribute(Mesh::ATTRIBUTE_NORMAL)
            .and_then(|a| a.as_float3())
            .map(|s| s.to_vec())
            .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()])
    };

    let linear_mat3 = Mat3::from_mat4(world);
    let normal_mat = if linear_mat3.determinant().abs() > 1e-12 {
        linear_mat3.inverse().transpose()
    } else {
        Mat3::IDENTITY
    };
    for n in local_normals {
        let wn = (normal_mat * Vec3::from_array(n)).normalize_or(Vec3::Y);
        batch.normals.push(wn.to_array());
    }

    batch.uvs.extend_from_slice(uvs);

    // 若存在负缩放镜像，翻转三角形绕序
    for tri in indices.chunks_exact(3) {
        if flip_winding {
            batch.indices.push(base_vertex + tri[0]);
            batch.indices.push(base_vertex + tri[2]);
            batch.indices.push(base_vertex + tri[1]);
        } else {
            batch.indices.push(base_vertex + tri[0]);
            batch.indices.push(base_vertex + tri[1]);
            batch.indices.push(base_vertex + tri[2]);
        }
    }
}

fn register_or_get_material(gltf_mat: &gltf::Material, ctx: &mut SceneContext<'_>) -> u32 {
    let pbr = gltf_mat.pbr_metallic_roughness();
    let base_color = pbr.base_color_factor();
    let metallic = pbr.metallic_factor();
    let roughness = pbr.roughness_factor();
    let emissive = gltf_mat.emissive_factor();

    let base_color_tex = pbr.base_color_texture().map(|t| t.texture().source().index());
    let normal_tex = gltf_mat.normal_texture().map(|t| t.texture().source().index());
    let mr_tex = pbr
        .metallic_roughness_texture()
        .map(|t| t.texture().source().index());
    let occlusion_tex = gltf_mat
        .occlusion_texture()
        .map(|t| t.texture().source().index());
    let emissive_tex = gltf_mat
        .emissive_texture()
        .map(|t| t.texture().source().index());
    let double_sided = gltf_mat.double_sided();

    let key = MaterialKey {
        base_color_bits: base_color.map(f32::to_bits),
        metallic_bits: metallic.to_bits(),
        roughness_bits: roughness.to_bits(),
        emissive_bits: emissive.map(f32::to_bits),
        base_color_tex,
        normal_tex,
        mr_tex,
        occlusion_tex,
        emissive_tex,
        double_sided,
    };

    if let Some(&id) = ctx.material_map.get(&key) {
        return id;
    }

    let id = ctx.materials.len() as u32;
    let tex_path = |idx: Option<usize>| -> Option<String> {
        idx.and_then(|i| ctx.exported_tex_paths.get(i).cloned().flatten())
    };

    ctx.materials.push(SceneMaterial {
        id,
        name: gltf_mat
            .name()
            .map(ToString::to_string)
            .unwrap_or_else(|| format!("material_{}", id)),
        base_color,
        metallic,
        roughness,
        emissive,
        base_color_texture: tex_path(base_color_tex),
        normal_texture: tex_path(normal_tex),
        metallic_roughness_texture: tex_path(mr_tex),
        occlusion_texture: tex_path(occlusion_tex),
        emissive_texture: tex_path(emissive_tex),
        double_sided,
    });
    ctx.material_map.insert(key, id);
    id
}

fn save_meshlet_mesh(
    meshlet_mesh: &MeshletMesh,
    out_file_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let saver = MeshletMeshSaver;
    let mut memory_writer = MemoryWriter(Vec::new());
    let saved_asset = SavedAsset::from_asset(meshlet_mesh);
    bevy_tasks::block_on(saver.save(
        &mut memory_writer,
        saved_asset,
        &(),
        AssetPath::default(),
    ))?;
    std::fs::write(out_file_path, &memory_writer.0)?;
    Ok(())
}

fn save_gltf_image_as_png(
    data: &gltf::image::Data,
    out_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use gltf::image::Format;
    let (w, h) = (data.width, data.height);
    match data.format {
        Format::R8 => {
            image::save_buffer_with_format(
                out_path,
                &data.pixels,
                w,
                h,
                image::ColorType::L8,
                image::ImageFormat::Png,
            )?;
        }
        Format::R8G8 => {
            let mut rgb = Vec::with_capacity((w * h * 3) as usize);
            for chunk in data.pixels.chunks_exact(2) {
                rgb.extend_from_slice(&[chunk[0], chunk[1], 0]);
            }
            image::save_buffer_with_format(
                out_path,
                &rgb,
                w,
                h,
                image::ColorType::Rgb8,
                image::ImageFormat::Png,
            )?;
        }
        Format::R8G8B8 => {
            image::save_buffer_with_format(
                out_path,
                &data.pixels,
                w,
                h,
                image::ColorType::Rgb8,
                image::ImageFormat::Png,
            )?;
        }
        Format::R8G8B8A8 => {
            image::save_buffer_with_format(
                out_path,
                &data.pixels,
                w,
                h,
                image::ColorType::Rgba8,
                image::ImageFormat::Png,
            )?;
        }
        Format::R16 | Format::R16G16 | Format::R16G16B16 | Format::R16G16B16A16 => {
            let channels: usize = match data.format {
                Format::R16 => 1,
                Format::R16G16 => 2,
                Format::R16G16B16 => 3,
                Format::R16G16B16A16 => 4,
                _ => unreachable!(),
            };
            let mut out8 = Vec::with_capacity((w * h) as usize * channels.max(3));
            for pixel in data.pixels.chunks_exact(2 * channels) {
                let c0 = u16::from_le_bytes([pixel[0], pixel[1]]) >> 8;
                if channels == 1 {
                    out8.push(c0 as u8);
                } else if channels == 2 {
                    let c1 = u16::from_le_bytes([pixel[2], pixel[3]]) >> 8;
                    out8.extend_from_slice(&[c0 as u8, c1 as u8, 0]);
                } else {
                    for c in 0..channels {
                        let val = u16::from_le_bytes([pixel[c * 2], pixel[c * 2 + 1]]) >> 8;
                        out8.push(val as u8);
                    }
                }
            }
            let color_type = match channels {
                1 => image::ColorType::L8,
                2 | 3 => image::ColorType::Rgb8,
                _ => image::ColorType::Rgba8,
            };
            image::save_buffer_with_format(
                out_path,
                &out8,
                w,
                h,
                color_type,
                image::ImageFormat::Png,
            )?;
        }
        Format::R32G32B32FLOAT | Format::R32G32B32A32FLOAT => {
            let channels = if matches!(data.format, Format::R32G32B32FLOAT) {
                3
            } else {
                4
            };
            let mut out8 = Vec::with_capacity((w * h * channels) as usize);
            for ch in data.pixels.chunks_exact(4) {
                let f = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
                out8.push((f.clamp(0.0, 1.0) * 255.0) as u8);
            }
            let color_type = if channels == 3 {
                image::ColorType::Rgb8
            } else {
                image::ColorType::Rgba8
            };
            image::save_buffer_with_format(
                out_path,
                &out8,
                w,
                h,
                color_type,
                image::ImageFormat::Png,
            )?;
        }
    }
    Ok(())
}

