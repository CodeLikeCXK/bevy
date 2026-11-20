# makemeshlet 使用指南

**中文** | [English](README.en.md)

`makemeshlet` 是面向 Bevy 引擎的离线场景与几何预处理工具，用于将 Blender 导出的 `.gltf` / `.glb` 模型及大型场景自动清洗、分流并烘焙为：

- **Meshlet 资产（`.meshlet_mesh`）**：支持自动退化三角面清洗、空间网格分块、批次面数上限拆分、自动实例化去重与离线逐顶点 AO 烘焙；
- **标准多级 LOD 资产（`.mesh3d_lod`）**：针对植被、镂空（`AlphaMode::Mask`）及玻璃/半透明件（`AlphaMode::Blend`）自动使用 `meshopt` 生成多级 LOD；
- **静态表面虚拟光照贴图（`.vlm.bin`）**：免 UV2 的空间哈希 8×8 表面砖块离线路径追踪烘焙器，支持 headless `wgpu` GPU 加速（自动回退多线程 CPU），同时输出间接漫反射 GI + 局部光阴影（RGB）与主平行光软阴影遮罩（A）；
- **统一场景清单（`.meshlet_scene.json`）**：记录去重 PBR 材质、Meshlet / Mesh3d 实例、空间分块、灯光、`World HDRI`、Gameplay 标记点与 `.vlm.bin` 引用，供渲染器直接流送加载。

---

## 目录

1. [编译与快速开始](#1-编译与快速开始)
2. [常用工作流示例](#2-常用工作流示例)
3. [完整命令行参数手册](#3-完整命令行参数手册)
   - [3.1 基础输入输出与量化参数](#31-基础输入输出与量化参数)
   - [3.2 场景导出与分块流送参数（`--scene`）](#32-场景导出与分块流送参数--scene)
   - [3.3 静态逐顶点 AO 烘焙参数](#33-静态逐顶点-ao-烘焙参数)
   - [3.4 Virtual Lightmap（VLM）光照贴图烘焙参数](#34-virtual-lightmapvlm光照贴图烘焙参数)
4. [Blender 节点角标与自动分流规则](#4-blender-节点角标与自动分流规则)
5. [输出目录结构](#5-输出目录结构)
6. [Virtual Lightmap 技术思路来源与授权状态说明](#6-virtual-lightmap-技术思路来源与授权状态说明)

---

## 1. 编译与快速开始

在 Bevy 仓库根目录下编译 Release 版本（强烈建议使用 `--release`，否则高模简化与射线追踪耗时显著增加）：

```powershell
cargo build --release --package makemeshlet
```

查看内置帮助信息：

```powershell
.\target\release\makemeshlet.exe --help
```

> **Windows 链接环境提示**：若本地终端中存在被其他工具链污染的 `LIB` / `INCLUDE` 环境变量导致 `LNK1104: cannot open file 'msvcrt.lib'`，可先在当前终端清理后再编译：
> ```powershell
> Remove-Item Env:LIB, Env:LIBPATH, Env:INCLUDE -ErrorAction SilentlyContinue
> cargo build --release --package makemeshlet
> ```

---

## 2. 常用工作流示例

### 2.1 单模型转换模式（默认）
将单个 `.gltf` / `.glb` 的各个图元直接转换为 `.meshlet_mesh`（默认附带离线逐顶点 AO 烘焙）：

```powershell
.\target\release\makemeshlet.exe `
  -i "path\to\model.glb" `
  -o "assets\models\"
```

### 2.2 完整场景烘焙模式（Meshlet + LOD + 贴图 + AO + VLM）
针对完整关卡（如千万级三角面 Sponza 场景），一次完成网格分块、贴图降采样导出、逐顶点 AO 与 GPU 加速 VLM 光照贴图烘焙：

```powershell
.\target\release\makemeshlet.exe `
  --scene `
  --export-textures `
  --max-texture-size 1024 `
  --grid-size 12.0 `
  --max-batch-tris 250000 `
  -i "path\to\scene.glb" `
  -o "path\to\scene_out"
```

### 2.3 仅重烘光照贴图模式（`--vlm-only`）
在网格已导出、仅需迭代光照参数或重新烘焙 `.vlm.bin` 时，使用 `--vlm-only` 跳过全部贴图导出、逐顶点 AO 与 Meshlet/LOD 构建，大幅缩短迭代耗时：

```powershell
.\target\release\makemeshlet.exe `
  --vlm-only `
  --vlm-texel-size 0.06 `
  --vlm-samples 32 `
  --vlm-bounces 2 `
  -i "path\to\scene.glb" `
  -o "path\to\scene_out"
```

> 若输出目录中已存在 `<stem>.meshlet_scene.json`，`--vlm-only` 会**无损保留**原有的 `meshes/`、材质与实例列表，仅更新清单内的 `vlm_file`（以及空缺时的 `world_hdri`）字段。

### 2.4 强制纯 CPU 烘焙 VLM 或禁用 VLM / AO
```powershell
# 禁用 VLM，仅导出场景网格与逐顶点 AO
.\target\release\makemeshlet.exe --scene --no-vlm -i "scene.glb" -o "scene_out"

# 仅重烘 VLM 且强制使用多线程 CPU（跳过 headless wgpu GPU 后端）
.\target\release\makemeshlet.exe --vlm-only --vlm-cpu -i "scene.glb" -o "scene_out"

# 快速导出草稿场景（同时关闭 AO 与 VLM）
.\target\release\makemeshlet.exe --scene --no-ao --no-vlm -i "scene.glb" -o "scene_out"
```

---

## 3. 完整命令行参数手册

### 3.1 基础输入输出与量化参数

| 参数 | 短选项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `--input <PATH>` | `-i` | 路径 | **必填** | 输入的 `.gltf` 或 `.glb` 文件路径。 |
| `--output <PATH>` | `-o` | 路径 | 输入文件所在目录 | 单模型模式下可为输出文件或目录；场景模式（`--scene` / `--vlm-only`）下为输出根目录。 |
| `--quantization <U8>` | `-q` | `u8` | `4` | Meshlet 顶点坐标定点数对齐量化因子：`x` 表示 $1 / 2^x\text{ cm}$ 精度（默认 `4` = `1/16 cm = 0.0625 cm`）。值越大精度越高但体积略增。 |

---

### 3.2 场景导出与分块流送参数（`--scene`）

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--scene` | Flag | `false` | 启用场景级导出模式：递归遍历节点树变换、解析 Blender 角标、合批与分流、提取材质/灯光/HDRI、烘焙 AO 与 VLM 并生成 `<stem>.meshlet_scene.json`。 |
| `--no-merge` | Flag | `false` | 场景模式下禁用按材质静态合并，保留各节点独立实例与世界位姿（仍会对共享同一 Mesh 图元的资产去重）。 |
| `--dynamic-prefix <PREFIX>` | `Vec<String>` | 空 | 不参与静态合并的节点名称前缀，可重复多次指定（例如 `--dynamic-prefix Wheel_ --dynamic-prefix Door_`），命中节点将强制走动态独立实例化分支且不参与静态 VLM 遮挡。 |
| `--export-textures` | Flag | `false` | 场景模式下将 glTF/GLB 内嵌或引用贴图导出为 PNG 至 `<output>/textures/` 子目录。 |
| `--max-texture-size <U32>` | `u32` | `1024` | 导出贴图的最大边长（像素）；超过此尺寸自动按比例双线性降采样，`0` 表示不限制。针对核显/轻薄本推荐 `1024` 或 `512`。 |
| `--grid-size <F32>` | `f32` | `12.0` | 静态合批的空间自动分块网格边长（米），用于把默认 `chunk="main"` 的大场景自动切分为空间区块以支持按距离流送卸载；`0` 表示禁用自动空间切分。 |
| `--max-batch-tris <USIZE>` | `usize` | `250000` | 单个静态 Meshlet 合并批次的最大三角形数上限，超出后自动开启新批次；`0` 表示不限制。 |
| `--auto-instance-min <USIZE>` | `usize` | `3` | 同一图元在场景非镜像可渲染节点中出现次数 $\ge N$ 时，自动走局部实例化复用而非复制进静态合并大网格；`0` 表示禁用自动实例化。 |
| `--lod-keep-blend` | Flag | `false` | 允许显式标记为 `MESH3D_LOD` 的物体保留 `AlphaMode::Blend`。默认关闭（自动转为 `AlphaMode::Mask` 以避免透明通道过绘瓶颈）。 |

---

### 3.3 静态逐顶点 AO 烘焙参数

默认在单模型模式与 `--scene` 模式下均开启（`--vlm-only` 模式下自动关闭）。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--no-ao` | Flag | `false` | 禁用离线静态逐顶点 AO 烘焙（所有顶点 AO 保持 `1.0`）。 |
| `--ao-samples <U32>` | `u32` | `32` | 每个顶点的余弦加权半球射线采样数。 |
| `--ao-radius <F32>` | `f32` | `1.5` | 最大遮蔽检测半径（米）。建议控制在接触遮蔽尺度，避免与全局 GI / IrradianceVolume 重复压暗室内。 |
| `--ao-power <F32>` | `f32` | `1.0` | AO 对比度幂指数（`> 1.0` 会加深墙角与缝隙暗部对比）。 |
| `--ao-bias <F32>` | `Option<f32>` | 自动计算 | 射线起点沿法线的偏移量（米）；未指定时自动取 $\operatorname{clamp}(\text{ao\_radius} \times 0.0025,\ 0.0005,\ 0.01)$。 |

---

### 3.4 Virtual Lightmap（VLM）光照贴图烘焙参数

在 `--scene` 模式下默认开启，或通过 `--vlm-only` 单独触发。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--no-vlm` | Flag | `false` | 禁用离线静态表面 Virtual Lightmap（`.vlm.bin`）烘焙。若同时指定 `--vlm-only`，以 `--vlm-only` 为准。 |
| `--vlm-only` | Flag | `false` | 仅烘焙 `<stem>.vlm.bin`，自动隐含 `--scene` 并关闭贴图导出、逐顶点 AO 与 Meshlet/LOD 构建；若输出目录已有 `.meshlet_scene.json`，仅无损更新其 `vlm_file` 字段。 |
| `--vlm-texel-size <F32>` | `f32` | `0.06` | Level 0 单纹素物理尺寸（米，默认 `0.06 m = 6 cm`，单个 `8×8` 砖块覆盖 `48 cm × 48 cm`）。当场景静态总表面积或实际体素化砖块数超出硬件安全上限（`240,000` 砖块）时，工具会自动按比例放宽实际 `texel_size`。 |
| `--vlm-levels <U32>` | `u32` | `2` | 空间多级 LOD 层数（有效范围 `1..=3`）。默认 `2` 层：Level 0 近景精细采样 + Level 1（`2×` 纹素尺寸）远景与兜底回退。 |
| `--vlm-samples <U32>` | `u32` | `32` | 每个有效纹素在单次间接反弹中的半球路径追踪采样数（配合离线切平面 `3×3` 边缘保持双边降噪）。 |
| `--vlm-bounces <U32>` | `u32` | `2` | 基于空间哈希缓存自举的多跳间接漫反射迭代次数（`0` 表示仅烘焙直接局部光与太阳软阴影，不计算间接反弹）。 |
| `--vlm-sun-angle-deg <F32>` | `f32` | `0.8` | 主平行光（Sun）软阴影视锥角半径（度）。`> 0` 时发射 8 条锥形软阴影射线，`0` 时退化为单条硬阴影射线。 |
| `--vlm-gi-scale <F32>` | `f32` | `1.0` | 间接漫反射反弹强度倍率。 |
| `--vlm-light-clearance <F32>` | `f32` | `0.35` | 静态点光源 / 聚光灯阴影射线终点处的灯具避让半径（米），用于跳过铁艺灯笼/灯罩自身几何体，消除灯架遮挡产生的圆点走样。 |
| `--vlm-light-radius <F32>` | `f32` | `0.15` | 静态点光源 / 聚光灯球面软阴影采样半径（米，6-ray 采样生成柔和半影）。 |
| `--vlm-cpu` | Flag | `false` | 强制使用多线程 CPU 后端烘焙 VLM。默认优先启用 headless `wgpu` Compute Shader GPU 加速（支持 DX12 512 MB 双三角形分片缓冲与分批提交防 TDR），当硬件不满足条件或初始化失败时自动回退至 CPU。 |

---

## 4. Blender 节点角标与自动分流规则

在 `--scene` 模式下，`makemeshlet` 会自动读取并向下继承 glTF 节点 `extras` 中的自定义属性（可配合 `tools/blender_bevy_tagger.py` 在 Blender 中一键打标）：

| `bevy_tag` 值 | 处理方式 |
| --- | --- |
| `MESHLET_STATIC`（默认） | 不透明静态高模：按 `(chunk, material_id, cast_shadows, batch_seq)` 合并为静态 `.meshlet_mesh`，并参与全局 AO 与 VLM 烘焙。 |
| `MESHLET_DYN` | 动态独立高模：保留局部坐标与节点独立实例，去重构建 `.meshlet_mesh`，不参与静态合并与静态 VLM 烘焙。 |
| `MESH3D_LOD` | 标准多级 LOD 网格（植被/镂空/中低模件）：使用 `meshopt` 自动生成多级 `.mesh3d_lod`（默认比例 `1.0, 0.4, 0.15`，距离 `18, 45, 120 m`）。不透明静态 `MESH3D_LOD` 同样会参与全局 AO 与 VLM 遮挡。 |
| `MESH3D_TRANS` | 半透明/玻璃件：保存为单级 `.mesh3d_lod` 并附带剔除距离（默认 `18, 30 m`）。 |
| `GAME_SPAWN` | 仅提取世界位姿与 `bevy_spawn_group` 写入 `gameplay_markers`，不导出渲染网格。 |
| `GAME_COLLIDER` | 跳过渲染网格导出。 |

**自动材质兜底规则**：若图元材质为 `AlphaMode::Mask` 或 `AlphaMode::Blend`（非 `Opaque`），即使节点未手动打标，也会自动分流至 `Mesh3d` 分支，确保 Meshlet 管线始终仅处理不透明几何体。

可选节点 `extras` 字段还包括：`bevy_chunk`、`bevy_lod_ratios`、`bevy_lod_dists`、`bevy_spawn_group`、`bevy_cast_shadow`。

---

## 5. 输出目录结构

以 `--scene --export-textures -i "sponza.glb" -o "out/"` 为例，导出目录结构如下：

```text
out/
├── sponza.meshlet_scene.json        # 场景清单（材质、实例、灯光、HDRI、VLM 索引）
├── sponza.vlm.bin                   # 离线烘焙的 Virtual Lightmap（BVVLM001 + LZ4 压缩）
├── meshes/
│   ├── sponza_chunk_*_mat_*_b*.meshlet_mesh   # 静态空间分块 Meshlet 资产
│   ├── sponza_mesh_*_prim_*.meshlet_mesh      # 独立 / 动态 / 自动实例化 Meshlet 资产
│   └── sponza_m3d_*_*_*.mesh3d_lod            # 标准 Mesh3d 多级 LOD 资产
├── textures/
│   └── sponza_tex_*.png                       # 导出并按需降采样后的 PBR 贴图
└── environment/
    └── <hdri_file>                            # 从 scene.extras 或旁路 .bevy_hdri.json 打包的环境贴图
```

---

## 6. Virtual Lightmap 技术思路来源与授权状态说明

### 思路来源与致谢

本工具及配套 Bevy 渲染管线中的 **Surface Tangent-Plane SpatialHash Virtual Lightmap（VLM）** 模块，在高层架构思路上受到以下研究项目的启发：

- **参考项目**：[johanholwerda/virtualLightmaps — Irradiance cache with hashed lightmap tiles](https://github.com/johanholwerda/virtualLightmaps)（作者：Johan Holwerda）
- **受到启发的核心思路**：摆脱传统手工/自动展开的第二套光照 UV（UV2），直接以世界空间坐标与法向构建开放寻址空间哈希表来索引紧凑的 `8×8` 光照贴图砖块（Hashed Lightmap Tiles），并在多跳间接光照计算阶段复用该空间哈希结构本身作为 Radiance Cache 进行自举迭代。

### 独立实现说明（Independent Clean-Room Implementation）

本仓库中的所有相关代码——包括离线 Rust 烘焙器（`tools/makemeshlet/src/vlm_baker.rs`）、headless `wgpu` Compute Shader 加速后端（`tools/makemeshlet/src/vlm_gpu.rs`、`vlm_bake.wgsl`）、`BVVLM001` 二进制压缩格式以及 Bevy 运行时采样管线（`crates/bevy_pbr/src/light_probe/virtual_lightmap.rs`、`virtual_lightmap.wesl`）——均面向 Bevy Meshlet 渲染架构独立设计与从零编写，**未直接使用、复制或逐行移植原仓库的源代码**（底层哈希函数仅采用公开通用的 Thomas Wang 与 PCG 整数哈希原语）。同时，本实现针对 Bevy 离线静态烘焙与低功耗 GPU 运行时需求进行了多项独立扩展设计：

- 6 主轴法向切平面 + 深度切片（`w_slice`）表面光栅化，以及面向高模 3D 浮雕网格的最外层表皮择优与跨切片提升机制；
- 带 1px 跨邻块无缝边框的 `10×10` 物理砖块与硬件双线性 `Rgba16Float` 2D 图集打包；
- `RGB`（局部光阴影漫反射 + 多跳间接 GI）与 `A`（主平行光软阴影遮罩 Shadow Mask）四通道联合编码；
- 适配 DX12 512 MB 单 Storage Buffer 上限的双三角形分片缓冲与分批提交调度。

### 原项目许可证与沟通状态（Attribution & License Status）

> [!NOTE]
> 截至本文档编写时，原参考仓库 [`johanholwerda/virtualLightmaps`](https://github.com/johanholwerda/virtualLightmaps) 尚未在其 GitHub 页面明确声明开源许可证（License）。
>
> 本模块借鉴其高层架构思路并重新实现。出于对原作者原创研究贡献的尊重，维护者已通过公开社交渠道（X / Twitter）主动联系原作者 Johan Holwerda 说明致谢与参考关系，并同步确认署名与许可意愿。后续将根据原作者的回复及时更新本节说明。
