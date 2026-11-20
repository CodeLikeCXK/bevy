# makemeshlet User Guide

[中文](README.md) | **English**

`makemeshlet` is a standalone offline scene and geometry preprocessing CLI tool for Bevy. It cleans, partitions, and bakes Blender-exported `.gltf` / `.glb` models and large-scale scenes into:

- **Meshlet assets (`.meshlet_mesh`)**: degenerate-triangle sanitization, spatial grid chunking, per-batch triangle capping, automatic instancing deduplication, and offline static per-vertex AO baking.
- **Standard multi-level LOD assets (`.mesh3d_lod`)**: automatic `meshopt` LOD generation for foliage, alpha-masked geometry (`AlphaMode::Mask`), and transmissive/translucent geometry (`AlphaMode::Blend`).
- **Static Surface Virtual Lightmaps (`.vlm.bin`)**: UV2-free spatial-hashed `8×8` surface tile path-tracing baker with headless `wgpu` GPU Compute acceleration (with automatic multi-threaded CPU fallback), packing static local-light + multi-bounce indirect diffuse GI (`RGB`) and primary directional sun soft-shadow visibility (`A`).
- **Unified scene manifest (`.meshlet_scene.json`)**: deduplicated PBR materials, Meshlet & Mesh3d instances, spatial chunk metadata, punctual lights, `World HDRI`, gameplay markers, and `.vlm.bin` references for direct streaming in Bevy.

---

## Table of Contents

1. [Build & Quick Start](#1-build--quick-start)
2. [Common Workflows](#2-common-workflows)
3. [Complete CLI Reference](#3-complete-cli-reference)
   - [3.1 Basic I/O & Quantization](#31-basic-io--quantization)
   - [3.2 Scene Export & Spatial Streaming (`--scene`)](#32-scene-export--spatial-streaming---scene)
   - [3.3 Static Per-Vertex AO Baking](#33-static-per-vertex-ao-baking)
   - [3.4 Virtual Lightmap (VLM) Baking](#34-virtual-lightmap-vlm-baking)
4. [Blender Node Tags & Automatic Routing](#4-blender-node-tags--automatic-routing)
5. [Output Directory Layout](#5-output-directory-layout)
6. [Virtual Lightmap Conceptual Inspiration & Attribution](#6-virtual-lightmap-conceptual-inspiration--attribution)

---

## 1. Build & Quick Start

Build the release binary from the Bevy repository root (`--release` is strongly recommended for high-poly Meshlet simplification and ray tracing):

```powershell
cargo build --release --package makemeshlet
```

Print the built-in CLI help:

```powershell
.\target\release\makemeshlet.exe --help
```

> **Windows MSVC Linker Tip**: If polluted `LIB` / `INCLUDE` environment variables in your shell cause `LNK1104: cannot open file 'msvcrt.lib'`, clear them in the current terminal before building:
> ```powershell
> Remove-Item Env:LIB, Env:LIBPATH, Env:INCLUDE -ErrorAction SilentlyContinue
> cargo build --release --package makemeshlet
> ```

---

## 2. Common Workflows

### 2.1 Single-Model Conversion (Default Mode)
Convert each primitive of a `.gltf` / `.glb` file into `.meshlet_mesh` assets (with offline per-vertex AO enabled by default):

```powershell
.\target\release\makemeshlet.exe `
  -i "path\to\model.glb" `
  -o "assets\models\"
```

### 2.2 Full Scene Bake (Meshlets + LODs + Textures + AO + VLM)
Process an entire level (e.g., multi-million-triangle Sponza) in one pass: spatial chunking, resized texture export, global per-vertex AO, and GPU-accelerated Virtual Lightmap baking:

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

### 2.3 Lightmap-Only Re-Bake (`--vlm-only`)
When scene meshes have already been exported and you only want to iterate on lighting parameters or regenerate `.vlm.bin`, use `--vlm-only` to skip texture export, per-vertex AO, and all Meshlet/LOD builds:

```powershell
.\target\release\makemeshlet.exe `
  --vlm-only `
  --vlm-texel-size 0.06 `
  --vlm-samples 32 `
  --vlm-bounces 2 `
  -i "path\to\scene.glb" `
  -o "path\to\scene_out"
```

> If `<stem>.meshlet_scene.json` already exists in the output directory, `--vlm-only` preserves all existing `meshes/`, materials, and instance lists and updates only `vlm_file` (and `world_hdri` if previously unset).

### 2.4 CPU-Only VLM or Draft Scene Export
```powershell
# Export scene meshes + vertex AO without Virtual Lightmap
.\target\release\makemeshlet.exe --scene --no-vlm -i "scene.glb" -o "scene_out"

# Re-bake VLM only using the multi-threaded CPU backend (skipping headless wgpu GPU)
.\target\release\makemeshlet.exe --vlm-only --vlm-cpu -i "scene.glb" -o "scene_out"

# Fast draft scene export (disable both AO and VLM)
.\target\release\makemeshlet.exe --scene --no-ao --no-vlm -i "scene.glb" -o "scene_out"
```

---

## 3. Complete CLI Reference

### 3.1 Basic I/O & Quantization

| Flag | Short | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `--input <PATH>` | `-i` | Path | **Required** | Input `.gltf` or `.glb` file path. |
| `--output <PATH>` | `-o` | Path | Input directory | Output file or directory (single-model mode), or root output directory (`--scene` / `--vlm-only` mode). |
| `--quantization <U8>` | `-q` | `u8` | `4` | Fixed-point vertex position quantization factor for Meshlets: `x` means $1 / 2^x\text{ cm}$ precision (default `4` = `1/16 cm = 0.0625 cm`). |

---

### 3.2 Scene Export & Spatial Streaming (`--scene`)

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--scene` | Flag | `false` | Enable scene export mode: traverse node transforms, parse Blender custom tags, batch & route geometry, extract materials/lights/HDRI, bake AO & VLM, and write `<stem>.meshlet_scene.json`. |
| `--no-merge` | Flag | `false` | Disable per-material static batch merging in scene mode, preserving individual node instances while still deduplicating shared Mesh primitives. |
| `--dynamic-prefix <PREFIX>` | `Vec<String>` | Empty | Node name prefix excluded from static merging (repeatable, e.g. `--dynamic-prefix Wheel_ --dynamic-prefix Door_`). Matching nodes are routed to independent dynamic instances and excluded from static VLM occlusion. |
| `--export-textures` | Flag | `false` | Export embedded or referenced glTF textures as PNG files under `<output>/textures/`. |
| `--max-texture-size <U32>` | `u32` | `1024` | Maximum texture edge length in pixels when exporting; larger textures are proportionally downscaled using bilinear filtering. `0` disables resizing (`1024` or `512` recommended for integrated GPUs). |
| `--grid-size <F32>` | `f32` | `12.0` | Spatial grid cell size (meters) for automatically splitting default `chunk="main"` static batches into streamable chunks. `0` disables spatial splitting. |
| `--max-batch-tris <USIZE>` | `usize` | `250000` | Maximum triangle count per static Meshlet batch before automatically splitting into a new batch. `0` disables the cap. |
| `--auto-instance-min <USIZE>` | `usize` | `3` | Automatically route any primitive referenced $\ge N$ times (by non-mirrored renderable nodes) to local instancing instead of static merging. `0` disables automatic instancing. |
| `--lod-keep-blend` | Flag | `false` | Allow explicitly tagged `MESH3D_LOD` meshes to keep `AlphaMode::Blend`. When off (default), `Blend` on `MESH3D_LOD` is converted to `AlphaMode::Mask` to avoid transparent overdraw. |

---

### 3.3 Static Per-Vertex AO Baking

Enabled by default in both single-model and `--scene` modes (automatically disabled in `--vlm-only` mode).

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--no-ao` | Flag | `false` | Disable offline static per-vertex AO baking (all vertex AO values default to `1.0`). |
| `--ao-samples <U32>` | `u32` | `32` | Cosine-weighted hemisphere ray count per vertex. |
| `--ao-radius <F32>` | `f32` | `1.5` | Maximum occlusion ray distance (meters). Keeping this at contact-shadow scale avoids double-darkening interiors alongside GI / IrradianceVolume. |
| `--ao-power <F32>` | `f32` | `1.0` | AO contrast exponent (`> 1.0` darkens crevices more strongly). |
| `--ao-bias <F32>` | `Option<f32>` | Auto | Ray origin offset along surface normal (meters). Defaults to $\operatorname{clamp}(\text{ao\_radius} \times 0.0025,\ 0.0005,\ 0.01)$. |

---

### 3.4 Virtual Lightmap (VLM) Baking

Enabled by default in `--scene` mode, or invoked standalone via `--vlm-only`.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--no-vlm` | Flag | `false` | Disable offline Virtual Lightmap (`.vlm.bin`) baking. Overridden if `--vlm-only` is specified. |
| `--vlm-only` | Flag | `false` | Bake only `<stem>.vlm.bin` (implies `--scene`, skips texture export, vertex AO, and Meshlet/LOD builds). Non-destructively updates `vlm_file` in an existing `.meshlet_scene.json` if present. |
| `--vlm-texel-size <F32>` | `f32` | `0.06` | Level 0 texel world size (meters; default `0.06 m = 6 cm`, so one `8×8` tile covers `48 cm × 48 cm`). Automatically relaxed if scene surface area or generated tile count exceeds the hardware safety budget (`240,000` tiles). |
| `--vlm-levels <U32>` | `u32` | `2` | Number of spatial LOD levels (`1..=3`). Default `2`: fine Level 0 near camera + `2×` coarser Level 1 for distance & fallback. |
| `--vlm-samples <U32>` | `u32` | `32` | Hemisphere path-tracing samples per active texel per bounce (paired with tangent-plane `3×3` edge-preserving bilateral filtering). |
| `--vlm-bounces <U32>` | `u32` | `2` | Multi-bounce indirect diffuse iterations bootstrapped from the spatial-hash radiance cache (`0` bakes direct local lights & sun shadows only). |
| `--vlm-sun-angle-deg <F32>` | `f32` | `0.8` | Primary directional sun angular half-cone radius in degrees (`> 0` casts 8 soft-shadow cone rays; `0` casts 1 hard-shadow ray). |
| `--vlm-gi-scale <F32>` | `f32` | `1.0` | Multiplier applied to baked indirect diffuse bounce irradiance. |
| `--vlm-light-clearance <F32>` | `f32` | `0.35` | Occlusion clearance radius (meters) around static Point/Spot light positions to skip self-shadowing from lantern cages / light fixtures. |
| `--vlm-light-radius <F32>` | `f32` | `0.15` | Spherical soft-shadow sampling radius (meters) for static Point/Spot lights (6 rays per light). |
| `--vlm-cpu` | Flag | `false` | Force the multi-threaded CPU VLM baker. By default, headless `wgpu` GPU Compute acceleration is used first (with DX12 512 MB dual triangle buffers and TDR-safe chunked dispatch), falling back automatically to CPU if unavailable. |

---

## 4. Blender Node Tags & Automatic Routing

In `--scene` mode, `makemeshlet` parses and inherits custom properties from glTF node `extras` (compatible with `tools/blender_bevy_tagger.py`):

| `bevy_tag` | Behavior |
| --- | --- |
| `MESHLET_STATIC` (default) | Opaque static high-poly meshes: merged by `(chunk, material_id, cast_shadows, batch_seq)` into `.meshlet_mesh` assets and included in global AO & VLM baking. |
| `MESHLET_DYN` | Dynamic high-poly meshes: kept in local coordinates as deduplicated independent `.meshlet_mesh` instances; excluded from static merging and VLM baking. |
| `MESH3D_LOD` | Standard multi-LOD meshes (foliage, cutouts, mid/low-poly props): simplified with `meshopt` into `.mesh3d_lod` (default ratios `1.0, 0.4, 0.15`; distances `18, 45, 120 m`). Opaque static `MESH3D_LOD` meshes still act as AO & VLM occluders. |
| `MESH3D_TRANS` | Translucent/glass meshes: exported as single-level `.mesh3d_lod` with culling distance (default `18, 30 m`). |
| `GAME_SPAWN` | Records world transform and `bevy_spawn_group` into `gameplay_markers`; no render mesh exported. |
| `GAME_COLLIDER` | Skipped from render mesh export. |

**Automatic Material Fallback**: Any primitive using `AlphaMode::Mask` or `AlphaMode::Blend` is automatically routed to the `Mesh3d` branch even without explicit node tags, keeping the Meshlet pipeline strictly opaque.

Additional supported node `extras` fields: `bevy_chunk`, `bevy_lod_ratios`, `bevy_lod_dists`, `bevy_spawn_group`, `bevy_cast_shadow`.

---

## 5. Output Directory Layout

Example output for `--scene --export-textures -i "sponza.glb" -o "out/"`:

```text
out/
├── sponza.meshlet_scene.json        # Scene manifest (materials, instances, lights, HDRI, VLM path)
├── sponza.vlm.bin                   # Offline-baked Virtual Lightmap (BVVLM001 + LZ4 compressed)
├── meshes/
│   ├── sponza_chunk_*_mat_*_b*.meshlet_mesh   # Static spatially-chunked Meshlet batches
│   ├── sponza_mesh_*_prim_*.meshlet_mesh      # Independent / dynamic / auto-instanced Meshlets
│   └── sponza_m3d_*_*_*.mesh3d_lod            # Standard Mesh3d multi-LOD assets
├── textures/
│   └── sponza_tex_*.png                       # Exported (and optionally downscaled) PBR textures
└── environment/
    └── <hdri_file>                            # Bundled World HDRI from scene.extras or .bevy_hdri.json
```

---

## 6. Virtual Lightmap Conceptual Inspiration & Attribution

### Conceptual Inspiration (*Inspired by*)

The **Surface Tangent-Plane SpatialHash Virtual Lightmap (VLM)** module in this tool and the companion Bevy rendering pipeline is **conceptually inspired by**:

- **Reference Project**: [johanholwerda/virtualLightmaps — Irradiance cache with hashed lightmap tiles](https://github.com/johanholwerda/virtualLightmaps) by Johan Holwerda.
- **High-Level Ideas**: Eliminating traditional secondary lightmap UVs (UV2) by indexing compact `8×8` lightmap tiles through an open-addressing world-space spatial hash table, and bootstrapping multi-bounce indirect lighting by querying the spatial hash table itself as a radiance cache.

### Independent Clean-Room Implementation

All code in this repository—including the offline Rust baker (`tools/makemeshlet/src/vlm_baker.rs`), the headless `wgpu` compute backend (`tools/makemeshlet/src/vlm_gpu.rs`, `vlm_bake.wgsl`), the `BVVLM001` compressed binary format, and the Bevy runtime sampling pipeline (`crates/bevy_pbr/src/light_probe/virtual_lightmap.rs`, `virtual_lightmap.wesl`)—was designed and written independently from scratch for Bevy's Meshlet renderer. **No source code was copied or directly ported from the reference repository** (underlying integer hashing uses standard public-domain Thomas Wang and PCG hash primitives). In addition, this implementation introduces several independent architectural extensions tailored for offline static baking and low-power GPUs:

- 6-axis surface tangent-plane rasterization with depth slicing (`w_slice`), outermost-shell filtering for high-poly 3D relief meshes, and cross-slice surface promotion;
- `10×10` physical atlas tiles (8×8 core + 1px cross-tile neighbor border) packed into a hardware bilinearly filterable `Rgba16Float` 2D texture atlas;
- Joint 4-channel encoding of static local-light + multi-bounce indirect diffuse irradiance (`RGB`) and primary directional sun soft-shadow visibility (`A`);
- Dual triangle storage-buffer splitting and TDR-safe chunked GPU dispatch respecting DX12's 512 MB single storage-buffer limit.

### Attribution & License Status

> [!NOTE]
> At the time of writing, the reference repository [`johanholwerda/virtualLightmaps`](https://github.com/johanholwerda/virtualLightmaps) does not declare an explicit open-source license on GitHub.
>
> This VLM implementation inspired by high-level conceptual inspiration and is written independently (not a derivative work of the reference source code), out of respect for the original author's research contribution, the maintainer has reached out to Johan Holwerda via X / Twitter to acknowledge the inspiration and confirm preferred attribution/licensing terms. This notice will be updated once a response is received.
