# makemeshlet

独立离线预处理工具：将 Blender 导出的 `.gltf` / `.glb` 3D 模型自动清洗并烘焙为 Bevy MeshletMesh 资产（`.meshlet_mesh`）。

完整说明文档已汇总至根目录：
👉 **[MAKEMESHLET_GUIDE.md](../../MAKEMESHLET_GUIDE.md)**

## 快速使用

```powershell
# 编译
cargo build --release --package makemeshlet

# 运行帮助
.\target\release\makemeshlet.exe --help

# 烘焙指定模型
.\target\release\makemeshlet.exe -i "path/to/model.gltf" -o "assets/models/"
```
