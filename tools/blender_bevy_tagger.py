# -*- coding: utf-8 -*-
"""
Blender -> Bevy (Meshlet + Mesh3d LOD) 资产打标与一键导出插件

设计目的：
1. 在 Solid（实体）视图下完成场景整理、资产清洗与打标，避免在低算力集显上触发 EEVEE 着色器编译卡死。
2. 所有角标（Tag）均集中在顶部的 `BEVY_TAG_REGISTRY` 配置表中，后续只需在此增删条目即可自动生成 UI 按钮。
3. 导出 GLB 时将自定义属性写入节点的 `extras` JSON 字段，供 `makemeshlet --scene` 自动分流处理：
   - 静态不透明高模 -> 按 chunk + material 合并为 .meshlet_mesh
   - 植物/镂空件     -> 使用 meshopt 自动生成多级 LOD (.mesh3d_lod) + VisibilityRange
   - 半透明玻璃/酒瓶 -> 保留透明材质 + 远距剔除
   - Gameplay 标记点 -> 仅导出坐标供引擎编辑器读取
"""

bl_info = {
    "name": "Bevy Pipeline Tagger & GLB Exporter",
    "author": "ActionRenderer Pipeline",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar (N) > Bevy Pipeline",
    "description": "为 Bevy Meshlet / Mesh3d LOD 混合管线批量打标、清洗 FBX 材质并导出 GLB",
    "category": "Import-Export",
}

import array
import json
import math
import os
import random
from collections import defaultdict, deque
import bmesh
import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty


# ==============================================================================
# 1. 角标类型注册表（人工扩展点）
#    如需新增角标类型，直接在此列表中追加元组即可，面板按钮会自动更新：
#    (枚举ID, 显示名称, 说明文字, Solid视图颜色RGBA, 默认写入物体的自定义属性字典)
# ==============================================================================
BEVY_TAG_REGISTRY = [
    (
        "MESHLET_STATIC",
        "Meshlet 静态高模",
        "静态不透明建筑/大件：按 chunk + 材质合并烘焙为 .meshlet_mesh",
        (0.20, 0.55, 0.95, 1.0),  # 蓝色
        {"bevy_chunk": "main"},
    ),
    (
        "MESH3D_LOD",
        "Mesh3d 多级 LOD",
        "植物/树叶/镂空件：离线通过 meshopt 生成多级 LOD 并用 VisibilityRange 切换",
        (0.20, 0.80, 0.35, 1.0),  # 绿色
        {"bevy_chunk": "main", "bevy_lod_ratios": "1.0,0.4,0.15", "bevy_lod_dists": "18,45,120"},
    ),
    (
        "MESH3D_TRANS",
        "Mesh3d 半透明件",
        "玻璃/酒瓶/水面等半透明材质：不走 Meshlet，保留半透明并按距离剔除",
        (0.15, 0.85, 0.90, 1.0),  # 青色
        {"bevy_chunk": "main", "bevy_lod_ratios": "1.0", "bevy_lod_dists": "18,30"},
    ),
    (
        "MESHLET_DYN",
        "Meshlet 动态独立件",
        "需要独立运动/旋转的高模零件（如车轮/门）：不参与静态合并，保留局部坐标",
        (0.95, 0.55, 0.15, 1.0),  # 橙色
        {"bevy_chunk": "dynamic"},
    ),
    (
        "GAME_SPAWN",
        "Gameplay 标记点",
        "出生点/刷怪点/相机锚点：不导出渲染网格，仅写入场景清单的 gameplay_markers",
        (0.95, 0.85, 0.15, 1.0),  # 黄色
        {"bevy_spawn_group": "default"},
    ),
    (
        "GAME_COLLIDER",
        "仅物理碰撞体",
        "空气墙/碰撞盒：跳过渲染转换，仅保留用于物理生成",
        (0.90, 0.20, 0.20, 1.0),  # 红色
        {},
    ),
]


# 材质名称中包含这些关键字时，FBX 修复脚本会保留其金属度；其余默认视为非金属
METAL_KEYWORDS = (
    "metal", "iron", "steel", "brass", "copper", "gold", "silver",
    "chrome", "alu", "bronze", "tin", "zinc", "titanium",
)

# 材质或物体名称包含这些关键字时，一键清洗脚本会将其识别为植物（转为 AlphaClip）
FOLIAGE_KEYWORDS = (
    "leaf", "leaves", "foliage", "ivy", "tree", "bush", "plant",
    "fern", "grass", "flower", "vine", "branch", "hedge",
)


def _find_tag_entry(tag_id: str):
    """根据 tag_id 从注册表查找对应配置。"""
    for entry in BEVY_TAG_REGISTRY:
        if entry[0] == tag_id:
            return entry
    return None


def _collect_target_objects(context, include_collection: bool):
    """收集需要打标的对象：选中对象，或当前激活集合下的全部对象。"""
    if include_collection and context.collection:
        return list(context.collection.all_objects)
    return list(context.selected_objects)


def _ensure_solid_object_color_view(context):
    """将当前 3D 视口设为 Solid + Object Color 模式，方便直观查看各物体的角标分类。"""
    for area in context.screen.areas:
        if area.type == "VIEW_3D":
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.shading.type = "SOLID"
                    space.shading.color_type = "OBJECT"


# ==============================================================================
# 2. Blender 属性组（存储在 Scene 中，控制面板交互参数）
# ==============================================================================
class BevyPipelineSettings(bpy.types.PropertyGroup):
    target_scope: EnumProperty(
        name="作用范围",
        description="选择将角标应用到当前选中物体，还是整个激活 Collection",
        items=[
            ("SELECTED", "选中物体", "仅对当前选中的物体打标"),
            ("COLLECTION", "激活集合", "对当前 Outliner 激活的 Collection 内所有物体打标"),
        ],
        default="SELECTED",
    )
    chunk_name: StringProperty(
        name="Chunk 分块名",
        description="用于引擎流送的区块标识（如 exterior_main, interior, wine）",
        default="main",
    )
    lod_ratios: StringProperty(
        name="LOD 面数比例",
        description="逗号分隔的简化比例（仅 MESH3D_LOD 生效，例如 1.0,0.4,0.15）",
        default="1.0,0.4,0.15",
    )
    lod_dists: StringProperty(
        name="LOD 切换距离",
        description="逗号分隔的距离上限（米），最后一级表示完全剔除距离（例如 18,45,120）",
        default="18,45,120",
    )
    spawn_group: StringProperty(
        name="Gameplay 分组",
        description="用于 GAME_SPAWN 标记点的分组名称",
        default="default",
    )
    export_path: StringProperty(
        name="导出 GLB 路径",
        description="导出 .glb 文件的目标路径（支持 // 相对路径）",
        subtype="FILE_PATH",
        default="//scene_tagged.glb",
    )
    export_apply_modifiers: BoolProperty(
        name="应用修改器",
        description="导出 GLB 时应用物体上的修改器",
        default=True,
    )
    export_visible_only: BoolProperty(
        name="仅导出可见/可渲染物体",
        description="过滤隐藏或关闭渲染图标的备份高模、切割盒与辅助体",
        default=True,
    )
    max_subsurf_level: IntProperty(
        name="细分修改器上限",
        description="限制 SUBSURF 修改器的最高细分级别，避免导出千万面爆炸",
        default=1,
        min=0,
        max=4,
    )
    merge_dist: FloatProperty(
        name="重合点焊接距离",
        description="拓扑净化时合并重合顶点的阈值（米），0 表示不焊接",
        default=0.0001,
        min=0.0,
        max=0.01,
        precision=5,
    )
    auto_create_uv: BoolProperty(
        name="缺失 UV 时自动补投影",
        description="Meshlet 强依赖 UV_0，对缺少 UV 层的网格自动补三平面投影 UV",
        default=True,
    )
    leaf_remove_ratio: FloatProperty(
        name="抽稀删面比例",
        description="要随机剔除的离散叶片比例（0.80 表示删掉 80% 叶片，保留 20%）",
        default=0.80,
        min=0.05,
        max=0.98,
        subtype="FACTOR",
    )
    leaf_keep_scale: FloatProperty(
        name="保留叶片放大倍率",
        description="对保留下来的叶片绕各自质心放大倍率，以补偿抽稀后的视觉蓬松度与外形",
        default=1.35,
        min=0.5,
        max=3.0,
        precision=2,
    )
    leaf_max_island_faces: IntProperty(
        name="单叶片最大面数",
        description="连通多边形数量超过该阈值的块视为连续树干/大枝干，将完整保留不做抽稀与缩放",
        default=32,
        min=1,
        max=1000,
    )
    leaf_random_seed: IntProperty(
        name="抽稀随机种子",
        description="空间分层抽稀的随机数种子，修改可切换保留叶片组合",
        default=42,
        min=0,
    )
    mat_merge_scalar_tol: FloatProperty(
        name="同质材质合并容差",
        description="将贴图通道一致且颜色/粗糙度差异在该容差内的材质合并（减少 Meshlet 全屏 Draw Pass）",
        default=0.08,
        min=0.0,
        max=0.5,
        precision=2,
    )
    mat_merge_ignore_vertex_color: BoolProperty(
        name="合并忽略顶点色混合",
        description="Meshlet 管线暂不保留顶点色，勾选可将仅顶点色节点有差异的同贴图材质合并",
        default=True,
    )
    audit_summary: StringProperty(default="")
    audit_heavy_top: StringProperty(default="")


# ==============================================================================
# 3. 操作算子（Operators）
# ==============================================================================
class BEVY_OT_apply_tag(bpy.types.Operator):
    """为选中物体或集合写入 Bevy 角标及参数"""
    bl_idname = "bevy.apply_tag"
    bl_label = "应用 Bevy 角标"
    bl_options = {"REGISTER", "UNDO"}

    tag_id: StringProperty()

    def execute(self, context):
        entry = _find_tag_entry(self.tag_id)
        if not entry:
            self.report({"ERROR"}, f"未知角标类型: {self.tag_id}")
            return {"CANCELLED"}

        _, label, _, color_rgba, defaults = entry
        settings = context.scene.bevy_pipeline_settings
        use_collection = (settings.target_scope == "COLLECTION")
        targets = _collect_target_objects(context, use_collection)

        if not targets:
            self.report({"WARNING"}, "未选中任何物体或激活集合为空")
            return {"CANCELLED"}

        for obj in targets:
            # 写入核心角标 ID
            obj["bevy_tag"] = self.tag_id

            # 写入分块与参数（优先采用注册表默认值，若用户面板有修改则按场景需求覆盖）
            if "bevy_chunk" in defaults:
                obj["bevy_chunk"] = settings.chunk_name.strip() or defaults["bevy_chunk"]
            if "bevy_lod_ratios" in defaults:
                obj["bevy_lod_ratios"] = settings.lod_ratios.strip() or defaults["bevy_lod_ratios"]
            if "bevy_lod_dists" in defaults:
                obj["bevy_lod_dists"] = settings.lod_dists.strip() or defaults["bevy_lod_dists"]
            if "bevy_spawn_group" in defaults:
                obj["bevy_spawn_group"] = settings.spawn_group.strip() or defaults["bevy_spawn_group"]

            # 设置 Solid 视图颜色，便于无材质着色器编译环境下可视化检查分类结果
            obj.color = color_rgba

        _ensure_solid_object_color_view(context)
        self.report({"INFO"}, f"已为 {len(targets)} 个物体打标为 [{label}]")
        return {"FINISHED"}


class BEVY_OT_clear_tag(bpy.types.Operator):
    """清除选中物体上的 Bevy 角标属性并恢复默认颜色"""
    bl_idname = "bevy.clear_tag"
    bl_label = "清除 Bevy 角标"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.bevy_pipeline_settings
        targets = _collect_target_objects(context, settings.target_scope == "COLLECTION")
        keys = ("bevy_tag", "bevy_chunk", "bevy_lod_ratios", "bevy_lod_dists", "bevy_spawn_group")
        count = 0
        for obj in targets:
            for k in keys:
                if k in obj:
                    del obj[k]
            obj.color = (1.0, 1.0, 1.0, 1.0)
            count += 1
        self.report({"INFO"}, f"已清除 {count} 个物体的 Bevy 角标")
        return {"FINISHED"}


class BEVY_OT_select_by_tag(bpy.types.Operator):
    """选中场景中所有带指定角标的物体（用于检查是否漏标）"""
    bl_idname = "bevy.select_by_tag"
    bl_label = "按角标筛选选中"
    bl_options = {"REGISTER", "UNDO"}

    tag_id: StringProperty()

    def execute(self, context):
        bpy.ops.object.select_all(action="DESELECT")
        matched = 0
        for obj in context.scene.objects:
            if obj.get("bevy_tag") == self.tag_id:
                obj.select_set(True)
                matched += 1
        self.report({"INFO"}, f"已选中 {matched} 个角标为 {self.tag_id} 的物体")
        return {"FINISHED"}


class BEVY_OT_fix_fbx_materials(bpy.types.Operator):
    """清洗 FBX 导入材质：修复非金属物体的错误金属度，并将植物材质设为 AlphaClip"""
    bl_idname = "bevy.fix_fbx_materials"
    bl_label = "一键清洗 FBX 材质 (金属度 + 植物AlphaClip)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        fixed_metallic = 0
        fixed_foliage = 0

        for mat in bpy.data.materials:
            if not mat.use_nodes or not mat.node_tree:
                continue

            mat_name_lower = mat.name.lower()
            is_metal = any(kw in mat_name_lower for kw in METAL_KEYWORDS)
            is_foliage = any(kw in mat_name_lower for kw in FOLIAGE_KEYWORDS)

            # 查找 Principled BSDF 节点调整 Metallic 与 Alpha
            for node in mat.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    metallic_input = node.inputs.get("Metallic")
                    if metallic_input and not is_metal and not metallic_input.is_linked:
                        if metallic_input.default_value > 0.0:
                            metallic_input.default_value = 0.0
                            fixed_metallic += 1

            # 兼容不同 Blender 版本的 Alpha Clip 设置（导出 glTF 时会映射为 AlphaMode::Mask）
            if is_foliage and hasattr(mat, "blend_method"):
                if mat.blend_method == "BLEND":
                    mat.blend_method = "CLIP"
                    if hasattr(mat, "alpha_threshold"):
                        mat.alpha_threshold = 0.5
                    fixed_foliage += 1

        self.report(
            {"INFO"},
            f"材质清洗完成：修复 {fixed_metallic} 个非金属度，转换 {fixed_foliage} 个植物 AlphaClip",
        )
        return {"FINISHED"}


def _is_object_exportable(obj, visible_only: bool) -> bool:
    """判断物体在导出过滤下是否保留。"""
    if not visible_only:
        return True
    if obj.hide_render:
        return False
    try:
        return bool(obj.visible_get())
    except Exception:
        return not obj.hide_viewport


def _estimate_obj_tris(obj) -> int:
    """快速估算物体三角面数（含 SUBSURF 倍率，无需触发重型 Depsgraph 求值）。"""
    if obj.type != "MESH" or not obj.data:
        return 0
    base_tris = sum(max(0, len(p.vertices) - 2) for p in obj.data.polygons)
    mult = 1
    for mod in obj.modifiers:
        if mod.type == "SUBSURF" and (mod.show_render or mod.show_viewport):
            lvl = max(0, getattr(mod, "render_levels", getattr(mod, "levels", 0)))
            mult *= 4 ** min(lvl, 4)
    return base_tris * mult


def _material_has_gltf_risk(mat) -> bool:
    """检查材质是否包含 glTF 导出可能丢失的节点结构（如无有效 Principled BSDF 或纯程序节点）。"""
    if not mat:
        return False
    if not mat.use_nodes or not mat.node_tree:
        return False
    nodes = mat.node_tree.nodes
    has_principled = any(n.type == "BSDF_PRINCIPLED" for n in nodes)
    if not has_principled:
        return True
    has_procedural = any(
        n.type in {"TEX_NOISE", "TEX_VORONOI", "TEX_WAVE", "TEX_MUSGRAVE", "MIX_SHADER"}
        for n in nodes
    )
    has_image = any(n.type == "TEX_IMAGE" for n in nodes)
    return has_procedural and not has_image


def _mesh_has_degenerate_polys(mesh) -> bool:
    """用底层数组快速检测网格是否存在零面积退化多边形。"""
    poly_count = len(mesh.polygons)
    if poly_count == 0:
        return False
    areas = array.array("f", [0.0]) * poly_count
    mesh.polygons.foreach_get("area", areas)
    return any(a <= 1e-8 for a in areas)


def _obj_has_bad_scale(obj) -> bool:
    """检查物体是否存在零缩放或负缩放（镜像）。"""
    sx, sy, sz = obj.scale.x, obj.scale.y, obj.scale.z
    if min(abs(sx), abs(sy), abs(sz)) < 1e-4:
        return True
    return (sx * sy * sz) < 0.0


def _collect_audit_issues(context, visible_only: bool, max_subsurf: int):
    """扫描当前场景并返回体检统计与按问题分类的物体集合。"""
    issues = {
        "HEAVY_SUBSURF": [],
        "BAD_MATERIAL": [],
        "MISSING_UV": [],
        "BAD_SCALE": [],
        "DEGENERATE_MESH": [],
        "ZERO_OR_AREA_LIGHT": [],
    }
    mesh_deg_cache = {}
    mat_risk_cache = {}
    unique_export_mats = set()
    heavy_list = []
    hidden_skipped = 0
    total_est_tris = 0
    export_mesh_count = 0

    for obj in context.scene.objects:
        if not _is_object_exportable(obj, visible_only):
            hidden_skipped += 1
            continue

        if obj.type == "LIGHT" and obj.data:
            if obj.data.type == "AREA" or obj.data.energy <= 0.01:
                issues["ZERO_OR_AREA_LIGHT"].append(obj)
            continue

        if obj.type != "MESH" or not obj.data:
            continue

        export_mesh_count += 1
        mesh = obj.data
        est_tris = _estimate_obj_tris(obj)
        total_est_tris += est_tris
        heavy_list.append((est_tris, obj.name))

        # 1. 高倍细分修改器
        if any(
            m.type == "SUBSURF"
            and max(getattr(m, "levels", 0), getattr(m, "render_levels", 0)) > max_subsurf
            for m in obj.modifiers
        ):
            issues["HEAVY_SUBSURF"].append(obj)

        # 2. 缺失 UV
        if not mesh.uv_layers:
            issues["MISSING_UV"].append(obj)

        # 3. 异常缩放
        if _obj_has_bad_scale(obj):
            issues["BAD_SCALE"].append(obj)

        # 4. 零面积退化面（按 Mesh ID 缓存）
        m_ptr = mesh.as_pointer()
        if m_ptr not in mesh_deg_cache:
            mesh_deg_cache[m_ptr] = _mesh_has_degenerate_polys(mesh)
        if mesh_deg_cache[m_ptr]:
            issues["DEGENERATE_MESH"].append(obj)

        # 5. 非 glTF 兼容材质风险
        obj_bad_mat = False
        for slot in obj.material_slots:
            mat = slot.material
            if not mat:
                continue
            unique_export_mats.add(mat.name)
            mp = mat.as_pointer()
            if mp not in mat_risk_cache:
                mat_risk_cache[mp] = _material_has_gltf_risk(mat)
            if mat_risk_cache[mp]:
                obj_bad_mat = True
        if obj_bad_mat:
            issues["BAD_MATERIAL"].append(obj)

    heavy_list.sort(key=lambda item: item[0], reverse=True)
    return {
        "export_mesh_count": export_mesh_count,
        "hidden_skipped": hidden_skipped,
        "total_est_tris": total_est_tris,
        "unique_mats": len(unique_export_mats),
        "heavy_top5": heavy_list[:5],
        "issues": issues,
    }


class BEVY_OT_audit_scene(bpy.types.Operator):
    """扫描场景潜在的非游戏资产陷阱，并在面板输出诊断报告"""
    bl_idname = "bevy.audit_scene"
    bl_label = "场景体检 (导出前诊断)"

    def execute(self, context):
        settings = context.scene.bevy_pipeline_settings
        report = _collect_audit_issues(
            context, settings.export_visible_only, settings.max_subsurf_level
        )
        issues = report["issues"]
        lines = [
            f"网格: {report['export_mesh_count']} (过滤隐藏 {report['hidden_skipped']}) | 估算面数: {report['total_est_tris']:,} | 材质: {report['unique_mats']}",
            f"高细分(>L{settings.max_subsurf_level}): {len(issues['HEAVY_SUBSURF'])} | 材质丢失风险: {len(issues['BAD_MATERIAL'])}",
            f"缺UV: {len(issues['MISSING_UV'])} | 退化面Mesh: {len(issues['DEGENERATE_MESH'])} | 异常缩放: {len(issues['BAD_SCALE'])} | 问题灯: {len(issues['ZERO_OR_AREA_LIGHT'])}",
        ]
        settings.audit_summary = "\n".join(lines)
        if report["heavy_top5"]:
            top_str = ", ".join(f"{n}({t:,})" for t, n in report["heavy_top5"][:3])
            settings.audit_heavy_top = f"最重Top3: {top_str}"
        else:
            settings.audit_heavy_top = ""

        self.report({"INFO"}, "场景体检完成，详情已更新至面板诊断区")
        return {"FINISHED"}


class BEVY_OT_select_audit_issue(bpy.types.Operator):
    """在 3D 视口选中指定体检问题分类下的所有物体"""
    bl_idname = "bevy.select_audit_issue"
    bl_label = "选中该类问题物体"
    bl_options = {"REGISTER", "UNDO"}

    issue_kind: StringProperty()

    def execute(self, context):
        settings = context.scene.bevy_pipeline_settings
        report = _collect_audit_issues(
            context, settings.export_visible_only, settings.max_subsurf_level
        )
        matched = report["issues"].get(self.issue_kind, [])
        bpy.ops.object.select_all(action="DESELECT")
        for obj in matched:
            try:
                obj.select_set(True)
            except Exception:
                pass
        if matched:
            context.view_layer.objects.active = matched[0]
        self.report({"INFO"}, f"已选中 {len(matched)} 个 [{self.issue_kind}] 问题对象")
        return {"FINISHED"}


class BEVY_OT_clamp_subsurf(bpy.types.Operator):
    """一键将 SUBSURF 细分修改器级别压低至上限以下，防止千万面爆炸"""
    bl_idname = "bevy.clamp_subsurf"
    bl_label = "限制高倍细分修改器"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.bevy_pipeline_settings
        limit = settings.max_subsurf_level
        targets = (
            list(context.selected_objects)
            if context.selected_objects
            else list(context.scene.objects)
        )
        clamped = 0
        for obj in targets:
            for mod in obj.modifiers:
                if mod.type == "SUBSURF":
                    changed = False
                    if mod.levels > limit:
                        mod.levels = limit
                        changed = True
                    if mod.render_levels > limit:
                        mod.render_levels = limit
                        changed = True
                    if changed:
                        clamped += 1
        self.report({"INFO"}, f"已将 {clamped} 个 SUBSURF 修改器限制为 <= {limit}")
        return {"FINISHED"}


class BEVY_OT_clean_topology(bpy.types.Operator):
    """用 bmesh 净化拓扑：清理零面积退化面、焊接重合点、删孤立点并按需补投影 UV"""
    bl_idname = "bevy.clean_topology"
    bl_label = "一键网格拓扑净化 (防退化崩+补UV)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.bevy_pipeline_settings
        targets = _collect_target_objects(context, settings.target_scope == "COLLECTION")
        if not targets:
            self.report({"WARNING"}, "未选中任何物体或激活集合为空（可全选或切换作用范围）")
            return {"CANCELLED"}

        processed_meshes = set()
        cleaned_meshes = 0
        removed_deg_faces = 0
        removed_loose_verts = 0
        added_uvs = 0

        for obj in targets:
            if obj.type != "MESH" or not obj.data:
                continue
            mesh = obj.data
            m_ptr = mesh.as_pointer()
            if m_ptr in processed_meshes:
                continue
            processed_meshes.add(m_ptr)

            bm = bmesh.new()
            try:
                bm.from_mesh(mesh)
                # 1. 删除零面积退化面
                deg_faces = [f for f in bm.faces if f.calc_area() <= 1e-8]
                if deg_faces:
                    removed_deg_faces += len(deg_faces)
                    bmesh.ops.delete(bm, geom=deg_faces, context="FACES")

                # 2. 焊接极近重合顶点
                if settings.merge_dist > 0.0 and len(bm.verts) > 0:
                    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=settings.merge_dist)

                # 3. 清理游离孤立顶点
                loose_verts = [v for v in bm.verts if not v.link_faces]
                if loose_verts:
                    removed_loose_verts += len(loose_verts)
                    bmesh.ops.delete(bm, geom=loose_verts, context="VERTS")

                # 4. 缺失 UV 时用纯 bmesh 三平面投影补全基础 UVMap（无需切 Edit 模式）
                if settings.auto_create_uv and len(bm.loops.layers.uv) == 0 and len(bm.faces) > 0:
                    uv_layer = bm.loops.layers.uv.verify()
                    for face in bm.faces:
                        n = face.normal
                        ax, ay, az = abs(n.x), abs(n.y), abs(n.z)
                        for loop in face.loops:
                            co = loop.vert.co
                            if az >= ax and az >= ay:
                                loop[uv_layer].uv = (co.x, co.y)
                            elif ay >= ax and ay >= az:
                                loop[uv_layer].uv = (co.x, co.z)
                            else:
                                loop[uv_layer].uv = (co.y, co.z)
                    added_uvs += 1

                bm.to_mesh(mesh)
                mesh.update()
                cleaned_meshes += 1
            finally:
                bm.free()

        self.report(
            {"INFO"},
            f"拓扑净化完成：{cleaned_meshes} 个 Mesh，清理退化面 {removed_deg_faces} 个，孤立点 {removed_loose_verts} 个，补全 UV {added_uvs} 个",
        )
        return {"FINISHED"}


class BEVY_OT_decimate_discrete_leaves(bpy.types.Operator):
    """对选中植物的离散小叶片按空间分层随机抽稀并放大保留叶片，在大幅减面的同时保住树冠外形"""
    bl_idname = "bevy.decimate_discrete_leaves"
    bl_label = "离散叶片随机抽稀 (减面+放大保外形)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.bevy_pipeline_settings
        targets = _collect_target_objects(context, settings.target_scope == "COLLECTION")
        if not targets:
            self.report({"WARNING"}, "请先在视口选中植物物体（如 IvySim_Leaves / Cypress_LF）")
            return {"CANCELLED"}

        keep_ratio = max(0.02, min(0.95, 1.0 - settings.leaf_remove_ratio))
        leaf_scale = settings.leaf_keep_scale
        max_faces = settings.leaf_max_island_faces
        rng = random.Random(settings.leaf_random_seed)

        processed_meshes = set()
        mesh_count = 0
        total_before_polys = 0
        total_after_polys = 0
        total_kept_islands = 0
        total_dropped_islands = 0
        total_large_preserved = 0

        for obj in targets:
            if obj.type != "MESH" or not obj.data:
                continue
            mesh = obj.data
            m_ptr = mesh.as_pointer()
            if m_ptr in processed_meshes:
                continue
            processed_meshes.add(m_ptr)

            bm = bmesh.new()
            try:
                bm.from_mesh(mesh)
                bm.verts.ensure_lookup_table()
                bm.verts.index_update()
                bm.faces.index_update()

                n_verts = len(bm.verts)
                before_polys = len(bm.faces)
                total_before_polys += before_polys
                if n_verts == 0 or before_polys == 0:
                    total_after_polys += before_polys
                    continue

                # 利用物体局部包围盒对角线计算空间体素尺度，用于分层抽稀防止树冠出现空洞塌陷
                bb = obj.bound_box
                xs = [p[0] for p in bb]
                ys = [p[1] for p in bb]
                zs = [p[2] for p in bb]
                dx = max(xs) - min(xs)
                dy = max(ys) - min(ys)
                dz = max(zs) - min(zs)
                diag = (dx * dx + dy * dy + dz * dz) ** 0.5
                cell_size = max(diag / 24.0, 0.05)

                all_verts = list(bm.verts)
                visited = bytearray(n_verts)
                voxel_buckets = defaultdict(list)

                # 1. 遍历连通岛（每片独立小叶片作为一个连通岛；超过阈值的大树干自动跳过）
                for start_idx in range(n_verts):
                    if visited[start_idx]:
                        continue
                    visited[start_idx] = 1
                    queue = deque([start_idx])
                    island_verts = [start_idx]
                    island_faces = set()
                    exceeded = False

                    while queue:
                        cur = queue.popleft()
                        v = all_verts[cur]
                        if not exceeded:
                            for f in v.link_faces:
                                island_faces.add(f.index)
                                if len(island_faces) > max_faces:
                                    exceeded = True
                                    island_faces.clear()
                                    island_verts.clear()
                                    break
                        for e in v.link_edges:
                            nb = e.other_vert(v).index
                            if not visited[nb]:
                                visited[nb] = 1
                                queue.append(nb)
                                if not exceeded:
                                    island_verts.append(nb)

                    if exceeded:
                        total_large_preserved += 1
                        continue
                    if not island_faces:
                        continue

                    inv_n = 1.0 / len(island_verts)
                    cx = sum(all_verts[vi].co.x for vi in island_verts) * inv_n
                    cy = sum(all_verts[vi].co.y for vi in island_verts) * inv_n
                    cz = sum(all_verts[vi].co.z for vi in island_verts) * inv_n
                    cell_key = (
                        int(cx // cell_size),
                        int(cy // cell_size),
                        int(cz // cell_size),
                    )
                    voxel_buckets[cell_key].append((island_verts, cx, cy, cz))

                # 2. 按空间体素内随机打乱 + 累加配额采样，既精准命中目标减面率又保持全局分布均匀
                kept_islands = []
                drop_vert_objs = []
                accum = 0.5

                for cell_key in sorted(voxel_buckets.keys()):
                    bucket = voxel_buckets[cell_key]
                    rng.shuffle(bucket)
                    for item in bucket:
                        accum += keep_ratio
                        if accum >= 1.0:
                            accum -= 1.0
                            kept_islands.append(item)
                        else:
                            for vi in item[0]:
                                drop_vert_objs.append(all_verts[vi])
                            total_dropped_islands += 1

                total_kept_islands += len(kept_islands)

                # 3. 直接在局部坐标围绕叶片各自质心放大保留叶片，保住树冠覆盖面积与轮廓
                if abs(leaf_scale - 1.0) > 1e-4:
                    for vert_indices, cx, cy, cz in kept_islands:
                        for vi in vert_indices:
                            co = all_verts[vi].co
                            co.x = cx + (co.x - cx) * leaf_scale
                            co.y = cy + (co.y - cy) * leaf_scale
                            co.z = cz + (co.z - cz) * leaf_scale

                # 4. 一次性批量删除剔除叶片的顶点（自动级联清理关联边与面）
                if drop_vert_objs:
                    bmesh.ops.delete(bm, geom=drop_vert_objs, context="VERTS")

                total_after_polys += len(bm.faces)
                bm.to_mesh(mesh)
                mesh.update()
                mesh_count += 1
            finally:
                bm.free()

        if mesh_count == 0:
            self.report({"WARNING"}, "选中对象中没有可处理的网格")
            return {"CANCELLED"}

        reduced_pct = (
            (1.0 - total_after_polys / total_before_polys) * 100.0
            if total_before_polys > 0
            else 0.0
        )
        self.report(
            {"INFO"},
            f"叶片抽稀完成 ({mesh_count} 个 Mesh): 面数 {total_before_polys:,} -> {total_after_polys:,} (减少 {reduced_pct:.1f}%) | 保留叶片 {total_kept_islands:,}, 剔除 {total_dropped_islands:,}, 跳过连续枝干 {total_large_preserved}",
        )
        return {"FINISHED"}


def _quantize_float(val: float, step: float) -> float:
    if step <= 1e-6:
        return round(float(val), 4)
    return round(round(float(val) / step) * step, 4)


def _trace_socket_signature(socket, ignore_vertex_color: bool, max_depth: int = 5):
    """沿节点输入反向追踪贴图引用与非贴图程序节点，生成通道结构指纹。"""
    if not socket or not socket.is_linked:
        return ("UNLINKED",)

    images = []
    proc_tokens = []
    visited = set()
    stack = [(socket.links[0].from_node, 0)]

    while stack:
        node, depth = stack.pop()
        if not node or depth > max_depth:
            continue
        ptr = node.as_pointer()
        if ptr in visited:
            continue
        visited.add(ptr)

        ntype = node.type
        if ntype == "TEX_IMAGE":
            img = getattr(node, "image", None)
            if img:
                img_id = bpy.path.abspath(img.filepath) if img.filepath else img.name
                images.append(img_id or img.name)
            else:
                images.append("<EMPTY_IMAGE>")
            continue

        if ignore_vertex_color and ntype in {"VERTEX_COLOR", "ATTRIBUTE"}:
            continue

        if ntype in {"TEX_NOISE", "TEX_VORONOI", "TEX_WAVE", "TEX_MUSGRAVE", "TEX_BRICK", "VALTORGB"}:
            proc_tokens.append(f"{ntype}:{node.name}")

        for inp in node.inputs:
            if inp.is_linked:
                for lk in inp.links:
                    stack.append((lk.from_node, depth + 1))

    return ("LINKED", tuple(sorted(set(images))), tuple(sorted(set(proc_tokens))))


def _material_merge_signature(mat, scalar_tol: float, ignore_vertex_color: bool):
    """生成材质语义指纹：不同贴图/透明模式/双面设置绝不混淆，仅合并同贴图同质材质。"""
    if not mat or not mat.use_nodes or not mat.node_tree:
        return None

    principled = None
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            principled = node
            break
    if not principled:
        return None

    step = max(scalar_tol, 0.0)
    channel_names = (
        "Base Color",
        "Metallic",
        "Roughness",
        "Normal",
        "Alpha",
        "Emission Color",
        "Emission",
    )
    channels_sig = []
    for cname in channel_names:
        sock = principled.inputs.get(cname)
        if not sock:
            continue
        trace = _trace_socket_signature(sock, ignore_vertex_color)
        if trace == ("UNLINKED",):
            dv = sock.default_value
            if hasattr(dv, "__len__"):
                qval = tuple(_quantize_float(x, step) for x in dv)
            else:
                qval = _quantize_float(dv, step)
            channels_sig.append((cname, "VAL", qval))
        else:
            channels_sig.append((cname, trace))

    blend_method = getattr(mat, "blend_method", "OPAQUE")
    alpha_thresh = _quantize_float(getattr(mat, "alpha_threshold", 0.5), 0.05)
    backface_cull = bool(getattr(mat, "use_backface_culling", False))

    return (blend_method, alpha_thresh, backface_cull, tuple(channels_sig))


def _canonical_sort_key(mat):
    """优先保留无 `.001` 数字后缀且名字最短的材质作为合并后的主材质。"""
    name = mat.name
    has_numeric_suffix = len(name) > 4 and name[-4] == "." and name[-3:].isdigit()
    return (1 if has_numeric_suffix else 0, len(name), name)


class BEVY_OT_merge_similar_materials(bpy.types.Operator):
    """合并贴图与视觉参数相同的冗余/同质材质，降低 Meshlet 主 Pass 全屏 Draw 次数"""
    bl_idname = "bevy.merge_similar_materials"
    bl_label = "同质近似材质合并 (压低 Meshlet Draw)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.bevy_pipeline_settings
        tol = settings.mat_merge_scalar_tol
        ignore_vcol = settings.mat_merge_ignore_vertex_color

        # 1. 统计当前场景正在使用的唯一材质
        used_mats = {}
        for obj in context.scene.objects:
            for slot in obj.material_slots:
                if slot.material:
                    used_mats[slot.material.as_pointer()] = slot.material

        before_count = len(used_mats)
        if before_count < 2:
            self.report({"INFO"}, "场景材质数少于 2 个，无需合并")
            return {"FINISHED"}

        # 2. 按语义指纹分组
        groups = defaultdict(list)
        for mat in used_mats.values():
            sig = _material_merge_signature(mat, tol, ignore_vcol)
            if sig is not None:
                groups[sig].append(mat)

        # 3. 为每组选取主材质并建立重映射表
        remap = {}
        merged_groups = 0
        for sig, mats in groups.items():
            if len(mats) < 2:
                continue
            mats.sort(key=_canonical_sort_key)
            canonical = mats[0]
            c_ptr = canonical.as_pointer()
            for dup in mats[1:]:
                if dup.as_pointer() != c_ptr:
                    remap[dup.as_pointer()] = canonical
            merged_groups += 1

        if not remap:
            self.report(
                {"INFO"},
                f"检查了 {before_count} 个场景材质，未发现可合并的同质材质（可适当调大容差）",
            )
            return {"FINISHED"}

        # 4. 替换物体及 Mesh 数据块上的材质槽引用
        replaced_slots = 0
        for obj in context.scene.objects:
            for idx, slot in enumerate(obj.material_slots):
                m = slot.material
                if not m:
                    continue
                target = remap.get(m.as_pointer())
                if target:
                    if slot.link == "DATA" and obj.data and hasattr(obj.data, "materials"):
                        if idx < len(obj.data.materials):
                            obj.data.materials[idx] = target
                    else:
                        slot.material = target
                    replaced_slots += 1

        # 5. 清理已无任何引用的孤立冗余材质
        removed_orphan = 0
        for dup_ptr in list(remap.keys()):
            mat = used_mats.get(dup_ptr)
            if mat and mat.users == 0:
                bpy.data.materials.remove(mat)
                removed_orphan += 1

        # 重新统计合并后剩余场景材质数
        after_mats = set()
        for obj in context.scene.objects:
            for slot in obj.material_slots:
                if slot.material:
                    after_mats.add(slot.material.as_pointer())

        self.report(
            {"INFO"},
            f"同质材质合并完成：场景材质 {before_count} -> {len(after_mats)} 个（{merged_groups} 组，替换 {replaced_slots} 个槽位，删除孤立材质 {removed_orphan} 个）",
        )
        return {"FINISHED"}


class BEVY_OT_fix_zero_lights(bpy.types.Operator):
    """修复 FBX 导入后强度/功率为 0 的灯光（如 SUN 与壁灯点光源）"""
    bl_idname = "bevy.fix_zero_lights"
    bl_label = "一键修复零强度灯光 (SUN/点光源)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        fixed_sun = 0
        fixed_local = 0
        for obj in context.scene.objects:
            if obj.type != "LIGHT" or not obj.data:
                continue
            light = obj.data
            if light.energy > 0.01:
                continue
            name_lower = obj.name.lower()
            if light.type == "SUN":
                light.energy = 5.0
                fixed_sun += 1
            elif "hdri_sky" not in name_lower and light.type in {"POINT", "SPOT"}:
                light.energy = 120.0
                fixed_local += 1

        self.report(
            {"INFO"},
            f"零强度灯光修复完成：SUN {fixed_sun} 个 (Strength=5.0), Point/Spot {fixed_local} 个 (Power=120W)",
        )
        return {"FINISHED"}


def _find_world_env_texture_and_mapping(start_node):
    """沿着 World 节点树上游追踪，定位环境纹理节点 (ShaderNodeTexEnvironment) 与其 Mapping 旋转。"""
    visited = set()
    queue = deque([start_node])
    env_node = None
    while queue:
        node = queue.popleft()
        if node is None or node in visited:
            continue
        visited.add(node)
        if node.type == "TEX_ENVIRONMENT" and getattr(node, "image", None) is not None:
            env_node = node
            break
        for sock in node.inputs:
            if sock.is_linked:
                for link in sock.links:
                    queue.append(link.from_node)

    if env_node is None:
        return None, 0.0

    # 追踪 Vector 输入上的 Mapping / VectorRotate Z 轴旋转
    yaw_rad = 0.0
    vec_sock = env_node.inputs.get("Vector")
    if vec_sock and vec_sock.is_linked:
        vec_visited = set()
        vec_queue = deque([link.from_node for link in vec_sock.links])
        while vec_queue:
            vnode = vec_queue.popleft()
            if vnode is None or vnode in vec_visited:
                continue
            vec_visited.add(vnode)
            if vnode.type == "MAPPING":
                rot_sock = vnode.inputs.get("Rotation")
                if rot_sock and hasattr(rot_sock, "default_value"):
                    yaw_rad += -float(rot_sock.default_value[2])
            elif vnode.type == "VECTOR_ROTATE":
                angle_sock = vnode.inputs.get("Angle")
                if angle_sock and hasattr(angle_sock, "default_value"):
                    yaw_rad += -float(angle_sock.default_value)
            for sock in vnode.inputs:
                if sock.is_linked:
                    for link in sock.links:
                        vec_queue.append(link.from_node)

    return env_node, math.degrees(yaw_rad)


def _extract_world_hdri_config(scene, glb_filepath):
    """若场景 World 使用了 HDRI 环境贴图，则提取贴图路径、强度与旋转配置；否则返回 None。"""
    world = getattr(scene, "world", None)
    if world is None or not getattr(world, "use_nodes", False) or world.node_tree is None:
        return None

    nodes = world.node_tree.nodes
    out_node = next(
        (n for n in nodes if n.type == "OUTPUT_WORLD" and getattr(n, "is_active_output", False)),
        None,
    ) or next((n for n in nodes if n.type == "OUTPUT_WORLD"), None)
    if out_node is None:
        return None

    surf_sock = out_node.inputs.get("Surface")
    if surf_sock is None or not surf_sock.is_linked:
        return None

    bg_node = surf_sock.links[0].from_node
    if bg_node is None:
        return None

    # 兼容直接连接 Background 或经由 Mix Shader 连接到 World Output
    strength = 1.0
    bg_queue = deque([bg_node])
    bg_visited = set()
    chosen_bg = None
    while bg_queue:
        n = bg_queue.popleft()
        if n is None or n in bg_visited:
            continue
        bg_visited.add(n)
        if n.type in {"BACKGROUND", "EMISSION"}:
            chosen_bg = n
            break
        for sock in n.inputs:
            if sock.is_linked:
                for link in sock.links:
                    bg_queue.append(link.from_node)

    search_root = chosen_bg if chosen_bg is not None else bg_node
    if chosen_bg is not None:
        str_sock = chosen_bg.inputs.get("Strength")
        if str_sock is not None and hasattr(str_sock, "default_value"):
            strength = float(str_sock.default_value)

    env_node, yaw_degrees = _find_world_env_texture_and_mapping(search_root)
    if env_node is None or env_node.image is None:
        return None

    img = env_node.image
    abs_path = ""
    if getattr(img, "filepath", ""):
        abs_path = os.path.normpath(bpy.path.abspath(img.filepath, library=img.library))

    resolved_path = abs_path if abs_path and os.path.isfile(abs_path) else ""
    if not resolved_path and getattr(img, "packed_file", None) is not None:
        stem = os.path.splitext(os.path.basename(glb_filepath))[0]
        ext = os.path.splitext(abs_path or img.name)[1] or ".hdr"
        unpacked_path = os.path.join(os.path.dirname(glb_filepath), f"{stem}_world_hdri{ext}")
        with open(unpacked_path, "wb") as f:
            f.write(img.packed_file.data)
        resolved_path = os.path.normpath(unpacked_path)

    if not resolved_path:
        return None

    projection = "equirectangular" if getattr(env_node, "projection", "EQUIRECTANGULAR") == "EQUIRECTANGULAR" else "mirror_ball"
    return {
        "image": resolved_path,
        "strength": max(0.0, strength),
        "yaw_degrees": yaw_degrees,
        "projection": projection,
    }


class BEVY_OT_export_glb(bpy.types.Operator):
    """一键导出含 Custom Properties (extras) 和灯光的 GLB 文件"""
    bl_idname = "bevy.export_glb"
    bl_label = "导出 GLB (含角标与灯光)"

    def execute(self, context):
        settings = context.scene.bevy_pipeline_settings
        filepath = bpy.path.abspath(settings.export_path)
        if not filepath:
            self.report({"ERROR"}, "请先指定有效的导出 GLB 路径")
            return {"CANCELLED"}

        out_dir = os.path.dirname(filepath)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # 导出前快速检查零强度灯并提醒
        zero_lights = sum(
            1
            for obj in context.scene.objects
            if _is_object_exportable(obj, settings.export_visible_only)
            and obj.type == "LIGHT"
            and obj.data
            and obj.data.energy <= 0.01
            and "hdri_sky" not in obj.name.lower()
        )

        hdri_cfg = _extract_world_hdri_config(context.scene, filepath)
        had_prev_prop = "bevy_world_hdri" in context.scene
        prev_prop_val = context.scene.get("bevy_world_hdri")
        if hdri_cfg is not None:
            context.scene["bevy_world_hdri"] = json.dumps(hdri_cfg, ensure_ascii=False)
        elif had_prev_prop:
            del context.scene["bevy_world_hdri"]

        export_kwargs = {
            "filepath": filepath,
            "export_format": "GLB",
            "export_extras": True,      # 关键：将 bevy_* 自定义属性写入 glTF extras
            "export_lights": True,      # 导出 KHR_lights_punctual 灯光
            "export_yup": True,
            "export_apply": settings.export_apply_modifiers,
        }
        # 动态兼容不同 Blender 版本的可见/可渲染过滤开关
        try:
            rna_props = bpy.ops.export_scene.gltf.get_rna_type().properties
            if "use_visible" in rna_props:
                export_kwargs["use_visible"] = settings.export_visible_only
            if "use_renderable" in rna_props:
                export_kwargs["use_renderable"] = settings.export_visible_only
        except Exception:
            pass

        try:
            bpy.ops.export_scene.gltf(**export_kwargs)
        finally:
            if had_prev_prop:
                context.scene["bevy_world_hdri"] = prev_prop_val
            elif "bevy_world_hdri" in context.scene:
                del context.scene["bevy_world_hdri"]

        sidecar_path = os.path.splitext(filepath)[0] + ".bevy_hdri.json"
        if hdri_cfg is not None:
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(hdri_cfg, f, indent=2, ensure_ascii=False)
        elif os.path.isfile(sidecar_path):
            try:
                os.remove(sidecar_path)
            except OSError:
                pass

        hdri_note = f"（含 World HDRI: {os.path.basename(hdri_cfg['image'])}）" if hdri_cfg else ""
        if zero_lights > 0:
            self.report(
                {"WARNING"},
                f"GLB 已导出: {filepath}{hdri_note}（提示：仍有 {zero_lights} 盏灯强度为 0，建议先点“修复零强度灯光”）",
            )
        else:
            self.report({"INFO"}, f"GLB 导出成功: {filepath}{hdri_note}")
        return {"FINISHED"}


# ==============================================================================
# 4. 3D 视口侧边栏 UI 面板
# ==============================================================================
class BEVY_PT_pipeline_panel(bpy.types.Panel):
    bl_label = "Bevy Pipeline Tagger"
    bl_idname = "BEVY_PT_pipeline_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Bevy Pipeline"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.bevy_pipeline_settings

        # 显示当前活动物体的角标状态
        active = context.active_object
        status_box = layout.box()
        if active:
            cur_tag = active.get("bevy_tag", "未打标 (默认 MESHLET_STATIC)")
            status_box.label(text=f"当前物体: {active.name}", icon="OBJECT_DATA")
            status_box.label(text=f"当前角标: {cur_tag}", icon="BOOKMARKS")
            if "bevy_chunk" in active:
                status_box.label(text=f"Chunk: {active['bevy_chunk']}")
        else:
            status_box.label(text="未选中活动物体", icon="INFO")

        # 作用范围与参数区
        param_box = layout.box()
        param_box.label(text="打标参数设置", icon="PREFERENCES")
        param_box.prop(settings, "target_scope", expand=True)
        param_box.prop(settings, "chunk_name")
        param_box.prop(settings, "lod_ratios")
        param_box.prop(settings, "lod_dists")
        param_box.prop(settings, "spawn_group")

        # 动态渲染注册表中的所有角标按钮
        tag_box = layout.box()
        tag_box.label(text="一键批量打标（按颜色区分）", icon="COLOR")
        for tag_id, label, _desc, _color, _defaults in BEVY_TAG_REGISTRY:
            row = tag_box.row(align=True)
            op = row.operator("bevy.apply_tag", text=label)
            op.tag_id = tag_id
            sel_op = row.operator("bevy.select_by_tag", text="", icon="RESTRICT_SELECT_OFF")
            sel_op.tag_id = tag_id

        tag_box.operator("bevy.clear_tag", icon="X")

        # 场景体检与非游戏资产净化区
        audit_box = layout.box()
        audit_box.label(text="场景体检与拓扑净化", icon="VIEWZOOM")
        audit_box.operator("bevy.audit_scene", icon="INFO")
        if settings.audit_summary:
            for line in settings.audit_summary.split("\n"):
                audit_box.label(text=line)
            if settings.audit_heavy_top:
                audit_box.label(text=settings.audit_heavy_top)
            sel_row = audit_box.row(align=True)
            op = sel_row.operator("bevy.select_audit_issue", text="高细分")
            op.issue_kind = "HEAVY_SUBSURF"
            op = sel_row.operator("bevy.select_audit_issue", text="风险材质")
            op.issue_kind = "BAD_MATERIAL"
            op = sel_row.operator("bevy.select_audit_issue", text="退化面")
            op.issue_kind = "DEGENERATE_MESH"
            sel_row2 = audit_box.row(align=True)
            op = sel_row2.operator("bevy.select_audit_issue", text="缺UV")
            op.issue_kind = "MISSING_UV"
            op = sel_row2.operator("bevy.select_audit_issue", text="异常缩放")
            op.issue_kind = "BAD_SCALE"
            op = sel_row2.operator("bevy.select_audit_issue", text="问题灯")
            op.issue_kind = "ZERO_OR_AREA_LIGHT"

        audit_box.prop(settings, "max_subsurf_level")
        audit_box.operator("bevy.clamp_subsurf", icon="MOD_SUBSURF")
        audit_box.prop(settings, "merge_dist")
        audit_box.prop(settings, "auto_create_uv")
        audit_box.operator("bevy.clean_topology", icon="MESH_DATA")

        # 植物离散叶片抽稀区（专治千万面独立叶片无法被 meshopt 简化的场景）
        leaf_box = layout.box()
        leaf_box.label(text="植物离散叶片抽稀 (保外形减面)", icon="OUTLINER_OB_POINTCLOUD")
        leaf_box.prop(settings, "leaf_remove_ratio")
        leaf_box.prop(settings, "leaf_keep_scale")
        leaf_box.prop(settings, "leaf_max_island_faces")
        leaf_box.prop(settings, "leaf_random_seed")
        leaf_box.operator("bevy.decimate_discrete_leaves", icon="MOD_DECIM")

        # FBX 清洗与 GLB 导出区
        util_box = layout.box()
        util_box.label(text="材质灯光修复与导出", icon="EXPORT")
        util_box.operator("bevy.fix_fbx_materials", icon="MATERIAL")
        util_box.prop(settings, "mat_merge_scalar_tol")
        util_box.prop(settings, "mat_merge_ignore_vertex_color")
        util_box.operator("bevy.merge_similar_materials", icon="NODE_MATERIAL")
        util_box.operator("bevy.fix_zero_lights", icon="LIGHT")
        util_box.prop(settings, "export_path")
        util_box.prop(settings, "export_visible_only")
        util_box.prop(settings, "export_apply_modifiers")
        util_box.operator("bevy.export_glb", icon="FILE_3D")


CLASSES = (
    BevyPipelineSettings,
    BEVY_OT_apply_tag,
    BEVY_OT_clear_tag,
    BEVY_OT_select_by_tag,
    BEVY_OT_audit_scene,
    BEVY_OT_select_audit_issue,
    BEVY_OT_clamp_subsurf,
    BEVY_OT_clean_topology,
    BEVY_OT_decimate_discrete_leaves,
    BEVY_OT_merge_similar_materials,
    BEVY_OT_fix_fbx_materials,
    BEVY_OT_fix_zero_lights,
    BEVY_OT_export_glb,
    BEVY_PT_pipeline_panel,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.bevy_pipeline_settings = bpy.props.PointerProperty(type=BevyPipelineSettings)


def unregister():
    del bpy.types.Scene.bevy_pipeline_settings
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
