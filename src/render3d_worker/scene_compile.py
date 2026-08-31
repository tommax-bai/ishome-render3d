"""编场景包：设计包（几何 + 尺子 + 高度 + 家具 + 材质 + 机位）→ 场景包。

`scene-compile` 这个 activity 的全部业务就在这一个函数里。它做三件事，一件都不多：

1. **攒网格**——壳体与家具都交给 `mesh`，这一层不碰坐标换算，也不认识"墙该多厚"。
2. **指派材质**——按 :class:`~render3d_worker.models.MaterialAssignment` 从细到粗匹配，
   一条都匹配不上就落到中性材质；**中性材质会如实进场景包的材质表**，不静默。
3. **算自证数**——地板面积、面积吻合率、墙段数、洞数、三角形数。

**不设死阈值。** 面积对不上到什么程度算失败，要有真跑数据才定（《纪律·阈值有数据才定》）——
今天只把 `area_match_ratio` 算出来带出去。真该炸的是"根本编不出来"那几种：几何为空、
面积为 0、一间房都没有；那些直接抛 :class:`SceneCompileError`，不返回一个空场景包
（空包往下游走，底渲会渲出一张全空的图，而没有一步能说出它为什么空）。
"""

from __future__ import annotations

import logging
import math

from render3d_worker import mesh
from render3d_worker.models import (
    DesignPackage,
    FloorplanGeometry,
    MaterialAssignment,
    MaterialSurface,
    Mesh,
    MeshSemantic,
    ScenePackage,
    SurfaceMaterial,
)

NEUTRAL_MATERIAL = SurfaceMaterial(id="material:neutral", base_color_hex="#B0B0B0")
"""没人指派材质的面用它。**这不是"这户人家的材质"，是"这块面没被指派"的可见记号**——
底渲的几何那一路上它是一片没分色的灰，看得出来；它也会出现在场景包的材质表里，
"哪些面没指派"因此答得出来，而不是悄悄借用了隔壁的颜色。"""

_SURFACE_OF_SEMANTIC: dict[MeshSemantic, MaterialSurface] = {
    "floor": "floor",
    "ceiling": "ceiling",
    "wall": "wall",
    "reveal": "wall",
    "furnishing": "furnishing",
}
"""网格语义 → 材质面。`reveal`（洞壁）归到 `wall`：洞壁是墙被切开露出的内表面，
它就是墙的材质；单给它一类，等于要求上游为每个门洞多写一条指派。"""

_REPORT_DECIMALS = 4
"""自证数留四位。它们是给人看、给日志比对的数，不是几何。"""

_log = logging.getLogger(__name__)


class SceneCompileError(Exception):
    """场景包编不出来。响亮失败，不返回一个空包。"""


def _material_specificity(assignment: MaterialAssignment) -> int:
    """一条指派有多"具体"：指名了房间算一分，指名了品类算一分。

    匹配时取分最高的那条，同分取输入里靠前的——**先细后粗、同细看次序**，
    比"房间优先还是品类优先"少一个要记的规矩，也不必为家具和面各写一套。
    """
    return int(assignment.room is not None) + int(assignment.category is not None)


def _pick_material_id(
    surface: MaterialSurface,
    room: str | None,
    category: str | None,
    assignments: list[MaterialAssignment],
) -> str:
    best_id: str | None = None
    best_score = -1
    for assignment in assignments:
        if assignment.surface != surface:
            continue
        if assignment.room is not None and assignment.room != room:
            continue
        if assignment.category is not None and assignment.category != category:
            continue
        score = _material_specificity(assignment)
        if score > best_score:
            best_id, best_score = assignment.material_id, score
    return best_id if best_id is not None else NEUTRAL_MATERIAL.id


def _check_material_table(package: DesignPackage) -> None:
    """材质表自身对不上头就当场炸：重复的 id、指向不存在材质的指派。

    这两样都是包填错了，不是几何的事；早一步炸在边界上，好过带着一个指不到东西的
    材质 id 一路走到底渲（口径同 models.py `_Contract` 的 `extra=forbid`）。
    """
    seen: set[str] = set()
    for material in package.surface_materials:
        if material.id in seen:
            raise SceneCompileError(f"材质表里 {material.id} 出现了不止一次")
        seen.add(material.id)
    for assignment in package.materials:
        if assignment.material_id not in seen:
            raise SceneCompileError(
                f"材质指派指向 {assignment.material_id}，材质表里没有这一条——不替它挑一个像的顶上"
            )


def _assign_materials(meshes: list[Mesh], package: DesignPackage) -> tuple[list[Mesh], bool]:
    """给每块网格挂上材质 id。返回改过的网格清单，以及中性材质有没有真被用到。"""
    category_of = {placement.id: placement.category for placement in package.furnishings}
    assigned: list[Mesh] = []
    used_neutral = False
    for block in meshes:
        surface = _SURFACE_OF_SEMANTIC[block.semantic]
        material_id = _pick_material_id(
            surface, block.room, category_of.get(block.id), package.materials
        )
        used_neutral = used_neutral or material_id == NEUTRAL_MATERIAL.id
        assigned.append(block.model_copy(update={"material_id": material_id}))
    return assigned, used_neutral


def _check_unique_ids(meshes: list[Mesh]) -> None:
    """网格 id 撞了就炸：底渲的遮罩索引表按 id 回指网格，撞了就指错。

    最可能撞的是家具——它的 id 直接用输入里 placement 的 id。
    """
    seen: set[str] = set()
    for block in meshes:
        if block.id in seen:
            raise SceneCompileError(f"网格 id {block.id} 出现了不止一次：遮罩索引表会指错")
        seen.add(block.id)


def _triangle_area_sqm(
    a: tuple[float, float, float], b: tuple[float, float, float], c: tuple[float, float, float]
) -> float:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    return math.sqrt(nx * nx + ny * ny + nz * nz) / 2


def _floor_area_sqm(meshes: list[Mesh]) -> float:
    """地板网格的面积和。按三角形实算而不是按矩形回推——回推等于信任建网格那一步，
    而这个数存在的意义正是去查它。"""
    total = 0.0
    for block in meshes:
        if block.semantic != "floor":
            continue
        for i, j, k in block.triangles:
            total += _triangle_area_sqm(block.vertices[i], block.vertices[j], block.vertices[k])
    return total


def _bounds_m(meshes: list[Mesh]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    points = [point for block in meshes for point in block.vertices]
    if not points:
        raise SceneCompileError("一个顶点都没有：编出来的是个空场景，不往下游传")
    low = (min(p[0] for p in points), min(p[1] for p in points), min(p[2] for p in points))
    high = (max(p[0] for p in points), max(p[1] for p in points), max(p[2] for p in points))
    return low, high


def _warn_if_scale_falls_back(plan: FloorplanGeometry) -> None:
    """尺子锚不到外轮廓、退回外接框时打一条警告。

    **退路不是错，静默才是错**：外接框是外接的，户型不是满铺矩形时它比套内建筑面积大，
    编出来的一切都会小一号，而场景包里没有一个数会说这件事（`area_match_ratio` 照样是
    房间遮罩 ÷ 套内面积，两边一起缩，比值不变）。选日志而不是加字段的理由见
    :func:`~render3d_worker.mesh.scale_anchor`。
    """
    anchor = mesh.scale_anchor(plan)
    if anchor.source == "outline":
        return
    _log.warning(
        "尺子退回了 plan_box 外接框（外轮廓 %d 段，围不出面积）："
        "外接框比套内建筑面积大，编出来的房子会偏小",
        len(plan.outline),
    )


def compile_scene_package(package: DesignPackage) -> ScenePackage:
    """设计包 → 场景包。**确定性：同一份包编两次，`model_dump_json()` 逐字相同。**

    自证数随包带出，不在这儿判它够不够好——门槛要有真跑数据才定。
    """
    plan = package.plan
    if not plan.rooms:
        raise SceneCompileError(
            "户型里一间房都没有：地板与吊顶都从房间遮罩长出来，没有房间就没有场景"
        )
    _check_material_table(package)
    _warn_if_scale_falls_back(plan)

    try:
        unit_m = mesh.metre_per_unit(plan, package.scale)
        blocks = [
            *mesh.build_shell(plan, package.scale, package.heights),
            *mesh.build_furnishings(package.furnishings, plan, package.scale, package.heights),
        ]
        openings_by_kind = mesh.count_openings_by_kind(plan, package.scale, package.heights)
        target_area_sqm = mesh.usable_area_sqm(package.scale)
    except mesh.MeshBuildError as error:
        raise SceneCompileError(f"网格建不出来：{error}") from error

    if not blocks:
        raise SceneCompileError("一块网格都没有：几何里既没有房间遮罩也没有墙")
    _check_unique_ids(blocks)

    floor_area_sqm = _floor_area_sqm(blocks)
    if floor_area_sqm <= 0:
        raise SceneCompileError("编出来的地板面积是 0：房间遮罩要么是空的，要么被尺子压成了一个点")

    meshes, used_neutral = _assign_materials(blocks, package)
    low_m, high_m = _bounds_m(meshes)
    materials = list(package.surface_materials)
    if used_neutral:
        materials.append(NEUTRAL_MATERIAL)

    return ScenePackage(
        revision_id=package.revision_id,
        scale_anchor_source=mesh.scale_anchor(plan).source,
        meshes=meshes,
        materials=materials,
        # 相机是输入不是产物：换个机位不该重编场景包，原样带过去
        cameras=list(package.cameras),
        bounds_min_m=low_m,
        bounds_max_m=high_m,
        metre_per_unit=round(unit_m, 6),
        floor_area_sqm=round(floor_area_sqm, _REPORT_DECIMALS),
        area_match_ratio=round(floor_area_sqm / target_area_sqm, _REPORT_DECIMALS),
        wall_segment_count=sum(1 for block in meshes if block.semantic == "wall"),
        degenerate_wall_count=len(mesh.degenerate_wall_ids(plan, package.scale)),
        opening_count_by_kind=openings_by_kind,
        triangle_count=sum(len(block.triangles) for block in meshes),
    )
