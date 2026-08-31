"""2D 户型几何 + 一把尺子 + 高度规则 → 三角网格。**确定性，零模型调用。**

这一层回答的问题只有一个：**归一化的平面，怎么变成米制的体块**。它不认识材质、
不认识相机、不知道场景包长什么样——那些是编场景包（`scene_compile`）的事。

三条口径写死在这儿，别处不再复述：

1. **换算只经 :class:`_Ruler` 一处。** x 按图宽归一、y 按图高归一，两个方向除的不是
   同一个数；不按 `frame_*_px` 修正长宽比，一张长方形户型会被拉成正方形——2D 那侧
   （render2d `plan_master._Frame`）踩过这个坑，这儿照它的口径来。
2. **不做布尔求并。** `outline` 与 `walls` 重合的段会生成两块重叠的墙体，允许；理由见
   models.py 里 :class:`~render3d_worker.models.FloorplanGeometry` 的 docstring
   （底渲只看最近面，重叠不影响四路输出，省一整套数值稳健性问题）。
3. **开洞是纯算术的切段，不引布尔运算库。** 一段带洞的墙沿洞的起讫切成
   "洞左 / 洞右 / 洞上过梁 / 洞下窗下墙"几块长方体，洞壁另出一块 `reveal` 套框网格。
   确定性、可数、可测——墙块数与洞高都是能在测试里断言的数。
4. **洞有两种形态，两条路都得走通。**上游的墙可能压着洞（洞要从墙里切出来），也可能
   在洞处本来就断开（洞落在两段墙之间的空隙里，什么也不用切、但缺过梁和窗下墙）。
   判据与做法见 :func:`_layout_openings`。

坐标系：米制右手系，x 向右、y 向里、z 向上，地面 z=0（models.py :class:`Mesh` 写死）。
平面图的"下"（y_ratio 增大）就是三维的 +y。

**缺的数不在这儿编。** 尺寸一律来自输入包，缺了、是 0、自相矛盾就抛
:class:`MeshBuildError`——编一个默认值填上，等于让一张错图一路走到出图才被发现。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from render3d_worker.models import (
    FloorplanGeometry,
    FurnishingPlacement,
    HeightRules,
    Mesh,
    MeshSemantic,
    OpeningKind,
    PlanAxis,
    PlanOpening,
    PlanScale,
    PlanWall,
)

UNASSIGNED_MATERIAL_ID = "material:unassigned"
"""建网格时挂的占位材质。**指派材质是编场景包那一层的事**（它才知道房间与家具品类），
这一层不做，也不留空串——留空串会让"没指派"和"指派成了空"长得一样。"""

_SAME_LINE_TOLERANCE_RATIO = 0.006
"""判"这个洞在哪道墙上"时位置的容差。取值与口径同 2D 母版（render2d
`plan_master._SAME_LINE_TOLERANCE`）：外轮廓给的是墙带中心线，与网格投票出来的线
差半个墙厚是常态，不给容差就没有一个洞落得到外墙上。"""

_MIN_SEGMENT_M = 1e-4
"""比 0.1 毫米还短的段不是墙，是切段留下的数值残渣——不出网格。

**这不是"多小算小"的阈值**（那种要真跑数据才敢定），是浮点残差的下界：洞正好开在
墙头时，"洞左"那一段的长度理论上就是 0。"""

_COORD_DECIMALS = 6
"""顶点坐标留到微米。再往下的位数是浮点残差不是几何，留着只会让两次编译的
JSON 差在末位——确定性是这一层的红线。"""

_KIND_ORDER: tuple[OpeningKind, ...] = ("door", "window", "pass")
"""洞的种类在报数时的固定次序。次序写死是为了同一份输入数出来的字典逐字相同。"""


class MeshBuildError(Exception):
    """网格建不出来。响亮失败，不给一堆"差不多的"体块。

    编场景包那一层会把它裹成 `SceneCompileError` 再抛（两层各自的失败各有各的名字，
    调用方只需要认识最外面那一个）。
    """


# ---------------------------------------------------------------------------
# 一、尺子：归一化 → 米
# ---------------------------------------------------------------------------


def usable_area_sqm(scale: PlanScale) -> float:
    """套内面积 = 建筑面积 × 得房率。**这条换算只写在这儿一处。**

    得房率取百分数不取小数（80 不是 0.8），口径同 models.py
    :class:`~render3d_worker.models.PlanScale`。
    """
    if scale.building_area_sqm <= 0:
        raise MeshBuildError(
            f"建筑面积是 {scale.building_area_sqm}㎡：尺子由面积反推，没有面积就没有米"
        )
    if scale.usable_area_percent <= 0:
        raise MeshBuildError(f"得房率是 {scale.usable_area_percent}%：套内面积会算成 0，量不出尺子")
    return scale.building_area_sqm * scale.usable_area_percent / 100.0


ScaleAnchorSource = Literal["outline", "plan-box"]


@dataclass(frozen=True)
class ScaleAnchor:
    """尺子拿哪块面积当锚、那块面积有多大（归一化到 x 单位的平方）。"""

    source: ScaleAnchorSource
    area_x_units_sq: float


def _y_units_per_x_unit(plan: FloorplanGeometry) -> float:
    """y 方向的归一化 1.0 折成多少个 x 方向单位。**长宽比的修正只写在这一处。**"""
    if plan.frame_width_px <= 0 or plan.frame_height_px <= 0:
        raise MeshBuildError(
            f"参照系是空的：frame {plan.frame_width_px}×{plan.frame_height_px}px，"
            f"没有它长宽比就是错的"
        )
    return plan.frame_height_px / plan.frame_width_px


def _crossings_left_of(x_ratio: float, y_ratio: float, verticals: list[PlanWall]) -> int:
    return sum(
        1
        for wall in verticals
        if wall.position_ratio < x_ratio
        and min(wall.start_ratio, wall.end_ratio) < y_ratio < max(wall.start_ratio, wall.end_ratio)
    )


def _crossings_above(x_ratio: float, y_ratio: float, horizontals: list[PlanWall]) -> int:
    return sum(
        1
        for wall in horizontals
        if wall.position_ratio < y_ratio
        and min(wall.start_ratio, wall.end_ratio) < x_ratio < max(wall.start_ratio, wall.end_ratio)
    )


def _rectilinear_area_ratio_sq(walls: list[PlanWall]) -> float:
    """一组轴对齐中心线段围合的面积（归一化，x 与 y 各按自己的分母）。围不成一圈就返回 0。

    做法：拿所有段的坐标把平面切成网格，逐格判"在不在屋里"，在的把格子面积加起来。
    **判据是横竖两个方向的奇偶都说"在里面"才算在里面**——横着往左数穿过几段竖墙，
    竖着往上数穿过几段横墙，两个都得是奇数。

    为什么要两个方向都判：**段会在墙角互相探出头**。2026-08-30 那份真跑存档的四条外轮廓，
    两段竖墙沿 y 铺满了整个 `plan_box`，而两段横墙的中心线缩在里面约 6 个源像素（半个墙厚）——
    只按一个方向判奇偶，墙角外那一圈会被算进屋里，量出来的就成了"竖墙的宽 × 外接框的高"。
    两个方向都判，探出头的那一截自动被切掉，剩下的正好是中心线围的那块。
    """
    verticals = [wall for wall in walls if wall.axis == "vertical"]
    horizontals = [wall for wall in walls if wall.axis == "horizontal"]
    if not verticals or not horizontals:
        return 0.0
    x_edges = sorted(
        {wall.position_ratio for wall in verticals}
        | {ratio for wall in horizontals for ratio in (wall.start_ratio, wall.end_ratio)}
    )
    y_edges = sorted(
        {wall.position_ratio for wall in horizontals}
        | {ratio for wall in verticals for ratio in (wall.start_ratio, wall.end_ratio)}
    )
    total = 0.0
    for x0_ratio, x1_ratio in zip(x_edges, x_edges[1:], strict=False):
        center_x_ratio = (x0_ratio + x1_ratio) / 2
        for y0_ratio, y1_ratio in zip(y_edges, y_edges[1:], strict=False):
            center_y_ratio = (y0_ratio + y1_ratio) / 2
            if not _crossings_left_of(center_x_ratio, center_y_ratio, verticals) % 2:
                continue
            if not _crossings_above(center_x_ratio, center_y_ratio, horizontals) % 2:
                continue
            total += (x1_ratio - x0_ratio) * (y1_ratio - y0_ratio)
    return total


def scale_anchor(plan: FloorplanGeometry) -> ScaleAnchor:
    """尺子拿哪块面积当锚：**外轮廓中心线围的那块**，围不出来才退回 `plan_box` 外接框。

    为什么是外轮廓而不是外接框：**"外墙中心线围合的面积"就是套内建筑面积的几何定义**
    （套内建筑面积 = 套内使用面积 + 内墙占地 + 外墙一半），两边说的是同一个量，所以这条
    不引入任何系数。外接框是**外接**的——户型只要不是满铺矩形（凹角、阳台外那块空白），
    框住的面积就比套内大，尺子跟着偏小，出图里家具会显得挤。首个真样例上外接框比外轮廓
    大约 4%，尺子因此偏小约 2%。

    退回外接框是给老产物留的路：`outline` 是 2026-08-31 才在产出侧补上的，之前的几何
    没有它。**退了要看得见**——本函数把 `source` 说出来，`scene_compile` 在退路上打一条
    警告日志。选日志而不是场景包字段，是因为加字段要动 `models.py` 那份两侧共用的契约
    （另一路正在写底渲），契约得两边一起改才算数；升成字段的时点写死＝`ScenePackage`
    下一次动契约那一批。
    """
    outline_area_x_units_sq = _rectilinear_area_ratio_sq(plan.outline) * _y_units_per_x_unit(plan)
    if outline_area_x_units_sq > 0:
        return ScaleAnchor("outline", outline_area_x_units_sq)
    left, top, right, bottom = plan.plan_box
    box_area_x_units_sq = (right - left) * (bottom - top) * _y_units_per_x_unit(plan)
    return ScaleAnchor("plan-box", box_area_x_units_sq)


class _Ruler:
    """归一化平面坐标 → 米。**全仓唯一的换算处。**

    `metre_per_unit` 的口径：**x 方向归一化 1.0（整张图的宽）等于多少米**。y 方向的
    1.0 不等于同一个数——它等于 `metre_per_unit × frame_height_px / frame_width_px`。
    两轴分母不同这件事只在这个类里存在，出了这个类全是米。

    原点仍是 `plan_box` 的左上角——**锚点换了不等于原点换了**：锚点决定一格有多大，
    原点只决定从哪儿起算，换原点会白白把所有坐标平移一遍。
    """

    def __init__(self, plan: FloorplanGeometry, scale: PlanScale) -> None:
        left, top, right, bottom = plan.plan_box
        if right - left <= 0 or bottom - top <= 0:
            raise MeshBuildError(f"图幅是空的：plan_box={plan.plan_box}")
        self._left_ratio = left
        self._top_ratio = top
        self._y_units_per_x_unit = _y_units_per_x_unit(plan)
        self.usable_area_sqm = usable_area_sqm(scale)
        self.anchor = scale_anchor(plan)
        if self.anchor.area_x_units_sq <= 0:
            raise MeshBuildError(f"锚不到面积：外轮廓围不出一块地，外接框 {plan.plan_box} 也是空的")
        self.metre_per_unit = math.sqrt(self.usable_area_sqm / self.anchor.area_x_units_sq)

    def x_m(self, x_ratio: float) -> float:
        """图上的 x 归一化坐标 → 米（图幅左边缘为 0）。"""
        return (x_ratio - self._left_ratio) * self.metre_per_unit

    def y_m(self, y_ratio: float) -> float:
        """图上的 y 归一化坐标 → 米（图幅上边缘为 0，往下为 +y）。"""
        return (y_ratio - self._top_ratio) * self._y_units_per_x_unit * self.metre_per_unit

    def across_x_m(self, ratio: float) -> float:
        """按图宽归一的长度（竖墙的厚度）→ 米。"""
        return ratio * self.metre_per_unit

    def across_y_m(self, ratio: float) -> float:
        """按图高归一的长度（横墙的厚度）→ 米。"""
        return ratio * self._y_units_per_x_unit * self.metre_per_unit

    def along_m(self, axis: PlanAxis, ratio: float) -> float:
        """沿墙走向的归一化坐标 → 米（竖墙沿 y、横墙沿 x）。"""
        return self.y_m(ratio) if axis == "vertical" else self.x_m(ratio)


def metre_per_unit(plan: FloorplanGeometry, scale: PlanScale) -> float:
    """归一化 1.0（整张图的宽）等于多少米。

    由面积反推：套内建筑面积 ÷ 锚那块地的归一化面积（按 `frame_*_px` 修正长宽比后）再开方。
    锚是外轮廓中心线围的那块，理由与退路见 :func:`scale_anchor`。比例尺不是模型能给的数，
    面积才是上游真有的数（匿名画像带建筑面积与得房率）。
    """
    return _Ruler(plan, scale).metre_per_unit


# ---------------------------------------------------------------------------
# 二、面片攒网格：一块网格 = 若干四边形
# ---------------------------------------------------------------------------


class _Faces:
    """攒面片。四边形进来，两个三角形出去，**朝向由调用方指定的法向定**。

    朝向不靠"记住顶点该按什么顺序写"——那种约定在第七个面上一定会写反一次；
    调用方说"这个面该朝哪儿"，这儿算叉积、不对就把顶点倒过来。
    """

    def __init__(self) -> None:
        self.vertices: list[tuple[float, float, float]] = []
        self.triangles: list[tuple[int, int, int]] = []

    def add_quad(
        self,
        corners: list[tuple[float, float, float]],
        toward: tuple[float, float, float],
    ) -> None:
        ordered = corners if _faces_toward(corners, toward) else list(reversed(corners))
        base = len(self.vertices)
        self.vertices.extend(_rounded(point) for point in ordered)
        self.triangles.append((base, base + 1, base + 2))
        self.triangles.append((base, base + 2, base + 3))


def _rounded(point: tuple[float, float, float]) -> tuple[float, float, float]:
    # 加 0.0 是为了把 -0.0 归成 0.0：两者数值相等但 JSON 里长得不一样
    x, y, z = point
    return (
        round(x, _COORD_DECIMALS) + 0.0,
        round(y, _COORD_DECIMALS) + 0.0,
        round(z, _COORD_DECIMALS) + 0.0,
    )


def _faces_toward(
    corners: list[tuple[float, float, float]], toward: tuple[float, float, float]
) -> bool:
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = corners[0], corners[1], corners[2]
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    return nx * toward[0] + ny * toward[1] + nz * toward[2] > 0


def _signed_area_sqm(base_xy: list[tuple[float, float]]) -> float:
    total = 0.0
    for index, (x0, y0) in enumerate(base_xy):
        x1, y1 = base_xy[(index + 1) % len(base_xy)]
        total += x0 * y1 - x1 * y0
    return total / 2


def _prism_faces(base_xy: list[tuple[float, float]], z_bottom_m: float, z_top_m: float) -> _Faces:
    """底面四边形沿 z 拉伸成长方体（底面可以是旋转过的矩形）。

    先把底面摆正成从 +z 看去的逆时针序：侧面的外法向是从边的走向推出来的
    （边 (dx,dy) 的外法向 = (dy,-dx)），底面倒了侧面就全朝里。
    """
    corners = base_xy if _signed_area_sqm(base_xy) >= 0 else list(reversed(base_xy))
    bottom = [(x, y, z_bottom_m) for x, y in corners]
    top = [(x, y, z_top_m) for x, y in corners]
    faces = _Faces()
    faces.add_quad(bottom, (0.0, 0.0, -1.0))
    faces.add_quad(top, (0.0, 0.0, 1.0))
    for index in range(len(corners)):
        following = (index + 1) % len(corners)
        (x0, y0), (x1, y1) = corners[index], corners[following]
        outward = (y1 - y0, -(x1 - x0), 0.0)
        faces.add_quad([bottom[index], bottom[following], top[following], top[index]], outward)
    return faces


def _mesh_of(mesh_id: str, semantic: MeshSemantic, room: str | None, faces: _Faces) -> Mesh:
    return Mesh(
        id=mesh_id,
        semantic=semantic,
        material_id=UNASSIGNED_MATERIAL_ID,
        room=room,
        vertices=faces.vertices,
        triangles=faces.triangles,
    )


def _axis_xy(axis: PlanAxis, along_m: float, across_m: float) -> tuple[float, float]:
    """沿墙、跨墙两个量 → 平面坐标。竖墙横墙只差一次 x/y 对调，对调只写在这一处。"""
    if axis == "vertical":
        return (across_m, along_m)
    return (along_m, across_m)


def _axis_point(
    axis: PlanAxis, along_m: float, across_m: float, z_m: float
) -> tuple[float, float, float]:
    x_m, y_m = _axis_xy(axis, along_m, across_m)
    return (x_m, y_m, z_m)


def _axis_base_xy(
    axis: PlanAxis, along0_m: float, along1_m: float, across0_m: float, across1_m: float
) -> list[tuple[float, float]]:
    return [
        _axis_xy(axis, along0_m, across0_m),
        _axis_xy(axis, along1_m, across0_m),
        _axis_xy(axis, along1_m, across1_m),
        _axis_xy(axis, along0_m, across1_m),
    ]


# ---------------------------------------------------------------------------
# 三、洞：门 / 窗 / 过口
# ---------------------------------------------------------------------------


def opening_kind(opening: PlanOpening, heights: HeightRules) -> OpeningKind:
    """这个洞算门、算窗还是算过口。**确定性推法，只写在这一处。**

    产出侧这一层只分洞在外墙还是内墙、不分门与窗（口径见 models.py
    :class:`~render3d_worker.models.PlanOpening`），而三维绕不开这件事——挖多高的洞
    取决于它是门还是窗。规则本身是数据（`HeightRules.outer_opening_kind` 与
    `inner_opening_kind`），换法只换包。产出侧补上门窗识别那一批之后改这一处。
    """
    return heights.outer_opening_kind if opening.is_on_outer_wall else heights.inner_opening_kind


def _opening_z_range(kind: OpeningKind, heights: HeightRules) -> tuple[float, float]:
    if kind == "door":
        return (0.0, heights.door_height_m)
    if kind == "pass":
        return (0.0, heights.pass_height_m)
    return (heights.window_sill_height_m, heights.window_head_height_m)


def _check_heights(heights: HeightRules) -> None:
    """竖向那几个数自相矛盾就当场炸。它们是常识档位，错了通常是包填错而不是这户特别。"""
    ceiling = heights.ceiling_height_m
    if ceiling <= 0:
        raise MeshBuildError(f"层高是 {ceiling}m：墙拉不起来")
    for name, height in (("门高", heights.door_height_m), ("过口高", heights.pass_height_m)):
        if height <= 0 or height > ceiling:
            raise MeshBuildError(f"{name} {height}m 不在 (0, 层高 {ceiling}m] 内")
    if heights.window_sill_height_m < 0:
        raise MeshBuildError(f"窗台高 {heights.window_sill_height_m}m 是负的")
    if heights.window_head_height_m <= heights.window_sill_height_m:
        raise MeshBuildError(
            f"窗顶 {heights.window_head_height_m}m 不高于窗台 {heights.window_sill_height_m}m"
        )
    if heights.window_head_height_m > ceiling:
        raise MeshBuildError(f"窗顶 {heights.window_head_height_m}m 高过层高 {ceiling}m")


@dataclass(frozen=True)
class _WallLine:
    """一道墙线：来自 `outline` 还是 `walls`，加它在那份清单里的位置。

    来源要留着，因为两处会有重合的段（不做布尔求并），网格 id 得分得开它们。
    """

    source: str
    index: int
    wall: PlanWall


@dataclass(frozen=True)
class _Cut:
    """一道墙上的一个洞：沿墙的起讫（归一化）+ 竖向的起讫（米）。"""

    opening_index: int
    kind: OpeningKind
    start_ratio: float
    end_ratio: float
    z_bottom_m: float
    z_top_m: float


@dataclass(frozen=True)
class _GapFill:
    """一个落在墙段空隙里的洞：不用切墙，反过来要**补**过梁与窗下墙。

    `position_ratio` 与 `thickness_ratio` **不是自己定的，是抄同直线上最近那段墙的**——
    补出来的块必须和邻墙共面，否则洞口两侧会看得见一道台阶。
    """

    opening_index: int
    kind: OpeningKind
    axis: PlanAxis
    start_ratio: float
    end_ratio: float
    position_ratio: float
    thickness_ratio: float
    anchor_id: str
    z_bottom_m: float
    z_top_m: float


@dataclass(frozen=True)
class _OpeningLayout:
    """整户的洞怎么落地：哪些墙线要切、切成什么，哪些洞要补、补在哪儿。

    切与补在一处算完，是因为**"要不要补"取决于"切不切得着"**——一个洞在任何一道墙上
    都切不着，才轮到补。分两处算就得把这个判据写两遍。
    """

    lines: list[_WallLine]
    cuts_per_line: list[list[_Cut]]
    fills: list[_GapFill]
    unplaced_opening_indices: list[int]


def _wall_lines(plan: FloorplanGeometry) -> list[_WallLine]:
    """外轮廓在前、网格墙在后。次序与 2D 母版 `(*outline, *walls)` 一致。"""
    return [
        *(_WallLine("outline", index, wall) for index, wall in enumerate(plan.outline)),
        *(_WallLine("grid", index, wall) for index, wall in enumerate(plan.walls)),
    ]


def _is_degenerate(line: _WallLine, ruler: _Ruler) -> bool:
    """这道墙线是不是退化段（长度为零）。判据只写在这一处。

    **上游真的会给出零长墙段**：网格投票产的 `walls` 里，2026-08-30 那份真跑存档 61 段
    有 5 段 `start_ratio == end_ratio`（第 32/36/47/49/55 段）。2D 那侧不受影响——画一条
    零长的线等于没画；三维要起体，零长拉出来是个零体积的退化块，会污染网格与遮罩。

    所以**跳过、计数、不静默**：跳了哪几段随场景包带出去
    （:attr:`ScenePackage.degenerate_wall_count`），数目异常就是上游那一步坏了，看得见。
    """
    ends_m = (
        ruler.along_m(line.wall.axis, line.wall.start_ratio),
        ruler.along_m(line.wall.axis, line.wall.end_ratio),
    )
    return abs(ends_m[1] - ends_m[0]) <= _MIN_SEGMENT_M


def degenerate_wall_ids(plan: FloorplanGeometry, scale: PlanScale) -> list[str]:
    """被跳过的退化墙线 id（`来源:序号`）——与 :func:`build_shell` 同一个判据。"""
    ruler = _Ruler(plan, scale)
    return [
        f"{line.source}:{line.index}" for line in _wall_lines(plan) if _is_degenerate(line, ruler)
    ]


def _live_wall_lines(plan: FloorplanGeometry, ruler: _Ruler) -> list[_WallLine]:
    """真会起体的那些墙线（退化段不算）。

    洞往哪儿落、补出来的块跟谁对齐，都只看这一份清单：**拿一段自己都没起体的墙当基准，
    对齐的是个不存在的东西**。"""
    return [line for line in _wall_lines(plan) if not _is_degenerate(line, ruler)]


def _cuts_on_line(line: _WallLine, openings: list[PlanOpening], heights: HeightRules) -> list[_Cut]:
    """落在这道墙上的洞，按位置排好、去掉互相压着的。

    **重叠的洞取排序后的第一个，后来的丢掉**：两个洞高度不同时合不出一个高度，取第一个
    是确定性的。真出现重叠说明几何产物里同一处出了两个洞——那是产出侧要修的事，
    这儿不替它猜。丢掉的洞不会计入 `cut_opening_kinds`，数上看得见。
    """
    wall = line.wall
    low_ratio = min(wall.start_ratio, wall.end_ratio)
    high_ratio = max(wall.start_ratio, wall.end_ratio)
    found: list[_Cut] = []
    for index, opening in enumerate(openings):
        if opening.axis != wall.axis:
            continue
        if abs(opening.position_ratio - wall.position_ratio) > _SAME_LINE_TOLERANCE_RATIO:
            continue
        start_ratio = max(low_ratio, min(opening.start_ratio, opening.end_ratio))
        end_ratio = min(high_ratio, max(opening.start_ratio, opening.end_ratio))
        if end_ratio <= start_ratio:
            continue
        kind = opening_kind(opening, heights)
        z_bottom_m, z_top_m = _opening_z_range(kind, heights)
        found.append(_Cut(index, kind, start_ratio, end_ratio, z_bottom_m, z_top_m))
    found.sort(key=lambda cut: (cut.start_ratio, cut.end_ratio, cut.opening_index))
    accepted: list[_Cut] = []
    for cut in found:
        if accepted and cut.start_ratio < accepted[-1].end_ratio:
            continue
        accepted.append(cut)
    return accepted


def _gap_ratio(opening: PlanOpening, wall: PlanWall) -> float:
    """洞与这段墙沿墙方向隔着多远（归一化）。负数＝压着，0＝恰好相接。"""
    open_low, open_high = sorted((opening.start_ratio, opening.end_ratio))
    wall_low, wall_high = sorted((wall.start_ratio, wall.end_ratio))
    return max(open_low, wall_low) - min(open_high, wall_high)


def _nearest_same_line_wall(opening: PlanOpening, lines: list[_WallLine]) -> _WallLine | None:
    """同一条直线（同轴、位置在容差内）上离这个洞最近的那段墙。一段都没有就返回 None。

    同距的取清单里靠前的那一段——外轮廓在前、网格墙在后，次序是写死的，所以这个选择
    是确定性的。真数据里洞两侧那两段墙的厚度与位置完全相同（2026-08-30 那份存档 10 个
    落空隙的洞全是如此），取哪一段都对齐得上；两侧真给出不同厚度时，补出来的块会跟其中
    一侧齐、跟另一侧差一点——那是上游同一道墙给了两个厚度，得回上游修。
    """
    best_line: _WallLine | None = None
    best_gap_ratio = math.inf
    for line in lines:
        wall = line.wall
        if wall.axis != opening.axis:
            continue
        if abs(wall.position_ratio - opening.position_ratio) > _SAME_LINE_TOLERANCE_RATIO:
            continue
        gap_ratio = _gap_ratio(opening, wall)
        if gap_ratio < best_gap_ratio:
            best_gap_ratio, best_line = gap_ratio, line
    return best_line


def _layout_openings(
    plan: FloorplanGeometry, ruler: _Ruler, heights: HeightRules
) -> _OpeningLayout:
    """每个洞该切进哪几道墙、还是该补成一块。**"要不要补"的判据只写在这儿。**

    两种形态都真出现过，所以两条路都得走通：

    - **墙压着洞**——墙段跨过洞的位置。把墙沿洞的起讫切开（:func:`_wall_pieces`），
      过梁、窗下墙、洞壁都由切出来的块与 `reveal` 套框给出。
    - **墙在洞处本来就断开**——洞落在两段墙之间的空隙里，与同直线上任何一段墙都不相交。
      2026-08-30 那份真跑存档 15 个洞里有 10 个是这样：洞与最近那段墙沿墙方向差
      **恰好一个源图像素**（竖向 0.00068、横向 0.00093 归一化，即 1080×1466 那张图的 1px），
      是网格量化留下的缝，不是"这儿没有墙"。这种洞不用切——**它缺的是过梁和窗下墙**，
      不补的话门窗就是从地面通到天花的大开口。

    补的那两块，位置与厚度**抄同直线上最近那段墙**（:func:`_nearest_same_line_wall`），
    不自己编一个：编出来的厚度会在洞口两侧留一道台阶，而"取不到"这件事本身就是要看见的
    信号——同直线上一段墙都找不到，说明这个洞落在了没有墙的地方，它会进
    `unplaced_opening_indices`，也就**不计入洞数**。

    **补出来的洞不生成 `reveal`**：洞壁由两侧邻墙的端面充当（那两段墙各自是完整的
    长方体，端面本来就朝着洞口）。再补一层 reveal 只会与端面重叠打架。
    """
    lines = _live_wall_lines(plan, ruler)
    cuts_per_line = [_cuts_on_line(line, plan.openings, heights) for line in lines]
    cut_indices = {cut.opening_index for cuts in cuts_per_line for cut in cuts}

    fills: list[_GapFill] = []
    unplaced: list[int] = []
    # 只拿 set 做"在不在"的判断，出场次序一律跟着输入列表——迭代 set 不是确定性的
    for index, opening in enumerate(plan.openings):
        if index in cut_indices:
            continue
        anchor = _nearest_same_line_wall(opening, lines)
        if anchor is None:
            unplaced.append(index)
            continue
        kind = opening_kind(opening, heights)
        z_bottom_m, z_top_m = _opening_z_range(kind, heights)
        start_ratio, end_ratio = sorted((opening.start_ratio, opening.end_ratio))
        fills.append(
            _GapFill(
                opening_index=index,
                kind=kind,
                axis=opening.axis,
                start_ratio=start_ratio,
                end_ratio=end_ratio,
                position_ratio=anchor.wall.position_ratio,
                thickness_ratio=anchor.wall.thickness_ratio,
                anchor_id=f"{anchor.source}:{anchor.index}",
                z_bottom_m=z_bottom_m,
                z_top_m=z_top_m,
            )
        )
    return _OpeningLayout(lines, cuts_per_line, fills, unplaced)


def built_opening_kinds(
    plan: FloorplanGeometry, scale: PlanScale, heights: HeightRules
) -> list[OpeningKind]:
    """**真做出来的**那些洞各是什么种类，按输入次序（切出来的和补出来的都算）。

    数的不是输入里有几个洞，是几个洞真落到了墙上。少掉的那些在
    :attr:`_OpeningLayout.unplaced_opening_indices` 里，判读方式是一句减法：
    `sum(opening_count_by_kind.values())` 对不上 `len(plan.openings)`，就是有洞落在了
    没有墙的地方。**选"计数上少掉"而不是另出一个自证数**，理由有两条：这个数已经能把
    事情说清楚（少几个就是漏几个），而再加一个字段要改 `models.py` 的契约，得两侧一起动。

    一个洞同时落在外轮廓与网格墙上（重合的段）只数一次。
    """
    ruler = _Ruler(plan, scale)
    layout = _layout_openings(plan, ruler, heights)
    placed = {cut.opening_index for cuts in layout.cuts_per_line for cut in cuts}
    placed.update(fill.opening_index for fill in layout.fills)
    return [
        opening_kind(opening, heights)
        for index, opening in enumerate(plan.openings)
        if index in placed
    ]


def count_openings_by_kind(
    plan: FloorplanGeometry, scale: PlanScale, heights: HeightRules
) -> dict[str, int]:
    """真做出来的洞按种类报数。次序按 :data:`_KIND_ORDER` 写死，一个都没有的种类不出现。"""
    kinds = built_opening_kinds(plan, scale, heights)
    return {kind: kinds.count(kind) for kind in _KIND_ORDER if kind in kinds}


# ---------------------------------------------------------------------------
# 四、壳体：地板 / 吊顶 / 墙 / 洞壁
# ---------------------------------------------------------------------------


def _floor_and_ceiling(plan: FloorplanGeometry, ruler: _Ruler, heights: HeightRules) -> list[Mesh]:
    """房间遮罩的每个矩形出一块地板 + 一块吊顶。

    地板法向朝上、吊顶法向朝下——两者都朝屋里。屋里是相机待的地方，朝外的面在底渲里
    只会挡住自己。
    """
    meshes: list[Mesh] = []
    for room in plan.rooms:
        for index, box in enumerate(room.boxes):
            left, top, right, bottom = box
            x0_m, x1_m = ruler.x_m(left), ruler.x_m(right)
            y0_m, y1_m = ruler.y_m(top), ruler.y_m(bottom)
            if abs(x1_m - x0_m) <= _MIN_SEGMENT_M or abs(y1_m - y0_m) <= _MIN_SEGMENT_M:
                continue
            corners = [(x0_m, y0_m), (x1_m, y0_m), (x1_m, y1_m), (x0_m, y1_m)]
            floor = _Faces()
            floor.add_quad([(x, y, 0.0) for x, y in corners], (0.0, 0.0, 1.0))
            meshes.append(_mesh_of(f"floor:{room.name}:{index}", "floor", room.name, floor))
            ceiling = _Faces()
            ceiling.add_quad(
                [(x, y, heights.ceiling_height_m) for x, y in corners], (0.0, 0.0, -1.0)
            )
            meshes.append(_mesh_of(f"ceiling:{room.name}:{index}", "ceiling", room.name, ceiling))
    return meshes


def _reveal_mesh(
    line: _WallLine,
    cut: _Cut,
    along0_m: float,
    along1_m: float,
    across0_m: float,
    across1_m: float,
) -> Mesh:
    """洞壁：两侧洞口套 + 洞顶（过梁底面），窗还多一块窗台面。

    **落在地面上的洞（门、过口）不出下面那块面**：它与地板共面，两块共面的片在光栅里
    会互相闪，而那块面本来就是地板。
    """
    axis = line.wall.axis
    faces = _Faces()
    z0_m, z1_m = cut.z_bottom_m, cut.z_top_m
    for along_m, toward_sign in ((along0_m, 1.0), (along1_m, -1.0)):
        faces.add_quad(
            [
                _axis_point(axis, along_m, across0_m, z0_m),
                _axis_point(axis, along_m, across1_m, z0_m),
                _axis_point(axis, along_m, across1_m, z1_m),
                _axis_point(axis, along_m, across0_m, z1_m),
            ],
            _axis_point(axis, toward_sign, 0.0, 0.0),
        )

    def horizontal_face(z_m: float, toward: tuple[float, float, float]) -> None:
        faces.add_quad(
            [
                _axis_point(axis, along0_m, across0_m, z_m),
                _axis_point(axis, along1_m, across0_m, z_m),
                _axis_point(axis, along1_m, across1_m, z_m),
                _axis_point(axis, along0_m, across1_m, z_m),
            ],
            toward,
        )

    horizontal_face(z1_m, (0.0, 0.0, -1.0))
    if z0_m > _MIN_SEGMENT_M:
        horizontal_face(z0_m, (0.0, 0.0, 1.0))
    mesh_id = f"reveal:{cut.kind}:{line.source}:{line.index}:{cut.opening_index}"
    return _mesh_of(mesh_id, "reveal", None, faces)


def _across_band_m(
    axis: PlanAxis, position_ratio: float, thickness_ratio: float, ruler: _Ruler, what: str
) -> tuple[float, float]:
    """一道墙线跨墙方向占的那一条带（米）。**切出来的块与补出来的块都走这一处**——
    共面对齐靠的就是"位置与厚度经同一段算式"，各算各的迟早差半个厚度。"""
    thickness_m = (
        ruler.across_x_m(thickness_ratio)
        if axis == "vertical"
        else ruler.across_y_m(thickness_ratio)
    )
    if thickness_m <= 0:
        raise MeshBuildError(
            f"{what} 的厚度是 {thickness_ratio}：没有厚度的墙起不了体，也不在这儿替它编一个厚度"
        )
    center_m = ruler.x_m(position_ratio) if axis == "vertical" else ruler.y_m(position_ratio)
    return center_m - thickness_m / 2, center_m + thickness_m / 2


def _wall_solid(
    mesh_id: str,
    axis: PlanAxis,
    along_span_m: tuple[float, float],
    across_span_m: tuple[float, float],
    z_span_m: tuple[float, float],
) -> Mesh | None:
    """一块墙体长方体。薄到只剩残渣的（沿墙或竖向）不出网格，返回 None。"""
    (along0_m, along1_m), (z0_m, z1_m) = along_span_m, z_span_m
    if along1_m - along0_m <= _MIN_SEGMENT_M or z1_m - z0_m <= _MIN_SEGMENT_M:
        return None
    base = _axis_base_xy(axis, along0_m, along1_m, across_span_m[0], across_span_m[1])
    return _mesh_of(mesh_id, "wall", None, _prism_faces(base, z0_m, z1_m))


def _wall_pieces(
    line: _WallLine, cuts: list[_Cut], ruler: _Ruler, heights: HeightRules
) -> list[Mesh]:
    """一道墙线 → 若干长方体 + 若干洞壁。**墙压着洞**那条路的切段就在这儿发生。

    切法：沿墙走一遍，洞之前的那段起一块整墙，洞上留过梁、洞下留窗下墙（门没有窗下墙），
    最后收尾那段再起一块。全是加减法，没有布尔运算。
    """
    wall = line.wall
    across_span_m = _across_band_m(
        wall.axis,
        wall.position_ratio,
        wall.thickness_ratio,
        ruler,
        f"{line.source} 第 {line.index} 道墙",
    )
    ends_m = (ruler.along_m(wall.axis, wall.start_ratio), ruler.along_m(wall.axis, wall.end_ratio))
    along0_m, along1_m = min(ends_m), max(ends_m)
    # 零长的退化段在 build_shell 就跳掉了（判据 `_is_degenerate` 只写在那一处）

    ceiling_m = heights.ceiling_height_m
    pieces: list[Mesh] = []
    span_count = 0

    def solid(along_from_m: float, along_to_m: float, z0_m: float, z1_m: float, part: str) -> None:
        nonlocal span_count
        mesh_id = f"wall:{line.source}:{line.index}:{part}:{span_count}"
        block = _wall_solid(
            mesh_id, wall.axis, (along_from_m, along_to_m), across_span_m, (z0_m, z1_m)
        )
        if block is None:
            return
        pieces.append(block)
        span_count += 1

    cursor_m = along0_m
    for cut in cuts:
        cut0_m = max(along0_m, ruler.along_m(wall.axis, cut.start_ratio))
        cut1_m = min(along1_m, ruler.along_m(wall.axis, cut.end_ratio))
        solid(cursor_m, cut0_m, 0.0, ceiling_m, "span")
        solid(cut0_m, cut1_m, cut.z_top_m, ceiling_m, "lintel")
        solid(cut0_m, cut1_m, 0.0, cut.z_bottom_m, "sill")
        pieces.append(_reveal_mesh(line, cut, cut0_m, cut1_m, across_span_m[0], across_span_m[1]))
        cursor_m = max(cursor_m, cut1_m)
    solid(cursor_m, along1_m, 0.0, ceiling_m, "span")
    return pieces


def _fill_pieces(fill: _GapFill, ruler: _Ruler, heights: HeightRules) -> list[Mesh]:
    """一个落在墙段空隙里的洞 → 过梁 +（窗才有的）窗下墙。**墙在洞处断开**那条路。

    这儿只补洞上下那两块，不补洞本身那一段整墙——洞就是要空着的地方。洞壁不补，理由见
    :func:`_layout_openings`（邻墙的端面就是洞壁）。位置与厚度全部来自 `fill`，而 `fill`
    抄的是邻墙，所以补出来的块与邻墙经的是同一段算式（:func:`_across_band_m`），共面。
    """
    across_span_m = _across_band_m(
        fill.axis,
        fill.position_ratio,
        fill.thickness_ratio,
        ruler,
        f"补洞 {fill.opening_index} 依据的墙 {fill.anchor_id}",
    )
    along_span_m = (
        ruler.along_m(fill.axis, fill.start_ratio),
        ruler.along_m(fill.axis, fill.end_ratio),
    )
    parts = (
        ("lintel", (fill.z_top_m, heights.ceiling_height_m)),
        ("sill", (0.0, fill.z_bottom_m)),
    )
    pieces: list[Mesh] = []
    for part, z_span_m in parts:
        block = _wall_solid(
            f"wall:fill:{fill.opening_index}:{part}",
            fill.axis,
            along_span_m,
            across_span_m,
            z_span_m,
        )
        if block is not None:
            pieces.append(block)
    return pieces


def build_shell(plan: FloorplanGeometry, scale: PlanScale, heights: HeightRules) -> list[Mesh]:
    """户型的壳：地板、吊顶、墙体、洞壁。同一份输入建两次，网格逐字相同。

    出场次序写死为"地板与吊顶 → 墙线（外轮廓在前）→ 补出来的洞（按输入次序）"，不是为了
    好看：网格清单的次序会一路带进场景包的 JSON，次序漂了确定性就没了。补的块排在最后，
    因为它们不属于任何一道墙线，插在中间就得为"插在哪儿"再定一条规矩。
    """
    ruler = _Ruler(plan, scale)
    _check_heights(heights)
    meshes = _floor_and_ceiling(plan, ruler, heights)
    layout = _layout_openings(plan, ruler, heights)
    for line, cuts in zip(layout.lines, layout.cuts_per_line, strict=True):
        meshes.extend(_wall_pieces(line, cuts, ruler, heights))
    for fill in layout.fills:
        meshes.extend(_fill_pieces(fill, ruler, heights))
    return meshes


# ---------------------------------------------------------------------------
# 五、家具：一件一个体块
# ---------------------------------------------------------------------------


def build_furnishings(
    placements: list[FurnishingPlacement],
    plan: FloorplanGeometry,
    scale: PlanScale,
    heights: HeightRules,
) -> list[Mesh]:
    """每件家具一个长方体，绕竖轴按 `yaw_deg` 转，坐在地上（z=0)。

    **体块就是今天的正确形态**：三维资产库不存在，编一个更像沙发的形状等于让这一层
    替产品做没人拍过板的决定；而底渲四路要的是"这儿有个多大的东西、属于哪间房"，
    体块答得完整。资产库到位那一批再换形状，换的是这一个函数。

    `yaw_deg` 绕 +z 转，0 度时进深朝 +y（平面图的"下"），从 +z 俯视逆时针为正——
    口径同 models.py :class:`~render3d_worker.models.FurnishingPlacement`。
    """
    ruler = _Ruler(plan, scale)
    _check_heights(heights)
    meshes: list[Mesh] = []
    for placement in placements:
        sizes = (placement.width_m, placement.depth_m, placement.height_m)
        if min(sizes) <= 0:
            raise MeshBuildError(f"家具 {placement.id} 的尺寸有非正数：{sizes}m")
        if placement.height_m > heights.ceiling_height_m:
            raise MeshBuildError(
                f"家具 {placement.id} 高 {placement.height_m}m，超过层高 "
                f"{heights.ceiling_height_m}m：多半是尺寸的单位填错了"
            )
        center_x_m = ruler.x_m(placement.center_x_ratio)
        center_y_m = ruler.y_m(placement.center_y_ratio)
        yaw_rad = math.radians(placement.yaw_deg)
        cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)
        half_width_m, half_depth_m = placement.width_m / 2, placement.depth_m / 2
        local = [
            (-half_width_m, -half_depth_m),
            (half_width_m, -half_depth_m),
            (half_width_m, half_depth_m),
            (-half_width_m, half_depth_m),
        ]
        base = [
            (center_x_m + x * cos_yaw - y * sin_yaw, center_y_m + x * sin_yaw + y * cos_yaw)
            for x, y in local
        ]
        faces = _prism_faces(base, 0.0, placement.height_m)
        meshes.append(_mesh_of(placement.id, "furnishing", placement.room, faces))
    return meshes
