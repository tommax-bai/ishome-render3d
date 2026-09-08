"""确定性 mock 摆场：`DesignPackage` 的几何 → 一组 `FurnishingPlacement`。

**为什么要有这份 mock**：上游（DeepDesign 的软装决策 + 三维资产库）今天不存在，
`DesignPackage.furnishings` 因此常是空的——底渲出的是只有墙、洞、房间的"光壳"。真跑
发现光壳喂给下游写实化模型时，模型因为可参照的信息太少，会把内部隔墙一并改掉；同一条
流程喂带家具方块的场景包时，几何锁得明显更好。**结论：底渲里有没有家具，直接影响下游
几何保真度**——这与 `HeightRules` 那段"常规住宅档位 mock，先用来做测试"的处理同形。

**边界，写在最前面**：家具摆放是设计决策，不是渲染层的职责。这份 mock **只是测试脚手架**：
- 只在 CLI 显式加 `--mock-furnishing` 时才会被调用（见 `cli.py`），默认关闭、不进
  `scene_compile` 的生产路径——`scene_compile.py` / `mesh.py` 一行没改，它们不知道
  这个模块存在。
- 尺寸是**常规住宅档位**（下面每个常量的 docstring 都写着口径来源），不是这户人家
  的软装决策，也不是量出来的。
- 房间名认不得就不摆——**不瞎猜房型**，认不出的房间数随 `MockFurnishingReport` 带出去，
  CLI 打自证时如实报。

**确定性**：不用 `random`，不看字典遍历序（分组后一律 `sorted()` 排过），不看时间。
同一份输入永远算出同一组摆放（同一份场景包"编两次逐字节相同"那条口径，参见
`scene_compile.compile_scene_package` 的 docstring）。

**摆法**：家具只贴墙或摆在房间正中，`yaw_deg` 只取 0/90/180/270（贴哪面墙决定朝向，
见 `_WALL_YAW_DEG`），足迹因此恒为轴对齐矩形——足迹落在哪一间房的 `boxes` 并集内、
挡没挡门，都能用简单的矩形运算测出来，不用引几何库。**这是刻意的简化**：真实软装
里家具会斜摆、会贴着不规则墙角走，这份 mock 都做不到，做不到就不做（见模块底部
"没做到什么"那几句）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from render3d_worker import mesh
from render3d_worker.models import (
    DesignPackage,
    FloorplanGeometry,
    FurnishingPlacement,
    HeightRules,
    OpeningKind,
)

type Rect = tuple[float, float, float, float]
"""一个矩形：`(left, top, right, bottom)`，坐标口径同 `RoomOutline.boxes`——归一化平面
坐标（0~1），不是毫米。本模块几乎全部运算都在这个坐标系里做矩形运算，只在需要知道
"多大算合适"（家具尺寸、留多少净空）时才换算成毫米。"""

# ---------------------------------------------------------------------------
# 一、家具尺寸档位：常规住宅 mock，不是这户人家的实测/设计值
# ---------------------------------------------------------------------------

BED_WIDTH_MM, BED_DEPTH_MM, BED_HEIGHT_MM = 1800, 2000, 450
"""双人床。1.8×2.0m 是国标双人床最常见的档位；0.45m 是床垫+床架的常见高度。"""

NIGHTSTAND_WIDTH_MM, NIGHTSTAND_DEPTH_MM, NIGHTSTAND_HEIGHT_MM = 450, 400, 550
WARDROBE_WIDTH_MM, WARDROBE_DEPTH_MM, WARDROBE_HEIGHT_MM = 1200, 600, 2200
"""两门衣柜的常规进深与高度；宽度取偏保守的 1.2m，不按房间大小放大——放大是软装的活。"""

SOFA_WIDTH_MM, SOFA_DEPTH_MM, SOFA_HEIGHT_MM = 2200, 900, 800
"""三人位布艺沙发的常规档位。"""

COFFEE_TABLE_WIDTH_MM, COFFEE_TABLE_DEPTH_MM, COFFEE_TABLE_HEIGHT_MM = 1200, 600, 420
TV_CABINET_WIDTH_MM, TV_CABINET_DEPTH_MM, TV_CABINET_HEIGHT_MM = 1800, 400, 500

DINING_TABLE_WIDTH_MM, DINING_TABLE_DEPTH_MM, DINING_TABLE_HEIGHT_MM = 1400, 800, 750
"""四人餐桌的常规档位。"""
DINING_CHAIR_WIDTH_MM, DINING_CHAIR_DEPTH_MM, DINING_CHAIR_HEIGHT_MM = 450, 450, 900

KITCHEN_CABINET_WIDTH_MM, KITCHEN_CABINET_DEPTH_MM, KITCHEN_CABINET_HEIGHT_MM = 1800, 600, 850
"""一段直跑橱柜台面的常规档位。厨房布局（一字/L 形/U 形）是软装决策，这儿只给一段
台面体块——"这儿有橱柜"比"猜一个像样的橱柜形状"更诚实。"""

BATHROOM_FIXTURE_WIDTH_MM, BATHROOM_FIXTURE_DEPTH_MM, BATHROOM_FIXTURE_HEIGHT_MM = 1200, 500, 850
"""洁具体块：马桶+洗手台合并成一段贴墙体块的常规尺寸。**不拆成马桶/洗手盆/淋浴分开摆**——
真实布局要看给排水点位（这层几何不含这个信息），拆开摆等于替设计瞎猜点位。"""

DESK_WIDTH_MM, DESK_DEPTH_MM, DESK_HEIGHT_MM = 1200, 600, 750
BOOKSHELF_WIDTH_MM, BOOKSHELF_DEPTH_MM, BOOKSHELF_HEIGHT_MM = 800, 300, 2000

WASHER_WIDTH_MM, WASHER_DEPTH_MM, WASHER_HEIGHT_MM = 600, 600, 850

WALL_CLEARANCE_MM = 40.0
"""家具背贴墙面留的缝。取值同这条线此前给 `design-package-full.json` 摆家具时用的口径
（见 `tests/fixtures/README.md`）：0 缝会让家具与墙面共面，光栅两个面会互相闪。"""

ITEM_GAP_MM = 50.0
"""同一间房里两件家具之间留的缝，避免贴面打架，量级同 `WALL_CLEARANCE_MM`。"""

DOOR_CLEARANCE_MM = 600.0
"""门洞前留的净空——常规住宅通行净宽的下限，不是这户实测。摆家具时**当作门两侧都是
房间内部**对称留出（不判断门到底朝哪边开），因此偏保守：宁可少摆一件，也不堵门。"""

# ---------------------------------------------------------------------------
# 二、房间名 → 家具品类：只认这张表里的名字，认不得就不摆
# ---------------------------------------------------------------------------

_ROOM_CATEGORY: dict[str, str] = {
    "卧室": "bedroom",
    "主卧": "bedroom",
    "主卧室": "bedroom",
    "次卧": "bedroom",
    "次卧室": "bedroom",
    "三卧": "bedroom",
    "客卧": "bedroom",
    "小孩房": "bedroom",
    "儿童房": "bedroom",
    "老人房": "bedroom",
    "客厅": "living_room",
    "起居室": "living_room",
    "餐厅": "dining_room",
    "厨房": "kitchen",
    "卫生间": "bathroom",
    "洗手间": "bathroom",
    "厕所": "bathroom",
    "主卫": "bathroom",
    "次卫": "bathroom",
    "公卫": "bathroom",
    "书房": "study",
    "阳台": "balcony",
}
"""**显式白名单，不做子串/通配匹配**——"XX房"不必然是卧室（厨房、书房本身就带"房"字），
瞎猜出来的类别比不摆更糟。真实几何产出的房间名如果不在这张表里（玄关、储物间、
飘窗一类），本模块原样跳过，不摆任何家具，缺口随 `MockFurnishingReport.unrecognized_rooms`
带出去。"""

# ---------------------------------------------------------------------------
# 三、房间几何：把网格切碎的 boxes 尽量拼回大块，取其中最大的一块当摆场
# ---------------------------------------------------------------------------

_MERGE_KEY_DECIMALS = 9
_MERGE_TOUCH_TOLERANCE_RATIO = 1e-6
_MAX_MERGE_PASSES = 8
"""收敛上限，不是判据——真实房间的 boxes 数远小于能撑满这个上限的规模，拼不动时循环
自己会提前退出（见 `_merge_adjacent`）。"""


def _merge_along(axis: Literal["x", "y"], rects: list[Rect]) -> tuple[list[Rect], bool]:
    """把 `axis` 方向上首尾相接、另一维完全同界的矩形拼成一块。

    只在整条边界重合（容差内）时才拼——拼出来的矩形因此严格是原并集的一部分，从不会
    比原 boxes 圈出的地方大一寸。"""
    groups: dict[tuple[float, float], list[Rect]] = {}
    for rect in rects:
        key = (rect[1], rect[3]) if axis == "x" else (rect[0], rect[2])
        rounded_key = (round(key[0], _MERGE_KEY_DECIMALS), round(key[1], _MERGE_KEY_DECIMALS))
        groups.setdefault(rounded_key, []).append(rect)

    merged_any = False
    result: list[Rect] = []
    for key in sorted(groups):  # 迭代字典必须先排序：顺序不许随哈希跑
        group = sorted(groups[key], key=lambda r: r[0] if axis == "x" else r[1])
        run: list[Rect] = []
        for rect in group:
            if run and _touches(run[-1], rect, axis):
                run[-1] = _joined(run[-1], rect, axis)
                merged_any = True
            else:
                run.append(rect)
        result.extend(run)
    return result, merged_any


def _touches(a: Rect, b: Rect, axis: Literal["x", "y"]) -> bool:
    gap = (b[0] - a[2]) if axis == "x" else (b[1] - a[3])
    return abs(gap) <= _MERGE_TOUCH_TOLERANCE_RATIO


def _joined(a: Rect, b: Rect, axis: Literal["x", "y"]) -> Rect:
    if axis == "x":
        return (a[0], a[1], b[2], a[3])
    return (a[0], a[1], a[2], b[3])


def _merge_adjacent(rects: list[Rect]) -> list[Rect]:
    current = list(rects)
    for _ in range(_MAX_MERGE_PASSES):
        current, merged_x = _merge_along("x", current)
        current, merged_y = _merge_along("y", current)
        if not (merged_x or merged_y):
            break
    return current


def _largest_usable_rect(boxes: list[Rect]) -> Rect | None:
    """房间里能摆家具的那一块地。**只取拼出来的最大一块，不用整个并集**：并集在 L 形、
    T 形房间里不是矩形，家具是轴对齐长方体，摆不出非矩形的地——宁可只用房间的一部分，
    也不让家具的角落探出房间外墙（同"做不到就别硬做"那条口径，代价是异形房间会有一角
    量出来的空间没被用上）。"""
    if not boxes:
        return None
    merged = _merge_adjacent(boxes)

    def area(rect: Rect) -> float:
        return (rect[2] - rect[0]) * (rect[3] - rect[1])

    return max(merged, key=lambda r: (area(r), -r[0], -r[1], -r[2], -r[3]))


def _rects_overlap(a: Rect, b: Rect) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# ---------------------------------------------------------------------------
# 四、房间坐标系：局部毫米制（原点＝摆场矩形左上角）↔ 全局归一化坐标
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RoomFrame:
    """一间房的摆场：矩形范围 + 这张图的毫米制换算。局部坐标原点在 `rect` 左上角，
    x 向右、y 向下，单位毫米——摆放算法只在这套坐标里想问题，只在算好之后才转回归一化坐标
    （`FurnishingPlacement` 存的是归一化中心点 + 毫米制尺寸，两套单位并存是契约本身的口径）。
    """

    rect: Rect
    mm_per_x: float
    m_per_y: float

    @property
    def width_mm(self) -> float:
        return (self.rect[2] - self.rect[0]) * self.mm_per_x

    @property
    def depth_mm(self) -> float:
        return (self.rect[3] - self.rect[1]) * self.m_per_y

    def to_ratio(self, local_x_mm: float, local_y_mm: float) -> tuple[float, float]:
        return (
            self.rect[0] + local_x_mm / self.mm_per_x,
            self.rect[1] + local_y_mm / self.m_per_y,
        )

    def to_local(self, ratio_x: float, ratio_y: float) -> tuple[float, float]:
        return (
            (ratio_x - self.rect[0]) * self.mm_per_x,
            (ratio_y - self.rect[1]) * self.m_per_y,
        )

    def footprint_ratio(
        self, local_x_mm: float, local_y_mm: float, half_x_mm: float, half_y_mm: float
    ) -> Rect:
        cx_ratio, cy_ratio = self.to_ratio(local_x_mm, local_y_mm)
        return (
            cx_ratio - half_x_mm / self.mm_per_x,
            cy_ratio - half_y_mm / self.m_per_y,
            cx_ratio + half_x_mm / self.mm_per_x,
            cy_ratio + half_y_mm / self.m_per_y,
        )

    def expand(self, rect: Rect, gap_mm: float) -> Rect:
        gap_x_ratio, gap_y_ratio = gap_mm / self.mm_per_x, gap_mm / self.m_per_y
        return (
            rect[0] - gap_x_ratio,
            rect[1] - gap_y_ratio,
            rect[2] + gap_x_ratio,
            rect[3] + gap_y_ratio,
        )


_DOORLIKE_KINDS: frozenset[OpeningKind] = frozenset({"door", "passage", "entry-door"})
"""留净空只为"人要走过去的洞"——窗不算，家具靠窗摆是室内设计常规做法。"""


def _door_clear_rects(
    plan: FloorplanGeometry, heights: HeightRules, room_name: str, mm_per_x: float, m_per_y: float
) -> list[Rect]:
    """这间房挨着的每个门洞，往房间里留一圈不许摆家具的矩形。"""
    clear_x_ratio, clear_y_ratio = DOOR_CLEARANCE_MM / mm_per_x, DOOR_CLEARANCE_MM / m_per_y
    rects: list[Rect] = []
    for opening in plan.openings:
        if room_name not in opening.connects:
            continue
        if mesh.opening_kind(opening, heights) not in _DOORLIKE_KINDS:
            continue
        start_ratio, end_ratio = sorted((opening.start_ratio, opening.end_ratio))
        if opening.axis == "vertical":
            rects.append(
                (
                    opening.position_ratio - clear_x_ratio,
                    start_ratio,
                    opening.position_ratio + clear_x_ratio,
                    end_ratio,
                )
            )
        else:
            rects.append(
                (
                    start_ratio,
                    opening.position_ratio - clear_y_ratio,
                    end_ratio,
                    opening.position_ratio + clear_y_ratio,
                )
            )
    return rects


# ---------------------------------------------------------------------------
# 五、两种摆法：贴墙、居中悬空
# ---------------------------------------------------------------------------

_WALL_YAW_DEG: dict[str, float] = {"top": 0.0, "bottom": 180.0, "left": 270.0, "right": 90.0}
"""贴哪面墙决定家具背朝哪个方向。推导见 `mesh.build_furnishings` 的旋转公式：`yaw_deg=0`
时家具进深朝 +y（贴 `top` 墙、背对小 y 一侧、正面朝房间内部）；顺着这张表转 90°/270°/180°，
背分别贴 `right`/`left`/`bottom` 墙。**摆放时只选墙，朝向随之唯一确定**——不会出现
位置摆对了、朝向转错的情况，也不用另外为"朝向"开一层判断。"""

_DEFAULT_WALL_ORDER: tuple[str, ...] = ("top", "left", "right", "bottom")
_OFFSET_FRACTIONS: tuple[float, ...] = (0.5, 0.25, 0.75, 0.1, 0.9)
"""沿墙试位置的固定候选表：先试墙正中，再试偏两侧——固定表 + 固定次序是确定性的来源之一
（不搜索、不逼近，最多试 `len(_DEFAULT_WALL_ORDER) × len(_OFFSET_FRACTIONS)` 个位置，
第一个满足"不挡门、不撞已摆家具"的就用它）。"""


@dataclass(frozen=True)
class _Slot:
    ratio_x: float
    ratio_y: float
    yaw_deg: float
    footprint: Rect
    wall: str


def _place_against_wall(
    frame: _RoomFrame,
    occupied: list[Rect],
    walls: tuple[str, ...],
    width_mm: float,
    depth_mm: float,
) -> _Slot | None:
    """按 `walls` 给的次序试贴墙，每面墙再按 `_OFFSET_FRACTIONS` 试位置，
    取第一个不挡门、不撞 `occupied` 里已摆家具的槽位。找不到就是这一步摆不下，返回 `None`
    ——不降级摆到墙外，也不无视碰撞硬摆。
    """
    for wall in walls:
        along_avail_mm = frame.width_mm if wall in ("top", "bottom") else frame.depth_mm
        across_avail_mm = frame.depth_mm if wall in ("top", "bottom") else frame.width_mm
        if depth_mm + 2 * WALL_CLEARANCE_MM > across_avail_mm:
            continue
        low_mm = WALL_CLEARANCE_MM + width_mm / 2
        high_mm = along_avail_mm - WALL_CLEARANCE_MM - width_mm / 2
        if low_mm > high_mm:
            continue
        for fraction in _OFFSET_FRACTIONS:
            along_mm = low_mm + fraction * (high_mm - low_mm)
            across_mm = (
                WALL_CLEARANCE_MM + depth_mm / 2
                if wall in ("top", "left")
                else across_avail_mm - WALL_CLEARANCE_MM - depth_mm / 2
            )
            local_x_mm, local_y_mm = (
                (along_mm, across_mm) if wall in ("top", "bottom") else (across_mm, along_mm)
            )
            half_x_mm, half_y_mm = (
                (width_mm / 2, depth_mm / 2)
                if wall in ("top", "bottom")
                else (depth_mm / 2, width_mm / 2)
            )
            footprint = frame.footprint_ratio(local_x_mm, local_y_mm, half_x_mm, half_y_mm)
            if any(_rects_overlap(footprint, blocker) for blocker in occupied):
                continue
            ratio_x, ratio_y = frame.to_ratio(local_x_mm, local_y_mm)
            return _Slot(ratio_x, ratio_y, _WALL_YAW_DEG[wall], footprint, wall)
    return None


_CENTER_SHIFT_FRACTIONS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (-0.15, 0.0),
    (0.15, 0.0),
    (0.0, -0.15),
    (0.0, 0.15),
)
"""悬空家具（茶几/餐桌）先试房间正中，四个方向各偏房间尺寸的 15% 再试——躲开门净空或
已摆家具时挪的幅度有限，不会为了躲一个障碍物就贴到墙角去（贴墙是贴墙家具的事）。"""


def _place_centered(
    frame: _RoomFrame, occupied: list[Rect], width_mm: float, depth_mm: float
) -> _Slot | None:
    half_x_mm, half_y_mm = width_mm / 2, depth_mm / 2
    if (
        width_mm + 2 * WALL_CLEARANCE_MM > frame.width_mm
        or depth_mm + 2 * WALL_CLEARANCE_MM > frame.depth_mm
    ):
        return None
    for shift_x_frac, shift_y_frac in _CENTER_SHIFT_FRACTIONS:
        local_x_mm = frame.width_mm / 2 + shift_x_frac * frame.width_mm
        local_y_mm = frame.depth_mm / 2 + shift_y_frac * frame.depth_mm
        in_bounds = (
            WALL_CLEARANCE_MM + half_x_mm
            <= local_x_mm
            <= frame.width_mm - WALL_CLEARANCE_MM - half_x_mm
            and WALL_CLEARANCE_MM + half_y_mm
            <= local_y_mm
            <= frame.depth_mm - WALL_CLEARANCE_MM - half_y_mm
        )
        if not in_bounds:
            continue
        footprint = frame.footprint_ratio(local_x_mm, local_y_mm, half_x_mm, half_y_mm)
        if any(_rects_overlap(footprint, blocker) for blocker in occupied):
            continue
        ratio_x, ratio_y = frame.to_ratio(local_x_mm, local_y_mm)
        return _Slot(ratio_x, ratio_y, 0.0, footprint, "center")
    return None


def _deprioritize(wall: str) -> tuple[str, ...]:
    """把 `wall` 挪到候选序列末尾——不是不让选，是"这面墙已经放了一件，换一面更像样"。"""
    return (*(w for w in _DEFAULT_WALL_ORDER if w != wall), wall)


def _placement(
    room_name: str,
    category: str,
    slot: _Slot,
    width_mm: int,
    depth_mm: int,
    height_mm: int,
    *,
    id_suffix: str = "",
) -> FurnishingPlacement:
    """摆一件家具。**尺寸收 `int`**：家具尺寸是整数毫米的数据（同 estate-svc 资产表），
    浮点毫米只可能是"米忘了换算"或一个没有来源的半毫米精度，让类型把它挡在外面。"""
    suffix = f":{id_suffix}" if id_suffix else ""
    return FurnishingPlacement(
        id=f"mock-furn:{room_name}:{category}{suffix}",
        category=category,
        room=room_name,
        center_x_ratio=slot.ratio_x,
        center_y_ratio=slot.ratio_y,
        width_mm=width_mm,
        depth_mm=depth_mm,
        height_mm=height_mm,
        yaw_deg=slot.yaw_deg,
    )


# ---------------------------------------------------------------------------
# 六、按房间品类摆家具
# ---------------------------------------------------------------------------

type _RoomRecipe = Callable[
    [str, _RoomFrame, list[Rect]], tuple[list[FurnishingPlacement], list[str]]
]
"""一个品类的摆法：房间名 + 摆场 + 门净空清单 → (摆出来的家具, 摆不下而跳过的记号)。"""


def _bedroom(
    room_name: str, frame: _RoomFrame, doors: list[Rect]
) -> tuple[list[FurnishingPlacement], list[str]]:
    placements: list[FurnishingPlacement] = []
    skipped: list[str] = []
    occupied = list(doors)

    bed_slot = _place_against_wall(frame, occupied, _DEFAULT_WALL_ORDER, BED_WIDTH_MM, BED_DEPTH_MM)
    if bed_slot is None:
        skipped.append(f"{room_name}:bed")
    else:
        placements.append(
            _placement(room_name, "bed", bed_slot, BED_WIDTH_MM, BED_DEPTH_MM, BED_HEIGHT_MM)
        )
        occupied.append(frame.expand(bed_slot.footprint, ITEM_GAP_MM))
        # 床头柜必须贴床所在那面墙——它答的是"床头两侧"这个位置关系，摆到别的墙就不是床头柜了
        nightstand_slot = _place_against_wall(
            frame, occupied, (bed_slot.wall,), NIGHTSTAND_WIDTH_MM, NIGHTSTAND_DEPTH_MM
        )
        if nightstand_slot is None:
            skipped.append(f"{room_name}:nightstand")
        else:
            placements.append(
                _placement(
                    room_name,
                    "nightstand",
                    nightstand_slot,
                    NIGHTSTAND_WIDTH_MM,
                    NIGHTSTAND_DEPTH_MM,
                    NIGHTSTAND_HEIGHT_MM,
                )
            )
            occupied.append(frame.expand(nightstand_slot.footprint, ITEM_GAP_MM))

    wardrobe_walls = _deprioritize(bed_slot.wall) if bed_slot is not None else _DEFAULT_WALL_ORDER
    wardrobe_slot = _place_against_wall(
        frame, occupied, wardrobe_walls, WARDROBE_WIDTH_MM, WARDROBE_DEPTH_MM
    )
    if wardrobe_slot is None:
        skipped.append(f"{room_name}:wardrobe")
    else:
        placements.append(
            _placement(
                room_name,
                "wardrobe",
                wardrobe_slot,
                WARDROBE_WIDTH_MM,
                WARDROBE_DEPTH_MM,
                WARDROBE_HEIGHT_MM,
            )
        )
    return placements, skipped


_OPPOSITE_WALL: dict[str, str] = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}


def _living_room(
    room_name: str, frame: _RoomFrame, doors: list[Rect]
) -> tuple[list[FurnishingPlacement], list[str]]:
    placements: list[FurnishingPlacement] = []
    skipped: list[str] = []
    occupied = list(doors)

    sofa_slot = _place_against_wall(
        frame, occupied, _DEFAULT_WALL_ORDER, SOFA_WIDTH_MM, SOFA_DEPTH_MM
    )
    if sofa_slot is None:
        skipped.append(f"{room_name}:sofa")
        tv_walls = _DEFAULT_WALL_ORDER
    else:
        placements.append(
            _placement(room_name, "sofa", sofa_slot, SOFA_WIDTH_MM, SOFA_DEPTH_MM, SOFA_HEIGHT_MM)
        )
        occupied.append(frame.expand(sofa_slot.footprint, ITEM_GAP_MM))
        # 电视柜要摆在沙发对面墙——沙发才"面向电视墙"；对面墙摆不下再退回随便一面墙
        opposite = _OPPOSITE_WALL[sofa_slot.wall]
        tv_walls = (
            opposite,
            *(w for w in _DEFAULT_WALL_ORDER if w not in (opposite, sofa_slot.wall)),
        )

    tv_slot = _place_against_wall(
        frame, occupied, tv_walls, TV_CABINET_WIDTH_MM, TV_CABINET_DEPTH_MM
    )
    if tv_slot is None:
        skipped.append(f"{room_name}:tv-cabinet")
    else:
        placements.append(
            _placement(
                room_name,
                "tv-cabinet",
                tv_slot,
                TV_CABINET_WIDTH_MM,
                TV_CABINET_DEPTH_MM,
                TV_CABINET_HEIGHT_MM,
            )
        )
        occupied.append(frame.expand(tv_slot.footprint, ITEM_GAP_MM))

    coffee_slot = _place_centered(frame, occupied, COFFEE_TABLE_WIDTH_MM, COFFEE_TABLE_DEPTH_MM)
    if coffee_slot is None:
        skipped.append(f"{room_name}:coffee-table")
    else:
        placements.append(
            _placement(
                room_name,
                "coffee-table",
                coffee_slot,
                COFFEE_TABLE_WIDTH_MM,
                COFFEE_TABLE_DEPTH_MM,
                COFFEE_TABLE_HEIGHT_MM,
            )
        )
    return placements, skipped


_CHAIR_SIDES: tuple[str, ...] = ("top", "bottom", "left", "right")
"""相对餐桌的四个方位，复用 `_WALL_YAW_DEG` 同一张朝向表——椅子"背对桌外、面朝桌"
跟家具"背贴墙、面朝房间"是同一种几何关系，只是这里的"墙"换成了桌子的边。"""


def _chair_beside_table(
    frame: _RoomFrame, occupied: list[Rect], table_x_mm: float, table_y_mm: float, side: str
) -> _Slot | None:
    half_table_w_mm, half_table_d_mm = DINING_TABLE_WIDTH_MM / 2, DINING_TABLE_DEPTH_MM / 2
    offset_mm = ITEM_GAP_MM + DINING_CHAIR_DEPTH_MM / 2
    if side == "top":
        local_x_mm, local_y_mm = table_x_mm, table_y_mm - half_table_d_mm - offset_mm
    elif side == "bottom":
        local_x_mm, local_y_mm = table_x_mm, table_y_mm + half_table_d_mm + offset_mm
    elif side == "left":
        local_x_mm, local_y_mm = table_x_mm - half_table_w_mm - offset_mm, table_y_mm
    else:
        local_x_mm, local_y_mm = table_x_mm + half_table_w_mm + offset_mm, table_y_mm

    half_x_mm, half_y_mm = (
        (DINING_CHAIR_WIDTH_MM / 2, DINING_CHAIR_DEPTH_MM / 2)
        if side in ("top", "bottom")
        else (DINING_CHAIR_DEPTH_MM / 2, DINING_CHAIR_WIDTH_MM / 2)
    )
    in_bounds = (
        half_x_mm <= local_x_mm <= frame.width_mm - half_x_mm
        and half_y_mm <= local_y_mm <= frame.depth_mm - half_y_mm
    )
    if not in_bounds:
        return None
    footprint = frame.footprint_ratio(local_x_mm, local_y_mm, half_x_mm, half_y_mm)
    if any(_rects_overlap(footprint, blocker) for blocker in occupied):
        return None
    ratio_x, ratio_y = frame.to_ratio(local_x_mm, local_y_mm)
    return _Slot(ratio_x, ratio_y, _WALL_YAW_DEG[side], footprint, side)


def _dining_room(
    room_name: str, frame: _RoomFrame, doors: list[Rect]
) -> tuple[list[FurnishingPlacement], list[str]]:
    placements: list[FurnishingPlacement] = []
    skipped: list[str] = []
    occupied = list(doors)

    table_slot = _place_centered(frame, occupied, DINING_TABLE_WIDTH_MM, DINING_TABLE_DEPTH_MM)
    if table_slot is None:
        skipped.append(f"{room_name}:dining-table")
        skipped.extend(f"{room_name}:dining-chair:{side}" for side in _CHAIR_SIDES)
        return placements, skipped

    placements.append(
        _placement(
            room_name,
            "dining-table",
            table_slot,
            DINING_TABLE_WIDTH_MM,
            DINING_TABLE_DEPTH_MM,
            DINING_TABLE_HEIGHT_MM,
        )
    )
    occupied.append(frame.expand(table_slot.footprint, ITEM_GAP_MM))
    table_x_mm, table_y_mm = frame.to_local(table_slot.ratio_x, table_slot.ratio_y)

    for side in _CHAIR_SIDES:
        chair_slot = _chair_beside_table(frame, occupied, table_x_mm, table_y_mm, side)
        if chair_slot is None:
            skipped.append(f"{room_name}:dining-chair:{side}")
            continue
        placements.append(
            _placement(
                room_name,
                "dining-chair",
                chair_slot,
                DINING_CHAIR_WIDTH_MM,
                DINING_CHAIR_DEPTH_MM,
                DINING_CHAIR_HEIGHT_MM,
                id_suffix=side,
            )
        )
        occupied.append(frame.expand(chair_slot.footprint, ITEM_GAP_MM))
    return placements, skipped


def _kitchen(
    room_name: str, frame: _RoomFrame, doors: list[Rect]
) -> tuple[list[FurnishingPlacement], list[str]]:
    slot = _place_against_wall(
        frame, list(doors), _DEFAULT_WALL_ORDER, KITCHEN_CABINET_WIDTH_MM, KITCHEN_CABINET_DEPTH_MM
    )
    if slot is None:
        return [], [f"{room_name}:kitchen-cabinet"]
    placement = _placement(
        room_name,
        "kitchen-cabinet",
        slot,
        KITCHEN_CABINET_WIDTH_MM,
        KITCHEN_CABINET_DEPTH_MM,
        KITCHEN_CABINET_HEIGHT_MM,
    )
    return [placement], []


def _bathroom(
    room_name: str, frame: _RoomFrame, doors: list[Rect]
) -> tuple[list[FurnishingPlacement], list[str]]:
    slot = _place_against_wall(
        frame,
        list(doors),
        _DEFAULT_WALL_ORDER,
        BATHROOM_FIXTURE_WIDTH_MM,
        BATHROOM_FIXTURE_DEPTH_MM,
    )
    if slot is None:
        return [], [f"{room_name}:bathroom-fixture"]
    placement = _placement(
        room_name,
        "bathroom-fixture",
        slot,
        BATHROOM_FIXTURE_WIDTH_MM,
        BATHROOM_FIXTURE_DEPTH_MM,
        BATHROOM_FIXTURE_HEIGHT_MM,
    )
    return [placement], []


def _study(
    room_name: str, frame: _RoomFrame, doors: list[Rect]
) -> tuple[list[FurnishingPlacement], list[str]]:
    placements: list[FurnishingPlacement] = []
    skipped: list[str] = []
    occupied = list(doors)

    desk_slot = _place_against_wall(
        frame, occupied, _DEFAULT_WALL_ORDER, DESK_WIDTH_MM, DESK_DEPTH_MM
    )
    if desk_slot is None:
        skipped.append(f"{room_name}:desk")
        bookshelf_walls = _DEFAULT_WALL_ORDER
    else:
        placements.append(
            _placement(room_name, "desk", desk_slot, DESK_WIDTH_MM, DESK_DEPTH_MM, DESK_HEIGHT_MM)
        )
        occupied.append(frame.expand(desk_slot.footprint, ITEM_GAP_MM))
        bookshelf_walls = _deprioritize(desk_slot.wall)

    bookshelf_slot = _place_against_wall(
        frame, occupied, bookshelf_walls, BOOKSHELF_WIDTH_MM, BOOKSHELF_DEPTH_MM
    )
    if bookshelf_slot is None:
        skipped.append(f"{room_name}:bookshelf")
    else:
        placements.append(
            _placement(
                room_name,
                "bookshelf",
                bookshelf_slot,
                BOOKSHELF_WIDTH_MM,
                BOOKSHELF_DEPTH_MM,
                BOOKSHELF_HEIGHT_MM,
            )
        )
    return placements, skipped


def _balcony(
    room_name: str, frame: _RoomFrame, doors: list[Rect]
) -> tuple[list[FurnishingPlacement], list[str]]:
    slot = _place_against_wall(
        frame, list(doors), _DEFAULT_WALL_ORDER, WASHER_WIDTH_MM, WASHER_DEPTH_MM
    )
    if slot is None:
        return [], [f"{room_name}:washer"]
    placement = _placement(
        room_name, "washer", slot, WASHER_WIDTH_MM, WASHER_DEPTH_MM, WASHER_HEIGHT_MM
    )
    return [placement], []


_RECIPES: dict[str, _RoomRecipe] = {
    "bedroom": _bedroom,
    "living_room": _living_room,
    "dining_room": _dining_room,
    "kitchen": _kitchen,
    "bathroom": _bathroom,
    "study": _study,
    "balcony": _balcony,
}


# ---------------------------------------------------------------------------
# 七、入口：一份 DesignPackage → 一份摆场报告
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockFurnishingReport:
    """这份 mock 摆场做了什么、没做什么——自证数，CLI 直接打出来。

    `skipped_items` 的记号是 `房间名:品类` 或 `房间名:品类:方位`，不是给人读的句子——
    真出现异常（一片房间一件都摆不出）时，看这份清单能定位到具体哪件、哪间房。
    """

    placements: list[FurnishingPlacement]
    unrecognized_rooms: list[str]
    skipped_items: list[str]


def build_mock_furnishings(package: DesignPackage) -> MockFurnishingReport:
    """纯函数：同一份 `package` 永远算出同一份报告。**不改 `package` 本身**，
    调用方（`cli.py`）自己决定要不要拿 `report.placements` 去替换 `package.furnishings`。
    """
    plan = package.plan
    mm_per_x = mesh.mm_per_unit(plan, package.scale)
    m_per_y = mm_per_x * plan.frame_height_px / plan.frame_width_px

    placements: list[FurnishingPlacement] = []
    unrecognized_rooms: list[str] = []
    skipped_items: list[str] = []

    for room in plan.rooms:
        category = _ROOM_CATEGORY.get(room.name)
        if category is None:
            unrecognized_rooms.append(room.name)
            continue
        rect = _largest_usable_rect(room.boxes)
        if rect is None:
            skipped_items.append(f"{room.name}:(没有 boxes，摆不了)")
            continue
        frame = _RoomFrame(rect=rect, mm_per_x=mm_per_x, m_per_y=m_per_y)
        doors = _door_clear_rects(plan, package.heights, room.name, mm_per_x, m_per_y)
        room_placements, room_skipped = _RECIPES[category](room.name, frame, doors)
        placements.extend(room_placements)
        skipped_items.extend(room_skipped)

    return MockFurnishingReport(
        placements=placements, unrecognized_rooms=unrecognized_rooms, skipped_items=skipped_items
    )


# ---------------------------------------------------------------------------
# 没做到什么（如实记在这儿，不是留在读者猜）
# ---------------------------------------------------------------------------
#
# - 只用房间 boxes 里能拼出的最大一块矩形当摆场：L 形、T 形房间会有一角量出来的空间
#   没被用上，家具数量因此可能比真实软装少。
# - 每个品类的家具件数与尺寸是固定的（一张床、一个衣柜……），不随房间实际面积放大缩小。
# - 床头柜只贴床所在那面墙的某个位置，不保证紧挨床头（`_OFFSET_FRACTIONS` 是固定候选表，
#   不是"贴着已摆家具找空隙"的搜索）。
# - 门净空按"两侧都留"处理，不判断门实际朝哪边开、开合会不会扫到家具。
# - 厨房与卫生间只给一段贴墙体块，不区分橱柜分区、不区分马桶/洗手盆/淋浴——那要看
#   给排水点位，这层几何没有这个信息。
