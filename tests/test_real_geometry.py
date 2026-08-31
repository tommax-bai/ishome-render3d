"""真跑那份包编出来是什么样——把关键数字钉死。

与 `test_scene_compile.py` 分工：那边现搭最小几何，量的是**编译这一步的规则**（尺子、
长宽比、切段、补洞）；这边吃 `tests/fixtures/design-package-full.json` 那份 92㎡ 三室的
拟真包，量的是**整户跑下来还成不成立**。规则对而整户垮掉、或整户能跑而某条规则悄悄变了，
是两种不同的坏法，各由一边照出来。

**这些数字是快照，不是门槛。** 它们变了不一定是错，但一定要有人说得出为什么变——
所以每个数后面都写了它是怎么来的。真出现改判，改这儿的数并在版本行留对照。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from render3d_worker import mesh
from render3d_worker.models import DesignPackage
from render3d_worker.scene_compile import compile_scene_package

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "design-package-full.json"

EXPECTED_ROOM_COUNT = 9
EXPECTED_OPENING_COUNT = 15
EXPECTED_DEGENERATE_WALL_COUNT = 5
"""上游给的零长墙线：61 段网格墙里的第 32/36/47/49/55 段（2026-08-30 存档）。

2D 那侧画一条零长的线等于没画，所以这件事只有三维会撞上。这个数变了就是产出侧
那一步变了——可能修好了，也可能坏得更多，都得有人去看。"""


def _load_package() -> DesignPackage:
    """每次现读现解析，不做模块级缓存：确定性那条要的是"同一份输入"，不是"同一个对象"。"""
    return DesignPackage.model_validate(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))


@pytest.fixture
def package() -> DesignPackage:
    return _load_package()


def test_the_real_package_still_compiles(package: DesignPackage) -> None:
    """整户编得出来，房间一间不少。"""
    scene = compile_scene_package(package)
    assert len(package.plan.rooms) == EXPECTED_ROOM_COUNT
    floors = [block for block in scene.meshes if block.semantic == "floor"]
    assert {block.room for block in floors} == {room.name for room in package.plan.rooms}
    assert scene.triangle_count > 0
    assert scene.unit == "m"


def test_compiling_the_real_package_twice_is_byte_identical() -> None:
    """真几何上确定性也得成立——最小几何跑得过、真几何跑不过的实现是有的
    （排序不稳、拿 set 决定次序），那种坑只有段数上百时才露头。"""
    first = compile_scene_package(_load_package())
    second = compile_scene_package(_load_package())
    assert first.model_dump_json() == second.model_dump_json()


def test_every_opening_lands_somewhere(package: DesignPackage) -> None:
    """15 个洞一个不少地做出来：外墙 9 个按窗算、内墙 6 个按门算。

    **这条是那次返工的回归。** 头一版只数出 5 个 window——真数据里墙在洞处本来就是断开的，
    洞落在两段墙之间的空隙里，与任何一段墙都不相交（沿墙方向差恰好一个源像素），
    于是十个洞既没被切、也没被补，门窗成了从地面通到天花的大开口。

    "洞数 == 输入里的洞数"就是判读方式：对不上就是有洞落在了没有墙的地方。
    """
    scene = compile_scene_package(package)
    assert len(package.plan.openings) == EXPECTED_OPENING_COUNT
    assert scene.opening_count_by_kind == {"door": 6, "window": 9}
    assert sum(scene.opening_count_by_kind.values()) == len(package.plan.openings)


def test_openings_in_wall_gaps_get_their_lintels(package: DesignPackage) -> None:
    """落在空隙里的那十个洞补出了过梁与窗下墙，且每一块都补在正常的高度上。

    六个门只欠过梁（门落地，没有窗下墙），四个窗欠过梁 + 窗下墙 —— 6 + 4×2 = 14 块。
    """
    scene = compile_scene_package(package)
    fills = [block for block in scene.meshes if block.id.startswith("wall:fill:")]
    assert len(fills) == 14
    ceiling_m = package.heights.ceiling_height_m
    for block in fills:
        z_values = [vertex[2] for vertex in block.vertices]
        assert 0.0 <= min(z_values) < max(z_values) <= ceiling_m


def test_the_ruler_anchors_on_the_real_outline(package: DesignPackage) -> None:
    """真包的尺子锚在外轮廓上，不是外接框。

    这份包的外轮廓是四条边的中心线，比 `plan_box` 外接框小 4.1%（中心线内缩半个墙厚），
    所以尺子比锚外接框时大 2.0%。等产出侧给出带飘窗台阶那种真外轮廓，这个差会变大，
    而口径不用再改一次——今天换的是锚点，不是系数。
    """
    plan = package.plan
    anchor = mesh.scale_anchor(plan)
    assert anchor.source == "outline"

    left, top, right, bottom = plan.plan_box
    aspect = plan.frame_height_px / plan.frame_width_px
    box_area_x_units_sq = (right - left) * (bottom - top) * aspect
    assert anchor.area_x_units_sq < box_area_x_units_sq
    assert anchor.area_x_units_sq / box_area_x_units_sq == pytest.approx(0.9604, abs=5e-4)


def test_the_real_scene_is_the_size_of_a_92_sqm_flat(package: DesignPackage) -> None:
    """整户的规模对得上一套 92㎡ 的房子：套内 73.6㎡，屋子九米见方上下，层高 2.8 米。

    这几条不是精确值而是量级——它们防的是"尺子整体错了一个数量级"那种坏法
    （单位当成厘米、长宽比没修正、锚点选错），那种错一眼看得出来，也只有这种数量级的
    断言拦得住，写死到小数点后四位反而会被无关的改动天天绊倒。
    """
    scene = compile_scene_package(package)
    assert mesh.usable_area_sqm(package.scale) == pytest.approx(73.6)

    span_x_m = scene.bounds_max_m[0] - scene.bounds_min_m[0]
    span_y_m = scene.bounds_max_m[1] - scene.bounds_min_m[1]
    assert 7.0 < span_x_m < 9.0
    assert 9.0 < span_y_m < 11.0
    # 长宽比对得上图幅，但不会分毫不差：墙有厚度，最外圈那几段会探出图幅一点点
    # （这份包上最多探出一个源像素，约 1.5 厘米），所以留 1% 的余量而不是写死
    box = package.plan.plan_box
    box_aspect = (
        (box[2] - box[0])
        * package.plan.frame_width_px
        / ((box[3] - box[1]) * package.plan.frame_height_px)
    )
    assert span_x_m / span_y_m == pytest.approx(box_aspect, rel=0.01)
    assert scene.bounds_max_m[2] == pytest.approx(package.heights.ceiling_height_m)


def test_the_real_floor_area_is_short_of_the_usable_area(package: DesignPackage) -> None:
    """房间遮罩加起来 54.3㎡，套内 73.6㎡，吻合率 0.737。

    **这个数今天就是不到 1，而且不该被凑。** 差的那 26% 里既有墙占的地，也有房间遮罩
    本来就没盖满的部分（几何那侧的自证数 `cell_coverage_ratio` 说它只盖住九成七）。
    `area_match_ratio` 是照出问题的那个数，不是要被调成 1 的数——**不设死阈值**
    （《纪律·阈值有数据才定》），这儿只钉住它别悄悄漂走。
    """
    scene = compile_scene_package(package)
    assert scene.floor_area_sqm == pytest.approx(54.26, abs=0.5)
    assert scene.area_match_ratio == pytest.approx(0.737, abs=0.01)


def test_the_real_geometry_keeps_its_degenerate_walls_visible(package: DesignPackage) -> None:
    """上游给的五段零长墙线被跳过了，而且跳了几段说得出来。"""
    scene = compile_scene_package(package)
    assert scene.degenerate_wall_count == EXPECTED_DEGENERATE_WALL_COUNT
    assert mesh.degenerate_wall_ids(package.plan, package.scale) == [
        "grid:32",
        "grid:36",
        "grid:47",
        "grid:49",
        "grid:55",
    ]


def test_every_real_mesh_has_a_declared_material(package: DesignPackage) -> None:
    """每块网格挂的材质都能在场景包的材质表里查到——查不到的灰是说不清来路的灰。"""
    scene = compile_scene_package(package)
    declared = {material.id for material in scene.materials}
    assert {block.material_id for block in scene.meshes} <= declared
    assert {block.semantic for block in scene.meshes} == {
        "floor",
        "ceiling",
        "wall",
        "reveal",
        "furnishing",
    }
