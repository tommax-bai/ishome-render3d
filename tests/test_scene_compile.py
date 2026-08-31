"""编场景包的守门测试：确定性、尺子、长宽比、开洞。

**几何在测试里现搭，不吃 `tests/fixtures/`**：这四条量的是编译这一步本身，卷子换了
它们还得成立；拟真包是给"整户跑得通吗"那一类测试用的，两拨互不牵连。

断言写的是"外面看得见的事实"（面积落在哪个区间、包围盒长宽比是多少、墙块数变多了），
不是把实现的算式再抄一遍——抄一遍的断言只会跟着 bug 一起改。
"""

from __future__ import annotations

import logging
import math

import pytest

from render3d_worker import mesh
from render3d_worker.models import (
    CameraSpec,
    DesignPackage,
    FloorplanGeometry,
    FurnishingPlacement,
    HeightRules,
    MaterialAssignment,
    Mesh,
    PlanOpening,
    PlanScale,
    PlanWall,
    RoomOutline,
    SurfaceMaterial,
)
from render3d_worker.scene_compile import SceneCompileError, compile_scene_package

BUILDING_AREA_SQM = 100.0
USABLE_AREA_PERCENT = 80.0
USABLE_AREA_SQM = 80.0
"""三个数摆在一起，是为了让"100㎡ 的房子按 80% 得房率就是 80㎡ 套内"这句话在测试里
写得出来——尺子对不对，量的就是这一句。"""

_OUTER_LEFT, _OUTER_RIGHT = 0.01, 0.99
_OUTER_TOP, _OUTER_BOTTOM = 0.015, 0.985
_OUTER_V_THICKNESS_RATIO, _OUTER_H_THICKNESS_RATIO = 0.02, 0.03
_SPINE_X, _SPLIT_Y = 0.5, 0.5
_PARTITION_THICKNESS_RATIO = 0.015


def _two_bedroom_plan(frame_width_px: int = 1200, frame_height_px: int = 800) -> FloorplanGeometry:
    """一个方正的两室户型：中间一道竖墙分东西，东边再横着分成两间卧室。

    `plan_box` 取满整幅，好让"包围盒的长宽比应该等于 frame 的长宽比"这句话是句实话，
    不必再绕一层图幅裁剪。
    """
    return FloorplanGeometry(
        frame_width_px=frame_width_px,
        frame_height_px=frame_height_px,
        plan_box=(0.0, 0.0, 1.0, 1.0),
        outline=[
            PlanWall(
                axis="vertical",
                position_ratio=_OUTER_LEFT,
                start_ratio=0.0,
                end_ratio=1.0,
                thickness_ratio=_OUTER_V_THICKNESS_RATIO,
            ),
            PlanWall(
                axis="vertical",
                position_ratio=_OUTER_RIGHT,
                start_ratio=0.0,
                end_ratio=1.0,
                thickness_ratio=_OUTER_V_THICKNESS_RATIO,
            ),
            PlanWall(
                axis="horizontal",
                position_ratio=_OUTER_TOP,
                start_ratio=0.0,
                end_ratio=1.0,
                thickness_ratio=_OUTER_H_THICKNESS_RATIO,
            ),
            PlanWall(
                axis="horizontal",
                position_ratio=_OUTER_BOTTOM,
                start_ratio=0.0,
                end_ratio=1.0,
                thickness_ratio=_OUTER_H_THICKNESS_RATIO,
            ),
        ],
        walls=[
            PlanWall(
                axis="vertical",
                position_ratio=_SPINE_X,
                start_ratio=0.03,
                end_ratio=0.97,
                thickness_ratio=_PARTITION_THICKNESS_RATIO,
            ),
            PlanWall(
                axis="horizontal",
                position_ratio=_SPLIT_Y,
                start_ratio=_SPINE_X,
                end_ratio=0.98,
                thickness_ratio=_PARTITION_THICKNESS_RATIO,
            ),
        ],
        openings=[
            PlanOpening(
                axis="vertical",
                position_ratio=_SPINE_X,
                start_ratio=0.35,
                end_ratio=0.45,
                is_on_outer_wall=False,
                connects=["客厅", "主卧"],
            ),
            PlanOpening(
                axis="vertical",
                position_ratio=_SPINE_X,
                start_ratio=0.60,
                end_ratio=0.70,
                is_on_outer_wall=False,
                connects=["客厅", "次卧"],
            ),
            PlanOpening(
                axis="vertical",
                position_ratio=_OUTER_LEFT,
                start_ratio=0.30,
                end_ratio=0.60,
                is_on_outer_wall=True,
                connects=["客厅"],
            ),
        ],
        rooms=[
            RoomOutline(
                name="客厅",
                boxes=[(0.02, 0.03, 0.4925, 0.97)],
                area_ratio=0.5,
                centroid=(0.256, 0.5),
            ),
            RoomOutline(
                name="主卧",
                boxes=[(0.5075, 0.03, 0.98, 0.4925)],
                area_ratio=0.25,
                centroid=(0.744, 0.261),
            ),
            RoomOutline(
                name="次卧",
                boxes=[(0.5075, 0.5075, 0.98, 0.97)],
                area_ratio=0.25,
                centroid=(0.744, 0.739),
            ),
        ],
        cell_coverage_ratio=0.97,
    )


def _two_bedroom_package(frame_width_px: int = 1200, frame_height_px: int = 800) -> DesignPackage:
    """整包：几何 + 尺子 + 默认高度 + 两件家具 + 一小张材质表 + 一台俯瞰相机。

    材质故意留一个缺口（床没有指派），好让"找不到材质要如实反映"那条能被看见。
    """
    return DesignPackage(
        revision_id="rev:two-bedroom",
        plan=_two_bedroom_plan(frame_width_px, frame_height_px),
        scale=PlanScale(
            building_area_sqm=BUILDING_AREA_SQM, usable_area_percent=USABLE_AREA_PERCENT
        ),
        heights=HeightRules(),
        furnishings=[
            FurnishingPlacement(
                id="furnishing:客厅:沙发",
                category="sofa",
                room="客厅",
                center_x_ratio=0.20,
                center_y_ratio=0.40,
                width_m=2.2,
                depth_m=0.9,
                height_m=0.8,
            ),
            FurnishingPlacement(
                id="furnishing:主卧:床",
                category="bed",
                room="主卧",
                center_x_ratio=0.75,
                center_y_ratio=0.25,
                width_m=1.8,
                depth_m=2.0,
                height_m=0.5,
                yaw_deg=90.0,
            ),
        ],
        materials=[
            MaterialAssignment(surface="floor", material_id="material:oak"),
            MaterialAssignment(surface="wall", material_id="material:paint"),
            MaterialAssignment(
                surface="furnishing", material_id="material:fabric", category="sofa"
            ),
        ],
        surface_materials=[
            SurfaceMaterial(id="material:oak", base_color_hex="#C8A97E"),
            SurfaceMaterial(id="material:paint", base_color_hex="#F2F2F0"),
            SurfaceMaterial(id="material:fabric", base_color_hex="#6E7B8B"),
        ],
        cameras=[CameraSpec(id="camera:bird", kind="bird")],
    )


def _single_wall_package(*, with_door: bool) -> DesignPackage:
    """一间房、一道竖墙，门洞可有可无。开洞那条测试要的就是这么干净的对照。"""
    door = [
        PlanOpening(
            axis="vertical",
            position_ratio=0.5,
            start_ratio=0.40,
            end_ratio=0.60,
            is_on_outer_wall=False,
            connects=["东屋", "西屋"],
        )
    ]
    return DesignPackage(
        revision_id="rev:single-wall",
        plan=FloorplanGeometry(
            frame_width_px=1000,
            frame_height_px=1000,
            plan_box=(0.0, 0.0, 1.0, 1.0),
            walls=[
                PlanWall(
                    axis="vertical",
                    position_ratio=0.5,
                    start_ratio=0.0,
                    end_ratio=1.0,
                    thickness_ratio=0.02,
                )
            ],
            openings=door if with_door else [],
            rooms=[
                RoomOutline(
                    name="西屋", boxes=[(0.0, 0.0, 0.49, 1.0)], area_ratio=0.5, centroid=(0.25, 0.5)
                ),
                RoomOutline(
                    name="东屋", boxes=[(0.51, 0.0, 1.0, 1.0)], area_ratio=0.5, centroid=(0.75, 0.5)
                ),
            ],
        ),
        scale=PlanScale(
            building_area_sqm=BUILDING_AREA_SQM, usable_area_percent=USABLE_AREA_PERCENT
        ),
    )


# ---------------------------------------------------------------------------
# ① 确定性：同一份输入编两次，逐字相同
# ---------------------------------------------------------------------------


def test_compiling_twice_gives_byte_identical_json() -> None:
    """确定性是这一层的红线，不是优点：失效传播、缓存、跨天比对全靠它。

    两次分别现搭一个包（不复用同一个对象），量的才是"输入相同"而不是"对象相同"。
    """
    first = compile_scene_package(_two_bedroom_package())
    second = compile_scene_package(_two_bedroom_package())
    assert first.model_dump_json() == second.model_dump_json()


# ---------------------------------------------------------------------------
# ② 尺子：100㎡ 建筑面积 + 80% 得房率
# ---------------------------------------------------------------------------


def test_floor_area_lands_near_the_usable_area() -> None:
    """编出来的地板面积应当略小于套内面积，差的那点是墙占的地。

    区间怎么来的：房间遮罩不含墙体占地，所以吻合率必然小于 1；这份几何实测 **0.9270**
    （尺子锚到外轮廓那一版；锚在 `plan_box` 外接框上时是 0.8812，外接框大一圈、尺子偏小）。
    落在 0.80 以下说明尺子把面积算成了另一个数，落到 0.97 以上说明墙被算进了地板。
    **这个区间只对这份现搭的几何有效**——真户型的门槛要有真跑数据才定
    （《纪律·阈值有数据才定》），本仓今天不设死阈值。
    """
    scene = compile_scene_package(_two_bedroom_package())
    assert 0.80 < scene.area_match_ratio < 0.97
    # 吻合率的分母确实是"建筑面积 × 得房率"，不是建筑面积本身
    assert scene.floor_area_sqm / scene.area_match_ratio == pytest.approx(USABLE_AREA_SQM, rel=1e-3)
    assert scene.floor_area_sqm < USABLE_AREA_SQM


# ---------------------------------------------------------------------------
# ③ 长宽比：x 与 y 的归一化分母不同，不修正就会把长方形拉成正方形
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("frame_width_px", "frame_height_px"), [(1200, 800), (800, 1200)])
def test_bounds_keep_the_frame_aspect(frame_width_px: int, frame_height_px: int) -> None:
    """包围盒的长宽比要等于 frame 的长宽比（`plan_box` 取满整幅时两者就是同一件事）。

    这是一条回归：x 按图宽归一、y 按图高归一，换算时不拿 `frame_*_px` 修正，一张长方形
    户型会被编成正方形，而面积吻合率照样是对的——**面积对不出这个错，只有形状能**。
    横竖两种 frame 都量一遍，免得实现把长宽比写反了还能蒙对一边。
    """
    scene = compile_scene_package(_two_bedroom_package(frame_width_px, frame_height_px))
    span_x_m = scene.bounds_max_m[0] - scene.bounds_min_m[0]
    span_y_m = scene.bounds_max_m[1] - scene.bounds_min_m[1]
    assert span_x_m / span_y_m == pytest.approx(frame_width_px / frame_height_px, rel=1e-4)
    # 形状对了，面积也得对：外轮廓中心线围的那块地就是套内建筑面积
    enclosed_sqm = (_OUTER_RIGHT - _OUTER_LEFT) * (_OUTER_BOTTOM - _OUTER_TOP) * span_x_m * span_y_m
    assert enclosed_sqm == pytest.approx(USABLE_AREA_SQM, rel=1e-4)
    # 包围盒框的是整幅（外接框），它必然比外轮廓围的那块大——尺子锚的是后者
    assert span_x_m * span_y_m > USABLE_AREA_SQM
    assert scene.bounds_max_m[2] == pytest.approx(HeightRules().ceiling_height_m)


# ---------------------------------------------------------------------------
# ④ 开洞：墙被切开，洞的高度就是门高
# ---------------------------------------------------------------------------


def test_a_door_splits_the_wall_and_keeps_its_height() -> None:
    """带门洞的墙块数要比不带的多（切成了洞左/洞右/过梁），洞壁高度等于门高。

    墙块数是开洞唯一在包上看得见的痕迹——不数它，一个"洞挖了但没挖穿"的实现能一路
    过关到出图。
    """
    heights = HeightRules()
    plain = compile_scene_package(_single_wall_package(with_door=False))
    holed = compile_scene_package(_single_wall_package(with_door=True))

    assert plain.wall_segment_count == 1
    assert holed.wall_segment_count > plain.wall_segment_count
    assert holed.opening_count_by_kind == {"door": 1}

    reveals = [block for block in holed.meshes if block.semantic == "reveal"]
    assert len(reveals) == 1
    z_values = [vertex[2] for vertex in reveals[0].vertices]
    assert min(z_values) == pytest.approx(0.0)
    assert max(z_values) - min(z_values) == pytest.approx(heights.door_height_m)
    assert not [block for block in plain.meshes if block.semantic == "reveal"]


# ---------------------------------------------------------------------------
# 编不出来的时候要炸，不能返回一个空包
# ---------------------------------------------------------------------------


def test_a_plan_without_rooms_fails_loudly() -> None:
    """一间房都没有＝没有地板也没有吊顶。空包往下游走，底渲会渲出一张说不清为什么空的图。"""
    package = _two_bedroom_package()
    package.plan.rooms = []
    with pytest.raises(SceneCompileError):
        compile_scene_package(package)


def test_a_zero_building_area_fails_loudly() -> None:
    """尺子由面积反推，没有面积就没有米——不在这儿替它编一个比例尺。"""
    package = _two_bedroom_package()
    package.scale = PlanScale(building_area_sqm=0.0, usable_area_percent=USABLE_AREA_PERCENT)
    with pytest.raises(SceneCompileError):
        compile_scene_package(package)


# ---------------------------------------------------------------------------
# 材质：指派得上的挂上，指派不上的落到中性材质并如实反映
# ---------------------------------------------------------------------------


def test_unassigned_surfaces_fall_back_to_a_declared_neutral_material() -> None:
    """没人指派的面（这份包里是吊顶和那张床）用中性材质，而中性材质进材质表。

    落到中性不是错，**落到中性而材质表里查不到它**才是错——那样"这块灰是谁"就没人答得出。
    """
    scene = compile_scene_package(_two_bedroom_package())
    declared = {material.id for material in scene.materials}
    assert {block.material_id for block in scene.meshes} <= declared

    by_id = {block.id: block.material_id for block in scene.meshes}
    assert by_id["furnishing:客厅:沙发"] == "material:fabric"
    ceiling_ids = {block.material_id for block in scene.meshes if block.semantic == "ceiling"}
    assert ceiling_ids == {by_id["furnishing:主卧:床"]}
    assert "material:oak" not in ceiling_ids


def test_yaw_turns_the_furnishing_block() -> None:
    """转 90 度之后，宽和进深在世界坐标里换个位置——体块是有朝向的，不是个球。"""
    scene = compile_scene_package(_two_bedroom_package())
    bed = next(block for block in scene.meshes if block.id == "furnishing:主卧:床")
    span_x_m = max(v[0] for v in bed.vertices) - min(v[0] for v in bed.vertices)
    span_y_m = max(v[1] for v in bed.vertices) - min(v[1] for v in bed.vertices)
    assert span_x_m == pytest.approx(2.0)  # 进深转到了 x 上
    assert span_y_m == pytest.approx(1.8)  # 宽转到了 y 上


def test_metre_per_unit_is_the_frame_width_in_metres() -> None:
    """`metre_per_unit` 的口径：归一化 1.0（整幅图宽）等于多少米。

    这份几何取满整幅，所以它就该等于包围盒的 x 跨度。
    """
    scene = compile_scene_package(_two_bedroom_package())
    span_x_m = scene.bounds_max_m[0] - scene.bounds_min_m[0]
    assert scene.metre_per_unit == pytest.approx(span_x_m, rel=1e-5)


# ---------------------------------------------------------------------------
# 尺子的锚：外轮廓围的那块地，不是外接框
# ---------------------------------------------------------------------------


def _l_shaped_outline_plan(*, with_outline: bool) -> FloorplanGeometry:
    """一个 L 形外轮廓：占满外接框的 **3/4**（右下角那 1/4 缺着）。

    3/4 这个数是从形状上一眼数得出来的，不是从实现里抄的——尺子锚外轮廓还是锚外接框，
    差的就是这个 3/4。
    """
    outline = [
        PlanWall(
            axis="horizontal",
            position_ratio=0.0,
            start_ratio=0.0,
            end_ratio=1.0,
            thickness_ratio=0.01,
        ),
        PlanWall(
            axis="vertical",
            position_ratio=1.0,
            start_ratio=0.0,
            end_ratio=0.5,
            thickness_ratio=0.01,
        ),
        PlanWall(
            axis="horizontal",
            position_ratio=0.5,
            start_ratio=0.5,
            end_ratio=1.0,
            thickness_ratio=0.01,
        ),
        PlanWall(
            axis="vertical",
            position_ratio=0.5,
            start_ratio=0.5,
            end_ratio=1.0,
            thickness_ratio=0.01,
        ),
        PlanWall(
            axis="horizontal",
            position_ratio=1.0,
            start_ratio=0.0,
            end_ratio=0.5,
            thickness_ratio=0.01,
        ),
        PlanWall(
            axis="vertical",
            position_ratio=0.0,
            start_ratio=0.0,
            end_ratio=1.0,
            thickness_ratio=0.01,
        ),
    ]
    return FloorplanGeometry(
        frame_width_px=1000,
        frame_height_px=1000,
        plan_box=(0.0, 0.0, 1.0, 1.0),
        outline=outline if with_outline else [],
    )


def test_the_ruler_anchors_on_the_outline_not_the_bounding_box() -> None:
    """尺子由**外轮廓围的那块地**反推，不是由 `plan_box` 外接框。

    外接框是外接的：L 形户型的外接框比它自己大三分之一，锚在外接框上尺子就偏小，
    出图里家具会显得挤（真跑那份包上就是这么发现的）。外轮廓中心线围的面积正是
    套内建筑面积的几何定义，所以这条不引入任何系数。
    """
    plan = _l_shaped_outline_plan(with_outline=True)
    scale = PlanScale(building_area_sqm=BUILDING_AREA_SQM, usable_area_percent=USABLE_AREA_PERCENT)
    anchor = mesh.scale_anchor(plan)
    assert anchor.source == "outline"
    assert anchor.area_x_units_sq == pytest.approx(0.75)
    assert mesh.metre_per_unit(plan, scale) == pytest.approx(math.sqrt(USABLE_AREA_SQM / 0.75))
    # 锚错了会得到这个数（外接框面积 1.0）——两者差 15%，分得开
    assert mesh.metre_per_unit(plan, scale) != pytest.approx(math.sqrt(USABLE_AREA_SQM))


def test_the_ruler_falls_back_to_the_plan_box_and_says_so() -> None:
    """没有外轮廓（8-31 之前的老产物）时退回外接框，而且退了这件事说得出来。

    退回本身不是错——错的是退了没人知道：外接框比套内建筑面积大，编出来的一切都会小
    一号，而场景包里没有一个数会露馅（`area_match_ratio` 的分子分母一起缩，比值不变）。
    """
    plan = _l_shaped_outline_plan(with_outline=False)
    scale = PlanScale(building_area_sqm=BUILDING_AREA_SQM, usable_area_percent=USABLE_AREA_PERCENT)
    anchor = mesh.scale_anchor(plan)
    assert anchor.source == "plan-box"
    assert anchor.area_x_units_sq == pytest.approx(1.0)
    assert mesh.metre_per_unit(plan, scale) == pytest.approx(math.sqrt(USABLE_AREA_SQM))


def test_falling_back_to_the_plan_box_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """退回外接框要在日志里看得见——它是这条退路今天唯一的出口。"""
    package = _single_wall_package(with_door=False)  # 这份包没有 outline
    with caplog.at_level(logging.WARNING, logger="render3d_worker.scene_compile"):
        compile_scene_package(package)
    assert any("plan_box" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# 洞的第二种形态：墙在洞处本来就断开，洞落在两段墙之间的空隙里
# ---------------------------------------------------------------------------


def _broken_wall_package(*, on_outer_wall: bool) -> DesignPackage:
    """一道在洞处断开的墙：两段墙夹着一个洞，谁也不与洞相交。

    真数据长这样——2026-08-30 那份存档 15 个洞里有 10 个如此，缝恰好一个源像素宽。
    这儿把缝放大到 0.001（1000px 的图上就是 1px），形态一样、算得清。
    """
    return DesignPackage(
        revision_id="rev:broken-wall",
        plan=FloorplanGeometry(
            frame_width_px=1000,
            frame_height_px=1000,
            plan_box=(0.0, 0.0, 1.0, 1.0),
            walls=[
                PlanWall(
                    axis="vertical",
                    position_ratio=0.5,
                    start_ratio=0.0,
                    end_ratio=0.399,
                    thickness_ratio=0.02,
                ),
                PlanWall(
                    axis="vertical",
                    position_ratio=0.5,
                    start_ratio=0.601,
                    end_ratio=1.0,
                    thickness_ratio=0.02,
                ),
            ],
            openings=[
                PlanOpening(
                    axis="vertical",
                    position_ratio=0.5,
                    start_ratio=0.40,
                    end_ratio=0.60,
                    is_on_outer_wall=on_outer_wall,
                    connects=["西屋", "东屋"],
                )
            ],
            rooms=[
                RoomOutline(
                    name="西屋", boxes=[(0.0, 0.0, 0.49, 1.0)], area_ratio=0.5, centroid=(0.25, 0.5)
                ),
                RoomOutline(
                    name="东屋", boxes=[(0.51, 0.0, 1.0, 1.0)], area_ratio=0.5, centroid=(0.75, 0.5)
                ),
            ],
        ),
        scale=PlanScale(
            building_area_sqm=BUILDING_AREA_SQM, usable_area_percent=USABLE_AREA_PERCENT
        ),
    )


def test_an_opening_in_a_wall_gap_still_gets_a_lintel() -> None:
    """洞落在两段墙的空隙里时，切不出东西，但**过梁得补上**。

    不补的话门就是一个从地面通到天花的大开口——真跑第一版出的图正是这样。
    补出来的块与两段邻墙**共面**（厚度、中心位置同源），洞口两侧不能有台阶。
    """
    heights = HeightRules()
    scene = compile_scene_package(_broken_wall_package(on_outer_wall=False))

    assert scene.opening_count_by_kind == {"door": 1}
    # 洞没被切进任何一段墙，所以不出洞壁——洞壁由两段邻墙的端面充当
    assert not [block for block in scene.meshes if block.semantic == "reveal"]

    walls = [block for block in scene.meshes if block.semantic == "wall"]
    fills = [block for block in walls if block.id.startswith("wall:fill:")]
    assert len(walls) == 3  # 两段原墙 + 一块过梁
    assert len(fills) == 1

    lintel = fills[0]
    z_values = [vertex[2] for vertex in lintel.vertices]
    assert min(z_values) == pytest.approx(heights.door_height_m)
    assert max(z_values) == pytest.approx(heights.ceiling_height_m)

    # 共面：过梁跨墙方向占的那一条带，与邻墙逐字相同
    def across_span_m(block: Mesh) -> tuple[float, float]:
        xs = [vertex[0] for vertex in block.vertices]
        return (min(xs), max(xs))

    neighbours = [block for block in walls if block not in fills]
    assert {across_span_m(block) for block in neighbours} == {across_span_m(lintel)}


def test_an_opening_with_no_wall_on_its_line_is_not_counted() -> None:
    """洞落在压根没有墙的地方时，做不出东西，**洞数上就少一个**。

    选"计数上少掉"而不是另出一个自证数：判读只要一句减法（洞数对不上输入里的洞数），
    而加字段要动 `models.py` 那份两侧共用的契约，得两边一起改才算数。
    """
    package = _broken_wall_package(on_outer_wall=False)
    package.plan.openings[0].position_ratio = 0.9  # 这条线上一段墙也没有
    scene = compile_scene_package(package)
    assert scene.opening_count_by_kind == {}
    assert not [block for block in scene.meshes if block.id.startswith("wall:fill:")]


def test_a_window_in_a_wall_gap_gets_both_a_lintel_and_a_sill() -> None:
    """外墙上的洞按窗算，窗除了过梁还欠一块窗下墙——不补的话窗户会一直落到地面。"""
    heights = HeightRules()
    scene = compile_scene_package(_broken_wall_package(on_outer_wall=True))

    assert scene.opening_count_by_kind == {"window": 1}
    fills = [block for block in scene.meshes if block.id.startswith("wall:fill:")]
    assert len(fills) == 2

    spans = sorted(
        (min(v[2] for v in block.vertices), max(v[2] for v in block.vertices)) for block in fills
    )
    assert spans[0] == pytest.approx((0.0, heights.window_sill_height_m))
    assert spans[1] == pytest.approx((heights.window_head_height_m, heights.ceiling_height_m))
