"""mock 摆场的守门测试：确定性、按房间名分派、不穿墙、不堵门。

**几何在这儿现搭，不吃 `tests/fixtures/`**——同 `test_scene_compile.py` 的做法：
量的是摆放这一步本身，卷子换了它还得成立。房间统一用边长 3m 的正方形（`_package` /
`_single_room_package` 把 `metre_per_unit` 钉成 7.5，好让 ratio 与米之间能手算校验）。
"""

from __future__ import annotations

from render3d_worker.furnish_mock import (
    DOOR_CLEARANCE_M,
    MockFurnishingReport,
    Rect,
    build_mock_furnishings,
)
from render3d_worker.models import (
    DesignPackage,
    FloorplanGeometry,
    FurnishingPlacement,
    HeightRules,
    PlanOpening,
    PlanScale,
    RoomOutline,
)

# 3m×3m 正方形房间：frame 1000×1000、`plan_box` 取满整幅（得房率 100%）时，
# `metre_per_unit = sqrt(building_area_sqm)`；56.25㎡ 对应 7.5 米/单位，
# 0.4 的 ratio 边长换算正好是 3 米——这几个数字是刻意凑的，方便测试里手算校验。
_BUILDING_AREA_SQM = 56.25
_METRE_PER_UNIT = 7.5
_ROOM_BOX: tuple[float, float, float, float] = (0.1, 0.1, 0.5, 0.5)


def _package(rooms: list[RoomOutline], openings: list[PlanOpening] | None = None) -> DesignPackage:
    return DesignPackage(
        revision_id="rev:furnish-mock-test",
        plan=FloorplanGeometry(
            frame_width_px=1000,
            frame_height_px=1000,
            plan_box=(0.0, 0.0, 1.0, 1.0),
            openings=openings or [],
            rooms=rooms,
        ),
        scale=PlanScale(building_area_sqm=_BUILDING_AREA_SQM, usable_area_percent=100.0),
    )


def _single_room_package(
    room_name: str,
    boxes: list[tuple[float, float, float, float]],
    openings: list[PlanOpening] | None = None,
) -> DesignPackage:
    return _package([RoomOutline(name=room_name, boxes=boxes)], openings)


# ---------------------------------------------------------------------------
# ① 确定性：同一份输入摆两次，逐字相同
# ---------------------------------------------------------------------------


def test_building_twice_from_scratch_gives_the_same_placements() -> None:
    """确定性是这一层的红线：不看 `random`、不看字典遍历序、不看时间。

    两次分别现搭一个包（不复用同一个对象），量的才是"输入相同"而不是"对象相同"。
    """
    rooms = [
        RoomOutline(name="主卧", boxes=[_ROOM_BOX]),
        RoomOutline(name="客厅", boxes=[(0.5, 0.5, 0.9, 0.9)]),
        RoomOutline(name="餐厅", boxes=[(0.1, 0.5, 0.4, 0.8)]),
    ]
    first = build_mock_furnishings(_package(list(rooms)))
    second = build_mock_furnishings(_package(list(rooms)))

    assert [p.model_dump_json() for p in first.placements] == [
        p.model_dump_json() for p in second.placements
    ]
    assert first.unrecognized_rooms == second.unrecognized_rooms
    assert first.skipped_items == second.skipped_items
    # 至少真摆出了东西——确定性不是靠"两次都什么都没摆"混过去的
    assert first.placements


# ---------------------------------------------------------------------------
# ② 按房间名分派：认得的给对应家具，认不得的一件不摆、如实报出来
# ---------------------------------------------------------------------------


def test_dispatch_by_room_name_and_reports_unrecognized_rooms() -> None:
    package = _package(
        [
            RoomOutline(name="主卧", boxes=[_ROOM_BOX]),
            RoomOutline(name="客厅", boxes=[(0.5, 0.1, 0.9, 0.5)]),
            RoomOutline(name="阳光房", boxes=[(0.1, 0.5, 0.5, 0.9)]),  # 不在白名单里
        ]
    )
    report = build_mock_furnishings(package)

    assert report.unrecognized_rooms == ["阳光房"]
    assert not any(p.room == "阳光房" for p in report.placements)

    bedroom_categories = {p.category for p in report.placements if p.room == "主卧"}
    living_room_categories = {p.category for p in report.placements if p.room == "客厅"}
    assert bedroom_categories <= {"bed", "nightstand", "wardrobe"}
    assert "bed" in bedroom_categories  # 3m×3m 的房间，双人床应该摆得下
    assert living_room_categories <= {"sofa", "tv-cabinet", "coffee-table"}
    assert "sofa" in living_room_categories


def test_a_room_with_no_boxes_is_skipped_not_guessed() -> None:
    """房间名认得，但没有几何（`boxes` 是空的）——摆不出任何东西，不许编一个位置。"""
    package = _package([RoomOutline(name="主卧", boxes=[])])
    report = build_mock_furnishings(package)
    assert not report.placements
    assert not report.unrecognized_rooms  # 名字认得，问题不在"认不认识"
    assert report.skipped_items


# ---------------------------------------------------------------------------
# ③ 不许穿墙：家具足迹必须落在房间 boxes 并集内
# ---------------------------------------------------------------------------


def _point_in_any_box(x_ratio: float, y_ratio: float, boxes: list[Rect]) -> bool:
    return any(
        left <= x_ratio <= right and top <= y_ratio <= bottom for left, top, right, bottom in boxes
    )


def _footprint_ratio(placement: FurnishingPlacement, m_per_x: float, m_per_y: float) -> Rect:
    """从 `FurnishingPlacement` 独立反推足迹——不借 `furnish_mock` 内部的 `_Slot`，
    量的是契约本身（中心点 + 米制尺寸 + `yaw_deg`），跟 `mesh.build_furnishings`
    的旋转口径对齐：0°/180° 时宽沿 x、深沿 y，90°/270° 两者对调。
    """
    if placement.yaw_deg in (0.0, 180.0):
        half_x_m, half_y_m = placement.width_m / 2, placement.depth_m / 2
    else:
        half_x_m, half_y_m = placement.depth_m / 2, placement.width_m / 2
    return (
        placement.center_x_ratio - half_x_m / m_per_x,
        placement.center_y_ratio - half_y_m / m_per_y,
        placement.center_x_ratio + half_x_m / m_per_x,
        placement.center_y_ratio + half_y_m / m_per_y,
    )


def test_furnishings_never_leave_the_rooms_box_union() -> None:
    """主卧被网格切成 2×2 共 4 块（同真实几何产出的样子：房间常被切碎，但拼起来是矩形）。

    摆出来的每一件家具，足迹上采样一片格点，每个点都要落在这 4 块**原始** box 之一里——
    不是落在摆放算法自己拼出来的那块大矩形里（那样测的是算法信自己，不是信 boxes 本身）。
    """
    boxes = [
        (0.1, 0.1, 0.3, 0.3),
        (0.3, 0.1, 0.5, 0.3),
        (0.1, 0.3, 0.3, 0.5),
        (0.3, 0.3, 0.5, 0.5),
    ]
    package = _single_room_package("主卧", boxes)
    report = build_mock_furnishings(package)
    assert report.placements  # 3m×3m 卧室，床/床头柜/衣柜至少摆得出一部分

    for placement in report.placements:
        left, top, right, bottom = _footprint_ratio(placement, _METRE_PER_UNIT, _METRE_PER_UNIT)
        samples_x = [left + (right - left) * frac for frac in (0.02, 0.3, 0.5, 0.7, 0.98)]
        samples_y = [top + (bottom - top) * frac for frac in (0.02, 0.3, 0.5, 0.7, 0.98)]
        for x_ratio in samples_x:
            for y_ratio in samples_y:
                assert _point_in_any_box(x_ratio, y_ratio, boxes), (
                    f"{placement.id} 的足迹有一点 ({x_ratio:.4f},{y_ratio:.4f}) "
                    "落在了房间 boxes 并集之外"
                )


def test_an_l_shaped_room_never_gets_furniture_in_its_missing_corner() -> None:
    """L 形房间：右下角那一块根本不属于这间房，家具不能探到那儿去。"""
    top_bar = (0.1, 0.1, 0.5, 0.3)  # 宽条
    left_bar = (0.1, 0.3, 0.3, 0.5)  # 窄条，右下角 (0.3,0.3)-(0.5,0.5) 缺着
    package = _single_room_package("客厅", [top_bar, left_bar])
    report = build_mock_furnishings(package)
    assert report.placements

    for placement in report.placements:
        left, top, right, bottom = _footprint_ratio(placement, _METRE_PER_UNIT, _METRE_PER_UNIT)
        missing_corner = (0.3, 0.3, 0.5, 0.5)
        overlaps_missing_corner = (
            left < missing_corner[2]
            and missing_corner[0] < right
            and (top < missing_corner[3] and missing_corner[1] < bottom)
        )
        assert not overlaps_missing_corner, f"{placement.id} 的足迹伸进了这间房缺角的那一块"


# ---------------------------------------------------------------------------
# ④ 不许堵门：门口留的净空里不能有家具
# ---------------------------------------------------------------------------


def test_furnishings_avoid_the_door_clearance_zone() -> None:
    """房间顶墙开一扇几乎通宽的门，沙发理应躲开——门净空里不能有任何家具足迹。"""
    door = PlanOpening(
        axis="horizontal",
        position_ratio=0.1,  # 房间顶边
        start_ratio=0.15,
        end_ratio=0.45,
        is_on_outer_wall=False,
        connects=["客厅"],
    )
    package = _single_room_package("客厅", [_ROOM_BOX], openings=[door])
    report = build_mock_furnishings(package)
    assert report.placements  # 至少摆出点东西，不然这条测试测的是"什么都没摆"

    clear_y_ratio = DOOR_CLEARANCE_M / _METRE_PER_UNIT
    door_clear: Rect = (
        door.start_ratio,
        door.position_ratio - clear_y_ratio,
        door.end_ratio,
        door.position_ratio + clear_y_ratio,
    )

    for placement in report.placements:
        footprint = _footprint_ratio(placement, _METRE_PER_UNIT, _METRE_PER_UNIT)
        overlaps = (
            footprint[0] < door_clear[2]
            and door_clear[0] < footprint[2]
            and footprint[1] < door_clear[3]
            and door_clear[1] < footprint[3]
        )
        assert not overlaps, f"{placement.id} 的足迹压进了门净空"


def test_windows_do_not_need_clearance_only_doors_and_passes_do() -> None:
    """外墙洞默认按窗算（`HeightRules.outer_opening_kind`），窗前摆家具是常规做法——
    同一扇洞若判成窗，不该逼着算法绕开它。"""
    window = PlanOpening(
        axis="horizontal",
        position_ratio=0.1,
        start_ratio=0.15,
        end_ratio=0.45,
        is_on_outer_wall=True,  # 外墙洞，`HeightRules` 默认按窗算
        connects=["客厅"],
    )
    heights = HeightRules()
    assert heights.outer_opening_kind == "window"

    package = _single_room_package("客厅", [_ROOM_BOX], openings=[window])
    report = build_mock_furnishings(package)
    # 沙发不用为了躲窗户去挑别的墙——跟完全没有洞时选的是同一面墙、同一个偏移
    baseline = build_mock_furnishings(_single_room_package("客厅", [_ROOM_BOX]))
    sofa = next(p for p in report.placements if p.category == "sofa")
    baseline_sofa = next(p for p in baseline.placements if p.category == "sofa")
    assert (sofa.center_x_ratio, sofa.center_y_ratio, sofa.yaw_deg) == (
        baseline_sofa.center_x_ratio,
        baseline_sofa.center_y_ratio,
        baseline_sofa.yaw_deg,
    )


# ---------------------------------------------------------------------------
# 其余：一致性小项
# ---------------------------------------------------------------------------


def test_report_is_a_plain_dataclass_not_mutating_the_input() -> None:
    """`build_mock_furnishings` 是纯函数：不改 `package` 本身，`furnishings` 摆前摆后一样空。"""
    package = _single_room_package("主卧", [_ROOM_BOX])
    assert package.furnishings == []
    report = build_mock_furnishings(package)
    assert isinstance(report, MockFurnishingReport)
    assert package.furnishings == []  # 没被就地改掉
