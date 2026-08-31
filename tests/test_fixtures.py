"""拟真上游包的守门测试：契约收得下、规模与 README 对得上。

这两份 fixture 是开发期唯一的 `DesignPackage` 来源（上游今天不存在，见
`tests/fixtures/README.md`）。契约 `extra="forbid"`——多一个字段就炸在这里，
好过带着一个没人读的字段一路走到出图。

规模数字在这儿写死是有意的：README 里逐条写着这两份包多大，
改了一边不改另一边，测试就红。数字改动＝换了拟真包，那是要说清楚的事。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from render3d_worker import mesh
from render3d_worker.models import DesignPackage, FurnishingPlacement, RoomOutline

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# --- README 里写着的规模，逐条钉死 ---------------------------------------

# 几何两份包逐字相同：真跑存档 geometry-1.json（9 房间 / 61 段墙 / 15 个洞）
# 加派生补齐的 4 段外轮廓。洞按外墙 9、内墙 6 分。
EXPECTED_ROOM_COUNT = 9
EXPECTED_WALL_COUNT = 61
EXPECTED_OUTLINE_COUNT = 4
EXPECTED_OPENING_COUNT = 15
EXPECTED_OUTER_OPENING_COUNT = 9
EXPECTED_INNER_OPENING_COUNT = 6
EXPECTED_CELL_COVERAGE_RATIO = 0.9734
EXPECTED_FRAME_WIDTH_PX = 1080
EXPECTED_FRAME_HEIGHT_PX = 1466

EXPECTED_FULL_FURNISHING_COUNT = 23
EXPECTED_FULL_MATERIAL_COUNT = 24
EXPECTED_FULL_SURFACE_MATERIAL_COUNT = 7
EXPECTED_FULL_CAMERA_COUNT = 8
EXPECTED_FULL_ROOM_CAMERA_COUNT = 7

# 有 room 机位的七间，与没有的两间。**两间没有是有来由的**，README 里逐间写着为什么：
# 主卧的地板质心被双人床盖住（四面墙逐一试过都盖），小孩房近乎正方、长轴两头真渲都不过关。
EXPECTED_ROOMS_WITH_CAMERA = {"客厅", "次卧", "餐厅", "厨房", "卫生间", "玄关", "阳台"}
EXPECTED_ROOMS_WITHOUT_CAMERA = {"主卧", "小孩房"}

FIXTURE_NAMES = ["design-package-full.json", "design-package-minimal.json"]


def load_raw(name: str) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    return payload


def load_package(name: str) -> DesignPackage:
    return DesignPackage.model_validate(load_raw(name))


@pytest.fixture(scope="module")
def full_package() -> DesignPackage:
    return load_package("design-package-full.json")


@pytest.fixture(scope="module")
def minimal_package() -> DesignPackage:
    return load_package("design-package-minimal.json")


# --- 一、契约收不收得下 ---------------------------------------------------


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_validates_against_contract(name: str) -> None:
    """两份包都要被 `DesignPackage` 原样收下。"""
    assert load_package(name).revision_id


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_uses_camel_case_on_the_wire(name: str) -> None:
    """序列化口径是 camelCase（`alias_generator=to_camel`）——包里不许出现 snake_case 键。"""
    raw = load_raw(name)
    assert "revisionId" in raw
    assert "revision_id" not in raw
    assert "frameWidthPx" in raw["plan"]
    assert "buildingAreaSqm" in raw["scale"]


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_extra_field_is_refused(name: str) -> None:
    """`extra=forbid` 要真的挡住多出来的字段——多一个字段说明两侧对不上头。"""
    raw = load_raw(name)
    raw["upstreamHint"] = "上游多塞了一个字段"
    with pytest.raises(ValueError):
        DesignPackage.model_validate(raw)


# --- 二、几何规模（两份包逐字相同） ---------------------------------------


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_geometry_scale_matches_readme(name: str) -> None:
    plan = load_package(name).plan
    assert plan.frame_width_px == EXPECTED_FRAME_WIDTH_PX
    assert plan.frame_height_px == EXPECTED_FRAME_HEIGHT_PX
    assert len(plan.rooms) == EXPECTED_ROOM_COUNT
    assert len(plan.walls) == EXPECTED_WALL_COUNT
    assert len(plan.outline) == EXPECTED_OUTLINE_COUNT
    assert len(plan.openings) == EXPECTED_OPENING_COUNT
    assert plan.cell_coverage_ratio == EXPECTED_CELL_COVERAGE_RATIO


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_opening_split_between_outer_and_inner_walls(name: str) -> None:
    """外墙洞与内墙洞都要有——只有一种的话，门窗那条确定性推法测不到。"""
    openings = load_package(name).plan.openings
    outer = [opening for opening in openings if opening.is_on_outer_wall]
    inner = [opening for opening in openings if not opening.is_on_outer_wall]
    assert len(outer) == EXPECTED_OUTER_OPENING_COUNT
    assert len(inner) == EXPECTED_INNER_OPENING_COUNT


def test_two_fixtures_share_the_same_geometry(
    full_package: DesignPackage, minimal_package: DesignPackage
) -> None:
    """两份包的差别只在"三维要而二维没有的那半"，几何与尺度必须逐字相同。"""
    assert full_package.plan == minimal_package.plan
    assert full_package.scale == minimal_package.scale


# --- 三、几何自洽：房间名对得上、内墙洞连得起来 ---------------------------


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_connects_names_are_real_rooms(name: str) -> None:
    plan = load_package(name).plan
    room_names = {room.name for room in plan.rooms}
    for opening in plan.openings:
        assert set(opening.connects) <= room_names, opening

    outer_connect_counts = {
        len(opening.connects) for opening in plan.openings if opening.is_on_outer_wall
    }
    assert outer_connect_counts == {1}, "外墙洞只有一侧是房间"

    two_sided = [
        opening
        for opening in plan.openings
        if not opening.is_on_outer_wall and len(opening.connects) == 2
    ]
    assert len(two_sided) == 5, "内墙洞里 5 个连起两间房（第 6 个另一侧是未被认领的像素）"


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_rooms_have_boxes_inside_the_plan_box(name: str) -> None:
    """房间遮罩不许跑到图幅外面——跑出去的那次在产出侧是响亮失败，这里也得挡住。"""
    plan = load_package(name).plan
    left, top, right, bottom = plan.plan_box
    for room in plan.rooms:
        assert room.boxes, room.name
        for box_left, box_top, box_right, box_bottom in room.boxes:
            assert left <= box_left < box_right <= right, room.name
            assert top <= box_top < box_bottom <= bottom, room.name


# --- 四、完整包：家具、材质、相机 -----------------------------------------


def test_full_package_scale_matches_readme(full_package: DesignPackage) -> None:
    assert len(full_package.furnishings) == EXPECTED_FULL_FURNISHING_COUNT
    assert len(full_package.materials) == EXPECTED_FULL_MATERIAL_COUNT
    assert len(full_package.surface_materials) == EXPECTED_FULL_SURFACE_MATERIAL_COUNT
    assert len(full_package.cameras) == EXPECTED_FULL_CAMERA_COUNT


def test_every_room_gets_two_to_four_furnishings(full_package: DesignPackage) -> None:
    per_room: dict[str, int] = {}
    for placement in full_package.furnishings:
        per_room[placement.room] = per_room.get(placement.room, 0) + 1
    assert set(per_room) == {room.name for room in full_package.plan.rooms}
    assert all(2 <= count <= 4 for count in per_room.values()), per_room


def test_furnishings_sit_inside_their_own_room(full_package: DesignPackage) -> None:
    """摆到房间外面的家具是错的布置，不是"差不多"。"""
    boxes = {room.name: room.boxes for room in full_package.plan.rooms}
    for placement in full_package.furnishings:
        assert any(
            left <= placement.center_x_ratio <= right and top <= placement.center_y_ratio <= bottom
            for left, top, right, bottom in boxes[placement.room]
        ), placement.id


def test_furnishing_dimensions_are_in_the_package_not_the_code(
    full_package: DesignPackage,
) -> None:
    """尺寸写在包里、单位是米：三条边都得是正数且在常见家具的量级内。"""
    for placement in full_package.furnishings:
        assert 0.2 <= placement.width_m <= 3.0, placement.id
        assert 0.2 <= placement.depth_m <= 3.0, placement.id
        assert 0.2 <= placement.height_m <= 2.6, placement.id


def test_every_furnishing_id_is_unique_and_semantic(full_package: DesignPackage) -> None:
    """《红线·命名禁纯序号》：id 里要带房间与类别，不许是 fur-1 这种。"""
    ids = [placement.id for placement in full_package.furnishings]
    assert len(set(ids)) == len(ids)
    for placement in full_package.furnishings:
        assert placement.room in placement.id
        assert placement.category in placement.id


def test_material_assignments_resolve(full_package: DesignPackage) -> None:
    """每条指派都要指到包里真有的材质；地/墙/顶各有一条全局兜底。"""
    known = {material.id for material in full_package.surface_materials}
    assert len(known) == len(full_package.surface_materials)
    for assignment in full_package.materials:
        assert assignment.material_id in known, assignment

    fallbacks = {
        assignment.surface
        for assignment in full_package.materials
        if assignment.room is None and assignment.category is None
    }
    assert fallbacks == {"floor", "wall", "ceiling"}

    scoped_rooms = {
        assignment.room for assignment in full_package.materials if assignment.room is not None
    }
    assert scoped_rooms <= {room.name for room in full_package.plan.rooms}


def test_every_furnishing_category_has_a_material(full_package: DesignPackage) -> None:
    assigned = {
        assignment.category
        for assignment in full_package.materials
        if assignment.surface == "furnishing"
    }
    used = {placement.category for placement in full_package.furnishings}
    assert used <= assigned


def _room_extent_ratio(room: RoomOutline) -> tuple[float, float]:
    """房间遮罩外接框的 x / y 跨度（归一化）。"""
    return (
        max(box[2] for box in room.boxes) - min(box[0] for box in room.boxes),
        max(box[3] for box in room.boxes) - min(box[1] for box in room.boxes),
    )


def _floor_anchor_ratio(room: RoomOutline) -> tuple[float, float]:
    """房间地板的面积加权质心（归一化）——底渲把 room 机位的眼点钉在这儿。"""
    total = weighted_x = weighted_y = 0.0
    for left, top, right, bottom in room.boxes:
        area = (right - left) * (bottom - top)
        total += area
        weighted_x += (left + right) / 2 * area
        weighted_y += (top + bottom) / 2 * area
    return weighted_x / total, weighted_y / total


def _footprint_ratio(
    placement: FurnishingPlacement, metre_per_unit: float, y_units_per_x_unit: float
) -> tuple[float, float, float, float]:
    """家具足迹（归一化）。`yawDeg` 只取 0/90/180/270，足迹恒为轴对齐长方形。"""
    if placement.yaw_deg in (0.0, 180.0):
        half_x_m, half_y_m = placement.width_m / 2, placement.depth_m / 2
    else:
        half_x_m, half_y_m = placement.depth_m / 2, placement.width_m / 2
    half_x = half_x_m / metre_per_unit
    half_y = half_y_m / (metre_per_unit * y_units_per_x_unit)
    return (
        placement.center_x_ratio - half_x,
        placement.center_y_ratio - half_y,
        placement.center_x_ratio + half_x,
        placement.center_y_ratio + half_y,
    )


def test_room_cameras_face_the_mask_long_axis(full_package: DesignPackage) -> None:
    """room 机位朝的必须是房间遮罩的长轴——这条是"能把进深拍进去"的全部依据。

    长短轴要在**米**里比，不能在归一化里比：x 按图宽归一、y 按图高归一，
    两个方向除的不是同一个数（这里只需要长宽比，不需要绝对尺子）。
    """
    aspect = full_package.plan.frame_height_px / full_package.plan.frame_width_px
    rooms = {room.name: room for room in full_package.plan.rooms}
    for camera in full_package.cameras:
        if camera.kind != "room":
            continue
        assert camera.room is not None
        x_ratio, y_ratio = _room_extent_ratio(rooms[camera.room])
        long_axis_is_x = x_ratio >= y_ratio * aspect
        expected = {90.0, 270.0} if long_axis_is_x else {0.0, 180.0}
        assert camera.yaw_deg in expected, (
            f"{camera.id} 朝的不是长轴：yaw={camera.yaw_deg}，该轴上应为 {sorted(expected)}"
        )


def test_room_cameras_are_level(full_package: DesignPackage) -> None:
    """room 机位是平视——底渲对 room 机位恒用 pitch 0，包里写别的值只会误导读包的人。"""
    for camera in full_package.cameras:
        if camera.kind == "room":
            assert camera.pitch_deg == 0.0, camera.id


def test_no_room_camera_stands_inside_a_furnishing(full_package: DesignPackage) -> None:
    """给了机位的房间，眼点不许落在任何家具体块里——站在柜子里往外看，出的图没法看。

    尺子走 `mesh.metre_per_unit` 这个唯一真源，不在这儿抄一份公式：
    它按外轮廓围出的面积锚定，抄一份就会在它改口径时悄悄对不上。
    """
    metre_per_unit = mesh.metre_per_unit(full_package.plan, full_package.scale)
    aspect = full_package.plan.frame_height_px / full_package.plan.frame_width_px
    rooms = {room.name: room for room in full_package.plan.rooms}
    for camera in full_package.cameras:
        if camera.kind != "room":
            continue
        assert camera.room is not None
        eye_x, eye_y = _floor_anchor_ratio(rooms[camera.room])
        for placement in full_package.furnishings:
            if placement.room != camera.room:
                continue
            left, top, right, bottom = _footprint_ratio(placement, metre_per_unit, aspect)
            assert not (left <= eye_x <= right and top <= eye_y <= bottom), (
                f"{camera.id} 的眼点落在 {placement.id} 的体块内"
            )


def test_rooms_without_a_camera_are_the_documented_ones(full_package: DesignPackage) -> None:
    """少给的机位要是"记过账的少"，不是"忘了给"。名单变了就得回去改 README。"""
    with_camera = {camera.room for camera in full_package.cameras if camera.kind == "room"}
    all_rooms = {room.name for room in full_package.plan.rooms}
    assert with_camera == EXPECTED_ROOMS_WITH_CAMERA
    assert all_rooms - with_camera == EXPECTED_ROOMS_WITHOUT_CAMERA


def test_furnishings_do_not_overlap_each_other(full_package: DesignPackage) -> None:
    """同一间房里两件家具不许叠在一起——叠出来的体块在底渲里是一团说不清的东西。"""
    metre_per_unit = mesh.metre_per_unit(full_package.plan, full_package.scale)
    aspect = full_package.plan.frame_height_px / full_package.plan.frame_width_px
    by_room: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = {}
    for placement in full_package.furnishings:
        by_room.setdefault(placement.room, []).append(
            (placement.id, _footprint_ratio(placement, metre_per_unit, aspect))
        )
    for room, items in by_room.items():
        for index, (left_id, a) in enumerate(items):
            for right_id, b in items[index + 1 :]:
                apart = a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
                assert apart, f"{room}：{left_id} 与 {right_id} 的体块重叠"


def test_cameras_cover_one_bird_and_the_main_rooms(full_package: DesignPackage) -> None:
    birds = [camera for camera in full_package.cameras if camera.kind == "bird"]
    rooms = [camera for camera in full_package.cameras if camera.kind == "room"]
    assert len(birds) == 1
    assert len(rooms) == EXPECTED_FULL_ROOM_CAMERA_COUNT
    assert birds[0].room is None, "bird 俯瞰整户，不属于哪间房"
    room_names = {room.name for room in full_package.plan.rooms}
    seen: set[str] = set()
    for camera in rooms:
        assert camera.room is not None
        assert camera.room in room_names
        assert camera.room not in seen, camera.id
        seen.add(camera.room)
    assert len({camera.id for camera in full_package.cameras}) == len(full_package.cameras)


# --- 五、最小包：上游只做出了平面时，这条线还出不出得来 -------------------


def test_minimal_package_carries_only_plan_scale_and_one_bird(
    minimal_package: DesignPackage,
) -> None:
    assert minimal_package.furnishings == []
    assert minimal_package.materials == []
    assert minimal_package.surface_materials == []
    assert len(minimal_package.cameras) == 1
    assert minimal_package.cameras[0].kind == "bird"


def test_minimal_package_omits_heights_and_falls_back_to_contract_defaults(
    minimal_package: DesignPackage,
) -> None:
    """`heights` 整段不写——竖向那一维今天没有上游，吃契约默认档位。"""
    assert "heights" not in load_raw("design-package-minimal.json")
    assert minimal_package.heights.ceiling_height_m == 2.80
    assert minimal_package.heights.outer_opening_kind == "window"
    assert minimal_package.heights.inner_opening_kind == "door"


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_scale_is_a_percent_not_a_fraction(name: str) -> None:
    """得房率取百分数不取小数（80 不是 0.8）——少一层换算就少一处会漂移的口径。"""
    scale = load_package(name).scale
    assert scale.building_area_sqm == 92.0
    assert scale.usable_area_percent == 80.0
