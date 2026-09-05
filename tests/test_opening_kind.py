"""洞口类型从上游读、读不到才猜——这条路的守门测试（2026-09-05，上游 aipipe c9fc00c 起给 `kind`）。

守四件事：

1. **契约对面**：`kind` 缺省是 `unknown` 不是 `door`；闭集与上游逐字一致，旧值 `pass` 不再认；
   档位猜法的两个字段不许填 `unknown` / `entry-door`。
2. **先读后猜、猜了要看得见**：上游给了 `kind` 的洞一律照上游起体（来源 `upstream`），只有
   `unknown` 才按外墙＝窗 / 内墙＝门猜（来源 `guessed`），猜了几个、哪几个随场景包带出。
3. **过口与入户门的起体**：`passage` 落地、不出窗下墙、洞高按过口档位；`entry-door` 按门起体、
   种类照写 `entry-door`。
4. **控制稿按最终种类画**：门斜线、窗十字、过口无符号；落在墙段空隙里（补出来的）的过口也按
   洞口表画成过口，不再被"没有窗下墙就是门"那条老退路画成门。

真户型那两份基准包（19 洞旧产物、16 洞 c9fc00c 产物）各走一遍：旧包没有 `kind` 仍读得进、
全部走猜法；新包 15 个上游给的照读、1 个 `unknown`（管井小门）走猜法。
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError
from test_control_sketch import HEIGHT_PX, WIDTH_PX, _open_gray, _project, _white_near

from render3d_worker import base_render, mesh
from render3d_worker.base_render import render_base_views
from render3d_worker.models import (
    CameraSpec,
    DesignPackage,
    FloorplanGeometry,
    HeightRules,
    PlanOpening,
    PlanScale,
    PlanWall,
    RoomOutline,
)
from render3d_worker.scene_compile import compile_scene_package

BASELINE_DIR = Path(__file__).parent.parent / "_iteration" / "真户型-基准"
REAL_16_PACKAGE = BASELINE_DIR / "design-package-真值138-16洞.json"
REAL_19_PACKAGE = BASELINE_DIR / "design-package-真值138.json"

ROOM_NORTH = "北屋"
ROOM_SOUTH = "南屋"
ROOM_CAMERA_ID = "room-南屋"
WALL_Y = 0.5
WALL_THICKNESS_RATIO = 0.02
DOOR_X = (0.30, 0.38)
WINDOW_X = (0.46, 0.54)
PASSAGE_X = (0.62, 0.74)
"""三个洞都开在 y=0.5 那道内墙上，挤在画幅中段：室内机位站在南屋地板质心平视北面，
80° 张角下能同时框进三个洞。"""


def _wall(axis: str, position: float, start: float, end: float) -> PlanWall:
    return PlanWall(
        axis=axis,  # type: ignore[arg-type]
        position_ratio=position,
        start_ratio=start,
        end_ratio=end,
        thickness_ratio=WALL_THICKNESS_RATIO,
    )


def _opening(x_span: tuple[float, float], **fields: object) -> PlanOpening:
    return PlanOpening.model_validate(
        {
            "axis": "horizontal",
            "positionRatio": WALL_Y,
            "startRatio": x_span[0],
            "endRatio": x_span[1],
            "isOnOuterWall": False,
            "connects": [ROOM_NORTH, ROOM_SOUTH],
            **fields,
        }
    )


def _three_opening_package(
    openings: list[PlanOpening], heights: HeightRules | None = None
) -> DesignPackage:
    """两间房夹一道横墙，墙压着洞（切出来的形态），洞由调用方给。"""
    plan = FloorplanGeometry(
        frame_width_px=1000,
        frame_height_px=1000,
        plan_box=(0.0, 0.0, 1.0, 1.0),
        outline=[
            _wall("vertical", 0.01, 0.0, 1.0),
            _wall("vertical", 0.99, 0.0, 1.0),
            _wall("horizontal", 0.01, 0.0, 1.0),
            _wall("horizontal", 0.99, 0.0, 1.0),
        ],
        walls=[_wall("horizontal", WALL_Y, 0.01, 0.99)],
        openings=openings,
        rooms=[
            RoomOutline(name=ROOM_NORTH, boxes=[(0.02, 0.02, 0.98, 0.49)], centroid=(0.5, 0.25)),
            RoomOutline(name=ROOM_SOUTH, boxes=[(0.02, 0.51, 0.98, 0.98)], centroid=(0.5, 0.75)),
        ],
    )
    fields: dict[str, object] = {
        "revision_id": "rev:opening-kind",
        "plan": plan,
        "scale": PlanScale(building_area_sqm=100.0, usable_area_percent=80.0),
        "cameras": [
            CameraSpec(
                id=ROOM_CAMERA_ID,
                kind="room",
                room=ROOM_SOUTH,
                eye_height_m=1.55,
                yaw_deg=180.0,
                pitch_deg=0.0,
                fov_deg=80.0,
            )
        ],
    }
    if heights is not None:
        fields["heights"] = heights
    return DesignPackage.model_validate(fields)


def _gap_package(kind: str) -> DesignPackage:
    """墙在洞处断开的形态（洞落在两段墙之间的空隙里，补过梁那条路），洞的种类由调用方给。"""
    return DesignPackage(
        revision_id="rev:gap-opening",
        plan=FloorplanGeometry(
            frame_width_px=1000,
            frame_height_px=1000,
            plan_box=(0.0, 0.0, 1.0, 1.0),
            walls=[
                _wall("vertical", 0.5, 0.0, 0.399),
                _wall("vertical", 0.5, 0.601, 1.0),
            ],
            openings=[
                PlanOpening(
                    axis="vertical",
                    position_ratio=0.5,
                    start_ratio=0.40,
                    end_ratio=0.60,
                    is_on_outer_wall=False,
                    connects=["西屋", "东屋"],
                    kind=kind,  # type: ignore[arg-type]
                )
            ],
            rooms=[
                RoomOutline(name="西屋", boxes=[(0.0, 0.0, 0.49, 1.0)], centroid=(0.25, 0.5)),
                RoomOutline(name="东屋", boxes=[(0.51, 0.0, 1.0, 1.0)], centroid=(0.75, 0.5)),
            ],
        ),
        scale=PlanScale(building_area_sqm=100.0, usable_area_percent=80.0),
    )


# ---------------------------------------------------------------------------
# 一、契约对面
# ---------------------------------------------------------------------------


def test_上游没给kind时缺省是unknown不是door() -> None:
    opening = PlanOpening.model_validate(
        {
            "axis": "vertical",
            "positionRatio": 0.5,
            "startRatio": 0.1,
            "endRatio": 0.2,
            "isOnOuterWall": False,
        }
    )
    assert opening.kind == "unknown"
    assert opening.kind_evidence == ""


def test_kind与kindEvidence按产物的camelCase读() -> None:
    opening = _opening(DOOR_X, kind="entry-door", kindEvidence="门弧 16/16，弧在户外")
    assert opening.kind == "entry-door"
    assert opening.kind_evidence == "门弧 16/16，弧在户外"
    assert opening.model_dump(by_alias=True)["kindEvidence"] == "门弧 16/16，弧在户外"


def test_旧值pass不再认() -> None:
    """闭集与上游逐字一致：`pass` 已改名 `passage`，老拼法读进来要炸而不是悄悄当成别的。"""
    with pytest.raises(ValidationError):
        _opening(DOOR_X, kind="pass")


def test_产物侧的自证数openingKindCoverageRatio读得进() -> None:
    raw = json.loads(REAL_16_PACKAGE.read_text(encoding="utf-8"))
    package = DesignPackage.model_validate(raw)
    assert package.plan.opening_kind_coverage_ratio == pytest.approx(0.9375)
    # 同一份产物里墙线带了按段实测的墙带（994e66c 起），逐字对面要收得下——今天只收不消费
    assert any(wall.bands for wall in package.plan.walls)


@pytest.mark.parametrize("field", ["outer_opening_kind", "inner_opening_kind"])
@pytest.mark.parametrize("value", ["unknown", "entry-door"])
def test_档位猜法不许猜成unknown或入户门(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        HeightRules.model_validate({field: value})


# ---------------------------------------------------------------------------
# 二、先读后猜，猜了要看得见
# ---------------------------------------------------------------------------


def test_上游给了kind就照上游_unknown才按档位猜() -> None:
    heights = HeightRules()
    for kind in ("door", "window", "passage", "entry-door"):
        resolved = mesh.resolve_opening_kind(_opening(DOOR_X, kind=kind), heights)
        assert (resolved.kind, resolved.source) == (kind, "upstream")
    inner = mesh.resolve_opening_kind(_opening(DOOR_X), heights)
    outer = mesh.resolve_opening_kind(_opening(DOOR_X, isOnOuterWall=True), heights)
    assert (inner.kind, inner.source) == ("door", "guessed")
    assert (outer.kind, outer.source) == ("window", "guessed")


def test_场景包带洞口表并数出按档位猜了几个() -> None:
    package = _three_opening_package(
        [
            _opening(DOOR_X, kind="door"),
            _opening(WINDOW_X),  # 上游没给：内墙，猜成门
            _opening(PASSAGE_X, kind="passage"),
        ]
    )
    scene = compile_scene_package(package)
    assert [(o.opening_index, o.kind, o.kind_source, o.placed) for o in scene.openings] == [
        (0, "door", "upstream", True),
        (1, "door", "guessed", True),
        (2, "passage", "upstream", True),
    ]
    assert scene.guessed_opening_count == 1
    assert scene.guessed_opening_indices == [1]
    assert scene.opening_count_by_kind == {"door": 2, "passage": 1}
    assert "unknown" not in {o.kind for o in scene.openings}


def test_落不到墙上的洞在洞口表里标placed为假且不计数() -> None:
    package = _gap_package("door")
    package.plan.openings[0].position_ratio = 0.9  # 这条线上一段墙也没有
    scene = compile_scene_package(package)
    assert [(o.opening_index, o.placed) for o in scene.openings] == [(0, False)]
    assert scene.opening_count_by_kind == {}


# ---------------------------------------------------------------------------
# 三、过口与入户门的起体
# ---------------------------------------------------------------------------


def test_过口落地不出窗下墙_洞高按过口档位() -> None:
    heights = HeightRules()
    scene = compile_scene_package(_three_opening_package([_opening(PASSAGE_X, kind="passage")]))
    assert scene.opening_count_by_kind == {"passage": 1}
    assert not [block for block in scene.meshes if ":sill:" in block.id]
    reveals = [block for block in scene.meshes if block.semantic == "reveal"]
    assert [block.id.split(":")[1] for block in reveals] == ["passage"]
    z_values = [vertex[2] for vertex in reveals[0].vertices]
    assert min(z_values) == pytest.approx(0.0)
    assert max(z_values) == pytest.approx(heights.pass_height_m)


def test_补出来的过口只补过梁不补窗下墙() -> None:
    heights = HeightRules()
    scene = compile_scene_package(_gap_package("passage"))
    fills = [block for block in scene.meshes if block.id.startswith("wall:fill:")]
    assert [block.id for block in fills] == ["wall:fill:0:lintel"]
    assert min(v[2] for v in fills[0].vertices) == pytest.approx(heights.pass_height_m)


def test_入户门按门起体_种类照写() -> None:
    heights = HeightRules()
    scene = compile_scene_package(
        _three_opening_package([_opening(DOOR_X, kind="entry-door", isOnOuterWall=True)])
    )
    assert scene.opening_count_by_kind == {"entry-door": 1}
    assert scene.openings[0].kind_source == "upstream"
    reveals = [block for block in scene.meshes if block.semantic == "reveal"]
    assert [block.id.split(":")[1] for block in reveals] == ["entry-door"]
    z_values = [vertex[2] for vertex in reveals[0].vertices]
    assert (min(z_values), max(z_values)) == pytest.approx((0.0, heights.door_height_m))


# ---------------------------------------------------------------------------
# 四、控制稿按最终种类画
# ---------------------------------------------------------------------------


def test_控制稿三种符号各自出现_门斜线窗十字过口无() -> None:
    """同一道墙上门、窗、过口各一个（种类全由上游给）：门洞里有斜线、窗洞里有十字、过口洞里
    在"若是门就该有斜线"的那个像素上是黑的。三个洞都按同一份高度档位起体。"""
    package = _three_opening_package(
        [
            _opening(DOOR_X, kind="door"),
            _opening(WINDOW_X, kind="window"),
            _opening(PASSAGE_X, kind="passage"),
        ]
    )
    scene = compile_scene_package(package)
    frames = {frame.kind: frame for frame in base_render._opening_frames(scene)}
    assert set(frames) == {"door", "window", "passage"}

    views = render_base_views(scene, ROOM_CAMERA_ID, WIDTH_PX, HEIGHT_PX)
    sketch = _open_gray(views.sketch_png)

    def on_wall(along_m: float, z_m: float, frame: base_render._OpeningFrame) -> tuple[int, int]:
        return _project(scene, ROOM_CAMERA_ID, (along_m, frame.across_center_m, z_m))

    door = frames["door"]
    door_w = door.along_m[1] - door.along_m[0]
    on_diagonal = on_wall(door.along_m[0] + door_w * 0.25, door.z_m[1] * 0.25, door)
    assert _white_near(sketch, *on_diagonal), "门洞里没有斜线"

    window = frames["window"]
    mid_x = (window.along_m[0] + window.along_m[1]) * 0.5
    mid_z = (window.z_m[0] + window.z_m[1]) * 0.5
    assert _white_near(sketch, *on_wall(mid_x, window.z_m[0] + 0.3, window)), "窗洞里没有竖梃"
    assert _white_near(sketch, *on_wall(window.along_m[0] + 0.2, mid_z, window)), "窗洞里没有横梃"

    passage = frames["passage"]
    passage_w = passage.along_m[1] - passage.along_m[0]
    diagonal_point = on_wall(passage.along_m[0] + passage_w * 0.25, passage.z_m[1] * 0.25, passage)
    assert not _white_near(sketch, *diagonal_point), "过口洞里画了门扇斜线"
    assert not _white_near(
        sketch, *on_wall(passage.along_m[0] + passage_w * 0.5, passage.z_m[1] * 0.15, passage)
    ), "过口洞里出现了竖梃一类的线"


def test_同一个洞判成过口的控制稿是判成门的子集_只少一条斜线() -> None:
    """把过口高度与门高设成同一个数，两份包的几何逐字相同，控制稿的差别只能来自符号：
    过口稿上白的像素门稿上也白；门稿多出来的正是斜线。"""
    heights = HeightRules(door_height_m=2.2, pass_height_m=2.2)
    as_door = compile_scene_package(
        _three_opening_package([_opening(PASSAGE_X, kind="door")], heights)
    )
    as_passage = compile_scene_package(
        _three_opening_package([_opening(PASSAGE_X, kind="passage")], heights)
    )
    assert [m.vertices for m in as_door.meshes] == [m.vertices for m in as_passage.meshes]

    door_sketch = _open_gray(
        render_base_views(as_door, ROOM_CAMERA_ID, WIDTH_PX, HEIGHT_PX).sketch_png
    )
    passage_sketch = _open_gray(
        render_base_views(as_passage, ROOM_CAMERA_ID, WIDTH_PX, HEIGHT_PX).sketch_png
    )
    white = base_render.SKETCH_FOREGROUND_U8
    assert not np.any((passage_sketch == white) & (door_sketch != white)), "过口稿画了门稿没有的线"
    assert np.count_nonzero(door_sketch == white) > np.count_nonzero(passage_sketch == white)


def test_补出来的过口按洞口表画_不按有无窗下墙猜() -> None:
    """落在墙段空隙里的洞没有套框，老退路是"没有窗下墙就当门画"——过口正好没有窗下墙，
    老退路会给它画上门扇。有洞口表就按表。"""
    scene = compile_scene_package(_gap_package("passage"))
    frames = base_render._opening_frames(scene)
    assert [frame.kind for frame in frames] == ["passage"]
    assert base_render._opening_symbol_segments(frames[0]) == []

    # 老场景包（没有洞口表）走的还是老退路，这条路只给 2026-09-05 之前编的包用
    legacy = scene.model_copy(update={"openings": []})
    assert [frame.kind for frame in base_render._opening_frames(legacy)] == ["door"]


def test_洞口表与网格对不上就炸() -> None:
    scene = compile_scene_package(_gap_package("passage"))
    mismatched = scene.model_copy(
        update={"openings": [scene.openings[0].model_copy(update={"opening_index": 7})]}
    )
    with pytest.raises(base_render.BaseRenderError):
        base_render._opening_frames(mismatched)


# ---------------------------------------------------------------------------
# 五、真户型：旧产物仍能读，新产物全链渲通
# ---------------------------------------------------------------------------


def test_旧19洞产物没有kind仍读得进_全部走猜法() -> None:
    package = DesignPackage.model_validate(json.loads(REAL_19_PACKAGE.read_text(encoding="utf-8")))
    assert {opening.kind for opening in package.plan.openings} == {"unknown"}
    scene = compile_scene_package(package)
    assert scene.guessed_opening_count == 19
    assert scene.guessed_opening_indices == list(range(19))
    assert scene.opening_count_by_kind == {"door": 9, "window": 10}


def test_新16洞产物十五个照上游一个走猜法_全链渲通() -> None:
    """c9fc00c 产物：15 个上游给了种类，1 个 `unknown`（第 3 个，入户门旁的管井小门，两侧都不是
    房间）走猜法——它在外轮廓上，猜成窗。揭顶机位五路渲得出来，控制稿里有入户门的框。"""
    package = DesignPackage.model_validate(json.loads(REAL_16_PACKAGE.read_text(encoding="utf-8")))
    assert len(package.plan.openings) == 16
    scene = compile_scene_package(package)
    assert scene.opening_count_by_kind == {"door": 6, "window": 9, "entry-door": 1}
    assert sum(scene.opening_count_by_kind.values()) == 16
    assert scene.guessed_opening_count == 1
    assert scene.guessed_opening_indices == [3]
    assert scene.openings[3].kind == "window"
    assert all(entry.placed for entry in scene.openings)
    pairs = zip(package.plan.openings, scene.openings, strict=True)
    assert all(given.kind == final.kind for given, final in pairs if given.kind != "unknown")

    frames = base_render._opening_frames(scene)
    assert {frame.kind for frame in frames} == {"door", "window", "entry-door"}

    views = render_base_views(scene, "cam-bird-dollhouse", 320, 240)
    # 揭顶机位 45° 张角从远处俯瞰，整户只占画幅两成多（1280×960 上实测 0.225）；这儿只守"不是空图"
    assert views.covered_pixel_ratio > 0.1
    sketch = np.asarray(Image.open(io.BytesIO(views.sketch_png)))
    assert np.count_nonzero(sketch == base_render.SKETCH_FOREGROUND_U8) > 0
