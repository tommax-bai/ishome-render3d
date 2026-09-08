"""控制稿门窗符号方案（``sketch_symbols`` / CLI ``--sketch-symbols``）的守门测试。

每个方案都要：确定性、同编码、门与窗的符号互不相同、过口不画、四路一个字节不动；
默认方案是 ``frame-handle``（2026-09-06 用户裁决从 ``diagonal-cross`` 换的，来路与数据见
``base_render.DEFAULT_SKETCH_SYMBOLS``），再换默认要用户拍。各方案的画法见
``base_render.SKETCH_SYMBOL_SCHEMES``，这里按画法逐点验：该白的白、该黑的黑。

场景复用 test_control_sketch 现造的那间房（北墙上一门一窗，室内机位从南墙边平视北墙，
所以相机在墙厚方向的负侧、朝相机的墙面是 y=3.0 那一面）。
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image
from test_control_sketch import (
    DOOR_TOP_MM,
    DOOR_X_MM,
    HEIGHT_PX,
    NORTH_WALL_Y_MM,
    ROOM_CAMERA_ID,
    WALL_CENTER_Y_MM,
    WIDTH_PX,
    WINDOW_X_MM,
    WINDOW_Z_MM,
    _make_scene,
    _open_gray,
    _project,
    _white_near,
)
from test_opening_kind import (
    DOOR_X,
    WINDOW_X,
    _opening,
    _three_opening_package,
)
from test_opening_kind import (
    ROOM_CAMERA_ID as PACKAGE_CAMERA_ID,
)

from render3d_worker import base_render, cli
from render3d_worker.base_render import (
    DEFAULT_SKETCH_SYMBOLS,
    SKETCH_SYMBOL_SCHEMES,
    BaseRenderError,
    SketchSymbolScheme,
    render_base_views,
)
from render3d_worker.models import BaseRenderViews
from render3d_worker.scene_compile import compile_scene_package

OTHER_SCHEMES: tuple[SketchSymbolScheme, ...] = tuple(
    scheme for scheme in SKETCH_SYMBOL_SCHEMES if scheme != DEFAULT_SKETCH_SYMBOLS
)
"""闭集里除默认之外的方案（含旧默认 ``diagonal-cross``，它留着复现 2026-09-06 之前的样本）。"""
ROOM_FACE_Y_MM = NORTH_WALL_Y_MM[0]
"""朝相机那一面：机位在南边，北墙朝南的面是 y=3000。"""

WINDOW_INNER_X_MM = (
    WINDOW_X_MM[0] + base_render.SKETCH_FRAME_INSET_MM,
    WINDOW_X_MM[1] - base_render.SKETCH_FRAME_INSET_MM,
)
WINDOW_INNER_Z_MM = (
    WINDOW_Z_MM[0] + base_render.SKETCH_FRAME_INSET_MM,
    WINDOW_Z_MM[1] - base_render.SKETCH_FRAME_INSET_MM,
)
WINDOW_MID_X_MM = (WINDOW_X_MM[0] + WINDOW_X_MM[1]) * 0.5
WINDOW_MID_Z_MM = (WINDOW_Z_MM[0] + WINDOW_Z_MM[1]) * 0.5
SILL_LINE_Z_MM = WINDOW_Z_MM[0] - base_render.SKETCH_SILL_DROP_MM
DOOR_WIDTH_MM = DOOR_X_MM[1] - DOOR_X_MM[0]
DOOR_DIAGONAL_QUARTER = (DOOR_X_MM[0] + DOOR_WIDTH_MM * 0.25, WALL_CENTER_Y_MM, DOOR_TOP_MM * 0.25)
DOOR_LEAF_LINE = (DOOR_X_MM[0] + base_render.SKETCH_DOOR_LEAF_OFFSET_MM, WALL_CENTER_Y_MM, 1.0)
DOOR_HANDLE_MID = (
    DOOR_X_MM[1]
    - base_render.SKETCH_DOOR_HANDLE_EDGE_MM
    - base_render.SKETCH_DOOR_HANDLE_LENGTH_MM * 0.5,
    WALL_CENTER_Y_MM,
    base_render.SKETCH_DOOR_HANDLE_HEIGHT_MM,
)


def _views_by_scheme(scheme: SketchSymbolScheme) -> BaseRenderViews:
    return render_base_views(_make_scene(), ROOM_CAMERA_ID, WIDTH_PX, HEIGHT_PX, scheme)


def _sketch(scheme: SketchSymbolScheme) -> npt.NDArray[np.int64]:
    return _open_gray(_views_by_scheme(scheme).sketch_png)


def _white(sketch: npt.NDArray[np.int64], point_mm: tuple[float, float, float]) -> bool:
    return _white_near(sketch, *_project(_make_scene(), ROOM_CAMERA_ID, point_mm))


def test_默认方案是双框加把手_且显式给默认与不给逐字节相同() -> None:
    """守默认值本身：2026-09-06 用户裁决把默认从 ``diagonal-cross`` 换成 ``frame-handle``。
    不给方案渲出来的稿要与显式给 ``frame-handle`` 逐字节相同，且与旧默认那张不同——
    把 ``DEFAULT_SKETCH_SYMBOLS`` 改回 ``diagonal-cross`` 这三条会一起红。"""
    assert DEFAULT_SKETCH_SYMBOLS == "frame-handle"
    implicit = render_base_views(_make_scene(), ROOM_CAMERA_ID, WIDTH_PX, HEIGHT_PX)
    explicit = _views_by_scheme(DEFAULT_SKETCH_SYMBOLS)
    assert implicit.sketch_png == explicit.sketch_png
    assert implicit.sketch_png == _views_by_scheme("frame-handle").sketch_png
    assert implicit.sketch_png != _views_by_scheme("diagonal-cross").sketch_png


def test_认不出的方案炸() -> None:
    with pytest.raises(BaseRenderError, match="符号方案认不出"):
        render_base_views(
            _make_scene(),
            ROOM_CAMERA_ID,
            WIDTH_PX,
            HEIGHT_PX,
            "plain",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("scheme", OTHER_SCHEMES)
def test_换方案只动控制稿_四路逐字节不变(scheme: SketchSymbolScheme) -> None:
    default = _views_by_scheme(DEFAULT_SKETCH_SYMBOLS)
    other = _views_by_scheme(scheme)
    assert other.geometry_png == default.geometry_png
    assert other.depth_png == default.depth_png
    assert other.line_png == default.line_png
    assert other.mask_png == default.mask_png
    assert other.sketch_png != default.sketch_png


@pytest.mark.parametrize("scheme", SKETCH_SYMBOL_SCHEMES)
def test_每个方案确定性且同编码(scheme: SketchSymbolScheme) -> None:
    first = _views_by_scheme(scheme).sketch_png
    second = _views_by_scheme(scheme).sketch_png
    assert first == second
    image = Image.open(io.BytesIO(first))
    assert image.mode == "L" and image.size == (WIDTH_PX, HEIGHT_PX)
    values = set(np.unique(np.asarray(image)).tolist())
    assert values == {base_render.SKETCH_BACKGROUND_U8, base_render.SKETCH_FOREGROUND_U8}


@pytest.mark.parametrize("scheme", SKETCH_SYMBOL_SCHEMES)
def test_每个方案门与窗的符号互不相同_过口不画(scheme: SketchSymbolScheme) -> None:
    """同一个框按门画与按窗画，线段集合不同；过口一条都没有。"""
    box = (DOOR_X_MM, NORTH_WALL_Y_MM, (0.0, DOOR_TOP_MM))

    def segments(kind: str) -> set[tuple[float, ...]]:
        frame = base_render._OpeningFrame(kind, 0, *box)
        return {
            tuple(np.round(np.concatenate([start, end]), 6).tolist())
            for start, end in base_render._opening_symbol_segments(frame, scheme, -1.0)
        }

    door, window, entry = segments("door"), segments("window"), segments("entry-door")
    assert door and window
    assert door != window
    assert entry == door
    assert segments("passage") == set()


def test_frame_sill_窗是双框加窗台线_门是外框加门扇线() -> None:
    sketch = _sketch("frame-sill")
    assert _white(sketch, (WINDOW_INNER_X_MM[0], WALL_CENTER_Y_MM, WINDOW_MID_Z_MM)), (
        "窗内框左边没画"
    )
    assert _white(sketch, (WINDOW_MID_X_MM, WALL_CENTER_Y_MM, WINDOW_INNER_Z_MM[1])), (
        "窗内框顶边没画"
    )
    assert _white(sketch, (WINDOW_MID_X_MM, ROOM_FACE_Y_MM, SILL_LINE_Z_MM)), (
        "窗台线没画在朝相机的墙面上"
    )
    assert not _white(sketch, (WINDOW_MID_X_MM, WALL_CENTER_Y_MM, WINDOW_MID_Z_MM)), (
        "窗中心不该有十字"
    )
    assert not _white(sketch, (WINDOW_X_MM[0] + 250.0, WALL_CENTER_Y_MM, WINDOW_Z_MM[0] + 300.0)), (
        "玻璃面上不该有线"
    )

    assert _white(sketch, DOOR_LEAF_LINE), "门扇线没画"
    assert _white(sketch, (DOOR_X_MM[0] + DOOR_WIDTH_MM * 0.5, WALL_CENTER_Y_MM, DOOR_TOP_MM)), (
        "门外框顶边没画"
    )
    assert not _white(sketch, DOOR_DIAGONAL_QUARTER), "门洞里不该有斜线"
    assert not _white(sketch, (DOOR_X_MM[1] - 200.0, WALL_CENTER_Y_MM, 1000.0)), "门扇线只在一侧"


def test_frame_handle_窗照frame_sill_门是外框加门扇线加把手() -> None:
    """合成方案：窗的线段集合与 ``frame-sill`` 逐条相同；门的线段集合＝``frame-sill`` 的门
    ∪ ``glazing-hatch`` 的门（外框只算一次）；三个方案的控制稿两两不同。"""
    assert "frame-handle" in SKETCH_SYMBOL_SCHEMES

    def segments(kind: str, scheme: SketchSymbolScheme) -> set[tuple[float, ...]]:
        frame = base_render._OpeningFrame(kind, 0, DOOR_X_MM, NORTH_WALL_Y_MM, (0.0, DOOR_TOP_MM))
        if kind == "window":
            frame = base_render._OpeningFrame(kind, 0, WINDOW_X_MM, NORTH_WALL_Y_MM, WINDOW_Z_MM)
        return {
            tuple(np.round(np.concatenate([start, end]), 6).tolist())
            for start, end in base_render._opening_symbol_segments(frame, scheme, -1.0)
        }

    assert segments("window", "frame-handle") == segments("window", "frame-sill")
    assert segments("door", "frame-handle") == segments("door", "frame-sill") | segments(
        "door", "glazing-hatch"
    )
    assert segments("entry-door", "frame-handle") == segments("door", "frame-handle")
    assert len(segments("door", "frame-handle")) == len(segments("door", "frame-sill")) + 1

    sketch = _sketch("frame-handle")
    assert _white(sketch, (WINDOW_INNER_X_MM[0], WALL_CENTER_Y_MM, WINDOW_MID_Z_MM)), (
        "窗内框左边没画"
    )
    assert _white(sketch, (WINDOW_MID_X_MM, ROOM_FACE_Y_MM, SILL_LINE_Z_MM)), (
        "窗台线没画在朝相机的墙面上"
    )
    assert not _white(sketch, (WINDOW_MID_X_MM, WALL_CENTER_Y_MM, WINDOW_MID_Z_MM)), (
        "窗中心不该有十字"
    )
    assert _white(sketch, DOOR_LEAF_LINE), "门扇线没画"
    assert _white(sketch, DOOR_HANDLE_MID), "门把手没画"
    assert _white(sketch, (DOOR_X_MM[0] + DOOR_WIDTH_MM * 0.5, WALL_CENTER_Y_MM, DOOR_TOP_MM)), (
        "门外框顶边没画"
    )
    assert not _white(sketch, DOOR_DIAGONAL_QUARTER), "门洞里不该有斜线"

    frame_sill = _views_by_scheme("frame-sill").sketch_png
    glazing_hatch = _views_by_scheme("glazing-hatch").sketch_png
    frame_handle = _views_by_scheme("frame-handle").sketch_png
    assert len({frame_sill, glazing_hatch, frame_handle}) == 3, "三个方案的控制稿要两两不同"


def test_glazing_hatch_窗是双框加斜向短划_门是外框加把手() -> None:
    sketch = _sketch("glazing-hatch")
    inner_w = WINDOW_INNER_X_MM[1] - WINDOW_INNER_X_MM[0]
    inner_h = WINDOW_INNER_Z_MM[1] - WINDOW_INNER_Z_MM[0]
    step = base_render.SKETCH_HATCH_LENGTH_RATIO * min(inner_w, inner_h) * np.cos(np.radians(45.0))
    first_stroke_mid = (
        WINDOW_INNER_X_MM[0] + inner_w * 0.18 + step * 0.5,
        WALL_CENTER_Y_MM,
        WINDOW_INNER_Z_MM[0] + inner_h * 0.55 + step * 0.5,
    )
    assert _white(sketch, (WINDOW_INNER_X_MM[0], WALL_CENTER_Y_MM, WINDOW_MID_Z_MM)), (
        "窗内框左边没画"
    )
    assert _white(sketch, first_stroke_mid), "玻璃反光短划没画"
    assert not _white(sketch, (WINDOW_X_MM[1] - 200.0, WALL_CENTER_Y_MM, WINDOW_Z_MM[0] + 300.0)), (
        "短划之外的玻璃面不该有线"
    )
    assert not _white(sketch, (WINDOW_MID_X_MM, ROOM_FACE_Y_MM, SILL_LINE_Z_MM)), (
        "这个方案不画窗台线"
    )

    assert _white(sketch, DOOR_HANDLE_MID), "门把手没画"
    assert _white(sketch, (DOOR_X_MM[0] + DOOR_WIDTH_MM * 0.5, WALL_CENTER_Y_MM, DOOR_TOP_MM)), (
        "门外框顶边没画"
    )
    assert not _white(sketch, DOOR_LEAF_LINE), "这个方案不画门扇线"
    assert not _white(sketch, DOOR_DIAGONAL_QUARTER), "门洞里不该有斜线"


def test_leaf_swing_门扇朝相机开_窗是双框窗台线加中竖梃() -> None:
    sketch = _sketch("leaf-swing")
    swing = np.radians(base_render.SKETCH_DOOR_SWING_DEG)
    free_edge_mid = (
        DOOR_X_MM[0] + DOOR_WIDTH_MM * float(np.cos(swing)),
        WALL_CENTER_Y_MM - DOOR_WIDTH_MM * float(np.sin(swing)),
        1.0,
    )
    assert free_edge_mid[1] < ROOM_FACE_Y_MM, "门扇该开进相机所在的房间（y 更小的一侧）"
    assert _white(sketch, free_edge_mid), "门扇的自由竖边没画"
    assert _white(sketch, (DOOR_X_MM[0] + DOOR_WIDTH_MM * 0.5, WALL_CENTER_Y_MM, DOOR_TOP_MM)), (
        "门外框顶边没画"
    )
    assert not _white(sketch, (DOOR_X_MM[1] - 100.0, WALL_CENTER_Y_MM, 1000.0)), (
        "门扇之外的洞里不该有线"
    )

    assert _white(sketch, (WINDOW_MID_X_MM, WALL_CENTER_Y_MM, WINDOW_MID_Z_MM)), "中竖梃没画"
    assert _white(sketch, (WINDOW_MID_X_MM, ROOM_FACE_Y_MM, SILL_LINE_Z_MM)), "窗台线没画"
    assert not _white(sketch, (WINDOW_X_MM[0] + 250.0, WALL_CENTER_Y_MM, WINDOW_MID_Z_MM)), (
        "不该有横梃"
    )


def test_相机在墙另一侧时窗台线画在另一面_门扇朝另一边开() -> None:
    frame = base_render._OpeningFrame("window", 0, WINDOW_X_MM, NORTH_WALL_Y_MM, WINDOW_Z_MM)
    south = base_render._sill_segment(frame, -1.0)
    north = base_render._sill_segment(frame, 1.0)
    assert south[0][1] == NORTH_WALL_Y_MM[0] and north[0][1] == NORTH_WALL_Y_MM[1]

    door = base_render._OpeningFrame("door", 0, DOOR_X_MM, NORTH_WALL_Y_MM, (0.0, DOOR_TOP_MM))
    toward_south = base_render._opening_symbol_segments(door, "leaf-swing", -1.0)
    toward_north = base_render._opening_symbol_segments(door, "leaf-swing", 1.0)
    south_free_y = min(float(point[1]) for start, end in toward_south for point in (start, end))
    north_free_y = max(float(point[1]) for start, end in toward_north for point in (start, end))
    assert south_free_y < WALL_CENTER_Y_MM < north_free_y


def test_编场景包那条路的洞_每个方案门窗符号都出现(tmp_path: Path) -> None:
    """从输入包编出来的洞（套框网格 + 洞口表）走每个方案都画得出门与窗；CLI ``--sketch-symbols``
    出的字节与纯库同方案一致，默认与不给一致，不认的名字被 argparse 拒掉。"""
    package = _three_opening_package(
        [_opening(DOOR_X, kind="door"), _opening(WINDOW_X, kind="window")]
    )
    scene = compile_scene_package(package)
    frames = {frame.kind: frame for frame in base_render._opening_frames(scene)}
    assert set(frames) == {"door", "window"}
    width_px, height_px = 320, 240

    package_path = tmp_path / "package.json"
    package_path.write_text(package.model_dump_json(by_alias=True), encoding="utf-8")
    white = base_render.SKETCH_FOREGROUND_U8
    diagonal_sketch = _open_gray(
        render_base_views(
            scene, PACKAGE_CAMERA_ID, width_px, height_px, "diagonal-cross"
        ).sketch_png
    )
    for scheme in SKETCH_SYMBOL_SCHEMES:
        views = render_base_views(scene, PACKAGE_CAMERA_ID, width_px, height_px, scheme)
        sketch = _open_gray(views.sketch_png)
        if scheme != "diagonal-cross":
            assert np.count_nonzero(sketch == white) > np.count_nonzero(diagonal_sketch == white), (
                f"{scheme}：外框加符号的线该比一条斜线加一个十字多"
            )
        out_dir = tmp_path / scheme
        code = cli.main(
            [
                "--design",
                str(package_path),
                "-o",
                str(out_dir),
                "--camera",
                PACKAGE_CAMERA_ID,
                "--width-px",
                str(width_px),
                "--height-px",
                str(height_px),
                "--sketch-symbols",
                scheme,
            ]
        )
        assert code == 0
        assert (out_dir / PACKAGE_CAMERA_ID / cli.SKETCH_PNG).read_bytes() == views.sketch_png
        assert (out_dir / PACKAGE_CAMERA_ID / cli.LINE_PNG).read_bytes() == views.line_png

    default_out = tmp_path / "default"
    assert (
        cli.main(
            [
                "--design",
                str(package_path),
                "-o",
                str(default_out),
                "--camera",
                PACKAGE_CAMERA_ID,
                "--width-px",
                str(width_px),
                "--height-px",
                str(height_px),
            ]
        )
        == 0
    )
    assert (default_out / PACKAGE_CAMERA_ID / cli.SKETCH_PNG).read_bytes() == (
        tmp_path / DEFAULT_SKETCH_SYMBOLS / PACKAGE_CAMERA_ID / cli.SKETCH_PNG
    ).read_bytes()
    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            [
                "--design",
                str(package_path),
                "-o",
                str(tmp_path / "bad"),
                "--sketch-symbols",
                "plain",
            ]
        )
    assert exit_info.value.code == 2
    assert json.loads((default_out / cli.SCENE_PACKAGE_JSON).read_text(encoding="utf-8"))[
        "openings"
    ]
