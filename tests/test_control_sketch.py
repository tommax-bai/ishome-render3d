"""控制稿（sketch.png，第五路）的守门测试：编码同线稿、确定性、画法规则四条——地面上的
房间分界线不画、天花与墙的交线不画而地脚线画、门与窗的符号不同、家具体块的可见边画。

场景在测试里现造（同 test_base_render 的理由：证的是"给同一份网格，画法是定死的"）：两间房
共一条**没有墙**的地板接缝，北墙上一门一窗（带套框网格，id 按 mesh 那一层 ``reveal:{kind}:…``
的格式写），可选一件家具体块。规则本身写在 base_render 模块 docstring，这里只验它。
"""

from __future__ import annotations

import io

import numpy as np
import numpy.typing as npt
from PIL import Image

from render3d_worker import base_render
from render3d_worker.base_render import render_base_views, resolve_camera_pose
from render3d_worker.models import CameraSpec, Mesh, ScenePackage, SurfaceMaterial
from render3d_worker.raster import look_at_matrix, perspective_matrix

WIDTH_PX = 640
HEIGHT_PX = 480

ROOM_A = "客厅"
ROOM_B = "餐厅"
CEILING_M = 2.8
WALL_THICKNESS_M = 0.2
SEAM_X_M = 4.0
"""两间房地板相接的那条线：x = 4，**这儿没有墙**。"""

ROOM_A_X_M = (0.0, SEAM_X_M)
ROOM_B_X_M = (SEAM_X_M, 7.0)
ROOM_Y_M = (0.0, 3.0)
NORTH_WALL_Y_M = (3.0, 3.0 + WALL_THICKNESS_M)
WALL_CENTER_Y_M = 3.0 + WALL_THICKNESS_M * 0.5

DOOR_X_M = (0.5, 1.4)
DOOR_TOP_M = 2.05
WINDOW_X_M = (2.5, 3.5)
WINDOW_Z_M = (0.9, 2.1)

BOX_X_M = (2.6, 3.6)
BOX_Y_M = (1.8, 2.6)
BOX_TOP_Z_M = 0.8
"""家具体块摆在退景路径（x=2）东侧 0.6 米、离机位 1.45 米以上：不改变机位（家具余量
0.40 米），顶面前棱又落在 80° 画幅之内。"""

BIRD_CAMERA_ID = "bird"
ROOM_CAMERA_ID = "room-客厅"
ROOM_FOV_DEG = 80.0
"""室内那台给 80 度：从南墙边（y=0.35）平视北墙（y=3.0），要把门、窗、地脚线、天花交线
一起框进画面。"""

MATERIAL_ID = "mat"
NEAR_PX = 2
"""投影点周围找线的半径（像素）：投影落在像素中心附近，线宽 1 像素，两像素够吃掉取整。"""


def _quad(
    mesh_id: str,
    semantic: str,
    corners: list[tuple[float, float, float]],
    room: str | None = None,
) -> Mesh:
    return Mesh(
        id=mesh_id,
        semantic=semantic,  # type: ignore[arg-type]
        material_id=MATERIAL_ID,
        room=room,
        vertices=corners,
        triangles=[(0, 1, 2), (0, 2, 3)],
    )


def _box(
    mesh_id: str,
    semantic: str,
    x_m: tuple[float, float],
    y_m: tuple[float, float],
    z_m: tuple[float, float],
    room: str | None = None,
) -> Mesh:
    """长方体：8 顶点 12 三角形（绕序不统一，底渲不许依赖它）。"""
    x0, x1 = x_m
    y0, y1 = y_m
    z0, z1 = z_m
    vertices: list[tuple[float, float, float]] = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    triangles: list[tuple[int, int, int]] = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ]
    return Mesh(
        id=mesh_id,
        semantic=semantic,  # type: ignore[arg-type]
        material_id=MATERIAL_ID,
        room=room,
        vertices=vertices,
        triangles=triangles,
    )


def _reveal(mesh_id: str, x_m: tuple[float, float], z_m: tuple[float, float]) -> Mesh:
    """洞口套框：两侧洞壁 + 洞顶，落地的洞不出洞底（同 mesh ``_reveal_mesh`` 的形态）。"""
    x0, x1 = x_m
    y0, y1 = NORTH_WALL_Y_M
    z0, z1 = z_m
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []

    def face(corners: list[tuple[float, float, float]]) -> None:
        base = len(vertices)
        vertices.extend(corners)
        triangles.append((base, base + 1, base + 2))
        triangles.append((base, base + 2, base + 3))

    face([(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)])
    face([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)])
    face([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)])
    if z0 > 0.0:
        face([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)])
    return Mesh(
        id=mesh_id,
        semantic="reveal",
        material_id=MATERIAL_ID,
        room=None,
        vertices=vertices,
        triangles=triangles,
    )


def _make_scene(with_box: bool = False) -> ScenePackage:
    y0, y1 = ROOM_Y_M
    meshes: list[Mesh] = []
    for room, (x0, x1) in ((ROOM_A, ROOM_A_X_M), (ROOM_B, ROOM_B_X_M)):
        meshes.append(
            _quad(
                f"floor:{room}:0",
                "floor",
                [(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)],
                room,
            )
        )
        meshes.append(
            _quad(
                f"ceiling:{room}:0",
                "ceiling",
                [
                    (x0, y0, CEILING_M),
                    (x1, y0, CEILING_M),
                    (x1, y1, CEILING_M),
                    (x0, y1, CEILING_M),
                ],
                room,
            )
        )
    full_z = (0.0, CEILING_M)
    x_left, x_right = ROOM_A_X_M[0], ROOM_B_X_M[1]
    meshes.append(_box("wall:south", "wall", (x_left, x_right), (-WALL_THICKNESS_M, 0.0), full_z))
    meshes.append(_box("wall:west", "wall", (-WALL_THICKNESS_M, x_left), ROOM_Y_M, full_z))
    meshes.append(
        _box("wall:east", "wall", (x_right, x_right + WALL_THICKNESS_M), ROOM_Y_M, full_z)
    )
    north = NORTH_WALL_Y_M
    meshes.append(_box("wall:north:span:0", "wall", (x_left, DOOR_X_M[0]), north, full_z))
    meshes.append(_box("wall:north:lintel:0", "wall", DOOR_X_M, north, (DOOR_TOP_M, CEILING_M)))
    meshes.append(_box("wall:north:span:1", "wall", (DOOR_X_M[1], WINDOW_X_M[0]), north, full_z))
    meshes.append(_box("wall:north:sill:0", "wall", WINDOW_X_M, north, (0.0, WINDOW_Z_M[0])))
    meshes.append(
        _box("wall:north:lintel:1", "wall", WINDOW_X_M, north, (WINDOW_Z_M[1], CEILING_M))
    )
    meshes.append(_box("wall:north:span:2", "wall", (WINDOW_X_M[1], x_right), north, full_z))
    meshes.append(_reveal("reveal:door:outline:0:0", DOOR_X_M, (0.0, DOOR_TOP_M)))
    meshes.append(_reveal("reveal:window:outline:0:1", WINDOW_X_M, WINDOW_Z_M))
    if with_box:
        meshes.append(
            _box("furnishing:茶几", "furnishing", BOX_X_M, BOX_Y_M, (0.0, BOX_TOP_Z_M), ROOM_A)
        )
    return ScenePackage(
        revision_id="rev-test-控制稿",
        meshes=meshes,
        materials=[SurfaceMaterial(id=MATERIAL_ID, base_color_hex="#D0D0D0")],
        cameras=[
            CameraSpec(id=BIRD_CAMERA_ID, kind="bird", yaw_deg=0.0, pitch_deg=-60.0, fov_deg=55.0),
            CameraSpec(
                id=ROOM_CAMERA_ID,
                kind="room",
                room=ROOM_A,
                eye_height_m=1.55,
                yaw_deg=0.0,
                fov_deg=ROOM_FOV_DEG,
            ),
        ],
    )


def _open_gray(png: bytes) -> npt.NDArray[np.int64]:
    return np.asarray(Image.open(io.BytesIO(png)), dtype=np.int64)


def _project(
    scene: ScenePackage, camera_id: str, point_m: tuple[float, float, float]
) -> tuple[int, int]:
    """世界点 → 像素下标，用底渲真正在用的那两个矩阵。"""
    aspect = WIDTH_PX / HEIGHT_PX
    pose = resolve_camera_pose(scene, camera_id, aspect)
    view = look_at_matrix(pose.eye_m, pose.target_m, pose.up_hint_xyz)
    proj = perspective_matrix(pose.fov_deg, aspect, pose.near_clip_m, pose.far_clip_m)
    clip = proj @ (view @ np.array([*point_m, 1.0], dtype=np.float64))
    ndc = clip[:3] / clip[3]
    return int((ndc[0] + 1.0) * 0.5 * WIDTH_PX), int((1.0 - ndc[1]) * 0.5 * HEIGHT_PX)


def _white_near(
    image: npt.NDArray[np.int64], x_px: int, y_px: int, radius_px: int = NEAR_PX
) -> bool:
    assert 0 <= x_px < image.shape[1] and 0 <= y_px < image.shape[0], "投影点落到画幅外了"
    window = image[
        max(0, y_px - radius_px) : y_px + radius_px + 1,
        max(0, x_px - radius_px) : x_px + radius_px + 1,
    ]
    return bool((window == base_render.SKETCH_FOREGROUND_U8).any())


def test_控制稿与线稿同尺寸同编码() -> None:
    """单通道、同宽高、只有 0 与 255 两个值，黑底白线。"""
    views = render_base_views(_make_scene(), ROOM_CAMERA_ID, WIDTH_PX, HEIGHT_PX)
    sketch_image = Image.open(io.BytesIO(views.sketch_png))
    line_image = Image.open(io.BytesIO(views.line_png))
    assert sketch_image.mode == line_image.mode == "L"
    assert sketch_image.size == line_image.size == (WIDTH_PX, HEIGHT_PX)
    sketch = np.asarray(sketch_image)
    assert set(np.unique(sketch).tolist()) <= {
        base_render.SKETCH_BACKGROUND_U8,
        base_render.SKETCH_FOREGROUND_U8,
    }
    assert int(sketch[0, 0]) == base_render.SKETCH_BACKGROUND_U8


def test_控制稿渲两次逐字节相同() -> None:
    for camera_id in (BIRD_CAMERA_ID, ROOM_CAMERA_ID):
        first = render_base_views(_make_scene(with_box=True), camera_id, WIDTH_PX, HEIGHT_PX)
        second = render_base_views(_make_scene(with_box=True), camera_id, WIDTH_PX, HEIGHT_PX)
        assert first.sketch_png == second.sketch_png


def test_地面上的房间分界线不画() -> None:
    """失效清单 B1：两间房的地板在 x=4 处共面相接、没有墙。线稿把这条网格边界画出来
    （它是几何事实边），控制稿一个像素都不许画——地面上的线会被模型读成台阶。"""
    scene = _make_scene()
    views = render_base_views(scene, BIRD_CAMERA_ID, WIDTH_PX, HEIGHT_PX)
    line = _open_gray(views.line_png)
    sketch = _open_gray(views.sketch_png)

    # 只取 y ≥ 1.8 那一段：揭顶机位从南面 60° 俯视，南墙（通高 2.8 米）挡住了离它
    # 1.6 米以内的地板，再往南的接缝本来就看不见
    seam_points = [(SEAM_X_M, float(y_m), 0.0) for y_m in np.linspace(1.8, 2.6, 5)]
    pixels = [_project(scene, BIRD_CAMERA_ID, point) for point in seam_points]
    assert all(_white_near(line, x, y, radius_px=1) for x, y in pixels), "线稿本该画出这条接缝"
    assert not any(_white_near(sketch, x, y, radius_px=1) for x, y in pixels), (
        "控制稿在没有墙的地面上画了线"
    )


def test_天花与墙的交线不画而地脚线画() -> None:
    """失效清单 B2：北墙与天花的交线（z=2.8）线稿画、控制稿不画；北墙与地面的交线（z=0）
    控制稿要画。取门与窗之间那段素墙（x=2.0）验，避开洞口轮廓。"""
    scene = _make_scene()
    views = render_base_views(scene, ROOM_CAMERA_ID, WIDTH_PX, HEIGHT_PX)
    line = _open_gray(views.line_png)
    sketch = _open_gray(views.sketch_png)

    top_x, top_y = _project(scene, ROOM_CAMERA_ID, (2.0, NORTH_WALL_Y_M[0], CEILING_M))
    bottom_x, bottom_y = _project(scene, ROOM_CAMERA_ID, (2.0, NORTH_WALL_Y_M[0], 0.0))
    assert _white_near(line, top_x, top_y), "线稿本该画出天花与墙的交线"
    assert not _white_near(sketch, top_x, top_y), "控制稿画了天花与墙的交线"
    assert _white_near(sketch, bottom_x, bottom_y), "控制稿没画地脚线"


def test_门与窗的符号不同() -> None:
    """失效清单 B4：同一张图上门与窗的符号不同。默认方案 ``frame-handle``（2026-09-06 用户裁决
    从 ``diagonal-cross`` 换的）下，门＝外框三边 + 门扇线 + 把手短横，窗＝外框 + 内框 + 窗台线；
    门洞里不该有斜线、窗洞中心不该有十字。点取在墙厚中心平面上（窗台线除外——它画在朝相机
    那一面的墙面上）。"""
    scene = _make_scene()
    views = render_base_views(scene, ROOM_CAMERA_ID, WIDTH_PX, HEIGHT_PX)
    sketch = _open_gray(views.sketch_png)
    y_m = WALL_CENTER_Y_M

    door_width_m = DOOR_X_M[1] - DOOR_X_M[0]
    on_leaf_line = (DOOR_X_M[0] + base_render.SKETCH_DOOR_LEAF_OFFSET_M, y_m, 1.0)
    on_handle = (
        DOOR_X_M[1]
        - base_render.SKETCH_DOOR_HANDLE_EDGE_M
        - base_render.SKETCH_DOOR_HANDLE_LENGTH_M * 0.5,
        y_m,
        base_render.SKETCH_DOOR_HANDLE_HEIGHT_M,
    )
    on_diagonal = (DOOR_X_M[0] + door_width_m * 0.25, y_m, DOOR_TOP_M * 0.25)
    assert _white_near(sketch, *_project(scene, ROOM_CAMERA_ID, on_leaf_line)), "门洞里没有门扇线"
    assert _white_near(sketch, *_project(scene, ROOM_CAMERA_ID, on_handle)), "门洞里没有把手"
    assert not _white_near(sketch, *_project(scene, ROOM_CAMERA_ID, on_diagonal)), (
        "门洞里出现了斜线——默认方案不画斜线"
    )

    window_mid_x_m = (WINDOW_X_M[0] + WINDOW_X_M[1]) * 0.5
    window_mid_z_m = (WINDOW_Z_M[0] + WINDOW_Z_M[1]) * 0.5
    on_inner_frame = (WINDOW_X_M[0] + base_render.SKETCH_FRAME_INSET_M, y_m, window_mid_z_m)
    on_sill_line = (
        window_mid_x_m,
        NORTH_WALL_Y_M[0],
        WINDOW_Z_M[0] - base_render.SKETCH_SILL_DROP_M,
    )
    off_symbol = (window_mid_x_m, y_m, window_mid_z_m)
    assert _white_near(sketch, *_project(scene, ROOM_CAMERA_ID, on_inner_frame)), "窗洞里没有内框"
    assert _white_near(sketch, *_project(scene, ROOM_CAMERA_ID, on_sill_line)), (
        "窗台线没画在朝相机那一面的墙面上"
    )
    assert not _white_near(sketch, *_project(scene, ROOM_CAMERA_ID, off_symbol)), (
        "窗洞中心出现了线——默认方案不画十字"
    )

    door_frame = base_render._OpeningFrame("door", 0, DOOR_X_M, NORTH_WALL_Y_M, (0.0, DOOR_TOP_M))
    window_frame = base_render._OpeningFrame("window", 0, WINDOW_X_M, NORTH_WALL_Y_M, WINDOW_Z_M)
    entry_frame = base_render._OpeningFrame(
        "entry-door", 0, DOOR_X_M, NORTH_WALL_Y_M, (0.0, DOOR_TOP_M)
    )
    passage_frame = base_render._OpeningFrame(
        "passage", 0, DOOR_X_M, NORTH_WALL_Y_M, (0.0, DOOR_TOP_M)
    )
    # 门＝外框三边（不画门槛线）+ 门扇线 + 把手；窗＝外框四边 + 内框四边 + 窗台线
    assert len(base_render._opening_symbol_segments(door_frame)) == 5
    assert len(base_render._opening_symbol_segments(entry_frame)) == 5
    assert len(base_render._opening_symbol_segments(window_frame)) == 9
    assert base_render._opening_symbol_segments(passage_frame) == []


def test_洞口形态从网格里读得出来() -> None:
    """套框网格 ``reveal:{kind}:…`` 直接读种类与框；两个洞的框与造场景时给的数一致。"""
    frames = base_render._opening_frames(_make_scene())
    by_kind = {frame.kind: frame for frame in frames}
    assert set(by_kind) == {"door", "window"}
    assert by_kind["door"].along_axis == 0
    assert by_kind["door"].along_m == DOOR_X_M
    assert by_kind["door"].z_m == (0.0, DOOR_TOP_M)
    assert by_kind["door"].across_m == NORTH_WALL_Y_M
    assert by_kind["door"].across_center_m == WALL_CENTER_Y_M
    assert by_kind["window"].along_m == WINDOW_X_M
    assert by_kind["window"].z_m == WINDOW_Z_M


def test_家具体块的可见边画() -> None:
    """上游给了家具体块就画它的可见边：体块顶面与正面的折边在控制稿上是白的；同一个像素
    在没有家具的场景里是黑的（那儿只是地板）。"""
    with_box = _make_scene(with_box=True)
    without_box = _make_scene(with_box=False)
    edge_point = ((BOX_X_M[0] + BOX_X_M[1]) * 0.5, BOX_Y_M[0], BOX_TOP_Z_M)
    x_px, y_px = _project(with_box, ROOM_CAMERA_ID, edge_point)
    assert _project(without_box, ROOM_CAMERA_ID, edge_point) == (x_px, y_px), (
        "有没有这件家具不该改变机位（它离退景路径够远）"
    )

    sketch_with = _open_gray(
        render_base_views(with_box, ROOM_CAMERA_ID, WIDTH_PX, HEIGHT_PX).sketch_png
    )
    sketch_without = _open_gray(
        render_base_views(without_box, ROOM_CAMERA_ID, WIDTH_PX, HEIGHT_PX).sketch_png
    )
    assert _white_near(sketch_with, x_px, y_px), "家具体块的棱没画"
    assert not _white_near(sketch_without, x_px, y_px), "没有家具的地板上不该有线"
