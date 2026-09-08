"""底渲四路输出的守门测试：确定性、四路一致、遮罩对得上、深度合理、失败响亮，
外加四条口径（室内机位、近平面裁剪、bird 剔天花、**观感只落在几何那一路**）。

场景包**在测试里现造**，不读 `tests/fixtures/`：这几条要证的是"给同一份网格，底渲的行为
是定死的"，与户型图怎么编成场景包无关。夹在一起，编包那边改一个三角化顺序就会把底渲的
测试连坐弄红，查错的人要先排除一整条不相干的线。**本文件一处都不钉网格块数**——墙体怎么
分块是 mesh 那一路的事，它改了不该连坐这里。
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from render3d_worker import base_render
from render3d_worker.base_render import (
    BaseRenderError,
    render_base_views,
    resolve_camera_pose,
)
from render3d_worker.models import (
    BaseRenderViews,
    CameraSpec,
    Mesh,
    ScenePackage,
    SurfaceMaterial,
)
from render3d_worker.raster import look_at_matrix, perspective_matrix

# 一间 4000×3000×2800 毫米的客厅：地板 + 四面墙 + 天花 + 一件 1000×1000×800 毫米的家具体块。
# 天花是**必须有**的：bird 剔天花那条裁决，没有天花就验不到（见 test_bird机位剔掉天花）。
ROOM_NAME = "客厅"
ROOM_WIDTH_MM = 4000.0
ROOM_DEPTH_MM = 3000.0
ROOM_HEIGHT_MM = 2800.0
BOX_X_MM = (2500.0, 3500.0)
BOX_Y_MM = (1000.0, 2000.0)
BOX_TOP_Z_MM = 800.0

FLOOR_CENTER_MM = (2000.0, 1500.0, 0.0)
BOX_TOP_CENTER_MM = (3000.0, 1500.0, BOX_TOP_Z_MM)

WIDTH_PX = 320
HEIGHT_PX = 240

BIRD_CAMERA_ID = "bird-整户"
ROOM_CAMERA_ID = "room-客厅"
ROOM_WIDE_CAMERA_ID = "room-客厅-广角"
ROOM_AUTO_CAMERA_ID = "room-客厅-自动"

# bird 机位取 -60 度而不是默认的 -30：这间房四面墙齐全，-30 度时视线会从南墙的**外侧**
# 擦过去，画面里只有一面墙背。-60 度越过墙顶看进屋里，才验得到室内的深度关系。
BIRD_PITCH_DEG = -60.0

# 广角室内机位：普通那台（竖直 60 度）在 1500 毫米外只看得见北墙的中段，够不着天花。
# 100 度才把天花与地板一起框进来——"room 机位天花仍在"这条断言要它才立得住。
ROOM_WIDE_FOV_DEG = 100.0

# 自动取景那台的张角：室内机位固定平视、眼高 1550 毫米，地板只从 眼高 ÷ tan(竖直半张角) 以外
# 才进画面——默认 55 度是 2980 毫米，这间 4000 毫米长的房只剩一条地板，过不了"拍到了地板"
# 那条门槛（ROOM_VIEW_MIN_TARGET_FLOOR_RATIO）。80 度把这个距离拉到 1850 毫米。
ROOM_AUTO_FOV_DEG = 80.0


def _quad_mesh(
    mesh_id: str,
    semantic: str,
    material_id: str,
    corners: list[tuple[float, float, float]],
    room: str | None = ROOM_NAME,
) -> Mesh:
    """四个角 → 两个三角形的一块面。"""
    return Mesh(
        id=mesh_id,
        semantic=semantic,  # type: ignore[arg-type]
        material_id=material_id,
        room=room,
        vertices=corners,
        triangles=[(0, 1, 2), (0, 2, 3)],
    )


def _box_mesh(mesh_id: str, material_id: str) -> Mesh:
    """家具体块：8 顶点 12 三角形。绕序刻意不统一——上游不保证，底渲也不许依赖它。"""
    x0, x1 = BOX_X_MM
    y0, y1 = BOX_Y_MM
    z0, z1 = 0.0, BOX_TOP_Z_MM
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
        semantic="furnishing",
        material_id=material_id,
        room=ROOM_NAME,
        vertices=vertices,
        triangles=triangles,
    )


def _make_scene() -> ScenePackage:
    width_mm, depth_mm, height_mm = ROOM_WIDTH_MM, ROOM_DEPTH_MM, ROOM_HEIGHT_MM
    meshes = [
        _quad_mesh(
            "floor-客厅",
            "floor",
            "mat-floor",
            [
                (0.0, 0.0, 0.0),
                (width_mm, 0.0, 0.0),
                (width_mm, depth_mm, 0.0),
                (0.0, depth_mm, 0.0),
            ],
        ),
        _quad_mesh(
            "wall-南",
            "wall",
            "mat-wall",
            [
                (0.0, 0.0, 0.0),
                (width_mm, 0.0, 0.0),
                (width_mm, 0.0, height_mm),
                (0.0, 0.0, height_mm),
            ],
        ),
        _quad_mesh(
            "wall-北",
            "wall",
            "mat-wall",
            [
                (0.0, depth_mm, 0.0),
                (width_mm, depth_mm, 0.0),
                (width_mm, depth_mm, height_mm),
                (0.0, depth_mm, height_mm),
            ],
        ),
        _quad_mesh(
            "wall-西",
            "wall",
            "mat-wall",
            [
                (0.0, 0.0, 0.0),
                (0.0, depth_mm, 0.0),
                (0.0, depth_mm, height_mm),
                (0.0, 0.0, height_mm),
            ],
        ),
        _quad_mesh(
            "wall-东",
            "wall",
            "mat-wall",
            [
                (width_mm, 0.0, 0.0),
                (width_mm, depth_mm, 0.0),
                (width_mm, depth_mm, height_mm),
                (width_mm, 0.0, height_mm),
            ],
        ),
        _quad_mesh(
            "ceiling-客厅",
            "ceiling",
            "mat-ceiling",
            [
                (0.0, 0.0, height_mm),
                (width_mm, 0.0, height_mm),
                (width_mm, depth_mm, height_mm),
                (0.0, depth_mm, height_mm),
            ],
        ),
        _box_mesh("furnishing-边柜", "mat-furnishing"),
    ]
    return ScenePackage(
        revision_id="rev-test-底渲",
        meshes=meshes,
        materials=[
            SurfaceMaterial(id="mat-floor", base_color_hex="#B58A5A"),
            SurfaceMaterial(id="mat-wall", base_color_hex="#EDE7DC"),
            SurfaceMaterial(id="mat-ceiling", base_color_hex="#F5F3EE"),
            SurfaceMaterial(id="mat-furnishing", base_color_hex="#7A6A58"),
        ],
        cameras=[
            CameraSpec(
                id=BIRD_CAMERA_ID, kind="bird", yaw_deg=0.0, pitch_deg=BIRD_PITCH_DEG, fov_deg=55.0
            ),
            CameraSpec(
                id=ROOM_CAMERA_ID,
                kind="room",
                room=ROOM_NAME,
                eye_height_mm=1550,
                yaw_deg=0.0,
                fov_deg=60.0,
            ),
            CameraSpec(
                id=ROOM_WIDE_CAMERA_ID,
                kind="room",
                room=ROOM_NAME,
                eye_height_mm=1550,
                yaw_deg=0.0,
                fov_deg=ROOM_WIDE_FOV_DEG,
            ),
            # 不给 yaw_deg：验自动取景那条路（camera.model_fields_set 里没有它）。
            CameraSpec(
                id=ROOM_AUTO_CAMERA_ID,
                kind="room",
                room=ROOM_NAME,
                eye_height_mm=1550,
                fov_deg=ROOM_AUTO_FOV_DEG,
            ),
            CameraSpec(id="room-卧室", kind="room", room="卧室"),
        ],
        bounds_min_mm=(0.0, 0.0, 0.0),
        bounds_max_mm=(width_mm, depth_mm, height_mm),
        mm_per_unit=1.0,
        floor_area_sqm=width_mm * depth_mm,
        triangle_count=24,
    )


def _open_gray(png: bytes) -> npt.NDArray[np.int64]:
    return np.asarray(Image.open(io.BytesIO(png)), dtype=np.int64)


def _project_pixel(
    point_mm: tuple[float, float, float],
    view_matrix: npt.NDArray[np.float64],
    proj_matrix: npt.NDArray[np.float64],
) -> tuple[int, int]:
    """世界点 → 像素下标。用的是底渲真正在用的那两个矩阵，取景算法不在测试里复制一份。"""
    clip = proj_matrix @ (view_matrix @ np.array([*point_mm, 1.0], dtype=np.float64))
    ndc = clip[:3] / clip[3]
    x_px = int((ndc[0] + 1.0) * 0.5 * WIDTH_PX)
    y_px = int((1.0 - ndc[1]) * 0.5 * HEIGHT_PX)
    return x_px, y_px


def _decode_depth_mm(views: BaseRenderViews, depth_u16: int) -> float:
    """按 :func:`_encode_depth_png` docstring 里公布的公式还原米数（近亮远暗，0 是背景）。"""
    assert depth_u16 >= 1, "背景像素没有深度可还原"
    brightness_ratio = (depth_u16 - 1) / 65534.0
    return views.near_mm + (1.0 - brightness_ratio) * (views.far_mm - views.near_mm)


@pytest.mark.parametrize("camera_id", [BIRD_CAMERA_ID, ROOM_WIDE_CAMERA_ID])
def test_渲两次逐字节相同(camera_id: str) -> None:
    """确定性：同一份场景包渲两次，五张 PNG 的字节完全相同。

    两台相机都跑：几何那一路的观感这一批（超采样、环境光遮蔽、家具投影）在两种机位下
    走的采样与查表都不一样，只验一台等于只验了其中一条路。**这条断言就是"不许随机数"
    那条红线的执行形态**——采样核换成 `random` 立刻红。
    """
    first = render_base_views(_make_scene(), camera_id, WIDTH_PX, HEIGHT_PX)
    second = render_base_views(_make_scene(), camera_id, WIDTH_PX, HEIGHT_PX)
    assert first.geometry_png == second.geometry_png
    assert first.depth_png == second.depth_png
    assert first.line_png == second.line_png
    assert first.mask_png == second.mask_png
    assert first.sketch_png == second.sketch_png
    assert first.covered_pixel_ratio == second.covered_pixel_ratio
    assert (first.near_mm, first.far_mm) == (second.near_mm, second.far_mm)


# 观感那一节的每个常量各挑一个"改了肯定看得出来"的值。**这张表的意义在于逐条证明
# 它们波及不到深度/线稿/遮罩**——那三路是被下游当数据读的（深度是米、遮罩是索引、
# 线稿是几何事实边），观感这边任何改动漏到那三路上，读出来的就是个假值。
APPEARANCE_KNOBS: list[tuple[str, Any]] = [
    ("GEOMETRY_SUPERSAMPLE_FACTOR", 3),
    ("GEOMETRY_BACKGROUND_RGB_U8", (200, 40, 40)),
    ("KEY_LIGHT_RGB", (1.0, 0.0, 0.0)),
    ("AMBIENT_SKY_RGB", (0.9, 0.9, 0.9)),
    ("SSAO_STRENGTH_RATIO", 0.0),
    ("SSAO_RADIUS_MM", 2.0),
    ("SHADOW_CASTER_SEMANTICS", frozenset()),
    ("SHADOW_MAP_PX", 256),
]


@pytest.mark.parametrize(("knob", "value"), APPEARANCE_KNOBS)
def test_观感常量只动得了几何那一路(knob: str, value: Any, monkeypatch: Any) -> None:
    """**红线守门**：改任意一个观感常量，几何图必须变，另外三路必须逐字节不变。

    为什么这条要按常量逐个来跑，而不是只验一次"三路没变"：三路挨着的路径不止一条
    （超采样是另渲一遍、遮蔽读的是 1 倍缓冲、投影另起一张深度图），每一条漏过去的
    形态都不一样。逐个改、逐个断，才说得清是哪一条串了台。

    自证数（`covered_pixel_ratio` / `near_mm` / `far_mm`）一并断：它们同样是数据不是观感，
    从 1 倍那次光栅来，不该被超采样那一遍碰到。
    """
    scene = _make_scene()
    baseline = render_base_views(scene, BIRD_CAMERA_ID, WIDTH_PX, HEIGHT_PX)
    monkeypatch.setattr(base_render, knob, value)
    tweaked = render_base_views(scene, BIRD_CAMERA_ID, WIDTH_PX, HEIGHT_PX)

    assert tweaked.geometry_png != baseline.geometry_png, f"{knob} 改了却没影响几何图"
    assert tweaked.depth_png == baseline.depth_png, f"{knob} 漏到深度那一路了"
    assert tweaked.line_png == baseline.line_png, f"{knob} 漏到线稿那一路了"
    assert tweaked.mask_png == baseline.mask_png, f"{knob} 漏到遮罩那一路了"
    assert tweaked.sketch_png == baseline.sketch_png, f"{knob} 漏到控制稿那一路了"
    assert tweaked.covered_pixel_ratio == baseline.covered_pixel_ratio
    assert (tweaked.near_mm, tweaked.far_mm) == (baseline.near_mm, baseline.far_mm)
    assert [entry.model_dump() for entry in tweaked.mask_index] == [
        entry.model_dump() for entry in baseline.mask_index
    ]


def test_三路数据图的背景恒为零而几何那一路不是() -> None:
    """深度/遮罩两路——逐像素答"有没有东西"——背景处处严格是 0；几何那一路是有意的例外
    （模块 docstring 写着）。

    线稿不进这条断言：轮廓线按 `_encode_line_png` 的约定标在**相邻像素对里下标小的
    那一侧**，轮廓另一侧恰好是背景时线就画到了背景像素上——这是几何事实边的正当形态，
    不是三路互相对不上（那条一致性只挑 mask/depth 来断，见 `test_四路覆盖像素数一致`，
    同一个理由）。这里只挑画幅角上离任何轮廓最远的一点验线稿，不对整块背景下断言。
    """
    views = render_base_views(_make_scene(), BIRD_CAMERA_ID, WIDTH_PX, HEIGHT_PX)
    mask = _open_gray(views.mask_png)
    background = mask == 0
    assert bool(background.any()), "bird 机位框住整户，四周本来就该有背景"

    assert int(np.count_nonzero(_open_gray(views.depth_png)[background])) == 0
    assert _open_gray(views.line_png)[0, 0] == 0, "画幅角上离几何最远，不该有轮廓线画到这儿"

    geometry = np.asarray(Image.open(io.BytesIO(views.geometry_png)))
    corner_rgb = tuple(int(channel) for channel in geometry[0, 0])
    assert corner_rgb == base_render.GEOMETRY_BACKGROUND_RGB_U8
    assert corner_rgb != (0, 0, 0), "纯黑那条口径已经退掉了，退回去说明改错了地方"


def test_四路覆盖像素数一致() -> None:
    """遮罩的非背景像素数 == 深度有值的像素数 == covered_pixel_ratio × 总像素。"""
    views = render_base_views(_make_scene(), BIRD_CAMERA_ID, WIDTH_PX, HEIGHT_PX)
    mask = _open_gray(views.mask_png)
    depth = _open_gray(views.depth_png)
    geometry = np.asarray(Image.open(io.BytesIO(views.geometry_png)))

    mask_covered_px = int(np.count_nonzero(mask))
    depth_covered_px = int(np.count_nonzero(depth >= 1))
    ratio_covered_px = views.covered_pixel_ratio * WIDTH_PX * HEIGHT_PX

    assert mask.shape == (HEIGHT_PX, WIDTH_PX)
    assert geometry.shape == (HEIGHT_PX, WIDTH_PX, 3)
    assert mask_covered_px == depth_covered_px
    assert mask_covered_px == pytest.approx(ratio_covered_px)
    assert 0.0 < views.covered_pixel_ratio <= 1.0


def test_遮罩索引表与图对得上() -> None:
    """表里的 index 集合 == 图里的非零值集合；pixel_count 之和 == 覆盖像素数。"""
    views = render_base_views(_make_scene(), BIRD_CAMERA_ID, WIDTH_PX, HEIGHT_PX)
    mask = _open_gray(views.mask_png)

    in_image = {int(value) for value in np.unique(mask) if value != 0}
    in_table = {entry.index for entry in views.mask_index}
    assert in_image == in_table
    assert len(in_table) == len(views.mask_index), "索引不许重复"

    covered_px = int(np.count_nonzero(mask))
    assert sum(entry.pixel_count for entry in views.mask_index) == covered_px
    for entry in views.mask_index:
        assert entry.pixel_count == int(np.count_nonzero(mask == entry.index))
    assert {entry.mesh_id for entry in views.mask_index} <= {
        mesh.id for mesh in _make_scene().meshes
    }


def test_地板中心比家具顶面更远() -> None:
    """深度合理：bird 机位下，地板中心的深度**大于**离相机更近的家具顶面深度。

    近亮远暗，所以同一条断言在 16 位灰度上是"地板中心更暗"。
    """
    scene = _make_scene()
    views = render_base_views(scene, BIRD_CAMERA_ID, WIDTH_PX, HEIGHT_PX)
    pose = resolve_camera_pose(scene, BIRD_CAMERA_ID, WIDTH_PX / HEIGHT_PX)
    view_matrix = look_at_matrix(pose.eye_mm, pose.target_mm, pose.up_hint_xyz)
    proj_matrix = perspective_matrix(
        pose.fov_deg, WIDTH_PX / HEIGHT_PX, pose.near_clip_mm, pose.far_clip_mm
    )

    floor_x, floor_y = _project_pixel(FLOOR_CENTER_MM, view_matrix, proj_matrix)
    box_x, box_y = _project_pixel(BOX_TOP_CENTER_MM, view_matrix, proj_matrix)

    mask = _open_gray(views.mask_png)
    by_index = {entry.index: entry.mesh_id for entry in views.mask_index}
    assert by_index[int(mask[floor_y, floor_x])] == "floor-客厅"
    assert by_index[int(mask[box_y, box_x])] == "furnishing-边柜"

    depth = _open_gray(views.depth_png)
    floor_depth_mm = _decode_depth_mm(views, int(depth[floor_y, floor_x]))
    box_depth_mm = _decode_depth_mm(views, int(depth[box_y, box_x]))
    assert floor_depth_mm > box_depth_mm
    assert depth[floor_y, floor_x] < depth[box_y, box_x], "近亮远暗：更远的地板要更暗"
    # 两点 y 相同、只差 800 毫米层高，俯角 60 度 → 深度差 = 800 × sin(60°) ≈ 693 毫米。
    assert floor_depth_mm - box_depth_mm == pytest.approx(
        BOX_TOP_Z_MM * np.sin(np.radians(-BIRD_PITCH_DEG)), abs=20.0
    )
    assert views.near_mm < views.far_mm


def test_相机id找不到就抛错() -> None:
    """失败响亮：不存在的 camera_id 抛 BaseRenderError，**不退化成默认相机**。"""
    scene = _make_scene()
    with pytest.raises(BaseRenderError, match="没有这台相机"):
        render_base_views(scene, "bird-不存在", WIDTH_PX, HEIGHT_PX)
    with pytest.raises(BaseRenderError, match="这间房没有地板"):
        render_base_views(scene, "room-卧室", WIDTH_PX, HEIGHT_PX)


def test_室内机位退到房间边缘朝内容看() -> None:
    """室内取景（观感提档 2026-09-01）：机位不再站在地板质心——那正是屋子正中，四面
    都是墙，站那儿只会看见一堵墙加半个柜子。改成退到镜头背后留出房间进深的那一侧，
    画面不再被一堵墙铺满，地板与家具都要露出来。

    这台相机的 ``yawDeg`` 是上游显式给的（0 度，朝北）：即便这间房的家具质心偏在东侧
    （见 test_室内机位没给yaw时自动看向家具），退景朝向也只认这个显式值，不被自动取景
    盖过去——验的正是"上游给了就听上游的"那条（``CameraSpec.yaw_deg`` 在
    ``model_fields_set`` 里）。
    """
    scene = _make_scene()
    pose = resolve_camera_pose(scene, ROOM_CAMERA_ID, WIDTH_PX / HEIGHT_PX)

    floor_centroid_x, floor_centroid_y = 2000.0, 1500.0
    assert pose.eye_mm[0] == pytest.approx(floor_centroid_x), "yaw 朝正北，退景不该带偏 x"
    assert pose.eye_mm[1] < floor_centroid_y, "没有退，还站在地板质心上"
    # 退景边界是沿射线定步长扫出来的（见 _room_retreat_distance_mm），落点比墙距差一个
    # 采样步长都算数：这里只验"确实退到了墙距附近"，不钉死浮点意义上的相等。
    assert pose.eye_mm[1] == pytest.approx(base_render.ROOM_EYE_WALL_MARGIN_MM, abs=20.0)
    assert pose.target_mm[0] == pytest.approx(pose.eye_mm[0])
    assert pose.target_mm[1] > pose.eye_mm[1], "上游给的是朝北（+y），没被自动取景扭到别的方向"
    # 上游给了 yaw 就不做候选评估：只量一次、不判（2026-09-05 取景规则改动后的口径）
    assert pose.room_view is not None
    assert pose.room_view.candidate_count == 1
    assert pose.room_view.eye_mm == pose.eye_mm

    views = render_base_views(scene, ROOM_CAMERA_ID, WIDTH_PX, HEIGHT_PX)
    assert views.camera_id == ROOM_CAMERA_ID
    assert views.room_view == pose.room_view
    semantics = {entry.semantic for entry in views.mask_index}
    assert {"wall", "furnishing"} <= semantics, "退到边缘朝内看，墙与家具都该露出来"
    # 房间四面都是封闭的墙+天花，画面本来就不会露出背景（covered_pixel_ratio 恒为
    # 1.0，这不是这条改动要验的事）；退没退开看的是网格种类——改前正对着贴身的墙，
    # 画面只有 wall-北 一块；改后退出了空间，家具与旁边的墙都该一起入画。
    assert len(views.mask_index) > 1, "改前只有一堵墙填满画面，改后不该只剩一块网格"


def test_室内机位没给yaw时按候选评估取景() -> None:
    """没给 ``yawDeg``（不在 ``model_fields_set`` 里）：走候选评估（2026-09-05，取代
    "自动看向家具质心"那版——那版只定朝向不看画面，小房间里退到贴着墙照样出图）。
    选出来的位姿必须达标：不贴墙、拍到了地板、主体是这间房；机位在房间里；评估的候选
    不止一个；渲出来的自证数与位姿上带的是同一份。
    """
    scene = _make_scene()
    pose = resolve_camera_pose(scene, ROOM_AUTO_CAMERA_ID, WIDTH_PX / HEIGHT_PX)

    check = pose.room_view
    assert check is not None
    assert check.passed
    assert check.candidate_count > 1
    assert check.min_depth_mm >= base_render.ROOM_EYE_WALL_MARGIN_MM
    assert check.target_floor_ratio >= base_render.ROOM_VIEW_MIN_TARGET_FLOOR_RATIO
    assert check.dominance_ratio >= base_render.ROOM_VIEW_MIN_DOMINANCE_RATIO
    assert 0.0 < pose.eye_mm[0] < ROOM_WIDTH_MM and 0.0 < pose.eye_mm[1] < ROOM_DEPTH_MM, (
        "机位退出房间外面去了"
    )

    views = render_base_views(scene, ROOM_AUTO_CAMERA_ID, WIDTH_PX, HEIGHT_PX)
    assert views.room_view == check
    semantics = {entry.semantic for entry in views.mask_index}
    assert {"floor", "wall"} <= semantics, "达标的位姿至少该同时看见地板和墙"


def test_近平面裁剪保住穿过相机的大面() -> None:
    """室内软光栅最容易出的那个错：横跨相机的大面必须留下**镜头前那一半**。

    造一块 10m × 10m 的地板，相机站在正中央——两个三角形都从相机背后一直伸到镜头前方，
    每一个都被近平面切开。裁剪写对了，画面下缘看得见地板，最近处落在
    ``眼高 / tan(竖直半张角)`` ≈ 2700 毫米；裁漏了（把整块面当"有顶点在背后"丢掉），
    地板就整块消失，下缘只剩背景——所以这条断言真的抓得住这个错。
    """
    span_mm = 5000.0
    eye_height_mm = 1550
    fov_deg = 60.0
    scene = ScenePackage(
        revision_id="rev-test-近裁剪",
        meshes=[
            _quad_mesh(
                "floor-大厅",
                "floor",
                "mat-floor",
                [
                    (-span_mm, -span_mm, 0.0),
                    (span_mm, -span_mm, 0.0),
                    (span_mm, span_mm, 0.0),
                    (-span_mm, span_mm, 0.0),
                ],
                room="大厅",
            ),
            _quad_mesh(
                "wall-远",
                "wall",
                "mat-wall",
                [
                    (-span_mm, span_mm, 0.0),
                    (span_mm, span_mm, 0.0),
                    (span_mm, span_mm, ROOM_HEIGHT_MM),
                    (-span_mm, span_mm, ROOM_HEIGHT_MM),
                ],
                room="大厅",
            ),
        ],
        materials=[
            SurfaceMaterial(id="mat-floor", base_color_hex="#B58A5A"),
            SurfaceMaterial(id="mat-wall", base_color_hex="#EDE7DC"),
        ],
        cameras=[
            CameraSpec(
                id="room-大厅",
                kind="room",
                room="大厅",
                eye_height_mm=eye_height_mm,
                yaw_deg=0.0,
                fov_deg=fov_deg,
            )
        ],
    )
    views = render_base_views(scene, "room-大厅", WIDTH_PX, HEIGHT_PX)

    by_mesh = {entry.mesh_id: entry.pixel_count for entry in views.mask_index}
    assert "floor-大厅" in by_mesh, "横跨近平面的地板被整块丢掉了——近裁剪没做或做漏了"
    assert by_mesh["floor-大厅"] > 0.1 * WIDTH_PX * HEIGHT_PX

    nearest_floor_mm = eye_height_mm / np.tan(np.radians(fov_deg) / 2.0)
    assert views.near_mm == pytest.approx(nearest_floor_mm, rel=0.02)


def test_bird机位剔掉天花() -> None:
    """裁决 2026-08-31：**bird 机位渲染时剔掉 ceiling，不加剖切面**。

    bird 的用途是从上往下看布局，天花挡在中间一点信息都不带；剖切面要多一个"切在哪个
    高度"的参数，今天没有依据能定（《纪律·阈值有数据才定》），剔天花是零参数的做法。

    剔的是**这一个机位的渲染**不是场景包——所以同一份包换 room 机位，天花必须还在。
    """
    scene = _make_scene()
    bird = render_base_views(scene, BIRD_CAMERA_ID, WIDTH_PX, HEIGHT_PX)
    by_semantic = {entry.semantic for entry in bird.mask_index}
    assert "ceiling" not in by_semantic, "bird 机位不该看得见天花"
    assert {"floor", "wall", "furnishing"} <= by_semantic, "剔掉天花就该看得见屋里的布局"
    assert "ceiling-客厅" not in {entry.mesh_id for entry in bird.mask_index}

    # 剔掉的网格是**根本没进 z-test**，不是画完被盖住：天花所占的索引整个缺席，
    # 其余网格的索引一个都不许挪位（索引恒等于网格在场景包里的下标 +1）。
    ceiling_index = [m.id for m in scene.meshes].index("ceiling-客厅") + 1
    mask = _open_gray(bird.mask_png)
    assert int(np.count_nonzero(mask == ceiling_index)) == 0
    for entry in bird.mask_index:
        assert scene.meshes[entry.index - 1].id == entry.mesh_id

    room = render_base_views(scene, ROOM_WIDE_CAMERA_ID, WIDTH_PX, HEIGHT_PX)
    room_semantics = {entry.semantic for entry in room.mask_index}
    assert "ceiling" in room_semantics, "room 机位站在屋里，天花本来就该看得见"
    assert "floor" in room_semantics
