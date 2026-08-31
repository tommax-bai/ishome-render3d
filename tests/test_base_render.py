"""底渲四路输出的守门测试：确定性、四路一致、遮罩对得上、深度合理、失败响亮，
外加三条口径（室内机位、近平面裁剪、bird 剔天花）。

场景包**在测试里现造**，不读 `tests/fixtures/`：这几条要证的是"给同一份网格，底渲的行为
是定死的"，与户型图怎么编成场景包无关。夹在一起，编包那边改一个三角化顺序就会把底渲的
测试连坐弄红，查错的人要先排除一整条不相干的线。**本文件一处都不钉网格块数**——墙体怎么
分块是 mesh 那一路的事，它改了不该连坐这里。
"""

from __future__ import annotations

import io

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

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

# 一间 4m × 3m × 2.8m 的客厅：地板 + 四面墙 + 天花 + 一件 1.0×1.0×0.8m 的家具体块。
# 天花是**必须有**的：bird 剔天花那条裁决，没有天花就验不到（见 test_bird机位剔掉天花）。
ROOM_NAME = "客厅"
ROOM_WIDTH_M = 4.0
ROOM_DEPTH_M = 3.0
ROOM_HEIGHT_M = 2.8
BOX_X_M = (2.5, 3.5)
BOX_Y_M = (1.0, 2.0)
BOX_TOP_Z_M = 0.8

FLOOR_CENTER_M = (2.0, 1.5, 0.0)
BOX_TOP_CENTER_M = (3.0, 1.5, BOX_TOP_Z_M)

WIDTH_PX = 320
HEIGHT_PX = 240

BIRD_CAMERA_ID = "bird-整户"
ROOM_CAMERA_ID = "room-客厅"
ROOM_WIDE_CAMERA_ID = "room-客厅-广角"

# bird 机位取 -60 度而不是默认的 -30：这间房四面墙齐全，-30 度时视线会从南墙的**外侧**
# 擦过去，画面里只有一面墙背。-60 度越过墙顶看进屋里，才验得到室内的深度关系。
BIRD_PITCH_DEG = -60.0

# 广角室内机位：普通那台（竖直 60 度）在 1.5 米外只看得见北墙的中段，够不着天花。
# 100 度才把天花与地板一起框进来——"room 机位天花仍在"这条断言要它才立得住。
ROOM_WIDE_FOV_DEG = 100.0


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
    x0, x1 = BOX_X_M
    y0, y1 = BOX_Y_M
    z0, z1 = 0.0, BOX_TOP_Z_M
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
    width_m, depth_m, height_m = ROOM_WIDTH_M, ROOM_DEPTH_M, ROOM_HEIGHT_M
    meshes = [
        _quad_mesh(
            "floor-客厅",
            "floor",
            "mat-floor",
            [(0.0, 0.0, 0.0), (width_m, 0.0, 0.0), (width_m, depth_m, 0.0), (0.0, depth_m, 0.0)],
        ),
        _quad_mesh(
            "wall-南",
            "wall",
            "mat-wall",
            [
                (0.0, 0.0, 0.0),
                (width_m, 0.0, 0.0),
                (width_m, 0.0, height_m),
                (0.0, 0.0, height_m),
            ],
        ),
        _quad_mesh(
            "wall-北",
            "wall",
            "mat-wall",
            [
                (0.0, depth_m, 0.0),
                (width_m, depth_m, 0.0),
                (width_m, depth_m, height_m),
                (0.0, depth_m, height_m),
            ],
        ),
        _quad_mesh(
            "wall-西",
            "wall",
            "mat-wall",
            [
                (0.0, 0.0, 0.0),
                (0.0, depth_m, 0.0),
                (0.0, depth_m, height_m),
                (0.0, 0.0, height_m),
            ],
        ),
        _quad_mesh(
            "wall-东",
            "wall",
            "mat-wall",
            [
                (width_m, 0.0, 0.0),
                (width_m, depth_m, 0.0),
                (width_m, depth_m, height_m),
                (width_m, 0.0, height_m),
            ],
        ),
        _quad_mesh(
            "ceiling-客厅",
            "ceiling",
            "mat-ceiling",
            [
                (0.0, 0.0, height_m),
                (width_m, 0.0, height_m),
                (width_m, depth_m, height_m),
                (0.0, depth_m, height_m),
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
                eye_height_m=1.55,
                yaw_deg=0.0,
                fov_deg=60.0,
            ),
            CameraSpec(
                id=ROOM_WIDE_CAMERA_ID,
                kind="room",
                room=ROOM_NAME,
                eye_height_m=1.55,
                yaw_deg=0.0,
                fov_deg=ROOM_WIDE_FOV_DEG,
            ),
            CameraSpec(id="room-卧室", kind="room", room="卧室"),
        ],
        bounds_min_m=(0.0, 0.0, 0.0),
        bounds_max_m=(width_m, depth_m, height_m),
        metre_per_unit=1.0,
        floor_area_sqm=width_m * depth_m,
        triangle_count=24,
    )


def _open_gray(png: bytes) -> npt.NDArray[np.int64]:
    return np.asarray(Image.open(io.BytesIO(png)), dtype=np.int64)


def _project_pixel(
    point_m: tuple[float, float, float],
    view_matrix: npt.NDArray[np.float64],
    proj_matrix: npt.NDArray[np.float64],
) -> tuple[int, int]:
    """世界点 → 像素下标。用的是底渲真正在用的那两个矩阵，取景算法不在测试里复制一份。"""
    clip = proj_matrix @ (view_matrix @ np.array([*point_m, 1.0], dtype=np.float64))
    ndc = clip[:3] / clip[3]
    x_px = int((ndc[0] + 1.0) * 0.5 * WIDTH_PX)
    y_px = int((1.0 - ndc[1]) * 0.5 * HEIGHT_PX)
    return x_px, y_px


def _decode_depth_m(views: BaseRenderViews, depth_u16: int) -> float:
    """按 :func:`_encode_depth_png` docstring 里公布的公式还原米数（近亮远暗，0 是背景）。"""
    assert depth_u16 >= 1, "背景像素没有深度可还原"
    brightness_ratio = (depth_u16 - 1) / 65534.0
    return views.near_m + (1.0 - brightness_ratio) * (views.far_m - views.near_m)


def test_渲两次逐字节相同() -> None:
    """确定性：同一份场景包渲两次，四张 PNG 的字节完全相同。"""
    first = render_base_views(_make_scene(), BIRD_CAMERA_ID, WIDTH_PX, HEIGHT_PX)
    second = render_base_views(_make_scene(), BIRD_CAMERA_ID, WIDTH_PX, HEIGHT_PX)
    assert first.geometry_png == second.geometry_png
    assert first.depth_png == second.depth_png
    assert first.line_png == second.line_png
    assert first.mask_png == second.mask_png
    assert first.covered_pixel_ratio == second.covered_pixel_ratio
    assert (first.near_m, first.far_m) == (second.near_m, second.far_m)


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
    view_matrix = look_at_matrix(pose.eye_m, pose.target_m, pose.up_hint_xyz)
    proj_matrix = perspective_matrix(
        pose.fov_deg, WIDTH_PX / HEIGHT_PX, pose.near_clip_m, pose.far_clip_m
    )

    floor_x, floor_y = _project_pixel(FLOOR_CENTER_M, view_matrix, proj_matrix)
    box_x, box_y = _project_pixel(BOX_TOP_CENTER_M, view_matrix, proj_matrix)

    mask = _open_gray(views.mask_png)
    by_index = {entry.index: entry.mesh_id for entry in views.mask_index}
    assert by_index[int(mask[floor_y, floor_x])] == "floor-客厅"
    assert by_index[int(mask[box_y, box_x])] == "furnishing-边柜"

    depth = _open_gray(views.depth_png)
    floor_depth_m = _decode_depth_m(views, int(depth[floor_y, floor_x]))
    box_depth_m = _decode_depth_m(views, int(depth[box_y, box_x]))
    assert floor_depth_m > box_depth_m
    assert depth[floor_y, floor_x] < depth[box_y, box_x], "近亮远暗：更远的地板要更暗"
    # 两点 y 相同、只差 0.8 米层高，俯角 60 度 → 深度差 = 0.8 × sin(60°) ≈ 0.693 米。
    assert floor_depth_m - box_depth_m == pytest.approx(
        BOX_TOP_Z_M * np.sin(np.radians(-BIRD_PITCH_DEG)), abs=0.02
    )
    assert views.near_m < views.far_m


def test_相机id找不到就抛错() -> None:
    """失败响亮：不存在的 camera_id 抛 BaseRenderError，**不退化成默认相机**。"""
    scene = _make_scene()
    with pytest.raises(BaseRenderError, match="没有这台相机"):
        render_base_views(scene, "bird-不存在", WIDTH_PX, HEIGHT_PX)
    with pytest.raises(BaseRenderError, match="这间房没有地板"):
        render_base_views(scene, "room-卧室", WIDTH_PX, HEIGHT_PX)


def test_室内机位站进房间里() -> None:
    """room 机位站在地板质心、眼高平视：正对 1.5 米外的北墙，画面被它铺满。"""
    views = render_base_views(_make_scene(), ROOM_CAMERA_ID, WIDTH_PX, HEIGHT_PX)
    assert views.covered_pixel_ratio == 1.0
    assert [entry.mesh_id for entry in views.mask_index] == ["wall-北"]
    assert views.camera_id == ROOM_CAMERA_ID

    line = _open_gray(views.line_png)
    assert set(np.unique(line)) <= {0, 255}
    # 整面平墙、没有遮挡也没有折角：线稿该是干净的，一条假线都不该有。
    assert int(np.count_nonzero(line)) == 0


def test_近平面裁剪保住穿过相机的大面() -> None:
    """室内软光栅最容易出的那个错：横跨相机的大面必须留下**镜头前那一半**。

    造一块 10m × 10m 的地板，相机站在正中央——两个三角形都从相机背后一直伸到镜头前方，
    每一个都被近平面切开。裁剪写对了，画面下缘看得见地板，最近处落在
    ``眼高 / tan(竖直半张角)`` ≈ 2.7 米；裁漏了（把整块面当"有顶点在背后"丢掉），
    地板就整块消失，下缘只剩背景——所以这条断言真的抓得住这个错。
    """
    span_m = 5.0
    eye_height_m = 1.55
    fov_deg = 60.0
    scene = ScenePackage(
        revision_id="rev-test-近裁剪",
        meshes=[
            _quad_mesh(
                "floor-大厅",
                "floor",
                "mat-floor",
                [
                    (-span_m, -span_m, 0.0),
                    (span_m, -span_m, 0.0),
                    (span_m, span_m, 0.0),
                    (-span_m, span_m, 0.0),
                ],
                room="大厅",
            ),
            _quad_mesh(
                "wall-远",
                "wall",
                "mat-wall",
                [
                    (-span_m, span_m, 0.0),
                    (span_m, span_m, 0.0),
                    (span_m, span_m, ROOM_HEIGHT_M),
                    (-span_m, span_m, ROOM_HEIGHT_M),
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
                eye_height_m=eye_height_m,
                yaw_deg=0.0,
                fov_deg=fov_deg,
            )
        ],
    )
    views = render_base_views(scene, "room-大厅", WIDTH_PX, HEIGHT_PX)

    by_mesh = {entry.mesh_id: entry.pixel_count for entry in views.mask_index}
    assert "floor-大厅" in by_mesh, "横跨近平面的地板被整块丢掉了——近裁剪没做或做漏了"
    assert by_mesh["floor-大厅"] > 0.1 * WIDTH_PX * HEIGHT_PX

    nearest_floor_m = eye_height_m / np.tan(np.radians(fov_deg) / 2.0)
    assert views.near_m == pytest.approx(nearest_floor_m, rel=0.02)


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
