"""室内机位取景规则（2026-09-05：**不许对墙**）的守门测试：小房间没有一个达标位姿就响亮
失败、上游显式给 yaw 时只量不判、两间房里选出来的位姿主体是目标房间、候选评估确定性。

场景在测试里现造（同 test_base_render 的理由）。规则本身写在 base_render
``resolve_camera_pose`` 的 docstring，判据的数在 ``ROOM_VIEW_*`` 常量上，这里只验它。
"""

from __future__ import annotations

import numpy as np
import pytest

from render3d_worker import base_render
from render3d_worker.base_render import BaseRenderError, render_base_views, resolve_camera_pose
from render3d_worker.models import CameraSpec, Mesh, ScenePackage, SurfaceMaterial

WIDTH_PX = 320
HEIGHT_PX = 240
ASPECT_RATIO = WIDTH_PX / HEIGHT_PX
CEILING_M = 2.8
WALL_THICKNESS_M = 0.2
MATERIAL_ID = "mat"

CLOSET_NAME = "储物间"
CLOSET_SIZE_M = 1.0
"""1 米见方的封闭小间：平视、眼高 1.55 米时地板进不了画面，怎么站都是墙面图。"""

TARGET_ROOM = "书房"
OTHER_ROOM = "客厅"
TARGET_X_M = (0.0, 3.0)
OTHER_X_M = (3.0 + WALL_THICKNESS_M, 9.0)
ROOM_Y_M = (0.0, 3.0)
PARTITION_X_M = (3.0, 3.0 + WALL_THICKNESS_M)
DOORWAY_Y_M = (1.0, 2.0)
DOOR_TOP_M = 2.05

AUTO_CAMERA_ID = "room-自动"
EXPLICIT_CAMERA_ID = "room-显式"
FOV_DEG = 80.0


def _quad(
    mesh_id: str, semantic: str, corners: list[tuple[float, float, float]], room: str | None
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
    x_m: tuple[float, float],
    y_m: tuple[float, float],
    z_m: tuple[float, float],
) -> Mesh:
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
        id=mesh_id, semantic="wall", material_id=MATERIAL_ID, vertices=vertices, triangles=triangles
    )


def _floor_and_ceiling(room: str, x_m: tuple[float, float], y_m: tuple[float, float]) -> list[Mesh]:
    x0, x1 = x_m
    y0, y1 = y_m
    return [
        _quad(
            f"floor:{room}:0",
            "floor",
            [(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)],
            room,
        ),
        _quad(
            f"ceiling:{room}:0",
            "ceiling",
            [(x0, y0, CEILING_M), (x1, y0, CEILING_M), (x1, y1, CEILING_M), (x0, y1, CEILING_M)],
            room,
        ),
    ]


def _outer_walls(x_m: tuple[float, float], y_m: tuple[float, float]) -> list[Mesh]:
    x0, x1 = x_m
    y0, y1 = y_m
    t = WALL_THICKNESS_M
    z = (0.0, CEILING_M)
    return [
        _box("wall:south", (x0 - t, x1 + t), (y0 - t, y0), z),
        _box("wall:north", (x0 - t, x1 + t), (y1, y1 + t), z),
        _box("wall:west", (x0 - t, x0), (y0, y1), z),
        _box("wall:east", (x1, x1 + t), (y0, y1), z),
    ]


def _scene(meshes: list[Mesh], cameras: list[CameraSpec]) -> ScenePackage:
    return ScenePackage(
        revision_id="rev-test-取景",
        meshes=meshes,
        materials=[SurfaceMaterial(id=MATERIAL_ID, base_color_hex="#D0D0D0")],
        cameras=cameras,
    )


def _closet_scene() -> ScenePackage:
    size = (0.0, CLOSET_SIZE_M)
    return _scene(
        [*_floor_and_ceiling(CLOSET_NAME, size, size), *_outer_walls(size, size)],
        [
            CameraSpec(id=AUTO_CAMERA_ID, kind="room", room=CLOSET_NAME, fov_deg=FOV_DEG),
            CameraSpec(
                id=EXPLICIT_CAMERA_ID, kind="room", room=CLOSET_NAME, yaw_deg=0.0, fov_deg=FOV_DEG
            ),
        ],
    )


def _two_room_scene() -> ScenePackage:
    """目标房间 3×3 米，隔一道带门洞的墙连着 5.8×3 米的大房间。"""
    z = (0.0, CEILING_M)
    meshes = [
        *_floor_and_ceiling(TARGET_ROOM, TARGET_X_M, ROOM_Y_M),
        *_floor_and_ceiling(OTHER_ROOM, OTHER_X_M, ROOM_Y_M),
        *_outer_walls((TARGET_X_M[0], OTHER_X_M[1]), ROOM_Y_M),
        _box("wall:partition:span:0", PARTITION_X_M, (ROOM_Y_M[0], DOORWAY_Y_M[0]), z),
        _box("wall:partition:lintel", PARTITION_X_M, DOORWAY_Y_M, (DOOR_TOP_M, CEILING_M)),
        _box("wall:partition:span:1", PARTITION_X_M, (DOORWAY_Y_M[1], ROOM_Y_M[1]), z),
    ]
    return _scene(
        meshes, [CameraSpec(id=AUTO_CAMERA_ID, kind="room", room=TARGET_ROOM, fov_deg=FOV_DEG)]
    )


def test_小房间没有达标位姿就响亮失败() -> None:
    """1 米见方的封闭小间：地板进不了画面（平视、眼高 1.55 米，地板只从
    1.55 ÷ tan(40°) ≈ 1.85 米以外才进画面），没有一个候选达标——抛错，报房间名与最接近的
    那个候选的数，**不出一张墙面图**。"""
    with pytest.raises(BaseRenderError, match=f"房间 {CLOSET_NAME} 无法取景") as caught:
        resolve_camera_pose(_closet_scene(), AUTO_CAMERA_ID, ASPECT_RATIO)
    message = str(caught.value)
    assert "地板占比" in message and "最近深度" in message, "失败要说出原因与最接近的那个候选"


def test_上游显式给yaw只量不判() -> None:
    """同一个小间，上游显式给了 yaw：照渲，取景自证数量出来（不达标），候选数记 1。"""
    views = render_base_views(_closet_scene(), EXPLICIT_CAMERA_ID, WIDTH_PX, HEIGHT_PX)
    check = views.room_view
    assert check is not None
    assert check.candidate_count == 1
    assert not check.passed
    assert check.target_floor_ratio < base_render.ROOM_VIEW_MIN_TARGET_FLOOR_RATIO


def test_两间房里选出来的位姿主体是目标房间() -> None:
    """自动取景选出来的位姿：站在目标房间里、达标、目标房间在有房间归属的像素里占大头，
    别的房间的地板与天花即便透过门洞看得见也只是零头。"""
    scene = _two_room_scene()
    pose = resolve_camera_pose(scene, AUTO_CAMERA_ID, ASPECT_RATIO)
    check = pose.room_view
    assert check is not None
    assert check.passed
    assert TARGET_X_M[0] < pose.eye_m[0] < TARGET_X_M[1]
    assert ROOM_Y_M[0] < pose.eye_m[1] < ROOM_Y_M[1]
    assert check.dominance_ratio >= base_render.ROOM_VIEW_MIN_DOMINANCE_RATIO
    assert check.other_room_ratio < check.target_room_ratio
    assert check.min_depth_m >= base_render.ROOM_EYE_WALL_MARGIN_M

    views = render_base_views(scene, AUTO_CAMERA_ID, WIDTH_PX, HEIGHT_PX)
    by_room = {entry.room for entry in views.mask_index if entry.semantic == "floor"}
    assert TARGET_ROOM in by_room, "最终那张图上得看得见目标房间的地板"


def test_候选评估确定性() -> None:
    """同一份场景包解两次位姿，位姿与自证数逐字相同；候选清单次序写死、起点排第一。"""
    first = resolve_camera_pose(_two_room_scene(), AUTO_CAMERA_ID, ASPECT_RATIO)
    second = resolve_camera_pose(_two_room_scene(), AUTO_CAMERA_ID, ASPECT_RATIO)
    assert first == second

    scene = _two_room_scene()
    floor_xy = base_render._room_floor_triangles_xy_m(scene, TARGET_ROOM)
    start_xy = base_render._room_view_start_xy_m(scene, TARGET_ROOM, floor_xy)
    poses = base_render._room_view_candidate_poses(
        floor_xy, base_render._room_furnishing_triangles_xy_m(scene, TARGET_ROOM), start_xy
    )
    assert poses, "至少有起点上的候选"
    assert np.allclose(poses[0][0], start_xy)
    assert len({(round(float(eye[0]), 4), round(float(eye[1]), 4), yaw) for eye, yaw in poses}) == (
        len(poses)
    ), "候选位姿不许重复"
    assert all(TARGET_X_M[0] <= eye[0] <= TARGET_X_M[1] for eye, _ in poses), (
        "候选位置都在目标房间里"
    )
