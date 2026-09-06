"""真户型 16 洞：``frame-handle`` 控制稿（书房 / 主卧 / 客厅 / 揭顶 + 入户门机位候选）+ ``frame-sill`` 揭顶与入户门对照
+ 三方案并排拼图。只在这一批用，写完不改。零模型调用。

跑法（仓根目录）：uv run python _iteration/run-2026-09-05-sketch-symbol-combo/渲控制稿与拼图.py

入户门机位不在机位包里：机位包的 ``room`` 相机要么自动取景（16 方向 × 16 朝向里挑，挑不到对着入户门的），要么显式给
yaw——显式给 yaw 那条路从**整间房**的面积质心退（``_room_floor_anchor_m``），"次卧"是两块不相连的地板，质心落在客厅里，
退不动，机位站在客厅——本脚本把这条路算出来只记录不用。本脚本的入户门机位**按几何定**：站在入户门（洞口表 #10，
``entry-door``）室内那一侧、门的垂直平分线上，眼高 / 张角 / 平视同其余 room 机位，退多远按两条规则各出一个候选：
「退到门高占画面高一半」与「退到撞墙、留墙距」；每个候选用仓里同一套取景判据（``_room_view_check``）量，过没过照实写进
``cam-room-入户门.pose.json``，过不了就不进真跑。渲图走 ``render_base_views`` 同一条光栅，只是位姿是给定的（本脚本抄它的函数体）。

自证：三台室内机位 ``frame-sill`` 重渲与上一批 ``run-2026-09-05-sketch-symbols/frame-sill/<机位>/sketch.png`` 逐字节相同
（抽门扇线 / 把手成函数没动字节）；``frame-handle`` 与 ``frame-sill`` 四路逐字节相同、控制稿白像素是超集（只多把手）。
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render3d_worker import base_render as br
from render3d_worker.models import DesignPackage, ScenePackage
from render3d_worker.raster import look_at_matrix, perspective_matrix
from render3d_worker.scene_compile import compile_scene_package

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
PACKAGE = ROOT / "_iteration/run-2026-09-05-opening-kind-consume/design-package-真值138-16洞-roomcams.json"
PRIOR_DEFAULT_DIR = ROOT / "_iteration/run-2026-09-05-opening-kind-consume"
PRIOR_SYMBOLS_DIR = ROOT / "_iteration/run-2026-09-05-sketch-symbols"
ROOM_CAMERA_IDS = ("cam-room-书房", "cam-room-主卧", "cam-room-客厅")
BIRD_CAMERA_ID = "cam-bird-dollhouse"
ENTRY_CAMERA_ID = "cam-room-入户门"
ENTRY_OPENING_INDEX = 10
ENTRY_RULES = ("退到门高占半幅", "退到撞墙留墙距")
SCHEME = "frame-handle"
CONTROL_SCHEME = "frame-sill"
WIDTH_PX, HEIGHT_PX = 1280, 960
ROOM_FOV_DEG = 65.0
ROOM_EYE_HEIGHT_M = 1.55
DOOR_HEIGHT_FRAME_RATIO = 0.5
"""「退到门高占半幅」：距离 ＝ 门高 / (2 · tan(竖直半张角) · 0.5)。"""
WALL_PROBE_STEP_M = 0.01
"""「退到撞墙留墙距」：从门的室内面沿法线每 1 cm 探一步，踩进任何墙块的平面包围盒就算撞墙，退回 ROOM_EYE_WALL_MARGIN_M。"""
TILE = (640, 480)
LABEL_H = 30


def _font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc"):
        try:
            return ImageFont.truetype(candidate, 22)
        except OSError:
            continue
    return ImageFont.load_default()


def _white_mask(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("L")) == br.SKETCH_FOREGROUND_U8


def _wall_boxes_xy(scene: ScenePackage) -> list[tuple[float, float, float, float]]:
    boxes = []
    for mesh in scene.meshes:
        if mesh.semantic != "wall" or not mesh.triangles:
            continue
        v = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
        boxes.append((float(v[:, 0].min()), float(v[:, 0].max()), float(v[:, 1].min()), float(v[:, 1].max())))
    return boxes


def _first_wall_hit_m(scene: ScenePackage, start_xy: np.ndarray, direction_xy: np.ndarray, max_m: float = 20.0) -> float:
    boxes = _wall_boxes_xy(scene)
    steps = int(max_m / WALL_PROBE_STEP_M)
    for i in range(1, steps + 1):
        d = i * WALL_PROBE_STEP_M
        p = start_xy + direction_xy * d
        if any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for x0, x1, y0, y1 in boxes):
            return d
    return max_m


def entry_door_candidates(scene: ScenePackage, aspect_ratio: float) -> list[tuple[str, br.CameraPose, dict[str, object]]]:
    """按几何定的入户门机位候选 + 各自取景判据的量数（过没过照实写）。"""
    entries = [row for row in scene.openings if row.kind == "entry-door"]
    assert [row.opening_index for row in entries] == [ENTRY_OPENING_INDEX], entries
    frames = [frame for frame in br._opening_frames(scene) if frame.kind == "entry-door"]
    assert len(frames) == 1, frames
    frame = frames[0]
    (a0, a1), (c0, c1), (z0, z1) = frame.along_m, frame.across_m, frame.z_m
    along_mid = (a0 + a1) * 0.5

    def xy(along_m: float, across_m: float) -> np.ndarray:
        point = np.zeros(2, dtype=np.float64)
        point[frame.along_axis] = along_m
        point[1 - frame.along_axis] = across_m
        return point

    # 室内那一侧：洞口两面墙面外 0.5 m 处，哪一侧踩在哪间房的地板上
    rooms = sorted({m.room for m in scene.meshes if m.semantic == "floor" and m.room})
    inside: tuple[str, float, float] | None = None
    for face_m, sign in ((c1, 1.0), (c0, -1.0)):
        probe = xy(along_mid, face_m + sign * 0.5)
        for room in rooms:
            if br._point_in_any_triangle_xy(br._room_floor_triangles_xy_m(scene, room), probe):
                assert inside is None, f"洞口两侧都有地板，不是入户门：{inside} / {room}"
                inside = (room, face_m, sign)
    assert inside is not None, "入户门两侧都没踩到地板"
    room, face_m, sign = inside

    door_height_m = z1 - z0
    half_fov_v = math.radians(ROOM_FOV_DEG) * 0.5
    inward_xy = xy(0.0, sign)
    face_xy = xy(along_mid, face_m)
    wall_hit_m = _first_wall_hit_m(scene, face_xy, inward_xy)
    distances = {
        ENTRY_RULES[0]: door_height_m / (2.0 * math.tan(half_fov_v) * DOOR_HEIGHT_FRAME_RATIO),
        ENTRY_RULES[1]: max(0.0, wall_hit_m - br.ROOM_EYE_WALL_MARGIN_M),
    }
    forward_xy = -inward_xy
    yaw_deg = math.degrees(math.atan2(float(forward_xy[0]), float(forward_xy[1]))) % 360.0
    floor_triangles_xy = br._room_floor_triangles_xy_m(scene, room)
    _centroid_xy, floor_z_m = br._room_floor_anchor_m(scene, room)
    eye_z_m = floor_z_m + ROOM_EYE_HEIGHT_M
    mesh_indices = br._rendered_mesh_indices(scene, "room")
    min_xyz_m, max_xyz_m = br._scene_bounds_m(scene, mesh_indices)
    diagonal_m = float(np.linalg.norm(max_xyz_m - min_xyz_m))
    far_clip_m = max(br.FAR_CLIP_MIN_M, diagonal_m * br.FAR_CLIP_DIAGONAL_RATIO)
    job = br._room_view_job(scene, room, mesh_indices, ROOM_FOV_DEG, aspect_ratio, far_clip_m)
    forward = br._yaw_pitch_direction(yaw_deg, br.ROOM_PITCH_DEG)

    out: list[tuple[str, br.CameraPose, dict[str, object]]] = []
    for rule, distance_m in distances.items():
        eye_xy = face_xy + inward_xy * distance_m
        check = br._room_view_check(job, eye_xy, eye_z_m, yaw_deg, candidate_count=1)
        eye_m = np.array([eye_xy[0], eye_xy[1], eye_z_m], dtype=np.float64)
        target_m = eye_m + forward * br.ROOM_TARGET_DISTANCE_M
        camera_id = f"{ENTRY_CAMERA_ID}-{rule}"
        pose = br.CameraPose(
            camera_id=camera_id,
            kind="room",
            eye_m=(float(eye_m[0]), float(eye_m[1]), float(eye_m[2])),
            target_m=(float(target_m[0]), float(target_m[1]), float(target_m[2])),
            up_hint_xyz=(0.0, 0.0, 1.0),
            fov_deg=ROOM_FOV_DEG,
            near_clip_m=br.NEAR_CLIP_M,
            far_clip_m=far_clip_m,
            room_view=check,
        )
        visible_height_at_door_m = 2.0 * distance_m * math.tan(half_fov_v)
        record: dict[str, object] = {
            "cameraId": camera_id,
            "rule": rule,
            "openingIndex": ENTRY_OPENING_INDEX,
            "openingKind": frame.kind,
            "openingAlongAxis": frame.along_axis,
            "openingAlongM": [a0, a1],
            "openingAcrossM": [c0, c1],
            "openingZM": [z0, z1],
            "insideRoom": room,
            "insideFaceAcrossM": face_m,
            "firstWallHitFromFaceM": wall_hit_m,
            "distanceToFaceM": distance_m,
            "visibleHeightAtDoorM": visible_height_at_door_m,
            "doorFitsVertically": visible_height_at_door_m >= door_height_m,
            "eyeM": list(pose.eye_m),
            "yawDeg": yaw_deg,
            "pitchDeg": br.ROOM_PITCH_DEG,
            "fovDeg": ROOM_FOV_DEG,
            "eyeOnRoomFloor": bool(br._point_in_any_triangle_xy(floor_triangles_xy, eye_xy)),
            "roomView": check.model_dump(),
        }
        out.append((rule, pose, record))
    return out


def package_route_pose(package: DesignPackage, cameras_json: list[dict[str, object]], yaw_deg: float) -> dict[str, object]:
    """对照：走机位包契约、``room`` 相机显式给 yaw 的那条路会站到哪儿（只记录不用）。"""
    extra = {
        "id": "cam-room-次卧-显式yaw",
        "kind": "room",
        "room": "次卧",
        "eyeHeightM": ROOM_EYE_HEIGHT_M,
        "yawDeg": yaw_deg,
        "pitchDeg": 0.0,
        "fovDeg": ROOM_FOV_DEG,
    }
    raw = json.loads(package.model_dump_json(by_alias=True))
    raw["cameras"] = [*cameras_json, extra]
    scene = compile_scene_package(DesignPackage.model_validate(raw))
    pose = br.resolve_camera_pose(scene, str(extra["id"]), WIDTH_PX / HEIGHT_PX)
    assert pose.room_view is not None
    return {"camera": extra, "eyeM": list(pose.eye_m), "roomView": pose.room_view.model_dump()}


def render_at_pose(scene: ScenePackage, pose: br.CameraPose, scheme: str) -> tuple[bytes, bytes, bytes]:
    """``render_base_views`` 的函数体，位姿给定；返回 (sketch, line, geometry)。"""
    aspect_ratio = WIDTH_PX / HEIGHT_PX
    mesh_indices = br._rendered_mesh_indices(scene, pose.kind)
    triangles_m, tri_mesh_ids = br._flatten_meshes(scene, mesh_indices)
    palette_ratio = br._mesh_palette_ratio(scene)
    shadow = br._build_shadow_map(scene, mesh_indices)
    view_matrix = look_at_matrix(pose.eye_m, pose.target_m, pose.up_hint_xyz)
    proj_matrix = perspective_matrix(pose.fov_deg, aspect_ratio, pose.near_clip_m, pose.far_clip_m)
    job = br._RasterJob(
        triangles_m=triangles_m,
        tri_mesh_ids=tri_mesh_ids,
        view_matrix=view_matrix,
        proj_matrix=proj_matrix,
        width_px=WIDTH_PX,
        height_px=HEIGHT_PX,
        near_clip_m=pose.near_clip_m,
    )
    buffers = job.rasterize_at(1)
    screen = br._screen_geometry(buffers, view_matrix, pose, aspect_ratio)
    sketch = br._encode_sketch_png(scene, buffers, screen, view_matrix, proj_matrix, pose, scheme)  # type: ignore[arg-type]
    line = br._encode_line_png(buffers, screen)
    geometry = br._encode_geometry_png(job, buffers, screen, pose, aspect_ratio, palette_ratio, shadow)
    return sketch, line, geometry


def main() -> None:
    raw = json.loads(PACKAGE.read_text(encoding="utf-8"))
    package = DesignPackage.model_validate(raw)
    scene = compile_scene_package(package)
    aspect_ratio = WIDTH_PX / HEIGHT_PX
    font = _font()
    sketches: dict[tuple[str, str], bytes] = {}

    # 三台室内 + 揭顶：走 render_base_views
    for camera_id in (*ROOM_CAMERA_IDS, BIRD_CAMERA_ID):
        views = {s: br.render_base_views(scene, camera_id, WIDTH_PX, HEIGHT_PX, s) for s in (CONTROL_SCHEME, SCHEME)}  # type: ignore[arg-type]
        a, b = views[CONTROL_SCHEME], views[SCHEME]
        assert (a.geometry_png, a.depth_png, a.line_png, a.mask_png) == (b.geometry_png, b.depth_png, b.line_png, b.mask_png)
        if camera_id in ROOM_CAMERA_IDS:
            prior = (PRIOR_SYMBOLS_DIR / CONTROL_SCHEME / camera_id / "sketch.png").read_bytes()
            assert prior == a.sketch_png, f"{camera_id}：frame-sill 重渲与上一批不同"
        for scheme, v in views.items():
            sketches[(scheme, camera_id)] = v.sketch_png
        out = HERE / SCHEME / camera_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "sketch.png").write_bytes(b.sketch_png)
        if camera_id == BIRD_CAMERA_ID:
            out = HERE / CONTROL_SCHEME / camera_id
            out.mkdir(parents=True, exist_ok=True)
            (out / "sketch.png").write_bytes(a.sketch_png)

    # 入户门机位：按几何定位姿，两条规则各一个候选
    candidates = entry_door_candidates(scene, aspect_ratio)
    entry_ids = []
    records: dict[str, object] = {
        "candidates": [record for _rule, _pose, record in candidates],
        "roomViewThresholds": {
            "minDepthM": br.ROOM_EYE_WALL_MARGIN_M,
            "minTargetFloorRatio": br.ROOM_VIEW_MIN_TARGET_FLOOR_RATIO,
            "minDominanceRatio": br.ROOM_VIEW_MIN_DOMINANCE_RATIO,
        },
        "packageRouteExplicitYaw": package_route_pose(package, raw["cameras"], float(candidates[0][2]["yawDeg"])),
        "anyPassed": any(pose.room_view is not None and pose.room_view.passed for _r, pose, _rec in candidates),
    }
    (HERE / f"{ENTRY_CAMERA_ID}.pose.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    for rule, pose, record in candidates:
        print(f"{pose.camera_id}: eye={record['eyeM']} yaw={record['yawDeg']} 距门面 {record['distanceToFaceM']:.2f} m "
              f"门放得下={record['doorFitsVertically']} 取景判据={record['roomView']}")
        entry_ids.append(pose.camera_id)
        for scheme in (CONTROL_SCHEME, SCHEME):
            sketch, line, geometry = render_at_pose(scene, pose, scheme)
            sketches[(scheme, pose.camera_id)] = sketch
            out = HERE / scheme / pose.camera_id
            out.mkdir(parents=True, exist_ok=True)
            (out / "sketch.png").write_bytes(sketch)
            if scheme == SCHEME:
                (out / "geometry.png").write_bytes(geometry)
    print(f"机位包契约那条路（次卧显式 yaw）：{records['packageRouteExplicitYaw']}")
    print(f"入户门候选有过判据的：{records['anyPassed']}")

    # 超集自证 + 白像素表 + 拼图
    camera_ids = (*ROOM_CAMERA_IDS, BIRD_CAMERA_ID, *entry_ids)
    montage = Image.new("RGB", (TILE[0] * 3, (TILE[1] + LABEL_H) * len(camera_ids)), (20, 20, 20))
    draw = ImageDraw.Draw(montage)
    rows = [
        "| 机位 | diagonal-cross 白像素（上一批） | frame-sill 白像素 | frame-handle 白像素 | 把手多画的像素 | frame-handle sha256 前 12 位 |",
        "|---|---|---|---|---|---|",
    ]
    for row, camera_id in enumerate(camera_ids):
        sill = _white_mask(sketches[(CONTROL_SCHEME, camera_id)])
        handle = _white_mask(sketches[(SCHEME, camera_id)])
        assert not np.any(sill & ~handle), f"{camera_id}：frame-handle 少画了 frame-sill 的线"
        default_path = PRIOR_DEFAULT_DIR / camera_id / "sketch.png"
        default_white = int(np.count_nonzero(_white_mask(default_path.read_bytes()))) if default_path.exists() else None
        digest = hashlib.sha256(sketches[(SCHEME, camera_id)]).hexdigest()[:12]
        rows.append(
            f"| {camera_id} | {default_white if default_white is not None else '—（机位包里没有这台）'} | "
            f"{int(np.count_nonzero(sill))} | {int(np.count_nonzero(handle))} | {int(np.count_nonzero(handle & ~sill)):+d} | {digest} |"
        )
        tiles: list[tuple[str, Image.Image | None]] = [
            (f"{camera_id} diagonal-cross（上一批）", Image.open(default_path) if default_path.exists() else None),
            (f"{camera_id} frame-sill", Image.open(io.BytesIO(sketches[(CONTROL_SCHEME, camera_id)]))),
            (f"{camera_id} frame-handle", Image.open(io.BytesIO(sketches[(SCHEME, camera_id)]))),
        ]
        y = row * (TILE[1] + LABEL_H)
        for col, (label, tile) in enumerate(tiles):
            x = col * TILE[0]
            draw.text((x + 8, y + 4), label, fill=(255, 200, 80), font=font)
            if tile is not None:
                montage.paste(tile.convert("RGB").resize(TILE), (x, y + LABEL_H))
    montage.save(HERE / "montage-控制稿三方案并排.png")
    print("\n".join(rows))
    print(f"拼图：{(HERE / 'montage-控制稿三方案并排.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
