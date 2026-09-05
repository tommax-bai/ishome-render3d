"""真户型 8 台机位按新取景规则重渲（geometry/line/sketch 三路）+「改前线稿 vs 改后控制稿」拼图
+ 数字表（老规则位姿按同一评估器量）。只在这一批用，写完不改。

跑法（仓根目录）：uv run python _iteration/run-2026-09-05-control-sketch/重渲与拼图.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render3d_worker import base_render as br
from render3d_worker.models import DesignPackage
from render3d_worker.scene_compile import compile_scene_package

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
PACKAGE = ROOT / "_iteration/run-2026-09-04-failure-catalogue/底渲-室内机位/design-package-真值138-roomcams.json"
OLD_LINE_DIR = ROOT / "_iteration/run-2026-09-04-failure-catalogue/底渲-室内机位"
OLD_BIRD_LINE = ROOT / "_iteration/真户型-基准/底渲-cam-bird-dollhouse/line.png"
OLD_POSES = HERE / "老规则位姿.json"
WIDTH_PX, HEIGHT_PX = 1280, 960
ASPECT = WIDTH_PX / HEIGHT_PX
FOOTPRINT_MARGIN_M = 0.30  # 足迹外扩一个墙厚（真户型墙厚中位 0.19 m、最大 0.64 m）——只为记录，不进库


def footprint_inside_ratio(scene, room, eye_m, yaw_deg, fov_deg):
    """击中点落在目标房间地板足迹（外扩一个墙厚）里的像素占击中像素的比例——记录用，
    库里没有这个量（它分不开 D2，见 run.md）。"""
    from render3d_worker.raster import look_at_matrix, perspective_matrix, rasterize

    mesh_indices = list(range(len(scene.meshes)))
    tris, ids = br._flatten_meshes(scene, mesh_indices)
    lo, hi = br._scene_bounds_m(scene, mesh_indices)
    far = max(br.FAR_CLIP_MIN_M, float(np.linalg.norm(hi - lo)) * br.FAR_CLIP_DIAGONAL_RATIO)
    eye = np.asarray(eye_m, dtype=np.float64)
    forward = br._yaw_pitch_direction(yaw_deg, 0.0)
    view = look_at_matrix(eye, eye + forward)
    h = br.ROOM_VIEW_EVAL_HEIGHT_PX
    w = int(round(h * ASPECT))
    buf = rasterize(tris, ids, view, perspective_matrix(fov_deg, ASPECT, br.NEAR_CLIP_M, far), w, h, br.NEAR_CLIP_M)
    ray = br._world_ray_xyz(buf, view, fov_deg, ASPECT)
    depth = np.where(buf.hit_mask, buf.depth_m, 0.0)
    xy = (eye.astype(np.float32) + ray * depth[..., None])[..., :2]
    inside = np.zeros(buf.hit_mask.shape, dtype=bool)
    for x0, x1, y0, y1 in br._room_floor_boxes_m(scene, room):
        inside |= (xy[..., 0] >= x0 - FOOTPRINT_MARGIN_M) & (xy[..., 0] <= x1 + FOOTPRINT_MARGIN_M) & (
            xy[..., 1] >= y0 - FOOTPRINT_MARGIN_M
        ) & (xy[..., 1] <= y1 + FOOTPRINT_MARGIN_M)
    hits = int(buf.hit_mask.sum())
    return float((inside & buf.hit_mask).sum()) / hits if hits else 0.0


def main() -> None:
    package = DesignPackage.model_validate(json.loads(PACKAGE.read_text(encoding="utf-8")))
    scene = compile_scene_package(package)
    old_poses = json.loads(OLD_POSES.read_text(encoding="utf-8"))

    rows = []
    tiles = []
    for camera in scene.cameras:
        out_dir = HERE / camera.id
        out_dir.mkdir(exist_ok=True)
        error = None
        views = None
        try:
            views = br.render_base_views(scene, camera.id, WIDTH_PX, HEIGHT_PX)
        except br.BaseRenderError as e:
            error = str(e)
        if views is not None:
            (out_dir / "geometry.png").write_bytes(views.geometry_png)
            (out_dir / "line.png").write_bytes(views.line_png)
            (out_dir / "sketch.png").write_bytes(views.sketch_png)

        # 老规则的位姿按同一评估器量（老位姿存在 老规则位姿.json，改前代码解出来的）
        old = None
        if camera.kind == "room":
            pose = old_poses[camera.id]
            eye = np.asarray(pose["eye_m"])
            fwd = np.asarray(pose["target_m"]) - eye
            yaw_old = math.degrees(math.atan2(fwd[0], fwd[1]))
            mesh_indices = br._rendered_mesh_indices(scene, camera.kind)
            lo, hi = br._scene_bounds_m(scene, mesh_indices)
            far = max(br.FAR_CLIP_MIN_M, float(np.linalg.norm(hi - lo)) * br.FAR_CLIP_DIAGONAL_RATIO)
            job = br._room_view_job(scene, camera.room, mesh_indices, camera.fov_deg, ASPECT, far)
            old_check = br._room_view_check(job, eye[:2], float(eye[2]), yaw_old, candidate_count=1)
            old = (old_check, footprint_inside_ratio(scene, camera.room, eye, yaw_old, camera.fov_deg))

        new = None
        if views is not None and views.room_view is not None:
            check = views.room_view
            new = (check, footprint_inside_ratio(scene, camera.room, check.eye_m, check.yaw_deg, camera.fov_deg))
        # 最终图上的目标房间地板占比（按遮罩索引表加像素）
        full_floor = None
        if views is not None and camera.kind == "room":
            px = sum(e.pixel_count for e in views.mask_index if e.semantic == "floor" and e.room == camera.room)
            full_floor = px / (WIDTH_PX * HEIGHT_PX)
        rows.append((camera.id, old, new, error, views.near_m if views else None, full_floor))

        # 拼图：改前线稿 | 改后控制稿
        old_line = OLD_BIRD_LINE if camera.kind == "bird" else OLD_LINE_DIR / camera.id / "line.png"
        left = Image.open(old_line).convert("L").resize((640, 480))
        if views is not None:
            import io

            right = Image.open(io.BytesIO(views.sketch_png)).convert("L").resize((640, 480))
        else:
            right = Image.new("L", (640, 480), 0)
        tiles.append((camera.id, left, right, error))

    font = None
    for candidate in ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc"):
        try:
            font = ImageFont.truetype(candidate, 22)
            break
        except OSError:
            continue
    montage = Image.new("RGB", (1280, 480 * len(tiles) + 30 * len(tiles)), (20, 20, 20))
    y = 0
    for camera_id, left, right, error in tiles:
        draw = ImageDraw.Draw(montage)
        label = f"{camera_id}    左：改前线稿 line.png（9-04 老取景）    右：改后控制稿 sketch.png（新取景）"
        if error:
            label += f"    ——新取景失败：{error[:60]}…"
        draw.text((8, y + 4), label, fill=(255, 200, 80), font=font)
        y += 30
        montage.paste(left.convert("RGB"), (0, y))
        montage.paste(right.convert("RGB"), (640, y))
        y += 480
    montage.save(HERE / "montage-改前线稿-vs-改后控制稿.png")

    print("| 机位 | 老规则：最小深度 / 地板占比 / 主体占比 / 足迹内占比 | 新规则：站位·朝向 / 候选数 / 最小深度 / 地板占比 / 主体占比 / 足迹内占比 | 最终图 near_m / 地板占比 |")
    print("|---|---|---|---|")
    for camera_id, old, new, error, near_m, full_floor in rows:
        o = "—" if old is None else f"{old[0].min_depth_m:.2f} / {old[0].target_floor_ratio:.3f} / {old[0].dominance_ratio:.3f} / {old[1]:.3f}"
        if error:
            n = f"**无法取景**：{error}"
        elif new is None:
            n = "—（揭顶）"
        else:
            c = new[0]
            n = f"({c.eye_m[0]:.2f}, {c.eye_m[1]:.2f}) 朝 {c.yaw_deg:.1f}° / {c.candidate_count} / {c.min_depth_m:.2f} / {c.target_floor_ratio:.3f} / {c.dominance_ratio:.3f} / {new[1]:.3f}"
        f = "—" if near_m is None else f"{near_m:.2f} / " + ("—" if full_floor is None else f"{full_floor:.3f}")
        print(f"| {camera_id} | {o} | {n} | {f} |")


if __name__ == "__main__":
    main()
