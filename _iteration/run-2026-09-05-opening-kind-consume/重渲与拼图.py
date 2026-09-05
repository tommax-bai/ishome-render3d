"""真户型 16 洞（c9fc00c 产物）8 台机位重渲 geometry/line/sketch 三路 +「19 洞旧控制稿 vs 16 洞新控制稿」拼图
+ 洞口表与取景表（打到 stdout，抄进 run.md）。只在这一批用，写完不改。零模型调用。

跑法（仓根目录）：uv run python _iteration/run-2026-09-05-opening-kind-consume/重渲与拼图.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from render3d_worker import base_render as br
from render3d_worker.models import DesignPackage
from render3d_worker.scene_compile import compile_scene_package

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
PACKAGE = HERE / "design-package-真值138-16洞-roomcams.json"
OLD_SKETCH_DIR = ROOT / "_iteration/run-2026-09-05-control-sketch"  # 19 洞那批的控制稿（同 8 台机位）
WIDTH_PX, HEIGHT_PX = 1280, 960
TILE = (640, 480)


def main() -> None:
    package = DesignPackage.model_validate(json.loads(PACKAGE.read_text(encoding="utf-8")))
    scene = compile_scene_package(package)

    print("## 场景包自证数")
    print(
        f"网格 {len(scene.meshes)} / 三角形 {scene.triangle_count} / 尺子 {scene.metre_per_unit:.4f} m/单位 / "
        f"地板 {scene.floor_area_sqm:.2f} ㎡（吻合 {scene.area_match_ratio:.3f}）/ 墙段 {scene.wall_segment_count}"
        f"（退化 {scene.degenerate_wall_count}）/ 洞 {scene.opening_count_by_kind} / "
        f"按档位猜 {scene.guessed_opening_count} 个 {scene.guessed_opening_indices}"
    )

    print()
    print("## 洞口表")
    print("| # | 轴 | 外墙 | connects | 上游 kind | 最终 kind | 来源 | 落到墙上 | 上游依据（摘） |")
    print("|---|---|---|---|---|---|---|---|---|")
    for entry, opening in zip(scene.openings, package.plan.openings, strict=True):
        print(
            f"| {entry.opening_index} | {opening.axis[0]} | {'是' if opening.is_on_outer_wall else '否'} | "
            f"{'/'.join(opening.connects) or '（空）'} | {opening.kind} | **{entry.kind}** | {entry.kind_source} | "
            f"{'是' if entry.placed else '否'} | {opening.kind_evidence[:40]} |"
        )

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
        rows.append((camera.id, views, error))

        old_path = OLD_SKETCH_DIR / camera.id / "sketch.png"
        old_missing = not old_path.exists()
        left = Image.new("L", TILE, 0) if old_missing else Image.open(old_path).convert("L").resize(TILE)
        right = Image.new("L", TILE, 0) if views is None else Image.open(io.BytesIO(views.sketch_png)).convert("L").resize(TILE)
        tiles.append((camera.id, left, right, old_missing, error))

    font = None
    for candidate in ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc"):
        try:
            font = ImageFont.truetype(candidate, 22)
            break
        except OSError:
            continue
    montage = Image.new("RGB", (TILE[0] * 2, (TILE[1] + 30) * len(tiles)), (20, 20, 20))
    y = 0
    for camera_id, left, right, old_missing, error in tiles:
        draw = ImageDraw.Draw(montage)
        label = f"{camera_id}    左：19 洞旧控制稿（9-05 control-sketch 那批）    右：16 洞新控制稿（c9fc00c 产物）"
        if old_missing:
            label += "    ——旧批无此机位（当时无法取景）"
        if error:
            label += f"    ——新批无法取景：{error[:50]}…"
        draw.text((8, y + 4), label, fill=(255, 200, 80), font=font)
        y += 30
        montage.paste(left.convert("RGB"), (0, y))
        montage.paste(right.convert("RGB"), (TILE[0], y))
        y += TILE[1]
    montage.save(HERE / "montage-19洞旧控制稿-vs-16洞新控制稿.png")

    print()
    print("## 取景")
    print("| 机位 | 站位·朝向 / 候选数 / 最小深度 / 地板占比 / 主体占比 | 最终图 near_m / 几何盖住 / 遮罩块数 |")
    print("|---|---|---|")
    for camera_id, views, error in rows:
        if error:
            print(f"| {camera_id} | **无法取景**：{error} | — |")
            continue
        assert views is not None
        if views.room_view is None:
            framing = "—（揭顶，位姿由包给）"
        else:
            c = views.room_view
            framing = (
                f"({c.eye_m[0]:.2f}, {c.eye_m[1]:.2f}) 朝 {c.yaw_deg:.1f}° / {c.candidate_count} / "
                f"{c.min_depth_m:.2f} / {c.target_floor_ratio:.3f} / {c.dominance_ratio:.3f}"
            )
        print(
            f"| {camera_id} | {framing} | {views.near_m:.2f} / {views.covered_pixel_ratio:.3f} / {len(views.mask_index)} |"
        )


if __name__ == "__main__":
    main()
