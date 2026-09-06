"""揭顶三方案 × 7 seed 的拼图、叠线与洞口放大页。只在这一批用，写完不改。零模型调用。

跑法（仓根目录）：uv run python _iteration/run-2026-09-06-sketch-symbol-birdview/拼图与叠线.py [叠线单张输出目录]
给第二个参数就把 21 张逐张叠线图（图幅尺寸）也写到那个目录——判读时逐张看用的就是它，不入库。

版式照抄 `run-2026-09-05-sketch-symbol-combo/拼图与叠线.py`（同一套投影、同一套洞号断言、同一种红色叠线）：
- `montage-揭顶-原图.png` / `montage-揭顶-叠线.png`：行＝方案，列＝控制稿 + seed1…7。
- `crops-揭顶-{门,窗}-<方案>.jpg`：行＝洞口，列＝控制稿 + seed1…7；洞口框四角用底渲那两个矩阵投到像素、外扩裁下放大。
  门那张 pad 大一圈（80 px），为的是把洞口两侧的地面也裁进来——本批要数"地面被符号切色"。

图从哪儿来（seed 1/2/3 三方案的揭顶图前两批已经出过，控制稿 sha256 逐字节相同 → 万相按 seed 出同图，本批不重跑）：
- `diagonal-cross` seed1/2/3：`run-2026-09-05-opening-kind-realism/cam-bird-dollhouse-seed{N}.png`
- `frame-sill` / `frame-handle` seed1/2/3：`run-2026-09-05-sketch-symbol-combo/真跑/<方案>/cam-bird-dollhouse/seed{N}.png`
- 三方案 seed4/5/6/7：本目录 `真跑/<方案>/seed{N}.png`
"""

from __future__ import annotations

import json
import re
import sys
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
PRIOR_REALISM_DIR = ROOT / "_iteration/run-2026-09-05-opening-kind-realism"
PRIOR_COMBO_RUN = ROOT / "_iteration/run-2026-09-05-sketch-symbol-combo/真跑"
CAMERA_ID = "cam-bird-dollhouse"
SCHEMES = ("diagonal-cross", "frame-sill", "frame-handle")
PRIOR_SEEDS = (1, 2, 3)
NEW_SEEDS = (4, 5, 6, 7)
SEEDS = (*PRIOR_SEEDS, *NEW_SEEDS)
WIDTH_PX, HEIGHT_PX = 1280, 960
TILE = (512, 384)
LABEL_H = 28
FRAME_ORDER = (0, 3, 4, 5, 6, 9, 10, 13, 14, 15, 1, 2, 7, 8, 11, 12)
"""``_opening_frames`` 的网格序 → 16 洞洞口表洞号（`run-2026-09-05-opening-kind-consume/run.md`）。"""
FRAME_KEYS = {
    0: ("window", 1, 4.42), 3: ("window", 1, 4.58), 4: ("window", 0, 1.76), 5: ("window", 0, 4.92),
    6: ("window", 0, 7.13), 9: ("window", 0, 0.27), 10: ("entry-door", 0, 9.22), 13: ("window", 0, 0.68),
    14: ("window", 0, 8.50), 15: ("window", 0, 4.46), 1: ("door", 1, 6.52), 2: ("door", 1, 6.53),
    7: ("door", 0, 3.34), 8: ("door", 0, 5.13), 11: ("door", 0, 8.30), 12: ("door", 0, 4.67),
}
DOOR_INDICES = (1, 2, 7, 8, 10, 11, 12)
WINDOW_INDICES = (0, 3, 4, 5, 6, 9, 13, 14, 15)


def _font(size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def sketch_path(scheme: str) -> Path:
    return HERE / "控制稿" / scheme / "sketch.png"


def image_path(scheme: str, seed: int) -> Path:
    if seed in NEW_SEEDS:
        return HERE / "真跑" / scheme / f"seed{seed}.png"
    if scheme == "diagonal-cross":
        return PRIOR_REALISM_DIR / f"{CAMERA_ID}-seed{seed}.png"
    return PRIOR_COMBO_RUN / scheme / CAMERA_ID / f"seed{seed}.png"


def log_path(scheme: str, seed: int) -> Path:
    if seed in NEW_SEEDS:
        return HERE / "真跑" / scheme / f"seed{seed}.png.log"
    if scheme == "diagonal-cross":
        return PRIOR_REALISM_DIR / f"{CAMERA_ID}-seed{seed}.png.log"  # 上一批没留 log
    return PRIOR_COMBO_RUN / scheme / CAMERA_ID / f"seed{seed}.png.log"


def overlay(image: Image.Image, sketch: Image.Image) -> Image.Image:
    base = image.convert("RGB")
    mask = sketch.convert("L").resize(base.size, Image.Resampling.NEAREST).point(lambda v: 255 if v > 127 else 0)
    red = Image.new("RGB", base.size, (255, 40, 40))
    return Image.composite(red, base, mask)


def _score_and_elapsed(log: Path) -> tuple[str, str, str]:
    if not log.exists():
        return "—", "—", "—"
    text = log.read_text(encoding="utf-8", errors="replace")
    score = re.search(r"fidelity_score=([0-9.]+)", text)
    elapsed = re.search(r"elapsed_seconds=([0-9.]+)", text)
    exit_code = re.search(r"exit=(\d+)", text)
    return (score.group(1) if score else "—", elapsed.group(1) if elapsed else "—", exit_code.group(1) if exit_code else "—")


def montages(overlay_dir: Path | None) -> None:
    font = _font()
    print("| 方案 | seed | 出处 | 退出码 | 后端耗时 s | 分数（对 sketch，尺子照 CLI；跨形态不可比） |")
    print("|---|---|---|---|---|---|")
    for name, want_overlay in (("原图", False), ("叠线", True)):
        columns = 1 + len(SEEDS)
        montage = Image.new("RGB", (TILE[0] * columns, (TILE[1] + LABEL_H) * len(SCHEMES)), (20, 20, 20))
        draw = ImageDraw.Draw(montage)
        for row, scheme in enumerate(SCHEMES):
            sketch = Image.open(sketch_path(scheme))
            y = row * (TILE[1] + LABEL_H)
            tiles: list[tuple[str, Image.Image | None]] = [(f"{scheme} 控制稿", sketch.convert("RGB"))]
            for seed in SEEDS:
                path = image_path(scheme, seed)
                score, elapsed, exit_code = _score_and_elapsed(log_path(scheme, seed))
                if not want_overlay:
                    origin = "本批" if seed in NEW_SEEDS else "前两批（同稿同 seed）"
                    print(f"| {scheme} | {seed} | {origin} | {exit_code} | {elapsed} | {score} |")
                if not path.exists():
                    tiles.append((f"seed{seed} 缺", None))
                    continue
                image = Image.open(path)
                mixed = overlay(image, sketch)
                tiles.append((f"seed{seed} 分数 {score}", mixed if want_overlay else image.convert("RGB")))
                if want_overlay and overlay_dir is not None:
                    mixed.save(overlay_dir / f"{scheme}-seed{seed}.overlay.png")
            for col, (label, tile) in enumerate(tiles):
                x = col * TILE[0]
                draw.text((x + 6, y + 3), label, fill=(255, 200, 80), font=font)
                if tile is not None:
                    montage.paste(tile.resize(TILE), (x, y + LABEL_H))
        out = HERE / f"montage-揭顶-{name}.png"
        montage.save(out)
        print(f"<!-- 拼图：{out.relative_to(ROOT)}（{montage.size[0]}×{montage.size[1]}） -->")


def _frames_by_index(scene: ScenePackage) -> dict[int, br._OpeningFrame]:
    frames = br._opening_frames(scene)
    assert len(frames) == len(FRAME_ORDER)
    out = {}
    for index, frame in zip(FRAME_ORDER, frames, strict=True):
        kind, axis, along0 = FRAME_KEYS[index]
        assert (frame.kind, frame.along_axis, round(frame.along_m[0], 2)) == (kind, axis, along0), (index, frame)
        out[index] = frame
    return out


def _projector(scene: ScenePackage):
    aspect = WIDTH_PX / HEIGHT_PX
    pose = br.resolve_camera_pose(scene, CAMERA_ID, aspect)
    view = look_at_matrix(pose.eye_m, pose.target_m, pose.up_hint_xyz)
    proj = perspective_matrix(pose.fov_deg, aspect, pose.near_clip_m, pose.far_clip_m)

    def project(point_m: np.ndarray) -> tuple[float, float] | None:
        clip = proj @ (view @ np.array([*point_m, 1.0], dtype=np.float64))
        if clip[3] <= 0.0:
            return None
        ndc = clip[:3] / clip[3]
        return (ndc[0] + 1.0) * 0.5 * WIDTH_PX, (1.0 - ndc[1]) * 0.5 * HEIGHT_PX

    return project


def opening_boxes(scene: ScenePackage, indices: tuple[int, ...], pad_px: int) -> list[tuple[str, tuple[int, int, int, int]]]:
    frames = _frames_by_index(scene)
    project = _projector(scene)
    boxes = []
    for index in indices:
        frame = frames[index]
        corners = [br._frame_point(frame, a, z, frame.across_center_m) for a in frame.along_m for z in frame.z_m]
        points = [p for p in (project(c) for c in corners) if p is not None]
        if len(points) < 4:
            continue
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        x0, y0 = max(0, int(min(xs)) - pad_px), max(0, int(min(ys)) - pad_px)
        x1, y1 = min(WIDTH_PX, int(max(xs)) + pad_px), min(HEIGHT_PX, int(max(ys)) + pad_px)
        if x1 - x0 < 2 * pad_px or y1 - y0 < 2 * pad_px:
            continue
        boxes.append((f"#{index}{'窗' if frame.kind == 'window' else '门'}", (x0, y0, x1, y1)))
    return boxes


def crops(scheme: str, sheet_name: str, boxes: list[tuple[str, tuple[int, int, int, int]]], scale: float) -> None:
    font = _font(18)
    columns: list[tuple[str, Path]] = [("控制稿", sketch_path(scheme))]
    columns += [(f"s{seed}", image_path(scheme, seed)) for seed in SEEDS]
    cell_w = int(max(x1 - x0 for _n, (x0, _y0, x1, _y1) in boxes) * scale)
    cell_h = int(max(y1 - y0 for _n, (_x0, y0, _x1, y1) in boxes) * scale)
    sheet = Image.new("RGB", (cell_w * len(columns), (cell_h + LABEL_H) * len(boxes)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for row, (name, (x0, y0, x1, y1)) in enumerate(boxes):
        y = row * (cell_h + LABEL_H)
        for col, (label, path) in enumerate(columns):
            x = col * cell_w
            draw.text((x + 4, y + 2), f"{name} {label}", fill=(255, 200, 80), font=font)
            if not path.exists():
                continue
            image = Image.open(path).convert("RGB")
            sx, sy = image.size[0] / WIDTH_PX, image.size[1] / HEIGHT_PX
            crop = image.crop((int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)))
            crop = crop.resize((int((x1 - x0) * scale), int((y1 - y0) * scale)), Image.Resampling.LANCZOS)
            sheet.paste(crop, (x, y + LABEL_H))
    out = HERE / f"crops-揭顶-{sheet_name}-{scheme}.jpg"
    sheet.save(out, quality=88)
    print(f"<!-- 放大页：{out.relative_to(ROOT)}（{sheet.size[0]}×{sheet.size[1]}）；方框：{boxes} -->")


def main() -> None:
    overlay_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    montages(overlay_dir)
    scene = compile_scene_package(DesignPackage.model_validate(json.loads(PACKAGE.read_text(encoding="utf-8"))))
    door_boxes = opening_boxes(scene, DOOR_INDICES, pad_px=80)
    window_boxes = opening_boxes(scene, WINDOW_INDICES, pad_px=45)
    for scheme in SCHEMES:
        crops(scheme, "门", door_boxes, scale=2.0)
        crops(scheme, "窗", window_boxes, scale=2.0)


if __name__ == "__main__":
    main()
