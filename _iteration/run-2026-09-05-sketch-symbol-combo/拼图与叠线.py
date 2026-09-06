"""真跑结果拼图：每机位一张（行＝方案；列＝控制稿、seed1/2/3、seed1/2/3 叠控制稿红），外加洞口放大页。
方案行：本批 ``frame-handle``（四台）与 ``frame-sill``（揭顶）；上一批 ``frame-sill``（三台室内，`run-2026-09-05-sketch-symbols/`）
与 ``diagonal-cross``（四台，图取 `run-2026-09-05-opening-kind-realism/`）同版式并排，好横向看。
同时把每张的分数与耗时从 `.log` 里抄出来打成表（抄进 run.md）。只在这一批用，写完不改。零模型调用。

跑法（仓根目录）：uv run python _iteration/run-2026-09-05-sketch-symbol-combo/拼图与叠线.py [叠线单张输出目录]
给了第二个参数就把逐张叠线图（图幅尺寸）也写到那个目录（判读用，不入库）。

洞口放大页（`crops-<机位>-<门|窗>.jpg`）：把每个洞口的框（墙厚中心平面上的矩形四角）用底渲真正在用的那两个矩阵投到像素上，
外扩一圈裁下来放大；行＝洞口，列＝各方案的控制稿 + 三 seed。只为看清每个洞渲成了什么。洞号照 16 洞洞口表
（`run-2026-09-05-opening-kind-consume/run.md`），框的次序是网格序，这里按几何对上号并断言。
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
RUN_DIR = HERE / "真跑"
PACKAGE = ROOT / "_iteration/run-2026-09-05-opening-kind-consume/design-package-真值138-16洞-roomcams.json"
PRIOR_SYMBOLS_DIR = ROOT / "_iteration/run-2026-09-05-sketch-symbols"
PRIOR_REALISM_DIR = ROOT / "_iteration/run-2026-09-05-opening-kind-realism"
PRIOR_DEFAULT_DIR = ROOT / "_iteration/run-2026-09-05-opening-kind-consume"
ROOM_CAMERA_IDS = ("cam-room-书房", "cam-room-主卧", "cam-room-客厅")
BIRD_CAMERA_ID = "cam-bird-dollhouse"
SEEDS = (1, 2, 3)
WIDTH_PX, HEIGHT_PX = 1280, 960
TILE = (512, 384)
LABEL_H = 28
COLUMNS = ("控制稿", "seed1", "seed2", "seed3", "seed1+叠线", "seed2+叠线", "seed3+叠线")
FRAME_ORDER = (0, 3, 4, 5, 6, 9, 10, 13, 14, 15, 1, 2, 7, 8, 11, 12)
"""``_opening_frames`` 的网格序 → 洞口表洞号；下面按 (种类, 沿墙轴, 沿墙起点) 断言对上。"""
FRAME_KEYS = {
    0: ("window", 1, 4.42), 3: ("window", 1, 4.58), 4: ("window", 0, 1.76), 5: ("window", 0, 4.92),
    6: ("window", 0, 7.13), 9: ("window", 0, 0.27), 10: ("entry-door", 0, 9.22), 13: ("window", 0, 0.68),
    14: ("window", 0, 8.50), 15: ("window", 0, 4.46), 1: ("door", 1, 6.52), 2: ("door", 1, 6.53),
    7: ("door", 0, 3.34), 8: ("door", 0, 5.13), 11: ("door", 0, 8.30), 12: ("door", 0, 4.67),
}


def _font(size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def overlay(image: Image.Image, sketch: Image.Image) -> Image.Image:
    base = image.convert("RGB")
    mask = sketch.convert("L").resize(base.size, Image.Resampling.NEAREST).point(lambda v: 255 if v > 127 else 0)
    red = Image.new("RGB", base.size, (255, 40, 40))
    return Image.composite(red, base, mask)


def _sources(camera_id: str) -> list[tuple[str, Path, Path, Path | None]]:
    """(行标签, 控制稿, 图目录, log 目录)——每机位要并排的方案行。"""
    rows: list[tuple[str, Path, Path, Path | None]] = [
        ("frame-handle（本批）", HERE / "frame-handle" / camera_id / "sketch.png", RUN_DIR / "frame-handle" / camera_id, RUN_DIR / "frame-handle" / camera_id)
    ]
    if camera_id == BIRD_CAMERA_ID:
        rows.append(("frame-sill（本批）", HERE / "frame-sill" / camera_id / "sketch.png", RUN_DIR / "frame-sill" / camera_id, RUN_DIR / "frame-sill" / camera_id))
        rows.append(("diagonal-cross（上一批）", PRIOR_DEFAULT_DIR / camera_id / "sketch.png", PRIOR_REALISM_DIR, None))
    else:
        rows.append(("frame-sill（上一批）", PRIOR_SYMBOLS_DIR / "frame-sill" / camera_id / "sketch.png", PRIOR_SYMBOLS_DIR / "frame-sill" / camera_id, PRIOR_SYMBOLS_DIR / "frame-sill" / camera_id))
        rows.append(("diagonal-cross（上一批）", PRIOR_SYMBOLS_DIR / "diagonal-cross" / camera_id / "sketch.png", PRIOR_REALISM_DIR, None))
    return rows


def _image_path(image_dir: Path, camera_id: str, seed: int) -> Path:
    if image_dir == PRIOR_REALISM_DIR:
        return image_dir / f"{camera_id}-seed{seed}.png"
    return image_dir / f"seed{seed}.png"


def _score_and_elapsed(log: Path | None) -> tuple[str, str, str]:
    if log is None or not log.exists():
        return "—", "—", "—"
    text = log.read_text(encoding="utf-8", errors="replace")
    score = re.search(r"fidelity_score=([0-9.]+)", text)
    elapsed = re.search(r"elapsed_seconds=([0-9.]+)", text)
    exit_code = re.search(r"exit=(\d+)", text)
    return (score.group(1) if score else "—", elapsed.group(1) if elapsed else "—", exit_code.group(1) if exit_code else "—")


def montages(overlay_dir: Path | None) -> None:
    font = _font()
    print("| 方案 | 机位 | seed | 退出码 | 耗时 s | 分数（对 sketch，尺子照 CLI） |")
    print("|---|---|---|---|---|---|")
    for camera_id in (*ROOM_CAMERA_IDS, BIRD_CAMERA_ID):
        rows = _sources(camera_id)
        montage = Image.new("RGB", (TILE[0] * len(COLUMNS), (TILE[1] + LABEL_H) * len(rows)), (20, 20, 20))
        draw = ImageDraw.Draw(montage)
        for row, (label, sketch_path, image_dir, log_dir) in enumerate(rows):
            sketch = Image.open(sketch_path)
            y = row * (TILE[1] + LABEL_H)
            tiles: list[tuple[str, Image.Image | None]] = [(f"{camera_id} {label} 控制稿", sketch.convert("RGB"))]
            overlays: list[tuple[str, Image.Image | None]] = []
            for seed in SEEDS:
                image_path = _image_path(image_dir, camera_id, seed)
                score, elapsed, exit_code = _score_and_elapsed(log_dir / f"seed{seed}.png.log" if log_dir else None)
                if "本批" in label:
                    print(f"| {label.split('（')[0]} | {camera_id} | {seed} | {exit_code} | {elapsed} | {score} |")
                if not image_path.exists():
                    tiles.append((f"seed{seed} 缺", None))
                    overlays.append((f"seed{seed}+叠线 缺", None))
                    continue
                image = Image.open(image_path)
                mixed = overlay(image, sketch)
                tiles.append((f"seed{seed}  分数 {score}", image.convert("RGB")))
                overlays.append((f"seed{seed}+叠线", mixed))
                if overlay_dir is not None:
                    mixed.save(overlay_dir / f"{label.split('（')[0]}-{camera_id}-seed{seed}.overlay.png")
            for col, (tile_label, tile) in enumerate(tiles + overlays):
                x = col * TILE[0]
                draw.text((x + 6, y + 3), tile_label, fill=(255, 200, 80), font=font)
                if tile is not None:
                    montage.paste(tile.resize(TILE), (x, y + LABEL_H))
        montage.save(HERE / f"montage-{camera_id}.png")
        print(f"<!-- 拼图：{(HERE / f'montage-{camera_id}.png').relative_to(ROOT)} -->")


def _frames_by_index(scene: ScenePackage) -> dict[int, br._OpeningFrame]:
    frames = br._opening_frames(scene)
    assert len(frames) == len(FRAME_ORDER)
    out = {}
    for index, frame in zip(FRAME_ORDER, frames, strict=True):
        kind, axis, along0 = FRAME_KEYS[index]
        assert (frame.kind, frame.along_axis, round(frame.along_m[0], 2)) == (kind, axis, along0), (index, frame)
        out[index] = frame
    return out


def _projector(scene: ScenePackage, camera_id: str):
    aspect = WIDTH_PX / HEIGHT_PX
    pose = br.resolve_camera_pose(scene, camera_id, aspect)
    view = look_at_matrix(pose.eye_m, pose.target_m, pose.up_hint_xyz)
    proj = perspective_matrix(pose.fov_deg, aspect, pose.near_clip_m, pose.far_clip_m)

    def project(point_m: np.ndarray) -> tuple[float, float] | None:
        clip = proj @ (view @ np.array([*point_m, 1.0], dtype=np.float64))
        if clip[3] <= 0.0:
            return None
        ndc = clip[:3] / clip[3]
        return (ndc[0] + 1.0) * 0.5 * WIDTH_PX, (1.0 - ndc[1]) * 0.5 * HEIGHT_PX

    return project


def opening_boxes(scene: ScenePackage, camera_id: str, indices: tuple[int, ...], pad_px: int) -> list[tuple[str, tuple[int, int, int, int]]]:
    """洞口框四角投到像素 → 外扩 pad 的方框（裁到画幅内）；整个框都在画幅外的洞跳过。"""
    frames = _frames_by_index(scene)
    project = _projector(scene, camera_id)
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


def crops(camera_id: str, sheet_name: str, boxes: list[tuple[str, tuple[int, int, int, int]]], scale: float) -> None:
    """每个方框一行；列＝各方案的控制稿裁块 + 三 seed 裁块（方框按图幅比例缩放到图上）。"""
    font = _font(18)
    rows = _sources(camera_id)
    columns: list[tuple[str, Path]] = []
    for label, sketch_path, image_dir, _log in rows:
        columns.append((f"{label.split('（')[0]} 控制稿", sketch_path))
        for seed in SEEDS:
            columns.append((f"{label.split('（')[0]} s{seed}", _image_path(image_dir, camera_id, seed)))
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
    out = HERE / f"crops-{camera_id}-{sheet_name}.jpg"
    sheet.save(out, quality=88)
    print(f"<!-- 放大页：{out.relative_to(ROOT)}（{sheet.size[0]}×{sheet.size[1]}）；方框（控制稿坐标）：{boxes} -->")


def main() -> None:
    overlay_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    montages(overlay_dir)
    scene = compile_scene_package(DesignPackage.model_validate(json.loads(PACKAGE.read_text(encoding="utf-8"))))
    crops("cam-room-客厅", "门窗", opening_boxes(scene, "cam-room-客厅", (8, 5, 2), pad_px=40), scale=2.0)
    crops("cam-room-主卧", "门", opening_boxes(scene, "cam-room-主卧", (2, 1), pad_px=40), scale=2.0)
    crops("cam-room-书房", "窗", opening_boxes(scene, "cam-room-书房", (6,), pad_px=40), scale=1.5)
    crops(BIRD_CAMERA_ID, "门", opening_boxes(scene, BIRD_CAMERA_ID, (1, 2, 7, 8, 10, 11, 12), pad_px=45), scale=2.0)
    crops(BIRD_CAMERA_ID, "窗", opening_boxes(scene, BIRD_CAMERA_ID, (0, 3, 4, 5, 6, 9, 13, 14, 15), pad_px=45), scale=2.0)


if __name__ == "__main__":
    main()
