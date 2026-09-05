"""真跑结果拼图：每个方案一张（行＝机位；列＝控制稿、seed1/2/3、seed1/2/3 叠控制稿红），
外加 `diagonal-cross` 那一张（图取自 `run-2026-09-05-opening-kind-realism/`、控制稿取自本目录默认方案），
四张同一版式好横向看。同时把每张的分数与耗时从 `.log` 里抄出来打成表（抄进 run.md）。
只在这一批用，写完不改。零模型调用。

跑法（仓根目录）：uv run python _iteration/run-2026-09-05-sketch-symbols/拼图与叠线.py [叠线单张输出目录]
给了第二个参数就把 1280×960 的逐张叠线图也写到那个目录（判读用，不入库）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
PRIOR_REALISM_DIR = ROOT / "_iteration/run-2026-09-05-opening-kind-realism"
CAMERA_IDS = ("cam-room-书房", "cam-room-主卧", "cam-room-客厅")
SEEDS = (1, 2, 3)
TILE = (512, 384)
LABEL_H = 28
COLUMNS = ("控制稿", "seed1", "seed2", "seed3", "seed1+叠线", "seed2+叠线", "seed3+叠线")


def _font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc"):
        try:
            return ImageFont.truetype(candidate, 20)
        except OSError:
            continue
    return ImageFont.load_default()


def overlay(image: Image.Image, sketch: Image.Image) -> Image.Image:
    base = image.convert("RGB")
    mask = sketch.convert("L").resize(base.size, Image.Resampling.NEAREST).point(lambda v: 255 if v > 127 else 0)
    red = Image.new("RGB", base.size, (255, 40, 40))
    return Image.composite(red, base, mask)


def _image_and_log(scheme: str, camera_id: str, seed: int) -> tuple[Path, Path | None]:
    if scheme == "diagonal-cross":
        return PRIOR_REALISM_DIR / f"{camera_id}-seed{seed}.png", None
    out = HERE / scheme / camera_id / f"seed{seed}.png"
    return out, out.with_suffix(".png.log")


def _score_and_elapsed(log: Path | None) -> tuple[str, str, str]:
    if log is None or not log.exists():
        return "—", "—", "—"
    text = log.read_text(encoding="utf-8", errors="replace")
    score = re.search(r"fidelity_score=([0-9.]+)", text)
    elapsed = re.search(r"elapsed_seconds=([0-9.]+)", text)
    exit_code = re.search(r"exit=(\d+)", text)
    return (
        score.group(1) if score else "—",
        elapsed.group(1) if elapsed else "—",
        exit_code.group(1) if exit_code else "—",
    )


def main() -> None:
    overlay_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    font = _font()
    schemes = ("diagonal-cross", "frame-sill", "glazing-hatch")
    print("| 方案 | 机位 | seed | 退出码 | 耗时 s | 分数（对 sketch，尺子照 CLI） |")
    print("|---|---|---|---|---|---|")
    for scheme in schemes:
        montage = Image.new("RGB", (TILE[0] * len(COLUMNS), (TILE[1] + LABEL_H) * len(CAMERA_IDS)), (20, 20, 20))
        draw = ImageDraw.Draw(montage)
        for row, camera_id in enumerate(CAMERA_IDS):
            sketch_path = HERE / scheme / camera_id / "sketch.png"
            sketch = Image.open(sketch_path)
            y = row * (TILE[1] + LABEL_H)
            tiles: list[tuple[str, Image.Image | None]] = [(f"{camera_id} {scheme} 控制稿", sketch.convert("RGB"))]
            overlays: list[tuple[str, Image.Image | None]] = []
            for seed in SEEDS:
                image_path, log = _image_and_log(scheme, camera_id, seed)
                score, elapsed, exit_code = _score_and_elapsed(log)
                if scheme != "diagonal-cross":
                    print(f"| {scheme} | {camera_id} | {seed} | {exit_code} | {elapsed} | {score} |")
                if not image_path.exists():
                    tiles.append((f"seed{seed} 缺", None))
                    overlays.append((f"seed{seed}+叠线 缺", None))
                    continue
                image = Image.open(image_path)
                mixed = overlay(image, sketch)
                tiles.append((f"seed{seed}  分数 {score}", image.convert("RGB")))
                overlays.append((f"seed{seed}+叠线", mixed))
                if overlay_dir is not None:
                    mixed.save(overlay_dir / f"{scheme}-{camera_id}-seed{seed}.overlay.png")
            for col, (label, tile) in enumerate(tiles + overlays):
                x = col * TILE[0]
                draw.text((x + 6, y + 3), label, fill=(255, 200, 80), font=font)
                if tile is not None:
                    montage.paste(tile.resize(TILE), (x, y + LABEL_H))
        montage.save(HERE / f"montage-{scheme}.png")
        print(f"<!-- 拼图：{(HERE / f'montage-{scheme}.png').relative_to(ROOT)} -->")


if __name__ == "__main__":
    main()
