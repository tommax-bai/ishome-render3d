"""拼图与叠线：奶油风 v2 五张 + modern 基线（16 洞）+ 奶油风 v1（19 洞）并排；下排同图叠控制稿红。

跑法（用 imagegen 的 venv，只要 PIL/numpy）：
    cd ~/codes/ishome-imagegen && uv run python <本文件> [--crops-dir <目录>]
`--crops-dir` 给了就顺带出 2× 放大的局部裁片，供逐张数残留用；不进本目录。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ITER = HERE.parent
STYLES = ITER / "run-2026-09-05-realism-styles"  # 奶油风 v1（19 洞控制稿）
MODERN = ITER / "run-2026-09-05-opening-kind-realism"  # modern 基线（16 洞控制稿）
SKETCH16 = ITER / "run-2026-09-05-opening-kind-consume"  # 16 洞控制稿

TILE = (584, 440)  # 每格缩到原图一半
FONT_PATH = "/System/Library/Fonts/STHeiti Medium.ttc"


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def overlay_sketch(result: Image.Image, sketch: Image.Image) -> Image.Image:
    """控制稿线处涂红叠到结果图上；控制稿按结果图尺寸缩放。

    render3d 落盘的 sketch.png 是黑底白线，imagegen CLI 留档的 .sketch.png 是白底黑线——按中值判极性。
    """
    sk = np.asarray(sketch.convert("L").resize(result.size, Image.Resampling.LANCZOS))
    mask = sk > 128 if np.median(sk) < 128 else sk < 128
    arr = np.asarray(result.convert("RGB")).astype(np.float32)
    red = np.array([255, 0, 0], dtype=np.float32)
    arr[mask] = arr[mask] * 0.15 + red * 0.85
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def tile(img: Image.Image, label: str) -> Image.Image:
    t = img.convert("RGB").resize(TILE, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (TILE[0], TILE[1] + 34), "white")
    canvas.paste(t, (0, 34))
    d = ImageDraw.Draw(canvas)
    d.text((8, 6), label, fill="black", font=font(22))
    return canvas


def montage(columns: list[tuple[str, Path, Path]], out: Path) -> None:
    """columns: (标签, 结果图, 控制稿)。上排原图，下排叠红。"""
    tiles_top = []
    tiles_bot = []
    for label, result_path, sketch_path in columns:
        result = Image.open(result_path)
        sketch = Image.open(sketch_path)
        tiles_top.append(tile(result, label))
        tiles_bot.append(tile(overlay_sketch(result, sketch), label + " 叠控制稿"))
    w = TILE[0] * len(columns)
    h = (TILE[1] + 34) * 2
    sheet = Image.new("RGB", (w, h), "white")
    for i, (a, b) in enumerate(zip(tiles_top, tiles_bot, strict=True)):
        sheet.paste(a, (i * TILE[0], 0))
        sheet.paste(b, (i * TILE[0], TILE[1] + 34))
    sheet.save(out)
    print(f"拼图已出：{out} {sheet.size}")


def crops(crops_dir: Path) -> None:
    """揭顶三张：房子区域上下两半各 2×；主卧两张：中段门洞区域 2×。"""
    crops_dir.mkdir(parents=True, exist_ok=True)
    for seed in (1, 2, 3):
        p = HERE / f"cam-bird-dollhouse-cream-warm-v2-seed{seed}.png"
        im = Image.open(p).convert("RGB")
        # 房子大致落在 x 230–930、y 170–810
        boxes = {"上半": (230, 170, 930, 500), "下半": (230, 470, 930, 810)}
        for name, box in boxes.items():
            c = im.crop(box)
            c = c.resize((c.width * 2, c.height * 2), Image.Resampling.LANCZOS)
            c.save(crops_dir / f"bird-seed{seed}-{name}.png")
    for seed in (1, 2):
        p = HERE / f"cam-room-主卧-cream-warm-v2-seed{seed}.png"
        im = Image.open(p).convert("RGB")
        c = im.crop((120, 250, 620, 720))
        c = c.resize((c.width * 2, c.height * 2), Image.Resampling.LANCZOS)
        c.save(crops_dir / f"room-seed{seed}-中段.png")
    print(f"裁片已出：{crops_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops-dir", type=Path, default=None)
    args = ap.parse_args()

    bird16 = SKETCH16 / "cam-bird-dollhouse" / "sketch.png"
    room16 = SKETCH16 / "cam-room-主卧" / "sketch.png"
    montage(
        [
            ("modern seed1（16洞基线）", MODERN / "cam-bird-dollhouse-seed1.png", bird16),
            (
                "cream v1 seed1（19洞）",
                STYLES / "cam-bird-dollhouse-cream-warm-seed1.png",
                STYLES / "cam-bird-dollhouse-cream-warm-seed1.sketch.png",
            ),
            ("cream v2 seed1", HERE / "cam-bird-dollhouse-cream-warm-v2-seed1.png", bird16),
            ("cream v2 seed2", HERE / "cam-bird-dollhouse-cream-warm-v2-seed2.png", bird16),
            ("cream v2 seed3", HERE / "cam-bird-dollhouse-cream-warm-v2-seed3.png", bird16),
        ],
        HERE / "montage-bird.png",
    )
    montage(
        [
            ("modern seed1（16洞基线）", MODERN / "cam-room-主卧-seed1.png", room16),
            (
                "cream v1 seed1（19洞）",
                STYLES / "cam-room-主卧-cream-warm-seed1.png",
                STYLES / "cam-room-主卧-cream-warm-seed1.sketch.png",
            ),
            ("cream v2 seed1", HERE / "cam-room-主卧-cream-warm-v2-seed1.png", room16),
            ("cream v2 seed2", HERE / "cam-room-主卧-cream-warm-v2-seed2.png", room16),
        ],
        HERE / "montage-room.png",
    )
    if args.crops_dir:
        crops(args.crops_dir)


if __name__ == "__main__":
    main()
