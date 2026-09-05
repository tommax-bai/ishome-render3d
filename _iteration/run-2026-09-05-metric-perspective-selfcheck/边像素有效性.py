"""线稿边像素里，尺子实际计入分母的（"走向有定义"）是哪些——按走向分。

尺子第 4 步：线稿 3x3 均值模糊后做 Sobel，梯度幅值为 0 的线像素"方向无定义"，不进分母。
一条 1 px 宽、正竖/正横的直线，模糊后是 3 px 宽的平台，正中那一列（也就是线像素本身）
Sobel 为 0——所以这类像素**全部不进分母**。这里把线像素按局部形状分成
"正竖直线段 / 正横直线段 / 其他（斜线、拐点、2 px 粗处）"，数各类里有多少进了分母。
产出 边像素有效性.md / 边像素有效性.png。用法：uv run python 边像素有效性.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("selfcheck", HERE / "度量自证-室内视角.py")
assert spec is not None and spec.loader is not None
S = importlib.util.module_from_spec(spec)
sys.modules["selfcheck"] = S
spec.loader.exec_module(S)


def shape_classes(line_mask: np.ndarray) -> dict[str, np.ndarray]:
    up = np.zeros_like(line_mask); up[1:] = line_mask[:-1]
    down = np.zeros_like(line_mask); down[:-1] = line_mask[1:]
    left = np.zeros_like(line_mask); left[:, 1:] = line_mask[:, :-1]
    right = np.zeros_like(line_mask); right[:, :-1] = line_mask[:, 1:]
    vertical = line_mask & up & down & ~left & ~right
    horizontal = line_mask & left & right & ~up & ~down
    other = line_mask & ~vertical & ~horizontal
    return {"正竖直线段": vertical, "正横直线段": horizontal, "其他（斜线/拐点/粗处）": other}


def main() -> int:
    md = ["## 线稿边像素：哪些进了尺子的分母（valid＝模糊后 Sobel 幅值 > 0）\n",
          "| 线稿 | 线像素总数 | 进分母 | 占比 | 正竖直线段：总/进分母 | 正横直线段：总/进分母 | 其他：总/进分母 |",
          "|---|---|---|---|---|---|---|"]
    panels = []
    jobs = [("客厅 line.png", S.ROOM_LINE["客厅"]), ("主卧 line.png", S.ROOM_LINE["主卧"]), ("揭顶 line.png", S.BIRD_LINE)]
    for name, path in jobs:
        arr, size = S.load_line(path)
        lf = S.line_features(arr)
        mask, valid = lf["mask"], lf["valid"]
        cls = shape_classes(mask)
        cells = []
        for c, m in cls.items():
            cells.append(f"{int(m.sum())} / {int((m & valid).sum())}")
        md.append(f"| {name} | {int(mask.sum())} | {int(valid.sum())} | {valid.sum() / mask.sum():.3f} | " + " | ".join(cells) + " |")
        canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        canvas[mask & ~valid] = (110, 110, 110)   # 灰＝线像素但不进分母
        canvas[valid] = (255, 255, 0)              # 黄＝进分母
        panels.append((name, Image.fromarray(canvas)))
    (HERE / "边像素有效性.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    w, h = panels[0][1].size
    out = Image.new("RGB", (w * 3 + 20, h + 40), (20, 20, 20))
    d = ImageDraw.Draw(out)
    font = S._font(24)
    for k, (name, im) in enumerate(panels):
        out.paste(im, (k * (w + 10), 40))
        d.text((k * (w + 10) + 6, 6), f"{name}  黄＝进分母  灰＝线像素但方向无定义、不进分母", fill=(255, 255, 255), font=font)
    out.save(HERE / "边像素有效性.png")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
