"""16-bit 深度图 / 索引遮罩图 → 8-bit 可视化，喂给扩散模型当条件图用。

depth.png 是 16-bit 灰度：0 = 背景（没打中任何 mesh），非零 = 近/远（数值越大越远，
实测范围约 19000~65535，背景的 0 只是"没打中"的哨兵值，不是"无穷远"，所以不能直接
线性拉伸——那样背景会被读成"最近"，整张图语义反了）。
mask.png 是 16-bit 索引图：0 = 背景，非零 = mesh 下标+1，同一堵墙/同一个家具的所有
像素共享一个索引，数值本身没有大小含义，直接当灰度看毫无意义（两个相邻索引可能是
99 和 100，灰度上几乎看不出差别；也可能是 5 和 300，灰度上却隔了一大截）。

转换规则：
- 深度：背景（0）固定映到黑（0）；前景在 [near, far] 上线性拉伸到 [1, 255]，近处更亮
  （近大远小是深度可视化的常见约定，跟人眼直觉一致：离镜头近的东西"抢眼"）。near/far
  取前景像素的实际 min/max，不用百分位——深度图没有孤立噪声点需要裁掉，min/max 就是
  真实的最近点和最远点。
- 遮罩：背景固定映到黑；每个非零索引按索引数值算一个稳定的色相（HSV 转 RGB），同一个
  索引在任何一次调用里颜色都一样（不依赖出现顺序），相邻索引的颜色也大概率不相近——
  用黄金分割率乘索引数再取小数部分当色相，是让"编号相邻的两个 mesh"不必然颜色相近的
  常见技巧。
"""

from __future__ import annotations

import argparse
import colorsys
import pathlib

import numpy as np
from PIL import Image

_GOLDEN_CONJUGATE = 0.6180339887498949
"""黄金分割共轭，用来把连续递增的索引打散到色相环上，让相邻索引不撞色。"""


def depth_to_visual(depth: np.ndarray) -> np.ndarray:
    """16-bit 深度（0=背景哨兵，非零=近/远）-> 8-bit 灰度（0=背景，1~255=近~远線性拉伸，近亮远暗）。"""
    fg = depth > 0
    if not fg.any():
        raise ValueError("深度图里没有前景像素（全是背景哨兵值 0），量不出近远")
    near = int(depth[fg].min())
    far = int(depth[fg].max())
    out = np.zeros(depth.shape, dtype=np.uint8)
    if far == near:
        out[fg] = 255
    else:
        # 近 -> 255，远 -> 1；254 份线性刻度，0 留给背景专用，两者不会混淆
        scale = (far - depth[fg].astype(np.float64)) / (far - near)
        out[fg] = (1 + scale * 254).round().astype(np.uint8)
    return out


def mask_to_visual(mask: np.ndarray) -> np.ndarray:
    """16-bit 索引遮罩（0=背景，索引=mesh下标+1）-> 8-bit 伪彩 RGB（背景黑，每个索引一个稳定色相）。"""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    ids = np.unique(mask)
    ids = ids[ids != 0]
    for idx in ids:
        hue = (int(idx) * _GOLDEN_CONJUGATE) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        out[mask == idx] = [int(r * 255), int(g * 255), int(b * 255)]
    return out


def convert_dir(src_dir: pathlib.Path) -> None:
    depth_path = src_dir / "depth.png"
    mask_path = src_dir / "mask.png"
    if depth_path.exists():
        depth = np.array(Image.open(depth_path))
        vis = depth_to_visual(depth)
        out_path = src_dir / "depth-vis.png"
        Image.fromarray(vis, mode="L").save(out_path)
        print(f"depth -> {out_path}")
    if mask_path.exists():
        mask = np.array(Image.open(mask_path))
        vis = mask_to_visual(mask)
        out_path = src_dir / "mask-vis.png"
        Image.fromarray(vis, mode="RGB").save(out_path)
        print(f"mask  -> {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="depth.png / mask.png -> 8-bit 可视化条件图")
    parser.add_argument("src_dir", type=pathlib.Path, help="含 depth.png / mask.png 的目录")
    args = parser.parse_args()
    convert_dir(args.src_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
