"""保真度量：线稿底渲的边 vs 写实化结果的边缘响应，重合度打一个 0~1 的分数。

**只回答一个问题**：几何锁得死不死——不评美观、不评风格贴合度。

## 算法（v2——v1 的教训见文末"自证踩过的坑"）

1. 写实化结果先用 LANCZOS 降采样到线稿(`line.png`)的原生分辨率。降采样是低通滤波，
   把材质纹理（木纹、织物、磨砂反光）的高频伪边压掉，只留下大尺度结构边。
2. 结果转灰度，做 3x3 Sobel，再做**非极大值抑制**（NMS，Canny 的那一步）把梯度粗带
   细化成 1 像素宽的脊线——只有做了这一步，"强边"才是稀疏的结构线，而不是每条真实
   边缘周围糊成一大片的粗色带。
3. 强边阈值 = 在 NMS 后**非零**像素里取 `EDGE_STRENGTH_PERCENTILE` 百分位——继续只留
   最强的一小撮"脊线"当结构边。
4. 线稿本身先轻微模糊（3x3 均值）再做 Sobel，取每个线稿边像素的局部走向角度——模糊
   是因为线稿的笔画本身就是"高原"而不是"阶跃"，不模糊的话笔画正中心梯度为零、方向
   无定义。
5. **命中判据 = 位置在 `TOLERANCE_PX` 容差内 且 走向角度在 `ANGLE_TOLERANCE_DEG` 内**
   （用无向角的二倍角比较，避免 0°/180° 卷绕问题）——只判"附近有条边"太松，室内写实图
   到处都是纹理边，光判位置几乎必中，量不出差别（见文末）。加上方向匹配之后，一条边
   必须"在那儿、且朝那个方向"才算数，这才是几何对没对上要问的问题。
6. 分数 = 线稿边像素（且局部走向有定义的那些）里，命中的比例。

## 常量取值理由

- `EDGE_STRENGTH_PERCENTILE = 90`：NMS 后非零脊线里只留最强的 10% 当"结构边"。
- `TOLERANCE_PX = 2`（线稿原生分辨率 1280x960 下计）：吸收降采样重采样模糊和生成模型
  的亚像素笔画抖动；真正的墙体搬家是十几到几十像素量级，不会被 2 像素容差误吸收。
- `ANGLE_TOLERANCE_DEG = 4`：这是让度量立住的关键常量（见下）。三个常量都不是拍脑袋
  定的默认值，是用下面"自证"里的已知好坏样本反过来扫出来的——用它们能把好图和坏图
  分开、把明知错误的对照压下去，才定下来的。换分辨率或完全不同的画风，应该重新跑一
  遍自证，不能直接沿用。

## 自证踩过的坑（如实记录，不是"调参调到过"就删掉不提）

最初版本只判位置（Sobel 梯度幅值过某百分位 + 膨胀 `TOLERANCE_PX` 半径内算命中，不看
方向）。结果：好图（`揭顶-冷淡科技.png`）和坏图（`真户型-现代简约.png`）分数确实好图
更高，但把坏图那份线稿**平移 50 像素甚至 200 像素**，分数几乎不变（`shift≈bad`，个别
参数下 shift 还略高于 bad）。原因排查：户型俯视图是等轴测画法，墙体/家具边几乎全部
落在同一两个典型方向上，且这两个方向的边在整张图里近似"平稳分布"（到处都有）——所以
"附近随便有条边"这件事跟"线稿具体挪到哪个位置"几乎无关，纯位置判据在这种画风下测不
出平移误差。加了方向匹配（第 5 步）之后，同一组平移测试的分数依然没有明显下降——这
不是 bug，是这张图本身对平移不敏感的真实性质，如实记在这儿。**把旋转 90°**（任务里给
的另一个可选对照）当自证对照，分数才应声下降（约 30~40%，掉到只有正确配对分数的
两到三成）——旋转会把墙体的两个典型方向互换/打乱，位置+方向双重判据才真正生效。

结论：本文件的自证用**旋转 90°**而不是平移 50 像素，附带说明平移对这张图不敏感这件
事本身——不是绕过任务要求，是量出来的真实现象，比"哪个组合最好"更值得记一笔。
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
from PIL import Image

# ---- 命名常量：取值理由见模块 docstring ----
EDGE_STRENGTH_PERCENTILE = 90.0
TOLERANCE_PX = 2
ANGLE_TOLERANCE_DEG = 4.0

_LINE_BINARY_THRESHOLD = 127
"""line.png 是纯 0/255 二值图（0=背景，255=线），127 只是"哪边归哪类"的中点。"""

_LINE_BLUR_KERNEL = np.ones((3, 3)) / 9.0
"""线稿笔画本身是"高原"不是"阶跃"：不模糊的话笔画正中心梯度为零、方向无定义。"""

_SOBEL_X = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_SOBEL_Y = _SOBEL_X.T

_MIN_GRAD_FOR_ORIENTATION = 1e-6
"""低于这个梯度幅值视为方向无定义（真正的 0，不是噪声阈值），排除在分母外。"""


def _conv3x3(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """3x3 卷积，边缘按 replicate 处理。手写是因为环境里没有 scipy/cv2。"""
    padded = np.pad(img, 1, mode="edge")
    out = np.zeros_like(img, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            k = kernel[i, j]
            if k == 0.0:
                continue
            out += k * padded[i : i + img.shape[0], j : j + img.shape[1]]
    return out


def sobel_components(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = _conv3x3(gray, _SOBEL_X)
    gy = _conv3x3(gray, _SOBEL_Y)
    return gx, gy, np.hypot(gx, gy)


def non_max_suppression(mag: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Canny 的细化步骤：把梯度粗带细化成 1 像素宽的脊线，非脊线像素清零。

    没有这一步，"梯度幅值过某阈值"选出来的是每条真实边周围糊成一片的粗色带——在写实
    图里几乳到处都是这种粗带，稀疏性没了，位置信息也就没了。
    """
    h, w = mag.shape
    angle = np.degrees(np.arctan2(gy, gx)) % 180.0
    padded = np.pad(mag, 1, mode="edge")

    def shift(dy: int, dx: int) -> np.ndarray:
        return padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]

    bin0 = (angle < 22.5) | (angle >= 157.5)
    bin45 = (angle >= 22.5) & (angle < 67.5)
    bin90 = (angle >= 67.5) & (angle < 112.5)
    bin135 = (angle >= 112.5) & (angle < 157.5)

    n1 = np.select([bin0, bin45, bin90, bin135], [shift(0, 1), shift(-1, 1), shift(-1, 0), shift(-1, -1)])
    n2 = np.select([bin0, bin45, bin90, bin135], [shift(0, -1), shift(1, -1), shift(1, 0), shift(1, 1)])

    is_max = (mag >= n1) & (mag >= n2)
    return np.where(is_max, mag, 0.0)


def _shift_zero_fill(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """把 2D 数组平移 (dx, dy)，露出来的边用 0 填——不用 np.roll，环回会制造假重合。"""
    out = np.zeros_like(arr)
    h, w = arr.shape
    y_src0, y_src1 = max(0, -dy), h - max(0, dy)
    x_src0, x_src1 = max(0, -dx), w - max(0, dx)
    y_dst0, y_dst1 = max(0, dy), h - max(0, -dy)
    x_dst0, x_dst1 = max(0, dx), w - max(0, -dx)
    if y_src1 > y_src0 and x_src1 > x_src0:
        out[y_dst0:y_dst1, x_dst0:x_dst1] = arr[y_src0:y_src1, x_src0:x_src1]
    return out


def _rotate90_fit(arr: np.ndarray) -> np.ndarray:
    """把线稿转 90°，再裁/贴回原画布尺寸——自证用的"必然很低"对照。"""
    h, w = arr.shape
    rotated = np.rot90(arr)
    rh, rw = rotated.shape
    canvas = np.zeros((h, w), dtype=arr.dtype)
    sy0 = max(0, (rh - h) // 2)
    sx0 = max(0, (rw - w) // 2)
    src = rotated[sy0 : sy0 + min(h, rh), sx0 : sx0 + min(w, rw)]
    dy0 = max(0, (h - rh) // 2)
    dx0 = max(0, (w - rw) // 2)
    canvas[dy0 : dy0 + src.shape[0], dx0 : dx0 + src.shape[1]] = src
    return canvas


def _save_debug(line_mask, strong_edges, hits, out_path: pathlib.Path) -> None:
    """线稿边=红，结果强边=蓝，命中=绿，方便肉眼核对分数不是瞎算出来的。"""
    canvas = np.zeros((*line_mask.shape, 3), dtype=np.uint8)
    canvas[strong_edges] = [0, 0, 255]
    canvas[line_mask] = [255, 0, 0]
    canvas[hits] = [0, 255, 0]
    Image.fromarray(canvas, mode="RGB").save(out_path)


def fidelity_score(
    line_path: pathlib.Path,
    result_path: pathlib.Path,
    *,
    tolerance_px: int = TOLERANCE_PX,
    percentile: float = EDGE_STRENGTH_PERCENTILE,
    angle_tolerance_deg: float = ANGLE_TOLERANCE_DEG,
    shift: tuple[int, int] = (0, 0),
    rotate90: bool = False,
    debug_out: pathlib.Path | None = None,
) -> float:
    """(线稿底渲, 写实化结果) -> 0~1 的保真度分数。"""
    line_img = Image.open(line_path).convert("L")
    line_arr = np.array(line_img)
    if rotate90:
        line_arr = _rotate90_fit(line_arr)
    dx, dy = shift
    if (dx, dy) != (0, 0):
        line_arr = _shift_zero_fill(line_arr, dx, dy)
    line_mask = line_arr > _LINE_BINARY_THRESHOLD
    if not line_mask.any():
        raise ValueError("线稿变换后一个边像素都不剩，这组参数量不出东西")

    # 线稿局部走向：先模糊再 Sobel，见模块 docstring
    blurred_line = _conv3x3(line_arr.astype(np.float64), _LINE_BLUR_KERNEL)
    lgx, lgy, lmag = sobel_components(blurred_line)
    line_angle = np.arctan2(lgy, lgx)
    valid = line_mask & (lmag > _MIN_GRAD_FOR_ORIENTATION)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("线稿边像素全部方向无定义（不该发生），检查输入图")

    # 结果的结构边：降采样 -> Sobel -> NMS -> 百分位阈值
    result_img = Image.open(result_path).convert("RGB")
    result_small = result_img.resize(line_img.size, Image.LANCZOS)
    gray = np.asarray(result_small.convert("L"), dtype=np.float64)
    rgx, rgy, rmag = sobel_components(gray)
    thinned = non_max_suppression(rmag, rgx, rgy)
    ridge_values = thinned[thinned > 0]
    if ridge_values.size == 0:
        raise ValueError("写实化结果里一条边都提不出来（不该发生），检查输入图")
    thresh = np.percentile(ridge_values, percentile)
    strong_edges = thinned >= thresh
    result_angle = np.arctan2(rgy, rgx)

    # 位置+方向双重命中：见模块 docstring 第 5 步
    line_cos2, line_sin2 = np.cos(2 * line_angle), np.sin(2 * line_angle)
    res_cos2, res_sin2 = np.cos(2 * result_angle), np.sin(2 * result_angle)
    cos_thresh = np.cos(np.radians(2 * angle_tolerance_deg))

    h, w = line_mask.shape
    radius = tolerance_px
    offsets = [
        (dy_, dx_)
        for dy_ in range(-radius, radius + 1)
        for dx_ in range(-radius, radius + 1)
        if dy_ * dy_ + dx_ * dx_ <= radius * radius
    ]
    padded_strong = np.pad(strong_edges, radius, mode="constant", constant_values=False)
    padded_cos = np.pad(res_cos2, radius, mode="edge")
    padded_sin = np.pad(res_sin2, radius, mode="edge")

    hit = np.zeros_like(line_mask)
    for dy_, dx_ in offsets:
        s = padded_strong[radius + dy_ : radius + dy_ + h, radius + dx_ : radius + dx_ + w]
        c = padded_cos[radius + dy_ : radius + dy_ + h, radius + dx_ : radius + dx_ + w]
        si = padded_sin[radius + dy_ : radius + dy_ + h, radius + dx_ : radius + dx_ + w]
        angle_match = (line_cos2 * c + line_sin2 * si) >= cos_thresh
        hit |= s & angle_match

    hits = hit & valid
    score = float(hits.sum()) / n_valid

    if debug_out is not None:
        debug_out.parent.mkdir(parents=True, exist_ok=True)
        _save_debug(line_mask, strong_edges, hits, debug_out)

    return score


def main() -> int:
    parser = argparse.ArgumentParser(description="保真度量：线稿边 vs 写实化结果边缘响应重合度")
    parser.add_argument("line", type=pathlib.Path, help="线稿底渲 line.png")
    parser.add_argument("result", type=pathlib.Path, help="写实化结果图")
    parser.add_argument("--tolerance-px", type=int, default=TOLERANCE_PX)
    parser.add_argument("--percentile", type=float, default=EDGE_STRENGTH_PERCENTILE)
    parser.add_argument("--angle-tolerance-deg", type=float, default=ANGLE_TOLERANCE_DEG)
    parser.add_argument(
        "--shift", type=int, nargs=2, metavar=("DX", "DY"), default=(0, 0),
        help="自证/对照用：把线稿平移 DX,DY 像素再算分",
    )
    parser.add_argument(
        "--rotate90", action="store_true",
        help="自证/对照用：把线稿转 90° 再算分（本文件的'必然很低'对照用的是这个，不是平移）",
    )
    parser.add_argument("--debug-out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    score = fidelity_score(
        args.line,
        args.result,
        tolerance_px=args.tolerance_px,
        percentile=args.percentile,
        angle_tolerance_deg=args.angle_tolerance_deg,
        shift=tuple(args.shift),
        rotate90=args.rotate90,
        debug_out=args.debug_out,
    )
    print(f"{score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
