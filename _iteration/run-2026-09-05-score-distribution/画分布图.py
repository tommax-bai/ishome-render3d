"""把 分布.csv 画成按分档的点图：每行一个分档×对照物，三栏＝v2 / v3 原位 / v3 配准；正常●、错配×、转 90°▲。

跑法：`cd ~/codes/ishome-imagegen && uv run --with matplotlib python 画分布图.py`（matplotlib 不在本仓依赖里，临时拉）。
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

matplotlib.use("Agg")

OUT_DIR = Path("/Users/baitianxing/codes/ishome-render3d.wt-score-dist/_iteration/run-2026-09-05-score-distribution")
CSV_PATH = OUT_DIR / "分布.csv"
PNG_PATH = OUT_DIR / "分布.png"

# 中文字体：按机器上有的取
_CANDIDATES = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "Heiti SC", "STHeiti", "Noto Sans CJK SC"]
_available = {f.name for f in font_manager.fontManager.ttflist}
_chosen = [c for c in _CANDIDATES if c in _available]
print("可用中文字体:", _chosen)
plt.rcParams["font.family"] = _chosen + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
rows = [r for r in rows if r["对照类型"] != "核对"]

# 行＝分档×对照物；顺序：视角、控制稿版本、风格、陈设、对照物
groups: "OrderedDict[str, dict[str, list[dict]]]" = OrderedDict()
keyed = sorted(rows, key=lambda r: (r["视角"], r["控制稿版本"], r["风格"], r["陈设"], r["对照物"]))
for r in keyed:
    label = f'{r["视角"]}·{r["控制稿版本"]}·{r["风格"]}·{r["陈设"]}  [{r["对照物"]}]'
    groups.setdefault(label, {"正常": [], "错配": [], "转90": []})[r["对照类型"]].append(r)

labels = list(groups.keys())
n_rows = len(labels)
metrics = [("v2", "v2"), ("v3原位", "v3 原位"), ("v3配准", "v3 配准（±24 px 平移）")]
fig, axes = plt.subplots(1, 3, figsize=(17, 0.42 * n_rows + 2.2), sharey=True)
style = {"正常": dict(marker="o", s=34, facecolors="#1f5fa8", edgecolors="#1f5fa8", alpha=0.85, zorder=3),
         "错配": dict(marker="x", s=44, c="#c0392b", linewidths=1.4, zorder=4),
         "转90": dict(marker="^", s=40, facecolors="none", edgecolors="#7f8c8d", linewidths=1.2, zorder=4)}
for ax, (col, title) in zip(axes, metrics):
    for yi, label in enumerate(labels):
        g = groups[label]
        n = len(g["正常"])
        if n < 5:
            ax.axhspan(yi - 0.5, yi + 0.5, color="#f2f2f2", zorder=0)
        for kind, lst in g.items():
            xs = [float(r[col]) for r in lst]
            if not xs:
                continue
            # 同档内小抖动，免得点叠死
            ys = [yi + (i - (len(xs) - 1) / 2) * (0.5 / max(len(xs), 1)) for i in range(len(xs))]
            ax.scatter(xs, ys, label=kind if (yi == 0 and ax is axes[0]) else None, **style[kind])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"{l}  n={len(groups[l]['正常'])}" for l in labels], fontsize=8)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xlim(0, 0.92)
    ax.set_xlabel(title, fontsize=10)
    ax.grid(axis="x", color="#e5e5e5", zorder=0)
    ax.tick_params(axis="x", labelsize=8)
axes[0].legend(loc="lower right", fontsize=8, frameon=True, title="●正常 ×错配 ▲转90°", title_fontsize=8)
fig.suptitle("保真度分数分布（按输出形态分档；灰底＝n<5 不能定线；分数只能同档横向比）", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(PNG_PATH, dpi=130)
print("写", PNG_PATH)
