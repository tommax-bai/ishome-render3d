"""平移搜索：把线稿在 ±40 px 内挪，找分数最高的位置——看散布里有多少是"整图错位"。

粗扫步长 8（±40），再在最优点周围步长 2（±6）细扫。默认常量。
产出 平移搜索.md / 平移搜索.json。用法：uv run python 平移搜索.py
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np

import importlib.util
import sys

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("selfcheck", HERE / "度量自证-室内视角.py")
assert spec is not None and spec.loader is not None
S = importlib.util.module_from_spec(spec)
sys.modules["selfcheck"] = S
spec.loader.exec_module(S)

COARSE_STEP, COARSE_RANGE = 8, 40
FINE_STEP, FINE_RANGE = 2, 6


def search(line_arr: np.ndarray, rf: dict, p: float, t: int, a: float) -> tuple[float, tuple[int, int], float, dict]:
    cache: dict[tuple[int, int], float] = {}

    def score_at(dx: int, dy: int) -> float:
        if (dx, dy) not in cache:
            lf = S.line_features(S.transform_line(line_arr, shift=(dx, dy)))
            cache[(dx, dy)] = S.score_of(S.hit_array(lf, rf, p, t, a), lf["valid"])
        return cache[(dx, dy)]

    base = score_at(0, 0)
    grid = range(-COARSE_RANGE, COARSE_RANGE + 1, COARSE_STEP)
    best = max(((score_at(dx, dy), (dx, dy)) for dx in grid for dy in grid), key=lambda x: x[0])
    bx, by = best[1]
    fine = [(dx, dy) for dx in range(bx - FINE_RANGE, bx + FINE_RANGE + 1, FINE_STEP) for dy in range(by - FINE_RANGE, by + FINE_RANGE + 1, FINE_STEP)]
    best = max(((score_at(dx, dy), (dx, dy)) for dx, dy in fine), key=lambda x: x[0])
    return base, best[1], best[0], {f"{k[0]},{k[1]}": v for k, v in cache.items()}


def main() -> int:
    t0 = time.time()
    P, T, A = S.DEFAULTS
    size = S.load_line(S.ROOM_LINE["客厅"])[1]
    rows = []
    md = ["## 平移搜索（默认常量；线稿挪 (dx,dy) 后的最高分；dx 向右、dy 向下为正）\n",
          "| 出图 | 原位分数 | 最优位移 (dx,dy) px | 最优分数 | 提升 |", "|---|---|---|---|---|"]
    jobs = []
    for r in S.ROOM_IMAGES:
        arr = S.load_line(S.ROOM_LINE[r])[0]
        for name in S.ROOM_IMAGES[r]:
            jobs.append((name, arr, S.result_features(S.ROOM_IMG_DIR / f"{name}.png", size)))
        jobs.append((f"{r} geometry.png（自量）", arr, S.result_features(S.ROOM_GEOM[r], size)))
    bird_arr = S.load_line(S.BIRD_LINE)[0]
    for name in S.BIRD_IMAGES:
        jobs.append((f"揭顶 {name}", bird_arr, S.result_features(S.BIRD_IMG_DIR / f"{name}.png", size)))
    for name, arr, rf in jobs:
        base, best_xy, best, grid = search(arr, rf, P, T, A)
        rows.append({"图": name, "原位": base, "最优位移": best_xy, "最优分数": best, "网格": grid})
        md.append(f"| {name} | {base:.4f} | ({best_xy[0]:+d},{best_xy[1]:+d}) | {best:.4f} | {best - base:+.4f} |")
        print(f"{name}: 原位 {base:.4f} 最优 {best_xy} {best:.4f} ({time.time() - t0:.0f}s)", flush=True)
    (HERE / "平移搜索.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (HERE / "平移搜索.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
