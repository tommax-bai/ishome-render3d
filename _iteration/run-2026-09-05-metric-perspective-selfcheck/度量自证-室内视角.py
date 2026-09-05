"""保真度尺子在室内透视视角下的自证（2026-09-05）。只做分析，不改尺子、不调模型。

尺子 = `../run-2026-09-01-condition-matrix/保真度量.py`（按路径 import，常量与函数原样用）。
本脚本另写一遍同样的流水线只为拿到逐像素的命中数组（尺子的 `fidelity_score` 只返回分数）；
每一组默认常量下的分数都与 `fidelity_score` 逐一对账（`assert`），对不上就崩，不会静默偏离。

产出（全在本目录）：
    结果.json                 全部数字
    结果表.md                 自动生成的表格（run.md 引用其中的表）
    debug/*.png               尺子自带 --debug-out 的调试图（线稿红、强边蓝、命中绿）
    montage-调试.png          8 张室内出图的调试图拼图
    montage-叠原图.png        调试图叠在出图上（半透明），看命中落在图上什么东西
    线稿边分类.png            客厅/主卧线稿按遮罩语义分类上色

用法：uv run python 度量自证-室内视角.py
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import pathlib
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).resolve().parent
ITER = HERE.parent
METRIC_PATH = ITER / "run-2026-09-01-condition-matrix" / "保真度量.py"

spec = importlib.util.spec_from_file_location("baozhen", METRIC_PATH)
assert spec is not None and spec.loader is not None
M = importlib.util.module_from_spec(spec)
sys.modules["baozhen"] = M
spec.loader.exec_module(M)

# ---- 样本（都是真跑存档，只读） ----
CAT = ITER / "run-2026-09-04-failure-catalogue"
ROOM_IMG_DIR = CAT / "万相-室内机位"
ROOM_LINE = {r: CAT / "底渲-室内机位" / f"cam-room-{r}" / "line.png" for r in ("客厅", "主卧")}
ROOM_GEOM = {r: CAT / "底渲-室内机位" / f"cam-room-{r}" / "geometry.png" for r in ("客厅", "主卧")}
ROOM_MASK = {r: HERE / "底渲-室内机位-四路" / f"cam-room-{r}" for r in ("客厅", "主卧")}
ROOM_IMAGES = {
    "客厅": ["客厅-nofurn-seed1", "客厅-nofurn-seed2", "客厅-nofurn-seed3", "客厅-default-seed1"],
    "主卧": ["主卧-nofurn-seed1", "主卧-nofurn-seed2", "主卧-nofurn-seed3", "主卧-default-seed1"],
}
OTHER = {"客厅": "主卧", "主卧": "客厅"}

BIRD = ITER / "真户型-基准" / "底渲-cam-bird-dollhouse"
BIRD_LINE = BIRD / "line.png"
BIRD_GEOM = BIRD / "geometry.png"
BIRD_IMG_DIR = ITER / "run-2026-09-02-control-path-survey" / "万相-doodle"
BIRD_IMAGES = ["run1-seed12345", "run2-seed12345-repeat", "run4-seed1001", "run5-seed1002"]

SHIFTS_PX = (20, 50, 100)
SWEEP_PERCENTILE = (85.0, 90.0, 95.0)
SWEEP_TOL = (2, 3, 4)
SWEEP_ANGLE = (4.0, 6.0, 8.0)
DEFAULTS = (M.EDGE_STRENGTH_PERCENTILE, M.TOLERANCE_PX, M.ANGLE_TOLERANCE_DEG)

# 遮罩语义 → 边的类别。线稿的线标在两块面交界处，看线像素半径 2 内出现了哪些语义。
SEM_CODE = {"bg": 0, "wall": 1, "floor": 2, "ceiling": 3, "reveal": 4, "furnishing": 5}
CLASS_NAMES = ["墙面边", "天花边", "地面边", "洞口边", "背景边"]
CLASS_COLOR = {
    "墙面边": (255, 255, 255),
    "天花边": (255, 80, 80),
    "地面边": (80, 160, 255),
    "洞口边": (255, 200, 0),
    "背景边": (0, 220, 0),
}
CLASS_RADIUS_PX = 2


# ---------------------------------------------------------------- 流水线（与尺子同一算法，只是把中间量露出来）
def load_line(path: pathlib.Path) -> tuple[np.ndarray, tuple[int, int]]:
    img = Image.open(path).convert("L")
    return np.array(img), img.size


def transform_line(arr: np.ndarray, *, rotate90: bool = False, shift=(0, 0)) -> np.ndarray:
    if rotate90:
        arr = M._rotate90_fit(arr)
    if tuple(shift) != (0, 0):
        arr = M._shift_zero_fill(arr, shift[0], shift[1])
    return arr


def line_features(line_arr: np.ndarray) -> dict:
    line_mask = line_arr > M._LINE_BINARY_THRESHOLD
    blurred = M._conv3x3(line_arr.astype(np.float64), M._LINE_BLUR_KERNEL)
    lgx, lgy, lmag = M.sobel_components(blurred)
    ang = np.arctan2(lgy, lgx)
    valid = line_mask & (lmag > M._MIN_GRAD_FOR_ORIENTATION)
    return {"mask": line_mask, "cos2": np.cos(2 * ang), "sin2": np.sin(2 * ang), "valid": valid}


def result_features(path: pathlib.Path, size: tuple[int, int]) -> dict:
    img = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    gray = np.asarray(img.convert("L"), dtype=np.float64)
    rgx, rgy, rmag = M.sobel_components(gray)
    thinned = M.non_max_suppression(rmag, rgx, rgy)
    ang = np.arctan2(rgy, rgx)
    return {"thinned": thinned, "ridge": thinned[thinned > 0], "cos2": np.cos(2 * ang), "sin2": np.sin(2 * ang)}


def hit_array(lf: dict, rf: dict, percentile: float, tol: int, angle_deg: float) -> np.ndarray:
    thresh = np.percentile(rf["ridge"], percentile)
    strong = rf["thinned"] >= thresh
    cos_thresh = np.cos(np.radians(2 * angle_deg))
    h, w = lf["mask"].shape
    r = tol
    offsets = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dy * dy + dx * dx <= r * r]
    ps = np.pad(strong, r, mode="constant", constant_values=False)
    pc = np.pad(rf["cos2"], r, mode="edge")
    psn = np.pad(rf["sin2"], r, mode="edge")
    hit = np.zeros_like(lf["mask"])
    for dy, dx in offsets:
        s = ps[r + dy : r + dy + h, r + dx : r + dx + w]
        c = pc[r + dy : r + dy + h, r + dx : r + dx + w]
        si = psn[r + dy : r + dy + h, r + dx : r + dx + w]
        hit |= s & ((lf["cos2"] * c + lf["sin2"] * si) >= cos_thresh)
    return hit & lf["valid"]


def score_of(hits: np.ndarray, valid: np.ndarray, where: np.ndarray | None = None) -> float | None:
    if where is not None:
        hits, valid = hits & where, valid & where
    n = int(valid.sum())
    return None if n == 0 else float(hits.sum()) / n


# ---------------------------------------------------------------- 线稿边按遮罩语义分类
def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    h, w = mask.shape
    p = np.pad(mask, r, mode="constant", constant_values=False)
    out = np.zeros_like(mask)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx <= r * r:
                out |= p[r + dy : r + dy + h, r + dx : r + dx + w]
    return out


def classify_line_pixels(mask_dir: pathlib.Path) -> np.ndarray:
    """返回整幅的类别图（0..4 对应 CLASS_NAMES），只在线像素处有意义。"""
    mask_arr = np.array(Image.open(mask_dir / "mask.png"))
    index = json.loads((mask_dir / "mask-index.json").read_text(encoding="utf-8"))
    lut = np.zeros(int(mask_arr.max()) + 1, dtype=np.int64)
    for e in index:
        lut[e["index"]] = SEM_CODE[e["semantic"]]
    sem = lut[mask_arr]
    has = {name: _dilate(sem == code, CLASS_RADIUS_PX) for name, code in SEM_CODE.items()}
    cls = np.full(mask_arr.shape, CLASS_NAMES.index("墙面边"), dtype=np.int64)
    # 优先级：天花 > 地面 > 洞口 > 背景 > 只剩墙
    cls[has["bg"]] = CLASS_NAMES.index("背景边")
    cls[has["reveal"]] = CLASS_NAMES.index("洞口边")
    cls[has["floor"]] = CLASS_NAMES.index("地面边")
    cls[has["ceiling"]] = CLASS_NAMES.index("天花边")
    return cls


# ---------------------------------------------------------------- 主流程
def fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.4f}"


def main() -> int:
    t0 = time.time()
    out: dict = {"常量默认": {"percentile": DEFAULTS[0], "tolerance_px": DEFAULTS[1], "angle_deg": DEFAULTS[2]}}
    md: list[str] = []

    # ---- 载入 ----
    lines = {r: load_line(p) for r, p in ROOM_LINE.items()}
    size = lines["客厅"][1]
    assert lines["主卧"][1] == size
    bird_arr, bird_size = load_line(BIRD_LINE)
    assert bird_size == size

    lf_room = {r: line_features(arr) for r, (arr, _) in lines.items()}
    lf_bird = line_features(bird_arr)
    cls_room = {r: classify_line_pixels(ROOM_MASK[r]) for r in ROOM_LINE}

    rf_room = {name: result_features(ROOM_IMG_DIR / f"{name}.png", size) for r in ROOM_IMAGES for name in ROOM_IMAGES[r]}
    rf_geom = {r: result_features(ROOM_GEOM[r], size) for r in ROOM_LINE}
    rf_bird = {name: result_features(BIRD_IMG_DIR / f"{name}.png", size) for name in BIRD_IMAGES}
    rf_bird_geom = result_features(BIRD_GEOM, size)
    print(f"载入完成 {time.time() - t0:.1f}s", flush=True)

    P, T, A = DEFAULTS

    # ---- 对账：自写流水线 vs 尺子本体（默认常量，正确配对 8 张 + 转 90° 2 张 + 平移 1 张）----
    checks = []
    for r in ROOM_IMAGES:
        for name in ROOM_IMAGES[r]:
            mine = score_of(hit_array(lf_room[r], rf_room[name], P, T, A), lf_room[r]["valid"])
            ref = M.fidelity_score(ROOM_LINE[r], ROOM_IMG_DIR / f"{name}.png")
            checks.append((name, mine, ref))
            assert abs(mine - ref) < 1e-12, (name, mine, ref)
    for r in ROOM_IMAGES:
        name = ROOM_IMAGES[r][0]
        lf_rot = line_features(transform_line(lines[r][0], rotate90=True))
        mine = score_of(hit_array(lf_rot, rf_room[name], P, T, A), lf_rot["valid"])
        ref = M.fidelity_score(ROOM_LINE[r], ROOM_IMG_DIR / f"{name}.png", rotate90=True)
        assert abs(mine - ref) < 1e-12, ("rot90", name, mine, ref)
        lf_sh = line_features(transform_line(lines[r][0], shift=(50, 0)))
        mine = score_of(hit_array(lf_sh, rf_room[name], P, T, A), lf_sh["valid"])
        ref = M.fidelity_score(ROOM_LINE[r], ROOM_IMG_DIR / f"{name}.png", shift=(50, 0))
        assert abs(mine - ref) < 1e-12, ("shift", name, mine, ref)
    out["对账"] = [{"图": n, "自写": m, "尺子": r} for n, m, r in checks]
    print("对账通过：自写流水线与 保真度量.py 逐一相等", flush=True)

    # ---- 表一：室内视角，默认常量，正对照 + 负对照 ----
    md.append("## 表一 室内视角 · 默认常量（percentile 90 / tol 2 px / angle 4°）\n")
    md.append("| 出图 | 正确配对 | 错配（另一间的线稿） | 转 90° | 平移 x20 | x50 | x100 | 平移 y20 | y50 | y100 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    table1: dict = {}
    lf_variants: dict = {}
    for r in ROOM_LINE:
        arr = lines[r][0]
        lf_variants[r] = {
            "rot90": line_features(transform_line(arr, rotate90=True)),
            **{f"x{s}": line_features(transform_line(arr, shift=(s, 0))) for s in SHIFTS_PX},
            **{f"y{s}": line_features(transform_line(arr, shift=(0, s))) for s in SHIFTS_PX},
        }
    hits_default: dict = {}
    for r in ROOM_IMAGES:
        for name in ROOM_IMAGES[r]:
            rf = rf_room[name]
            row = {}
            h_ok = hit_array(lf_room[r], rf, P, T, A)
            hits_default[name] = h_ok
            row["正确配对"] = score_of(h_ok, lf_room[r]["valid"])
            o = OTHER[r]
            h_mis = hit_array(lf_room[o], rf, P, T, A)
            hits_default[f"{name}|错配"] = h_mis
            row["错配"] = score_of(h_mis, lf_room[o]["valid"])
            for key, lf in lf_variants[r].items():
                row[key] = score_of(hit_array(lf, rf, P, T, A), lf["valid"])
            table1[name] = row
            md.append(
                f"| {name} | {fmt(row['正确配对'])} | {fmt(row['错配'])} | {fmt(row['rot90'])} | "
                + " | ".join(fmt(row[f"x{s}"]) for s in SHIFTS_PX)
                + " | "
                + " | ".join(fmt(row[f"y{s}"]) for s in SHIFTS_PX)
                + " |"
            )
    # 底渲 geometry 自量
    md.append("")
    md.append("| 底渲 geometry 自量 | 自家 line | 另一间 line | 转 90° | 平移 x20 | x50 | x100 |")
    md.append("|---|---|---|---|---|---|---|")
    table1_geom: dict = {}
    for r in ROOM_LINE:
        rf = rf_geom[r]
        row = {
            "自家": score_of(hit_array(lf_room[r], rf, P, T, A), lf_room[r]["valid"]),
            "错配": score_of(hit_array(lf_room[OTHER[r]], rf, P, T, A), lf_room[OTHER[r]]["valid"]),
            "rot90": score_of(hit_array(lf_variants[r]["rot90"], rf, P, T, A), lf_variants[r]["rot90"]["valid"]),
            **{f"x{s}": score_of(hit_array(lf_variants[r][f"x{s}"], rf, P, T, A), lf_variants[r][f"x{s}"]["valid"]) for s in SHIFTS_PX},
        }
        hits_default[f"geometry-{r}"] = hit_array(lf_room[r], rf, P, T, A)
        table1_geom[r] = row
        md.append(
            f"| {r} geometry.png | {fmt(row['自家'])} | {fmt(row['错配'])} | {fmt(row['rot90'])} | "
            + " | ".join(fmt(row[f"x{s}"]) for s in SHIFTS_PX)
            + " |"
        )
    out["表一_室内_默认常量"] = table1
    out["表一_室内_geometry自量"] = table1_geom
    print(f"表一完成 {time.time() - t0:.1f}s", flush=True)

    # ---- 表二：揭顶视角同一套对照（作对比基线）----
    md.append("\n## 表二 揭顶视角 · 同一套对照（默认常量）\n")
    md.append("| 出图 | 正确配对 | 转 90° | 平移 x20 | x50 | x100 | 平移 y20 | y50 | y100 |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    table2: dict = {}
    bird_variants = {
        "rot90": line_features(transform_line(bird_arr, rotate90=True)),
        **{f"x{s}": line_features(transform_line(bird_arr, shift=(s, 0))) for s in SHIFTS_PX},
        **{f"y{s}": line_features(transform_line(bird_arr, shift=(0, s))) for s in SHIFTS_PX},
    }
    for name, rf in list(rf_bird.items()) + [("geometry.png（自量）", rf_bird_geom)]:
        row = {"正确配对": score_of(hit_array(lf_bird, rf, P, T, A), lf_bird["valid"])}
        for key, lf in bird_variants.items():
            row[key] = score_of(hit_array(lf, rf, P, T, A), lf["valid"])
        table2[name] = row
        md.append(
            f"| {name} | {fmt(row['正确配对'])} | {fmt(row['rot90'])} | "
            + " | ".join(fmt(row[f"x{s}"]) for s in SHIFTS_PX)
            + " | "
            + " | ".join(fmt(row[f"y{s}"]) for s in SHIFTS_PX)
            + " |"
        )
    out["表二_揭顶_默认常量"] = table2
    print(f"表二完成 {time.time() - t0:.1f}s", flush=True)

    # ---- 表三：线稿边按遮罩语义分类，各类命中率 ----
    md.append("\n## 表三 线稿边按遮罩语义分类（半径 2 px 内出现的语义；优先级 天花 > 地面 > 洞口 > 背景 > 只剩墙）\n")
    table3: dict = {}
    for r in ROOM_IMAGES:
        valid = lf_room[r]["valid"]
        cls = cls_room[r]
        shares = {c: int((valid & (cls == i)).sum()) for i, c in enumerate(CLASS_NAMES)}
        n_valid = int(valid.sum())
        md.append(f"\n### {r}（线稿有效边像素 {n_valid}）\n")
        md.append("| 类别 | 像素占比 | " + " | ".join(ROOM_IMAGES[r]) + " | 极差 | 错配（另一间线稿，同类） | geometry 自量 |")
        md.append("|---|---|" + "---|" * len(ROOM_IMAGES[r]) + "---|---|---|")
        table3[r] = {"有效边像素": n_valid, "类别": {}}
        for i, c in enumerate(CLASS_NAMES):
            where = cls == i
            per = [score_of(hits_default[name], valid, where) for name in ROOM_IMAGES[r]]
            o = OTHER[r]
            per_mis = [score_of(hits_default[f"{name}|错配"], lf_room[o]["valid"], cls_room[o] == i) for name in ROOM_IMAGES[r]]
            geom = score_of(hits_default[f"geometry-{r}"], valid, where)
            vals = [v for v in per if v is not None]
            rng = (max(vals) - min(vals)) if vals else None
            mis_vals = [v for v in per_mis if v is not None]
            mis_txt = f"{min(mis_vals):.4f}–{max(mis_vals):.4f}" if mis_vals else "—"
            table3[r]["类别"][c] = {
                "像素数": shares[c],
                "占比": shares[c] / n_valid,
                "各图命中率": dict(zip(ROOM_IMAGES[r], per)),
                "极差": rng,
                "错配各图": dict(zip(ROOM_IMAGES[r], per_mis)),
                "geometry自量": geom,
            }
            md.append(
                f"| {c} | {shares[c] / n_valid:.3f}（{shares[c]}） | "
                + " | ".join(fmt(v) for v in per)
                + f" | {fmt(rng)} | {mis_txt} | {fmt(geom)} |"
            )
        # 各类对整体分数散布的贡献：分数 = Σ 占比_c × 命中率_c
        md.append("")
        md.append("| 贡献（占比×命中率） | " + " | ".join(ROOM_IMAGES[r]) + " | 极差 |")
        md.append("|---|" + "---|" * len(ROOM_IMAGES[r]) + "---|")
        contrib: dict = {}
        for i, c in enumerate(CLASS_NAMES):
            w = shares[c] / n_valid
            per = [score_of(hits_default[name], valid, cls == i) for name in ROOM_IMAGES[r]]
            vals = [w * (v or 0.0) for v in per]
            contrib[c] = vals
            md.append(f"| {c} | " + " | ".join(f"{v:.4f}" for v in vals) + f" | {max(vals) - min(vals):.4f} |")
        totals = [sum(contrib[c][k] for c in CLASS_NAMES) for k in range(len(ROOM_IMAGES[r]))]
        md.append("| 合计（＝表一正确配对） | " + " | ".join(f"{v:.4f}" for v in totals) + f" | {max(totals) - min(totals):.4f} |")
        table3[r]["贡献"] = contrib
        # 只算墙面边的分数 vs 全边分数
        wall_only = [score_of(hits_default[name], valid, cls == CLASS_NAMES.index("墙面边")) for name in ROOM_IMAGES[r]]
        wall_mis = [score_of(hits_default[f"{name}|错配"], lf_room[OTHER[r]]["valid"], cls_room[OTHER[r]] == CLASS_NAMES.index("墙面边")) for name in ROOM_IMAGES[r]]
        table3[r]["只算墙面边"] = {"正确配对": dict(zip(ROOM_IMAGES[r], wall_only)), "错配": dict(zip(ROOM_IMAGES[r], wall_mis))}
    out["表三_按边类别"] = table3
    print(f"表三完成 {time.time() - t0:.1f}s", flush=True)

    # ---- 表四：常量扫描 ----
    md.append("\n## 表四 常量扫描（室内 8 张；每格＝该组常量下的分数汇总）\n")
    md.append(
        "| percentile | tol px | angle ° | 正确 min | 正确 mean | 正确 max | 正确极差 | 错配 max | 错配 mean | 转90 max | 平移x50 max | "
        "间隙 min正−max错 | 比值 min正/max错 | 墙面边：正确 min | 墙面边：错配 max | 墙面边：间隙 |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    sweep: list[dict] = []
    wall_idx = CLASS_NAMES.index("墙面边")
    for p, t, a in itertools.product(SWEEP_PERCENTILE, SWEEP_TOL, SWEEP_ANGLE):
        ok, mis, rot, sh50, ok_w, mis_w = [], [], [], [], [], []
        per_image = {}
        for r in ROOM_IMAGES:
            o = OTHER[r]
            for name in ROOM_IMAGES[r]:
                rf = rf_room[name]
                h = hit_array(lf_room[r], rf, p, t, a)
                s_ok = score_of(h, lf_room[r]["valid"])
                s_ok_w = score_of(h, lf_room[r]["valid"], cls_room[r] == wall_idx)
                hm = hit_array(lf_room[o], rf, p, t, a)
                s_mis = score_of(hm, lf_room[o]["valid"])
                s_mis_w = score_of(hm, lf_room[o]["valid"], cls_room[o] == wall_idx)
                s_rot = score_of(hit_array(lf_variants[r]["rot90"], rf, p, t, a), lf_variants[r]["rot90"]["valid"])
                s_sh = score_of(hit_array(lf_variants[r]["x50"], rf, p, t, a), lf_variants[r]["x50"]["valid"])
                ok.append(s_ok); mis.append(s_mis); rot.append(s_rot); sh50.append(s_sh); ok_w.append(s_ok_w); mis_w.append(s_mis_w)
                per_image[name] = {"正确": s_ok, "错配": s_mis, "转90": s_rot, "平移x50": s_sh, "墙面边正确": s_ok_w, "墙面边错配": s_mis_w}
        row = {
            "percentile": p, "tol": t, "angle": a,
            "正确min": min(ok), "正确mean": float(np.mean(ok)), "正确max": max(ok), "正确极差": max(ok) - min(ok),
            "错配max": max(mis), "错配mean": float(np.mean(mis)), "转90max": max(rot), "平移x50max": max(sh50),
            "间隙": min(ok) - max(mis), "比值": min(ok) / max(mis) if max(mis) > 0 else None,
            "墙面边正确min": min(ok_w), "墙面边错配max": max(mis_w), "墙面边间隙": min(ok_w) - max(mis_w),
            "各图": per_image,
        }
        sweep.append(row)
        md.append(
            f"| {p:.0f} | {t} | {a:.0f} | {row['正确min']:.4f} | {row['正确mean']:.4f} | {row['正确max']:.4f} | {row['正确极差']:.4f} | "
            f"{row['错配max']:.4f} | {row['错配mean']:.4f} | {row['转90max']:.4f} | {row['平移x50max']:.4f} | "
            f"{row['间隙']:+.4f} | {fmt(row['比值'])} | {row['墙面边正确min']:.4f} | {row['墙面边错配max']:.4f} | {row['墙面边间隙']:+.4f} |"
        )
        print(f"  扫描 p{p:.0f} t{t} a{a:.0f}：间隙 {row['间隙']:+.4f} 比值 {fmt(row['比值'])} {time.time() - t0:.0f}s", flush=True)
    out["表四_常量扫描"] = sweep

    # ---- 调试图与拼图 ----
    debug_dir = HERE / "debug"
    debug_dir.mkdir(exist_ok=True)
    names_all = [n for r in ROOM_IMAGES for n in ROOM_IMAGES[r]]
    for r in ROOM_IMAGES:
        for name in ROOM_IMAGES[r]:
            M.fidelity_score(ROOM_LINE[r], ROOM_IMG_DIR / f"{name}.png", debug_out=debug_dir / f"{name}.debug.png")
    font = _font(22)
    tile_w, tile_h = 640, 480
    montage = Image.new("RGB", (tile_w * 4, tile_h * 2 + 2 * 30), (20, 20, 20))
    overlay = Image.new("RGB", (tile_w * 4, tile_h * 2 + 2 * 30), (20, 20, 20))
    draw_m, draw_o = ImageDraw.Draw(montage), ImageDraw.Draw(overlay)
    for k, name in enumerate(names_all):
        col, rowi = k % 4, k // 4
        x, y = col * tile_w, rowi * (tile_h + 30)
        dbg = Image.open(debug_dir / f"{name}.debug.png").convert("RGB")
        montage.paste(dbg.resize((tile_w, tile_h), Image.NEAREST), (x, y + 30))
        base = Image.open(ROOM_IMG_DIR / f"{name}.png").convert("RGB").resize(size, Image.LANCZOS)
        dim = Image.blend(base, Image.new("RGB", size, (0, 0, 0)), 0.5)
        dbg_arr = np.array(dbg)
        base_arr = np.array(dim)
        marked = (dbg_arr[..., 0] > 0) | (dbg_arr[..., 1] > 0)  # 红线与绿命中；蓝强边不叠，免得糊
        base_arr[marked] = dbg_arr[marked]
        overlay.paste(Image.fromarray(base_arr).resize((tile_w, tile_h), Image.LANCZOS), (x, y + 30))
        label = f"{name}  分数 {table1[name]['正确配对']:.4f}  错配 {table1[name]['错配']:.4f}  转90 {table1[name]['rot90']:.4f}"
        draw_m.text((x + 6, y + 4), label, fill=(255, 255, 255), font=font)
        draw_o.text((x + 6, y + 4), label, fill=(255, 255, 255), font=font)
    montage.save(HERE / "montage-调试.png")
    overlay.save(HERE / "montage-叠原图.png")

    # 线稿边分类图
    cls_img = Image.new("RGB", (size[0] * 2 + 10, size[1] + 40), (20, 20, 20))
    d = ImageDraw.Draw(cls_img)
    for k, r in enumerate(ROOM_LINE):
        canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        valid = lf_room[r]["valid"]
        for i, c in enumerate(CLASS_NAMES):
            canvas[valid & (cls_room[r] == i)] = CLASS_COLOR[c]
        cls_img.paste(Image.fromarray(canvas), (k * (size[0] + 10), 40))
        d.text((k * (size[0] + 10) + 6, 6), f"{r} 线稿边分类", fill=(255, 255, 255), font=font)
    legend = "  ".join(f"{c}" for c in CLASS_NAMES)
    d.text((size[0] // 2, 6), "白=墙面边 红=天花边 蓝=地面边 黄=洞口边 绿=背景边", fill=(200, 200, 200), font=font)
    cls_img.save(HERE / "线稿边分类.png")

    (HERE / "结果.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (HERE / "结果表.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"完成 {time.time() - t0:.1f}s；表在 结果表.md，数在 结果.json")
    return 0


def _font(px: int) -> ImageFont.ImageFont:
    for p in ("/System/Library/Fonts/STHeiti Medium.ttc", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"):
        try:
            return ImageFont.truetype(p, px)
        except OSError:
            continue
    return ImageFont.load_default()


if __name__ == "__main__":
    raise SystemExit(main())
