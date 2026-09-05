"""分数分布：把已存档的万相 / Seedream 真跑图在两把尺子（v2 `fidelity_metric`、v3 `fidelity_metric_v3`）上量齐，
按输出形态分档（视角 × 控制稿版本 × 风格 × 陈设），并给每档配"错配"与"转 90°"两种负对照。

只算不调模型。跑法：`cd ~/codes/ishome-imagegen && uv run python 分数分布.py`。
输出到 render3d `_iteration/run-2026-09-05-score-distribution/分布.csv`，并把分档汇总表打到 stdout（贴进 run.md）。

对照物两种：
- `sketch`＝送模型的那张稿（反色回黑底白线；9-02 / 9-04 / Seedream 批送的就是 line.png 反色，已逐像素核过，
  所以那几批只有一行、对照物记 `line(=送模型稿反色)`）；
- `line`＝同机位 `line.png`（9-05 三批控制稿 ≠ 线稿，另量一遍）。
"""

from __future__ import annotations

import csv
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from imagegen_worker.fidelity_metric import score_fidelity
from imagegen_worker.fidelity_metric_v3 import score_fidelity_v3

ROOT = Path("/Users/baitianxing/codes/ishome-render3d.wt-score-dist/_iteration")
OUT_DIR = ROOT / "run-2026-09-05-score-distribution"
CSV_PATH = OUT_DIR / "分布.csv"
WORKERS = 5

BASE_LINE = "真户型-基准/底渲-cam-bird-dollhouse/line.png"
BASE_GEOM = "真户型-基准/底渲-cam-bird-dollhouse/geometry.png"


def l04(room: str) -> str:
    return f"run-2026-09-04-failure-catalogue/底渲-室内机位/cam-room-{room}/line.png"


def cs(cam: str, f: str) -> str:
    return f"run-2026-09-05-control-sketch/{cam}/{f}"


def ok(cam: str, f: str) -> str:
    return f"run-2026-09-05-opening-kind-consume/{cam}/{f}"


V_LINE19 = "19洞线稿反色"
V_LINE19_REF = "19洞线稿(参考图通路)"
V_GEOM19 = "19洞geometry自抽"
V_CS19 = "19洞控制稿"
V_CS16 = "16洞控制稿"


@dataclass(frozen=True)
class Sample:
    path: str
    视角: str
    控制稿版本: str
    风格: str
    陈设: str
    机位: str
    通路: str
    提示词来源: str
    sketch: str  # 送模型的稿（黑底白线形态）
    line: str | None  # 同机位 line.png；None＝与 sketch 是同一张
    备注: str = ""

    @property
    def bin_key(self) -> tuple[str, str, str, str]:
        return (self.视角, self.控制稿版本, self.风格, self.陈设)


def build_samples() -> list[Sample]:
    s: list[Sample] = []
    d = "run-2026-09-02-control-path-survey/万相-doodle"
    for f, note in [
        ("run1-seed12345", ""),
        ("run2-seed12345-repeat", "与 run1 逐像素相同（同 seed 复跑）"),
        ("run4-seed1001", ""),
        ("run5-seed1002", ""),
    ]:
        s.append(Sample(f"{d}/{f}.png", "揭顶", V_LINE19, "modern-minimal", "默认", "cam-bird-dollhouse",
                        "万相doodle", "探针手写", BASE_LINE, None, note))
    s.append(Sample(f"{d}/run6-geometry-autosketch-seed12345.png", "揭顶", V_GEOM19, "modern-minimal", "默认",
                    "cam-bird-dollhouse", "万相doodle(is_sketch=false)", "探针手写", BASE_LINE, None,
                    "送的是 geometry.png 让模型自抽线稿，按任务指定对 line.png 量"))
    d = "run-2026-09-02-control-path-survey/Seedream-5.0-lite-对照"
    for i in (1, 2, 3):
        s.append(Sample(f"{d}/lite-run{i}.png", "揭顶", V_LINE19_REF, "modern-minimal", "默认", "cam-bird-dollhouse",
                        "Seedream-5.0-lite参考图", "探针手写", BASE_LINE, None, "参考图通路对照组；出图 2304×1728"))
    d = "run-2026-09-04-failure-catalogue/万相-室内机位"
    for room in ("客厅", "主卧"):
        for f, furn in [("default-seed1", "默认"), ("nofurn-seed1", "无陈设"), ("nofurn-seed2", "无陈设"),
                        ("nofurn-seed3", "无陈设")]:
            s.append(Sample(f"{d}/{room}-{f}.png", "室内", V_LINE19, "modern-minimal", furn, f"cam-room-{room}",
                            "万相doodle", "探针手写", l04(room), None))
    d = "run-2026-09-04-failure-catalogue/万相-揭顶-无陈设"
    for i in (1, 2):
        s.append(Sample(f"{d}/bird-nofurn-seed{i}.png", "揭顶", V_LINE19, "modern-minimal", "无陈设",
                        "cam-bird-dollhouse", "万相doodle", "探针手写", BASE_LINE, None))
    d = "run-2026-09-05-control-sketch-realism"
    for cam in ("cam-bird-dollhouse", "cam-room-客厅", "cam-room-主卧", "cam-room-次卧", "cam-room-书房"):
        view = "揭顶" if cam == "cam-bird-dollhouse" else "室内"
        for i in (1, 2, 3):
            s.append(Sample(f"{d}/{cam}-seed{i}.png", view, V_CS19, "modern-minimal", "无陈设", cam,
                            "万相doodle经网关", "imagegen模板", cs(cam, "sketch.png"), cs(cam, "line.png")))
    d = "run-2026-09-05-realism-styles"
    for cam, style, i in [("cam-bird-dollhouse", "nordic-light", 1), ("cam-bird-dollhouse", "nordic-light", 2),
                          ("cam-bird-dollhouse", "cream-warm", 1), ("cam-room-主卧", "nordic-light", 1),
                          ("cam-room-主卧", "cream-warm", 1)]:
        view = "揭顶" if cam == "cam-bird-dollhouse" else "室内"
        s.append(Sample(f"{d}/{cam}-{style}-seed{i}.png", view, V_CS19, style, "无陈设", cam,
                        "万相doodle经网关", "imagegen模板", cs(cam, "sketch.png"), cs(cam, "line.png")))
    d = "run-2026-09-05-opening-kind-realism"
    for cam in ("cam-bird-dollhouse", "cam-room-客厅", "cam-room-主卧", "cam-room-书房"):
        view = "揭顶" if cam == "cam-bird-dollhouse" else "室内"
        for i in (1, 2, 3):
            s.append(Sample(f"{d}/{cam}-seed{i}.png", view, V_CS16, "modern-minimal", "无陈设", cam,
                            "万相doodle经网关", "imagegen模板", ok(cam, "sketch.png"), ok(cam, "line.png")))
    return s


def mismatch_ref(version: str, cam: str, kind: str) -> tuple[str, str]:
    """错配：同一控制稿版本、另一台机位的稿。返回 (另一台机位, 稿路径)。"""
    if version in (V_LINE19, V_LINE19_REF, V_GEOM19):
        alt = {"cam-bird-dollhouse": "cam-room-客厅", "cam-room-客厅": "cam-room-主卧",
               "cam-room-主卧": "cam-room-客厅"}[cam]
        return alt, (BASE_LINE if alt == "cam-bird-dollhouse" else l04(alt.removeprefix("cam-room-")))
    if version == V_CS19:
        alt = {"cam-bird-dollhouse": "cam-room-客厅", "cam-room-客厅": "cam-room-主卧", "cam-room-主卧": "cam-room-客厅",
               "cam-room-次卧": "cam-room-书房", "cam-room-书房": "cam-room-次卧"}[cam]
        return alt, cs(alt, f"{kind}.png")
    if version == V_CS16:
        alt = {"cam-bird-dollhouse": "cam-room-客厅", "cam-room-客厅": "cam-room-主卧", "cam-room-主卧": "cam-room-书房",
               "cam-room-书房": "cam-room-客厅"}[cam]
        return alt, ok(alt, f"{kind}.png")
    raise ValueError(version)


@dataclass(frozen=True)
class Job:
    sample: Sample
    对照类型: str  # 正常 / 错配 / 转90 / 核对
    对照物: str  # sketch / line / line(=送模型稿反色)
    对照物路径: str
    rotate90: bool
    错配机位: str = ""


def score_job(job: Job) -> dict:
    t0 = time.time()
    ref = (ROOT / job.对照物路径).read_bytes()
    res = (ROOT / job.sample.path).read_bytes()
    v2 = score_fidelity(ref, res, rotate90=job.rotate90)
    v3 = score_fidelity_v3(ref, res, rotate90=job.rotate90)
    row = {
        "样本路径": "_iteration/" + job.sample.path,
        "视角": job.sample.视角,
        "控制稿版本": job.sample.控制稿版本,
        "风格": job.sample.风格,
        "陈设": job.sample.陈设,
        "机位": job.sample.机位,
        "通路": job.sample.通路,
        "提示词来源": job.sample.提示词来源,
        "对照类型": job.对照类型,
        "错配机位": job.错配机位,
        "对照物": job.对照物,
        "对照物路径": "_iteration/" + job.对照物路径,
        "v2": round(v2, 4),
        "v3原位": round(v3.score_at_origin, 4),
        "v3配准": round(v3.score, 4),
        "dx": v3.offset_px[0],
        "dy": v3.offset_px[1],
        "sx": v3.scale_xy[0],
        "sy": v3.scale_xy[1],
        "线像素": v3.n_line_pixels,
        "v3进分母": v3.n_scored_pixels,
        "备注": job.sample.备注,
        "耗时s": round(time.time() - t0, 1),
    }
    return row


def build_jobs(samples: list[Sample]) -> list[Job]:
    jobs: list[Job] = []
    by_bin: dict[tuple, list[Sample]] = defaultdict(list)
    for smp in samples:
        by_bin[smp.bin_key].append(smp)
        kinds = [("line(=送模型稿反色)", smp.sketch)] if smp.line is None else [("sketch", smp.sketch), ("line", smp.line)]
        for kind, p in kinds:
            jobs.append(Job(smp, "正常", kind, p, False))
    # 负对照：每档取 2 张（机位尽量不同），错配 + 转 90°，对照物种类与正常行一致
    for key, lst in by_bin.items():
        picked: list[Sample] = []
        seen_cams: set[str] = set()
        for smp in lst:
            if smp.机位 not in seen_cams:
                picked.append(smp)
                seen_cams.add(smp.机位)
            if len(picked) == 2:
                break
        for smp in lst:
            if len(picked) == 2:
                break
            if smp not in picked and "逐像素相同" not in smp.备注:
                picked.append(smp)
        for smp in picked:
            kinds = [("line(=送模型稿反色)", "line")] if smp.line is None else [("sketch", "sketch"), ("line", "line")]
            for kind_label, kind_file in kinds:
                alt, altp = mismatch_ref(smp.控制稿版本, smp.机位, kind_file)
                jobs.append(Job(smp, "错配", kind_label, altp, False, alt))
                own = smp.sketch if kind_file == "sketch" or smp.line is None else smp.line
                jobs.append(Job(smp, "转90", kind_label, own, True))
    # 核对：揭顶 geometry 自量（已知 v2 0.0607 / v3 原位 0.0675）
    geom = Sample(BASE_GEOM, "揭顶", V_LINE19, "—", "—", "cam-bird-dollhouse", "底渲(非模型)", "—", BASE_LINE, None,
                  "核对用：几何全对的底渲自量，不进分布")
    jobs.append(Job(geom, "核对", "line", BASE_LINE, False))
    return jobs


def fmt(x: float) -> str:
    return f"{x:.4f}"


def summarize(rows: list[dict]) -> None:
    def stats(vals: list[float]) -> str:
        if not vals:
            return "—"
        if len(vals) == 1:
            return f"{fmt(vals[0])}（n=1）"
        return f"{fmt(min(vals))} / {fmt(statistics.median(vals))} / {fmt(max(vals))}，极差 {fmt(max(vals) - min(vals))}"

    bins: dict[tuple, dict[str, dict[str, list[dict]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        if r["对照类型"] == "核对":
            continue
        key = (r["视角"], r["控制稿版本"], r["风格"], r["陈设"])
        bins[key][r["对照物"]][r["对照类型"]].append(r)

    order = sorted(bins.keys(), key=lambda k: (k[0], k[1], k[2], k[3]))
    print("\n## 汇总（每格：最小 / 中位 / 最大，极差）\n")
    for metric in ("v2", "v3原位", "v3配准"):
        print(f"\n### {metric}\n")
        print("| 视角 | 控制稿版本 | 风格 | 陈设 | 对照物 | n | 正常 | 错配 | 转90 | 能否定线 |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for key in order:
            for kind, groups in bins[key].items():
                normal = [r[metric] for r in groups.get("正常", [])]
                mis = [r[metric] for r in groups.get("错配", [])]
                rot = [r[metric] for r in groups.get("转90", [])]
                n = len(normal)
                flag = "n≥5" if n >= 5 else "**n<5 不能定线**"
                print(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {kind} | {n} | {stats(normal)} | "
                      f"{stats(mis)} | {stats(rot)} | {flag} |")

    print("\n### 位移（v3 配准的最优位移，正常行；每档列出全部）\n")
    print("| 视角 | 控制稿版本 | 风格 | 陈设 | 对照物 | (dx,dy) 逐样本 |")
    print("|---|---|---|---|---|---|")
    for key in order:
        for kind, groups in bins[key].items():
            offs = ", ".join(f"({r['dx']:+d},{r['dy']:+d})" for r in groups.get("正常", []))
            print(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {kind} | {offs} |")

    print("\n### 室内档按机位拆（正常行；补充数据，不是分档）\n")
    print("| 控制稿版本 | 风格 | 陈设 | 对照物 | 机位 | n | v2 | v3原位 | v3配准 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for key in order:
        if key[0] != "室内":
            continue
        for kind, groups in bins[key].items():
            by_cam: dict[str, list[dict]] = defaultdict(list)
            for r in groups.get("正常", []):
                by_cam[r["机位"]].append(r)
            for cam, lst in sorted(by_cam.items()):
                print(f"| {key[1]} | {key[2]} | {key[3]} | {kind} | {cam} | {len(lst)} | "
                      f"{stats([r['v2'] for r in lst])} | {stats([r['v3原位'] for r in lst])} | "
                      f"{stats([r['v3配准'] for r in lst])} |")

    print("\n### Seedream 参考图通路 vs 万相控制通路（揭顶、对 line.png、v2）\n")
    for r in rows:
        if r["视角"] == "揭顶" and r["控制稿版本"] in (V_LINE19, V_LINE19_REF, V_GEOM19) and r["对照类型"] == "正常":
            print(f"- {r['通路']} · {r['陈设']} · {Path(r['样本路径']).name}: v2 {fmt(r['v2'])} · v3原位 {fmt(r['v3原位'])} "
                  f"· v3配准 {fmt(r['v3配准'])} @({r['dx']:+d},{r['dy']:+d})")

    print("\n### 核对\n")
    for r in rows:
        if r["对照类型"] == "核对":
            print(f"- {r['样本路径']} 对 {r['对照物路径']}: v2 {fmt(r['v2'])}（已知 0.0607） v3原位 {fmt(r['v3原位'])}（已知 0.0675） "
                  f"v3配准 {fmt(r['v3配准'])} @({r['dx']:+d},{r['dy']:+d})")
    for r in rows:
        if r["样本路径"].endswith("run1-seed12345.png") and r["对照类型"] == "正常":
            print(f"- 19 洞万相 run1: v2 {fmt(r['v2'])}（已知 0.2396）")


def main() -> None:
    samples = build_samples()
    jobs = build_jobs(samples)
    print(f"样本 {len(samples)} 张，量分任务 {len(jobs)} 个（含负对照与核对），{WORKERS} 进程", file=sys.stderr)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, row in enumerate(ex.map(score_job, jobs), 1):
            rows.append(row)
            print(f"[{i}/{len(jobs)}] {row['对照类型']:2s} {row['对照物']:<14s} {Path(row['样本路径']).name:45s} "
                  f"v2 {row['v2']:.4f} v3 {row['v3原位']:.4f}->{row['v3配准']:.4f} @({row['dx']:+d},{row['dy']:+d}) "
                  f"{row['耗时s']}s", file=sys.stderr)
    print(f"总耗时 {time.time() - t0:.0f} s", file=sys.stderr)

    order = {"正常": 0, "错配": 1, "转90": 2, "核对": 3}
    rows.sort(key=lambda r: (r["视角"], r["控制稿版本"], r["风格"], r["陈设"], r["机位"], r["样本路径"],
                             r["对照物"], order[r["对照类型"]]))
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"写 {CSV_PATH}（{len(rows)} 行）", file=sys.stderr)
    summarize(rows)


if __name__ == "__main__":
    main()
