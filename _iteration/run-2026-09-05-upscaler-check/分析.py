"""对 scratchpad `upscale/out/<机位>/` 里各候选的放大结果做全部量化与拼图，写本目录 `分析.json` 与拼图。纯本地。

每个候选每张图：
  ① 输出尺寸、耗时（run1 含 MPS 首次编译、run2 为热态）、加载时间
  ② 确定性：run1/run2 数组 sha256 逐字节比（放大.py 已记，这里再读文件复核一次）
  ③ 放大图 LANCZOS 缩回 1168×880 与原图逐像素比：均值差 / 差>10 占比 / 差>30 占比 / 完全相同占比
     （口径照抄 run-2026-09-05-wanx-superres-batch/比对与拼图.py：RGB 三通道取最大差）
  ④ 保真度：run-2026-09-01-condition-matrix/保真度量.py 的 fidelity_score 对 line.png，量原图与放大图
     （尺子先把结果 LANCZOS 缩到线稿 1280×960 再量，分数只能横向比）
  ⑤ 叠线：线稿 BILINEAR 缩放到放大图尺寸、线像素按 55% 红混入；整图拼图 + 固定区域 1:1 局部拼图
  补：结构边新增/丢失比（不是既有尺子，本文件自加）：两图各缩到线稿尺寸做 Sobel+NMS 脊线；阈值 T 取
     原图脊线的第 90 百分位（绝对值，两图共用，不各取各的百分位——否则锐化把原有边推进前 10% 也会算"新增"）。
     新增边比 = 放大图脊线 ≥T 里、原图 2 px 内连弱脊线（≥T/4）都没有的占比；丢失边比 = 原图脊线 ≥T 里、
     放大图 2 px 内连弱脊线都没有的占比。
×4 候选另派生一个 "→×2(LANCZOS)" 变体：×4 结果 LANCZOS 缩到 2336×1760，耗时 = ×4 耗时 + 缩放耗时。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).resolve().parent
SCRATCH = pathlib.Path(
    "/private/tmp/claude-501/-Users-baitianxing-codes-ishome/5043122a-4d91-4234-8b80-a5febe79f775/scratchpad/upscale"
)
OUT = SCRATCH / "out"
ITER = pathlib.Path("/Users/baitianxing/codes/ishome-render3d/_iteration")
REALISM = ITER / "run-2026-09-05-control-sketch-realism"
SKETCH = ITER / "run-2026-09-05-control-sketch"
FIDELITY_PY = ITER / "run-2026-09-01-condition-matrix" / "保真度量.py"

SAMPLES = {
    # 机位: (原图, 线稿, 局部裁区 (x, y, w, h) 原图坐标)
    "cam-bird-dollhouse": (REALISM / "cam-bird-dollhouse-seed1.png", SKETCH / "cam-bird-dollhouse" / "line.png", (400, 430, 320, 240)),
    "cam-room-主卧": (REALISM / "cam-room-主卧-seed1.png", SKETCH / "cam-room-主卧" / "line.png", (100, 250, 320, 240)),
}
CANDIDATES = ["lanczos-x2", "realesrgan-x2plus", "swinir-m-x2-realsr", "realesrgan-x4plus", "realesr-general-x4v3"]
TARGET_X2 = (1168 * 2, 880 * 2)
OVERLAY_ALPHA = 0.55
LABEL_FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"  # 拼图标签用系统中文字体；缺了退回 PIL 默认（中文会成方块）
EDGE_TOL_PX = 2


def _load_fidelity_module():
    spec = importlib.util.spec_from_file_location("fidelity", FIDELITY_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FID = _load_fidelity_module()


def sha256_arr(im: Image.Image) -> str:
    return hashlib.sha256(np.asarray(im.convert("RGB")).tobytes()).hexdigest()


def to_size(im: Image.Image, size) -> Image.Image:
    im = im.convert("RGB")
    return im if im.size == tuple(size) else im.resize(tuple(size), Image.LANCZOS)


def compare(a: Image.Image, b: Image.Image, size) -> dict:
    aa = np.asarray(to_size(a, size), dtype=np.int16)
    bb = np.asarray(to_size(b, size), dtype=np.int16)
    d = np.abs(aa - bb)
    dmax = d.max(axis=2)
    return {
        "compare_size": list(size),
        "mean_abs_diff": round(float(d.mean()), 3),
        "pct_pixels_maxch_diff_gt10": round(float((dmax > 10).mean() * 100), 2),
        "pct_pixels_maxch_diff_gt30": round(float((dmax > 30).mean() * 100), 2),
        "pct_pixels_identical": round(float((dmax == 0).mean() * 100), 2),
    }


def ridge_map(im: Image.Image, size) -> np.ndarray:
    """与保真度量.py 第 2 步同口径：缩到线稿尺寸 -> 灰度 -> Sobel -> NMS，返回脊线幅值图（非脊线为 0）。"""
    gray = np.asarray(to_size(im, size).convert("L"), dtype=np.float64)
    gx, gy, mag = FID.sobel_components(gray)
    return FID.non_max_suppression(mag, gx, gy)


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    h, w = mask.shape
    padded = np.pad(mask, r, mode="constant", constant_values=False)
    out = np.zeros_like(mask)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx <= r * r:
                out |= padded[r + dy : r + dy + h, r + dx : r + dx + w]
    return out


def edge_change(orig: Image.Image, cand: Image.Image, size) -> dict:
    ro, rc = ridge_map(orig, size), ridge_map(cand, size)
    T = float(np.percentile(ro[ro > 0], FID.EDGE_STRENGTH_PERCENTILE))
    so, sc = ro >= T, rc >= T
    wo, wc = ro >= T / 4, rc >= T / 4
    new_ratio = float((sc & ~dilate(wo, EDGE_TOL_PX)).sum()) / max(1, int(sc.sum()))
    lost_ratio = float((so & ~dilate(wc, EDGE_TOL_PX)).sum()) / max(1, int(so.sum()))
    return {"threshold_from_original": round(T, 2), "pct_strong_edges_new": round(new_ratio * 100, 2),
            "pct_strong_edges_lost": round(lost_ratio * 100, 2),
            "n_strong_edges_orig": int(so.sum()), "n_strong_edges_cand": int(sc.sum())}


def overlay(line: Image.Image, res: Image.Image) -> Image.Image:
    mask = np.asarray(line.convert("L").resize(res.size, Image.BILINEAR)) > 127
    arr = np.asarray(res.convert("RGB"), dtype=np.float32)
    red = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    arr[mask] = arr[mask] * (1 - OVERLAY_ALPHA) + red * OVERLAY_ALPHA
    return Image.fromarray(arr.round().astype(np.uint8))


def grid(tiles: list[tuple[Image.Image, str]], per_row: int, out_path: pathlib.Path, gap: int = 10) -> None:
    tw = max(t.width for t, _ in tiles)
    th = max(t.height for t, _ in tiles)
    rows = (len(tiles) + per_row - 1) // per_row
    canvas = Image.new("RGB", (per_row * (tw + gap) + gap, rows * (th + gap + 36) + gap), (30, 30, 30))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(LABEL_FONT, 22)
    except OSError:
        font = ImageFont.load_default()
    for i, (t, name) in enumerate(tiles):
        r, c = divmod(i, per_row)
        x, y = gap + c * (tw + gap), gap + r * (th + gap + 36)
        canvas.paste(t, (x, y))
        d.text((x + 2, y + th + 6), name, fill=(235, 235, 235), font=font)
    canvas.save(out_path, compress_level=6)


def main() -> int:
    results: dict = {"samples": {}, "notes": {
        "compare": "放大图与原图各自 LANCZOS 缩到 1168×880，RGB 三通道取最大差",
        "fidelity": "保真度量.py fidelity_score 对 line.png；尺子先缩到 1280×960 再量",
        "edge_change": f"补充量（本文件自加）：线稿尺寸下 Sobel+NMS 脊线，阈值 T=原图脊线第 90 百分位（绝对值两图共用），新增=放大图≥T 且原图 {EDGE_TOL_PX} px 内无 ≥T/4 脊线；丢失反之",
        "overlay": f"线稿 BILINEAR 缩放到结果尺寸，线像素按 {OVERLAY_ALPHA:.0%} 红混入（不是实心盖住，方便看红线下面模型自己的边）",
    }}
    for sample, (orig_path, line_path, crop) in SAMPLES.items():
        orig = Image.open(orig_path).convert("RGB")
        line = Image.open(line_path).convert("L")
        ox, oy, ow, oh = crop
        s_res: dict = {
            "original": str(orig_path), "line": str(line_path), "original_size": list(orig.size),
            "fidelity_original": round(FID.fidelity_score(line_path, orig_path), 4),
            "crop_region_in_original": list(crop),
            "candidates": {},
        }
        print(f"\n##### {sample}  原图 {orig.size}  保真度 {s_res['fidelity_original']:.4f}")
        full_tiles = [(overlay(line, orig), f"原图 1168×880")]
        crop_tiles = [(orig.crop((ox, oy, ox + ow, oy + oh)).resize((ow * 2, oh * 2), Image.NEAREST), "原图 NEAREST×2（对照）")]
        crop_ov_tiles = [(overlay(line, orig).crop((ox, oy, ox + ow, oy + oh)).resize((ow * 2, oh * 2), Image.NEAREST), "原图 NEAREST×2 叠线")]
        entries: list[tuple[str, Image.Image, dict]] = []
        for cand in CANDIDATES:
            meta_path = OUT / sample / f"{cand}.meta.json"
            if not meta_path.exists():
                print(f"  {cand}: 无 meta（没跑成）")
                s_res["candidates"][cand] = {"status": "missing"}
                continue
            meta = json.loads(meta_path.read_text())
            run1 = Image.open(OUT / sample / f"{cand}-run1.png").convert("RGB")
            run2_path = OUT / sample / f"{cand}-run2.png"
            det = sha256_arr(run1) == sha256_arr(Image.open(run2_path).convert("RGB")) if run2_path.exists() else None
            base = {
                "status": "ok", "output_size": list(run1.size), "device": meta["device"], "tile": meta["tile"],
                "load_seconds": meta["load_seconds"],
                "seconds_run1": meta["runs"][0]["seconds"], "seconds_run2": meta["runs"][1]["seconds"] if len(meta["runs"]) > 1 else None,
                "deterministic_meta": meta["deterministic_across_runs"], "deterministic_recheck": det,
                "png_bytes": meta["runs"][0]["png_bytes"], "array_sha256_run1": meta["runs"][0]["array_sha256"],
                "mps_driver_allocated_mb": meta["runs"][-1].get("mps_driver_allocated_mb"),
                "model_info": meta["model_info"],
            }
            entries.append((cand, run1, base))
            if run1.size != TARGET_X2:
                t0 = time.perf_counter()
                down = run1.resize(TARGET_X2, Image.LANCZOS)
                resize_s = time.perf_counter() - t0
                entries.append((f"{cand}→x2(LANCZOS)", down, {
                    "status": "ok", "derived_from": cand, "output_size": list(down.size), "device": meta["device"], "tile": meta["tile"],
                    "load_seconds": meta["load_seconds"],
                    "seconds_run1": round(base["seconds_run1"] + resize_s, 3),
                    "seconds_run2": round(base["seconds_run2"] + resize_s, 3) if base["seconds_run2"] is not None else None,
                    "resize_seconds": round(resize_s, 3),
                    "deterministic_meta": base["deterministic_meta"], "deterministic_recheck": det,
                }))
        for name, im, rec in entries:
            rec["vs_original_downscaled"] = compare(orig, im, orig.size)
            rec["fidelity"] = round(FID.fidelity_score(line_path, _tmp_save(im, sample, name)), 4)
            rec["edge_change"] = edge_change(orig, im, line.size)
            s_res["candidates"][name] = rec
            ov = overlay(line, im)
            full_tiles.append((ov, f"{name} {im.width}×{im.height}"))
            sc = im.width / orig.width
            box = (round(ox * sc), round(oy * sc), round((ox + ow) * sc), round((oy + oh) * sc))
            crop_im = im.crop(box)
            crop_ov = ov.crop(box)
            if crop_im.size != (ow * 2, oh * 2):
                crop_im = crop_im.resize((ow * 2, oh * 2), Image.LANCZOS)
                crop_ov = crop_ov.resize((ow * 2, oh * 2), Image.LANCZOS)
            crop_tiles.append((crop_im, f"{name} 局部 1:1" + ("" if sc == 2 else f"（×{sc:g} 缩到 ×2 显示）")))
            crop_ov_tiles.append((crop_ov, f"{name} 局部叠线"))
            c = rec["vs_original_downscaled"]
            print(f"  {name:32s} {im.size}  run1 {rec['seconds_run1']:>6.2f}s run2 {rec['seconds_run2']!s:>6}s  det={rec['deterministic_recheck']}  "
                  f"mean {c['mean_abs_diff']:.3f} >10 {c['pct_pixels_maxch_diff_gt10']:.2f}% >30 {c['pct_pixels_maxch_diff_gt30']:.2f}% same {c['pct_pixels_identical']:.2f}%  "
                  f"fid {rec['fidelity']:.4f}  new-edge {rec['edge_change']['pct_strong_edges_new']:.1f}% lost {rec['edge_change']['pct_strong_edges_lost']:.1f}%")
        results["samples"][sample] = s_res
        h = 480
        grid([(t.resize((round(t.width * h / t.height), h), Image.LANCZOS), n) for t, n in full_tiles], 4, HERE / f"montage-叠线-{sample}.png")
        grid(crop_tiles, 4, HERE / f"montage-局部-{sample}.png")
        grid(crop_ov_tiles, 4, HERE / f"montage-局部叠线-{sample}.png")
    (HERE / "分析.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


def _tmp_save(im: Image.Image, sample: str, name: str) -> pathlib.Path:
    """fidelity_score 只吃路径；派生变体没有文件，落到 scratchpad 再量。"""
    p = OUT / sample / f"_derived-{name.replace('→', '-').replace('(', '').replace(')', '')}.png"
    if not p.exists():
        im.save(p, compress_level=1)
    return p


if __name__ == "__main__":
    raise SystemExit(main())
