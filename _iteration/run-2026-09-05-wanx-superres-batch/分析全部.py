"""对 out/ 里所有出图做：与 run1 逐像素比、保真度、叠线图、拼图。结果写 out/分析.json 并打印。纯本地。"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import 比对与拼图 as T

HERE = pathlib.Path(__file__).parent
OUT = HERE / "out"
RUN1 = pathlib.Path("/Users/baitianxing/codes/ishome-render3d/_iteration/run-2026-09-02-control-path-survey/万相-doodle/run1-seed12345.png")
RUN2 = RUN1.with_name("run2-seed12345-repeat.png")
LINE = pathlib.Path("/Users/baitianxing/codes/ishome-render3d/_iteration/真户型-基准/底渲-cam-bird-dollhouse/line.png")
fid = T.load_fidelity()

results = {"baseline_run1_fidelity": fid(LINE, RUN1)}
print(f"run1 保真度（基线）: {results['baseline_run1_fidelity']:.4f}")

for png in sorted(OUT.glob("*.png")):
    if png.name.startswith("overlay-") or png.name.startswith("montage"):
        continue
    cmp = T.compare(RUN1, png)
    score = fid(LINE, png)
    ov = OUT / f"overlay-{png.stem}.png"
    T.overlay(LINE, png, ov)
    results[png.name] = {"vs_run1": cmp, "fidelity": score, "overlay": ov.name}
    print(f"\n== {png.name} 尺寸 {cmp['b_size']}  保真度 {score:.4f}")
    for k in ("compare_size", "mean_abs_diff", "pct_pixels_maxch_diff_gt10", "pct_pixels_maxch_diff_gt30", "pct_pixels_identical"):
        v = cmp[k]
        print(f"   {k}: {v:.3f}" if isinstance(v, float) else f"   {k}: {v}")

pairs = [("doodle-n2-seed12345-1.png", "doodle-n2-seed12345-2.png"),
         ("sr-x2-of-run1.png", "sr-x3-of-run1.png")]
for a, b in pairs:
    pa, pb = OUT / a, OUT / b
    if pa.exists() and pb.exists():
        cmp = T.compare(pa, pb)
        results[f"{a} vs {b}"] = cmp
        print(f"\n== {a} vs {b}: mean {cmp['mean_abs_diff']:.3f}  >10: {cmp['pct_pixels_maxch_diff_gt10']:.2f}%  identical: {cmp['pct_pixels_identical']:.2f}%")

cmp = T.compare(RUN1, RUN2)
results["对照 run1 vs run2-repeat（9-02 同 seed n=1 重复跑）"] = cmp
print(f"\n== 对照 run1 vs run2-repeat: mean {cmp['mean_abs_diff']:.3f}  >10: {cmp['pct_pixels_maxch_diff_gt10']:.2f}%  identical: {cmp['pct_pixels_identical']:.2f}%")

(OUT / "分析.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))

# 叠线拼图：基线 run1 也叠一张，三组各拼一张
T.overlay(LINE, RUN1, OUT / "overlay-run1-baseline.png")
montages = {
    "montage-超分叠线.png": ["overlay-run1-baseline.png", "overlay-sr-x2-of-run1.png", "overlay-sr-x3-of-run1.png", "overlay-sr-x2-of-run1-small683.png"],
    "montage-大线稿叠线.png": ["overlay-run1-baseline.png", "overlay-doodle-2560-seed12345.png"],
    "montage-n2叠线.png": ["overlay-run1-baseline.png", "overlay-doodle-n2-seed12345-1.png", "overlay-doodle-n2-seed12345-2.png"],
}
for name, parts in montages.items():
    paths = [OUT / p for p in parts if (OUT / p).exists()]
    print(name, T.montage(OUT / name, paths))
