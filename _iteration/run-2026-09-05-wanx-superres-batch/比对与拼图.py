"""比对两张图（缩到同尺寸逐像素）、量保真度、出叠线拼图。纯本地，不调 API。

用法：
  compare A B [--size W H]      两图各自 LANCZOS 缩到 --size（默认 A 的尺寸）后逐像素比
  fidelity LINE RESULT          调 run-2026-09-01-condition-matrix/保真度量.py 的 fidelity_score
  overlay LINE RESULT OUT       把线稿缩放到 RESULT 尺寸、线像素染红叠在 RESULT 上
  montage OUT IMG [IMG ...]     横排拼图，每张等比缩到高 640，底部写文件名
"""
import argparse, importlib.util, pathlib
import numpy as np
from PIL import Image, ImageDraw

FIDELITY_PY = pathlib.Path("/Users/baitianxing/codes/ishome-render3d/_iteration/run-2026-09-01-condition-matrix/保真度量.py")


def load_fidelity():
    spec = importlib.util.spec_from_file_location("fidelity", FIDELITY_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fidelity_score


def to_size(im: Image.Image, size) -> Image.Image:
    im = im.convert("RGB")
    return im if im.size == tuple(size) else im.resize(tuple(size), Image.LANCZOS)


def compare(a_path, b_path, size=None):
    a = Image.open(a_path); b = Image.open(b_path)
    size = tuple(size) if size else a.size
    aa = np.asarray(to_size(a, size), dtype=np.int16)
    bb = np.asarray(to_size(b, size), dtype=np.int16)
    d = np.abs(aa - bb)
    dmax = d.max(axis=2)
    return {
        "a": pathlib.Path(a_path).name, "a_size": list(a.size),
        "b": pathlib.Path(b_path).name, "b_size": list(b.size),
        "compare_size": list(size),
        "mean_abs_diff": float(d.mean()),
        "pct_pixels_maxch_diff_gt10": float((dmax > 10).mean() * 100),
        "pct_pixels_maxch_diff_gt30": float((dmax > 30).mean() * 100),
        "pct_pixels_identical": float((dmax == 0).mean() * 100),
    }


def overlay(line_path, result_path, out_path):
    res = Image.open(result_path).convert("RGB")
    line = Image.open(line_path).convert("L").resize(res.size, Image.BILINEAR)
    mask = np.asarray(line) > 127
    arr = np.array(res)
    arr[mask] = [255, 0, 0]
    Image.fromarray(arr).save(out_path)
    return res.size


def montage(out_path, paths, height=640):
    tiles = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        w = round(im.width * height / im.height)
        tiles.append((im.resize((w, height), Image.LANCZOS), pathlib.Path(p).name))
    W = sum(t.width for t, _ in tiles) + 10 * (len(tiles) + 1)
    canvas = Image.new("RGB", (W, height + 40), (30, 30, 30))
    d = ImageDraw.Draw(canvas)
    x = 10
    for t, name in tiles:
        canvas.paste(t, (x, 10))
        d.text((x, height + 16), name, fill=(230, 230, 230))
        x += t.width + 10
    canvas.save(out_path)
    return canvas.size


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("compare"); c.add_argument("a"); c.add_argument("b"); c.add_argument("--size", type=int, nargs=2)
    f = sp.add_parser("fidelity"); f.add_argument("line"); f.add_argument("result")
    o = sp.add_parser("overlay"); o.add_argument("line"); o.add_argument("result"); o.add_argument("out")
    m = sp.add_parser("montage"); m.add_argument("out"); m.add_argument("imgs", nargs="+")
    a = p.parse_args()
    if a.cmd == "compare":
        for k, v in compare(a.a, a.b, a.size).items():
            print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
    elif a.cmd == "fidelity":
        print(f"{load_fidelity()(pathlib.Path(a.line), pathlib.Path(a.result)):.4f}")
    elif a.cmd == "overlay":
        print("overlay", overlay(a.line, a.result, a.out))
    else:
        print("montage", montage(a.out, a.imgs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
