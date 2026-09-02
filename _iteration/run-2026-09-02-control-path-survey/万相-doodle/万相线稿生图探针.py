"""万相 wanx2.1-imageedit function=doodle 探针：把 render3d 的 line.png 当线稿送进去。
只在 scratchpad 用。判据：旋转 90° 出图跟不跟着转；同 seed 逐像素；三跑极差。
"""
import argparse, base64, io, json, os, pathlib, sys, time, urllib.request, urllib.error
from PIL import Image, ImageOps

BASE = "https://dashscope.aliyuncs.com"
SUBMIT = BASE + "/api/v1/services/aigc/image2image/image-synthesis"

PROMPT = ("揭顶式户型鸟瞰效果图：去掉天花板从上方斜俯视整套公寓，墙体、门窗洞口和房间划分严格按线稿，"
          "不增减墙体、不合并或拆分房间。现代简约风格，暖白哑光墙面、浅橡木宽板地板、灰色亚麻布艺、"
          "黑色金属细节，柔和自然光，写实建筑可视化渲染。无文字、无水印、无标注。")

def prep_sketch(line_png: pathlib.Path, rotate90: bool, raw: bool = False) -> bytes:
    if raw:
        im = Image.open(line_png).convert("RGB")
        buf = io.BytesIO(); im.save(buf, format="PNG"); return buf.getvalue()
    im = Image.open(line_png).convert("L")
    im = ImageOps.invert(im)            # 白底黑线
    if rotate90:
        im = im.rotate(90, expand=True)
    buf = io.BytesIO(); im.convert("RGB").save(buf, format="PNG"); return buf.getvalue()

def call(api_key, sketch_png, seed, is_sketch=True, n=1):
    body = {"model": "wanx2.1-imageedit",
            "input": {"function": "doodle", "prompt": PROMPT,
                      "base_image_url": "data:image/png;base64," + base64.b64encode(sketch_png).decode()},
            "parameters": {"is_sketch": is_sketch, "n": n, "watermark": False}}
    if seed is not None: body["parameters"]["seed"] = seed
    req = urllib.request.Request(SUBMIT, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}", "X-DashScope-Async": "enable"})
    with urllib.request.urlopen(req, timeout=60) as r: sub = json.load(r)
    task_id = sub["output"]["task_id"]
    t0 = time.monotonic()
    while True:
        time.sleep(3)
        q = urllib.request.Request(BASE + f"/api/v1/tasks/{task_id}", headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(q, timeout=60) as r: st = json.load(r)
        s = st["output"]["task_status"]
        if s in ("SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"): break
        if time.monotonic() - t0 > 600: raise RuntimeError("超时 600s")
    return st, time.monotonic() - t0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--line", required=True, type=pathlib.Path)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--rotate90", action="store_true")
    p.add_argument("--no-sketch", action="store_true", help="is_sketch=false：让模型自己抽线稿（对照）")
    p.add_argument("--raw", action="store_true", help="原图直送，不反色（给 geometry.png 用）")
    p.add_argument("-o", "--out", required=True, type=pathlib.Path)
    a = p.parse_args()
    key = os.environ["DASHSCOPE_API_KEY"]
    sketch = prep_sketch(a.line, a.rotate90, raw=a.raw)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".sketch.png").write_bytes(sketch)
    try:
        st, el = call(key, sketch, a.seed, is_sketch=not a.no_sketch)
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:800], file=sys.stderr); return 3
    meta = {k: v for k, v in st.items()}
    a.out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    if st["output"]["task_status"] != "SUCCEEDED":
        print("失败：", json.dumps(st["output"], ensure_ascii=False)[:800], file=sys.stderr); return 4
    url = st["output"]["results"][0]["url"]
    img = urllib.request.urlopen(url, timeout=120).read()
    a.out.write_bytes(img)
    im = Image.open(io.BytesIO(img))
    print(f"出图 {a.out} {im.size} {im.format} ({len(img)/1024:.0f} KB, {el:.1f}s) usage={st.get('usage')} seed={a.seed}")
    return 0

if __name__ == "__main__": raise SystemExit(main())
