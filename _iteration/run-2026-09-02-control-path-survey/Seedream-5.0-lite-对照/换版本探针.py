"""换 Seedream 版本探针：复用 方差探针.py 的提示词，直连方舟（/api/v3/images/generations）、换 model id。只在 scratchpad 用。"""
import sys, pathlib, os, time, json, argparse, base64, urllib.request, urllib.error
sys.path.insert(0, "/Users/baitianxing/codes/ishome-render3d/_iteration/run-2026-09-01-variance-knobs")
import 方差探针 as P
B = pathlib.Path("/Users/baitianxing/codes/ishome-render3d/_iteration/真户型-基准/底渲-cam-bird-dollhouse")
a = argparse.ArgumentParser(); a.add_argument("--model", required=True); a.add_argument("-o", required=True, type=pathlib.Path); a.add_argument("--extra", default="{}")
args = a.parse_args()
prompt = P.build_prompt("bird", P.STYLES["现代简约"], room_facts=False)
srcs = [(B/"geometry.png").read_bytes(), (B/"line.png").read_bytes()]
body = {"model": args.model, "prompt": prompt, "image": ["data:image/png;base64," + base64.b64encode(b).decode() for b in srcs],
        "size": "2K", "watermark": False, "response_format": "b64_json", **json.loads(args.extra)}
req = urllib.request.Request("https://ark.cn-beijing.volces.com/api/v3/images/generations", data=json.dumps(body).encode(),
      headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['ARK_API_KEY']}"})
t0 = time.monotonic()
try:
    with urllib.request.urlopen(req, timeout=400) as r: payload = json.load(r)
except urllib.error.HTTPError as e:
    print("失败：HTTP", e.code, e.read().decode()[:600]); raise SystemExit(3)
items = payload.get("data") or []
if not items or not items[0].get("b64_json"): print("失败：空图片列表", json.dumps(payload)[:400]); raise SystemExit(4)
img = base64.b64decode(items[0]["b64_json"]); meta = {k: v for k, v in payload.items() if k != "data"}
args.o.parent.mkdir(parents=True, exist_ok=True); args.o.write_bytes(img); args.o.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
print(f"出图 {args.o} {len(img)/1024:.0f}KB {time.monotonic()-t0:.1f}s usage={meta.get('usage')}")
