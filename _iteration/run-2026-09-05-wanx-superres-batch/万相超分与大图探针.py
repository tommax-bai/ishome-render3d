"""万相 wanx2.1-imageedit 三件探针：super_resolution 超分 / 大线稿直送 doodle / doodle n=2。
提交与轮询写法照抄 run-2026-09-02-control-path-survey/万相-doodle/万相线稿生图探针.py。
每次调用落 <out>.meta.json：请求参数（去掉 base64）、提交回执、轮询终态、耗时、usage。
n>1 时出图落 <out>-1.png、<out>-2.png。
"""
import argparse, base64, io, json, os, pathlib, sys, time, urllib.request, urllib.error
from PIL import Image

BASE = "https://dashscope.aliyuncs.com"
SUBMIT = BASE + "/api/v1/services/aigc/image2image/image-synthesis"

DOODLE_PROMPT = ("揭顶式户型鸟瞰效果图：去掉天花板从上方斜俯视整套公寓，墙体、门窗洞口和房间划分严格按线稿，"
                 "不增减墙体、不合并或拆分房间。现代简约风格，暖白哑光墙面、浅橡木宽板地板、灰色亚麻布艺、"
                 "黑色金属细节，柔和自然光，写实建筑可视化渲染。无文字、无水印、无标注。")
SR_PROMPT = "图像超分"


def load_key_from_env_file(path: pathlib.Path = pathlib.Path.home() / ".ishome" / "llm-local.env") -> str:
    """凭证在 ~/.ishome/llm-local.env（KEY=VALUE 一行一条）；这里只读 DASHSCOPE_API_KEY，不打印。"""
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("DASHSCOPE_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"{path} 里没有 DASHSCOPE_API_KEY")


def data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def submit_and_poll(api_key: str, body: dict):
    req = urllib.request.Request(SUBMIT, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}",
                 "X-DashScope-Async": "enable"})
    t_submit = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as r:
        sub = json.load(r)
    task_id = sub["output"]["task_id"]
    t0 = time.monotonic()
    while True:
        time.sleep(3)
        q = urllib.request.Request(BASE + f"/api/v1/tasks/{task_id}", headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(q, timeout=60) as r:
            st = json.load(r)
        s = st["output"]["task_status"]
        if s in ("SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"):
            break
        if time.monotonic() - t0 > 600:
            raise RuntimeError("超时 600s")
    return sub, st, time.monotonic() - t0, t0 - t_submit


def run(api_key: str, body: dict, out: pathlib.Path, note: dict) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    shown = json.loads(json.dumps(body))
    shown["input"]["base_image_url"] = f"<data:image/png;base64, {len(body['input']['base_image_url'])} chars>"
    meta = {"request": shown, **note}
    try:
        sub, st, poll_s, submit_s = submit_and_poll(api_key, body)
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        meta["http_error"] = {"code": e.code, "body": err[:2000]}
        out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        print("HTTP", e.code, err[:800], file=sys.stderr)
        return 3
    meta.update({"submit_response": sub, "final": st, "poll_seconds": round(poll_s, 1),
                 "submit_seconds": round(submit_s, 2)})
    if st["output"]["task_status"] != "SUCCEEDED":
        out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        print("失败：", json.dumps(st["output"], ensure_ascii=False)[:800], file=sys.stderr)
        return 4
    results = st["output"]["results"]
    files = []
    for i, res in enumerate(results, 1):
        img = urllib.request.urlopen(res["url"], timeout=180).read()
        p = out if len(results) == 1 else out.with_name(f"{out.stem}-{i}{out.suffix}")
        p.write_bytes(img)
        im = Image.open(io.BytesIO(img))
        files.append({"file": p.name, "size": list(im.size), "format": im.format, "bytes": len(img)})
        print(f"出图 {p} {im.size} {im.format} ({len(img)/1024:.0f} KB)")
    meta["outputs"] = files
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"轮询 {poll_s:.1f}s（提交往返 {submit_s:.2f}s） usage={st.get('usage')}")
    return 0


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("sr", help="function=super_resolution")
    a.add_argument("--image", required=True, type=pathlib.Path)
    a.add_argument("--factor", type=int, required=True)
    a.add_argument("-o", "--out", required=True, type=pathlib.Path)
    b = sp.add_parser("doodle", help="function=doodle is_sketch=true，线稿图原样直送（不反色不缩放）")
    b.add_argument("--sketch", required=True, type=pathlib.Path)
    b.add_argument("--seed", type=int, default=None)
    b.add_argument("--n", type=int, default=1)
    b.add_argument("-o", "--out", required=True, type=pathlib.Path)
    args = p.parse_args()
    key = os.environ.get("DASHSCOPE_API_KEY") or load_key_from_env_file()

    if args.cmd == "sr":
        raw = args.image.read_bytes()
        im = Image.open(io.BytesIO(raw))
        body = {"model": "wanx2.1-imageedit",
                "input": {"function": "super_resolution", "prompt": SR_PROMPT, "base_image_url": data_url(raw)},
                "parameters": {"upscale_factor": args.factor, "n": 1, "watermark": False}}
        note = {"input_file": str(args.image), "input_size": list(im.size), "input_bytes": len(raw)}
    else:
        raw = args.sketch.read_bytes()
        im = Image.open(io.BytesIO(raw))
        body = {"model": "wanx2.1-imageedit",
                "input": {"function": "doodle", "prompt": DOODLE_PROMPT, "base_image_url": data_url(raw)},
                "parameters": {"is_sketch": True, "n": args.n, "watermark": False}}
        if args.seed is not None:
            body["parameters"]["seed"] = args.seed
        note = {"input_file": str(args.sketch), "input_size": list(im.size), "input_bytes": len(raw)}
    return run(key, body, args.out, note)


if __name__ == "__main__":
    raise SystemExit(main())
