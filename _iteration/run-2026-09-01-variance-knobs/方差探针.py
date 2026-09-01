"""方差探针：独立于 imagegen-pinned（那个 checkout 目前不在场）复刻 realism-试跑.py 的调用逻辑，
额外开一个 extra_body 口子用来探测 seed / guidance_scale 等非标准参数。

只在 scratchpad 里用，不进任何 git 仓。提示词常量与 realism-试跑.py 保持字面一致（照抄），
这样跟已知噪声底（0.2197/0.1201/0.0479，条件矩阵/真户型-现代简约-geo+line*.png）严格可比。
"""
from __future__ import annotations

import argparse
import base64
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

REALISM_MODEL = "realism-pass.default"

_GEOMETRY_CLAUSE = (
    "The input image is an untextured, flat-shaded 3D base render of a real apartment "
    "interior. It is the ONLY source of geometry, camera and layout. Keep the camera "
    "position, viewing angle and field of view EXACTLY as given. Keep every wall, floor, "
    "ceiling, door opening, window opening and furniture piece at exactly its given "
    "position, size, proportion and orientation. Do not add, remove, move, resize or "
    "reshape any wall, opening or furniture piece. Do not invent additional rooms, "
    "windows or doors. Do not change the room's footprint."
)

_TASK_CLAUSE = (
    "Task: turn this base render into a photorealistic architectural interior "
    "visualization, as if photographed with a full-frame camera and a 24mm tilt-shift "
    "lens. Replace the flat placeholder colors with believable physically-based "
    "materials, and light the scene with soft natural daylight entering through the "
    "window openings plus subtle warm interior fill. Add realistic contact shadows, "
    "soft global illumination, accurate material reflections and fine surface texture."
)

_NO_TEXT_CLAUSE = (
    "Render NO text of any kind: no letters, no Chinese characters, no numbers, "
    "no labels, no captions, no watermarks, no dimension annotations."
)

_NEGATIVES = (
    "changing the camera angle or focal length",
    "adding or removing walls, doors, windows or furniture",
    "wide-angle or fisheye distortion, curved walls, tilted verticals",
    "cartoon, illustration, painterly or CGI-clay look",
    "people, pets, text, logos, watermarks",
    "blown-out highlights or crushed blacks that hide the geometry",
)

_SCENE_CLAUSES = {
    "room": (
        "This is an eye-level interior view standing inside the room. Ground the "
        "furniture with believable contact shadows on the floor."
    ),
    "bird": (
        "This is a top-down dollhouse view of a whole apartment with the ceiling removed: "
        "the walls are cut open so the interior is visible from above. Keep it reading as "
        "an architectural model of one floor — do NOT close the ceiling back up, and do "
        "NOT turn it into a flat 2D floor plan."
    ),
}

STYLES = {
    "现代简约": (
        "contemporary minimalist Chinese urban apartment. Warm off-white matte walls, "
        "wide-plank light oak flooring, soft grey linen upholstery, black powder-coated "
        "metal details, one or two muted sage accents. Calm, bright, uncluttered."
    ),
}

# ---- 候选改法一：房间数 + 外轮廓形状显式约束（数字来自 real-run/底渲/scene-package.json，
# 不是编的：7 个房间名各出现一次；实际地面面积/外接矩形面积 = 86.7/119.8 ≈ 0.72，
# 外轮廓不是矩形，缺角比例不小）----
_ROOM_FACTS_CLAUSE = (
    "Ground-truth floor plan facts (read from the geometry, do not override them): this "
    "apartment has EXACTLY 7 rooms, no more and no fewer — living room, bathroom, study, "
    "balcony, primary bedroom, secondary bedroom, kitchen. Do not add a room, remove a "
    "room, split a room into two, or merge two rooms into one open-plan space. The "
    "apartment's outer footprint is NOT a simple rectangle: its actual floor area is only "
    "about 72% of its bounding rectangle's area, i.e. a real notch is cut out of one "
    "corner of the outline. Do not straighten this notch into a rectangle."
)


# ---- 候选改法二：把第二张图（line.png）的角色在提示词里点明为墙体轴线的权威来源 ----
_LINE_AUTHORITY_CLAUSE = (
    "You are given two images of the exact same view. The FIRST image is the flat-shaded "
    "3D base render (materials and camera reference). The SECOND image is a line drawing "
    "of the wall centerlines — it is the single authoritative source for where every wall, "
    "opening and room boundary actually is. Wherever the two images could be read as "
    "disagreeing, follow the SECOND image's line positions exactly for wall placement and "
    "room count; never let material shading in the first image imply a different wall "
    "layout than the lines in the second image show."
)


def build_prompt(scene_kind: str, style: str, *, room_facts: bool, line_authority: bool = False) -> str:
    lines = [_GEOMETRY_CLAUSE]
    if room_facts:
        lines.append(_ROOM_FACTS_CLAUSE)
    if line_authority:
        lines.append(_LINE_AUTHORITY_CLAUSE)
    lines += [
        _TASK_CLAUSE,
        _SCENE_CLAUSES[scene_kind],
        f"Interior style: {style}",
        _NO_TEXT_CLAUSE,
    ]
    lines.extend(f"Avoid: {item}" for item in _NEGATIVES)
    return "\n".join(lines)


class GatewayError(Exception):
    def __init__(self, details: list[str]) -> None:
        super().__init__("；".join(details))
        self.details = details


def generate(
    *,
    prompt: str,
    source_pngs: list[bytes],
    size: str,
    api_key: str,
    gateway_url: str,
    extra_body: dict,
) -> tuple[bytes, dict]:
    body = {
        "model": REALISM_MODEL,
        "prompt": prompt,
        "image": ["data:image/png;base64," + base64.b64encode(b).decode() for b in source_pngs],
        "size": size,
        "watermark": False,
        "n": 1,
        "response_format": "b64_json",
        **extra_body,
    }
    request = urllib.request.Request(
        gateway_url + "/v1/images/generations",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as e:
        raise GatewayError([f"网关拒绝（HTTP {e.code}）：{e.read().decode()[:800]}"]) from e
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise GatewayError([f"网关连不上或回了不认识的东西：{e}"]) from e

    sent = payload.get("usage", {}).get("input_images")
    if sent != len(source_pngs):
        raise GatewayError(
            [f"条件图没全送到模型（送了 {len(source_pngs)} 张，回执 input_images={sent!r}）"]
        )
    items = payload.get("data") or []
    if not items or not items[0].get("b64_json"):
        raise GatewayError(["网关回了空的图片列表"])
    image_bytes = base64.b64decode(items[0]["b64_json"])
    meta = {k: v for k, v in payload.items() if k != "data"}
    meta["revised_prompt"] = items[0].get("revised_prompt") or ""
    return image_bytes, meta


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, type=pathlib.Path, nargs="+")
    p.add_argument("--scene", default="bird", choices=sorted(_SCENE_CLAUSES))
    p.add_argument("--style", default="现代简约", choices=sorted(STYLES))
    p.add_argument("--size", default="2K")
    p.add_argument("--room-facts", action="store_true", help="候选改法一：加房间数+外轮廓约束句")
    p.add_argument("--line-authority", action="store_true", help="候选改法二：点明 line.png 是墙体轴线权威来源")
    p.add_argument("--extra-body", default="{}", help="JSON 字典，原样并入请求体，用于探参数")
    p.add_argument("-o", "--out", required=True, type=pathlib.Path)
    p.add_argument("--gateway", default="http://127.0.0.1:4000")
    args = p.parse_args()

    import os

    api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("LITELLM_MASTER_KEY")
    if not api_key:
        print("没有网关凭证", file=sys.stderr)
        return 2

    extra_body = json.loads(args.extra_body)
    source_pngs = [ip.read_bytes() for ip in args.images]
    prompt = build_prompt(
        args.scene, STYLES[args.style], room_facts=args.room_facts, line_authority=args.line_authority
    )

    started = time.monotonic()
    try:
        image_bytes, meta = generate(
            prompt=prompt,
            source_pngs=source_pngs,
            size=args.size,
            api_key=api_key,
            gateway_url=args.gateway,
            extra_body=extra_body,
        )
    except GatewayError as e:
        for d in e.details:
            print(f"失败：{d}", file=sys.stderr)
        return 3
    elapsed = time.monotonic() - started

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(image_bytes)
    args.out.with_suffix(".prompt.txt").write_text(prompt, encoding="utf-8")
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"出图 {args.out}（{len(image_bytes)/1024:.0f} KB，{elapsed:.1f}s）")
    print(f"extra_body={extra_body}")
    print(f"回执 meta（去掉图片数据）: {json.dumps(meta, ensure_ascii=False)[:500]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
