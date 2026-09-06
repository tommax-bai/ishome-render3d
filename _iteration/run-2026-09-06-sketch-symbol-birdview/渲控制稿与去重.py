"""揭顶机位（``cam-bird-dollhouse``）三方案控制稿 + 控制稿 sha256 去重清单。只在这一批用，写完不改。零模型调用。

跑法（仓根目录）：uv run python _iteration/run-2026-09-06-sketch-symbol-birdview/渲控制稿与去重.py

本批要回答的是"揭顶视角下 ``diagonal-cross`` 对 ``frame-sill`` / ``frame-handle``"，几何输入与前两批同：
16 洞机位包 ``run-2026-09-05-opening-kind-consume/design-package-真值138-16洞-roomcams.json``，1280×960，风格 modern。

**去重**（`run-2026-09-05-sketch-symbol-combo/run.md` 的教训：万相按 seed 确定，同稿同 seed 逐像素同图）：
先渲控制稿、算 sha256，与前两批落盘的揭顶控制稿逐字节比——相同就说明 seed 1/2/3 那九张图已经在仓里，
本批不重复调用，只花钱买**新 seed**。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from render3d_worker import base_render as br
from render3d_worker.models import DesignPackage
from render3d_worker.scene_compile import compile_scene_package

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
PACKAGE = ROOT / "_iteration/run-2026-09-05-opening-kind-consume/design-package-真值138-16洞-roomcams.json"
BIRD_CAMERA_ID = "cam-bird-dollhouse"
SCHEMES = ("diagonal-cross", "frame-sill", "frame-handle")
WIDTH_PX, HEIGHT_PX = 1280, 960

# 前两批落盘的同机位控制稿（用来证明本批控制稿与它们逐字节相同 → 那九张图可直接复用）
PRIOR = {
    "diagonal-cross": ROOT / "_iteration/run-2026-09-05-opening-kind-consume" / BIRD_CAMERA_ID / "sketch.png",
    "frame-sill": ROOT / "_iteration/run-2026-09-05-sketch-symbol-combo/frame-sill" / BIRD_CAMERA_ID / "sketch.png",
    "frame-handle": ROOT / "_iteration/run-2026-09-05-sketch-symbol-combo/frame-handle" / BIRD_CAMERA_ID / "sketch.png",
}
# 前两批已出的揭顶真跑图（seed 1/2/3），本批直接引用不重跑
PRIOR_RUNS = {
    "diagonal-cross": ROOT / "_iteration/run-2026-09-05-opening-kind-realism",
    "frame-sill": ROOT / "_iteration/run-2026-09-05-sketch-symbol-combo/真跑/frame-sill" / BIRD_CAMERA_ID,
    "frame-handle": ROOT / "_iteration/run-2026-09-05-sketch-symbol-combo/真跑/frame-handle" / BIRD_CAMERA_ID,
}


def _white(png: bytes) -> int:
    import io

    return int(np.count_nonzero(np.asarray(Image.open(io.BytesIO(png)).convert("L")) == br.SKETCH_FOREGROUND_U8))


def main() -> None:
    package = DesignPackage.model_validate(json.loads(PACKAGE.read_text(encoding="utf-8")))
    scene = compile_scene_package(package)

    manifest: dict[str, object] = {
        "cameraId": BIRD_CAMERA_ID,
        "widthPx": WIDTH_PX,
        "heightPx": HEIGHT_PX,
        "package": str(PACKAGE.relative_to(ROOT)),
        "schemes": {},
    }
    four_routes: dict[str, tuple[bytes, bytes, bytes, bytes]] = {}
    rows = [
        "| 方案 | 控制稿 sha256 | 白像素 | 与前两批落盘稿逐字节 | 前两批已出 seed 1/2/3 |",
        "|---|---|---|---|---|",
    ]
    for scheme in SCHEMES:
        views = br.render_base_views(scene, BIRD_CAMERA_ID, WIDTH_PX, HEIGHT_PX, scheme)  # type: ignore[arg-type]
        four_routes[scheme] = (views.geometry_png, views.depth_png, views.line_png, views.mask_png)
        out = HERE / "控制稿" / scheme
        out.mkdir(parents=True, exist_ok=True)
        (out / "sketch.png").write_bytes(views.sketch_png)
        digest = hashlib.sha256(views.sketch_png).hexdigest()
        prior_path = PRIOR[scheme]
        same = prior_path.exists() and prior_path.read_bytes() == views.sketch_png
        prior_seeds = sorted(
            p.name
            for p in PRIOR_RUNS[scheme].glob("*seed[123].png")
            if "sketch" not in p.name and BIRD_CAMERA_ID in f"{p.parent.name}/{p.name}"
        )
        manifest["schemes"][scheme] = {  # type: ignore[index]
            "sketchSha256": digest,
            "whitePixels": _white(views.sketch_png),
            "priorSketch": str(prior_path.relative_to(ROOT)),
            "byteIdenticalToPrior": same,
            "priorRunDir": str(PRIOR_RUNS[scheme].relative_to(ROOT)),
            "priorSeedImages": prior_seeds,
        }
        rows.append(
            f"| {scheme} | `{digest[:16]}…` | {_white(views.sketch_png)} | "
            f"{'同' if same else '**不同**'} | {'、'.join(prior_seeds) if prior_seeds else '无'} |"
        )

    # 四路（几何/深度/线稿/遮罩）跨方案逐字节相同——差别只在控制稿
    base = four_routes[SCHEMES[0]]
    for scheme in SCHEMES[1:]:
        assert four_routes[scheme] == base, f"{scheme}：四路与 {SCHEMES[0]} 不同"
    manifest["fourRoutesIdenticalAcrossSchemes"] = True

    (HERE / "控制稿sha清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(rows))
    print("四路（几何/深度/线稿/遮罩）跨三方案逐字节相同：是")


if __name__ == "__main__":
    main()
