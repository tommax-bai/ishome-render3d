"""真户型 16 洞（c9fc00c 产物）书房 / 主卧 / 客厅三台机位 × 全部符号方案的控制稿 + 「方案并排」拼图。
只在这一批用，写完不改。零模型调用。

跑法（仓根目录）：uv run python _iteration/run-2026-09-05-sketch-symbols/渲控制稿与拼图.py

自证：默认方案 ``diagonal-cross`` 渲出来的 sketch.png 与 ``run-2026-09-05-opening-kind-consume/<机位>/sketch.png``
逐字节相同（换方案没有动默认那条路），四路（几何/深度/线稿/遮罩）在各方案之间逐字节相同。
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from render3d_worker import base_render as br
from render3d_worker.models import DesignPackage
from render3d_worker.scene_compile import compile_scene_package

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
PACKAGE = ROOT / "_iteration/run-2026-09-05-opening-kind-consume/design-package-真值138-16洞-roomcams.json"
PRIOR_SKETCH_DIR = ROOT / "_iteration/run-2026-09-05-opening-kind-consume"
CAMERA_IDS = ("cam-room-书房", "cam-room-主卧", "cam-room-客厅")
WIDTH_PX, HEIGHT_PX = 1280, 960
TILE = (640, 480)
LABEL_H = 30


def _font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc"):
        try:
            return ImageFont.truetype(candidate, 22)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    package = DesignPackage.model_validate(json.loads(PACKAGE.read_text(encoding="utf-8")))
    scene = compile_scene_package(package)
    schemes = br.SKETCH_SYMBOL_SCHEMES
    font = _font()

    montage = Image.new("RGB", (TILE[0] * len(schemes), (TILE[1] + LABEL_H) * len(CAMERA_IDS)), (20, 20, 20))
    draw = ImageDraw.Draw(montage)
    print("| 机位 | 方案 | 白像素数 | 与默认方案的白像素差 | sha256 前 12 位 |")
    print("|---|---|---|---|---|")
    for row, camera_id in enumerate(CAMERA_IDS):
        default_views = None
        for col, scheme in enumerate(schemes):
            views = br.render_base_views(scene, camera_id, WIDTH_PX, HEIGHT_PX, scheme)
            out_dir = HERE / scheme / camera_id
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "sketch.png").write_bytes(views.sketch_png)
            if scheme == br.DEFAULT_SKETCH_SYMBOLS:
                default_views = views
                prior = PRIOR_SKETCH_DIR / camera_id / "sketch.png"
                same = prior.read_bytes() == views.sketch_png
                print(f"<!-- {camera_id} 默认方案与 {prior.relative_to(ROOT)} 逐字节相同：{same} -->")
                assert same, f"{camera_id}：默认方案的控制稿变了"
            else:
                assert default_views is not None
                assert views.geometry_png == default_views.geometry_png
                assert views.depth_png == default_views.depth_png
                assert views.line_png == default_views.line_png
                assert views.mask_png == default_views.mask_png
            image = Image.open(io.BytesIO(views.sketch_png)).convert("L")
            white = sum(1 for value in image.getdata() if value == br.SKETCH_FOREGROUND_U8)
            default_white = sum(
                1
                for value in Image.open(io.BytesIO(default_views.sketch_png)).convert("L").getdata()
                if value == br.SKETCH_FOREGROUND_U8
            ) if default_views is not None else white
            digest = hashlib.sha256(views.sketch_png).hexdigest()[:12]
            print(f"| {camera_id} | {scheme} | {white} | {white - default_white:+d} | {digest} |")

            x = col * TILE[0]
            y = row * (TILE[1] + LABEL_H)
            draw.text((x + 8, y + 4), f"{camera_id}  {scheme}" + ("（默认）" if col == 0 else ""), fill=(255, 200, 80), font=font)
            montage.paste(image.resize(TILE).convert("RGB"), (x, y + LABEL_H))
    montage.save(HERE / "montage-方案并排.png")
    print(f"拼图：{(HERE / 'montage-方案并排.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
