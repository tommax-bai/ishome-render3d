"""把 1280x960 的线稿反色图用 NEAREST 放大到 2560x1920，做"大线稿直送"的条件图。"""
import os, pathlib
from PIL import Image

SRC = pathlib.Path("/Users/baitianxing/codes/ishome-render3d/_iteration/run-2026-09-02-control-path-survey/万相-doodle/条件图-线稿反色.png")
OUT = pathlib.Path(__file__).with_name("条件图-线稿反色-2560x1920-nearest.png")

im = Image.open(SRC).convert("RGB")
big = im.resize((2560, 1920), Image.NEAREST)
big.save(OUT)
print(big.size, os.path.getsize(OUT) // 1024, "KB", OUT)
