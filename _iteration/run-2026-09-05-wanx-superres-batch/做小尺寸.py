"""把 run1（1168x880）LANCZOS 缩到 683x512（高刚好到下限 512，保持 4:3），试超分对小图放不放大——
584x440 那次被拒（高 <512），这次踩着下限。"""
import os, pathlib
from PIL import Image

SRC = pathlib.Path("/Users/baitianxing/codes/ishome-render3d/_iteration/run-2026-09-02-control-path-survey/万相-doodle/run1-seed12345.png")
OUT = pathlib.Path(__file__).with_name("run1-small-683x512.png")
im = Image.open(SRC).convert("RGB").resize((683, 512), Image.LANCZOS)
im.save(OUT)
print(im.size, os.path.getsize(OUT) // 1024, "KB", OUT)
