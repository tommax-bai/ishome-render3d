"""把 run1（1168x880）LANCZOS 缩到一半 584x440，用来试超分对小图放不放大——回答"输出是不是被封顶"。"""
import os, pathlib
from PIL import Image

SRC = pathlib.Path("/Users/baitianxing/codes/ishome-render3d/_iteration/run-2026-09-02-control-path-survey/万相-doodle/run1-seed12345.png")
OUT = pathlib.Path(__file__).with_name("run1-half-584x440.png")
im = Image.open(SRC).convert("RGB").resize((584, 440), Image.LANCZOS)
im.save(OUT)
print(im.size, os.path.getsize(OUT) // 1024, "KB", OUT)
