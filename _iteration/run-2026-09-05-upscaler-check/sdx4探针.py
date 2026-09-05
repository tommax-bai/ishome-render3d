"""SD x4 upscaler 可行性探针：不放整图，只量"一块 tile 要多久、两次是否一致、内容改没改"。

整图 1168×880 直接送 UNet 不可能：这条管线的潜空间分辨率 = 输入分辨率（VAE 缩放因子 4，输出 ×4），
UNet 在 1/2 分辨率层就有自注意力，584×440 = 257k token 的注意力矩阵（fp16）= 257k² × 2 B ≈ 132 GB，
16 GB 机器上只能分块。所以量的是 tile：256×256 输入 → 1024×1024 输出，20 步（diffusers 文档示例步数；
管线默认 75 步），guidance 默认 9，seed 固定，跑两次比 sha256。

用法：sdx4_probe.py --input IMG --out-dir DIR [--tile 256] [--steps 20] [--runs 2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np
import torch
from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--origin", type=int, nargs=2, default=(400, 430))
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--attention-slicing", action="store_true", help="pipe.enable_attention_slicing('max')，按头切注意力省显存")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    from diffusers import StableDiffusionUpscalePipeline

    device = "mps"
    t0 = time.perf_counter()
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16, variant="fp16"
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    if a.attention_slicing:
        pipe.enable_attention_slicing("max")
    load_s = time.perf_counter() - t0

    img = Image.open(a.input).convert("RGB")
    x0, y0 = a.origin
    tile = img.crop((x0, y0, x0 + a.tile, y0 + a.tile))
    tile.save(a.out_dir / f"sdx4-tile{a.tile}-input.png")

    meta = {"pipeline": "StableDiffusionUpscalePipeline", "model": "stabilityai/stable-diffusion-x4-upscaler(fp16)",
            "device": device, "torch": torch.__version__, "attention_slicing": a.attention_slicing, "tile_input": [a.tile, a.tile], "origin": list(a.origin),
            "steps": a.steps, "seed": a.seed, "load_seconds": round(load_s, 2), "runs": []}
    for k in range(1, a.runs + 1):
        gen = torch.Generator(device="cpu").manual_seed(a.seed)
        t0 = time.perf_counter()
        out = pipe(prompt="", image=tile, num_inference_steps=a.steps, generator=gen).images[0]
        torch.mps.synchronize()
        secs = time.perf_counter() - t0
        p = a.out_dir / f"sdx4-tile{a.tile}-run{k}.png"
        out.save(p)
        h = hashlib.sha256(np.asarray(out).tobytes()).hexdigest()
        # 内容改没改：×4 结果 LANCZOS 缩回 tile 尺寸与输入逐像素比
        back = np.asarray(out.resize(tile.size, Image.LANCZOS), dtype=np.int16)
        d = np.abs(back - np.asarray(tile, dtype=np.int16)); dmax = d.max(axis=2)
        rec = {"run": k, "seconds": round(secs, 2), "output_size": list(out.size), "array_sha256": h,
               "vs_input_downscaled": {"mean_abs_diff": round(float(d.mean()), 3),
                                       "pct_pixels_maxch_diff_gt10": round(float((dmax > 10).mean() * 100), 2),
                                       "pct_pixels_maxch_diff_gt30": round(float((dmax > 30).mean() * 100), 2)},
               "mps_driver_allocated_mb": round(torch.mps.driver_allocated_memory() / 2**20)}
        meta["runs"].append(rec)
        print(f"run{k}: {out.size} {secs:.1f}s sha256[:12]={h[:12]} mean_diff={rec['vs_input_downscaled']['mean_abs_diff']} "
              f">10={rec['vs_input_downscaled']['pct_pixels_maxch_diff_gt10']}% mps={rec['mps_driver_allocated_mb']}MB", flush=True)
    meta["deterministic_across_runs"] = len({r["array_sha256"] for r in meta["runs"]}) == 1
    # 外推：整图按 tile 无重叠平铺的块数（有重叠只会更多）
    import math
    n_tiles = math.ceil(1168 / a.tile) * math.ceil(880 / a.tile)
    warm = meta["runs"][-1]["seconds"]
    meta["extrapolation_full_image"] = {"tiles_no_overlap": n_tiles, "seconds_at_warm_tile_time": round(n_tiles * warm, 1),
                                        "output_size_x4": [1168 * 4, 880 * 4]}
    (a.out_dir / f"sdx4-tile{a.tile}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"deterministic={meta['deterministic_across_runs']} load={load_s:.1f}s extrapolation={meta['extrapolation_full_image']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
