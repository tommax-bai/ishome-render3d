"""本机确定性放大器真跑驱动：一个候选 × 一张输入 × N 次重复，记尺寸/耗时/哈希。纯本地，不调 API。

用法：
  放大.py --model lanczos-x2|realesrgan-x2plus|realesrgan-x4plus|realesr-general-x4v3|swinir-m-x2-realsr \
          --input IMG --out-dir DIR [--device mps|cpu] [--tile N --tile-pad P] [--runs 2]

产物（写进 --out-dir）：`<模型>-run<k>.png` 与 `<模型>.meta.json`（模型信息、每次的耗时、输出尺寸、
数组 sha256、PNG 文件 sha256、MPS 驱动占用）。确定性判据 = 各次数组 sha256 逐字节相同。

神经网络候选一律 fp32、`torch.inference_mode()`，不开 half（MPS 上 half 的卷积累加顺序不保证一致）。
权重放在 scratchpad `upscale/weights/`（下载记录见同目录 download.log），用 spandrel 0.4.2 读取；
spandrel 的 ImageModelDescriptor 会按各架构的尺寸要求（SwinIR 窗口 8 的整数倍）自动补边再裁回。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import time

import numpy as np
import torch
from PIL import Image

WEIGHTS_DIR = pathlib.Path(
    "/private/tmp/claude-501/-Users-baitianxing-codes-ishome/5043122a-4d91-4234-8b80-a5febe79f775/scratchpad/upscale/weights"
)
MODELS: dict[str, pathlib.Path | None] = {
    "lanczos-x2": None,
    "realesrgan-x2plus": WEIGHTS_DIR / "RealESRGAN_x2plus.pth",
    "realesrgan-x4plus": WEIGHTS_DIR / "RealESRGAN_x4plus.pth",
    "realesr-general-x4v3": WEIGHTS_DIR / "realesr-general-x4v3.pth",
    "swinir-m-x2-realsr": WEIGHTS_DIR / "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth",
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_model(name: str, device: str):
    """返回 (可调用对象, 倍数, 模型信息 dict)。lanczos 返回 None 作为可调用对象。"""
    path = MODELS[name]
    if path is None:
        return None, 2, {"kind": "PIL.Image.resize LANCZOS", "scale": 2}
    import spandrel

    desc = spandrel.ModelLoader().load_from_file(str(path))
    desc = desc.to(device).eval()
    info = {
        "kind": "spandrel",
        "weight_file": path.name,
        "weight_sha256": sha256_bytes(path.read_bytes()),
        "architecture": desc.architecture.name,
        "tags": list(desc.tags),
        "scale": desc.scale,
        "input_channels": desc.input_channels,
        "output_channels": desc.output_channels,
        "tiling": str(desc.tiling),
        "size_requirements": repr(desc.size_requirements),
        "n_params": int(sum(p.numel() for p in desc.model.parameters())),
    }
    return desc, desc.scale, info


def tiled_forward(model, x: torch.Tensor, scale: int, tile: int, pad: int) -> torch.Tensor:
    """带重叠的分块推理：每块外扩 pad 像素送模型，只取中心区贴回。顺序固定、无随机，结果确定。"""
    _, c, h, w = x.shape
    out = torch.zeros((1, c, h * scale, w * scale), dtype=x.dtype, device=x.device)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            ys, ye = max(0, y0 - pad), min(h, y1 + pad)
            xs, xe = max(0, x0 - pad), min(w, x1 + pad)
            o = model(x[:, :, ys:ye, xs:xe])
            oy, ox = (y0 - ys) * scale, (x0 - xs) * scale
            out[:, :, y0 * scale : y1 * scale, x0 * scale : x1 * scale] = o[
                :, :, oy : oy + (y1 - y0) * scale, ox : ox + (x1 - x0) * scale
            ]
    return out


def run_once(model, scale: int, img: Image.Image, device: str, tile: int, pad: int) -> tuple[Image.Image, float]:
    t0 = time.perf_counter()
    if model is None:
        out = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
        return out, time.perf_counter() - t0
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    with torch.inference_mode():
        y = tiled_forward(model, x, scale, tile, pad) if tile > 0 else model(x)
        y = y.clamp_(0.0, 1.0)
        if device == "mps":
            torch.mps.synchronize()
        y = y.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out = Image.fromarray((y * 255.0).round().astype(np.uint8), mode="RGB")
    return out, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--input", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--tile", type=int, default=0, help="0=整图一次过；>0 为分块边长（输入坐标）")
    ap.add_argument("--tile-pad", type=int, default=32)
    ap.add_argument("--runs", type=int, default=2)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(a.input).convert("RGB")
    t0 = time.perf_counter()
    model, scale, info = load_model(a.model, a.device)
    load_s = time.perf_counter() - t0

    meta = {
        "model": a.model,
        "model_info": info,
        "input": str(a.input),
        "input_size": list(img.size),
        "device": a.device if model is not None else "cpu(PIL)",
        "tile": a.tile,
        "tile_pad": a.tile_pad,
        "load_seconds": round(load_s, 3),
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pillow": Image.__version__ if hasattr(Image, "__version__") else None,
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
        },
        "runs": [],
    }
    try:
        import spandrel  # noqa: F401

        meta["env"]["spandrel"] = spandrel.__version__
    except Exception:  # pragma: no cover
        pass

    for k in range(1, a.runs + 1):
        out, secs = run_once(model, scale, img, a.device, a.tile, a.tile_pad)
        png_path = a.out_dir / f"{a.model}-run{k}.png"
        out.save(png_path, compress_level=6)
        arr_hash = sha256_bytes(np.asarray(out).tobytes())
        rec = {
            "run": k,
            "seconds": round(secs, 3),
            "output_size": list(out.size),
            "array_sha256": arr_hash,
            "png_sha256": sha256_bytes(png_path.read_bytes()),
            "png_bytes": png_path.stat().st_size,
        }
        if a.device == "mps" and model is not None:
            rec["mps_driver_allocated_mb"] = round(torch.mps.driver_allocated_memory() / 2**20)
        meta["runs"].append(rec)
        print(f"{a.model} run{k}: {out.size} {secs:.2f}s sha256[:12]={arr_hash[:12]}", flush=True)

    hashes = {r["array_sha256"] for r in meta["runs"]}
    meta["deterministic_across_runs"] = len(hashes) == 1
    (a.out_dir / f"{a.model}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"deterministic={meta['deterministic_across_runs']} load={load_s:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
