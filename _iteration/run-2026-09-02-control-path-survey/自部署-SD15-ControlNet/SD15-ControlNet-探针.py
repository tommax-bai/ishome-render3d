"""自部署机制验证：SD1.5 + ControlNet(depth + mlsd)，吃 render3d 的 depth/line 两路当控制图。
只证机制（控制通道是否真约束几何），不证质量。M1 Pro 16GB, MPS。"""
import argparse, pathlib, time, json
import numpy as np, torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

B = pathlib.Path("/Users/baitianxing/codes/ishome-render3d/_iteration/真户型-基准/底渲-cam-bird-dollhouse")
PROMPT = ("photorealistic dollhouse view of an apartment interior with the ceiling removed, seen from above at an angle, "
          "contemporary minimalist style, warm off-white matte walls, light oak plank flooring, soft daylight, "
          "architectural visualization, highly detailed")
NEG = "text, watermark, blurry, cartoon, people, distorted walls, extra rooms"

def cond_images(rot90: bool, size):
    d16 = np.asarray(Image.open(B/"depth.png")).astype(np.float64)
    valid = d16 > 0
    lo, hi = d16[valid].min(), d16[valid].max()
    near_bright = np.zeros_like(d16); near_bright[valid] = 1.0 - (d16[valid]-lo)/(hi-lo)   # ControlNet depth: 近亮远暗
    depth = Image.fromarray((near_bright*255).astype(np.uint8)).convert("RGB")
    line = Image.open(B/"line.png").convert("RGB")                                        # MLSD: 黑底白线
    if rot90:
        depth = depth.rotate(90, expand=True); line = line.rotate(90, expand=True)
        size = (size[1], size[0])
    return depth.resize(size, Image.BILINEAR), line.resize(size, Image.NEAREST), size

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True); p.add_argument("--rotate90", action="store_true")
    p.add_argument("--steps", type=int, default=20); p.add_argument("--width", type=int, default=512); p.add_argument("--height", type=int, default=384)
    p.add_argument("--scales", default="0.8,0.8", help="depth,mlsd 的 conditioning_scale")
    p.add_argument("-o", "--out", required=True, type=pathlib.Path)
    a = p.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    depth, line, size = cond_images(a.rotate90, (a.width, a.height))
    cn = [ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16, variant="fp16"),
          ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_mlsd", torch_dtype=torch.float16, variant="fp16")]
    pipe = StableDiffusionControlNetPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=cn,
            torch_dtype=torch.float16, variant="fp16", safety_checker=None).to(dev)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    print("device", dev, "unet dtype", pipe.unet.dtype, flush=True)
    scales = [float(s) for s in a.scales.split(",")]
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    t0 = time.monotonic()
    img = pipe(PROMPT, negative_prompt=NEG, image=[depth, line], controlnet_conditioning_scale=scales,
               num_inference_steps=a.steps, width=size[0], height=size[1], generator=g).images[0]
    el = time.monotonic() - t0
    a.out.parent.mkdir(parents=True, exist_ok=True); img.save(a.out)
    depth.save(a.out.with_suffix(".cond-depth.png")); line.save(a.out.with_suffix(".cond-line.png"))
    a.out.with_suffix(".meta.json").write_text(json.dumps({"seed": a.seed, "rotate90": a.rotate90, "steps": a.steps, "size": size, "scales": scales, "device": dev, "seconds": round(el,1)}))
    print(f"出图 {a.out} {img.size} {el:.1f}s dev={dev}")

if __name__ == "__main__": main()
