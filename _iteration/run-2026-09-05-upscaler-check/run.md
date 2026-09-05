# 本机确定性放大器真跑：1168×880 → 2K 级，动不动墙（2026-09-05）

本文件写完不改。回答一件事：写实化出图封在 1168×880（`run-2026-09-05-wanx-superres-batch/run.md`：万相超分不放大、送大线稿也不放大），
产品要 2K——**万相之外有没有一个本机/自部署可跑、确定性、不改几何、单张 ≤ 约 60 s 的放大器**。

- 样本：`run-2026-09-05-control-sketch-realism/cam-bird-dollhouse-seed1.png` 与 `cam-room-主卧-seed1.png`（1168×880，万相 doodle 出图）；
  线稿 `run-2026-09-05-control-sketch/<机位>/line.png`（1280×960）。
- 尺子：`run-2026-09-01-condition-matrix/保真度量.py` 的 `fidelity_score`（结果先 LANCZOS 缩到线稿尺寸再量，分数只能横向比）。
- 逐像素比：放大图 LANCZOS 缩回 1168×880 与原图比，RGB 三通道取最大差，报"均值差 / 差>10 占比 / 差>30 占比 / 完全相同占比"——口径照抄
  9-05 万相超分记录，那份的参照值是：万相超分 ×2 对 run1 均值差 2.650、>10 占 3.35 %、>30 占 0.50 %，判为"不动墙"。
- 确定性判据：同一进程内跑两次，输出数组 sha256 逐字节相同（PNG 文件 sha256 也相同）。
- 脚本：`放大.py`（跑候选、记耗时与哈希）、`分析.py`（全部量化与拼图）、`sdx4探针.py`（SD x4 upscaler 探针）；
  `meta/` 里是每个候选每张图的 `.meta.json`（模型信息、权重 sha256、每次耗时、MPS 占用）、安装/下载/运行日志、跑批脚本。
- 全部数字在 `分析.json`。本记录不调任何付费接口，费用 0。

## 环境

| 项 | 值 |
|---|---|
| 机器 | Apple M1 Pro，16 GB 统一内存，macOS 15.7.3；跑的时候本机还有别的常驻进程（ishome-dev 栈），空闲内存很少（vm_stat free ≈ 60 MB） |
| Python / torch | 3.12.14 / **2.13.0**（MPS 可用；神经网络候选全部 fp32 + `torch.inference_mode()`，不开 half） |
| 加载权重 | **spandrel 0.4.2**（`realesrgan` 包依赖 basicsr，没装、没试；spandrel 直接读官方 .pth，四个权重全部一次读成，`meta/install.log`） |
| 其他 | numpy 2.5.2、pillow 12.3.0；SD 探针用 scratchpad `ctrl/.venv`：diffusers 0.40.0、transformers 5.16.1 |
| 权重来源 | GitHub releases 经本机代理 7890 下载，四个文件 4–10 s 各一次成功，sha256 记在 `meta/weights-download.log`：<br>`RealESRGAN_x2plus.pth`（xinntao/Real-ESRGAN v0.2.1，64 MB）· `RealESRGAN_x4plus.pth`（v0.1.0，64 MB）· `realesr-general-x4v3.pth`（v0.2.5.0，4.7 MB）· `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth`（JingyunLiang/SwinIR v0.0，64 MB） |
| SD x4 upscaler | `stabilityai/stable-diffusion-x4-upscaler` fp16 safetensors（unet 903 MB + text_encoder 649 MB + vae 106 MB）经代理下到 HF 缓存；`hf download` 第一次在 unet 402 MB 处停住 11 min 没动，杀掉重跑续传成功；`--include` 被 CLI 忽略（"Ignoring --include since filenames have been explicitly set"），配置 json 是第二次按文件名补下的 |

## 候选：测了五个，一个没测到，其余没测

| 候选 | 倍数 | 架构 / 参数量 | 测没测 |
|---|---|---|---|
| Lanczos ×2（PIL） | 2 | 插值 | 测了——下限对照 |
| Real-ESRGAN x2plus | 2 | ESRGAN(RRDBNet 23 块 64 特征，pixel-unshuffle) / 16.7 M | 测了，整图一次过 |
| SwinIR-M ×2 real-SR GAN | 2 | SwinIR / 11.7 M（spandrel 标 tiling=DISCOURAGED） | 整图**没跑完**（见下），改分块 256 测了 |
| Real-ESRGAN x4plus | 4 | ESRGAN(RRDBNet 23 块) / 16.7 M | 测了，另派生"×4 再 LANCZOS 缩到 ×2" |
| realesr-general-x4v3 | 4 | RealESRGAN Compact(SRVGGNet) / 1.2 M | 测了，另派生"×4 再 LANCZOS 缩到 ×2" |
| SD x4 upscaler（diffusers `StableDiffusionUpscalePipeline`） | 4 | 扩散，UNet 473 M | **没测到**：256×256 一块都跑不起来，原文见下 |
| HAT / DAT / 其他 transformer SR | — | — | 没测：SwinIR-M（11.7 M）整图已经把 16 GB 机器跑到换页，HAT-L 40 M 参数、窗口注意力更重，同一台机器上只会更慢，先不花这个时间 |
| 万相超分 | — | — | 不在本记录范围：9-05 已量过，不放大 |

## 数字

耗时是单张推理（不含加载权重，加载记在表后），run1 是进程里第一次（含 MPS 首次编译），run2 是热态。
`>10` / `>30` / `同` 是逐像素比的三个占比，`保真` 是对 `line.png` 的分数（原图：揭顶 **0.2109**、主卧 **0.2634**）。
`新增边 / 丢失边` 是本记录自加的补充量（不是既有尺子）：两图各缩到线稿尺寸做 Sobel+NMS 脊线，阈值 T 取**原图**脊线第 90 百分位、两图共用；
放大图 ≥T 的脊线里、原图 2 px 内连弱脊线（≥T/4）都没有的占比＝新增边比，反过来＝丢失边比。Lanczos 那行（≈1 %）就是这个量的本底。

### 揭顶 cam-bird-dollhouse（原图保真 0.2109）

| 候选 | 出图 | run1 | run2 | 确定性 | 均值差 | >10 | >30 | 同 | 保真 | 新增边 | 丢失边 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Lanczos ×2 | 2336×1760 | 0.59 s | 0.15 s | 同 | 0.149 | 0.03 % | 0.00 % | 87.95 % | 0.2115 | 1.0 % | 1.1 % |
| **Real-ESRGAN x2plus** | 2336×1760 | 13.36 s | **7.94 s** | 同 | 2.193 | 1.98 % | 0.17 % | 1.74 % | 0.2528 | 3.2 % | 2.1 % |
| SwinIR-M ×2（分块 256/16） | 2336×1760 | 58.83 s | 56.35 s | 同 | 1.157 | 1.26 % | 0.02 % | 8.76 % | 0.2480 | 2.9 % | 2.7 % |
| Real-ESRGAN x4plus | 4672×3520 | 43.17 s | **89.08 s** | 同 | 2.583 | 1.70 % | 0.07 % | 0.74 % | 0.2155 | 2.9 % | 4.1 % |
| x4plus → ×2(LANCZOS) | 2336×1760 | +0.16 s | +0.16 s | 同 | 2.573 | 1.72 % | 0.08 % | 0.74 % | 0.2256 | 3.1 % | 3.8 % |
| realesr-general-x4v3 | 4672×3520 | 3.64 s | **1.58 s** | 同 | 2.339 | 3.25 % | 0.61 % | 1.59 % | 0.2112 | 4.2 % | 4.8 % |
| x4v3 → ×2(LANCZOS) | 2336×1760 | +0.16 s | +0.16 s | 同 | 2.324 | 3.24 % | 0.60 % | 1.59 % | 0.2200 | 4.4 % | 4.5 % |

### 主卧 cam-room-主卧（原图保真 0.2634）

| 候选 | 出图 | run1 | run2 | 确定性 | 均值差 | >10 | >30 | 同 | 保真 | 新增边 | 丢失边 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Lanczos ×2 | 2336×1760 | 0.99 s | 0.50 s | 同 | 0.133 | 0.02 % | 0.00 % | 85.21 % | 0.2674 | 0.8 % | 0.8 % |
| **Real-ESRGAN x2plus** | 2336×1760 | 10.68 s | **8.18 s** | 同 | 1.797 | 1.39 % | 0.10 % | 0.46 % | 0.2628 | 3.4 % | 2.9 % |
| SwinIR-M ×2（分块 256/16） | 2336×1760 | 57.00 s | 57.03 s | 同 | 1.309 | 1.21 % | 0.06 % | 2.41 % | 0.2798 | 3.6 % | 2.3 % |
| Real-ESRGAN x4plus | 4672×3520 | 24.65 s | **67.80 s** | 同 | 1.559 | 1.64 % | 0.18 % | 1.07 % | 0.2734 | 2.9 % | 4.6 % |
| x4plus → ×2(LANCZOS) | 2336×1760 | +0.16 s | +0.16 s | 同 | 1.565 | 1.64 % | 0.19 % | 1.12 % | 0.2747 | 3.0 % | 4.4 % |
| realesr-general-x4v3 | 4672×3520 | 1.53 s | **1.29 s** | 同 | 1.344 | 1.97 % | 0.29 % | 4.57 % | 0.2606 | 2.7 % | 5.2 % |
| x4v3 → ×2(LANCZOS) | 2336×1760 | +0.17 s | +0.17 s | 同 | 1.304 | 1.97 % | 0.29 % | 5.40 % | 0.2657 | 2.7 % | 5.0 % |

加载与占用（`meta/*.meta.json`）：x2plus 加载 33.2 s / 17.9 s（首次含磁盘冷读，都是 `torch.load` 一个 64 MB pickle），MPS 驱动占用 4.1 GB；
SwinIR 分块加载 1.6 s、占用 3.2 GB、`time -l` 峰值内存 3.9 GB；x4plus 加载 2.6 s、**占用 14.3 GB**；x4v3 加载 5.7 s / 2.8 s、占用 1.0 GB。

CPU 对照（同一台机、`--device cpu`、只跑一次，看没有 GPU 的部署机大概什么量级）：见文末"CPU 对照"一节。

## 两件跑不动 / 没跑到的，原文

**SwinIR 整图**：`放大.py --model swinir-m-x2-realsr`（不分块）在揭顶图上从 18:56:34 跑到 19:12:48 没出第一次结果，手动杀掉。
`time -l`：real 1007.9 s、sys 167.5 s、max RSS 1.05 GB；杀之前 `ps` 看 CPU 13 %、RSS 已缩到 7 MB——页面被换出去了，是内存换页不是在算。
spandrel 对这个架构标的就是 `ModelTiling.DISCOURAGED`（分块会在块缝上有差别），整图在 16 GB 机器上又跑不动，只能分块跑：
tile 256 + pad 16，两张图都是 **57 s 上下**，刚好卡在 60 s 线上，且这个数是在同一台机器同时跑着 `分析.py`（CPU）时量的，干净环境只会略快、不会到 30 s 量级。

**SD x4 upscaler**：管线 fp16 能装进 MPS（加载 ~60 s，`time -l` 峰值 4.9 GB），但一块 256×256 输入、20 步（diffusers 文档示例步数；管线默认 75 步）在第一步 UNet 注意力处报错：

```
RuntimeError: Invalid buffer size: 16.00 GiB
  File "diffusers/models/attention_processor.py", line 607, in forward
  File "diffusers/models/attention_processor.py", line 2767, in __call__
```

加 `pipe.enable_attention_slicing("max")` 重跑（`meta/sdx4-probe-attempt2-attention-slicing.log`）同一处同一报错。两次尝试都在 `sdx4探针.py`，原文在 `meta/sdx4-probe-attempt1.log` / `-attempt2-…log`。
到此没再试（128 块、CPU fp32 都没跑）——原因：这条管线的潜空间分辨率＝输入分辨率（VAE 缩放因子 4、出 ×4），整图 1168×880 无论如何要切成 ≥ 20 块、每块 20 步扩散，
就算一块能跑起来也不可能落在 60 s 内；而且它是扩散生成——同 seed 的确定性、改不改内容都没量到，**这条记录对它只能说"本机 16 GB 上一块都跑不起来"，不能说别的**。
出的也是 ×4（4672×3520），要 2K 还得再缩。

## 叠线拼图怎么读

- `montage-叠线-<机位>.png`：原图 + 七个候选（含两个派生）各自叠线稿红线（线稿 BILINEAR 缩放到结果尺寸，线像素 55 % 红混入，不是实心盖住——红线下面模型自己的边看得见），每格缩到高 480。
- `montage-局部-<机位>.png`：固定区域 1:1 局部（揭顶 (400,430)+320×240 → 卫生间+走廊门；主卧 (100,250)+320×240 → 黑门+竖窗+墙角），
  第一格是原图 NEAREST ×2 做对照；×4 候选那格是 ×4 结果按 ×2 缩着显示。
- `montage-局部叠线-<机位>.png`：同一区域叠线。
- 全分辨率成品只留了推荐候选的：`cam-bird-dollhouse-realesrgan-x2plus.png`、`cam-room-主卧-realesrgan-x2plus.png`（run1，sha256 在 `meta/`；其余候选的成品在 scratchpad，确定性已证、按 `放大.py` 可复现出逐字节同一张）。

看到的（两张图都一样）：

1. **墙、门、窗一条都没挪**：八格里红线落点相同，红线下面每个候选自己的边都压在红线上；数字上新增边/丢失边 1–5 %，Lanczos 本底 1 %，
   逐像素差 >10 级的像素 1.2–3.3 %、>30 级 ≤ 0.6 %，全部低于万相超分那次（3.35 % / 0.50 %）——那次已判为"不动墙"。
2. 保真度：x2plus 揭顶 +0.042、主卧 −0.001；SwinIR +0.037 / +0.016；x4plus→×2 +0.015 / +0.011；x4v3→×2 +0.009 / +0.002。
   同万相超分的解释一样：锐化了边、尺子多命中几条，不是几何变了（几何变了逐像素差和新增边会一起跳）。
3. 差别在纹理，不在几何：
   - x2plus / x4plus 把主卧竖窗外面的树影抹成偏白的平面（局部图第 3、5 格）——窗外内容被"净化"了，室内几何不受影响；
   - x4v3 把揭顶地板木纹磨平成蜡面（局部图第 7 格），边也偏软，"丢失边"两张都是最高（4.8 % / 5.2 %）；
   - SwinIR 最接近原图（均值差最低 1.16 / 1.31，"完全相同"像素最多），边细，但 57 s；
   - Lanczos 就是原图放大，没有新细节，作为下限。
4. 分块 SwinIR 的块缝（tile 256、pad 16）在拼图上没看出线，逐像素比也没有块状差异；没有做专门的块缝量化。

## 判断（不是裁决，我提的）

**可作为 imagegen 写实化之后的确定性放大步的：Real-ESRGAN x2plus。**
本机 M1 Pro 热态 8 s（冷 11–13 s）、2336×1760、两次逐字节相同、逐像素差与保真度都在"不动墙"范围内、权重 64 MB、spandrel 一行读入、MPS 4 GB。
其余三个各差一条：SwinIR 质量最贴原图但 57 s 卡线、整图跑不动；x4plus 占 14 GB、第二次 68–89 s；x4v3 最快（1.3–1.6 s）但把木纹磨平、丢边最多——
如果部署机没 GPU、8 s 变几分钟，x4v3 是退路（见 CPU 对照）。SD x4 本机跑不起来，且它是生成不是放大，就算跑得动也不该放在"确定性放大"这一格。

**落点**（按《方案/架构对齐》"逻辑异质→物理隔离"两判据分列判，我提的，不是裁决）：

- 放大是**确定性图像处理**：无 seed、无 prompt、同输入同输出、不联网、不知道用户是谁；写实化是**生成**：走万相、带 seed、同 prompt 不同 seed 几何散布。
  两者逻辑异质，不应合进同一个调用里——放大不能藏在 `realism` 那一步的尾巴上，它得是 Temporal 里**独立的一个 activity**（写实图进、2K 图出），
  这样挑图门禁（已判必须做）可以在放大之前对 1168×880 判、放大只对过门禁的图做，付费图和放大图分开落盘、分开重试。
- 代码归属：这是 torch 模型推理，imagegen 现在的写实化只是经 LiteLLM 网关调万相、进程里没有 torch；把 64 MB 权重 + torch 塞进 imagegen worker 会改变它的运行时形态（内存从几十 MB 到 4 GB、要 GPU）。
  按"是否起服务另过服务存在性判据"：放大器本身无状态、可以是一个纯函数库（`spandrel` 读权重 → 推理 → 返回数组），**先以库形式独立成仓**（放大是确定性的、可单元测试的——同输入 sha256 相同），
  由一个独立的放大 worker 进程跑那个 activity；是否成为常驻服务看部署机形态——有 GPU 的机器上权重驻留 8 s 一张，没 GPU 的机器要按 CPU 对照的数另算，那时才有"起不起服务"的问题。
- 万相已确认封在 1168×880，所以这一步不是"暂时"的，出图管线里 2K 只能从这儿来；放大器换型号（比如以后换 SwinIR 或别家）只换库不动 activity 契约。

## CPU 对照

（写在此处的数字来自 `meta/run_all.log` 末尾"CPU 对照"段，只跑一次、同一台 M1 Pro、`--device cpu`、fp32。）

| 候选 | 出图 | CPU 一次 | 对比 MPS 热态 | CPU 结果 vs MPS 结果（全分辨率逐像素） |
|---|---|---|---|---|
| Real-ESRGAN x2plus | 2336×1760 | **115.8 s** | 7.9 s | 均值差 0.0002、最大差 1 级、99.93 % 像素完全相同 |
| realesr-general-x4v3 | 4672×3520 | **10.0 s** | 1.6 s | 均值差 0.0000、最大差 1 级、99.99 % 像素完全相同 |

- 确定性是**同一设备内**的：MPS 两次逐字节同，CPU 与 MPS 之间 sha256 不同，差只在最后 1 级（浮点累加顺序），肉眼与尺子都量不出，但"逐字节相同"这条只能在固定后端上承诺——部署时后端要钉死。
- 没 GPU 的部署机上 x2plus 约 2 分钟一张（M1 Pro 的 CPU，x86 服务器另量），超出 60 s；x4v3 10 s 但纹理被磨平。要么部署机配 GPU，要么接受 x4v3 的质量，二选一是部署时的事，这里只给数。

## 文件

- `run.md`（本文件）· `分析.json`（全部数字）
- 脚本：`放大.py` `分析.py` `sdx4探针.py`；`meta/run_all.sh` `meta/download_weights.sh`（跑批与下载）
- 成品：`cam-bird-dollhouse-realesrgan-x2plus.png` `cam-room-主卧-realesrgan-x2plus.png`（2336×1760）
- 拼图：`montage-叠线-*.png` `montage-局部-*.png` `montage-局部叠线-*.png`（各两张机位）
- `meta/`：10 份 MPS `.meta.json` + 2 份 CPU 对照 `cpu-*.meta.json`、`install.log`、`weights-download.log`、`run_all.log`、`sdx4-probe-attempt1.log`、`sdx4-probe-attempt2-attention-slicing.log`
- 没进仓的：权重与 venv 在 scratchpad `upscale/`，其余候选成品在 scratchpad `upscale/out/`
