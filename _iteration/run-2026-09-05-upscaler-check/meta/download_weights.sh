#!/bin/zsh
# 下载候选权重（走本机代理 7890）。每条记 http 码 / 字节数 / 耗时；结尾记 sha256。
set -u
cd "$(dirname "$0")/weights"
export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890
GH=https://github.com
URLS=(
  "$GH/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
  "$GH/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
  "$GH/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
  "$GH/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth"
)
for u in "${URLS[@]}"; do
  f=$(basename "$u")
  echo "== $f  <- $u" | tee -a download.log
  curl -sSL -m 600 -o "$f" -w "http %{http_code} size %{size_download} time %{time_total}s\n" "$u" 2>&1 | tee -a download.log
done
ls -la
shasum -a 256 *.pth | tee -a download.log
