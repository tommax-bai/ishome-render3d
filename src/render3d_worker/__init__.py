"""render3d_worker：三维管线 activity 执行进程（render3d-svc）。

V1.4 裁决（2026-08-23）：绘图能力物理拆分——本仓承接 scene-compile 与
base-render（三维场景编译与底渲：几何/深度/线稿/遮罩），独立部署 Temporal
worker，专属 task queue `render3d-activities`，无对外 RPC 端口、无数据库
schema、无状态（算完即焚，产物写 OSS + 注册 ArtifactRegistry）。伸缩轴：
GPU + 三维引擎重依赖。
"""
