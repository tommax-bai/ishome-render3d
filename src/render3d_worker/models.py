"""render3d_worker activity 出入参模型（pydantic）。

跨 domain 纪律：worker 不 import 其他 domain 的内部模块，activity 入参出参
以本模块与（后续）contracts 生成 SDK 为准。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

RenderTier = Literal["preview", "final"]
"""渲染两档：失效传播默认只重算 preview，final 由用户显式请求或交付节点触发。"""


class BaseRenderRequest(BaseModel):
    """base-render 输入：三维底渲（几何/深度/线稿/遮罩输出）。"""

    scene_package_key: str
    camera_id: str
    render_tier: RenderTier = "preview"
