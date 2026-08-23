"""Temporal activities：所有 IO 与重计算收口在此。

注册名唯一真源：ishome-contracts `activities/registry.md`，**只增不改**——改注册名
会破坏历史 workflow 重放，等同于改线上协议；新增走 contracts 仓 PR 评审。
命名规则（规范 §2.4）：注册名 = kebab-case 显式声明；函数名 = 同词 snake_case
动词前置。
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from temporalio import activity

from render3d_worker.models import BaseRenderRequest

ActivityResult = dict[str, Any]


@activity.defn(name="scene-compile")
async def compile_scene(deep_revision_id: str) -> ActivityResult:
    """DeepDesign → Scene Graph → 场景包编译（交互引擎专用）。"""
    raise NotImplementedError


@activity.defn(name="base-render")
async def render_base(request: BaseRenderRequest) -> ActivityResult:
    """三维底渲：几何/深度/线稿/遮罩输出（交互引擎专用）。"""
    raise NotImplementedError


ACTIVITY_REGISTRY: dict[str, Callable[..., Coroutine[Any, Any, ActivityResult]]] = {
    "scene-compile": compile_scene,
    "base-render": render_base,
}
"""注册名 → 实现。键与 contracts 注册表逐字一致（tests/test_activity_registry.py 断言）。"""
