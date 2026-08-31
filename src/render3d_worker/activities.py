"""Temporal activities：所有 IO 与重计算收口在此。

注册名唯一真源：ishome-contracts `activities/registry.md`，**只增不改**——改注册名
会破坏历史 workflow 重放，等同于改线上协议；新增走 contracts 仓 PR 评审。
命名规则（规范 §2.4）：注册名 = kebab-case 显式声明；函数名 = 同词 snake_case
动词前置。

**两个 activity 今天都是存根，纯库那条路已经能跑**（`cli.py`：本地一份 JSON 进、
场景包与四路图出）。存根不是"还没写"，是**接线时点写死**：

- 缺的第一件是落桶。本仓还没有对象存储那一层（reportrender 的 `book_store` 是可照抄的
  形态：只写不签，签名是"给谁看、看多久"，属业务侧）；
- 缺的第二件是键的登记。`design_package_key` / `scene_package_key` / 底渲四路的键模板
  **都还没进 contracts `registries/object_keys.md`**（那张表今天只有报告册与用户上传件
  两行），产物类型也还没进 `registries/artifacts.md`。

**接通那一批要做的三件**：①加只依赖 oss2 的 store 模块；②contracts 两张表各增行；
③把下面两个函数体换成"取键 → 调纯库 → 写桶 → 返回键与自证数"。在那之前
`raise NotImplementedError` 是诚实的形态——半接的 activity 会让派发方以为它能用。
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from temporalio import activity

from render3d_worker.models import BaseRenderRequest, SceneCompileRequest

ActivityResult = dict[str, Any]


@activity.defn(name="scene-compile")
async def compile_scene(request: SceneCompileRequest) -> ActivityResult:
    """DeepDesign → Scene Graph → 场景包编译（交互引擎专用）。

    接通后：按 `design_package_key` 取输入包 → `scene_compile.compile_scene_package`
    → 场景包写桶 → 返回 `{"scenePackageKey": ..., 自证数}`。纯库那半已经能跑。
    """
    raise NotImplementedError


@activity.defn(name="base-render")
async def render_base(request: BaseRenderRequest) -> ActivityResult:
    """三维底渲：几何/深度/线稿/遮罩输出（交互引擎专用）。

    接通后：按 `scene_package_key` 取场景包 → `base_render.render_base_views`
    → 四路 PNG 与遮罩索引表写桶 → 返回四个键与自证数。纯库那半已经能跑。
    """
    raise NotImplementedError


ACTIVITY_REGISTRY: dict[str, Callable[..., Coroutine[Any, Any, ActivityResult]]] = {
    "scene-compile": compile_scene,
    "base-render": render_base,
}
"""注册名 → 实现。键与 contracts 注册表逐字一致（tests/test_activity_registry.py 断言）。"""
