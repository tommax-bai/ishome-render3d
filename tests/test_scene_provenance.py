"""场景包要能回答"这些数字是上游给的还是 mock 的"。

**mock 与真值分得开是这条服务的底线**：三维的上游今天大半不存在，产出上必须看得出
哪一块是补的（同报告线"draft 落点进正文、同页必须挂依据标注"那条口径）。
判据一律看**上游填包时写没写那一段**，不看值长什么样——值碰巧等于默认值不代表它是猜的。
"""

from __future__ import annotations

import json
from pathlib import Path

from render3d_worker.models import DesignPackage, HeightRules
from render3d_worker.scene_compile import compile_scene_package

FIXTURES = Path(__file__).parent / "fixtures"


def _package(name: str) -> DesignPackage:
    with (FIXTURES / f"{name}.json").open(encoding="utf-8") as f:
        return DesignPackage.model_validate(json.load(f))


def test_不写高度那一段就是吃常规住宅档位() -> None:
    scene = compile_scene_package(_package("design-package-minimal"))
    assert scene.heights_source == "mock-default"
    # 吃的确实是契约里那档常规住宅的净高，墙就起这么高
    assert scene.bounds_max_m[2] == HeightRules().ceiling_height_m


def test_上游写了高度就记成上游给的() -> None:
    scene = compile_scene_package(_package("design-package-full"))
    assert scene.heights_source == "upstream"


def test_值等于默认值也仍算上游给的() -> None:
    """**判据是写没写，不是值等不等**：上游明明白白报了 2.80，不该被记成猜的。"""
    raw = _package("design-package-minimal").model_dump(by_alias=True)
    raw["heights"] = HeightRules().model_dump(by_alias=True)
    scene = compile_scene_package(DesignPackage.model_validate(raw))
    assert scene.heights_source == "upstream"


def test_尺子锚不到外轮廓时场景包上看得见() -> None:
    """同一条口径管尺子：退回外接框不是错，静默才是错。"""
    raw = _package("design-package-minimal").model_dump(by_alias=True)
    raw["plan"]["outline"] = []
    scene = compile_scene_package(DesignPackage.model_validate(raw))
    assert scene.scale_anchor_source == "plan-box"
