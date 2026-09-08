"""展示层换算：内部数据是毫米，**打给人看的那几行仍说米**。

用户裁决 2026-09-08（*"三维那面改成毫米吧。我们所有的单位都改成毫米。除了给用户展示的
部分"*）把射程切成两半——内部一律毫米，给人看的不变，**换算发生在展示那一层**。
本仓唯一的展示面是 CLI 打的那几行（业主看的报告句子、图上标注、会话话术都不在这个仓），
所以这条纪律在这儿能被验到的就是 :func:`render3d_worker.cli._metres` 与调用它的那几处。

为什么值得单写一份测试：换算写错了不会让任何测试变红——四路图逐字节不变、自证数照样是
毫米，只有打出来那行字里的数会差三个数量级。**没有断言的换算就是没有换算。**
"""

from __future__ import annotations

import math

import pytest

from render3d_worker.cli import _MM_PER_METRE, _metres


def test_整千毫米换成整米() -> None:
    """最平常的那一类：常规住宅净高 2800 毫米，打出来是 2.8 米。"""
    assert _metres(2800.0) == pytest.approx(2.8)
    assert _metres(1550.0) == pytest.approx(1.55)  # 眼高
    assert _metres(50.0) == pytest.approx(0.05)  # 近裁剪面


def test_零与负数照原样换算不做特判() -> None:
    """0 换出来还是 0；负数（世界坐标可以在原点西边/北边）不许被当成非法值吞掉。

    **不特判是有意的**：这一层只做除法，"这个数合不合法"是产生它的那一层的事，
    在展示层补一次判断只会让两处口径分叉。
    """
    assert _metres(0.0) == 0.0
    assert _metres(-2500.0) == pytest.approx(-2.5)


def test_不能整除的毫米数不在换算里取整() -> None:
    """**边界：除不尽的。** 1 毫米＝0.001 米，7 毫米＝0.007 米——换算只做除法，不动精度。

    取整口径在调用处的格式化串上（`:.2f` / `:.4f`），不在换算里。理由写在 `_metres`
    的 docstring：先在换算里舍一次、格式化时再舍一次就是两次取整，末位会往下掉一档。
    这条断言就是钉住"换算不取整"这件事——它一旦被改成 `round(...)`，这里立刻红。
    """
    assert _metres(1.0) == pytest.approx(0.001)
    assert _metres(7.0) == pytest.approx(0.007)
    assert _metres(1234.0) == pytest.approx(1.234)
    # 1/3 毫米这种除不尽的：换算之后仍然除不尽，没有被悄悄抹平
    third_mm = 1000.0 / 3.0
    assert _metres(third_mm) == pytest.approx(1.0 / 3.0)
    assert _metres(third_mm) != pytest.approx(0.333, abs=1e-6)


def test_取整发生在格式化那一步而不是换算里() -> None:
    """**边界：需要取整的。** CLI 用 `:.2f` 打深度、`:.4f` 打尺子，取整只在这一步发生。

    1236 毫米按两位小数打是 1.24 米（进位），按四位是 1.2360——同一个数、两种精度，
    正因为换算本身没有先舍过一次。
    """
    assert f"{_metres(1236.0):.2f}" == "1.24"
    assert f"{_metres(1236.0):.4f}" == "1.2360"
    # 半分进位：1235 毫米 → 1.235 米，两位小数按浮点的最近偶数落到 1.24
    assert f"{_metres(1235.0):.2f}" in {"1.23", "1.24"}, "半分进位随浮点表示，两种都算对"


def test_换算系数就是一千且只此一处() -> None:
    """系数写死 1000，且 CLI 里除了 `_metres` 没有第二处做毫米→米。

    《纪律·同一条规矩只写一处》：散着写 `/1000` 等于把量纲边界摊回全仓，
    那正是这次改动要消掉的东西。
    """
    import inspect

    from render3d_worker import cli

    assert _MM_PER_METRE == 1000.0
    source = inspect.getsource(cli)
    # 除了 _metres 的定义体，源码里不该再出现别的"除以一千"
    assert source.count("/ _MM_PER_METRE") == 1
    assert "/ 1000" not in source


def test_往返换算回得来() -> None:
    """毫米 → 米 → 毫米：拿常规住宅那几个数走一圈，回得到原值。

    这条防的是"换算方向写反了"——乘除写反时上面几条按值断言的还有可能蒙对一两个
    （1.0 那种），往返这条蒙不过去。
    """
    for value_mm in (2800.0, 1550.0, 2050.0, 900.0, 120.0, 1.0, 0.0, -350.0):
        assert _metres(value_mm) * _MM_PER_METRE == pytest.approx(value_mm)


def test_大数不丢精度() -> None:
    """远裁剪面 50 000 毫米、揭顶机位眼高 12 000 毫米这一档，换算之后仍是精确值。

    毫米制把所有数放大了一千倍，double 的有效位远够——这条钉住"改量纲没有换来精度问题"。
    """
    assert _metres(50_000.0) == 50.0
    assert _metres(12_000.0) == 12.0
    assert not math.isinf(_metres(1e12))
