"""四路（几何/深度/线稿/遮罩）与加控制稿之前**逐字节相同**的锁。

基线＝加第五路之前的代码（2026-09-05，提交 ef693a7）渲出来的字节的 sha256：拟真包
``design-package-full.json`` 全部 8 台相机（都是上游显式给 yaw 的，取景规则的改动碰不到它们）
在 320×240 上，以及 test_base_render 那间现造的房里三台位姿不随取景规则变的相机。

四路一个字节都不许变：线稿是保真度尺子的输入，深度/遮罩是被当数据读的，几何是写实化的参考。
控制稿是**另出的一路**，不是改出来的。这条测试红了，先问"是谁动了四路"，不是改基线；
真要改判（比如几何路观感提档），改这儿的数并在提交信息里说清楚。

## 2026-09-08 换过一次基线：只换 `depth` 一路，只因量纲改毫米

用户裁决 2026-09-08 把内部数据的长度量纲从米改成毫米（*"我们所有的单位都改成毫米。
除了给用户展示的部分"*）。**几何/线稿/遮罩三路 11 台相机全部逐字节不变**——形状、边、
索引都不是浮点末位说了算的东西，量纲换了它们一个字节都没动，这也正是"这次只换了单位、
没换几何"的证据。

`depth` 那一路换了 5 台（本表 `cam-bird-overview` / `cam-room-厨房` / `cam-room-餐厅`，
下表 `bird-整户` / `room-客厅-广角`）。原因：深度存的是 16 位归一化值，归一化的分子分母
现在都是毫米制浮点，末位舍入跟米制时不同。**差多少量过**：改前改后逐像素比，不同的像素
最多 0.65%，且每一个的差都**正好是 1**（量程 65535，即 0.0015%）。这是量化末位的抖动，
不是几何变了——真变了形状，`line` 与 `mask` 不可能一个字节都不动。

所以这次换基线是**记录一次已知的量纲改动**，不是放宽这条锁：往后这四路再红，仍然先问
"是谁动了四路"。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_base_render import (
    BIRD_CAMERA_ID,
    HEIGHT_PX,
    ROOM_CAMERA_ID,
    ROOM_WIDE_CAMERA_ID,
    WIDTH_PX,
    _make_scene,
)

from render3d_worker.base_render import render_base_views
from render3d_worker.models import BaseRenderViews, DesignPackage, ScenePackage
from render3d_worker.scene_compile import compile_scene_package

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "design-package-full.json"
FIXTURE_WIDTH_PX = 320
FIXTURE_HEIGHT_PX = 240

FIXTURE_BASELINE_SHA256: dict[str, tuple[str, str, str, str]] = {
    "cam-bird-overview": (
        "07247b1a45c40cd229b32642d39988d354526c21ff82f8f5f0a0e8c94e381084",
        "fb07ea3bb6e6b63ace58df10dc71dd9a09c14138232a3f47769a9a6a303cb072",
        "43d5713321bb91d0e1276fd46b69098b2ddbbd0bf727b01a070e5abc26b45af9",
        "572acf2b50032eeee01f2283abf7c910b320f53352ae0a9d034a7585b6e869bd",
    ),
    "cam-room-客厅": (
        "22dff860d58d31a07964e2a5a9be6f82e551a723f123bb71152c59737ff6e234",
        "96601b2830f829b8527630eacd3dc5e5ee051a6f60a0010471a659d3a88736b7",
        "a456bb50d87a2e6c03a358bf0e07f9e844283a15d707771784a7ac02f13e5c9d",
        "b027fd0159f2be84214ac1abe7e3e8428bf5499715fe1952876c61c22ae34bad",
    ),
    "cam-room-次卧": (
        "3dfb7e77b3051bf92c5688d6438d3190b5c756df0cb3c21cbd8634bff4cceab6",
        "ff1c83d6c4aa07b42afb5f5e3ecfcfa2e6acc6834d23b1a87c1f3461fa546049",
        "64e92eca78172c2eb44940c78f68245513257c572a43433ce5724ab613b977a3",
        "f7f1aebfe4bb0754b6a738b5d2ce521fc4b04a89f473af061ef027fd165f40f4",
    ),
    "cam-room-餐厅": (
        "8cc36b04fc671f23acc7aa34662b46961aec40c9740e7a218207b3eb67fcd75b",
        "6d971bb33a7d61c49c677057a0800177dd87e7e90ca629d41457e4b349d132b7",
        "ab67323144bbbed352ec73b02add201b09b9a9c905cf3a410ca64e1914a804a2",
        "caa1f4415df1e3f6090e8a727d6b403954f0383b7712a72763b595ebcc445206",
    ),
    "cam-room-厨房": (
        "d57334dbea4920b61d2b0d15009e4cb81a94c3ae09ccf0b2a45e9c8b765ed52f",
        "8a909c2049f99b621cc4a9e3d3d3829c7fd0c6370f409562034ac0d0e9ec8186",
        "ba15dd4de13511816ba59258cd19b563eb029addc9b1ef1a216fd20571b51400",
        "721758c4897b91d9e3d075b4baffc25ccb56f9d469a251f121985c2c28ea3925",
    ),
    "cam-room-卫生间": (
        "5d1ec9f01eee24f1ecad6290a3d9fe9f79661aed99ef8b13bdc24a601d9e5bfc",
        "5756c01c4edb4cc1ac32e9436234bd89c4ccc7b2dd8440dcbf419bad67408fb0",
        "36d58a4a3e00ab0c6cdf1748d47c0a129fbeb6a904e6938e348ffe9747964918",
        "f33f0f5ff530f760a1de9572f186b5836aa3bc372a57a7f256da2fe498774215",
    ),
    "cam-room-玄关": (
        "b3266bd23e8d068e7f51c430744d819e02abac1673d924342b89ad61bf9b6e8e",
        "85e8fe3fbb17d802acde741b0b42d467f52260981b3a4401c58c7533df3f5c03",
        "3e9ffe6f02a00cfbd5969a3f1a5dc9e7df4374c23c8db20c14602b2f7ec11847",
        "e790db6a8edf11b9a852ad1a742ab7164d790f5d8e22cef1d9d972886d3e7239",
    ),
    "cam-room-阳台": (
        "67ecb3b832d7cbbb4833289b593af5b399f172ed63a93f6885635f2c8bedf738",
        "c1caf1ac2041ffe5d20234a4e8d2427fb165e9caedd71e80830e210195c563e1",
        "c4db3f34f014afa12e0340875acfd3d4c4fc9072a529c5942614ebc3c263ae1b",
        "5ac15152c3d7c3d49b3ad817e3a687b9dbfb0868007c06d0fda3554c65e82393",
    ),
}
"""拟真包 8 台相机在 320×240 上四路 PNG 的 sha256，次序＝(几何, 深度, 线稿, 遮罩)。"""

TEST_SCENE_BASELINE_SHA256: dict[str, tuple[str, str, str, str]] = {
    BIRD_CAMERA_ID: (
        "4ca06e3b6d2255aac4d209b992bea0db3590450cc36ccabbf343ebc889291d0a",
        "68a54d872d5c7205ed14af3fa22b87fbf7837bf24f9b13f568b5975990592954",
        "aeab2dd41ed5492188e020321b34a2ceefd059f1fe31f07c60e2ff5c59d18a9e",
        "34cf0fe76a22647e3b023f893b887cb53cafe23468e22130dfc7be33715c7841",
    ),
    ROOM_CAMERA_ID: (
        "46db270467f561005c5e65c1e2afa040d7575e609f7c516cc659d2028e8de440",
        "82d0e569c5a7e1ca5c6be16ca1c23a04c74299d2b75aadb64996d04fc91080ef",
        "8a4bd10dc2047bc14dca1a64bc903e5c9609d064929370be712853782d0c0498",
        "6ecce7ae29cf30ae3d58da51c527376767517e1778f6e56cf5e9922b65c752d3",
    ),
    ROOM_WIDE_CAMERA_ID: (
        "f0b77fedfcf0d94022c2eb8c180c6697ac89859edf1bd8e2ed3ecf2de1b37166",
        "9654781c4e53b80c362684d031ff899c00b0442239b4256c27757097ea9933ba",
        "420aaa9fcbb5da0698090b46506d8d33a8167207faebd51e10a2e74221b6fc67",
        "3b62a6ea60eff08f721861123cf65972dfade3d90f2132531e6222baa6a5ef52",
    ),
}
"""test_base_render 那间房三台相机在 320×240 上四路 PNG 的 sha256，次序同上。"""


def _four_view_digests(views: BaseRenderViews) -> tuple[str, str, str, str]:
    return (
        hashlib.sha256(views.geometry_png).hexdigest(),
        hashlib.sha256(views.depth_png).hexdigest(),
        hashlib.sha256(views.line_png).hexdigest(),
        hashlib.sha256(views.mask_png).hexdigest(),
    )


@pytest.fixture(scope="module")
def fixture_scene() -> ScenePackage:
    package = DesignPackage.model_validate(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))
    return compile_scene_package(package)


@pytest.mark.parametrize("camera_id", sorted(FIXTURE_BASELINE_SHA256))
def test_拟真包四路与加控制稿之前逐字节相同(fixture_scene: ScenePackage, camera_id: str) -> None:
    views = render_base_views(fixture_scene, camera_id, FIXTURE_WIDTH_PX, FIXTURE_HEIGHT_PX)
    assert _four_view_digests(views) == FIXTURE_BASELINE_SHA256[camera_id], (
        f"相机 {camera_id} 的四路里有一路变了——控制稿是另出的一路，四路一个字节都不许动"
    )
    assert len(views.sketch_png) > 0


@pytest.mark.parametrize("camera_id", sorted(TEST_SCENE_BASELINE_SHA256))
def test_现造那间房四路与加控制稿之前逐字节相同(camera_id: str) -> None:
    views = render_base_views(_make_scene(), camera_id, WIDTH_PX, HEIGHT_PX)
    assert _four_view_digests(views) == TEST_SCENE_BASELINE_SHA256[camera_id]
