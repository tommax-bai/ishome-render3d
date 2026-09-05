"""两个 activity 的全链测试：取键 → 调纯库 → 写桶 → 回键与自证数，用内存假桶跑到底。

桶是桩件：这一层要验的是"产物出来之后往哪走、走不通怎么说"；真桶验的是凭证与网络，
那件事由真跑留档。纯库**不桩**——吃的是 `tests/fixtures/` 里的拟真包，整户真编真渲，
所以这里也顺带证了"CLI 那条路与 activity 这条路出的是同一份东西"。

四类错误各有一条（`violations[].check`）：取键失败 / 契约校验失败 / 纯库失败 / 写桶失败。
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from render3d_worker.activities import (
    ACTIVITY_BASE_RENDER,
    ACTIVITY_REGISTRY,
    ACTIVITY_SCENE_COMPILE,
    BaseViewRenderer,
    SceneCompiler,
    activity_registry,
)
from render3d_worker.activity_models import (
    CHECK_BAD_INPUT,
    CHECK_BASE_RENDER_FAILED,
    CHECK_CONTRACT_INVALID,
    CHECK_OBJECT_STORE_GET,
    CHECK_OBJECT_STORE_PUT,
    CHECK_SCENE_COMPILE_FAILED,
)
from render3d_worker.base_render import render_base_views
from render3d_worker.models import DesignPackage, ScenePackage
from render3d_worker.object_store import ObjectStoreError, content_type_of
from render3d_worker.scene_compile import compile_scene_package

FIXTURE_DIR = Path(__file__).parent / "fixtures"
MINIMAL_FIXTURE = FIXTURE_DIR / "design-package-minimal.json"
FULL_FIXTURE = FIXTURE_DIR / "design-package-full.json"
MINIMAL_REVISION = "rev-fixture-92sqm-3b2l1b-minimal"
FULL_REVISION = "rev-fixture-92sqm-3b2l1b-full"

_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
PREFIX = f"uploads/{_SHA}"
"""前缀是上游的地盘，本仓不解释它；测试里照上传件那条键的形态给一个。"""

BIRD_CAMERA_ID = "cam-bird-overview"
LIVING_ROOM_CAMERA_ID = "cam-room-客厅"
"""full 包里 8 台机位挑两台：bird 一台、room 一台（客厅是最大的那间，README 实测两头都过关）。"""

WIDTH_PX = 320
HEIGHT_PX = 240
"""小画幅：这里验的是链路不是画质，整户 1024×768 一台机位要好几秒。"""

VIEW_FILES = ("geometry.png", "depth.png", "line.png", "mask.png", "mask-index.json")
VIEW_KEY_FIELDS = ("geometry_key", "depth_key", "line_key", "mask_key", "mask_index_key")


class InMemoryObjectStore:
    """内存假桶：满足 `ObjectStore` 协议，记下写了什么与头是什么，或按需当场失败。"""

    def __init__(
        self, *, get_fails_with: str | None = None, put_fails_with: str | None = None
    ) -> None:
        self.objects: dict[str, bytes] = {}
        self.content_types: dict[str, str] = {}
        self.get_count = 0
        self._get_fails_with = get_fails_with
        self._put_fails_with = put_fails_with

    @property
    def bucket_name(self) -> str:
        return "ishome-test"

    def get(self, object_key: str, what: str) -> bytes:
        self.get_count += 1
        if self._get_fails_with is not None:
            raise ObjectStoreError([self._get_fails_with])
        if object_key not in self.objects:
            raise ObjectStoreError([f"{what}不在私有桶 `{self.bucket_name}` 里（键 {object_key}）"])
        return self.objects[object_key]

    def put(self, object_key: str, data: bytes) -> str:
        if self._put_fails_with is not None:
            raise ObjectStoreError([self._put_fails_with])
        self.content_types[object_key] = content_type_of(data)
        self.objects[object_key] = data
        return object_key


def _design_package_key(revision_id: str, prefix: str = PREFIX) -> str:
    return f"{prefix}/render3d/{revision_id}/design-package.json"


def _scene_package_key(revision_id: str, prefix: str = PREFIX) -> str:
    return f"{prefix}/render3d/{revision_id}/scene-package.json"


def _view_key(revision_id: str, camera_id: str, file_name: str) -> str:
    return f"{PREFIX}/render3d/{revision_id}/{camera_id}/{file_name}"


def _seed(store: InMemoryObjectStore, revision_id: str, package_bytes: bytes) -> str:
    key = _design_package_key(revision_id)
    store.objects[key] = package_bytes
    return key


def _minimal_json() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(MINIMAL_FIXTURE.read_text(encoding="utf-8"))
    return payload


async def _compile(store: InMemoryObjectStore, key: str, revision_id: str) -> dict[str, Any]:
    return await SceneCompiler(store).compile_scene(
        {"designPackageKey": key, "revisionId": revision_id}
    )


async def _render(
    store: InMemoryObjectStore,
    scene_package_key: str,
    camera_ids: list[str] | None,
    *,
    width_px: int = WIDTH_PX,
    height_px: int = HEIGHT_PX,
) -> dict[str, Any]:
    return await BaseViewRenderer(store).render_base(
        {
            "scenePackageKey": scene_package_key,
            "cameraIds": camera_ids,
            "widthPx": width_px,
            "heightPx": height_px,
        }
    )


async def _compile_and_render(
    store: InMemoryObjectStore,
    fixture: Path,
    revision_id: str,
    camera_ids: list[str] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    key = _seed(store, revision_id, fixture.read_bytes())
    compiled = await _compile(store, key, revision_id)
    assert compiled["verdict"] == "ok", compiled
    rendered = await _render(store, compiled["scene_package_key"], camera_ids)
    return compiled, rendered


def _only_violation(receipt: dict[str, Any]) -> dict[str, str]:
    assert receipt["verdict"] == "failed", receipt
    violations: list[dict[str, str]] = receipt["violations"]
    assert len(violations) == 1, violations
    return violations[0]


# ---------------------------------------------------------------------------
# 正路径
# ---------------------------------------------------------------------------


async def test_scene_compile_写场景包并回键与自证数() -> None:
    store = InMemoryObjectStore()
    key = _seed(store, MINIMAL_REVISION, MINIMAL_FIXTURE.read_bytes())

    receipt = await _compile(store, key, MINIMAL_REVISION)

    assert receipt["verdict"] == "ok", receipt
    scene_key = _scene_package_key(MINIMAL_REVISION)
    assert receipt["scene_package_key"] == scene_key
    assert receipt["bucket"] == "ishome-test"
    assert receipt["design_package_key"] == key
    assert receipt["revision_id"] == MINIMAL_REVISION
    assert set(store.objects) == {key, scene_key}
    assert store.content_types[scene_key] == "application/json"

    scene = ScenePackage.model_validate_json(store.objects[scene_key])
    assert scene.revision_id == MINIMAL_REVISION
    assert receipt["mesh_count"] == len(scene.meshes) > 0
    assert receipt["triangle_count"] == scene.triangle_count > 0
    assert receipt["metre_per_unit"] == scene.metre_per_unit > 0
    assert receipt["floor_area_sqm"] == scene.floor_area_sqm > 0
    assert receipt["area_match_ratio"] == scene.area_match_ratio > 0
    assert receipt["wall_segment_count"] == scene.wall_segment_count
    assert receipt["degenerate_wall_count"] == scene.degenerate_wall_count
    # README：15 个洞（外墙 9 / 内墙 6）
    assert sum(receipt["opening_count_by_kind"].values()) == 15
    assert receipt["heights_source"] == "mock-default", "minimal 包整段不写 heights，吃的是默认"
    assert receipt["scale_anchor_source"] == "outline"
    assert receipt["camera_ids"] == [BIRD_CAMERA_ID]
    assert receipt["scene_package_size_bytes"] == len(store.objects[scene_key])
    assert receipt["elapsed_seconds"] >= 0.0


async def test_桶里的场景包与CLI写本地的逐字相同() -> None:
    """两条路同一份契约、同一段代码、同一种序列化——对账时不必先猜是不是序列化不同。"""
    store = InMemoryObjectStore()
    key = _seed(store, MINIMAL_REVISION, MINIMAL_FIXTURE.read_bytes())
    receipt = await _compile(store, key, MINIMAL_REVISION)

    package = DesignPackage.model_validate(_minimal_json())
    local = compile_scene_package(package).model_dump_json(by_alias=True, indent=2)
    assert store.objects[receipt["scene_package_key"]] == local.encode("utf-8")


async def test_全链_minimal包一台bird机位全渲() -> None:
    store = InMemoryObjectStore()
    compiled, rendered = await _compile_and_render(store, MINIMAL_FIXTURE, MINIMAL_REVISION, None)

    assert rendered["verdict"] == "ok", rendered
    assert rendered["scene_package_key"] == compiled["scene_package_key"]
    assert rendered["bucket"] == "ishome-test"
    assert rendered["revision_id"] == MINIMAL_REVISION
    assert rendered["elapsed_seconds"] >= 0.0
    assert len(rendered["renders"]) == 1
    view = rendered["renders"][0]
    assert view["camera_id"] == BIRD_CAMERA_ID
    assert (view["width_px"], view["height_px"]) == (WIDTH_PX, HEIGHT_PX)

    for field, file_name in zip(VIEW_KEY_FIELDS, VIEW_FILES, strict=True):
        expected = _view_key(MINIMAL_REVISION, BIRD_CAMERA_ID, file_name)
        assert view[field] == expected, field
        assert expected in store.objects, field
    for field in VIEW_KEY_FIELDS[:-1]:
        assert store.content_types[view[field]] == "image/png", field
    assert store.content_types[view["mask_index_key"]] == "application/json"

    mask_index = json.loads(store.objects[view["mask_index_key"]])
    assert isinstance(mask_index, list)
    assert len(mask_index) == view["mask_entry_count"] > 0
    assert {"index", "meshId", "semantic", "pixelCount"} <= set(mask_index[0])
    assert 0.0 < view["covered_pixel_ratio"] <= 1.0
    assert 0.0 < view["near_m"] < view["far_m"]
    assert view["elapsed_seconds"] >= 0.0

    # 第五路控制稿：合流后是纯库的必填产物，与其余四路一样必写、不许为 None
    sketch_key = view["sketch_key"]
    assert sketch_key == _view_key(MINIMAL_REVISION, BIRD_CAMERA_ID, "sketch.png")
    assert store.content_types[sketch_key] == "image/png"
    assert view["room_view"] is None or isinstance(view["room_view"], dict)


async def test_全链_full包只渲指定的两台机位() -> None:
    store = InMemoryObjectStore()
    requested = [BIRD_CAMERA_ID, LIVING_ROOM_CAMERA_ID]
    _, rendered = await _compile_and_render(store, FULL_FIXTURE, FULL_REVISION, requested)

    assert rendered["verdict"] == "ok", rendered
    assert [view["camera_id"] for view in rendered["renders"]] == requested
    living = rendered["renders"][1]
    assert living["depth_key"] == _view_key(FULL_REVISION, LIVING_ROOM_CAMERA_ID, "depth.png")
    assert living["depth_key"] in store.objects
    written_cameras = {
        key.split("/")[-2]
        for key in store.objects
        if key.startswith(f"{PREFIX}/render3d/{FULL_REVISION}/") and key.count("/") == 5
    }
    assert written_cameras == set(requested), "没点名的机位不该被渲、更不该被写"


async def test_渲两次键与字节相同() -> None:
    """确定性：两只空桶各跑一遍全链，键集合相同、每个键底下的字节相同、回执除耗时外相同。"""
    first_store, second_store = InMemoryObjectStore(), InMemoryObjectStore()
    first = await _compile_and_render(first_store, MINIMAL_FIXTURE, MINIMAL_REVISION, None)
    second = await _compile_and_render(second_store, MINIMAL_FIXTURE, MINIMAL_REVISION, None)

    assert set(first_store.objects) == set(second_store.objects)
    assert len(first_store.objects) >= 7, "输入包 + 场景包 + 至少五路"
    for key, data in first_store.objects.items():
        assert second_store.objects[key] == data, key

    def _without_elapsed(receipt: dict[str, Any]) -> dict[str, Any]:
        stripped = {k: v for k, v in receipt.items() if k != "elapsed_seconds"}
        if "renders" in stripped:
            stripped["renders"] = [
                {k: v for k, v in view.items() if k != "elapsed_seconds"}
                for view in stripped["renders"]
            ]
        return stripped

    assert _without_elapsed(first[0]) == _without_elapsed(second[0])
    assert _without_elapsed(first[1]) == _without_elapsed(second[1])


async def test_16位深度与遮罩逐字节保真() -> None:
    """深度与遮罩被下游当数据读：桶里的字节必须与纯库回的**逐字节相同**，且仍是 16 位灰度。"""
    store = InMemoryObjectStore()
    compiled, rendered = await _compile_and_render(store, MINIMAL_FIXTURE, MINIMAL_REVISION, None)
    view = rendered["renders"][0]

    scene = ScenePackage.model_validate_json(store.objects[compiled["scene_package_key"]])
    direct = render_base_views(scene, BIRD_CAMERA_ID, WIDTH_PX, HEIGHT_PX)
    assert store.objects[view["depth_key"]] == direct.depth_png
    assert store.objects[view["mask_key"]] == direct.mask_png
    assert store.objects[view["geometry_key"]] == direct.geometry_png
    assert store.objects[view["line_key"]] == direct.line_png

    depth = Image.open(io.BytesIO(store.objects[view["depth_key"]]))
    mask = Image.open(io.BytesIO(store.objects[view["mask_key"]]))
    assert depth.mode == "I;16", "深度不再是 16 位：被转码了"
    assert mask.mode == "I;16", "遮罩不再是 16 位：被转码了"
    assert int(np.asarray(depth).max()) > 255, "16 位深度的值域该超出 8 位"
    assert (view["near_m"], view["far_m"]) == (direct.near_m, direct.far_m)


def test_registry_绑定到装好桶的实现件上() -> None:
    store = InMemoryObjectStore()
    compiler, renderer = SceneCompiler(store), BaseViewRenderer(store)
    bound = activity_registry(compiler, renderer)
    assert set(bound) == set(ACTIVITY_REGISTRY) == {ACTIVITY_SCENE_COMPILE, ACTIVITY_BASE_RENDER}
    assert bound[ACTIVITY_SCENE_COMPILE].__func__ is ACTIVITY_REGISTRY[ACTIVITY_SCENE_COMPILE]  # type: ignore[attr-defined]
    assert bound[ACTIVITY_BASE_RENDER].__func__ is ACTIVITY_REGISTRY[ACTIVITY_BASE_RENDER]  # type: ignore[attr-defined]
    assert bound[ACTIVITY_SCENE_COMPILE].__self__ is compiler  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 请求本身不合：bad-input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "request_",
    [
        {"designPackageKey": _design_package_key(MINIMAL_REVISION)},
        {
            "designPackageKey": _design_package_key(MINIMAL_REVISION),
            "revisionId": MINIMAL_REVISION,
            "tier": "final",
        },
        {"design_package_key_typo": "x", "revisionId": MINIMAL_REVISION},
    ],
)
async def test_scene_compile_请求不合就是bad_input(request_: dict[str, Any]) -> None:
    store = InMemoryObjectStore()
    receipt = await SceneCompiler(store).compile_scene(request_)
    assert _only_violation(receipt)["check"] == CHECK_BAD_INPUT
    assert store.get_count == 0, "入参都没解开就不该去碰桶"


@pytest.mark.parametrize(
    "overrides",
    [
        {"widthPx": 0},
        {"heightPx": -1},
        {"cameraIds": []},
        {"cameraIds": [BIRD_CAMERA_ID, BIRD_CAMERA_ID]},
        {"renderTier": "preview"},
        {"widthPx": None},
    ],
)
async def test_base_render_请求不合就是bad_input(overrides: dict[str, Any]) -> None:
    store = InMemoryObjectStore()
    request: dict[str, Any] = {
        "scenePackageKey": _scene_package_key(MINIMAL_REVISION),
        "cameraIds": None,
        "widthPx": WIDTH_PX,
        "heightPx": HEIGHT_PX,
    }
    request.update(overrides)
    receipt = await BaseViewRenderer(store).render_base(request)
    assert _only_violation(receipt)["check"] == CHECK_BAD_INPUT
    assert store.get_count == 0


# ---------------------------------------------------------------------------
# 契约校验失败：contract-invalid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_key",
    [
        "design-package.json",
        f"render3d/{MINIMAL_REVISION}/design-package.json",
        _scene_package_key(MINIMAL_REVISION),
        f"{PREFIX}/render3d/{MINIMAL_REVISION}/design-package.json/",
    ],
)
async def test_输入包键形态不合(bad_key: str) -> None:
    store = InMemoryObjectStore()
    receipt = await _compile(store, bad_key, MINIMAL_REVISION)
    violation = _only_violation(receipt)
    assert violation["check"] == CHECK_CONTRACT_INVALID
    assert "不合约定形态" in violation["detail"]
    assert store.get_count == 0, "键不对就不该发请求"


async def test_请求的修订号与键里的对不上() -> None:
    store = InMemoryObjectStore()
    key = _seed(store, MINIMAL_REVISION, MINIMAL_FIXTURE.read_bytes())
    receipt = await _compile(store, key, "rev-别的")
    violation = _only_violation(receipt)
    assert violation["check"] == CHECK_CONTRACT_INVALID
    assert "rev-别的" in violation["detail"] and MINIMAL_REVISION in violation["detail"]
    assert store.get_count == 0


async def test_包里写的修订号与请求的对不上() -> None:
    payload = _minimal_json()
    payload["revisionId"] = "rev-包里写的另一个"
    store = InMemoryObjectStore()
    key = _seed(store, MINIMAL_REVISION, json.dumps(payload).encode("utf-8"))
    receipt = await _compile(store, key, MINIMAL_REVISION)
    violation = _only_violation(receipt)
    assert violation["check"] == CHECK_CONTRACT_INVALID
    assert "rev-包里写的另一个" in violation["detail"]
    assert _scene_package_key(MINIMAL_REVISION) not in store.objects


def _package_with_extra_field() -> bytes:
    payload = _minimal_json()
    payload["renderTier"] = "final"
    return json.dumps(payload).encode("utf-8")


def _package_without_plan() -> bytes:
    payload = _minimal_json()
    del payload["plan"]
    return json.dumps(payload).encode("utf-8")


@pytest.mark.parametrize(
    ("package_bytes", "reason"),
    [
        (b"not json at all", "输入包不合契约"),
        (b'"a json string"', "输入包不合契约"),
        (_package_without_plan(), "plan"),
        (_package_with_extra_field(), "renderTier"),
    ],
)
async def test_坏输入包响亮失败_契约那一类(package_bytes: bytes, reason: str) -> None:
    store = InMemoryObjectStore()
    key = _seed(store, MINIMAL_REVISION, package_bytes)
    receipt = await _compile(store, key, MINIMAL_REVISION)
    violation = _only_violation(receipt)
    assert violation["check"] == CHECK_CONTRACT_INVALID
    assert reason in violation["detail"]
    assert set(store.objects) == {key}, "契约没过就不该往桶里写任何东西"


async def test_裹在designPackage里的包也认() -> None:
    """同 CLI 的口径：上游把包裹在 `{"designPackage": {...}}` 里，直接喂那一份也认。"""
    store = InMemoryObjectStore()
    wrapped = json.dumps({"designPackage": _minimal_json()}).encode("utf-8")
    key = _seed(store, MINIMAL_REVISION, wrapped)
    receipt = await _compile(store, key, MINIMAL_REVISION)
    assert receipt["verdict"] == "ok", receipt


async def test_坏输入包响亮失败_纯库那一类() -> None:
    """几何过了契约但编不出来（一间房都没有）：纯库的话原文进 detail，分类是纯库失败。"""
    payload = _minimal_json()
    payload["plan"]["rooms"] = []
    store = InMemoryObjectStore()
    key = _seed(store, MINIMAL_REVISION, json.dumps(payload).encode("utf-8"))
    receipt = await _compile(store, key, MINIMAL_REVISION)
    violation = _only_violation(receipt)
    assert violation["check"] == CHECK_SCENE_COMPILE_FAILED
    assert "一间房都没有" in violation["detail"]
    assert _scene_package_key(MINIMAL_REVISION) not in store.objects


@pytest.mark.parametrize(
    "bad_key",
    [
        "scene-package.json",
        _design_package_key(MINIMAL_REVISION),
        f"{PREFIX}/render3d/{MINIMAL_REVISION}/cam/scene-package.json/",
    ],
)
async def test_场景包键形态不合(bad_key: str) -> None:
    store = InMemoryObjectStore()
    receipt = await _render(store, bad_key, None)
    violation = _only_violation(receipt)
    assert violation["check"] == CHECK_CONTRACT_INVALID
    assert store.get_count == 0


async def test_场景包不合契约或修订号与键对不上() -> None:
    store = InMemoryObjectStore()
    scene_key = _scene_package_key(MINIMAL_REVISION)

    store.objects[scene_key] = b'{"revisionId": "x", "unit": "ft"}'
    violation = _only_violation(await _render(store, scene_key, None))
    assert violation["check"] == CHECK_CONTRACT_INVALID
    assert "场景包不合契约" in violation["detail"]

    mismatched = ScenePackage(revision_id="rev-别的").model_dump_json(by_alias=True)
    store.objects[scene_key] = mismatched.encode("utf-8")
    violation = _only_violation(await _render(store, scene_key, None))
    assert violation["check"] == CHECK_CONTRACT_INVALID
    assert "rev-别的" in violation["detail"]


async def test_点名的机位不在场景包里() -> None:
    store = InMemoryObjectStore()
    _, rendered = await _compile_and_render(
        store, MINIMAL_FIXTURE, MINIMAL_REVISION, ["cam-不存在", BIRD_CAMERA_ID]
    )
    violation = _only_violation(rendered)
    assert violation["check"] == CHECK_CONTRACT_INVALID
    assert "cam-不存在" in violation["detail"] and BIRD_CAMERA_ID in violation["detail"]
    assert not any(key.endswith(".png") for key in store.objects), "一台都不该渲"


async def test_场景包里一台相机都没有_全渲就是渲不了() -> None:
    payload = _minimal_json()
    payload["cameras"] = []
    store = InMemoryObjectStore()
    key = _seed(store, MINIMAL_REVISION, json.dumps(payload).encode("utf-8"))
    compiled = await _compile(store, key, MINIMAL_REVISION)
    assert compiled["verdict"] == "ok" and compiled["camera_ids"] == []
    violation = _only_violation(await _render(store, compiled["scene_package_key"], None))
    assert violation["check"] == CHECK_CONTRACT_INVALID
    assert "一台相机都没有" in violation["detail"]


# ---------------------------------------------------------------------------
# 纯库失败逐机位：base-render-failed，不吞
# ---------------------------------------------------------------------------


async def test_一台机位取景失败原样成它的violation_其余照渲() -> None:
    """room 机位指的房间在场景里没有地板——纯库响亮失败；这一台成一条 violation，
    bird 那台照渲照写，整份回执按 failed 回报且 renders 里列着写进去的那台。"""
    payload = _minimal_json()
    bad_camera_id = "cam-room-没有这间房"
    payload["cameras"].append({"id": bad_camera_id, "kind": "room", "room": "没有这间房"})
    store = InMemoryObjectStore()
    key = _seed(store, MINIMAL_REVISION, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    compiled = await _compile(store, key, MINIMAL_REVISION)
    assert compiled["verdict"] == "ok", compiled

    rendered = await _render(store, compiled["scene_package_key"], None)

    assert rendered["verdict"] == "failed"
    violations = rendered["violations"]
    assert len(violations) == 1, violations
    assert violations[0]["check"] == CHECK_BASE_RENDER_FAILED
    assert bad_camera_id in violations[0]["detail"]
    assert "没有这间房" in violations[0]["detail"]
    assert [view["camera_id"] for view in rendered["renders"]] == [BIRD_CAMERA_ID]
    assert rendered["renders"][0]["depth_key"] in store.objects
    assert _view_key(MINIMAL_REVISION, bad_camera_id, "depth.png") not in store.objects
    assert rendered["scene_package_key"] == compiled["scene_package_key"]


# ---------------------------------------------------------------------------
# 桶失败：object-store-get-failed / object-store-put-failed
# ---------------------------------------------------------------------------


async def test_取不到输入包() -> None:
    store = InMemoryObjectStore()
    receipt = await _compile(store, _design_package_key(MINIMAL_REVISION), MINIMAL_REVISION)
    violation = _only_violation(receipt)
    assert violation["check"] == CHECK_OBJECT_STORE_GET
    assert "不在私有桶" in violation["detail"]

    broken = InMemoryObjectStore(get_fails_with="桶连不上")
    violation = _only_violation(
        await _compile(broken, _design_package_key(MINIMAL_REVISION), MINIMAL_REVISION)
    )
    assert violation == {"check": CHECK_OBJECT_STORE_GET, "detail": "桶连不上"}


async def test_取不到场景包() -> None:
    store = InMemoryObjectStore()
    violation = _only_violation(await _render(store, _scene_package_key(MINIMAL_REVISION), None))
    assert violation["check"] == CHECK_OBJECT_STORE_GET
    assert "场景包不在私有桶" in violation["detail"]


async def test_场景包写不进去不当成功() -> None:
    store = InMemoryObjectStore(put_fails_with="桶满了")
    key = _seed(store, MINIMAL_REVISION, MINIMAL_FIXTURE.read_bytes())
    violation = _only_violation(await _compile(store, key, MINIMAL_REVISION))
    assert violation == {"check": CHECK_OBJECT_STORE_PUT, "detail": "桶满了"}
    assert set(store.objects) == {key}


async def test_底渲写不进去当场停_已写的照样列出来() -> None:
    good = InMemoryObjectStore()
    compiled, _ = await _compile_and_render(good, MINIMAL_FIXTURE, MINIMAL_REVISION, None)

    broken = InMemoryObjectStore(put_fails_with="桶满了")
    broken.objects[compiled["scene_package_key"]] = good.objects[compiled["scene_package_key"]]
    rendered = await _render(broken, compiled["scene_package_key"], None)

    assert rendered["verdict"] == "failed"
    assert rendered["violations"] == [
        {"check": CHECK_OBJECT_STORE_PUT, "detail": f"机位 {BIRD_CAMERA_ID}：桶满了"}
    ]
    assert rendered["renders"] == []
    assert set(broken.objects) == {compiled["scene_package_key"]}
