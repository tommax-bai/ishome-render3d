"""对象存储那一层的守门测试：键形态、Content-Type 按首部判、凭证缺一即败、真桶口的错误映射。

真桶用桩件（monkeypatch 掉 `oss2.Auth` / `oss2.Bucket`）：这里验的是"本层往 SDK 递了什么、
SDK 抛了什么本层怎么说"；凭证、网络、桶权限对不对由真跑留档，不是单测的题目。
"""

from __future__ import annotations

from typing import Any

import oss2
import pytest

from render3d_worker.object_store import (
    DESIGN_PACKAGE_KEY_TEMPLATE,
    RENDER_VIEW_KEY_TEMPLATE,
    SCENE_PACKAGE_KEY_TEMPLATE,
    KeyRoot,
    ObjectStoreError,
    OssObjectStore,
    OssSettings,
    check_key_segment,
    content_type_of,
    key_root_of_design_package_key,
    key_root_of_scene_package_key,
)

_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
_PREFIX = f"uploads/{_SHA}"
_REVISION = "rev-fixture-92sqm-3b2l1b-minimal"
_PNG_HEAD = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

_ENV_NAMES = (
    "ISHOME_OSS_ENDPOINT",
    "ISHOME_OSS_BUCKET_PRIVATE",
    "ISHOME_OSS_ACCESS_KEY_ID",
    "ISHOME_OSS_ACCESS_KEY_SECRET",
)


# ---------------------------------------------------------------------------
# 键
# ---------------------------------------------------------------------------


def test_输入包键解出键根并派生其余键() -> None:
    root = key_root_of_design_package_key(f"{_PREFIX}/render3d/{_REVISION}/design-package.json")
    assert root == KeyRoot(prefix=_PREFIX, revision_id=_REVISION)
    assert root.design_package_key == f"{_PREFIX}/render3d/{_REVISION}/design-package.json"
    assert root.scene_package_key == f"{_PREFIX}/render3d/{_REVISION}/scene-package.json"
    assert (
        root.view_key("cam-room-客厅", "depth.png")
        == f"{_PREFIX}/render3d/{_REVISION}/cam-room-客厅/depth.png"
    )


def test_场景包键解出的键根与输入包键的相同() -> None:
    from_design = key_root_of_design_package_key(
        f"{_PREFIX}/render3d/{_REVISION}/design-package.json"
    )
    from_scene = key_root_of_scene_package_key(f"{_PREFIX}/render3d/{_REVISION}/scene-package.json")
    assert from_design == from_scene


def test_键模板与派生逻辑说的是同一件事() -> None:
    """docstring 里那三条模板是给人读的形态说明——它们与代码派生出来的键不许分家。"""
    root = KeyRoot(prefix=_PREFIX, revision_id=_REVISION)
    assert root.design_package_key == DESIGN_PACKAGE_KEY_TEMPLATE.format(
        prefix=_PREFIX, revision_id=_REVISION
    )
    assert root.scene_package_key == SCENE_PACKAGE_KEY_TEMPLATE.format(
        prefix=_PREFIX, revision_id=_REVISION
    )
    assert root.view_key("cam-bird-overview", "mask-index.json") == RENDER_VIEW_KEY_TEMPLATE.format(
        prefix=_PREFIX,
        revision_id=_REVISION,
        camera_id="cam-bird-overview",
        file_name="mask-index.json",
    )


def test_前缀是上游的地盘_从后往前解() -> None:
    """前缀里碰巧也有一段叫 render3d，键也不许解错。"""
    root = key_root_of_design_package_key("a/render3d/b/render3d/rev-1/design-package.json")
    assert root == KeyRoot(prefix="a/render3d/b", revision_id="rev-1")


@pytest.mark.parametrize(
    "bad_key",
    [
        "design-package.json",
        "render3d/rev-1/design-package.json",
        f"{_PREFIX}/render3d/{_REVISION}/scene-package.json",
        f"{_PREFIX}/render3d/{_REVISION}/other.json",
        f"{_PREFIX}/render2d/{_REVISION}/design-package.json",
        f"{_PREFIX}/render3d//design-package.json",
        f"{_PREFIX}/render3d/../design-package.json",
        f"{_PREFIX}/render3d/{_REVISION}/design-package.json ",
        f"/{_PREFIX}/render3d/{_REVISION}/design-package.json",
    ],
)
def test_输入包键形态不合当场失败(bad_key: str) -> None:
    with pytest.raises(ObjectStoreError, match="不合约定形态"):
        key_root_of_design_package_key(bad_key)


@pytest.mark.parametrize(
    ("segment", "reason"),
    [
        ("", "空"),
        (".", "当不了"),
        ("..", "当不了"),
        ("cam/room", "劈开"),
        ("cam\\room", "劈开"),
        (" cam", "空白"),
        ("cam\n", "空白"),
        ("cam\x01room", "控制字符"),
    ],
)
def test_机位id当不了键段就当场失败(segment: str, reason: str) -> None:
    with pytest.raises(ObjectStoreError, match=reason):
        check_key_segment(segment, "机位 id")
    with pytest.raises(ObjectStoreError):
        KeyRoot(prefix=_PREFIX, revision_id=_REVISION).view_key(segment, "depth.png")


def test_机位id不限字符集_房间名照收() -> None:
    check_key_segment("cam-room-客厅", "机位 id")
    check_key_segment("Cam_Room.01", "机位 id")


# ---------------------------------------------------------------------------
# Content-Type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "content_type"),
    [
        (_PNG_HEAD, "image/png"),
        (b'{"revisionId": "rev-1"}', "application/json"),
        (b'  \n[{"index": 1}]', "application/json"),
    ],
)
def test_content_type按首部判(data: bytes, content_type: str) -> None:
    assert content_type_of(data) == content_type


@pytest.mark.parametrize("data", [b"", b"GIF89a", b"\xff\xd8\xff\xe0JFIF", b"revisionId: rev-1"])
def test_认不出的字节不给猜(data: bytes) -> None:
    with pytest.raises(ObjectStoreError, match="既不是 PNG 也不是 JSON"):
        content_type_of(data)


# ---------------------------------------------------------------------------
# 凭证
# ---------------------------------------------------------------------------


def test_凭证缺一即败_并点名缺哪个(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_NAMES:
        monkeypatch.setenv(name, "x")
    monkeypatch.setenv("ISHOME_OSS_ACCESS_KEY_SECRET", "   ")
    monkeypatch.delenv("ISHOME_OSS_BUCKET_PRIVATE")
    with pytest.raises(ObjectStoreError) as excinfo:
        OssSettings.from_env()
    message = str(excinfo.value)
    assert "ISHOME_OSS_BUCKET_PRIVATE" in message
    assert "ISHOME_OSS_ACCESS_KEY_SECRET" in message
    assert "ISHOME_OSS_ENDPOINT" not in message


def test_凭证齐了就装得上(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISHOME_OSS_ENDPOINT", " https://oss.test ")
    monkeypatch.setenv("ISHOME_OSS_BUCKET_PRIVATE", "ishome-private")
    monkeypatch.setenv("ISHOME_OSS_ACCESS_KEY_ID", "id")
    monkeypatch.setenv("ISHOME_OSS_ACCESS_KEY_SECRET", "secret")
    settings = OssSettings.from_env()
    assert settings == OssSettings("https://oss.test", "ishome-private", "id", "secret")


# ---------------------------------------------------------------------------
# 真桶口（SDK 桩件）
# ---------------------------------------------------------------------------


class _FakeOssObject:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class _FakeOssBucket:
    """桩件 `oss2.Bucket`：记下递进来的键、字节与头；没有的键抛 SDK 自己的 NoSuchKey。"""

    instances: list[_FakeOssBucket] = []

    def __init__(self, auth: Any, endpoint: str, bucket_name: str) -> None:
        self.endpoint = endpoint
        self.bucket_name = bucket_name
        self.objects: dict[str, bytes] = {}
        self.headers: dict[str, dict[str, str]] = {}
        self.fail_put_with: Exception | None = None
        _FakeOssBucket.instances.append(self)

    def put_object(self, key: str, data: bytes, headers: dict[str, str]) -> None:
        if self.fail_put_with is not None:
            raise self.fail_put_with
        self.objects[key] = data
        self.headers[key] = headers

    def get_object(self, key: str) -> _FakeOssObject:
        if key not in self.objects:
            raise oss2.exceptions.NoSuchKey(404, {}, b"", {})
        return _FakeOssObject(self.objects[key])


@pytest.fixture
def stubbed_store(monkeypatch: pytest.MonkeyPatch) -> tuple[OssObjectStore, _FakeOssBucket]:
    """装一只真形态的 `OssObjectStore`，底下的 SDK 换成桩件；连桩件一起交出来好断言递了什么。"""
    _FakeOssBucket.instances.clear()
    monkeypatch.setattr(oss2, "Auth", lambda key_id, secret: ("auth", key_id, secret))
    monkeypatch.setattr(oss2, "Bucket", _FakeOssBucket)
    store = OssObjectStore(OssSettings("https://oss.test", "ishome-private", "id", "secret"))
    return store, _FakeOssBucket.instances[-1]


def test_写桶带上按首部判的头_取回原字节(
    stubbed_store: tuple[OssObjectStore, _FakeOssBucket],
) -> None:
    store, bucket = stubbed_store
    assert store.bucket_name == "ishome-private"
    assert (bucket.endpoint, bucket.bucket_name) == ("https://oss.test", "ishome-private")

    depth_key = f"{_PREFIX}/render3d/{_REVISION}/cam/depth.png"
    index_key = f"{_PREFIX}/render3d/{_REVISION}/cam/mask-index.json"
    assert store.put(depth_key, _PNG_HEAD) == depth_key
    assert store.put(index_key, b"[]") == index_key
    assert bucket.headers[depth_key] == {"Content-Type": "image/png"}
    assert bucket.headers[index_key] == {"Content-Type": "application/json"}
    assert bucket.objects[depth_key] == _PNG_HEAD
    assert store.get(depth_key, "深度图") == _PNG_HEAD


def test_桶里没有这个键就说没有(stubbed_store: tuple[OssObjectStore, _FakeOssBucket]) -> None:
    store, _ = stubbed_store
    with pytest.raises(ObjectStoreError, match="输入包不在私有桶 `ishome-private` 里"):
        store.get(f"{_PREFIX}/render3d/{_REVISION}/design-package.json", "输入包")


def test_空对象不当成功(stubbed_store: tuple[OssObjectStore, _FakeOssBucket]) -> None:
    store, bucket = stubbed_store
    bucket.objects["k"] = b""
    with pytest.raises(ObjectStoreError, match="空对象"):
        store.get("k", "场景包")
    with pytest.raises(ObjectStoreError, match="字节是空的"):
        store.put("k2", b"")
    assert "k2" not in bucket.objects


def test_写失败上抛_不返回指向空气的键(
    stubbed_store: tuple[OssObjectStore, _FakeOssBucket],
) -> None:
    store, bucket = stubbed_store
    bucket.fail_put_with = oss2.exceptions.RequestError(OSError("connection reset"))
    with pytest.raises(ObjectStoreError, match="写不进私有桶"):
        store.put("k", _PNG_HEAD)
