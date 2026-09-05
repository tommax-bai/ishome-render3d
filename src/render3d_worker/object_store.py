"""出站边缘：**私有对象存储**（阿里云 OSS 私有桶，用户裁决 2026-08-30 晚）。

本仓**既读又写**：输入包（`design-package.json`）由上游写进私有桶，本仓按键取下来；场景包与
底渲各路图写回**同一个前缀**下。场景包几百 KB、每台机位几路图几 MB——走编排 payload 是不行的
（Temporal 单条 payload 有量级限制，且历史会一直背着它），所以图走桶、键走编排。

**只写不签。** 签名是"给谁看、看多久"的事，属业务侧——生成侧不知用户是谁。两边靠**确定性
对象键**接头（见下面的键模板），"这张图出没出来"问存储即知，不必另立台账。

**不转码。** 本层收字节、回字节：深度与遮罩是 16 位灰度 PNG，被下游当数据读，经过这一层
一个字节都不许变——Content-Type 只按字节首部判，不解码、不重编（测试逐字节断）。

依赖方向（import-linter 锁定）：本模块只依赖运行库（oss2），不感知上层，也不认识 temporalio。
形态照 imagegen 的 `image_store`，两仓各持一份（谁也不能 import 谁）。

键形态——**草案，待 contracts `registries/object_keys.md` 登记**（另一份登记草案正在
contracts 仓起，形态由用户拍板；登记时若形态变了，改本模块的模板与 `tests/test_object_store.py`
的守门测试，别处不留副本）：

    输入包    {prefix}/render3d/{revision_id}/design-package.json
    场景包    {prefix}/render3d/{revision_id}/scene-package.json
    各路图    {prefix}/render3d/{revision_id}/{camera_id}/geometry.png
                                                          depth.png
                                                          line.png
                                                          mask.png
                                                          mask-index.json
                                                          sketch.png（纯库回了才有）

`{prefix}` ＝ 输入包键去掉 `/render3d/…` 那段，本仓不解释它是什么（是上传件的
`uploads/{content_sha256}` 还是别的，由写输入包的一侧定）。三条判据同注册表开头那三条：
**确定性派生**（同键重跑覆盖同一个对象）、**不铸新流水号**、**键里不含用户身份与渠道方言**。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import oss2

RENDER3D_SEGMENT = "render3d"
"""本仓产物在前缀底下自己的那一段目录名。"""

DESIGN_PACKAGE_FILE = "design-package.json"
SCENE_PACKAGE_FILE = "scene-package.json"
GEOMETRY_FILE = "geometry.png"
DEPTH_FILE = "depth.png"
LINE_FILE = "line.png"
MASK_FILE = "mask.png"
MASK_INDEX_FILE = "mask-index.json"
SKETCH_FILE = "sketch.png"
"""各路产物的文件名。与 CLI 写本地目录用的是同一套名字（`cli.py` 顶部那组常量）——两处各持
一份而不互相 import：import-linter 锁死 activities 看不见 cli。名字变了两处一起改。"""

DESIGN_PACKAGE_KEY_TEMPLATE = "{prefix}/render3d/{revision_id}/design-package.json"
SCENE_PACKAGE_KEY_TEMPLATE = "{prefix}/render3d/{revision_id}/scene-package.json"
RENDER_VIEW_KEY_TEMPLATE = "{prefix}/render3d/{revision_id}/{camera_id}/{file_name}"
"""三条键模板（草案，待登记，见模块 docstring）。派生逻辑在 :class:`KeyRoot`，这三行是给人读的
形态说明与守门测试的对照物。"""

_ENDPOINT_ENV = "ISHOME_OSS_ENDPOINT"
_BUCKET_ENV = "ISHOME_OSS_BUCKET_PRIVATE"
_ACCESS_KEY_ID_ENV = "ISHOME_OSS_ACCESS_KEY_ID"
_ACCESS_KEY_SECRET_ENV = "ISHOME_OSS_ACCESS_KEY_SECRET"

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JSON_LEADING_BYTES = (b"{", b"[")


class ObjectStoreError(Exception):
    """取件或写件失败——响亮失败。写不进去就是这件产物没出来，不许当成功回报。"""

    def __init__(self, details: list[str]) -> None:
        super().__init__("；".join(details))
        self.details = details


# ---------------------------------------------------------------------------
# 键：形态校验与派生
# ---------------------------------------------------------------------------


def check_key_segment(segment: str, what: str) -> None:
    """一个值能不能当对象键的一段用。

    只拦"写进去就再也取不回来"的那几种：空、含 `/`（把键劈成两段）、`.`/`..`、首尾空白、
    控制字符、反斜杠。**不限定字符集**——机位 id 今天带房间名（拟真包里是 `cam-room-客厅`），
    OSS 的键收 UTF-8；限成小写字母数字会把现有输入全拦在门外。
    """
    if not segment:
        raise ObjectStoreError([f"{what}是空的，当不了对象键的一段"])
    if segment in (".", ".."):
        raise ObjectStoreError([f"{what}不能是 `{segment}`：当不了对象键的一段"])
    if "/" in segment or "\\" in segment:
        raise ObjectStoreError([f"{what}含 `/` 或 `\\`，会把对象键劈开：`{segment}`"])
    if segment != segment.strip():
        raise ObjectStoreError([f"{what}首尾带空白，写进去就取不回来：`{segment!r}`"])
    if any(ord(char) < 32 or ord(char) == 127 for char in segment):
        raise ObjectStoreError([f"{what}含控制字符，当不了对象键的一段：`{segment!r}`"])


@dataclass(frozen=True)
class KeyRoot:
    """一次修订的键根：前缀 + 修订号。本仓所有产物的键都从它派生，没有第二个来源。

    从输入包键解出来（:func:`key_root_of_design_package_key`）或从场景包键解出来
    （:func:`key_root_of_scene_package_key`）——两个 activity 各一个入口，出去的键形态相同。
    """

    prefix: str
    revision_id: str

    @property
    def design_package_key(self) -> str:
        return f"{self.prefix}/{RENDER3D_SEGMENT}/{self.revision_id}/{DESIGN_PACKAGE_FILE}"

    @property
    def scene_package_key(self) -> str:
        return f"{self.prefix}/{RENDER3D_SEGMENT}/{self.revision_id}/{SCENE_PACKAGE_FILE}"

    def view_key(self, camera_id: str, file_name: str) -> str:
        """某台机位的某一路产物的键。机位 id 先过一遍能不能当键段用。"""
        check_key_segment(camera_id, "机位 id")
        return f"{self.prefix}/{RENDER3D_SEGMENT}/{self.revision_id}/{camera_id}/{file_name}"


def _key_root_of(object_key: str, file_name: str, what: str) -> KeyRoot:
    """按 `{prefix}/render3d/{revision_id}/{file_name}` 从后往前解。

    从后往前而不是找第一个 `render3d`：前缀是上游的地盘，本仓不解释它——它里面碰巧也有一段
    叫 `render3d` 也不该把键解错。
    """
    template = f"{{prefix}}/{RENDER3D_SEGMENT}/{{revision_id}}/{file_name}"
    parts = object_key.split("/")
    if (
        len(parts) < 4
        or parts[-1] != file_name
        or parts[-3] != RENDER3D_SEGMENT
        or any(not part or part in (".", "..") for part in parts)
        or object_key != object_key.strip()
    ):
        raise ObjectStoreError([f"{what}键不合约定形态 `{template}`：`{object_key}`"])
    revision_id = parts[-2]
    check_key_segment(revision_id, f"{what}键里的修订号")
    return KeyRoot(prefix="/".join(parts[:-3]), revision_id=revision_id)


def key_root_of_design_package_key(design_package_key: str) -> KeyRoot:
    """输入包键 → 键根。形态不合当场失败，不猜、不容错：容错的结果是把产物写到没人会去读的地方。"""
    return _key_root_of(design_package_key, DESIGN_PACKAGE_FILE, "输入包")


def key_root_of_scene_package_key(scene_package_key: str) -> KeyRoot:
    """场景包键 → 键根。底渲各路的键从这儿派生，与场景包同前缀同修订。"""
    return _key_root_of(scene_package_key, SCENE_PACKAGE_FILE, "场景包")


# ---------------------------------------------------------------------------
# 字节：Content-Type 按首部判
# ---------------------------------------------------------------------------


def content_type_of(data: bytes) -> str:
    """这份字节的 Content-Type，**按首部判**：PNG 看魔数，JSON 看第一个非空白字节。

    不认得就抛——不给猜的。本仓只会写这两种（各路图是 PNG、场景包与遮罩索引表是 JSON）；
    出现第三种就是有人把不该写的东西送到了这儿。只判首部不解码：16 位 PNG 走过这里一个字节都不动。
    """
    if data.startswith(_PNG_MAGIC):
        return "image/png"
    if data.lstrip().startswith(_JSON_LEADING_BYTES):
        return "application/json"
    raise ObjectStoreError(
        [f"字节既不是 PNG 也不是 JSON（首部 {data[:8]!r}）：认不出格式就写不对头，不往桶里写"]
    )


# ---------------------------------------------------------------------------
# 桶
# ---------------------------------------------------------------------------


class ObjectStore(Protocol):
    """activity 看到的桶：取字节、写字节、报桶名。真桶是 :class:`OssObjectStore`；测试用内存假桶。

    只有这三个口。签名、列举、删除都不在——本层不知用户是谁，也不替谁收拾。
    """

    @property
    def bucket_name(self) -> str: ...

    def get(self, object_key: str, what: str) -> bytes:
        """按键取字节。取不到、取到空对象都上抛 :class:`ObjectStoreError`。"""
        ...

    def put(self, object_key: str, data: bytes) -> str:
        """写一份字节，返回键。空字节、写失败都上抛——不返回一个指向空气的键。"""
        ...


@dataclass(frozen=True)
class OssSettings:
    """私有桶连接口径。四个值全部来自环境，代码里不留任何默认桶名或端点。"""

    endpoint: str
    bucket: str
    access_key_id: str
    access_key_secret: str

    @staticmethod
    def from_env() -> OssSettings:
        """从环境读取；**缺一即起进程就失败**，不等到第一份产物写不进去才发现。"""
        values = {
            name: os.environ.get(name, "").strip()
            for name in (_ENDPOINT_ENV, _BUCKET_ENV, _ACCESS_KEY_ID_ENV, _ACCESS_KEY_SECRET_ENV)
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ObjectStoreError(
                [
                    f"私有对象存储没配全，缺：{'、'.join(missing)}——凭证放"
                    " ~/.ishome/oss-local.env（本机）或 /opt/ishome/env/oss.env（服务器），不入库"
                ]
            )
        return OssSettings(
            endpoint=values[_ENDPOINT_ENV],
            bucket=values[_BUCKET_ENV],
            access_key_id=values[_ACCESS_KEY_ID_ENV],
            access_key_secret=values[_ACCESS_KEY_SECRET_ENV],
        )


class OssObjectStore:
    """私有桶的读写口。签名不在这里——本层只读写不签（见模块 docstring）。"""

    def __init__(self, settings: OssSettings) -> None:
        auth = oss2.Auth(settings.access_key_id, settings.access_key_secret)
        self._bucket = oss2.Bucket(auth, settings.endpoint, settings.bucket)
        self._bucket_name = settings.bucket

    @property
    def bucket_name(self) -> str:
        return self._bucket_name

    def get(self, object_key: str, what: str) -> bytes:
        try:
            data: bytes = self._bucket.get_object(object_key).read()
        except oss2.exceptions.NoSuchKey as e:
            raise ObjectStoreError(
                [
                    f"{what}不在私有桶 `{self._bucket_name}` 里（键 {object_key}）——"
                    "要么写它的一侧没写成，要么两侧的键对不上"
                ]
            ) from e
        except oss2.exceptions.OssError as e:
            raise ObjectStoreError(
                [f"取{what}失败（桶 `{self._bucket_name}`，键 {object_key}）：{e}"]
            ) from e
        if not data:
            raise ObjectStoreError([f"{what}是空对象（键 {object_key}）"])
        return data

    def put(self, object_key: str, data: bytes) -> str:
        if not data:
            raise ObjectStoreError([f"字节是空的，不往桶里写（键 {object_key}）"])
        content_type = content_type_of(data)
        try:
            self._bucket.put_object(object_key, data, headers={"Content-Type": content_type})
        except oss2.exceptions.OssError as e:
            raise ObjectStoreError(
                [f"写不进私有桶 `{self._bucket_name}`（键 {object_key}）：{e}"]
            ) from e
        return object_key
