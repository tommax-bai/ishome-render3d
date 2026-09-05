"""两个 activity 的出入参：编排那一侧看到的契约。

与 :mod:`render3d_worker.models` 分开放的理由：那份是**上游与纯库之间**的契约（输入包、场景包），
这份是**编排与本仓之间**的契约（请求、回执）。两份各归各——纯库与 CLI 不该看见请求怎么长，
编排也不该看见网格怎么长（import-linter 把两者锁在同一层、互不可见）。

**请求收 camelCase、`extra=forbid`**（同输入包那条口径）：多出来的字段不是"宽容一点收下"的事，
它说明两侧对不上头，早一步炸在边界上，好过带着一个没人读的字段一路走到出图。

**回执是 snake_case 的普通字典**（同 imagegen 两个 activity 的回执形态）：`verdict` 二选一——
`ok` 带键与自证数，`failed` 带 `violations[{check, detail}]`。`check` 是错误分类，四类写死：
取键失败 / 契约校验失败 / 纯库失败 / 写桶失败（见 :data:`CHECK_OBJECT_STORE_GET` 起那组常量）。

**没有档位参数**（用户裁决 2026-09-04：两档合一档）；**不签链接、不知用户是谁**。

与 genpipe 编排那边共用的契约（2026-09-05）：

    scene-compile  入  designPackageKey, revisionId
                   出  scene_package_key, bucket, 自证数
    base-render    入  scenePackageKey, cameraIds（null＝全渲）, widthPx, heightPx
                   出  renders[{camera_id, geometry_key, depth_key, line_key, mask_key,
                                mask_index_key, sketch_key, 自证数}], 自证数
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

CHECK_BAD_INPUT = "bad-input"
"""请求本身解析不了：字段缺、多、类型错。这一类重试一万次也是同一个结果。"""

CHECK_OBJECT_STORE_GET = "object-store-get-failed"
"""取键失败：桶里没有这个键、桶连不上、取到空对象。"""

CHECK_CONTRACT_INVALID = "contract-invalid"
"""契约校验失败：键形态不合、包 JSON 解析不了或不合契约、修订号对不上、机位不在场景包里。"""

CHECK_SCENE_COMPILE_FAILED = "scene-compile-failed"
"""纯库失败（编场景包）：几何空、面积为 0、材质表坏——原文照抄 :class:`SceneCompileError`。"""

CHECK_BASE_RENDER_FAILED = "base-render-failed"
"""纯库失败（底渲）：取景失败、房间没地板、材质对不上——**逐机位一条**，原文照抄
:class:`BaseRenderError`，不吞。"""

CHECK_OBJECT_STORE_PUT = "object-store-put-failed"
"""写桶失败：产物出得再好、落不了地也按失败回报——回一个指向空气的键，下游会拿它去用。"""


class _ActivityContract(BaseModel):
    """请求基类：camelCase 别名对齐编排那一侧的序列化；`extra=forbid` 拒收多出来的字段。"""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class SceneCompileRequest(_ActivityContract):
    """scene-compile 请求：**吃对象键，不吃本地路径**（包本体在私有桶里，activity 拿键去取）。

    `revision_id` 与键里那一段**必须一致**，包里写的 `revisionId` 也必须一致——三处对不上就是
    派发方拿错了包，当场按契约校验失败回报，不猜哪一个是对的。
    """

    design_package_key: str
    revision_id: str


class BaseRenderRequest(_ActivityContract):
    """base-render 请求：一份场景包、若干台机位、一个画幅。

    `camera_ids` 为 `None` ＝ 场景包里的机位全渲；给了就只渲这些，每一个都得在场景包里。
    画幅两个数**必填**：默认值是"猜"的一种形态，编排要多大就得说出来。
    """

    scene_package_key: str
    camera_ids: list[str] | None = None
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)

    @field_validator("camera_ids")
    @classmethod
    def _camera_ids_non_empty_and_unique(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("cameraIds 给了空表：要全渲就给 null，一台都不渲就别派这一步")
        duplicates = sorted({camera_id for camera_id in value if value.count(camera_id) > 1})
        if duplicates:
            raise ValueError(f"cameraIds 里有重复：{'、'.join(duplicates)}")
        return value


class Violation(BaseModel):
    check: str
    detail: str


class FailedReceipt(BaseModel):
    """失败回执。`violations` 至少一条；同一次可以有多条（底渲逐机位一条）。"""

    verdict: Literal["failed"] = "failed"
    violations: list[Violation]


class SceneCompileReceipt(BaseModel):
    """scene-compile 成功回执：场景包的键 + 编包的自证数（全部来自场景包自带的那组数）。"""

    verdict: Literal["ok"] = "ok"
    scene_package_key: str
    bucket: str
    design_package_key: str
    revision_id: str
    mesh_count: int
    triangle_count: int
    metre_per_unit: float
    floor_area_sqm: float
    area_match_ratio: float
    """编出来的地板面积对输入套内面积之比。**不判**——门槛要有真跑数据才定，先带出去。"""

    wall_segment_count: int
    degenerate_wall_count: int
    opening_count_by_kind: dict[str, int]
    heights_source: str
    scale_anchor_source: str
    camera_ids: list[str]
    scene_package_size_bytes: int
    elapsed_seconds: float


class RenderedView(BaseModel):
    """一台机位的产物：五路图的键 + 遮罩索引表的键 + 这台机位的自证数。

    五路键都必有（第五路控制稿 `sketch_key` 自 2026-09-05 合流起与其余四路同列，不再可空）。
    `room_view` 只有 `room` 机位才有：取景自证数原样带出，`bird` 机位为 `None`。
    """

    camera_id: str
    geometry_key: str
    depth_key: str
    line_key: str
    mask_key: str
    mask_index_key: str
    sketch_key: str
    width_px: int
    height_px: int
    covered_pixel_ratio: float
    near_m: float
    far_m: float
    mask_entry_count: int
    room_view: dict[str, Any] | None
    elapsed_seconds: float


class BaseRenderReceipt(BaseModel):
    """base-render 成功回执：每台机位一条 :class:`RenderedView`。"""

    verdict: Literal["ok"] = "ok"
    scene_package_key: str
    bucket: str
    revision_id: str
    renders: list[RenderedView]
    elapsed_seconds: float
