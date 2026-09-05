"""Temporal activities：所有 IO 与重计算收口在此。

注册名唯一真源：ishome-contracts `activities/registry.md`，**只增不改**——改注册名
会破坏历史 workflow 重放，等同于改线上协议；新增走 contracts 仓 PR 评审。
命名规则（规范 §2.4）：注册名 = kebab-case 显式声明；函数名 = 同词 snake_case
动词前置。

**两个 activity 至此实装**（2026-09-05）。此前是"接线时点写死"的诚实存根，缺的三件——
落桶（:mod:`render3d_worker.object_store`）、键的登记（键形态草案写在那个模块的 docstring，
**待 contracts 登记**）、函数体——这一批补齐。形态照 imagegen 两个 activity 的先例：实现件是类，
进程级依赖（私有桶）由组合根 `worker` 注入并在**起进程时**当场校验，入参是**不透明字典**。
函数体就是一句话：**取键 → 调纯库 → 写桶 → 返回键与自证数**。纯库一行不改。

错误分四类、逐类落在回执 `violations[].check` 上（常量在 :mod:`render3d_worker.activity_models`）：
取键失败 / 契约校验失败 / 纯库失败 / 写桶失败。**都不上抛异常而是按 `failed` 回报**：这四类里
只有取键与写桶是重试可能好转的，要不要重试由编排侧显式决定，不由本仓用异常替它决定。

底渲**逐机位**：一台取景失败（另一条分支让小房间取景响亮失败）**原样传成该机位的一条
violation，不吞**，其余机位照渲照写；有任何一台失败整份回执 `verdict=failed`，写进桶的那些
仍列在 `renders` 里——不吞也不丢，派发方看得见哪台出了、哪台没出、为什么。写桶失败则
当场停：桶坏了后面每一台都会坏，重复几条同样的话没有意义。

**CLI 不废**：本地渲一张图走 `cli.py`，不起 Temporal、不碰桶；两条路共用同一份纯库代码，
分界由 import-linter 锁死（`cli` 看不见 `activities`）。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Coroutine
from typing import Any

from pydantic import ValidationError
from temporalio import activity

from render3d_worker.activity_models import (
    CHECK_BAD_INPUT,
    CHECK_BASE_RENDER_FAILED,
    CHECK_CONTRACT_INVALID,
    CHECK_OBJECT_STORE_GET,
    CHECK_OBJECT_STORE_PUT,
    CHECK_SCENE_COMPILE_FAILED,
    BaseRenderReceipt,
    BaseRenderRequest,
    FailedReceipt,
    RenderedView,
    SceneCompileReceipt,
    SceneCompileRequest,
    Violation,
)
from render3d_worker.base_render import BaseRenderError, render_base_views
from render3d_worker.models import BaseRenderViews, DesignPackage, ScenePackage
from render3d_worker.object_store import (
    DEPTH_FILE,
    GEOMETRY_FILE,
    LINE_FILE,
    MASK_FILE,
    MASK_INDEX_FILE,
    SKETCH_FILE,
    KeyRoot,
    ObjectStore,
    ObjectStoreError,
    key_root_of_design_package_key,
    key_root_of_scene_package_key,
)
from render3d_worker.scene_compile import SceneCompileError, compile_scene_package

ActivityResult = dict[str, Any]

ACTIVITY_SCENE_COMPILE = "scene-compile"
ACTIVITY_BASE_RENDER = "base-render"
"""contracts 注册名。字符串在此各声明一次，worker 与守门测试都引它。"""

_DESIGN_PACKAGE_WRAPPER_KEY = "designPackage"
"""上游若把包裹在 `{"designPackage": {...}}` 里（同几何 CLI 的裹法），直接喂那一份也认——
与 `cli.py` 读本地 JSON 的口径相同：两条路同一份契约、同一种裹法。"""

_ERROR_DETAIL_MAX_CHARS = 400
"""pydantic 的多行报告压成一行再进回执：回执躺在 Temporal 历史里，要一眼读得懂。"""


class SceneCompiler:
    """scene-compile 的实现件：私有桶由组合根（worker）注入。"""

    def __init__(self, store: ObjectStore) -> None:
        self._store = store

    @activity.defn(name=ACTIVITY_SCENE_COMPILE)
    async def compile_scene(self, request: dict[str, Any]) -> ActivityResult:
        """输入包（桶里）→ 场景包 → 写回同一前缀，返回**场景包的键**与编包的自证数。

        键从输入包键派生（:class:`~render3d_worker.object_store.KeyRoot`）；请求里的
        `revisionId`、键里的修订号、包里写的 `revisionId` 三处必须一致。
        """
        started = time.perf_counter()
        try:
            parsed = SceneCompileRequest.model_validate(request)
        except ValidationError as e:
            return _failed(CHECK_BAD_INPUT, f"入参解析失败：{_one_line(e)}")

        try:
            root = key_root_of_design_package_key(parsed.design_package_key)
        except ObjectStoreError as e:
            return _failed_many(CHECK_CONTRACT_INVALID, e.details)
        if root.revision_id != parsed.revision_id:
            return _failed(
                CHECK_CONTRACT_INVALID,
                f"请求的 revisionId `{parsed.revision_id}` 与输入包键里的修订号"
                f" `{root.revision_id}` 对不上（键 {parsed.design_package_key}）",
            )

        try:
            package_bytes = await asyncio.to_thread(
                self._store.get, parsed.design_package_key, "输入包"
            )
        except ObjectStoreError as e:
            return _failed_many(CHECK_OBJECT_STORE_GET, e.details)

        try:
            package = _parse_design_package(package_bytes)
        except (ValueError, ValidationError) as e:
            return _failed(
                CHECK_CONTRACT_INVALID,
                f"输入包不合契约（键 {parsed.design_package_key}）：{_one_line(e)}",
            )
        if package.revision_id != parsed.revision_id:
            return _failed(
                CHECK_CONTRACT_INVALID,
                f"输入包里写的 revisionId `{package.revision_id}` 与请求的"
                f" `{parsed.revision_id}` 对不上（键 {parsed.design_package_key}）",
            )

        try:
            # 编包是纯 CPU（整户几百块网格，秒级）：丢进线程里跑，不占事件循环。
            scene = await asyncio.to_thread(compile_scene_package, package)
        except SceneCompileError as e:
            return _failed(CHECK_SCENE_COMPILE_FAILED, str(e))

        # 与 CLI 写本地 `scene-package.json` 逐字同形（by_alias + 缩进 2）：同一份输入包，
        # 本地渲出来的场景包与桶里那份逐字节相同，对起账来不必先猜是不是序列化不同。
        scene_bytes = scene.model_dump_json(by_alias=True, indent=2).encode("utf-8")
        try:
            scene_package_key = await asyncio.to_thread(
                self._store.put, root.scene_package_key, scene_bytes
            )
        except ObjectStoreError as e:
            return _failed_many(CHECK_OBJECT_STORE_PUT, e.details)

        return SceneCompileReceipt(
            scene_package_key=scene_package_key,
            bucket=self._store.bucket_name,
            design_package_key=parsed.design_package_key,
            revision_id=scene.revision_id,
            mesh_count=len(scene.meshes),
            triangle_count=scene.triangle_count,
            metre_per_unit=scene.metre_per_unit,
            floor_area_sqm=scene.floor_area_sqm,
            area_match_ratio=scene.area_match_ratio,
            wall_segment_count=scene.wall_segment_count,
            degenerate_wall_count=scene.degenerate_wall_count,
            opening_count_by_kind=dict(scene.opening_count_by_kind),
            guessed_opening_count=scene.guessed_opening_count,
            guessed_opening_indices=list(scene.guessed_opening_indices),
            heights_source=scene.heights_source,
            scale_anchor_source=scene.scale_anchor_source,
            camera_ids=[camera.id for camera in scene.cameras],
            scene_package_size_bytes=len(scene_bytes),
            elapsed_seconds=_elapsed(started),
        ).model_dump()


class BaseViewRenderer:
    """base-render 的实现件：私有桶由组合根（worker）注入。"""

    def __init__(self, store: ObjectStore) -> None:
        self._store = store

    @activity.defn(name=ACTIVITY_BASE_RENDER)
    async def render_base(self, request: dict[str, Any]) -> ActivityResult:
        """场景包（桶里）+ 机位表 + 画幅 → 每台机位各路图 → 写回同一前缀，返回各路的键与自证数。

        几何由纯库定、零模型调用、无随机：同一份场景包同一画幅渲两次，键与字节逐字相同。
        深度与遮罩是 16 位灰度 PNG，**原字节写桶**，不经任何解码重编。
        """
        started = time.perf_counter()
        try:
            parsed = BaseRenderRequest.model_validate(request)
        except ValidationError as e:
            return _failed(CHECK_BAD_INPUT, f"入参解析失败：{_one_line(e)}")

        try:
            root = key_root_of_scene_package_key(parsed.scene_package_key)
        except ObjectStoreError as e:
            return _failed_many(CHECK_CONTRACT_INVALID, e.details)

        try:
            scene_bytes = await asyncio.to_thread(
                self._store.get, parsed.scene_package_key, "场景包"
            )
        except ObjectStoreError as e:
            return _failed_many(CHECK_OBJECT_STORE_GET, e.details)

        try:
            scene = ScenePackage.model_validate_json(scene_bytes)
        except ValidationError as e:
            return _failed(
                CHECK_CONTRACT_INVALID,
                f"场景包不合契约（键 {parsed.scene_package_key}）：{_one_line(e)}",
            )
        if scene.revision_id != root.revision_id:
            return _failed(
                CHECK_CONTRACT_INVALID,
                f"场景包里写的 revisionId `{scene.revision_id}` 与键里的修订号"
                f" `{root.revision_id}` 对不上（键 {parsed.scene_package_key}）",
            )

        known_camera_ids = [camera.id for camera in scene.cameras]
        if parsed.camera_ids is None:
            camera_ids = known_camera_ids
            if not camera_ids:
                # 包里一台相机都没有：这不是"渲个默认视角"能糊过去的，机位是输入不是产物
                return _failed(
                    CHECK_CONTRACT_INVALID,
                    "场景包里一台相机都没有，渲不了——机位是输入不是产物"
                    f"（键 {parsed.scene_package_key}）",
                )
        else:
            camera_ids = parsed.camera_ids
            unknown = [camera_id for camera_id in camera_ids if camera_id not in known_camera_ids]
            if unknown:
                return _failed(
                    CHECK_CONTRACT_INVALID,
                    f"场景包里没有这些机位：{'、'.join(unknown)}；"
                    f"已有：{'、'.join(known_camera_ids) or '（一台都没有）'}",
                )

        renders: list[RenderedView] = []
        violations: list[Violation] = []
        for camera_id in camera_ids:
            camera_started = time.perf_counter()
            try:
                # 底渲是纯 CPU（1024×768 上秒级）：丢进线程里跑，不占事件循环。
                views = await asyncio.to_thread(
                    render_base_views, scene, camera_id, parsed.width_px, parsed.height_px
                )
            except BaseRenderError as e:
                # 取景失败、房间没地板、材质对不上——**原样传成这台机位的一条 violation**，
                # 其余机位照渲。不上抛：这一类重试一万次也是同一个结果。
                violations.append(
                    Violation(check=CHECK_BASE_RENDER_FAILED, detail=f"机位 {camera_id}：{e}")
                )
                continue
            try:
                rendered = await asyncio.to_thread(self._put_views, root, views)
            except ObjectStoreError as e:
                # 桶坏了后面每一台都会坏：当场停，已写进去的那些照样列出来。
                violations.extend(
                    Violation(check=CHECK_OBJECT_STORE_PUT, detail=f"机位 {camera_id}：{detail}")
                    for detail in e.details
                )
                break
            renders.append(
                rendered.model_copy(update={"elapsed_seconds": _elapsed(camera_started)})
            )

        if violations:
            return {
                **FailedReceipt(violations=violations).model_dump(),
                "scene_package_key": parsed.scene_package_key,
                "bucket": self._store.bucket_name,
                "revision_id": scene.revision_id,
                "renders": [rendered.model_dump() for rendered in renders],
                "elapsed_seconds": _elapsed(started),
            }
        return BaseRenderReceipt(
            scene_package_key=parsed.scene_package_key,
            bucket=self._store.bucket_name,
            revision_id=scene.revision_id,
            renders=renders,
            elapsed_seconds=_elapsed(started),
        ).model_dump()

    def _put_views(self, root: KeyRoot, views: BaseRenderViews) -> RenderedView:
        """一台机位的各路写桶，返回键与自证数（耗时由调用方填）。任一路写失败即上抛。

        五路图逐路必写（第五路 `sketch.png` 自 2026-09-05 合流起是纯库的必填产物）；取景自证数
        `room_view` 只有 `room` 机位才有，`bird` 机位为 `None`，原样带出。
        遮罩索引表与 CLI 写本地 `mask-index.json` 逐字同形（by_alias + 缩进 2）。
        """
        camera_id = views.camera_id
        mask_index = [entry.model_dump(by_alias=True) for entry in views.mask_index]
        mask_index_bytes = json.dumps(mask_index, ensure_ascii=False, indent=2).encode("utf-8")
        return RenderedView(
            camera_id=camera_id,
            geometry_key=self._store.put(
                root.view_key(camera_id, GEOMETRY_FILE), views.geometry_png
            ),
            depth_key=self._store.put(root.view_key(camera_id, DEPTH_FILE), views.depth_png),
            line_key=self._store.put(root.view_key(camera_id, LINE_FILE), views.line_png),
            mask_key=self._store.put(root.view_key(camera_id, MASK_FILE), views.mask_png),
            mask_index_key=self._store.put(
                root.view_key(camera_id, MASK_INDEX_FILE), mask_index_bytes
            ),
            sketch_key=self._store.put(root.view_key(camera_id, SKETCH_FILE), views.sketch_png),
            width_px=views.width_px,
            height_px=views.height_px,
            covered_pixel_ratio=views.covered_pixel_ratio,
            near_m=views.near_m,
            far_m=views.far_m,
            mask_entry_count=len(views.mask_index),
            room_view=None if views.room_view is None else views.room_view.model_dump(),
            elapsed_seconds=0.0,
        )


def _parse_design_package(package_bytes: bytes) -> DesignPackage:
    payload: Any = json.loads(package_bytes)
    if isinstance(payload, dict) and _DESIGN_PACKAGE_WRAPPER_KEY in payload:
        payload = payload[_DESIGN_PACKAGE_WRAPPER_KEY]
    return DesignPackage.model_validate(payload)


def _one_line(error: Exception) -> str:
    return " ".join(str(error).split())[:_ERROR_DETAIL_MAX_CHARS]


def _elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 3)


def _failed(check: str, detail: str) -> ActivityResult:
    return FailedReceipt(violations=[Violation(check=check, detail=detail)]).model_dump()


def _failed_many(check: str, details: list[str]) -> ActivityResult:
    return FailedReceipt(
        violations=[Violation(check=check, detail=detail) for detail in details]
    ).model_dump()


ACTIVITY_REGISTRY: dict[str, Callable[..., Coroutine[Any, Any, ActivityResult]]] = {
    ACTIVITY_SCENE_COMPILE: SceneCompiler.compile_scene,
    ACTIVITY_BASE_RENDER: BaseViewRenderer.render_base,
}
"""注册名 → 实现（未绑定的方法）。键与 contracts 注册表逐字一致（tests/test_activity_registry.py
断言）。进程里绑定了桶的那份由 :func:`activity_registry` 给 worker。"""


def activity_registry(
    compiler: SceneCompiler, renderer: BaseViewRenderer
) -> dict[str, Callable[..., Coroutine[Any, Any, ActivityResult]]]:
    """本仓承接的 activity 全集（队列 `render3d-activities`），绑定到装好桶的实现件上。"""
    return {
        ACTIVITY_SCENE_COMPILE: compiler.compile_scene,
        ACTIVITY_BASE_RENDER: renderer.render_base,
    }
