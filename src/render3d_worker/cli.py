"""CLI：`render3d --design design-package.json -o out/`。

**工具形态先行**（同 render2d 母版那条路的形态）：三维先以命令行工具存在，
**接进 activity 的时点写死＝派发链路接通、且本仓能落桶那一批**。先有能渲出来的东西，
再谈它在编排里怎么被调——反过来是接一遍再改一遍。

这条路不碰 Temporal、不碰对象存储：吃一份本地 :class:`DesignPackage` JSON，
出场景包与底渲四路图。import-linter 锁死 `cli` 看不见 `activities`——从它能看见
那一层起，"本地渲一张图不需要起编排"就只是一句承诺而不是结构。

产出（每台相机一个子目录）：
    scene-package.json          场景包（米制、含自证数）
    {camera_id}/geometry.png    几何：材质分色 + 固定方向明暗
    {camera_id}/depth.png       深度：16 位，还原回米要用 near_m/far_m
    {camera_id}/line.png        线稿：几何边界确定性提取，不做图像滤波猜边
    {camera_id}/mask.png        遮罩：索引图，0 是背景
    {camera_id}/mask-index.json 索引表：index → 网格 id / 语义 / 房间 / 像素数
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from render3d_worker.base_render import BaseRenderError, render_base_views
from render3d_worker.models import BaseRenderViews, DesignPackage, ScenePackage
from render3d_worker.scene_compile import SceneCompileError, compile_scene_package

SCENE_PACKAGE_JSON = "scene-package.json"
GEOMETRY_PNG = "geometry.png"
DEPTH_PNG = "depth.png"
LINE_PNG = "line.png"
MASK_PNG = "mask.png"
MASK_INDEX_JSON = "mask-index.json"

EXIT_BAD_INPUT = 2
EXIT_COMPILE_FAILED = 3
EXIT_RENDER_FAILED = 4


def _load_package(path: Path) -> DesignPackage:
    with path.open(encoding="utf-8") as f:
        payload: Any = json.load(f)
    # 上游若把包裹在 {"designPackage": {...}} 里（同几何 CLI 的裹法），直接喂那一份也认
    if isinstance(payload, dict) and "designPackage" in payload:
        payload = payload["designPackage"]
    return DesignPackage.model_validate(payload)


def _write_views(out_dir: Path, views: BaseRenderViews) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / GEOMETRY_PNG).write_bytes(views.geometry_png)
    (out_dir / DEPTH_PNG).write_bytes(views.depth_png)
    (out_dir / LINE_PNG).write_bytes(views.line_png)
    (out_dir / MASK_PNG).write_bytes(views.mask_png)
    index = [entry.model_dump(by_alias=True) for entry in views.mask_index]
    (out_dir / MASK_INDEX_JSON).write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _print_scene_self_check(scene: ScenePackage) -> None:
    """把自证数打出来。**不判**——门槛要有真跑数据才定（《纪律·阈值有数据才定》）。"""
    print(f"场景包：{len(scene.meshes)} 块网格 / {scene.triangle_count} 个三角形")
    anchor = "外轮廓围合面积" if scene.scale_anchor_source == "outline" else "外接框（退路）"
    print(f"  尺子：归一化 1.0 = {scene.metre_per_unit:.4f} 米，锚＝{anchor}")
    print(
        f"  地板面积：{scene.floor_area_sqm:.2f} ㎡"
        f"（与输入套内面积之比 {scene.area_match_ratio:.3f}）"
    )
    built = sum(scene.opening_count_by_kind.values())
    print(
        f"  墙段：{scene.wall_segment_count}（跳过退化段 {scene.degenerate_wall_count}）"
        f"  洞：{scene.opening_count_by_kind} 共 {built} 个"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="render3d",
        description="三维底渲：输入包 → 场景包 → 几何/深度/线稿/遮罩四路（确定性、零模型调用）",
    )
    parser.add_argument("--design", required=True, type=Path, help="DesignPackage JSON")
    parser.add_argument("-o", "--out", type=Path, default=Path("out"), help="产物目录")
    parser.add_argument("--camera", action="append", help="只渲这台相机（可给多次），默认全渲")
    parser.add_argument("--width-px", type=int, default=1024)
    parser.add_argument("--height-px", type=int, default=768)
    parser.add_argument("--scene-only", action="store_true", help="只编场景包，不渲图")
    args = parser.parse_args(argv)

    try:
        package = _load_package(args.design)
    except (OSError, ValueError, ValidationError) as e:
        print(f"读输入包失败：{e}", file=sys.stderr)
        return EXIT_BAD_INPUT

    try:
        scene = compile_scene_package(package)
    except SceneCompileError as e:
        print(f"编场景包失败：{e}", file=sys.stderr)
        return EXIT_COMPILE_FAILED

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / SCENE_PACKAGE_JSON).write_text(
        scene.model_dump_json(by_alias=True, indent=2), encoding="utf-8"
    )
    _print_scene_self_check(scene)

    if args.scene_only:
        return 0

    camera_ids = args.camera or [camera.id for camera in scene.cameras]
    if not camera_ids:
        # 包里一台相机都没有：这不是"渲个默认视角"能糊过去的，机位是输入不是产物
        print("输入包里没有相机，渲不了——机位是输入不是产物", file=sys.stderr)
        return EXIT_BAD_INPUT

    for camera_id in camera_ids:
        try:
            views = render_base_views(scene, camera_id, args.width_px, args.height_px)
        except BaseRenderError as e:
            print(f"底渲失败（相机 {camera_id}）：{e}", file=sys.stderr)
            return EXIT_RENDER_FAILED
        _write_views(args.out / camera_id, views)
        print(
            f"相机 {camera_id}：{views.width_px}×{views.height_px}，"
            f"几何盖住 {views.covered_pixel_ratio:.3f}，"
            f"深度 {views.near_m:.2f}~{views.far_m:.2f} 米，"
            f"遮罩 {len(views.mask_index)} 块"
        )

    print(f"产物在 {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
