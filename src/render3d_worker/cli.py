"""CLI：`render3d --design design-package.json -o out/`。

**工具形态先行**（同 render2d 母版那条路的形态）：三维先以命令行工具存在，
**接进 activity 的时点写死＝派发链路接通、且本仓能落桶那一批**。先有能渲出来的东西，
再谈它在编排里怎么被调——反过来是接一遍再改一遍。

这条路不碰 Temporal、不碰对象存储：吃一份本地 :class:`DesignPackage` JSON，
出场景包与底渲五路图。import-linter 锁死 `cli` 看不见 `activities`——从它能看见
那一层起，"本地渲一张图不需要起编排"就只是一句承诺而不是结构。

产出（每台相机一个子目录）：
    scene-package.json          场景包（米制、含自证数）
    {camera_id}/geometry.png    几何：材质分色 + 固定方向明暗
    {camera_id}/depth.png       深度：16 位，还原回米要用 near_m/far_m
    {camera_id}/line.png        线稿：几何事实边，保真度尺子的输入，不做图像滤波猜边
    {camera_id}/sketch.png      控制稿：给线稿生图控制通道画的（画法见 base_render 模块 docstring；
                                门窗符号方案由 --sketch-symbols 选，默认 frame-handle
                                ——2026-09-06 用户裁决换掉 diagonal-cross，来路与数据见
                                base_render.DEFAULT_SKETCH_SYMBOLS）
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

from render3d_worker.base_render import (
    DEFAULT_SKETCH_SYMBOLS,
    SKETCH_SYMBOL_SCHEMES,
    BaseRenderError,
    render_base_views,
)
from render3d_worker.furnish_mock import MockFurnishingReport, build_mock_furnishings
from render3d_worker.models import BaseRenderViews, DesignPackage, ScenePackage
from render3d_worker.scene_compile import SceneCompileError, compile_scene_package

SCENE_PACKAGE_JSON = "scene-package.json"
GEOMETRY_PNG = "geometry.png"
DEPTH_PNG = "depth.png"
LINE_PNG = "line.png"
SKETCH_PNG = "sketch.png"
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
    (out_dir / SKETCH_PNG).write_bytes(views.sketch_png)
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
    heights = "上游给的" if scene.heights_source == "upstream" else "常规住宅档位（mock）"
    print(f"  高度：{heights}")
    built = sum(scene.opening_count_by_kind.values())
    print(
        f"  墙段：{scene.wall_segment_count}（跳过退化段 {scene.degenerate_wall_count}）"
        f"  洞：{scene.opening_count_by_kind} 共 {built} 个"
    )
    guessed = "、".join(str(index) for index in scene.guessed_opening_indices) or "无"
    print(
        f"  按档位猜的洞（上游 kind=unknown，外墙＝窗/内墙＝门）："
        f"{scene.guessed_opening_count} 个，下标 {guessed}"
    )


def _print_mock_furnishing_self_check(report: MockFurnishingReport) -> None:
    """把 mock 家具的自证数打出来。**家具是 mock，不是这户人家的软装决策**——摆放依据
    只是常规住宅档位（见 `furnish_mock.py` 里每个尺寸常量的 docstring），不是量出来的、
    也不是设计出来的；只在显式加 `--mock-furnishing` 时才会走到这儿。"""
    print(f"家具：mock 摆场，{len(report.placements)} 件（常规住宅档位，非实测/非软装设计）")
    if report.unrecognized_rooms:
        print(f"  认不出房型，未摆：{'、'.join(report.unrecognized_rooms)}")
    if report.skipped_items:
        print(f"  认得房型但摆不下，跳过：{'、'.join(report.skipped_items)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="render3d",
        description=(
            "三维底渲：输入包 → 场景包 → 几何/深度/线稿/遮罩/控制稿五路（确定性、零模型调用）"
        ),
    )
    parser.add_argument("--design", required=True, type=Path, help="DesignPackage JSON")
    parser.add_argument("-o", "--out", type=Path, default=Path("out"), help="产物目录")
    parser.add_argument("--camera", action="append", help="只渲这台相机（可给多次），默认全渲")
    parser.add_argument("--width-px", type=int, default=1024)
    parser.add_argument("--height-px", type=int, default=768)
    parser.add_argument("--scene-only", action="store_true", help="只编场景包，不渲图")
    parser.add_argument(
        "--sketch-symbols",
        choices=SKETCH_SYMBOL_SCHEMES,
        default=DEFAULT_SKETCH_SYMBOLS,
        help=(
            "控制稿门窗符号方案（画法见 base_render.SKETCH_SYMBOL_SCHEMES）；其余四路不受它影响。"
            "默认 frame-handle（2026-09-06 用户裁决）；diagonal-cross 是旧默认，留着复现历史样本"
        ),
    )
    parser.add_argument(
        "--mock-furnishing",
        action="store_true",
        help="上游没给家具时按常规档位摆一份确定性 mock（测试用，默认关，见 furnish_mock.py）",
    )
    args = parser.parse_args(argv)

    try:
        package = _load_package(args.design)
    except (OSError, ValueError, ValidationError) as e:
        print(f"读输入包失败：{e}", file=sys.stderr)
        return EXIT_BAD_INPUT

    if args.mock_furnishing:
        if package.furnishings:
            # mock 是整份摆场、不是补白——留着输入自带的那几件会和 mock 的挤在一起打架
            print(
                f"输入包自带 {len(package.furnishings)} 件家具，"
                "--mock-furnishing 用 mock 摆场整体替换",
                file=sys.stderr,
            )
        report = build_mock_furnishings(package)
        package = package.model_copy(update={"furnishings": report.placements})
        _print_mock_furnishing_self_check(report)

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

    failed_camera_ids: list[str] = []
    for camera_id in camera_ids:
        try:
            views = render_base_views(
                scene, camera_id, args.width_px, args.height_px, args.sketch_symbols
            )
        except BaseRenderError as e:
            # 一台失败不拦着别的机位：每台机位各自独立，失败的那台不出图、最后一并报出来
            # 并以非零退出——响亮，但不把能出的图也扣下
            print(f"底渲失败（相机 {camera_id}）：{e}", file=sys.stderr)
            failed_camera_ids.append(camera_id)
            continue
        _write_views(args.out / camera_id, views)
        print(
            f"相机 {camera_id}：{views.width_px}×{views.height_px}，"
            f"几何盖住 {views.covered_pixel_ratio:.3f}，"
            f"深度 {views.near_m:.2f}~{views.far_m:.2f} 米，"
            f"遮罩 {len(views.mask_index)} 块"
        )
        if views.room_view is not None:
            _print_room_view_self_check(views)

    print(f"产物在 {args.out}")
    if failed_camera_ids:
        print(f"底渲失败的机位：{'、'.join(failed_camera_ids)}", file=sys.stderr)
        return EXIT_RENDER_FAILED
    return 0


def _print_room_view_self_check(views: BaseRenderViews) -> None:
    """室内机位的取景自证数。自动取景的必然达标（不达标已经响亮失败了）；上游显式给 yaw
    的只量不判，打出来让人看见"上游给的机位按我们的判据是不是对着墙"。"""
    check = views.room_view
    assert check is not None
    verdict = "达标" if check.passed else "不达标（上游显式给的 yaw，只量不判）"
    print(
        f"  取景：站在 ({check.eye_m[0]:.2f}, {check.eye_m[1]:.2f}) 朝 {check.yaw_deg:.1f}°，"
        f"评估了 {check.candidate_count} 个候选；最近深度 {check.min_depth_m:.2f} 米，"
        f"目标房间地板占比 {check.target_floor_ratio:.3f}，"
        f"主体占比 {check.dominance_ratio:.3f}"
        f"（目标 {check.target_room_ratio:.3f} / 其他房间 {check.other_room_ratio:.3f}）——{verdict}"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
