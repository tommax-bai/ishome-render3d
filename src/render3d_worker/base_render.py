"""底渲（activity ``base-render``）：一份场景包 + 一个机位 → 几何/深度/线稿/遮罩四路图。

**四路是给下一步当条件图的，不是给人看的成品图**——写实化（``realism-pass``）在 imagegen
那个仓走生成模型。所以这一步既不需要 GPU 也不需要三维引擎：四路全是几何缓冲，纯 numpy
软光栅就出得来（光栅本身在 :mod:`render3d_worker.raster`，本模块只管"编码成哪四张图"）。

四路的编码口径**一律"没有东西 ＝ 0"**，四张图对齐同一条约定：

===== ================== =====================================================
路    图像形态            背景（没打到几何）
===== ================== =====================================================
几何  8 位 RGB PNG        纯黑 (0, 0, 0)
深度  16 位灰度 PNG       0（几何像素落在 1..65535，见 :func:`_encode_depth_png`）
线稿  8 位灰度 PNG        0（黑底白线）
遮罩  16 位灰度索引 PNG   0（索引 0 保留给背景）
===== ================== =====================================================

统一成这一条的理由：下游把四路当多通道条件图拼起来时，不用逐路记"这一路是反的"；
而且"背景＝0"让每一路单独都能答出"这个像素有没有东西"，四路一致性可自查（测试即断这条）。

失败要响亮：相机 id 找不到、``room`` 机位指的房间没有地板、网格引用的材质不在场景包里，
一律抛 :class:`BaseRenderError`，**不退化成默认相机、不编一个兜底颜色**——退化只会让一张
看着正常、其实渲错了机位的图流到下游（《纪律·拿不到就说没有，不许填猜的值》）。
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image

from render3d_worker.models import (
    BaseRenderViews,
    CameraKind,
    CameraSpec,
    MaskEntry,
    MeshSemantic,
    ScenePackage,
)
from render3d_worker.raster import (
    Float64Array,
    RasterBuffers,
    look_at_matrix,
    perspective_matrix,
    rasterize,
)

# ---------------------------------------------------------------------------
# 口径常量：这一节的每个数都决定"图长什么样"，集中在一处，代码里不再散着写。
# ---------------------------------------------------------------------------

NEAR_CLIP_M: float = 0.05
"""近裁剪面（米）。取 5 厘米：室内机位贴着墙也留得出余量，而比这更近的东西没有取景意义。
深度精度**不受它影响**——本仓的深度是视空间米数，不走 NDC z（见 :func:`perspective_matrix`）。"""

FAR_CLIP_DIAGONAL_RATIO: float = 4.0
FAR_CLIP_MIN_M: float = 50.0
"""远裁剪面 = max(场景包围盒对角线 × 4, 50 米)。远平面不参与裁剪也不参与深度归一化，
只是投影矩阵要一个数；给得宽松就不会有人因为它被切掉。"""

BIRD_FRAMING_MARGIN_RATIO: float = 1.06
"""bird 机位把整户外接球塞进画面后再退开 6%——留一圈边，免得墙角正好压在画幅上。"""

BIRD_HIDDEN_SEMANTICS: frozenset[MeshSemantic] = frozenset({"ceiling"})
"""**bird 机位渲染时剔掉天花**（用户裁决 2026-08-31）。

原话："bird 这个机位的用途就是'从上往下看这户人家的布局'，天花挡在中间它就一点信息都不带；
剖切面要多一个'切在哪个高度'的参数，而那个参数今天没有任何依据能定（《纪律·阈值有数据才定》），
剔 ceiling 是零参数、确定性的做法。"

剔的是**这一个机位的这一次渲染**，不是场景包：ceiling 网格照编不误，交互引擎与 room 机位
照常用得到（所以判据挂在 ``CameraSpec.kind == "bird"`` 上，不是挂在编包那一侧）。
遮罩索引不受影响——索引恒等于网格在 ``scene.meshes`` 里的下标 +1，剔掉的网格只是不出现在
``mask_index`` 里，剩下的网格索引一个都不挪位（不然同一份包换台相机，同一块墙的索引就变了）。"""

ROOM_PITCH_DEG: float = 0.0
"""``room`` 机位固定**平视**。``CameraSpec.pitch_deg`` 只对 ``bird`` 生效：站在房间里低头，
画面全是地板、形体读不出来，条件图就废了。要改室内俯仰改这一个常量。"""

ROOM_TARGET_DISTANCE_M: float = 1.0
"""``room`` 机位的注视点取眼前 1 米——只用来定视线方向，远近不影响透视。"""

LIGHT_FROM_DIR_XYZ: tuple[float, float, float] = (-0.40, -0.60, 0.70)
"""几何那一路的**唯一**光源方向（世界系，指向光源那一侧；模块内归一化）。

写死不随相机转的理由有两条：一是确定性——光跟着机位转，同一户换个机位就没法把两张图
的明暗对上；二是这盏灯只为"让形体读得出来"服务，不是为了写实（写实归 realism-pass）。
方向取左·前·上（-x, -y, +z）：三维制图惯例的主光位；带 -y 分量（朝观察者那一侧）是为了
让正对相机的墙面别糊成一片黑。"""

AMBIENT_RATIO: float = 0.35
"""环境光占比。背光面留 35% 亮度：条件图要的是"读得出这儿有一面墙"，不是真实的暗部。"""

GEOMETRY_BACKGROUND_RGB_U8: tuple[int, int, int] = (0, 0, 0)
LINE_BACKGROUND_U8: int = 0
LINE_FOREGROUND_U8: int = 255
"""线稿**黑底白线**。与"没有东西＝0"那条对齐，也和 canny 一类线稿条件图的通行形态一致，
下游不用先反相。"""

LINE_DEPTH_TOLERANCE_RATIO: float = 0.02
"""线稿的深度不连续判据：实测深度与**按法向外推的预测深度**差超过 2% 才算断开。

为什么不是"相邻像素深度差超过 X 米/X%"那种阈值：掠射的地板相邻两像素本来就能差好几个
百分点，定低了满屏假线、定高了近处的边丢掉，怎么定都不对。本仓所有几何都是三角面（平面），
所以"同一张面继续下去应该是多深"是**能精确算出来**的——连续面上残差恒为 0，只有真跨过
遮挡边界或折角才跳起来。2% 只是留给浮点误差的余量，不是靠调它来分边缘。"""

DEPTH_BACKGROUND_U16: int = 0
DEPTH_MIN_U16: int = 1
DEPTH_MAX_U16: int = 65535
"""深度图把 0 让给背景，几何像素落在 1..65535。这样"有没有几何"从深度图单独就答得出来，
不必再取一次遮罩；代价是丢掉 65536 分之一的动态范围，换一条能自查的一致性。"""

DEPTH_MIN_SPAN_M: float = 1e-6
"""深度归一化两端之间的最小跨度，防 ``far_m == near_m`` 时除零（整幅画只有一个深度）。"""

MASK_MAX_INDEX: int = 65535
"""遮罩用 16 位索引图：网格数上不封顶（每段墙、每个门窗套、每件家具都是一块），
8 位的 255 个索引不够用。代价只是文件大一倍，换"不会因为户型大一点就编不下"。"""


class BaseRenderError(Exception):
    """底渲失败：相机/房间/材质在场景包里对不上，或画幅参数不合法。"""


@dataclass(frozen=True)
class CameraPose:
    """一台相机解出来的实际机位：算矩阵要的全部数。

    单独暴露（而不是埋在 :func:`render_base_views` 里）的理由：bird 机位是**算**出来的，
    "这张图是从哪儿看的"必须答得出来——activity 要把它记进产物元数据，测试要拿它把世界点
    投到像素上验深度，都不该各自再实现一遍取景算法。
    """

    camera_id: str
    kind: CameraKind
    """机位类型跟着姿态一起带出来：**哪些网格要剔**是按它判的（见
    :data:`BIRD_HIDDEN_SEMANTICS`），取景与绘制得用同一个判据，否则机位框住的东西
    和画出来的东西不是一回事。"""

    eye_m: tuple[float, float, float]
    target_m: tuple[float, float, float]
    up_hint_xyz: tuple[float, float, float]
    fov_deg: float
    near_clip_m: float
    far_clip_m: float


@dataclass(frozen=True, eq=False)
class _ScreenGeometry:
    """逐像素的视空间量，几何与线稿两路共用，算一次传两处。"""

    ray_view_xyz: Float64Array
    """(H, W, 3) 每个像素的视空间方向，z 恒为 -1——所以"沿射线走 t"里的 t 就是正深度（米）。"""

    normal_view_xyz: Float64Array
    position_view_m: Float64Array
    facing_sign: Float64Array
    """(H, W) 法向朝不朝着相机：+1 保留、-1 翻转（上游绕序不保证一致，见 raster 不做背面剔除）。"""


def resolve_camera_pose(scene: ScenePackage, camera_id: str, aspect_ratio: float) -> CameraPose:
    """按 ``camera_id`` 解出实际机位。找不到相机／房间就抛 :class:`BaseRenderError`。

    - ``bird``：注视整户包围盒中心，方向用相机自带的 ``yaw_deg``/``pitch_deg``，**距离自动算**
      ——把包围盒外接球塞进视锥里较窄的那个方向（竖直与水平张角取小者），再退 6% 留边。
    - ``room``：站在该房间地板的**面积加权质心**上、抬到 ``eye_height_m``，按 ``yaw_deg`` 平视。

    包围盒**按网格顶点现算**，不读 ``bounds_min_m``/``bounds_max_m``：那两个字段是场景包的
    自证数（编包那一侧填的），取景必须框住真正会被画出来的东西，两者万一不一致，以画得出来
    的为准。同理，取景只框**这台相机会画的那些网格**——bird 剔了天花，包围盒就不该再被
    天花撑着。
    """
    if aspect_ratio <= 0.0:
        raise BaseRenderError(f"宽高比必须为正：aspect_ratio={aspect_ratio}")
    camera = _find_camera(scene, camera_id)
    min_xyz_m, max_xyz_m = _scene_bounds_m(scene, _rendered_mesh_indices(scene, camera.kind))
    diagonal_m = float(np.linalg.norm(max_xyz_m - min_xyz_m))
    far_clip_m = max(FAR_CLIP_MIN_M, diagonal_m * FAR_CLIP_DIAGONAL_RATIO)

    if camera.kind == "bird":
        center_m = (min_xyz_m + max_xyz_m) * 0.5
        forward = _yaw_pitch_direction(camera.yaw_deg, camera.pitch_deg)
        half_fov_v = math.radians(camera.fov_deg) * 0.5
        half_fov_h = math.atan(math.tan(half_fov_v) * aspect_ratio)
        half_fov_min = min(half_fov_v, half_fov_h)
        radius_m = max(diagonal_m * 0.5, DEPTH_MIN_SPAN_M)
        distance_m = radius_m / math.sin(half_fov_min) * BIRD_FRAMING_MARGIN_RATIO
        eye_m = center_m - forward * distance_m
        target_m = center_m
    else:
        if camera.room is None:
            raise BaseRenderError(f"room 机位没有指定房间：camera_id={camera_id}")
        centroid_xy_m, floor_z_m = _room_floor_anchor_m(scene, camera.room)
        eye_m = np.array(
            [centroid_xy_m[0], centroid_xy_m[1], floor_z_m + camera.eye_height_m],
            dtype=np.float64,
        )
        forward = _yaw_pitch_direction(camera.yaw_deg, ROOM_PITCH_DEG)
        target_m = eye_m + forward * ROOM_TARGET_DISTANCE_M

    return CameraPose(
        camera_id=camera.id,
        kind=camera.kind,
        eye_m=(float(eye_m[0]), float(eye_m[1]), float(eye_m[2])),
        target_m=(float(target_m[0]), float(target_m[1]), float(target_m[2])),
        up_hint_xyz=(0.0, 0.0, 1.0),
        fov_deg=camera.fov_deg,
        near_clip_m=NEAR_CLIP_M,
        far_clip_m=far_clip_m,
    )


def render_base_views(
    scene: ScenePackage,
    camera_id: str,
    width_px: int = 1024,
    height_px: int = 768,
) -> BaseRenderViews:
    """一份场景包 + 一个机位 → 四路图 + 遮罩索引表 + 自证数。

    零模型调用、无随机、无时间戳：**同一份场景包渲两次，四张 PNG 逐字节相同**
    （同 render2d 母版那条口径；测试直接断字节相等）。
    """
    if width_px <= 0 or height_px <= 0:
        raise BaseRenderError(f"画幅必须为正：width_px={width_px} height_px={height_px}")
    mesh_count = len(scene.meshes)
    if mesh_count + 1 > MASK_MAX_INDEX + 1:
        raise BaseRenderError(f"网格数超出 16 位遮罩索引上限：meshes={mesh_count}")

    aspect_ratio = width_px / height_px
    pose = resolve_camera_pose(scene, camera_id, aspect_ratio)
    triangles_m, tri_mesh_ids = _flatten_meshes(scene, _rendered_mesh_indices(scene, pose.kind))
    palette_ratio = _mesh_palette_ratio(scene)

    view_matrix = look_at_matrix(pose.eye_m, pose.target_m, pose.up_hint_xyz)
    proj_matrix = perspective_matrix(pose.fov_deg, aspect_ratio, pose.near_clip_m, pose.far_clip_m)
    buffers = rasterize(
        triangles_m,
        tri_mesh_ids,
        view_matrix,
        proj_matrix,
        width_px,
        height_px,
        pose.near_clip_m,
    )
    screen = _screen_geometry(buffers, view_matrix, pose, aspect_ratio)

    near_m, far_m, depth_png = _encode_depth_png(buffers)
    mask_png, mask_index = _encode_mask_png(buffers, scene)
    return BaseRenderViews(
        geometry_png=_encode_geometry_png(buffers, screen, palette_ratio),
        depth_png=depth_png,
        line_png=_encode_line_png(buffers, screen),
        mask_png=mask_png,
        width_px=width_px,
        height_px=height_px,
        camera_id=pose.camera_id,
        mask_index=mask_index,
        covered_pixel_ratio=buffers.covered_pixel_ratio,
        near_m=near_m,
        far_m=far_m,
    )


# ---------------------------------------------------------------------------
# 场景包 → 光栅入参
# ---------------------------------------------------------------------------


def _find_camera(scene: ScenePackage, camera_id: str) -> CameraSpec:
    for camera in scene.cameras:
        if camera.id == camera_id:
            return camera
    known = ", ".join(camera.id for camera in scene.cameras) or "（场景包里一台相机都没有）"
    raise BaseRenderError(f"场景包里没有这台相机：camera_id={camera_id}；已有：{known}")


def _rendered_mesh_indices(scene: ScenePackage, camera_kind: CameraKind) -> list[int]:
    """这台相机这一次要画哪些网格，返回它们在 ``scene.meshes`` 里的下标。

    **剔除的口径只写在这一处**：取景（包围盒）与绘制（光栅）都问它，两边就不会各判一次
    而判出不一样的答案。今天只有一条规则——bird 剔 :data:`BIRD_HIDDEN_SEMANTICS`；
    room 一块不剔（站在屋里本来就该看得见天花）。

    返回的是**下标**不是网格：遮罩索引与调色板都按下标对齐，剔除只能让某些下标缺席，
    绝不能让剩下的重新编号。
    """
    if camera_kind != "bird":
        return list(range(len(scene.meshes)))
    return [
        index
        for index, mesh in enumerate(scene.meshes)
        if mesh.semantic not in BIRD_HIDDEN_SEMANTICS
    ]


def _scene_bounds_m(
    scene: ScenePackage, mesh_indices: list[int]
) -> tuple[Float64Array, Float64Array]:
    """按网格顶点现算包围盒。一个顶点都没有就抛错——框不出画面的场景不该渲出一张黑图。"""
    lows: list[Float64Array] = []
    highs: list[Float64Array] = []
    for index in mesh_indices:
        mesh = scene.meshes[index]
        if not mesh.vertices:
            continue
        verts = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
        lows.append(verts.min(axis=0))
        highs.append(verts.max(axis=0))
    if not lows:
        raise BaseRenderError(
            f"这台相机要画的网格里一个顶点都没有，取不出机位：revision_id={scene.revision_id}"
        )
    return np.min(np.stack(lows), axis=0), np.max(np.stack(highs), axis=0)


def _room_floor_anchor_m(scene: ScenePackage, room: str) -> tuple[tuple[float, float], float]:
    """该房间地板的**面积加权**质心 (x, y) 与标高 z。

    面积加权而不是顶点平均：地板由若干矩形块三角化而来，块小的地方顶点密，顶点平均会把
    机位拽到细碎那一侧。面积加权只跟形状有关，跟怎么切三角形无关——换个三角化方式机位不动。
    """
    weighted_sum = np.zeros(3, dtype=np.float64)
    total_area_m2 = 0.0
    for mesh in scene.meshes:
        if mesh.semantic != "floor" or mesh.room != room or not mesh.triangles:
            continue
        verts = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
        index = np.asarray(mesh.triangles, dtype=np.int64).reshape(-1, 3)
        tris = verts[index]
        cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        areas_m2 = 0.5 * np.linalg.norm(cross, axis=1)
        centroids = tris.mean(axis=1)
        weighted_sum += (centroids * areas_m2[:, None]).sum(axis=0)
        total_area_m2 += float(areas_m2.sum())
    if total_area_m2 <= 0.0:
        rooms = sorted({m.room for m in scene.meshes if m.semantic == "floor" and m.room})
        known = ", ".join(rooms) or "（场景包里没有任何地板网格）"
        raise BaseRenderError(f"这间房没有地板，站不进去：room={room}；有地板的房间：{known}")
    anchor = weighted_sum / total_area_m2
    return (float(anchor[0]), float(anchor[1])), float(anchor[2])


def _yaw_pitch_direction(yaw_deg: float, pitch_deg: float) -> Float64Array:
    """朝向角 → 单位视线方向（世界系）。

    yaw 的口径与 :class:`~render3d_worker.models.FurnishingPlacement` **逐字一致**：0 度朝
    +y（户型图上的下方），俯视看去逆时针为正（0° → +y，90° → +x）。同一份包里家具朝向与
    相机朝向用两套角度约定，是最难查的一类错，所以只写这一处。
    pitch 绕水平轴，负值低头（``CameraSpec.pitch_deg`` 默认 -30 即俯视）。
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cos_pitch = math.cos(pitch)
    return np.array(
        [cos_pitch * math.sin(yaw), cos_pitch * math.cos(yaw), math.sin(pitch)],
        dtype=np.float64,
    )


def _flatten_meshes(
    scene: ScenePackage, mesh_indices: list[int]
) -> tuple[Float64Array, npt.NDArray[np.int32]]:
    """要画的网格摊平成 (N, 3, 3) 三角形 + (N,) 网格序号。序号 = 网格在场景包里的下标。

    序号就是遮罩索引减一，两处不再各排一次序：光栅只认整数，遮罩索引表回指网格身份。
    被 :func:`_rendered_mesh_indices` 剔掉的网格连三角形都不进来——不是画完再盖住，
    是根本不参与 z-test，所以它背后的东西该多深就是多深。
    """
    tri_blocks: list[Float64Array] = []
    id_blocks: list[npt.NDArray[np.int32]] = []
    for mesh_index in mesh_indices:
        mesh = scene.meshes[mesh_index]
        if not mesh.triangles or not mesh.vertices:
            continue
        verts = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
        index = np.asarray(mesh.triangles, dtype=np.int64).reshape(-1, 3)
        if index.size and (int(index.max()) >= verts.shape[0] or int(index.min()) < 0):
            raise BaseRenderError(
                f"网格的三角形索引越界：mesh_id={mesh.id} vertices={verts.shape[0]}"
            )
        tri_blocks.append(verts[index])
        id_blocks.append(np.full(index.shape[0], mesh_index, dtype=np.int32))
    if not tri_blocks:
        return np.zeros((0, 3, 3), dtype=np.float64), np.zeros(0, dtype=np.int32)
    return np.concatenate(tri_blocks, axis=0), np.concatenate(id_blocks, axis=0)


def _mesh_palette_ratio(scene: ScenePackage) -> Float64Array:
    """(1 + 网格数, 3) 调色板，0 号是背景色，往后按网格顺序放 ``base_color_hex``。

    背景占 0 号是为了让 ``id_buffer + 1`` 直接当下标用——省掉"未命中"那一路分支，也让
    调色板与遮罩索引共用同一套编号（图上索引 k 的颜色就是 palette[k]）。

    材质查不到不给兜底色：那说明场景包自己对不上（网格引用了不存在的材质），编一个灰色
    只会让错误一路走到出图。
    """
    by_id = {material.id: material for material in scene.materials}
    palette = np.zeros((len(scene.meshes) + 1, 3), dtype=np.float64)
    palette[0] = np.array(GEOMETRY_BACKGROUND_RGB_U8, dtype=np.float64) / 255.0
    for mesh_index, mesh in enumerate(scene.meshes):
        material = by_id.get(mesh.material_id)
        if material is None:
            known = ", ".join(sorted(by_id)) or "（场景包里没有材质）"
            raise BaseRenderError(
                f"网格引用了场景包里没有的材质：mesh_id={mesh.id} "
                f"material_id={mesh.material_id}；已有：{known}"
            )
        palette[mesh_index + 1] = _hex_to_rgb_ratio(material.base_color_hex, material.id)
    return palette


def _hex_to_rgb_ratio(base_color_hex: str, material_id: str) -> Float64Array:
    """``#RRGGBB`` → 0~1 的 RGB。写不对就抛错，不猜——颜色错了整张几何图就是错的。"""
    text = base_color_hex.strip().removeprefix("#")
    if len(text) != 6:
        raise BaseRenderError(
            f"材质颜色不是 #RRGGBB：material_id={material_id} hex={base_color_hex}"
        )
    try:
        value = int(text, 16)
    except ValueError as exc:
        raise BaseRenderError(
            f"材质颜色不是合法十六进制：material_id={material_id} hex={base_color_hex}"
        ) from exc
    return (
        np.array([(value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF], dtype=np.float64)
        / 255.0
    )


# ---------------------------------------------------------------------------
# 逐像素的视空间量
# ---------------------------------------------------------------------------


def _screen_geometry(
    buffers: RasterBuffers,
    view_matrix: Float64Array,
    pose: CameraPose,
    aspect_ratio: float,
) -> _ScreenGeometry:
    """每个像素的视线方向、视空间位置、视空间法向与朝向符号。

    位置能从深度反算出来，是因为深度存的是"沿相机前向的米数"而射线的 z 恒为 -1：
    ``position_view = ray_view × depth``，一次乘法，不必在光栅里额外存一张位置缓冲。
    """
    height_px, width_px = buffers.id_buffer.shape
    tan_v = math.tan(math.radians(pose.fov_deg) * 0.5)
    tan_h = tan_v * aspect_ratio
    x_ndc = (np.arange(width_px, dtype=np.float64) + 0.5) * (2.0 / width_px) - 1.0
    y_ndc = 1.0 - (np.arange(height_px, dtype=np.float64) + 0.5) * (2.0 / height_px)

    ray_view = np.empty((height_px, width_px, 3), dtype=np.float64)
    ray_view[..., 0] = (x_ndc * tan_h)[None, :]
    ray_view[..., 1] = (y_ndc * tan_v)[:, None]
    ray_view[..., 2] = -1.0

    hit = buffers.hit_mask
    depth_m = np.where(hit, buffers.depth_m, 0.0).astype(np.float64)
    position_view = ray_view * depth_m[..., None]

    normal_world = buffers.normal_unit_xyz.astype(np.float64)
    normal_view = np.einsum("ij,hwj->hwi", view_matrix[:3, :3], normal_world)
    toward = np.einsum("hwi,hwi->hw", normal_view, ray_view)
    # 视线与法向同向说明看到的是背面：翻过来。上游绕序不保证一致，所以朝向在这儿定，
    # 不在光栅里剔除（剔错了室内会直接看穿墙）。
    facing_sign = np.where(toward > 0.0, -1.0, 1.0)
    normal_view = normal_view * facing_sign[..., None]
    return _ScreenGeometry(
        ray_view_xyz=ray_view,
        normal_view_xyz=normal_view,
        position_view_m=position_view,
        facing_sign=facing_sign,
    )


# ---------------------------------------------------------------------------
# 四路编码
# ---------------------------------------------------------------------------


def _encode_geometry_png(
    buffers: RasterBuffers, screen: _ScreenGeometry, palette_ratio: Float64Array
) -> bytes:
    """几何路：材质底色 × 一盏写死方向的兰伯特明暗，8 位 RGB PNG。

    只要漫反射、不要高光/阴影/环境光遮蔽：这张图的任务是"让形体读得出来"，多一项都只是
    给写实化那一步添噪声（真实光影归 realism-pass）。
    """
    light = np.array(LIGHT_FROM_DIR_XYZ, dtype=np.float64)
    light = light / float(np.linalg.norm(light))
    normal_world = buffers.normal_unit_xyz.astype(np.float64) * screen.facing_sign[..., None]
    n_dot_l = np.clip(np.einsum("hwi,i->hw", normal_world, light), 0.0, 1.0)
    shade_ratio = AMBIENT_RATIO + (1.0 - AMBIENT_RATIO) * n_dot_l

    color_ratio = palette_ratio[buffers.id_buffer + 1] * shade_ratio[..., None]
    background = np.array(GEOMETRY_BACKGROUND_RGB_U8, dtype=np.float64) / 255.0
    color_ratio = np.where(buffers.hit_mask[..., None], color_ratio, background)
    rgb_u8 = np.rint(np.clip(color_ratio, 0.0, 1.0) * 255.0).astype(np.uint8)
    return _encode_png(Image.fromarray(rgb_u8, mode="RGB"))


def _encode_depth_png(buffers: RasterBuffers) -> tuple[float, float, bytes]:
    """深度路：**近处亮、远处暗**的 16 位灰度 PNG，返回 ``(near_m, far_m, png)``。

    近亮远暗的理由：这张图是给写实化当深度条件用的，depth-anything / MiDaS 一系的深度
    ControlNet 通行的就是近白远黑，跟着它下游不用再反相；而且"近＝信号强"符合直觉——
    画面主体（贴近相机的家具与墙面）落在高位，量化误差先吃在无关紧要的远景上。

    ``near_m``/``far_m`` 取的是**这一帧实际命中的最近/最远深度**（不是裁剪面）：把 16 位
    全部铺在真正用到的那段量程上，精度最高。代价是两端随机位变，所以它们必须随
    :class:`~render3d_worker.models.BaseRenderViews` 带出去——没有这两个数，图只是相对
    明暗，还原不回米。

    还原公式（下游照抄）::

        v = png[y, x]                       # uint16
        有几何 = v >= 1
        brightness = (v - 1) / 65534.0
        depth_m = near_m + (1 - brightness) * (far_m - near_m)
    """
    hit = buffers.hit_mask
    height_px, width_px = buffers.id_buffer.shape
    if not bool(hit.any()):
        # 一个像素都没打到：两端给 0，图全黑。自证数 covered_pixel_ratio 会是 0，
        # 由调用方按数据判失败（**不在这儿设死阈值**，《纪律·阈值有数据才定》）。
        zeros = np.zeros((height_px, width_px), dtype=np.uint16)
        return 0.0, 0.0, _encode_png(Image.fromarray(zeros))

    depth_m = buffers.depth_m.astype(np.float64)
    near_m = float(depth_m[hit].min())
    far_m = float(depth_m[hit].max())
    if far_m - near_m < DEPTH_MIN_SPAN_M:
        far_m = near_m + DEPTH_MIN_SPAN_M

    normalized = np.clip((depth_m - near_m) / (far_m - near_m), 0.0, 1.0)
    span_u16 = float(DEPTH_MAX_U16 - DEPTH_MIN_U16)
    value = DEPTH_MIN_U16 + np.rint((1.0 - normalized) * span_u16)
    depth_u16 = np.where(hit, value, float(DEPTH_BACKGROUND_U16)).astype(np.uint16)
    return near_m, far_m, _encode_png(Image.fromarray(depth_u16))


def _encode_line_png(buffers: RasterBuffers, screen: _ScreenGeometry) -> bytes:
    """线稿路：黑底白线的 8 位灰度 PNG，线**从几何缓冲直接取**，不做图像滤波。

    两条判据，都是确定性的几何事实（这是与 canny 那类"猜边缘"的关键区别——同一份场景包
    出来的线永远是同一批像素，不随对比度、不随滤波核变）：

    1. **网格边界**：``id_buffer`` 相邻不等即成线。轮廓（几何↔背景）与两块网格的交界一次
       全包了；
    2. **深度不连续**：按当前像素所在**平面**（法向已知）外推到邻居那条视线上，算"这张面
       继续下去应该是多深"，与邻居实测深度比。残差超过
       :data:`LINE_DEPTH_TOLERANCE_RATIO` 即断开。平面外推顺带把**同一块网格内部的折角**
       也画上了（箱体家具的竖棱、L 形墙的转角），不用再单加一条法向判据。

    线标在比较对里**下标小的那一侧**，线宽 1 像素：条件图上细线比粗线好，粗了会把靠得近的
    结构糊成一团。
    """
    id_buffer = buffers.id_buffer
    hit = buffers.hit_mask
    depth_m = buffers.depth_m.astype(np.float64)
    edge = np.zeros(id_buffer.shape, dtype=bool)

    edge[:, :-1] |= id_buffer[:, :-1] != id_buffer[:, 1:]
    edge[:-1, :] |= id_buffer[:-1, :] != id_buffer[1:, :]

    edge[:, :-1] |= _depth_break(
        screen.normal_view_xyz[:, :-1],
        screen.position_view_m[:, :-1],
        screen.ray_view_xyz[:, 1:],
        depth_m[:, 1:],
        hit[:, :-1] & hit[:, 1:],
    )
    edge[:-1, :] |= _depth_break(
        screen.normal_view_xyz[:-1, :],
        screen.position_view_m[:-1, :],
        screen.ray_view_xyz[1:, :],
        depth_m[1:, :],
        hit[:-1, :] & hit[1:, :],
    )

    line_u8 = np.where(edge, LINE_FOREGROUND_U8, LINE_BACKGROUND_U8).astype(np.uint8)
    return _encode_png(Image.fromarray(line_u8, mode="L"))


def _depth_break(
    normal_view_xyz: Float64Array,
    position_view_m: Float64Array,
    neighbour_ray_xyz: Float64Array,
    neighbour_depth_m: Float64Array,
    both_hit: npt.NDArray[np.bool_],
) -> npt.NDArray[np.bool_]:
    """当前像素所在平面外推到邻居视线上，预测深度对不上就是断开。

    ``分母 = n·r`` 趋零意味着这张面正好被视线擦着看（掠射），面在那儿本来就要结束——
    预测值发散，判为断开是对的，不是数值意外。
    """
    denom = np.einsum("hwi,hwi->hw", normal_view_xyz, neighbour_ray_xyz)
    numer = np.einsum("hwi,hwi->hw", normal_view_xyz, position_view_m)
    usable = np.abs(denom) > 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        predicted_m = np.where(usable, numer / np.where(usable, denom, 1.0), 0.0)
    residual_m = np.abs(neighbour_depth_m - predicted_m)
    continuous = (
        usable
        & (predicted_m > 0.0)
        & np.isfinite(residual_m)
        & (residual_m <= LINE_DEPTH_TOLERANCE_RATIO * neighbour_depth_m)
    )
    return np.asarray(both_hit & ~continuous, dtype=np.bool_)


def _encode_mask_png(buffers: RasterBuffers, scene: ScenePackage) -> tuple[bytes, list[MaskEntry]]:
    """遮罩路：16 位灰度索引 PNG（索引 0 ＝ 背景，网格 k 占索引 k+1）+ 索引表。

    索引表**只收真出现在图上的网格**：表里的 index 集合与图里的非零值集合相等、
    ``pixel_count`` 之和等于覆盖像素数——这条一致性是遮罩这一路的自证（测试直接断它）。
    被完全挡住的网格不进表，正是因为"这张图上有什么"才是下游要问的问题。
    """
    mask_u16 = np.where(buffers.hit_mask, buffers.id_buffer + 1, 0).astype(np.uint16)
    counts = np.bincount(mask_u16.reshape(-1).astype(np.int64), minlength=len(scene.meshes) + 1)
    entries: list[MaskEntry] = []
    for mesh_index, mesh in enumerate(scene.meshes):
        pixel_count = int(counts[mesh_index + 1])
        if pixel_count == 0:
            continue
        entries.append(
            MaskEntry(
                index=mesh_index + 1,
                mesh_id=mesh.id,
                semantic=mesh.semantic,
                room=mesh.room,
                pixel_count=pixel_count,
            )
        )
    return _encode_png(Image.fromarray(mask_u16)), entries


def _encode_png(image: Image.Image) -> bytes:
    """统一的 PNG 落字节口径：参数全部显式写死，逐字节可复现靠的就是这一处不飘。"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getvalue()


__all__ = [
    "BaseRenderError",
    "CameraPose",
    "render_base_views",
    "resolve_camera_pose",
]
