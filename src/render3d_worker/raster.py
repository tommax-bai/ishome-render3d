"""软光栅核心：吃三角网格与相机矩阵，吐几何缓冲，**不认识业务**。

本模块一个字都不知道什么是房间、什么是家具、什么叫遮罩索引——它只有三角形、矩阵、
缓冲区。把这条界划清的理由：底渲的口径（四路怎么编码、相机怎么摆）会随下游写实化的
需要变，光栅的口径（怎么裁近平面、深度相等谁赢）不会；混在一起改一次动两处。

**为什么不用 GPU / 三维引擎**：四路输出是给 realism-pass 当条件图的几何缓冲，不是给人看
的写实图；而且要求"同一份场景包渲两次逐字节相同"，这条在纯 numpy 软光栅里是免费的，
在任何驱动/GPU 栈上都要额外证明。

**确定性怎么保证的**（三条，缺一条就不成立）：

1. 遍历顺序写死——按调用方给的三角形顺序，从 0 到 N-1，不并行、不排序、不走集合；
2. 深度相等时**先来的赢**（严格 ``<`` 才覆盖），所以共面三角形谁盖住谁只由输入顺序决定；
3. 全程无随机数、无时间戳、无浮点归约顺序不定的操作（不用 ``np.add.reduceat`` 那类）。

坐标系（与 :class:`~render3d_worker.models.Mesh` 一致）：世界系米制右手系，x 向右、
y 向里、z 向上；视空间沿用 OpenGL 惯例——相机在原点、**看向 -z**，所以"正深度"＝ ``-z``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Float64Array = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]
Int32Array = npt.NDArray[np.int32]

Vector3Like = npt.ArrayLike
"""三维向量的入参形态：长度 3 的序列或 ndarray。单位由参数名带（``_m`` 即米）。"""

MISS_MESH_INDEX: int = -1
"""``id_buffer`` 里"这个像素没打到任何几何"的值。取 -1 而不是 0：0 是合法的网格序号。"""

_DEGENERATE_AREA_PX2 = 1e-12
"""屏幕面积（像素平方）小于它就当退化三角形丢掉——它张不出一个像素，只会把重心坐标除爆。"""

_PARALLEL_EPS = 1e-12


@dataclass(frozen=True, eq=False)
class RasterBuffers:
    """一次光栅的全部几何缓冲。四路图都是从这三张缓冲算出来的，不再回头看三角形。

    ``eq=False``：里面是 ndarray，默认的 ``__eq__`` 会返回数组、一比就炸；缓冲区之间
    没有"相等"这个概念，要比就比具体那一张。
    """

    depth_m: Float32Array
    """(H, W) 沿相机前向的正深度（米）。**未命中为 ``np.inf``**——取 inf 不取 nan 的理由：
    z-test 是 ``candidate < existing``，inf 让"空像素永远输"这件事不需要额外分支。"""

    id_buffer: Int32Array
    """(H, W) 命中像素所属**网格的序号**（调用方给的 ``tri_mesh_ids``），未命中为
    :data:`MISS_MESH_INDEX`。"""

    normal_unit_xyz: Float32Array
    """(H, W, 3) 命中处三角形的**世界系单位法向**，未命中为 (0, 0, 0)。

    存世界系不存视空间：法向是几何的属性，换个机位不该重算；着色那一步要视空间时
    自己乘视矩阵的旋转块（正交，转置即逆）。法向**不做正反归一**——网格的绕序由上游
    决定，本模块不假设它一致，朝向留给用光的那一步按视线翻（见 base_render）。"""

    width_px: int
    height_px: int

    @property
    def hit_mask(self) -> npt.NDArray[np.bool_]:
        """(H, W) 哪些像素被几何盖住。以 ``id_buffer`` 为准而不是深度：判据只留一处。"""
        return self.id_buffer >= 0

    @property
    def covered_pixel_ratio(self) -> float:
        """被几何盖住的像素比例——底渲的自证数。**不设死阈值**（《纪律·阈值有数据才定》）。"""
        total_px = self.width_px * self.height_px
        if total_px == 0:
            return 0.0
        return float(np.count_nonzero(self.hit_mask)) / float(total_px)


def look_at_matrix(
    eye_m: Vector3Like,
    target_m: Vector3Like,
    up_hint_xyz: Vector3Like = (0.0, 0.0, 1.0),
) -> Float64Array:
    """视图矩阵（世界系 → 视空间），OpenGL 惯例：相机落在原点、看向 -z、y 朝上。

    ``up_hint_xyz`` 只是**提示**不是结果：真正的上方向由 ``forward × right`` 反求，
    所以给一个不严格垂直于视线的粗略上方向也不会把画面拧歪。

    默认提示取世界 z（本项目的竖直轴）。视线与它平行时（正俯视，pitch = ±90）叉积退化，
    改用 **-y** 当提示——理由：那一刻画面就是一张户型图，让平面图的"上方"（-y，因为
    y 向里/在图上向下）落在画面上方，出来的俯视图与用户看惯的户型图朝向一致。
    """
    eye = np.asarray(eye_m, dtype=np.float64).reshape(3)
    target = np.asarray(target_m, dtype=np.float64).reshape(3)
    forward = target - eye
    forward_len = float(np.linalg.norm(forward))
    if forward_len < _PARALLEL_EPS:
        raise ValueError(
            f"相机位置与目标点重合，定不出视线：eye_m={eye.tolist()} target_m={target.tolist()}"
        )
    forward = forward / forward_len

    up_hint = np.asarray(up_hint_xyz, dtype=np.float64).reshape(3)
    right = np.cross(forward, up_hint)
    if float(np.linalg.norm(right)) < 1e-9:
        right = np.cross(forward, np.array([0.0, -1.0, 0.0]))
    right_len = float(np.linalg.norm(right))
    if right_len < _PARALLEL_EPS:
        raise ValueError(f"上方向与视线平行且备用轴也退化：up_hint_xyz={up_hint.tolist()}")
    right = right / right_len
    true_up = np.cross(right, forward)

    view = np.eye(4, dtype=np.float64)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = -forward
    view[0, 3] = -float(np.dot(right, eye))
    view[1, 3] = -float(np.dot(true_up, eye))
    view[2, 3] = float(np.dot(forward, eye))
    return view


def perspective_matrix(
    fov_deg: float, aspect_ratio: float, near_m: float, far_m: float
) -> Float64Array:
    """透视投影矩阵（视空间 → 裁剪空间），OpenGL 惯例：``w = -z``，即 w 就是正深度（米）。

    ``fov_deg`` 是**竖直**张角。横向张角由 ``aspect_ratio`` 撑出来——两个方向除的不是同一个
    数，取景要"框得住"时必须按两者里小的那一个算（见 base_render 的 bird 机位）。

    近平面只影响裁剪，**不影响深度精度**：本模块的深度是视空间米数（``-z``）直接透视校正
    插值出来的，不走 NDC 的 z，所以没有"近平面压死远处精度"那套 z-fighting 账。
    """
    if not 0.0 < fov_deg < 180.0:
        raise ValueError(f"竖直张角要落在 (0, 180) 度：fov_deg={fov_deg}")
    if aspect_ratio <= 0.0:
        raise ValueError(f"宽高比必须为正：aspect_ratio={aspect_ratio}")
    if not 0.0 < near_m < far_m:
        raise ValueError(f"裁剪面要满足 0 < near < far：near_m={near_m} far_m={far_m}")

    focal_ratio = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = focal_ratio / aspect_ratio
    proj[1, 1] = focal_ratio
    proj[2, 2] = (far_m + near_m) / (near_m - far_m)
    proj[2, 3] = 2.0 * far_m * near_m / (near_m - far_m)
    proj[3, 2] = -1.0
    return proj


def rasterize(
    triangles_m: npt.ArrayLike,
    tri_mesh_ids: npt.ArrayLike,
    view_matrix: Float64Array,
    proj_matrix: Float64Array,
    width_px: int,
    height_px: int,
    near_m: float,
) -> RasterBuffers:
    """把三角形画进 z-buffer。``triangles_m`` 形状 (N, 3, 3)，``tri_mesh_ids`` 形状 (N,)。

    ``near_m`` 是显式参数而不是从 ``proj_matrix`` 反解出来的：反解要假定矩阵长成标准
    OpenGL 那样，一旦调用方换了投影就悄悄算错。宁可多传一个数。

    **近裁剪不是可选项**：相机站在室内时墙一定会穿过近平面，不裁就会有顶点落到 w ≤ 0,
    透视除法把它甩到画面另一侧——屏幕上出现一块翻面的巨大三角形盖住整幅图。这是室内
    软光栅最容易也最难查的一个错，所以裁剪写在最里层、每个三角形都过。

    远平面**不裁**：深度是视空间米数，远处三角形照常光栅、照常写深度，没有必要为了一个
    只影响 NDC z 的常数把几何切掉。
    """
    if width_px <= 0 or height_px <= 0:
        raise ValueError(f"画幅必须为正：width_px={width_px} height_px={height_px}")
    if near_m <= 0.0:
        raise ValueError(f"近裁剪面必须为正：near_m={near_m}")

    tris_m = np.asarray(triangles_m, dtype=np.float64)
    if tris_m.size == 0:
        tris_m = tris_m.reshape(0, 3, 3)
    if tris_m.ndim != 3 or tris_m.shape[1:] != (3, 3):
        raise ValueError(f"triangles_m 形状必须是 (N, 3, 3)，收到 {tris_m.shape}")
    mesh_ids = np.asarray(tri_mesh_ids, dtype=np.int32).reshape(-1)
    if mesh_ids.shape[0] != tris_m.shape[0]:
        raise ValueError(
            f"三角形数与网格序号数对不上：triangles={tris_m.shape[0]} ids={mesh_ids.shape[0]}"
        )

    depth_m = np.full((height_px, width_px), np.inf, dtype=np.float32)
    id_buffer = np.full((height_px, width_px), MISS_MESH_INDEX, dtype=np.int32)
    normal_unit_xyz = np.zeros((height_px, width_px, 3), dtype=np.float32)

    if tris_m.shape[0] > 0:
        verts_view = _to_view_space(tris_m, view_matrix)
        normals_world = _triangle_unit_normals(tris_m)
        finite_tri = np.isfinite(tris_m).all(axis=(1, 2))
        for tri_index in range(tris_m.shape[0]):
            if not bool(finite_tri[tri_index]):
                continue
            mesh_index = int(mesh_ids[tri_index])
            normal_u32 = normals_world[tri_index].astype(np.float32)
            for piece_view in _clip_triangle_near(verts_view[tri_index], near_m):
                _fill_triangle(
                    piece_view,
                    mesh_index,
                    normal_u32,
                    proj_matrix,
                    depth_m,
                    id_buffer,
                    normal_unit_xyz,
                )

    return RasterBuffers(
        depth_m=depth_m,
        id_buffer=id_buffer,
        normal_unit_xyz=normal_unit_xyz,
        width_px=width_px,
        height_px=height_px,
    )


def _to_view_space(tris_m: Float64Array, view_matrix: Float64Array) -> Float64Array:
    """(N, 3, 3) 世界坐标 → (N, 3, 3) 视空间坐标。一次矩阵乘打完，不在循环里逐点乘。"""
    flat = tris_m.reshape(-1, 3)
    homogeneous = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=np.float64)], axis=1)
    view_flat: Float64Array = homogeneous @ view_matrix.T
    return view_flat[:, :3].reshape(tris_m.shape)


def _triangle_unit_normals(tris_m: Float64Array) -> Float64Array:
    """(N, 3) 世界系单位法向；退化三角形（零面积）给 (0, 0, 0)，着色时只吃到环境光。"""
    edge_a = tris_m[:, 1] - tris_m[:, 0]
    edge_b = tris_m[:, 2] - tris_m[:, 0]
    raw: Float64Array = np.cross(edge_a, edge_b)
    length = np.linalg.norm(raw, axis=1, keepdims=True)
    safe_length = np.where(length > 0.0, length, 1.0)
    return raw / safe_length


def _clip_triangle_near(tri_view: Float64Array, near_m: float) -> list[Float64Array]:
    """按近平面裁一个视空间三角形，返回 0~2 个三角形（Sutherland–Hodgman 后扇形三角化）。

    保留的是 ``-z >= near_m`` 那一侧。顶点顺序与新顶点的插入位置全部写死（按边 0-1、1-2、
    2-0 走一圈），所以同一个输入永远切出同一组三角形——这是逐字节可复现的前提之一。
    """
    signed_dist = -tri_view[:, 2] - near_m
    inside = signed_dist >= 0.0
    inside_count = int(np.count_nonzero(inside))
    if inside_count == 3:
        return [tri_view]
    if inside_count == 0:
        return []

    polygon: list[Float64Array] = []
    for i in range(3):
        j = (i + 1) % 3
        if bool(inside[i]):
            polygon.append(tri_view[i])
        if bool(inside[i]) != bool(inside[j]):
            denom = signed_dist[i] - signed_dist[j]
            ratio = 0.0 if denom == 0.0 else float(signed_dist[i] / denom)
            polygon.append(tri_view[i] + ratio * (tri_view[j] - tri_view[i]))
    if len(polygon) < 3:
        return []
    return [np.stack([polygon[0], polygon[k], polygon[k + 1]]) for k in range(1, len(polygon) - 1)]


def _fill_triangle(
    tri_view: Float64Array,
    mesh_index: int,
    normal_unit_xyz_tri: Float32Array,
    proj_matrix: Float64Array,
    depth_m: Float32Array,
    id_buffer: Int32Array,
    normal_unit_xyz: Float32Array,
) -> None:
    """把一个**已过近裁剪**的视空间三角形填进缓冲：包围盒 + 重心坐标 + z-test。

    深度用**透视校正**插值：屏幕上线性的是 1/w，不是深度本身。直接拿重心坐标插深度，
    地板这种大斜面会中间鼓起来一大截——线稿那一路靠深度的平面外推找边，插错了满屏假线。
    """
    height_px, width_px = id_buffer.shape
    homogeneous = np.concatenate([tri_view, np.ones((3, 1), dtype=np.float64)], axis=1)
    clip = homogeneous @ proj_matrix.T
    w_m = clip[:, 3]
    if not bool(np.all(w_m > 0.0)) or not bool(np.isfinite(clip).all()):
        return

    ndc = clip[:, :3] / w_m[:, None]
    x_px = (ndc[:, 0] + 1.0) * 0.5 * width_px
    y_px = (1.0 - ndc[:, 1]) * 0.5 * height_px
    depth_vert_m = -tri_view[:, 2]
    if not bool(np.isfinite(x_px).all()) or not bool(np.isfinite(y_px).all()):
        return

    # 像素中心在 (i + 0.5, j + 0.5)，故包围盒按"中心落在三角形跨度内"取整并夹回画幅。
    x_from = int(max(0.0, math.floor(float(x_px.min()) - 0.5)))
    x_to = int(min(float(width_px - 1), math.ceil(float(x_px.max()) - 0.5)))
    y_from = int(max(0.0, math.floor(float(y_px.min()) - 0.5)))
    y_to = int(min(float(height_px - 1), math.ceil(float(y_px.max()) - 0.5)))
    if x_to < x_from or y_to < y_from:
        return

    det = (y_px[1] - y_px[2]) * (x_px[0] - x_px[2]) + (x_px[2] - x_px[1]) * (y_px[0] - y_px[2])
    if abs(det) < _DEGENERATE_AREA_PX2:
        return

    center_x = np.arange(x_from, x_to + 1, dtype=np.float64)[None, :] + 0.5
    center_y = np.arange(y_from, y_to + 1, dtype=np.float64)[:, None] + 0.5
    lam0 = (
        (y_px[1] - y_px[2]) * (center_x - x_px[2]) + (x_px[2] - x_px[1]) * (center_y - y_px[2])
    ) / det
    lam1 = (
        (y_px[2] - y_px[0]) * (center_x - x_px[2]) + (x_px[0] - x_px[2]) * (center_y - y_px[2])
    ) / det
    lam2 = 1.0 - lam0 - lam1
    # 不做背面剔除：上游网格的绕序不保证一致，剔错了室内会直接看穿墙。两面都画，
    # 朝向问题留到着色那一步按视线翻法向（正确性只依赖 z-buffer，不依赖绕序）。
    inside = (lam0 >= 0.0) & (lam1 >= 0.0) & (lam2 >= 0.0)
    if not bool(inside.any()):
        return

    inv_w = lam0 / w_m[0] + lam1 / w_m[1] + lam2 / w_m[2]
    depth_over_w = (
        lam0 * depth_vert_m[0] / w_m[0]
        + lam1 * depth_vert_m[1] / w_m[1]
        + lam2 * depth_vert_m[2] / w_m[2]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        candidate_m = (depth_over_w / inv_w).astype(np.float32)
    inside &= np.isfinite(candidate_m) & (candidate_m > 0.0)

    existing_m = depth_m[y_from : y_to + 1, x_from : x_to + 1]
    # 严格 `<`：深度相等时**先来的赢**。共面三角形谁盖住谁于是只由输入顺序决定，
    # 而输入顺序是场景包写死的——这条就是"渲两次逐字节相同"里最容易漏掉的一半。
    winners = inside & (candidate_m < existing_m)
    if not bool(winners.any()):
        return
    existing_m[winners] = candidate_m[winners]
    id_buffer[y_from : y_to + 1, x_from : x_to + 1][winners] = mesh_index
    normal_unit_xyz[y_from : y_to + 1, x_from : x_to + 1][winners] = normal_unit_xyz_tri
