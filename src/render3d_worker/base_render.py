"""底渲（activity ``base-render``）：一份场景包 + 一个机位 → 几何/深度/线稿/遮罩四路图。

**四路是给下一步当条件图的，不是给人看的成品图**——写实化（``realism-pass``）在 imagegen
那个仓走生成模型。所以这一步既不需要 GPU 也不需要三维引擎：四路全是几何缓冲，纯 numpy
软光栅就出得来（光栅本身在 :mod:`render3d_worker.raster`，本模块只管"编码成哪四张图"）。

四路分**三路数据、一路观感**，这是本模块最要紧的一条分界：

===== ================== ================== =============================================
路    图像形态            背景（没打到几何）  下游怎么读它
===== ================== ================== =============================================
深度  16 位灰度 PNG       0                  **当数据读**：值是米（1..65535 归一化，两端随包带出）
遮罩  16 位灰度索引 PNG   0                  **当数据读**：索引 ≡ 网格下标 +1，回指网格身份
线稿  8 位灰度 PNG        0（黑底白线）      **当数据读**：像素要么是几何事实边、要么不是
几何  8 位 RGB PNG        柔和中性底          **当图看**：给写实化当色彩/形体参考，也给人调试看
===== ================== ================== =============================================

**三路数据图一律"没有东西 ＝ 0"**，且**一律不做任何观感处理**——不抗锯齿、不降采样、
不加环境光遮蔽。理由是这三路的每个像素都要经得起被当成一个量去读：深度边缘上一平均，
就造出一个现实中不存在的深度；遮罩索引一平均，指到的是一块不存在的网格；线稿糊一下，
"这是不是一条真边"就答不出来了。**"这个像素有没有东西"由这三路答**，三路互相自查
（测试即断这条一致性）。

几何那一路**退出"背景＝0"这条约定**（观感提档 2026-09-01）：纯黑底把每一条轮廓都变成
最高对比的硬边，人眼看着扎、写实化那一步也没有必要吃这个对比。它是四路里唯一
"给人和给生成模型看"的一路，所以抗锯齿、环境光遮蔽、调色都只落在它身上
（见 :data:`GEOMETRY_SUPERSAMPLE_FACTOR` 起的那一节常量）。

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
    Float32Array,
    Float64Array,
    RasterBuffers,
    look_at_matrix,
    orthographic_matrix,
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

ROOM_EYE_WALL_MARGIN_M: float = 0.35
"""室内机位退到房间边缘时离墙留的余量（米）。墙有厚度、镜头也有物理宽度，贴着地板边界
站会让近裁剪面（:data:`NEAR_CLIP_M` 5 厘米）咬穿墙面；0.35 米是室内摄影"背几乎靠墙但不贴
墙"的量级，也留够近裁剪面十倍以上的冗余。"""

ROOM_AUTO_DIRECTION_MIN_M: float = 0.05
"""家具重心与地板质心的偏移小于它，就当**没有方向可言**（没有家具，或家具本身摆在正中央），
改用形状本身的兜底方向（:func:`_room_fallback_direction_xy`），不把一个几乎为零的向量硬
归一化成一个由浮点噪声决定的朝向。"""

ROOM_EYE_FURNISHING_MARGIN_M: float = 0.40
"""室内机位退景时离任何家具至少留这么多距离（米）。**只按地板边界退是不够的**——真跑
这份仓的拟真户型包时，卫生间那台机位退到了离洗手台只有 0.2 米的地方（近裁剪几乎糊在
台面上），小房间里家具能占掉大半张地板，只避墙不避家具，镜头照样怼进柜子里。
0.40 米比 :data:`ROOM_EYE_WALL_MARGIN_M` 更大：墙是一整片竖直平面，贴近了好歹还能拍全，
家具是个有厚度的体块，镜头贴上去大概率是拍它的一个面，糊得更明显。"""

_ROOM_BOUNDARY_SEARCH_STEPS: int = 256
"""退景边界沿射线走的采样步数。典型房间（对角线 5～10 米）下给到 2～4 厘米的分辨率，
比 :data:`ROOM_EYE_WALL_MARGIN_M` 精细一个数量级，不会让退墙距离被采样步长本身吃掉。"""

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


# ---------------------------------------------------------------------------
# 几何那一路的观感常量。**这一节的数只进几何图**——深度/线稿/遮罩三路一个都不读它们
# （分界见模块 docstring）。要调观感只动这一节，动完只有几何那一张 PNG 会变。
# ---------------------------------------------------------------------------

GEOMETRY_SUPERSAMPLE_FACTOR: int = 2
"""几何路按 N 倍分辨率再光栅一遍、然后 N×N 盒式降采样（超采样抗锯齿）。

**取 2 不取 3 是实测的取舍**，不是随手挑的档：光栅代价是平方级，同一份包同一台室内
机位实测 0.22 秒（1 倍）→ 0.84 秒（2 倍）→ 2.11 秒（3 倍）。2 倍把一条斜边上的灰阶
从 2 级抬到 5 级，阶梯感已经压掉大半；3 倍到 10 级，1024px 幅面上肉眼分不出来，
每台相机却要多花两秒半。

**只有几何这一路超采样**：另外三路是被当数据读的（深度是米、遮罩是索引、线稿是几何
事实边），降采样会在轮廓上造出现实中不存在的值。所以它们照旧只走 1 倍那一次光栅——
不是"降采样时把它们跳过"，是**根本不参与这次超采样光栅**。"""

KEY_LIGHT_FROM_DIR_XYZ: tuple[float, float, float] = (-0.40, -0.60, 0.70)
"""主光方向（世界系，指向光源那一侧；模块内归一化）。

写死不随相机转的理由有两条：一是确定性——光跟着机位转，同一户换个机位就没法把两张图
的明暗对上；二是这盏灯只为"让形体读得出来"服务，不是为了写实（写实归 realism-pass）。
方向取左·前·上（-x, -y, +z）：三维制图惯例的主光位；带 -y 分量（朝观察者那一侧）是为了
让正对相机的墙面别糊成一片黑。"""

KEY_LIGHT_RGB: tuple[float, float, float] = (0.62, 0.60, 0.56)
"""主光的辐照度，**线性光**、微暖。三个分量都小于 1 是有意的：加上环境光与补光之后，
正对主光的白色天花刚好落在 0.96 上下不溢出——溢出的部分是白花花一片，形体反而读不出来。"""

FILL_LIGHT_FROM_DIR_XYZ: tuple[float, float, float] = (0.55, 0.70, 0.25)
FILL_LIGHT_RGB: tuple[float, float, float] = (0.16, 0.18, 0.22)
"""补光：方向与主光大致相对、略微偏冷、强度约主光的四分之一。

没有它，背对主光的那半间房只剩环境光，一面墙上读不出任何朝向；有了它，背光面之间
仍有明暗差，转角还立得住。偏冷是为了跟暖主光拉开——两面墙的差别于是不只是明暗，
还有色温，比单纯压暗更容易看清哪面朝哪儿。"""

AMBIENT_SKY_RGB: tuple[float, float, float] = (0.36, 0.38, 0.42)
AMBIENT_GROUND_RGB: tuple[float, float, float] = (0.25, 0.235, 0.215)
"""半球环境光：朝上的面吃 sky（偏冷）、朝下的面吃 ground（偏暖），中间按法向 z 线性过渡。

**替掉的是原来那个 0.35 的常数环境光**。常数环境光的毛病是天花、地板、四面墙的暗部
全是同一个值，暗部里一点形体都没有——那正是"发闷"的来源。半球式只多一个数，就让
天花（朝下、吃地面反射的暖调）与地板（朝上、吃天光的冷调）在不打光时也分得开。
ground 给到接近 sky 的量是室内的实情：室内的"天空"其实是天花与墙的互相反射，
不是户外那种上亮下暗。"""

GEOMETRY_BACKGROUND_RGB_U8: tuple[int, int, int] = (46, 50, 56)
"""几何路的背景（**四路里只有这一路的背景不是 0**，理由见模块 docstring）。

取偏冷的深中性灰而不是纯黑：纯黑与浅色墙面之间是满对比的硬边，鸟瞰图上整户像被剪刀
剪下来贴上去的。抬到这一档，轮廓仍然一眼分得出来（与最暗的家具也差着好几档），
边缘却不再扎眼。**不取浅底**：浅底会跟墙面撞在一起，"这块是墙还是空"就分不出来了。"""

SSAO_RADIUS_M: float = 0.55
"""环境光遮蔽的采样半径（米）。按室内实物的尺度定：墙角的阴角、家具与地面的接缝、
柜门与柜体的缝，量级都在几十厘米以内。给大了整间房都被压暗（远处的墙会算成遮挡物），
给小了只在一两个像素宽的边上有效果，看着像描了一圈黑边。"""

SSAO_BIAS_M: float = 0.02
"""判遮挡时先把采样点往外推 2 厘米。不推的话同一张平面会自己遮自己（深度缓冲是离散的，
邻近像素的深度差落在采样点前后都有可能），出来的是满屏细密条纹。"""

SSAO_STRENGTH_RATIO: float = 0.85
"""被遮蔽到底的地方还剩 15% 环境光。这是**观感参数不是物理量**：条件图要的是"看得出
这儿有个角"，不是真实的暗部；但也不能是 1.0——压到全黑，墙角里的形体就跟着一起没了。

遮蔽**只乘在环境光那一份上，不乘主光**（见 :func:`_shade_linear` 分两份返回）：
主光是有方向的，它被挡没被挡由阴影那一路答（:data:`SHADOW_CASTER_SEMANTICS`），
两件事各归各的，才不会在一块受光的墙角上把同一件事算两遍。"""

SSAO_BLUR_HALF_PX: int = 2
"""遮蔽图的均值模糊半径（像素）。要它是因为采样核按 4×4 的角度铺开
（:data:`SSAO_ROTATION_TILE_RAD`），不抹一下就看得见那张 4 像素的网格。
半径 2（5×5 窗口）刚好盖住一整个铺块。"""

SSAO_KERNEL_TANGENT: tuple[tuple[float, float, float], ...] = (
    (+0.298442, +0.000000, +0.062230),
    (-0.220335, +0.201845, +0.112941),
    (+0.026739, -0.304683, +0.156900),
    (+0.193452, +0.252323, +0.204023),
    (-0.328153, -0.058046, +0.258132),
    (+0.294969, -0.187635, +0.321577),
    (-0.094589, +0.351867, +0.396100),
    (-0.172484, -0.332108, +0.483126),
    (+0.351933, +0.128526, +0.583878),
    (-0.331663, +0.136906, +0.699447),
    (+0.133098, -0.284422, +0.830830),
    (+0.061091, +0.194768, +0.978945),
)
"""切空间里的 12 个采样偏移（z 是法向那一侧的半球）。**写死成常量数组，不是随机生成的**
——本仓的口径是"同一份场景包渲两次逐字节相同"，采样核只要碰一次随机数这条就没了。

怎么构造出来的（要重算或改个数时照这个来）：第 i 个点取 z = sqrt((i+0.5)/12) 的余弦
加权半球分布（AO 的权重本来就是余弦，按余弦布点等于把样本花在有权重的地方），
方位角按黄金角 i·π·(3−√5) 铺开（相邻样本方位差最大），再乘一个 0.30→1.00 的长度渐变
让近处比远处密——遮挡主要发生在近处。12 个是够用与够快之间的取舍：8 个在大平面上
看得出结构性的斑，16 个跟 12 个已经看不出差别，代价却多三分之一。"""

SSAO_ROTATION_TILE_RAD: tuple[tuple[float, float, float, float], ...] = (
    (0.000000, 3.141593, 1.570796, 4.712389),
    (0.785398, 3.926991, 2.356194, 5.497787),
    (0.392699, 3.534292, 1.963495, 5.105088),
    (1.178097, 4.319690, 2.748894, 5.890486),
)
"""采样核绕法向转多少度，按屏幕坐标 4×4 铺开（``角度 = 本表[y % 4][x % 4]``）。

**这是随机旋转的确定性替身**：所有像素用同一个核，平坦墙面上 12 个样本的落点处处相同，
遮蔽值就会呈现出核自己的形状（一圈一圈的环带）。通常的做法是给每个像素一个随机转角，
但随机数在本仓是红线。改用一张写死的铺块：16 个角度取 0..15 的比特翻转序（van der
Corput）× 2π/16 排进 4×4，**相邻格子的角度差最大**，所以铺块内部先自己散开一次，
剩下的网格感再由 :data:`SSAO_BLUR_HALF_PX` 的均值模糊抹掉。"""

_SSAO_AXIS_SWAP_COS: float = 0.9
"""法向与世界 z 轴的夹角余弦超过它，切线的参考轴就换成 x。叉积在两向量平行时退化成
零向量，切空间基就散了——地板与天花的法向正好贴着 z 轴，不换轴每一张水平面都会中招。"""

_SSAO_MIN_SAMPLE_DEPTH_M: float = 1e-4
"""采样点落到相机平面之后（深度 ≤ 0）就没法投回屏幕，直接判不遮挡。"""

SHADOW_CASTER_SEMANTICS: frozenset[MeshSemantic] = frozenset({"furnishing"})
"""**只有家具投影，壳体（地/顶/墙/洞壁）不投影。**

这不是省事，是这盏灯的处境决定的：主光是一束平行光，而户型的壳体把这户围成一个**闭合
的盒子**——真让壳体投影，室内每一个机位都整间房落在影子里，那张影子一点信息都不带
（同 bird 机位剔天花那条裁决的形状：挡在中间又不带信息的东西，剔掉）。

让家具投影，影子说的是"这件东西落在地上哪儿、离墙多远"——那是形体信息，正是底渲这一路
要交给写实化的东西。**接收方是全部几何**：家具的影子照样落在地板、墙面、别的家具上。"""

SHADOW_MAP_PX: int = 2048
"""从光源那一侧渲的深度图边长。整户（本仓实测约 11 米宽）铺在 2048 上是 5 毫米一格——
室内机位凑到 1.5 米看，一格约合画面上 3 个像素，影子边缘的台阶再经几何路的超采样
一平均就压掉了。给到 4096 只是把这台阶压到 1.5 像素，代价却是四倍的图；给到 1024
则是两厘米一格，室内机位上看得见明显的锯齿边。"""

SHADOW_BIAS_M: float = 0.012
"""比深度图里记的再近 1.2 厘米才算被挡住。**没有它就是满屏摩尔纹**：深度图是离散的，
一块平面上每个格子记的是格子中心那一点的深度，边上的点自然一半在前一半在后，
于是这块平面自己把自己遮成条纹。取 1.2 厘米是按 :data:`SHADOW_MAP_PX` 那一格的
斜掠误差量级定的——比它小压不住条纹，比它大影子会从物体底下浮起来一道缝。"""

_SHADOW_FIT_MARGIN_M: float = 0.05
"""光源那一侧的取景框往外放 5 厘米。框正好贴着包围盒时，最外圈的三角形会被裁掉半个像素，
影子边上于是缺一道。"""


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
class _RasterJob:
    """一次光栅要的全部入参，攒成一份。

    单独成型是因为**同一批三角形要按两个分辨率各光栅一遍**：1 倍那一次出的缓冲是
    深度/遮罩/线稿三路的唯一来源，:data:`GEOMETRY_SUPERSAMPLE_FACTOR` 倍那一次只喂几何路。
    两次之间除了画幅没有任何差别，所以入参只该攒一次、由 :meth:`rasterize_at` 换个倍数取。
    """

    triangles_m: Float64Array
    tri_mesh_ids: npt.NDArray[np.int32]
    view_matrix: Float64Array
    proj_matrix: Float64Array
    width_px: int
    height_px: int
    near_clip_m: float

    def rasterize_at(self, factor: int) -> RasterBuffers:
        """按 ``factor`` 倍画幅光栅一遍。投影矩阵不随倍数变——它只认宽高比。"""
        return rasterize(
            self.triangles_m,
            self.tri_mesh_ids,
            self.view_matrix,
            self.proj_matrix,
            self.width_px * factor,
            self.height_px * factor,
            self.near_clip_m,
        )


@dataclass(frozen=True, eq=False)
class _ShadowMap:
    """从主光那一侧渲出来的深度图，外加把世界点投进它所需的全部数：正交，故没有视锥。

    ``light_view_matrix`` 之外还要带半宽半高，是因为正交投影矩阵是**按要框住的东西现算**
    的（每台相机、每份包都不一样），而查表那一步要用同一组数把世界点换成格子下标——
    两处各算一次就会错开半格，影子会整体偏一点点。
    """

    depth_m: Float32Array
    hit_mask: npt.NDArray[np.bool_]
    light_view_matrix: Float64Array
    half_width_m: float
    half_height_m: float
    size_px: int


@dataclass(frozen=True, eq=False)
class _ScreenGeometry:
    """逐像素的视空间量，环境光遮蔽与线稿两路共用，算一次传两处。"""

    ray_view_xyz: Float64Array
    """(H, W, 3) 每个像素的视空间方向，z 恒为 -1——所以"沿射线走 t"里的 t 就是正深度（米）。"""

    normal_view_xyz: Float64Array
    """(H, W, 3) 视空间法向，**已按视线翻正**（上游绕序不保证一致，见 raster 不做背面剔除）。"""

    position_view_m: Float64Array


def resolve_camera_pose(scene: ScenePackage, camera_id: str, aspect_ratio: float) -> CameraPose:
    """按 ``camera_id`` 解出实际机位。找不到相机／房间就抛 :class:`BaseRenderError`。

    - ``bird``：注视整户包围盒中心，方向用相机自带的 ``yaw_deg``/``pitch_deg``，**距离自动算**
      ——把包围盒外接球塞进视锥里较窄的那个方向（竖直与水平张角取小者），再退 6% 留边。
    - ``room``：**退到房间边缘、朝房间内部看**（观感提档 2026-09-01，取代"站在地板质心上"
      那版——质心正是屋子正中，四面都是墙，站那儿只会看见一堵墙加半个柜子）。水平朝向
      ``forward_xy`` 分两种来路：``camera.yaw_deg`` 若是上游显式给出的（``model_fields_set``
      里有它，哪怕值恰好是默认的 0），就按它平视，**不被下面的自动取景盖过**；没给才自动指向
      该房间家具的面积加权质心（见 :func:`_room_auto_forward_xy`）。机位本身**总是**沿
      ``-forward_xy``（镜头背后那个方向）从地板质心退，退到离墙 :data:`ROOM_EYE_WALL_MARGIN_M`
      **且**离任何家具 :data:`ROOM_EYE_FURNISHING_MARGIN_M` 的最远处（:func:`_room_eye_xy_m`）
      ——这一步与 yaw 是不是自动无关：不管朝哪儿看，镜头背后都该有房间的进深，不是贴着墙
      或者贴着柜子站。俯仰仍固定用 :data:`ROOM_PITCH_DEG`。

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
        centroid_xy = np.asarray(centroid_xy_m, dtype=np.float64)
        floor_triangles_xy = _room_floor_triangles_xy_m(scene, camera.room)
        if "yaw_deg" in camera.model_fields_set:
            forward_xy = _yaw_pitch_direction(camera.yaw_deg, ROOM_PITCH_DEG)[:2]
        else:
            forward_xy = _room_auto_forward_xy(scene, camera.room, floor_triangles_xy, centroid_xy)
        eye_xy = _room_eye_xy_m(scene, camera.room, floor_triangles_xy, centroid_xy, forward_xy)
        eye_m = np.array([eye_xy[0], eye_xy[1], floor_z_m + camera.eye_height_m], dtype=np.float64)
        forward = np.array([forward_xy[0], forward_xy[1], 0.0], dtype=np.float64)
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
    （同 render2d 母版那条口径；测试直接断字节相等）。观感那一批（超采样、环境光遮蔽）
    照样一个随机数都没有——采样核是写死的常量数组，转角是写死的 4×4 铺块。

    光栅走**两遍**：1 倍那一遍出深度/遮罩/线稿三路（它们是被当数据读的，见模块 docstring），
    :data:`GEOMETRY_SUPERSAMPLE_FACTOR` 倍那一遍只出几何路。分两遍而不是"渲一遍高的再
    降采样给大家用"，是为了让三路数据图与观感提档之间**结构上没有接口**——观感这一节
    再怎么改，那三路走的还是原来那一次光栅的原始缓冲。
    """
    if width_px <= 0 or height_px <= 0:
        raise BaseRenderError(f"画幅必须为正：width_px={width_px} height_px={height_px}")
    mesh_count = len(scene.meshes)
    if mesh_count + 1 > MASK_MAX_INDEX + 1:
        raise BaseRenderError(f"网格数超出 16 位遮罩索引上限：meshes={mesh_count}")

    aspect_ratio = width_px / height_px
    pose = resolve_camera_pose(scene, camera_id, aspect_ratio)
    mesh_indices = _rendered_mesh_indices(scene, pose.kind)
    triangles_m, tri_mesh_ids = _flatten_meshes(scene, mesh_indices)
    palette_ratio = _mesh_palette_ratio(scene)
    shadow = _build_shadow_map(scene, mesh_indices)

    view_matrix = look_at_matrix(pose.eye_m, pose.target_m, pose.up_hint_xyz)
    proj_matrix = perspective_matrix(pose.fov_deg, aspect_ratio, pose.near_clip_m, pose.far_clip_m)
    job = _RasterJob(
        triangles_m=triangles_m,
        tri_mesh_ids=tri_mesh_ids,
        view_matrix=view_matrix,
        proj_matrix=proj_matrix,
        width_px=width_px,
        height_px=height_px,
        near_clip_m=pose.near_clip_m,
    )
    buffers = job.rasterize_at(1)
    screen = _screen_geometry(buffers, view_matrix, pose, aspect_ratio)

    near_m, far_m, depth_png = _encode_depth_png(buffers)
    mask_png, mask_index = _encode_mask_png(buffers, scene)
    return BaseRenderViews(
        geometry_png=_encode_geometry_png(
            job, buffers, screen, pose, aspect_ratio, palette_ratio, shadow
        ),
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


def _room_content_anchor_xy_m(scene: ScenePackage, room: str) -> Float64Array | None:
    """该房间家具的**面积加权**质心 (x, y)；房间里一件家具都没有就是 ``None``。

    权重取三角形自身面积，跟 :func:`_room_floor_anchor_m` 同一个理由：一张大茶几比十个
    小拉手更能代表"这间房主要在看什么"，跟拉手切了几个三角形无关。侧面（柜门、抽屉的
    立面）也一起算进去，不只算水平投影——这里求的是"这堆几何的质量分布在哪儿"，不是
    "占地面积"，一人高的衣柜本该比同样占地的矮凳把镜头往它那边多拉一点。
    """
    weighted_sum = np.zeros(2, dtype=np.float64)
    total_area_m2 = 0.0
    for mesh in scene.meshes:
        if mesh.semantic != "furnishing" or mesh.room != room or not mesh.triangles:
            continue
        verts = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
        index = np.asarray(mesh.triangles, dtype=np.int64).reshape(-1, 3)
        tris = verts[index]
        cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        areas_m2 = 0.5 * np.linalg.norm(cross, axis=1)
        centroids_xy = tris[:, :, :2].mean(axis=1)
        weighted_sum += (centroids_xy * areas_m2[:, None]).sum(axis=0)
        total_area_m2 += float(areas_m2.sum())
    if total_area_m2 <= 0.0:
        return None
    return weighted_sum / total_area_m2


def _room_floor_triangles_xy_m(scene: ScenePackage, room: str) -> Float64Array:
    """(N, 3, 2) 该房间地板三角形在 xy 平面的顶点，供 :func:`_room_retreat_distance_m`
    判"这一点还在地板上"。调用方保证房间有地板——:func:`_room_floor_anchor_m` 在这之前
    已经查过一遍同一个条件，这里不重复报错。
    """
    blocks: list[Float64Array] = []
    for mesh in scene.meshes:
        if mesh.semantic != "floor" or mesh.room != room or not mesh.triangles:
            continue
        verts = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)[:, :2]
        index = np.asarray(mesh.triangles, dtype=np.int64).reshape(-1, 3)
        blocks.append(verts[index])
    return np.concatenate(blocks, axis=0)


def _room_furnishing_triangles_xy_m(scene: ScenePackage, room: str) -> Float64Array:
    """(N, 3, 2) 该房间家具三角形在 xy 平面的投影，供 :func:`_room_retreat_distance_m`
    判"这一点离家具还有多远"。与地板不同，房间允许一件家具都没有——空场景返回
    ``(0, 3, 2)``，:func:`_min_distance_to_triangles_xy` 把它读成"没有障碍物"。
    """
    blocks: list[Float64Array] = []
    for mesh in scene.meshes:
        if mesh.semantic != "furnishing" or mesh.room != room or not mesh.triangles:
            continue
        verts = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)[:, :2]
        index = np.asarray(mesh.triangles, dtype=np.int64).reshape(-1, 3)
        blocks.append(verts[index])
    if not blocks:
        return np.zeros((0, 3, 2), dtype=np.float64)
    return np.concatenate(blocks, axis=0)


def _point_in_any_triangle_xy(triangles_xy_m: Float64Array, point_xy: Float64Array) -> bool:
    """点是否落在 ``triangles_xy_m`` 里任意一个三角形内（含边界）。

    向量化的同号判据：点相对三角形每条边的有向面积同号（或为零）即在内部，一次比完全部
    三角形，不逐个三角形早退——房间的地板三角形数量级在几十到几百，向量化比早退循环快，
    也不必操心"先查哪个三角形"这种会牵出遍历序的问题。
    """
    a, b, c = triangles_xy_m[:, 0], triangles_xy_m[:, 1], triangles_xy_m[:, 2]

    def _signed_area2(p: Float64Array, q: Float64Array, r: Float64Array) -> Float64Array:
        return (p[:, 0] - r[:, 0]) * (q[:, 1] - r[:, 1]) - (q[:, 0] - r[:, 0]) * (p[:, 1] - r[:, 1])

    point = np.broadcast_to(point_xy, a.shape)
    d1, d2, d3 = _signed_area2(point, a, b), _signed_area2(point, b, c), _signed_area2(point, c, a)
    has_negative = (d1 < 0.0) | (d2 < 0.0) | (d3 < 0.0)
    has_positive = (d1 > 0.0) | (d2 > 0.0) | (d3 > 0.0)
    return bool(np.any(~(has_negative & has_positive)))


def _point_segment_distance_xy(
    point_xy: Float64Array, a_xy: Float64Array, b_xy: Float64Array
) -> Float64Array:
    """点到线段 ``a_xy``–``b_xy`` 的最短距离，向量化到一批线段上。"""
    ab = b_xy - a_xy
    ab_len2 = np.einsum("ij,ij->i", ab, ab)
    ratio = np.einsum("ij,ij->i", point_xy - a_xy, ab) / np.where(ab_len2 > 0.0, ab_len2, 1.0)
    closest = a_xy + ab * np.clip(ratio, 0.0, 1.0)[:, None]
    return np.linalg.norm(point_xy - closest, axis=1)


def _min_distance_to_triangles_xy(triangles_xy_m: Float64Array, point_xy: Float64Array) -> float:
    """点到一批三角形（边界，含内部记 0）的最短距离。三角形数组为空记 ``inf``——
    "没有障碍物"与"障碍物远在天边"在退景那一步是同一件事。"""
    if triangles_xy_m.shape[0] == 0:
        return math.inf
    if _point_in_any_triangle_xy(triangles_xy_m, point_xy):
        return 0.0
    a, b, c = triangles_xy_m[:, 0], triangles_xy_m[:, 1], triangles_xy_m[:, 2]
    point = np.broadcast_to(point_xy, a.shape)
    edge_distances = np.minimum(
        _point_segment_distance_xy(point, a, b),
        np.minimum(
            _point_segment_distance_xy(point, b, c), _point_segment_distance_xy(point, c, a)
        ),
    )
    return float(edge_distances.min())


def _room_retreat_distance_m(
    floor_triangles_xy_m: Float64Array,
    furnishing_triangles_xy_m: Float64Array,
    start_xy_m: Float64Array,
    direction_xy: Float64Array,
) -> float:
    """从 ``start_xy_m``（保证在地板内，通常是地板质心）沿单位向量 ``direction_xy`` 退，
    退到**同时满足**"还在地板上，且离墙 :data:`ROOM_EYE_WALL_MARGIN_M`"与"离所有家具至少
    :data:`ROOM_EYE_FURNISHING_MARGIN_M`"这两条的最远处（米）。

    两条判据一起扫、撞上哪条就停在哪条，且**墙距只在真撞上墙的时候才扣**：家具挡道时
    ``_min_distance_to_triangles_xy`` 量出来的已经是"离家具多远"本身，那个数已经满足
    :data:`ROOM_EYE_FURNISHING_MARGIN_M` 的下限，不该再叠一层墙距去扣——这不是同一件事
    的两次计费，是两种障碍物各自的量尺。真跑数据验过反面：不分青红皂白统一扣墙距，
    多数房间退不到 5 厘米就被判定"到头了"（家具往往就摆在退景方向上，原始可退距离本来
    就没比墙距大多少，再扣一次直接归零），整条退景形同虚设。

    按 :data:`_ROOM_BOUNDARY_SEARCH_STEPS` 个固定步长扫，不解析求交：地板允许是任意多边形
    （L 形、带凹角的户型很常见），解析求交要先按凸凹分两套公式，固定步数一份代码就把两种
    形状都盖了，代价只是分辨率有限。非凸形状下"沿一条射线出去就再也不会回到地板里"这个
    假设可能不成立（射线可能先穿出一个凹角、后面又扫回地板），但退景只要一个"退到这儿
    一定还站得住"的下界——一撞到出界或者贴上家具就停，宁可少退一点，也不会把机位退到
    墙外面或者撞进柜子里。
    """
    span_m = float(np.linalg.norm(np.ptp(floor_triangles_xy_m.reshape(-1, 2), axis=0)))
    if span_m <= 0.0:
        return 0.0
    step_m = span_m / _ROOM_BOUNDARY_SEARCH_STEPS
    farthest_m = 0.0
    for step_index in range(1, _ROOM_BOUNDARY_SEARCH_STEPS + 1):
        candidate_m = step_index * step_m
        point_xy = start_xy_m + direction_xy * candidate_m
        if not _point_in_any_triangle_xy(floor_triangles_xy_m, point_xy):
            # 让给墙距：candidate_m 已经踩出地板了，退回 candidate_m - 墙距 才是"离墙还有
            # 余量"的点；跟上一步已经验过安全的 farthest_m 取小，防着房间小到一步的采样
            # 步长本身就比墙距还大，wall_safe_m 反而比上一步更远的极端情况。
            wall_safe_m = max(0.0, candidate_m - ROOM_EYE_WALL_MARGIN_M)
            farthest_m = min(farthest_m, wall_safe_m)
            break
        if _min_distance_to_triangles_xy(furnishing_triangles_xy_m, point_xy) < (
            ROOM_EYE_FURNISHING_MARGIN_M
        ):
            break
        farthest_m = candidate_m
    else:
        # 扫完 _ROOM_BOUNDARY_SEARCH_STEPS 步都没撞墙也没撞家具：span_m 是地板包围盒的
        # 对角线，从盒内一点沿任意方向走这么远必然已经出盒（多边形又整个落在盒里），
        # 正常情况下走不到这儿；真走到这儿说明浮点误差让最后一步刚好卡在边界上，同样
        # 按"撞墙"处理，让出墙距，不要把机位摆到搜索上限那个点上。
        farthest_m = max(0.0, farthest_m - ROOM_EYE_WALL_MARGIN_M)
    return farthest_m


def _room_fallback_direction_xy(
    triangles_xy_m: Float64Array, centroid_xy_m: Float64Array
) -> Float64Array:
    """没有家具可看时的兜底朝向：地板离质心最远的那个顶点方向。

    形状本身已经给出唯一确定的答案，不用另编一个默认方向（同《纪律·拿不到就说没有，
    不许填猜的值》的精神）：退到这个方向的边缘上、看回去，房间纵深最深的那一段正对镜头，
    比随手定一个 +y 更不容易一睁眼就对着一堵近墙。"""
    vertices_xy = triangles_xy_m.reshape(-1, 2)
    offsets = vertices_xy - centroid_xy_m
    farthest_index = int(np.argmax(np.einsum("ij,ij->i", offsets, offsets)))
    direction = offsets[farthest_index]
    norm = float(np.linalg.norm(direction))
    if norm < DEPTH_MIN_SPAN_M:
        return np.array([0.0, 1.0], dtype=np.float64)
    return direction / norm


def _room_auto_forward_xy(
    scene: ScenePackage,
    room: str,
    floor_triangles_xy_m: Float64Array,
    centroid_xy_m: Float64Array,
) -> Float64Array:
    """``camera.yaw_deg`` 没有显式给出时的水平朝向：指向该房间家具的面积加权质心；
    没有家具（或家具质心恰好落在地板质心上，见 :data:`ROOM_AUTO_DIRECTION_MIN_M`）
    就退回形状本身的兜底方向（:func:`_room_fallback_direction_xy`）。"""
    content_xy_m = _room_content_anchor_xy_m(scene, room)
    if content_xy_m is not None:
        offset = content_xy_m - centroid_xy_m
        norm = float(np.linalg.norm(offset))
        if norm >= ROOM_AUTO_DIRECTION_MIN_M:
            return offset / norm
    return _room_fallback_direction_xy(floor_triangles_xy_m, centroid_xy_m)


def _room_eye_xy_m(
    scene: ScenePackage,
    room: str,
    floor_triangles_xy_m: Float64Array,
    centroid_xy_m: Float64Array,
    forward_xy: Float64Array,
) -> Float64Array:
    """机位退到哪儿：从地板质心沿 ``-forward_xy``（镜头背后那个方向）退，退到离墙
    :data:`ROOM_EYE_WALL_MARGIN_M`、离家具 :data:`ROOM_EYE_FURNISHING_MARGIN_M` 处；
    房间小到退不了那么多，就退到能退的最远处（:func:`_room_retreat_distance_m` 找不到
    地方退时返回 0，机位留在质心上，不会退出房间，也不会撞进家具）。

    退的方向永远是**镜头背后**，不是"离家具最远"：家具决定的是看哪儿（``forward_xy``），
    退是为了让镜头背后空出一整间房的进深，两件事顺着同一个朝向走，退出来的画面才是
    "站在房间这一头看那一头"，不是斜着站在屋子中间。
    """
    retreat_xy = -forward_xy
    furnishing_triangles_xy_m = _room_furnishing_triangles_xy_m(scene, room)
    eye_distance_m = _room_retreat_distance_m(
        floor_triangles_xy_m, furnishing_triangles_xy_m, centroid_xy_m, retreat_xy
    )
    return centroid_xy_m + retreat_xy * eye_distance_m


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


def _build_shadow_map(scene: ScenePackage, mesh_indices: list[int]) -> _ShadowMap | None:
    """从主光那一侧正交渲一张深度图；这台相机没有任何投影体就返回 ``None``。

    投影体只取 :data:`SHADOW_CASTER_SEMANTICS`（今天＝家具），且**只取这台相机本来就要画
    的那些网格**——bird 剔了天花，天花就不该在光那一侧还立着挡光。

    取景框按投影体的包围盒在光空间里现算：正交投影没有"离多远"这回事，画幅只由要框住的
    东西定。相机站在包围盒往光源方向退开一个对角线的地方，保证所有投影体的深度为正
    （:func:`rasterize` 要 ``near_m > 0``）。

    返回 ``None`` 而不是一张空图：没有投影体时"整幅不在影子里"是个确定的事实，让调用方
    直接跳过查表，比渲一张全 inf 的图再查一遍便宜，也少一处"空图算不算遮挡"的分支。
    """
    caster_indices = [
        index
        for index in mesh_indices
        if scene.meshes[index].semantic in SHADOW_CASTER_SEMANTICS and scene.meshes[index].triangles
    ]
    if not caster_indices:
        return None
    triangles_m, tri_mesh_ids = _flatten_meshes(scene, caster_indices)
    if triangles_m.shape[0] == 0:
        return None

    light_from = np.asarray(KEY_LIGHT_FROM_DIR_XYZ, dtype=np.float64)
    light_from = light_from / float(np.linalg.norm(light_from))
    min_xyz_m, max_xyz_m = _scene_bounds_m(scene, mesh_indices)
    center_m = (min_xyz_m + max_xyz_m) * 0.5
    span_m = float(np.linalg.norm(max_xyz_m - min_xyz_m))
    eye_m = center_m + light_from * max(span_m, DEPTH_MIN_SPAN_M)
    light_view = look_at_matrix(eye_m, center_m)

    corners_m = np.stack(
        np.meshgrid(*zip(min_xyz_m, max_xyz_m, strict=True), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    corners_view = np.concatenate([corners_m, np.ones((corners_m.shape[0], 1))], axis=1) @ (
        light_view.T
    )
    half_width_m = float(np.abs(corners_view[:, 0]).max()) + _SHADOW_FIT_MARGIN_M
    half_height_m = float(np.abs(corners_view[:, 1]).max()) + _SHADOW_FIT_MARGIN_M
    depth_range_m = -corners_view[:, 2]
    near_m = max(NEAR_CLIP_M, float(depth_range_m.min()) - _SHADOW_FIT_MARGIN_M)
    far_m = float(depth_range_m.max()) + _SHADOW_FIT_MARGIN_M

    buffers = rasterize(
        triangles_m,
        tri_mesh_ids,
        light_view,
        orthographic_matrix(half_width_m, half_height_m, near_m, far_m),
        SHADOW_MAP_PX,
        SHADOW_MAP_PX,
        near_m,
    )
    return _ShadowMap(
        depth_m=buffers.depth_m,
        hit_mask=buffers.hit_mask,
        light_view_matrix=light_view,
        half_width_m=half_width_m,
        half_height_m=half_height_m,
        size_px=SHADOW_MAP_PX,
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
    normal_view = normal_view * np.where(toward > 0.0, -1.0, 1.0)[..., None]
    return _ScreenGeometry(
        ray_view_xyz=ray_view,
        normal_view_xyz=normal_view,
        position_view_m=position_view,
    )


# ---------------------------------------------------------------------------
# 四路编码
# ---------------------------------------------------------------------------


def _encode_geometry_png(
    job: _RasterJob,
    buffers: RasterBuffers,
    screen: _ScreenGeometry,
    pose: CameraPose,
    aspect_ratio: float,
    palette_ratio: Float64Array,
    shadow: _ShadowMap | None,
) -> bytes:
    """几何路：超采样 + 半球环境光 + 主/补光 + 环境光遮蔽 + 家具投影，8 位 RGB PNG。

    四步，每一步为什么在这儿：

    1. 按 :data:`GEOMETRY_SUPERSAMPLE_FACTOR` 倍**再光栅一遍**（``buffers`` 那一遍是给
       另外三路的，一个像素都不动）；
    2. 在**线性光**里着色——材质色是 sRGB 编码的，直接乘明暗等于在伽马空间里乘，中间调
       会被压暗一大截，这正是原来那张图"发闷"的一半来源。着色**分环境光与主光两份返回**，
       因为压它们的是两件不同的事（遮蔽 vs 投影），合在一起就分不开了；
    3. 两份各自盒式降采样，**也在线性光里做**：把 sRGB 值平均出来的边缘会比两侧都暗
       （伽马是凸的），抗锯齿反而画出一圈黑边；
    4. 遮蔽、背景在 1 倍分辨率上合成。遮蔽是低频量，按 1 倍算再乘上去与按 N 倍算肉眼无差，
       却省掉 N² 倍的采样。合成用的是**覆盖率加权**（降采样后的颜色已经带着覆盖率），
       所以边缘像素上背景与几何按子像素比例混，遮蔽只压几何那一份。
    """
    factor = GEOMETRY_SUPERSAMPLE_FACTOR
    high = job.rasterize_at(factor)
    palette_linear = _srgb_to_linear(palette_ratio.astype(np.float32))

    ambient_linear, key_linear = _shade_linear(
        high, job, pose, aspect_ratio, palette_linear, shadow
    )
    covered = high.hit_mask.astype(np.float32)
    coverage_ratio = _box_downsample(covered[..., None], factor)[..., 0]
    occlusion_ratio = _ambient_occlusion_ratio(buffers, screen, pose, aspect_ratio)

    background_linear = _srgb_to_linear(
        np.asarray(GEOMETRY_BACKGROUND_RGB_U8, dtype=np.float32) / 255.0
    )
    composited = (
        _box_downsample(ambient_linear, factor) * occlusion_ratio[..., None]
        + _box_downsample(key_linear, factor)
        + (1.0 - coverage_ratio)[..., None] * background_linear
    )
    rgb_u8 = np.rint(np.clip(_linear_to_srgb(composited), 0.0, 1.0) * 255.0).astype(np.uint8)
    return _encode_png(Image.fromarray(rgb_u8, mode="RGB"))


def _shade_linear(
    buffers: RasterBuffers,
    job: _RasterJob,
    pose: CameraPose,
    aspect_ratio: float,
    palette_linear: Float32Array,
    shadow: _ShadowMap | None,
) -> tuple[Float32Array, Float32Array]:
    """着色，返回**(环境光那一份, 主光那一份)**，都是 (H, W, 3) 线性光。

    没打到几何的像素两份都给 0——降采样之后它们就自动成了覆盖率加权的和，边缘上不必
    再单独记一次"这个子像素有没有东西"。

    分两份返回，是因为压它们的是两件不同的事：环境光被凹角挡住（:func:`_ambient_occlusion_ratio`），
    主光被家具挡住（:data:`SHADOW_CASTER_SEMANTICS`）。合成一份再乘两个系数，等于在一块
    既在墙角又在阴影里的地方把同一份光扣两遍。补光跟着环境光那一份走：它是"把背光面捞
    回来"的散射项，没有方向可言，投影对它不成立。

    三盏灯全部写死在世界系、不随相机转（理由见 :data:`KEY_LIGHT_FROM_DIR_XYZ`）。
    """
    ray_world = _world_ray_xyz(buffers, job.view_matrix, pose, aspect_ratio)
    facing = np.where(
        np.einsum("hwi,hwi->hw", buffers.normal_unit_xyz, ray_world) > 0.0,
        np.float32(-1.0),
        np.float32(1.0),
    )
    normal_world = buffers.normal_unit_xyz * facing[..., None]

    sky = np.asarray(AMBIENT_SKY_RGB, dtype=np.float32)
    ground = np.asarray(AMBIENT_GROUND_RGB, dtype=np.float32)
    up_ratio = (0.5 + 0.5 * normal_world[..., 2])[..., None]
    irradiance = ground + (sky - ground) * up_ratio
    irradiance += np.asarray(FILL_LIGHT_RGB, dtype=np.float32) * _lambert(
        normal_world, FILL_LIGHT_FROM_DIR_XYZ
    )

    key_irradiance = np.asarray(KEY_LIGHT_RGB, dtype=np.float32) * _lambert(
        normal_world, KEY_LIGHT_FROM_DIR_XYZ
    )
    if shadow is not None:
        # 空像素的深度是 inf，乘出来是 inf/nan，投回光空间会一路污染到取整。它们反正
        # 不着色，直接按 0 米算——落在相机自己那一点上，判出来的是"照得到"，无害。
        depth_m = np.where(buffers.hit_mask, buffers.depth_m, np.float32(0.0))
        position_world_m = np.asarray(pose.eye_m, dtype=np.float32) + ray_world * depth_m[..., None]
        key_irradiance *= _sunlit_ratio(position_world_m, shadow)[..., None]

    albedo = palette_linear[buffers.id_buffer + 1] * buffers.hit_mask[..., None]
    return albedo * irradiance, albedo * key_irradiance


def _lambert(
    normal_world_xyz: Float32Array, light_from_dir_xyz: tuple[float, float, float]
) -> Float32Array:
    """(H, W, 1) 兰伯特系数。光源方向在这儿归一化：常量按可读性写（不是单位长）。"""
    direction = np.asarray(light_from_dir_xyz, dtype=np.float32)
    direction = direction / np.float32(np.linalg.norm(direction))
    return np.asarray(
        np.clip(np.einsum("hwi,i->hw", normal_world_xyz, direction), 0.0, 1.0)[..., None],
        dtype=np.float32,
    )


def _world_ray_xyz(
    buffers: RasterBuffers, view_matrix: Float64Array, pose: CameraPose, aspect_ratio: float
) -> Float32Array:
    """(H, W, 3) 每个像素的**世界系**视线方向，长度归一到"走一米深度前进一米"。

    不走 :class:`_ScreenGeometry` 而另算一遍，是因为超采样那一遍的画幅是 N² 倍大：
    那个结构里三张 float64 缓冲在 2 倍超采样下就是两百多兆，而几何着色只要这一张。
    这里用 float32，并且把世界系射线**直接**由 ``right·x + up·y + forward`` 拼出来，
    不先摊一张视空间射线再乘旋转——省掉一整张中间缓冲。
    """
    height_px, width_px = buffers.id_buffer.shape
    right = view_matrix[0, :3].astype(np.float32)
    up = view_matrix[1, :3].astype(np.float32)
    forward = -view_matrix[2, :3].astype(np.float32)

    tan_v = np.float32(math.tan(math.radians(pose.fov_deg) * 0.5))
    tan_h = tan_v * np.float32(aspect_ratio)
    x_view = ((np.arange(width_px, dtype=np.float32) + 0.5) * (2.0 / width_px) - 1.0) * tan_h
    y_view = (1.0 - (np.arange(height_px, dtype=np.float32) + 0.5) * (2.0 / height_px)) * tan_v
    return np.asarray(
        right * x_view[None, :, None] + up * y_view[:, None, None] + forward, dtype=np.float32
    )


def _sunlit_ratio(position_world_m: Float32Array, shadow: _ShadowMap) -> Float32Array:
    """(H, W) 主光照不照得到：1 ＝ 照得到，0 ＝ 被家具挡住。

    把世界点换进光空间、按正交画幅算出格子下标，跟深度图里那一格比：图里记的更近，
    说明这条光线上先撞到了别的东西。落在取景框外的点判为**照得到**——框是按投影体的
    包围盒算的，框外就是没有投影体的地方，那儿本来就没有影子。
    """
    matrix = shadow.light_view_matrix.astype(np.float32)
    x_m = np.einsum("hwi,i->hw", position_world_m, matrix[0, :3]) + matrix[0, 3]
    y_m = np.einsum("hwi,i->hw", position_world_m, matrix[1, :3]) + matrix[1, 3]
    depth_m = -(np.einsum("hwi,i->hw", position_world_m, matrix[2, :3]) + matrix[2, 3])

    column = np.floor((x_m / shadow.half_width_m + 1.0) * 0.5 * shadow.size_px).astype(np.int64)
    row = np.floor((1.0 - y_m / shadow.half_height_m) * 0.5 * shadow.size_px).astype(np.int64)
    inside = (column >= 0) & (column < shadow.size_px) & (row >= 0) & (row < shadow.size_px)
    flat_index = np.clip(row, 0, shadow.size_px - 1) * shadow.size_px + np.clip(
        column, 0, shadow.size_px - 1
    )
    caster_m = shadow.depth_m.reshape(-1)[flat_index]
    blocked = (
        inside & shadow.hit_mask.reshape(-1)[flat_index] & (caster_m < depth_m - SHADOW_BIAS_M)
    )
    return np.where(blocked, np.float32(0.0), np.float32(1.0))


def _box_downsample(image: Float32Array, factor: int) -> Float32Array:
    """(N·h, N·w, C) → (h, w, C)，N×N 盒式平均。调用方保证画幅是 N 的整数倍。"""
    height_px, width_px, channels = image.shape
    blocks = image.reshape(height_px // factor, factor, width_px // factor, factor, channels)
    return np.asarray(blocks.mean(axis=(1, 3)), dtype=np.float32)


def _srgb_to_linear(value_ratio: Float32Array) -> Float32Array:
    """sRGB 编码值 → 线性光。材质色是按 ``#RRGGBB`` 给的，那是 sRGB，不是能直接相乘的量。"""
    clamped = np.clip(value_ratio, 0.0, 1.0)
    return np.asarray(
        np.where(clamped <= 0.04045, clamped / 12.92, ((clamped + 0.055) / 1.055) ** 2.4),
        dtype=np.float32,
    )


def _linear_to_srgb(value_ratio: Float32Array) -> Float32Array:
    clamped = np.clip(value_ratio, 0.0, 1.0)
    return np.asarray(
        np.where(clamped <= 0.0031308, clamped * 12.92, 1.055 * clamped ** (1.0 / 2.4) - 0.055),
        dtype=np.float32,
    )


def _ambient_occlusion_ratio(
    buffers: RasterBuffers, screen: _ScreenGeometry, pose: CameraPose, aspect_ratio: float
) -> Float32Array:
    """(H, W) 每个像素还剩多少环境光：1 ＝ 完全开阔，:data:`SSAO_STRENGTH_RATIO` 决定下限。

    屏幕空间做法：以本像素的视空间位置为中心，往法向那一侧的半球撒
    :data:`SSAO_KERNEL_TANGENT` 那 12 个点，把每个点投回屏幕、跟深度缓冲里那一格比——
    缓冲比采样点更近，说明这个方向上被挡住了。

    **这是"体积感"的来源**：墙角的两个面互相遮、家具与地面的接缝互相遮，于是有了接地感；
    单靠方向光做不到这件事（平行光对一个凹角里的两个面给的是同一个明暗）。

    远处的遮挡物按 ``半径 / 距离`` 衰减：不衰减的话对面那堵墙会把整间房算成被遮挡，
    出来的是一张整体压暗的图而不是墙角变暗。
    """
    height_px, width_px = buffers.id_buffer.shape
    hit = buffers.hit_mask
    if not bool(hit.any()):
        return np.ones((height_px, width_px), dtype=np.float32)

    normal = screen.normal_view_xyz.astype(np.float32)
    position_m = screen.position_view_m.astype(np.float32)
    # 空像素在深度缓冲里是 inf，两个 inf 相减是 nan；这里把它们换成 0，判遮挡那一步
    # 本来就把空像素排除在外（``hit_flat``），换掉只是别让衰减那一步算出 nan 来。
    depth_m = np.where(hit, buffers.depth_m, np.float32(0.0))
    axis_u, axis_v = _tangent_frame(normal, height_px, width_px)

    tan_v = math.tan(math.radians(pose.fov_deg) * 0.5)
    tan_h = tan_v * aspect_ratio
    depth_flat = depth_m.reshape(-1)
    hit_flat = hit.reshape(-1)
    occlusion = np.zeros((height_px, width_px), dtype=np.float32)

    for offset_u, offset_v, offset_n in SSAO_KERNEL_TANGENT:
        sample_m = position_m + (
            axis_u * np.float32(offset_u)
            + axis_v * np.float32(offset_v)
            + normal * np.float32(offset_n)
        ) * np.float32(SSAO_RADIUS_M)
        sample_depth_m = -sample_m[..., 2]
        ahead = sample_depth_m > _SSAO_MIN_SAMPLE_DEPTH_M
        safe_depth_m = np.where(ahead, sample_depth_m, 1.0)
        x_px = (sample_m[..., 0] / (safe_depth_m * tan_h) + 1.0) * 0.5 * width_px
        y_px = (1.0 - sample_m[..., 1] / (safe_depth_m * tan_v)) * 0.5 * height_px
        column = np.floor(x_px).astype(np.int64)
        row = np.floor(y_px).astype(np.int64)
        on_screen = ahead & (column >= 0) & (column < width_px) & (row >= 0) & (row < height_px)
        flat_index = np.clip(row, 0, height_px - 1) * width_px + np.clip(column, 0, width_px - 1)
        occluder_m = depth_flat[flat_index]
        blocked = on_screen & hit_flat[flat_index] & (occluder_m < sample_depth_m - SSAO_BIAS_M)
        falloff = np.float32(SSAO_RADIUS_M) / np.maximum(
            np.float32(SSAO_RADIUS_M), np.abs(depth_m - occluder_m)
        )
        occlusion += np.where(blocked, falloff, np.float32(0.0))

    open_ratio = 1.0 - SSAO_STRENGTH_RATIO * occlusion / len(SSAO_KERNEL_TANGENT)
    clamped: Float32Array = np.where(hit, np.clip(open_ratio, 0.0, 1.0), 1.0).astype(np.float32)
    return _box_blur(clamped, SSAO_BLUR_HALF_PX)


def _tangent_frame(
    normal_view_xyz: Float32Array, height_px: int, width_px: int
) -> tuple[Float32Array, Float32Array]:
    """逐像素的切空间两根轴，已按 :data:`SSAO_ROTATION_TILE_RAD` 绕法向转过。

    转角按屏幕位置查表而不是逐像素随机：随机数是本仓的红线，而不转的话每个像素的采样
    落点完全相同，平坦墙面上会显出采样核自己的形状。
    """
    reference = np.where(
        (np.abs(normal_view_xyz[..., 2]) > _SSAO_AXIS_SWAP_COS)[..., None],
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    )
    tangent = np.cross(reference, normal_view_xyz)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), np.float32(1e-8))
    bitangent = np.cross(normal_view_xyz, tangent)

    tile = np.asarray(SSAO_ROTATION_TILE_RAD, dtype=np.float32)
    angle_rad = tile[
        np.ix_(np.arange(height_px) % tile.shape[0], np.arange(width_px) % tile.shape[1])
    ]
    cos_a = np.cos(angle_rad)[..., None]
    sin_a = np.sin(angle_rad)[..., None]
    axis_u: Float32Array = tangent * cos_a + bitangent * sin_a
    return axis_u, np.asarray(np.cross(normal_view_xyz, axis_u), dtype=np.float32)


def _box_blur(image: Float32Array, half_px: int) -> Float32Array:
    """边缘按最外一行/列延拓的方形均值。可分离成两趟，两趟内部的求和顺序都写死。"""
    size = 2 * half_px + 1
    padded = np.pad(image, ((half_px, half_px), (0, 0)), mode="edge")
    rows = padded[0 : image.shape[0]].copy()
    for offset in range(1, size):
        rows += padded[offset : offset + image.shape[0]]
    rows /= np.float32(size)

    padded = np.pad(rows, ((0, 0), (half_px, half_px)), mode="edge")
    columns = padded[:, 0 : image.shape[1]].copy()
    for offset in range(1, size):
        columns += padded[:, offset : offset + image.shape[1]]
    columns /= np.float32(size)
    return np.asarray(columns, dtype=np.float32)


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
