"""底渲（activity ``base-render``）：一份场景包 + 一个机位 → 几何/深度/线稿/遮罩/控制稿五路图。

**五路是给下一步当条件图的，不是给人看的成品图**——写实化（``realism-pass``）在 imagegen
那个仓走生成模型。所以这一步既不需要 GPU 也不需要三维引擎：五路全是几何缓冲，纯 numpy
软光栅就出得来（光栅本身在 :mod:`render3d_worker.raster`，本模块只管"编码成哪几张图"）。

五路分**三路数据、一路观感、一路控制**，这是本模块最要紧的一条分界：

====== ================== ================== =============================================
路     图像形态            背景（没打到几何）  下游怎么读它
====== ================== ================== =============================================
深度   16 位灰度 PNG       0                  **当数据读**：值是米（1..65535 归一化，两端随包带出）
遮罩   16 位灰度索引 PNG   0                  **当数据读**：索引 ≡ 网格下标 +1，回指网格身份
线稿   8 位灰度 PNG        0（黑底白线）      **当数据读**：像素要么是几何事实边、要么不是
几何   8 位 RGB PNG        柔和中性底          **当图看**：给写实化当色彩/形体参考，也给人调试看
控制稿 8 位灰度 PNG        0（黑底白线）      **当控制条件送**：线稿生图通道锁什么，就画什么
====== ================== ================== =============================================

**三路数据图一律"没有东西 ＝ 0"**，且**一律不做任何观感处理**——不抗锯齿、不降采样、
不加环境光遮蔽。理由是这三路的每个像素都要经得起被当成一个量去读：深度边缘上一平均，
就造出一个现实中不存在的深度；遮罩索引一平均，指到的是一块不存在的网格；线稿糊一下，
"这是不是一条真边"就答不出来了。**"这个像素有没有东西"由这三路答**，三路互相自查
（测试即断这条一致性）。控制稿同样黑底、同样不做观感处理，只是它答的不是"有没有东西"，
是"控制通道该锁住什么"。

几何那一路**退出"背景＝0"这条约定**（观感提档 2026-09-01）：纯黑底把每一条轮廓都变成
最高对比的硬边，人眼看着扎、写实化那一步也没有必要吃这个对比。它是五路里唯一
"给人和给生成模型看"的一路，所以抗锯齿、环境光遮蔽、调色都只落在它身上
（见 :data:`GEOMETRY_SUPERSAMPLE_FACTOR` 起的那一节常量）。

**线稿与控制稿的分工**（2026-09-05，来路＝中控仓《评审/失效清单-控制图通路-2026-09-04》）。
线稿是**几何事实边**：每一条网格边界、每一处深度断开都画，它是保真度尺子的输入，
一个像素都不按"该不该给模型看"取舍，**一个字节都不动**。控制稿是**给控制通道画的**：
线稿生图那一类通道的性质是"线稿里画什么，出图就锁什么；画错什么，锁错什么"，失效清单里
构件级四条（地面分界线被画成台阶 B1、天花交线被画成灯槽 B2、透视里的天花斜线被读成斜顶 B3、
门窗不分 B4）来源全是线稿画了不该画的边、没画该画的区别。所以控制稿按下面的画法另出一路。

控制稿画法（:func:`_encode_sketch_png`；这是给控制通道画的，不是给人看的）：

- **画**：墙与地面的交线（地脚线）、墙与墙的竖直交线、洞口轮廓（洞口侧壁与墙面的折边、
  过梁底面与墙面的折边、门洞侧壁与地面的交线）、遮挡轮廓（近处几何盖住远处几何的边，
  透过门洞看到的远处轮廓也在内）、揭顶视角下的墙顶与墙厚（墙顶面与墙侧面的折边、墙顶面
  盖住地板的遮挡边）、家具体块的可见边（上游给了家具体块才有；今天上游不给，家具为空时
  自然一条都不画）。
- **不画**：地面上房间之间的分界线（两块地板共面相接——没有墙的地方，地面上不许有线）、
  天花与墙的交线、天花分区线、同一面墙被切成几块之后的共面接缝（墙段/过梁/窗下墙之间，
  外轮廓与网格墙重合的段之间）。
- **符号按洞的最终种类画**（判据＝同一张图上门和窗的符号不同）：**门**（``door``，入户门
  ``entry-door`` 同）画到地面——门洞侧壁与地面相交的地脚线画、地面上不画门槛线——洞口内画
  **外框（三边，不画门槛线）+ 一条通高的门扇线 + 一条把手短横**；**窗**（``window``）离地
  有窗台线（窗下墙顶面与窗下墙立面的折边），洞口内画**外框 + 内框 + 窗台线**；**过口**
  （``passage``）洞口内不画符号——它没有门扇、没有窗台，只有洞口轮廓。
  上游没给种类（``unknown``）的洞在编场景包时已按档位猜成了门或窗，这儿画的就是猜出来
  那一种的符号（猜了几个、哪几个，场景包 ``guessed_opening_indices`` 说得出）。
  符号画在墙厚的中心平面上（窗台线画在朝相机那一面的墙面上），按本机位的深度缓冲做遮挡
  判断——被墙挡住的洞口，符号也被挡住。
- **符号方案**（``sketch_symbols``，CLI ``--sketch-symbols``；2026-09-05 晚加，来路＝真跑
  ``_iteration/run-2026-09-05-opening-kind-realism/``：窗十字 2/3 被读成黑板或带格柜子、门斜线 1/3
  被画成实体斜条）。上面那段是默认方案 ``frame-handle``（2026-09-06 用户裁决换的，见
  :data:`DEFAULT_SKETCH_SYMBOLS`）；其余方案只换洞口内的符号，画的边、编码、尺寸、遮挡判断
  全同，四路一个字节不动。每个方案的画法在 :data:`SKETCH_SYMBOL_SCHEMES` 条目里写死；
  哪个方案当默认要用户拍，默认值不随实验换。
- 洞口的**种类**从场景包的洞口表读（``ScenePackage.openings``，2026-09-05 起编场景包时带出，
  每个洞一行：最终种类与来源）；洞口的**框**从网格里读（:func:`_opening_frames`）：切出来的洞
  有 ``reveal:{kind}:{来源}:{墙线号}:{洞号}`` 套框网格，补出来的洞（洞落在两段墙的空隙里，
  见 mesh ``_layout_openings``）没有套框，只有 ``wall:fill:{洞号}:lintel`` 过梁块与（窗才有的）
  ``wall:fill:{洞号}:sill`` 窗下墙块，洞号回指洞口表。场景包没有洞口表时（2026-09-05 之前编的
  老场景包）退回按网格 id 读：套框读 id 里的种类，补出来的洞有窗下墙按窗画、没有按门画——
  这条退路分不开门与过口，所以只给老包用。

线稿与控制稿用的是同一套几何量（法向、平面外推），差别只在**画不画的取舍**：线稿全画；
控制稿先把相邻像素对分成"共面接缝 / 折边 / 遮挡边 / 轮廓"四种（:func:`_sketch_pair_marks`），
再按两侧网格的语义（地板/天花/墙/洞壁/家具）取舍。

失败要响亮：相机 id 找不到、``room`` 机位指的房间没有地板、自动取景找不到一个达标的位姿、
网格引用的材质不在场景包里，一律抛 :class:`BaseRenderError`，**不退化成默认相机、不编一个
兜底颜色、不出一张对着墙的图**——退化只会让一张看着正常、其实渲错了机位的图流到下游
（《纪律·拿不到就说没有，不许填猜的值》）。
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from PIL import Image

from render3d_worker.models import (
    BaseRenderViews,
    CameraKind,
    CameraSpec,
    MaskEntry,
    MeshSemantic,
    RoomViewCheck,
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

ROOM_VIEW_RETREAT_DIRECTION_COUNT: int = 16
"""自动取景时从起点往几个方向退（等分 360°，22.5° 一档）。户型是正交的：8 档只有轴向与对角，
退到的位置只有墙的正中和墙角；16 档多出"略偏轴"那一档，退得到墙角旁边、避得开正对门洞的
位置。再加密到 32 档，退到的位置与 16 档相差不到一个避墙距离，只是多花一倍时间。"""

ROOM_VIEW_YAW_COUNT: int = 16
"""每个候选位置试几个水平朝向（等分 360°，22.5° 一档）。只保留朝着起点那一侧的（视线与
"位置→起点"的夹角不超过 90°）：背对房间主体看出去，画面必然是一堵墙，不必渲一遍来证明。"""

ROOM_VIEW_EVAL_HEIGHT_PX: int = 96
"""候选评估用的光栅高度（宽按最终画幅的宽高比算）。实测本仓软光栅的耗时由三角形数决定、
与画幅几乎无关（真户型 2368 个三角形：96×72 0.067 秒，256×192 0.077 秒）；96 高已经把
占比量到 1% 以内（一格 ≈ 1/12288），再高只是白花。"""

ROOM_VIEW_MIN_TARGET_FLOOR_RATIO: float = 0.04
"""目标房间的地板至少占画面多少才算"拍到了这间房、不是对着墙"。

**取值理由＝真户型 7 台室内机位实测下能分开好坏的最小值**（2026-09-05，老规则的位姿按
本评估器量，全表在 ``_iteration/run-2026-09-05-control-sketch/run.md``）。失效清单 D1 判为
"整幅是墙面"的两台在这个量上是 0.000（卫生间）与 0.031（厨房）；判为可用的四台是 0.046
（书房）、0.056（阳台）、0.145（主卧）、0.211（客厅）。0.04 落在 0.031 与 0.046 之间。
这不是"好图"的门槛，是"不是墙面图"的门槛。数这么小是因为室内机位固定平视
（:data:`ROOM_PITCH_DEG`）、眼高 1.55 m：竖直半张角 32.5°（65° 张角）之内地板只从 2.43 m
以外才进画面，小房间里能进画面的地板本来就只有一条。"""

ROOM_VIEW_MIN_DOMINANCE_RATIO: float = 0.5
"""目标房间在"有房间归属的像素"（各房间的地板 + 天花）里至少占多少才算画面主体。

**取值理由＝真户型 7 台室内机位实测下能分开好坏的最小值**（同上表）。老规则下次卧那台
站在整间房的质心上——上游把两处不相连的地板都标成"次卧"，质心落在它们之间、其实站在
客厅里——它在这个量上是 0.053；判为可用的机位最低是阳台 0.546，其余 0.706～0.996。
0.5 是"目标房间不少于其他房间之和"这句话的直译，落在两组之间。

这个量**分不出失效清单 D2**（客厅看穿三间房）：客厅老机位在这个量上是 0.957——遮罩里
墙不归任何房间，透过门洞看到的是隔壁的墙，遮罩数不到它。按"击中点落不落在目标房间地板
足迹里"另量了一遍（含墙），客厅老机位 0.91、主卧 0.99、书房 0.93、阳台 0.66，同样分不开
（数在 run.md）。D2 今天没有能用的判据，留给用户定。"""

_ROOM_VIEW_EYE_DEDUPE_M: float = 0.01
"""两个候选位置相距不到 1 厘米就当同一个位置（小房间里往几个方向退会退到同一处）。
1 厘米远小于任何取景意义上的差别，只是去重复。"""

_ROOM_FLOOR_TOUCH_TOLERANCE_M: float = 1e-3
"""两块地板矩形的边相距不到 1 毫米就算相接（同一间房的连通判据）。网格顶点留到微米
（mesh ``_COORD_DECIMALS``），相接的块共享的是同一个坐标，1 毫米只是留给浮点的余量。"""

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

SKETCH_BACKGROUND_U8: int = LINE_BACKGROUND_U8
SKETCH_FOREGROUND_U8: int = LINE_FOREGROUND_U8
"""控制稿与线稿同编码：黑底白线、单通道 0/255、同尺寸。下游把两路当同一种输入送。"""

SKETCH_SAME_PLANE_COS: float = 0.999
"""相邻两个像素的法向夹角余弦不小于它就当同一张平面。本仓的几何全是平面（墙块、地板、
天花、家具体块），相邻两张面要么共面（余弦 1，只差浮点误差）、要么以直角相接（余弦 0）；
0.999（约 2.6°）离两头都远，**不是靠调它来分边**。"""

SKETCH_CONTIGUITY_TOLERANCE_RATIO: float = LINE_DEPTH_TOLERANCE_RATIO
"""判两个相邻像素是"相接的两张面"还是"一前一后的两张面"：任一侧的平面外推到另一侧的
视线上、预测深度与实测差在这个比例内，就是相接。理由：相接的两张面共享一条交线，交线上
的点同时落在两张平面上，所以**至少有一个方向外推得准**；遮挡边两个方向都外推不准。
容差与线稿那一路同一个数、同一个理由——留给浮点误差的余量，不是靠它分边。"""

SKETCH_SYMBOL_DEPTH_TOLERANCE_RATIO: float = LINE_DEPTH_TOLERANCE_RATIO
"""门窗符号做遮挡判断时，符号上的点比深度缓冲远不超过这个比例仍算可见。符号画在墙厚的
中心平面上，端点正落在洞口侧壁那张面上，与缓冲里的深度只差浮点误差；容差用同一个数。"""

SketchSymbolScheme = Literal[
    "diagonal-cross", "frame-sill", "glazing-hatch", "leaf-swing", "frame-handle"
]
SKETCH_SYMBOL_SCHEMES: tuple[SketchSymbolScheme, ...] = (
    "diagonal-cross",
    "frame-sill",
    "glazing-hatch",
    "leaf-swing",
    "frame-handle",
)
"""控制稿门窗符号方案闭集。所有方案：确定性、同编码（黑底白线）、门与窗的符号互不相同、
过口不画符号、门不在地面上画门槛线。**外框**＝洞口边界在墙厚中心平面上的矩形（门三边、
窗四边），**内框**＝外框向内缩 :data:`SKETCH_FRAME_INSET_M` 的矩形（窗才有，表示玻璃边）。

- ``diagonal-cross``（2026-09-06 之前的默认，现在只作可选）：门＝洞内一条左下到右上的斜线；
  窗＝洞内十字（竖梃 + 横梃）。**留着是历史样本的复现依据**——2026-09-06 之前跑出来的图全
  是这个方案的稿，删了就复现不出来；新图不用它（模型常把斜线当成真东西，见
  :data:`DEFAULT_SKETCH_SYMBOLS` 那条的数据）。
- ``frame-sill``：窗＝外框 + 内框 + 窗台线（洞下沿再向下 :data:`SKETCH_SILL_DROP_M`、两端各
  伸出 :data:`SKETCH_SILL_OVERHANG_M`，画在朝相机那一侧的墙面上），不画十字；门＝外框 +
  门扇线（洞内一条通高竖线，离沿墙坐标小的那侧洞边 :data:`SKETCH_DOOR_LEAF_OFFSET_M`，
  表示开着的门扇的可见边），不画斜线。
- ``glazing-hatch``：窗＝外框 + 内框 + 三条 45° 斜向短划（玻璃反光；起点在内框宽 18%/34%/50%、
  内框高 55% 处，长 :data:`SKETCH_HATCH_LENGTH_RATIO` × 内框短边，左下向右上），不画十字、
  不画窗台线；门＝外框 + 门把手（离沿墙坐标大的那侧洞边 :data:`SKETCH_DOOR_HANDLE_EDGE_M`、
  长 :data:`SKETCH_DOOR_HANDLE_LENGTH_M`、高 :data:`SKETCH_DOOR_HANDLE_HEIGHT_M` 的一条短横）。
- ``leaf-swing``：门＝外框 + 一扇朝相机这一侧开到 :data:`SKETCH_DOOR_SWING_DEG` 的门扇（铰链在
  沿墙坐标小的那侧洞边；画门扇的顶边、底边、自由竖边三条线，是三维里的线段、不在墙面上）；
  窗＝外框 + 内框 + 窗台线 + 中竖梃（双扇平开窗的分扇线），不画横梃。
- ``frame-handle``（**默认**，2026-09-06 加，来路＝真跑
  ``_iteration/run-2026-09-05-sketch-symbols/``：``frame-sill`` 的窗 6/6 读对、门里无斜线残影，
  ``glazing-hatch`` 的把手 3/3 长成真把手）：
  窗＝照 ``frame-sill``（外框 + 内框 + 窗台线）；门＝外框 + 门扇线（照 ``frame-sill``）+ 把手
  短横（位置与尺寸照 ``glazing-hatch``）。门扇线在沿墙坐标小的那侧、把手在大的那侧，两件
  各自照抄、没有合成一扇门的几何。
"""

DEFAULT_SKETCH_SYMBOLS: SketchSymbolScheme = "frame-handle"
"""默认方案。**2026-09-06 用户裁决从 ``diagonal-cross`` 换成 ``frame-handle``**——问的是"门窗在
线描图上的画法，换不换（新画法 49 个门零失误，老画法 21 个残影 + 12 个被渲成实体）"，选中
"换（推荐）"（备选：保持现状／再看看别的户型）。连带：斜线被渲成实体那类失效消失，为它做
检测器也不必要了。

数据出处（三批真跑，写完不改）：
- ``_iteration/run-2026-09-05-sketch-symbols/``：室内 18 张。``diagonal-cross`` 门里符号残影 6/12、
  书房那扇窗 3 张只读对 1；``frame-sill`` 残影 0/12、窗 3/3。
- ``_iteration/run-2026-09-05-sketch-symbol-combo/``：合成方案 + 揭顶 15 张。两批合计 33 个门格，
  ``frame-sill``/``frame-handle`` 0 残影对 ``diagonal-cross`` 10/33。
- ``_iteration/run-2026-09-06-sketch-symbol-birdview/``：揭顶 7 个 seed、49 个门格。
  ``diagonal-cross`` 残影 21/49、被画成实体 12/49、7 张里 5 张地面被斜线切成两色、门形态 43/49；
  ``frame-sill``/``frame-handle`` 0/49、0/49、0 处、48/49。窗 168 格三方案全对，换默认由门定。
  两个新方案四栏同分，差别只在把手：没被遮挡的 21 格长出 16 格，无一例变差——默认取
  ``frame-handle``（是 ``frame-sill`` 的超集）。

换默认要用户拍，默认值不随实验换。旧方案 ``diagonal-cross`` 留在
:data:`SKETCH_SYMBOL_SCHEMES` 里作可选：2026-09-06 之前的图都是它渲的稿，它是那批历史样本的
复现依据。"""

SKETCH_FRAME_INSET_M: float = 0.08
"""内框向内缩的量：窗框型材可见宽 6～8 厘米的常规档位。"""

SKETCH_SILL_DROP_M: float = 0.06
SKETCH_SILL_OVERHANG_M: float = 0.04
"""窗台线离洞下沿的距离与两端伸出洞宽的量：窗台板厚 5～6 厘米、两端各出 3～5 厘米的常规档位。"""

SKETCH_DOOR_LEAF_OFFSET_M: float = 0.12
"""``frame-sill`` 的门扇线离洞边的距离：门扇开到八成时从正面看到的门扇投影宽的量级。"""

SKETCH_HATCH_LENGTH_RATIO: float = 0.30
"""``glazing-hatch`` 每条短划的长度 ＝ 内框短边 × 它；三条起点按内框尺寸定比例，所以短划
永远落在内框里（起点最远在宽 50%/高 55%，加 0.30 × 短边 × cos45° 仍不出框）。"""

SKETCH_DOOR_HANDLE_HEIGHT_M: float = 1.00
SKETCH_DOOR_HANDLE_LENGTH_M: float = 0.12
SKETCH_DOOR_HANDLE_EDGE_M: float = 0.06
"""门把手：离地 1 米、执手长 12 厘米、离门扇自由边 6 厘米的常规档位。"""

SKETCH_DOOR_SWING_DEG: float = 45.0
"""``leaf-swing`` 门扇开到的角度（从墙面量）：斜视机位下门扇既伸进房间、又看得出洞宽。"""

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
    room_view: RoomViewCheck | None = None
    """``room`` 机位的取景自证数（自动取景时是选中那个候选的评估结果；上游显式给 yaw 时
    只量不判）。``bird`` 机位为 ``None``。"""


@dataclass(frozen=True, eq=False)
class _RoomViewJob:
    """一间房自动取景要的全部不变量：摊平的三角形、每块网格归不归目标房间、张角、画幅。
    攒一次，每个候选位姿只换视图矩阵——候选有上百个，这些东西不该算上百遍。"""

    triangles_m: Float64Array
    tri_mesh_ids: npt.NDArray[np.int32]
    target_floor_of_index: npt.NDArray[np.bool_]
    """(1 + 网格数,) 这个遮罩索引是不是目标房间的地板。0 号（背景）恒为 False。"""

    target_room_of_index: npt.NDArray[np.bool_]
    other_room_of_index: npt.NDArray[np.bool_]
    """(1 + 网格数,) 这个遮罩索引归目标房间 / 归别的房间。墙不归任何房间，两边都是 False。"""

    fov_deg: float
    aspect_ratio: float
    near_clip_m: float
    far_clip_m: float
    width_px: int
    height_px: int


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
    - ``room``：**室内机位不许对墙**（2026-09-05，来路＝失效清单 D1/D2：老规则"退到房间
      边缘朝屋里看"在小房间里退到贴着墙、整幅是墙面；客厅那台看穿三间房、主体不是客厅）。
      分两条路：

      * ``camera.yaw_deg`` 是上游显式给出的（``model_fields_set`` 里有它，哪怕值恰好是默认
        的 0）：按它平视，机位沿 ``-forward_xy``（镜头背后）从地板质心退到离墙
        :data:`ROOM_EYE_WALL_MARGIN_M` **且**离任何家具 :data:`ROOM_EYE_FURNISHING_MARGIN_M`
        的最远处（:func:`_room_eye_xy_m`）——上游给了就听上游的，**不做候选评估、不判**，
        只把取景自证数量出来随位姿带出（``room_view.passed`` 可以是 False）。
      * 没给：**确定性的候选评估**（:func:`_room_view_candidate_poses` 起那一节）。从这间房
        最大那块连通地板的质心出发，往 :data:`ROOM_VIEW_RETREAT_DIRECTION_COUNT` 个方向各退到
        能退的最远处得到候选位置，每个位置配 :data:`ROOM_VIEW_YAW_COUNT` 档朝向里朝着起点
        那一侧的；每个候选用本仓自己的低分辨率深度/遮罩光栅评估三个数——最近深度、目标房间
        地板占比、目标房间在有房间归属的像素里的占比；三条判据（最近深度 ≥ 避墙距离、地板
        占比 ≥ :data:`ROOM_VIEW_MIN_TARGET_FLOOR_RATIO`、主体占比 ≥
        :data:`ROOM_VIEW_MIN_DOMINANCE_RATIO`）全过的里面取"地板占比 × 主体占比"最大的，
        同分取候选次序靠前的。**一个都不达标就抛错**，报"房间 X 无法取景"与最接近的那个候选
        的数，不出一张墙面图。

      俯仰两条路都固定用 :data:`ROOM_PITCH_DEG`。

    包围盒**按网格顶点现算**，不读 ``bounds_min_m``/``bounds_max_m``：那两个字段是场景包的
    自证数（编包那一侧填的），取景必须框住真正会被画出来的东西，两者万一不一致，以画得出来
    的为准。同理，取景只框**这台相机会画的那些网格**——bird 剔了天花，包围盒就不该再被
    天花撑着。
    """
    if aspect_ratio <= 0.0:
        raise BaseRenderError(f"宽高比必须为正：aspect_ratio={aspect_ratio}")
    camera = _find_camera(scene, camera_id)
    mesh_indices = _rendered_mesh_indices(scene, camera.kind)
    min_xyz_m, max_xyz_m = _scene_bounds_m(scene, mesh_indices)
    diagonal_m = float(np.linalg.norm(max_xyz_m - min_xyz_m))
    far_clip_m = max(FAR_CLIP_MIN_M, diagonal_m * FAR_CLIP_DIAGONAL_RATIO)
    room_view: RoomViewCheck | None = None

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
        job = _room_view_job(
            scene, camera.room, mesh_indices, camera.fov_deg, aspect_ratio, far_clip_m
        )
        eye_z_m = floor_z_m + camera.eye_height_m
        if "yaw_deg" in camera.model_fields_set:
            forward_xy = _yaw_pitch_direction(camera.yaw_deg, ROOM_PITCH_DEG)[:2]
            eye_xy = _room_eye_xy_m(scene, camera.room, floor_triangles_xy, centroid_xy, forward_xy)
            room_view = _room_view_check(job, eye_xy, eye_z_m, camera.yaw_deg, candidate_count=1)
        else:
            room_view = _room_view_pick(scene, camera.room, job, floor_triangles_xy, eye_z_m)
        eye_m = np.asarray(room_view.eye_m, dtype=np.float64)
        forward = _yaw_pitch_direction(room_view.yaw_deg, ROOM_PITCH_DEG)
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
        room_view=room_view,
    )


def render_base_views(
    scene: ScenePackage,
    camera_id: str,
    width_px: int = 1024,
    height_px: int = 768,
    sketch_symbols: SketchSymbolScheme = DEFAULT_SKETCH_SYMBOLS,
) -> BaseRenderViews:
    """一份场景包 + 一个机位 → 五路图 + 遮罩索引表 + 自证数。

    ``sketch_symbols`` 只管控制稿里门窗符号的画法（闭集 :data:`SKETCH_SYMBOL_SCHEMES`），
    其余四路与它无关。

    零模型调用、无随机、无时间戳：**同一份场景包渲两次，五张 PNG 逐字节相同**
    （同 render2d 母版那条口径；测试直接断字节相等）。观感那一批（超采样、环境光遮蔽）
    照样一个随机数都没有——采样核是写死的常量数组，转角是写死的 4×4 铺块。

    光栅走**两遍**：1 倍那一遍出深度/遮罩/线稿/控制稿四路（前三路是被当数据读的，控制稿
    是从同一份缓冲按另一套取舍画的，见模块 docstring），:data:`GEOMETRY_SUPERSAMPLE_FACTOR`
    倍那一遍只出几何路。分两遍而不是"渲一遍高的再降采样给大家用"，是为了让数据图与观感
    提档之间**结构上没有接口**——观感这一节再怎么改，那几路走的还是原来那一次光栅的原始缓冲。
    """
    if width_px <= 0 or height_px <= 0:
        raise BaseRenderError(f"画幅必须为正：width_px={width_px} height_px={height_px}")
    if sketch_symbols not in SKETCH_SYMBOL_SCHEMES:
        raise BaseRenderError(
            f"控制稿符号方案认不出：{sketch_symbols}；认得的：{SKETCH_SYMBOL_SCHEMES}"
        )
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
        sketch_png=_encode_sketch_png(
            scene, buffers, screen, view_matrix, proj_matrix, pose, sketch_symbols
        ),
        width_px=width_px,
        height_px=height_px,
        camera_id=pose.camera_id,
        mask_index=mask_index,
        covered_pixel_ratio=buffers.covered_pixel_ratio,
        near_m=near_m,
        far_m=far_m,
        room_view=pose.room_view,
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
    return np.asarray(np.linalg.norm(point_xy - closest, axis=1), dtype=np.float64)


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


# ---------------------------------------------------------------------------
# 室内机位的候选评估：不许对墙
# ---------------------------------------------------------------------------


def _room_view_job(
    scene: ScenePackage,
    room: str,
    mesh_indices: list[int],
    fov_deg: float,
    aspect_ratio: float,
    far_clip_m: float,
) -> _RoomViewJob:
    """攒一间房取景评估的不变量。画幅按 :data:`ROOM_VIEW_EVAL_HEIGHT_PX` 与最终宽高比定——
    评估用的张角与宽高比必须和最终那张图一样，否则量的是另一台相机的画面。"""
    triangles_m, tri_mesh_ids = _flatten_meshes(scene, mesh_indices)
    count = len(scene.meshes) + 1
    target_floor = np.zeros(count, dtype=np.bool_)
    target_room = np.zeros(count, dtype=np.bool_)
    other_room = np.zeros(count, dtype=np.bool_)
    for mesh_index, mesh in enumerate(scene.meshes):
        if mesh.room is None or mesh.semantic not in ("floor", "ceiling"):
            continue
        if mesh.room == room:
            target_room[mesh_index + 1] = True
            target_floor[mesh_index + 1] = mesh.semantic == "floor"
        else:
            other_room[mesh_index + 1] = True
    height_px = ROOM_VIEW_EVAL_HEIGHT_PX
    width_px = max(1, int(round(height_px * aspect_ratio)))
    return _RoomViewJob(
        triangles_m=triangles_m,
        tri_mesh_ids=tri_mesh_ids,
        target_floor_of_index=target_floor,
        target_room_of_index=target_room,
        other_room_of_index=other_room,
        fov_deg=fov_deg,
        aspect_ratio=aspect_ratio,
        near_clip_m=NEAR_CLIP_M,
        far_clip_m=far_clip_m,
        width_px=width_px,
        height_px=height_px,
    )


def _room_view_passes(
    min_depth_m: float, target_floor_ratio: float, dominance_ratio: float
) -> bool:
    """三条取景判据，只写在这一处：不贴墙、拍到了地板、画面主体是这间房。"""
    return (
        min_depth_m >= ROOM_EYE_WALL_MARGIN_M
        and target_floor_ratio >= ROOM_VIEW_MIN_TARGET_FLOOR_RATIO
        and dominance_ratio >= ROOM_VIEW_MIN_DOMINANCE_RATIO
    )


def _room_view_check(
    job: _RoomViewJob,
    eye_xy_m: Float64Array,
    eye_z_m: float,
    yaw_deg: float,
    candidate_count: int,
) -> RoomViewCheck:
    """按本仓自己的深度/遮罩光栅量一个位姿：最近深度、目标房间地板占比、主体占比。

    量的就是最终那台相机会画的东西（同一批三角形、同一个张角与宽高比），只是画幅小——
    所以"评估说主体是这间房"与"最终图上主体是这间房"是同一件事，差的只是采样粒度。
    """
    eye_m = np.array([eye_xy_m[0], eye_xy_m[1], eye_z_m], dtype=np.float64)
    forward = _yaw_pitch_direction(yaw_deg, ROOM_PITCH_DEG)
    view_matrix = look_at_matrix(eye_m, eye_m + forward * ROOM_TARGET_DISTANCE_M)
    proj_matrix = perspective_matrix(job.fov_deg, job.aspect_ratio, job.near_clip_m, job.far_clip_m)
    buffers = rasterize(
        job.triangles_m,
        job.tri_mesh_ids,
        view_matrix,
        proj_matrix,
        job.width_px,
        job.height_px,
        job.near_clip_m,
    )
    total_px = float(job.width_px * job.height_px)
    hit = buffers.hit_mask
    hit_px = int(np.count_nonzero(hit))
    min_depth_m = float(buffers.depth_m[hit].min()) if hit_px > 0 else math.inf
    index = buffers.id_buffer + 1
    target_floor_ratio = float(np.count_nonzero(job.target_floor_of_index[index])) / total_px
    target_room_ratio = float(np.count_nonzero(job.target_room_of_index[index])) / total_px
    other_room_ratio = float(np.count_nonzero(job.other_room_of_index[index])) / total_px
    attributed = target_room_ratio + other_room_ratio
    dominance_ratio = target_room_ratio / attributed if attributed > 0.0 else 0.0
    return RoomViewCheck(
        eye_m=(float(eye_m[0]), float(eye_m[1]), float(eye_m[2])),
        yaw_deg=float(yaw_deg),
        min_depth_m=min_depth_m,
        target_floor_ratio=target_floor_ratio,
        target_room_ratio=target_room_ratio,
        other_room_ratio=other_room_ratio,
        dominance_ratio=dominance_ratio,
        candidate_count=candidate_count,
        passed=_room_view_passes(min_depth_m, target_floor_ratio, dominance_ratio),
    )


def _room_floor_boxes_m(scene: ScenePackage, room: str) -> list[tuple[float, float, float, float]]:
    """该房间每块地板网格的平面包围矩形 ``(x0, x1, y0, y1)``。地板网格一块就是一个矩形
    （mesh ``_floor_and_ceiling``），包围矩形就是它本身。次序跟着网格序。"""
    boxes: list[tuple[float, float, float, float]] = []
    for mesh in scene.meshes:
        if mesh.semantic != "floor" or mesh.room != room or not mesh.vertices:
            continue
        verts = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
        boxes.append(
            (
                float(verts[:, 0].min()),
                float(verts[:, 0].max()),
                float(verts[:, 1].min()),
                float(verts[:, 1].max()),
            )
        )
    return boxes


def _room_view_start_xy_m(
    scene: ScenePackage, room: str, floor_triangles_xy_m: Float64Array
) -> Float64Array:
    """自动取景的起点：这间房**最大的那块连通地板**的面积加权质心；质心落在地板外
    （L 形）就退到那块连通地板里最大一块矩形的中心。

    为什么不直接用整间房的质心（:func:`_room_floor_anchor_m`）：上游的房间遮罩会把同名的
    几块不相连的地板算成一间房（真户型 2026-09-05 实测："次卧"两处、"卫生间"三处），
    整间房的质心落在它们之间的墙里，从墙里出发退到哪儿都不是这间房。连通判据是矩形相接
    （:data:`_ROOM_FLOOR_TOUCH_TOLERANCE_M`），地板网格一块就是一个矩形
    （mesh ``_floor_and_ceiling``）。
    """
    boxes = _room_floor_boxes_m(scene, room)
    component_of = list(range(len(boxes)))

    def _root(index: int) -> int:
        while component_of[index] != index:
            component_of[index] = component_of[component_of[index]]
            index = component_of[index]
        return index

    tolerance = _ROOM_FLOOR_TOUCH_TOLERANCE_M
    for i, (ax0, ax1, ay0, ay1) in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            bx0, bx1, by0, by1 = boxes[j]
            touching = (
                ax0 <= bx1 + tolerance
                and bx0 <= ax1 + tolerance
                and ay0 <= by1 + tolerance
                and by0 <= ay1 + tolerance
            )
            if touching:
                component_of[_root(i)] = _root(j)

    area_of_component: dict[int, float] = {}
    for index, (x0, x1, y0, y1) in enumerate(boxes):
        root = _root(index)
        area_of_component[root] = area_of_component.get(root, 0.0) + (x1 - x0) * (y1 - y0)
    # 同面积取先出现的连通块：dict 的插入序跟着网格序走，网格序是场景包写死的
    best_root = max(area_of_component, key=lambda root: (area_of_component[root], -root))

    weighted = np.zeros(2, dtype=np.float64)
    total_area = 0.0
    largest_box_center = np.zeros(2, dtype=np.float64)
    largest_box_area = -1.0
    for index, (x0, x1, y0, y1) in enumerate(boxes):
        if _root(index) != best_root:
            continue
        area = (x1 - x0) * (y1 - y0)
        center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)
        weighted += center * area
        total_area += area
        if area > largest_box_area:
            largest_box_area, largest_box_center = area, center
    centroid = weighted / total_area
    if _point_in_any_triangle_xy(floor_triangles_xy_m, centroid):
        return centroid
    return largest_box_center


def _room_view_candidate_poses(
    floor_triangles_xy_m: Float64Array,
    furnishing_triangles_xy_m: Float64Array,
    start_xy_m: Float64Array,
) -> list[tuple[Float64Array, float]]:
    """候选位姿 ``(eye_xy, yaw_deg)``，次序写死：起点本身在前，然后按退让方向的角序；
    每个位置按朝向的角序。次序进了"同分取靠前"那条规则，所以它是结果的一部分。"""
    eyes: list[Float64Array] = [start_xy_m]
    for step in range(ROOM_VIEW_RETREAT_DIRECTION_COUNT):
        direction_xy = _yaw_pitch_direction(360.0 * step / ROOM_VIEW_RETREAT_DIRECTION_COUNT, 0.0)[
            :2
        ]
        distance_m = _room_retreat_distance_m(
            floor_triangles_xy_m, furnishing_triangles_xy_m, start_xy_m, direction_xy
        )
        eye_xy = start_xy_m + direction_xy * distance_m
        if any(float(np.linalg.norm(eye_xy - known)) < _ROOM_VIEW_EYE_DEDUPE_M for known in eyes):
            continue
        eyes.append(eye_xy)

    poses: list[tuple[Float64Array, float]] = []
    for eye_xy in eyes:
        toward_start = start_xy_m - eye_xy
        at_start = float(np.linalg.norm(toward_start)) < _ROOM_VIEW_EYE_DEDUPE_M
        for step in range(ROOM_VIEW_YAW_COUNT):
            yaw_deg = 360.0 * step / ROOM_VIEW_YAW_COUNT
            forward_xy = _yaw_pitch_direction(yaw_deg, 0.0)[:2]
            if not at_start and float(np.dot(forward_xy, toward_start)) < 0.0:
                continue
            poses.append((eye_xy, yaw_deg))
    return poses


def _room_view_pick(
    scene: ScenePackage,
    room: str,
    job: _RoomViewJob,
    floor_triangles_xy_m: Float64Array,
    eye_z_m: float,
) -> RoomViewCheck:
    """评估全部候选，取达标里"地板占比 × 主体占比"最大的；一个都不达标就响亮失败。"""
    start_xy = _room_view_start_xy_m(scene, room, floor_triangles_xy_m)
    poses = _room_view_candidate_poses(
        floor_triangles_xy_m, _room_furnishing_triangles_xy_m(scene, room), start_xy
    )
    checks = [
        _room_view_check(job, eye_xy, eye_z_m, yaw_deg, candidate_count=len(poses))
        for eye_xy, yaw_deg in poses
    ]

    def _score(check: RoomViewCheck) -> float:
        return check.target_floor_ratio * check.dominance_ratio

    best: RoomViewCheck | None = None
    for check in checks:
        if check.passed and (best is None or _score(check) > _score(best)):
            best = check
    if best is not None:
        return best

    closest = checks[0]
    for check in checks[1:]:
        if _score(check) > _score(closest):
            closest = check
    raise BaseRenderError(
        f"房间 {room} 无法取景：{len(checks)} 个候选位姿没有一个达标"
        f"（判据：最近深度 ≥ {ROOM_EYE_WALL_MARGIN_M} m、目标房间地板占比 ≥ "
        f"{ROOM_VIEW_MIN_TARGET_FLOOR_RATIO}、主体占比 ≥ {ROOM_VIEW_MIN_DOMINANCE_RATIO}）；"
        f"最接近的一个站在 ({closest.eye_m[0]:.2f}, {closest.eye_m[1]:.2f}) 朝 "
        f"{closest.yaw_deg:.1f}°：最近深度 {closest.min_depth_m:.2f} m、地板占比 "
        f"{closest.target_floor_ratio:.3f}、主体占比 {closest.dominance_ratio:.3f}"
    )


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
    ray_world = _world_ray_xyz(buffers, job.view_matrix, pose.fov_deg, aspect_ratio)
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
    buffers: RasterBuffers, view_matrix: Float64Array, fov_deg: float, aspect_ratio: float
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

    tan_v = np.float32(math.tan(math.radians(fov_deg) * 0.5))
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


# ---------------------------------------------------------------------------
# 控制稿：给"线稿生图"控制通道画的那一路（画法规则见模块 docstring）
# ---------------------------------------------------------------------------

_SKETCH_BACKGROUND_CODE: int = 0
_SKETCH_SEMANTIC_CODE: dict[MeshSemantic, int] = {
    "floor": 1,
    "ceiling": 2,
    "wall": 3,
    "reveal": 4,
    "furnishing": 5,
}
"""网格语义 → 控制稿取舍用的整数码，0 留给背景（没打到几何）。"""

_OPENING_KINDS: tuple[str, ...] = ("door", "window", "passage", "entry-door")
"""控制稿认得的洞口种类（同 models ``OpeningKind`` 去掉 ``unknown``：起了体的洞没有"不知道"）。"""


@dataclass(frozen=True)
class _OpeningFrame:
    """一个洞口在三维里的框：沿墙的起讫、墙厚中心、竖向起讫。门窗符号画在这个框里。"""

    kind: str
    along_axis: int
    """洞口宽度沿哪根世界轴：0 ＝ x、1 ＝ y。另一根轴就是墙厚方向。"""

    along_m: tuple[float, float]
    across_m: tuple[float, float]
    """墙厚方向上墙的两张面（洞壁沿墙厚的起讫）。符号大多画在两者的中点（墙厚中心平面），
    窗台线画在朝相机那一面上。"""

    z_m: tuple[float, float]

    @property
    def across_center_m(self) -> float:
        return (self.across_m[0] + self.across_m[1]) * 0.5


_Segment = tuple[Float64Array, Float64Array]


def _sketch_semantic_code_of_index(scene: ScenePackage) -> npt.NDArray[np.int8]:
    """(1 + 网格数,) 遮罩索引 → 语义码。``id_buffer + 1`` 直接当下标用（同调色板那套编号）。"""
    codes = np.full(len(scene.meshes) + 1, _SKETCH_BACKGROUND_CODE, dtype=np.int8)
    for mesh_index, mesh in enumerate(scene.meshes):
        codes[mesh_index + 1] = _SKETCH_SEMANTIC_CODE[mesh.semantic]
    return codes


def _encode_sketch_png(
    scene: ScenePackage,
    buffers: RasterBuffers,
    screen: _ScreenGeometry,
    view_matrix: Float64Array,
    proj_matrix: Float64Array,
    pose: CameraPose,
    sketch_symbols: SketchSymbolScheme,
) -> bytes:
    """控制稿路：黑底白线的 8 位灰度 PNG，与线稿同尺寸同编码，**从同一份 1 倍缓冲取**。

    两步：相邻像素对按几何关系与两侧语义取舍（:func:`_sketch_pair_marks`，横竖各扫一遍），
    再把门窗符号按三维线段投影上去、过深度缓冲的遮挡判断（:func:`_draw_opening_symbols`）。
    线宽 1 像素，理由同线稿：条件图上细线比粗线好。
    """
    semantic = _sketch_semantic_code_of_index(scene)[buffers.id_buffer + 1]
    hit = buffers.hit_mask
    depth_m = buffers.depth_m.astype(np.float64)
    canvas = np.zeros(buffers.id_buffer.shape, dtype=np.bool_)

    mark_a, mark_b = _sketch_pair_marks(
        semantic[:, :-1],
        semantic[:, 1:],
        hit[:, :-1],
        hit[:, 1:],
        depth_m[:, :-1],
        depth_m[:, 1:],
        screen.normal_view_xyz[:, :-1],
        screen.normal_view_xyz[:, 1:],
        screen.position_view_m[:, :-1],
        screen.position_view_m[:, 1:],
        screen.ray_view_xyz[:, :-1],
        screen.ray_view_xyz[:, 1:],
    )
    canvas[:, :-1] |= mark_a
    canvas[:, 1:] |= mark_b

    mark_a, mark_b = _sketch_pair_marks(
        semantic[:-1, :],
        semantic[1:, :],
        hit[:-1, :],
        hit[1:, :],
        depth_m[:-1, :],
        depth_m[1:, :],
        screen.normal_view_xyz[:-1, :],
        screen.normal_view_xyz[1:, :],
        screen.position_view_m[:-1, :],
        screen.position_view_m[1:, :],
        screen.ray_view_xyz[:-1, :],
        screen.ray_view_xyz[1:, :],
    )
    canvas[:-1, :] |= mark_a
    canvas[1:, :] |= mark_b

    _draw_opening_symbols(canvas, scene, buffers, view_matrix, proj_matrix, pose, sketch_symbols)
    sketch_u8 = np.where(canvas, SKETCH_FOREGROUND_U8, SKETCH_BACKGROUND_U8).astype(np.uint8)
    return _encode_png(Image.fromarray(sketch_u8, mode="L"))


def _plane_residual_ratio(
    normal_view_xyz: Float64Array,
    position_view_m: Float64Array,
    neighbour_ray_xyz: Float64Array,
    neighbour_depth_m: Float64Array,
    both_hit: npt.NDArray[np.bool_],
) -> Float64Array:
    """当前像素所在平面外推到邻居视线上，预测深度与邻居实测深度的相对残差；外推不成立
    （掠射、外推到相机背后、任一侧没打到几何）记 ``inf``。与 :func:`_depth_break` 是同一个
    几何量，只是这里要的是数不是布尔——控制稿要拿它分"相接"与"遮挡"。"""
    denom = np.einsum("hwi,hwi->hw", normal_view_xyz, neighbour_ray_xyz)
    numer = np.einsum("hwi,hwi->hw", normal_view_xyz, position_view_m)
    usable = both_hit & (np.abs(denom) > 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        predicted_m = np.where(usable, numer / np.where(usable, denom, 1.0), 0.0)
        safe_depth_m = np.where(both_hit & (neighbour_depth_m > 0.0), neighbour_depth_m, 1.0)
        ratio = np.abs(neighbour_depth_m - predicted_m) / safe_depth_m
    valid = usable & (predicted_m > 0.0) & np.isfinite(ratio)
    return np.asarray(np.where(valid, ratio, np.inf), dtype=np.float64)


def _sketch_pair_marks(
    semantic_a: npt.NDArray[np.int8],
    semantic_b: npt.NDArray[np.int8],
    hit_a: npt.NDArray[np.bool_],
    hit_b: npt.NDArray[np.bool_],
    depth_a_m: Float64Array,
    depth_b_m: Float64Array,
    normal_a_xyz: Float64Array,
    normal_b_xyz: Float64Array,
    position_a_m: Float64Array,
    position_b_m: Float64Array,
    ray_a_xyz: Float64Array,
    ray_b_xyz: Float64Array,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """一批相邻像素对 (a, b) → 哪些要在 a 侧标线、哪些要在 b 侧标线。

    先把每一对分成四种几何关系（互斥）：

    - **轮廓**：一侧打到几何、另一侧是背景；
    - **共面接缝**：两侧法向相同（:data:`SKETCH_SAME_PLANE_COS`）且相接——同一张面，或者
      被切成几块的同一面墙、两块共面的地板；
    - **折边**：相接但法向不同——两张面沿一条交线相接（墙角、地脚线、家具的棱）；
    - **遮挡边**：不相接——一前一后两张面，近的盖住远的（相接的判据见
      :data:`SKETCH_CONTIGUITY_TOLERANCE_RATIO`）。

    再按模块 docstring 的画法取舍：轮廓画（天花的轮廓除外）；遮挡边画（两侧都是地板的
    除外——地板全在同一高度，本来也遮不住地板）；折边画，但**两侧任一侧是天花的不画**
    （天花与墙的交线）、**两侧都是地板的不画**（房间分界线）；共面接缝一律不画。

    线标在**近的那一侧**（轮廓标在几何那一侧、遮挡边标在遮挡物那一侧、折边两侧深度几乎
    相同则标 a 侧）：这条线属于看得见的那张面，不属于被它盖住的东西。
    """
    both_hit = hit_a & hit_b
    silhouette = hit_a != hit_b
    same_plane = both_hit & (
        np.einsum("hwi,hwi->hw", normal_a_xyz, normal_b_xyz) >= (SKETCH_SAME_PLANE_COS)
    )
    residual_ab = _plane_residual_ratio(normal_a_xyz, position_a_m, ray_b_xyz, depth_b_m, both_hit)
    residual_ba = _plane_residual_ratio(normal_b_xyz, position_b_m, ray_a_xyz, depth_a_m, both_hit)
    contiguous = both_hit & (
        (residual_ab <= SKETCH_CONTIGUITY_TOLERANCE_RATIO)
        | (residual_ba <= SKETCH_CONTIGUITY_TOLERANCE_RATIO)
    )
    fold = contiguous & ~same_plane
    occlusion = both_hit & ~contiguous

    ceiling_code = _SKETCH_SEMANTIC_CODE["ceiling"]
    floor_code = _SKETCH_SEMANTIC_CODE["floor"]
    ceiling_involved = (semantic_a == ceiling_code) | (semantic_b == ceiling_code)
    both_floor = (semantic_a == floor_code) & (semantic_b == floor_code)
    visible_semantic = np.where(hit_a, semantic_a, semantic_b)

    draw = (
        (silhouette & (visible_semantic != ceiling_code))
        | (occlusion & ~both_floor)
        | (fold & ~ceiling_involved & ~both_floor)
    )
    choose_a = hit_a & (~hit_b | (depth_a_m <= depth_b_m))
    return np.asarray(draw & choose_a, dtype=np.bool_), np.asarray(draw & ~choose_a, dtype=np.bool_)


def _reveal_along_axis(mesh_id: str, verts_m: Float64Array, triangles: Float64Array) -> int:
    """洞口套框的宽度沿哪根轴：套框的两侧洞壁是竖直面、法向沿墙走向，找到一张竖直面即得。"""
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edge_a, edge_b)
    lengths = np.linalg.norm(normals, axis=1)
    for normal, length in zip(normals, lengths, strict=True):
        if length <= 0.0:
            continue
        unit = normal / length
        if abs(float(unit[2])) < 0.5:
            return 0 if abs(float(unit[0])) >= abs(float(unit[1])) else 1
    raise BaseRenderError(f"洞口套框没有竖直的洞壁面，定不出洞口方向：mesh_id={mesh_id}")


def _opening_kind_of_index(scene: ScenePackage) -> dict[str, str]:
    """洞口表：洞号（输入包里的下标，字符串形态同 id 里的写法）→ 最终种类。老包为空。"""
    kinds: dict[str, str] = {}
    for entry in scene.openings:
        if entry.kind not in _OPENING_KINDS:
            raise BaseRenderError(
                f"洞口表里第 {entry.opening_index} 个洞的种类是 {entry.kind}：控制稿没有它的符号"
            )
        kinds[str(entry.opening_index)] = entry.kind
    return kinds


def _kind_from_table_or(scene_kinds: dict[str, str], opening_index: str, fallback: str) -> str:
    """洞口表里有这个洞就用表里的种类；没有洞口表（老包）才用从网格 id 读出来的。

    表里有别的洞、独缺这一个，是场景包自己前后不一致，炸——不替它挑一种。
    """
    if not scene_kinds:
        return fallback
    kind = scene_kinds.get(opening_index)
    if kind is None:
        raise BaseRenderError(f"场景包有洞口表，但第 {opening_index} 个洞不在表里：网格与表对不上")
    return kind


def _opening_frames(scene: ScenePackage) -> list[_OpeningFrame]:
    """从场景包里读出每个洞口的框与种类（来路与 id 格式见模块 docstring）。

    次序写死：切出来的洞按套框网格在场景包里的次序，补出来的洞按过梁块在场景包里的次序、
    排在后面——次序进了符号的绘制序，绘制序进了像素（后画的覆盖先画的，虽然都是白）。
    """
    scene_kinds = _opening_kind_of_index(scene)
    frames: list[_OpeningFrame] = []
    fills: dict[str, dict[str, Float64Array]] = {}
    for mesh in scene.meshes:
        parts = mesh.id.split(":")
        if not mesh.vertices or not mesh.triangles:
            continue
        verts_m = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
        if parts[0] == "reveal" and len(parts) >= 2:
            id_kind = parts[1]
            if id_kind not in _OPENING_KINDS:
                raise BaseRenderError(
                    f"洞口套框的 id 里种类认不出：mesh_id={mesh.id}；认得的：{_OPENING_KINDS}"
                )
            kind = _kind_from_table_or(scene_kinds, parts[-1], id_kind)
            index = np.asarray(mesh.triangles, dtype=np.int64).reshape(-1, 3)
            along_axis = _reveal_along_axis(mesh.id, verts_m, verts_m[index])
            across_axis = 1 - along_axis
            frames.append(
                _OpeningFrame(
                    kind=kind,
                    along_axis=along_axis,
                    along_m=(
                        float(verts_m[:, along_axis].min()),
                        float(verts_m[:, along_axis].max()),
                    ),
                    across_m=(
                        float(verts_m[:, across_axis].min()),
                        float(verts_m[:, across_axis].max()),
                    ),
                    z_m=(float(verts_m[:, 2].min()), float(verts_m[:, 2].max())),
                )
            )
        elif len(parts) >= 4 and parts[0] == "wall" and parts[1] == "fill":
            fills.setdefault(parts[2], {})[parts[3]] = verts_m

    for opening_index, blocks in fills.items():
        lintel_m = blocks.get("lintel")
        if lintel_m is None:
            # 过梁块薄到没起体（洞顶就是天花）：这个洞从地面通到天花，没有框可挂符号，
            # 洞口轮廓本身照样由折边与遮挡边画出来。
            continue
        extent_m = lintel_m.max(axis=0) - lintel_m.min(axis=0)
        # 过梁块的两条水平边里长的是洞宽、短的是墙厚——补出来的块厚度抄的是邻墙
        # （真户型实测墙厚 0.1～0.3 m，门洞宽 0.7～1.0 m），洞比墙厚是户型的常态。
        along_axis = 0 if float(extent_m[0]) >= float(extent_m[1]) else 1
        across_axis = 1 - along_axis
        sill_m = blocks.get("sill")
        z_bottom_m = float(sill_m[:, 2].max()) if sill_m is not None else 0.0
        frames.append(
            _OpeningFrame(
                kind=_kind_from_table_or(
                    scene_kinds, opening_index, "window" if sill_m is not None else "door"
                ),
                along_axis=along_axis,
                along_m=(
                    float(lintel_m[:, along_axis].min()),
                    float(lintel_m[:, along_axis].max()),
                ),
                across_m=(
                    float(lintel_m[:, across_axis].min()),
                    float(lintel_m[:, across_axis].max()),
                ),
                z_m=(z_bottom_m, float(lintel_m[:, 2].min())),
            )
        )
    return frames


def _opening_symbol_segments(
    frame: _OpeningFrame,
    scheme: SketchSymbolScheme = DEFAULT_SKETCH_SYMBOLS,
    viewer_across_sign: float = 1.0,
) -> list[_Segment]:
    """一个洞口的符号线段（世界系）。画法按方案，全文见 :data:`SKETCH_SYMBOL_SCHEMES`。
    过口任何方案都不画；认不出的种类或方案炸，不静默画成空。

    ``viewer_across_sign``：相机在墙厚方向的哪一侧（+1 ＝ 沿墙厚轴坐标大的那侧）。只有窗台线
    （画在朝相机那一面）与 ``leaf-swing`` 的门扇（朝相机开）用它；其余线段在墙厚中心平面上。
    """
    if scheme not in SKETCH_SYMBOL_SCHEMES:
        raise BaseRenderError(f"控制稿符号方案认不出：{scheme}；认得的：{SKETCH_SYMBOL_SCHEMES}")
    if frame.kind == "passage":
        return []
    if frame.kind == "window":
        return _window_symbol_segments(frame, scheme, viewer_across_sign)
    if frame.kind in ("door", "entry-door"):
        return _door_symbol_segments(frame, scheme, viewer_across_sign)
    raise BaseRenderError(f"洞口种类 {frame.kind} 没有控制稿符号；认得的：{_OPENING_KINDS}")


def _frame_point(frame: _OpeningFrame, along_m: float, z_m: float, across_m: float) -> Float64Array:
    xyz = np.zeros(3, dtype=np.float64)
    xyz[frame.along_axis] = along_m
    xyz[1 - frame.along_axis] = across_m
    xyz[2] = z_m
    return xyz


def _rect_segments(
    frame: _OpeningFrame,
    along_m: tuple[float, float],
    z_m: tuple[float, float],
    with_bottom: bool,
) -> list[_Segment]:
    """墙厚中心平面上的矩形：两根竖边 + 顶边，``with_bottom`` 才有底边（门的底边在地面上，
    那是门槛线，不画）。矩形退化（宽或高不为正）就一条也不画。"""
    (a0, a1), (z0, z1) = along_m, z_m
    if a1 <= a0 or z1 <= z0:
        return []
    c = frame.across_center_m
    segments: list[_Segment] = [
        (_frame_point(frame, a0, z0, c), _frame_point(frame, a0, z1, c)),
        (_frame_point(frame, a1, z0, c), _frame_point(frame, a1, z1, c)),
        (_frame_point(frame, a0, z1, c), _frame_point(frame, a1, z1, c)),
    ]
    if with_bottom:
        segments.append((_frame_point(frame, a0, z0, c), _frame_point(frame, a1, z0, c)))
    return segments


def _inner_rect_m(
    frame: _OpeningFrame,
) -> tuple[tuple[float, float], tuple[float, float]]:
    (a0, a1), (z0, z1) = frame.along_m, frame.z_m
    inset = SKETCH_FRAME_INSET_M
    return (a0 + inset, a1 - inset), (z0 + inset, z1 - inset)


def _sill_segment(frame: _OpeningFrame, viewer_across_sign: float) -> _Segment:
    """窗台线：洞下沿再向下一小段、两端各伸出洞宽一点，画在朝相机那一面的墙面上（画在墙厚
    中心平面会被窗下墙自己挡住）。"""
    (a0, a1), z0 = frame.along_m, frame.z_m[0]
    face_m = frame.across_m[1] if viewer_across_sign >= 0.0 else frame.across_m[0]
    z_m = z0 - SKETCH_SILL_DROP_M
    return (
        _frame_point(frame, a0 - SKETCH_SILL_OVERHANG_M, z_m, face_m),
        _frame_point(frame, a1 + SKETCH_SILL_OVERHANG_M, z_m, face_m),
    )


def _window_symbol_segments(
    frame: _OpeningFrame, scheme: SketchSymbolScheme, viewer_across_sign: float
) -> list[_Segment]:
    (a0, a1), (z0, z1) = frame.along_m, frame.z_m
    c = frame.across_center_m
    if scheme == "diagonal-cross":
        mid_along_m = (a0 + a1) * 0.5
        mid_z_m = (z0 + z1) * 0.5
        return [
            (_frame_point(frame, mid_along_m, z0, c), _frame_point(frame, mid_along_m, z1, c)),
            (_frame_point(frame, a0, mid_z_m, c), _frame_point(frame, a1, mid_z_m, c)),
        ]
    inner_along_m, inner_z_m = _inner_rect_m(frame)
    segments = _rect_segments(frame, frame.along_m, frame.z_m, with_bottom=True)
    segments += _rect_segments(frame, inner_along_m, inner_z_m, with_bottom=True)
    if scheme in ("frame-sill", "frame-handle"):
        segments.append(_sill_segment(frame, viewer_across_sign))
        return segments
    if scheme == "glazing-hatch":
        (g0, g1), (gz0, gz1) = inner_along_m, inner_z_m
        width_m, height_m = g1 - g0, gz1 - gz0
        if width_m <= 0.0 or height_m <= 0.0:
            return segments
        step_m = SKETCH_HATCH_LENGTH_RATIO * min(width_m, height_m) * math.cos(math.radians(45.0))
        for along_ratio in (0.18, 0.34, 0.50):
            start_along_m = g0 + width_m * along_ratio
            start_z_m = gz0 + height_m * 0.55
            segments.append(
                (
                    _frame_point(frame, start_along_m, start_z_m, c),
                    _frame_point(frame, start_along_m + step_m, start_z_m + step_m, c),
                )
            )
        return segments
    if scheme == "leaf-swing":
        segments.append(_sill_segment(frame, viewer_across_sign))
        mid_along_m = (a0 + a1) * 0.5
        segments.append(
            (_frame_point(frame, mid_along_m, z0, c), _frame_point(frame, mid_along_m, z1, c))
        )
        return segments
    raise BaseRenderError(f"控制稿符号方案认不出：{scheme}；认得的：{SKETCH_SYMBOL_SCHEMES}")


def _door_leaf_line_segments(frame: _OpeningFrame) -> list[_Segment]:
    """门扇线：离沿墙坐标小的那侧洞边 :data:`SKETCH_DOOR_LEAF_OFFSET_M` 的一条通高竖线，
    墙厚中心平面上；洞比这个偏移还窄就不画。"""
    (a0, a1), (z0, z1) = frame.along_m, frame.z_m
    c = frame.across_center_m
    leaf_along_m = a0 + SKETCH_DOOR_LEAF_OFFSET_M
    if leaf_along_m >= a1:
        return []
    return [(_frame_point(frame, leaf_along_m, z0, c), _frame_point(frame, leaf_along_m, z1, c))]


def _door_handle_segments(frame: _OpeningFrame) -> list[_Segment]:
    """门把手：离沿墙坐标大的那侧洞边 :data:`SKETCH_DOOR_HANDLE_EDGE_M`、长
    :data:`SKETCH_DOOR_HANDLE_LENGTH_M`、离地 :data:`SKETCH_DOOR_HANDLE_HEIGHT_M` 的一条短横，
    墙厚中心平面上；洞太窄或太矮放不下就不画。"""
    (a0, a1), (z0, z1) = frame.along_m, frame.z_m
    c = frame.across_center_m
    handle_z_m = z0 + SKETCH_DOOR_HANDLE_HEIGHT_M
    handle_to_m = a1 - SKETCH_DOOR_HANDLE_EDGE_M
    handle_from_m = max(a0, handle_to_m - SKETCH_DOOR_HANDLE_LENGTH_M)
    if handle_from_m >= handle_to_m or handle_z_m >= z1:
        return []
    return [
        (
            _frame_point(frame, handle_from_m, handle_z_m, c),
            _frame_point(frame, handle_to_m, handle_z_m, c),
        )
    ]


def _door_symbol_segments(
    frame: _OpeningFrame, scheme: SketchSymbolScheme, viewer_across_sign: float
) -> list[_Segment]:
    (a0, a1), (z0, z1) = frame.along_m, frame.z_m
    c = frame.across_center_m
    if scheme == "diagonal-cross":
        return [(_frame_point(frame, a0, z0, c), _frame_point(frame, a1, z1, c))]
    segments = _rect_segments(frame, frame.along_m, frame.z_m, with_bottom=False)
    if scheme in ("frame-sill", "frame-handle"):
        segments += _door_leaf_line_segments(frame)
    if scheme in ("glazing-hatch", "frame-handle"):
        segments += _door_handle_segments(frame)
    if scheme in ("frame-sill", "glazing-hatch", "frame-handle"):
        return segments
    if scheme == "leaf-swing":
        width_m = a1 - a0
        swing = math.radians(SKETCH_DOOR_SWING_DEG)
        sign = 1.0 if viewer_across_sign >= 0.0 else -1.0
        free_along_m = a0 + width_m * math.cos(swing)
        free_across_m = c + sign * width_m * math.sin(swing)
        hinge_bottom = _frame_point(frame, a0, z0, c)
        hinge_top = _frame_point(frame, a0, z1, c)
        free_bottom = _frame_point(frame, free_along_m, z0, free_across_m)
        free_top = _frame_point(frame, free_along_m, z1, free_across_m)
        segments += [(hinge_bottom, free_bottom), (hinge_top, free_top), (free_bottom, free_top)]
        return segments
    raise BaseRenderError(f"控制稿符号方案认不出：{scheme}；认得的：{SKETCH_SYMBOL_SCHEMES}")


def _draw_opening_symbols(
    canvas: npt.NDArray[np.bool_],
    scene: ScenePackage,
    buffers: RasterBuffers,
    view_matrix: Float64Array,
    proj_matrix: Float64Array,
    pose: CameraPose,
    scheme: SketchSymbolScheme,
) -> None:
    for frame in _opening_frames(scene):
        eye_across_m = float(pose.eye_m[1 - frame.along_axis])
        viewer_across_sign = 1.0 if eye_across_m >= frame.across_center_m else -1.0
        for start_m, end_m in _opening_symbol_segments(frame, scheme, viewer_across_sign):
            _draw_segment(
                canvas, start_m, end_m, view_matrix, proj_matrix, buffers, pose.near_clip_m
            )


def _draw_segment(
    canvas: npt.NDArray[np.bool_],
    start_m: Float64Array,
    end_m: Float64Array,
    view_matrix: Float64Array,
    proj_matrix: Float64Array,
    buffers: RasterBuffers,
    near_clip_m: float,
) -> None:
    """把一条世界系线段画进画布：近平面裁剪 → 投影 → 按画幅裁剪 → 逐像素采样 → 深度测试。

    深度沿线段按 1/w 线性插值（屏幕上线性的是 1/w，同光栅那一路的口径）。深度测试对着
    1 倍缓冲：比缓冲深的点被挡住，不画——所以被墙挡住的洞口，符号也被挡住。
    采样步长一像素，点数由裁剪后的屏幕长度定，全程无随机。
    """
    height_px, width_px = canvas.shape
    points_m = np.stack([start_m, end_m]).astype(np.float64)
    homogeneous = np.concatenate([points_m, np.ones((2, 1), dtype=np.float64)], axis=1)
    view_xyz = (homogeneous @ view_matrix.T)[:, :3]
    signed_m = -view_xyz[:, 2] - near_clip_m
    inside = signed_m >= 0.0
    if not bool(inside.any()):
        return
    if not bool(inside.all()):
        kept, cut = (0, 1) if bool(inside[0]) else (1, 0)
        ratio = float(signed_m[kept] / (signed_m[kept] - signed_m[cut]))
        view_xyz[cut] = view_xyz[kept] + ratio * (view_xyz[cut] - view_xyz[kept])

    clip = np.concatenate([view_xyz, np.ones((2, 1), dtype=np.float64)], axis=1) @ proj_matrix.T
    w_m = clip[:, 3]
    if not bool(np.all(w_m > 0.0)) or not bool(np.isfinite(clip).all()):
        return
    x_px = (clip[:, 0] / w_m + 1.0) * 0.5 * width_px
    y_px = (1.0 - clip[:, 1] / w_m) * 0.5 * height_px

    # Liang–Barsky：把参数 t 裁到画幅内，裁剪后的 t 仍是屏幕空间的参数，1/w 照样按它线性插
    dx_px, dy_px = float(x_px[1] - x_px[0]), float(y_px[1] - y_px[0])
    t_from, t_to = 0.0, 1.0
    for p, q in (
        (-dx_px, float(x_px[0])),
        (dx_px, float(width_px) - float(x_px[0])),
        (-dy_px, float(y_px[0])),
        (dy_px, float(height_px) - float(y_px[0])),
    ):
        if p == 0.0:
            if q < 0.0:
                return
            continue
        ratio = q / p
        if p < 0.0:
            t_from = max(t_from, ratio)
        else:
            t_to = min(t_to, ratio)
    if t_from > t_to:
        return

    length_px = math.hypot(dx_px, dy_px) * (t_to - t_from)
    sample_count = max(2, int(math.ceil(length_px)) + 1)
    t = t_from + (t_to - t_from) * np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    column = np.clip(np.floor(x_px[0] + t * dx_px).astype(np.int64), 0, width_px - 1)
    row = np.clip(np.floor(y_px[0] + t * dy_px).astype(np.int64), 0, height_px - 1)
    inverse_w = (1.0 - t) / w_m[0] + t / w_m[1]
    sample_depth_m = 1.0 / inverse_w

    buffer_depth_m = buffers.depth_m[row, column].astype(np.float64)
    visible = ~np.isfinite(buffer_depth_m) | (
        sample_depth_m <= buffer_depth_m * (1.0 + SKETCH_SYMBOL_DEPTH_TOLERANCE_RATIO)
    )
    canvas[row[visible], column[visible]] = True


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
    "DEFAULT_SKETCH_SYMBOLS",
    "SKETCH_SYMBOL_SCHEMES",
    "BaseRenderError",
    "CameraPose",
    "RoomViewCheck",
    "SketchSymbolScheme",
    "render_base_views",
    "resolve_camera_pose",
]
