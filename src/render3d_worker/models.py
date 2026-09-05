"""render3d_worker 的数据契约：三维这条服务要什么数据、交出什么东西。

**服务之间只有数据通信**（用户裁决 2026-08-31）：上游（定稿平面、三维资产库）今天
都不存在，本仓不等它们——把"三维需要什么"写成一份输入包契约，开发期喂拟真包
（`tests/fixtures/`），真派发时同一份包由上游填。上下游是通信关系，不是调用关系：
activity 收对象键、CLI 收 JSON 文件，两条路吃的是同一个 :class:`DesignPackage`。

跨 domain 纪律：worker 不 import 其他 domain 的内部模块。本模块里的几何那一族
（:class:`PlanWall` 起到 :class:`FloorplanGeometry`）是**产出侧那份的逐字对面**——
两个仓两种语言谁也不能 import 谁，对不上就是接不上头。字段名与口径改自
ishome-render2d 同名模型（同一个产出侧），改动只有一处：本仓不消费 `PlanNote`
一族（图上的字归 2D）。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class _Contract(BaseModel):
    """契约基类：camelCase 别名对齐产出侧的序列化；`extra=forbid` 拒收多出来的字段。

    多出来的字段不是"宽容一点收下"的事——它说明两侧对不上头，早一步炸在边界上，
    好过带着一个没人读的字段一路走到出图（同报告册对象键那条纪律）。
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


# ---------------------------------------------------------------------------
# 一、上游那半：户型几何（归一化，**没有任何绝对尺寸**）
# ---------------------------------------------------------------------------

PlanAxis = Literal["vertical", "horizontal"]


class PlanWallBand(_Contract):
    """墙上的一段实测墙带：同段内一个厚度，墨宽一变就换段（产出侧 2026-09-01 起给）。

    `start_ratio`/`end_ratio` 沿墙方向，与 :class:`PlanWall` 的起讫同分母；`face_low_ratio`/
    `face_high_ratio` 是墙带两面的位置（与 `position_ratio` 同分母），厚度＝两面之差。

    **本仓今天只收不消费**：起体仍按 :attr:`PlanWall.thickness_ratio`（整条线一个厚度）。
    收下它是因为几何那一族是产出侧的逐字对面（`extra=forbid`），产出侧写了这个字段、
    这儿不认就读不进来。按段厚度起体的时点＝墙体按段起体那一批（要用户拍）。
    """

    start_ratio: float
    end_ratio: float
    face_low_ratio: float
    face_high_ratio: float


class PlanWall(_Contract):
    """一段墙：轴向、所在位置、起讫、墙厚，全部归一化到整图（0~1，左上角为原点）。

    `axis` 为 `vertical` 时 `position_ratio` 是 x、`start_ratio`/`end_ratio` 是 y 的起讫，
    **`thickness_ratio` 按图宽归一**；`horizontal` 时全部反过来（厚度按图高归一）。
    两个方向除的不是同一个数，所以还原形状必须用 :attr:`FloorplanGeometry.frame_width_px`。
    """

    axis: PlanAxis
    position_ratio: float
    start_ratio: float
    end_ratio: float
    thickness_ratio: float
    bands: list[PlanWallBand] = Field(default_factory=list)


OpeningKind = Literal["door", "window", "passage", "entry-door", "unknown"]
"""洞口类型闭集，**与产出侧逐字一致**（aipipe `models.OpeningKind`，2026-09-05）。

`passage` 是没有门扇的过口；`entry-door` 是入户门（外轮廓上带门弧的洞，全户唯一）；
`unknown` 是**推不出**，不是"大概是门"——三维遇到它才退到 :class:`HeightRules` 的档位猜法，
并把"猜了"这件事带进场景包（:attr:`ScenePackage.guessed_opening_count`）。

版本对照：2026-09-05 之前本仓的闭集是 `door | window | pass`，`pass` 改名 `passage`
（跟产出侧的字），`entry-door` 与 `unknown` 是同一天新加的。`HeightRules.pass_height_m`
这个字段名**没改**：它是上游填的那半的字段，值的含义（过口净高）不变，改名只会让填包那侧
再对一次表。
"""

GuessedOpeningKind = Literal["door", "window", "passage"]
"""档位猜法能猜出来的种类：只有这三种有各自的高度档位。`unknown` 不许作猜的结果
（猜出一个"不知道"等于没猜），`entry-door` 也不许（入户门全户唯一，猜不出来）。"""


class PlanOpening(_Contract):
    """墙线上的一个洞：坐标口径同 :class:`PlanWall`。

    `kind` 是产出侧 2026-09-05 起给的洞口类型（由确定性代码从像素里推：门弧、跨洞平行线；
    推不出即 `unknown` 并在 `kind_evidence` 里写为什么）。三维绕不开类型——墙上挖多高的洞
    取决于它是门还是窗——所以本仓**先读 `kind`**，只有 `unknown` 才退到 :class:`HeightRules`
    里的档位猜法（外墙＝窗、内墙＝门），推法写在 mesh ``resolve_opening_kind`` 一处。

    **缺省是 `unknown` 不是 `door`**：没跑过类型推断的老产物读进来就是"不知道"，
    走猜法并计数，而不是悄悄全当门。
    """

    axis: PlanAxis
    position_ratio: float
    start_ratio: float
    end_ratio: float
    is_on_outer_wall: bool
    connects: list[str] = Field(default_factory=list)
    """这个洞两侧的房间名。三维用得上：门洞的两侧地面要连起来，遮罩才不会把一户切成孤岛。"""

    kind: OpeningKind = "unknown"
    kind_evidence: str = ""
    """给出 `kind` 的依据（哪样证据、量到多少）；`unknown` 时写的是为什么推不出。
    给人读的，不进规则。"""


class RoomOutline(_Contract):
    """一个房间占的地方：若干矩形块拼起来（房间不一定是矩形，L 形很常见）。

    `boxes` 拼起来就是房间遮罩，三维里它就是这间房的地板与吊顶轮廓；`centroid` 是锚点。
    `area_ratio` 是占户型内部自由面积的比例，**不是面积**——面积要靠 :class:`PlanScale`。
    """

    name: str
    boxes: list[tuple[float, float, float, float]] = Field(default_factory=list)
    area_ratio: float = 0.0
    centroid: tuple[float, float] = (0.0, 0.0)


class FloorplanGeometry(_Contract):
    """三维的唯一几何来源：轮廓、墙、洞、房间遮罩，**没有任何绝对尺寸**。

    `outline` 是户型外轮廓那一圈墙，与 `walls` 分列：`walls` 是网格投票出来的线，投不上
    票的外墙不在里面（飘窗那种墙往外折一个台阶的段）。**两处都当墙起体**，重合的段会生成
    两块重叠的墙体——底渲只看最近面，重叠不影响四路输出，故不做布尔求并（省一整套
    数值稳健性问题；真要出交互引擎用的干净网格时再做，时点＝场景包接进交互引擎那一批）。

    `frame_*_px` 是这一整套比例的参照系（像素）。缺了它形状就是错的：x 按图宽归一、
    y 按图高归一，一张长方形的户型会被拉成正方形。
    """

    frame_width_px: int
    frame_height_px: int
    plan_box: tuple[float, float, float, float]
    outline: list[PlanWall] = Field(default_factory=list)
    walls: list[PlanWall] = Field(default_factory=list)
    openings: list[PlanOpening] = Field(default_factory=list)
    rooms: list[RoomOutline] = Field(default_factory=list)
    cell_coverage_ratio: float = 0.0
    opening_kind_coverage_ratio: float = 0.0
    """产出侧的自证数：非 `unknown` 类型的洞占全部洞的比例。老产物没有这个数，读作 0。"""


# ---------------------------------------------------------------------------
# 二、三维要、二维没有的那半：尺度、高度、家具、材质、相机
# ---------------------------------------------------------------------------


class PlanScale(_Contract):
    """把归一化几何换成米的那把尺子。

    几何一个绝对尺寸都没有，而三维里"墙 2.8 米高"必须有米。尺子由**面积反推**：
    套内面积 = 建筑面积 × 得房率，再与 `plan_box` 框住的归一化面积相除得米/单位。
    面积是上游真有的数（匿名画像带建筑面积与得房率），比例尺不是——**不许由模型给**。

    `usable_area_percent` 取百分数不取小数（80 不是 0.8），同 contracts 数据包那条口径：
    少一层换算就少一处会漂移的口径。
    """

    building_area_sqm: float
    usable_area_percent: float = 80.0


class HeightRules(_Contract):
    """竖向那一维的全部数字，集中在一处，**没有一个散在代码里**。

    **这一整段是 mock，不是这户人家的实测值**（用户裁决 2026-08-31 夜：*"高度的话，我觉得
    我们可以按照常规住宅的高度做一个 mock，然后先用来做测试，后续的话我们要求上游给发过来"*）。
    下面的默认值是**常规住宅档位**：户型图上量不到高度——那是立面的事，平面图一个字都不说，
    所以在上游给出这户的实测高度之前，三维只能按常规档位起体，否则墙起不来、图出不来。

    **上游什么时候必须给**（触发条件写死，不留悬空）：楼盘资料或业主自己报出这套房的层高时，
    以它为准；那一步到位之前，本段一直是 mock。**用没用上默认值看得见**——包里不写这一段
    就是吃默认，场景包的 `heights_source` 会写着 `mock-default`（见 :class:`ScenePackage`）。

    集中在包里的理由：换一户、换一个楼盘只换数据、代码不动；也让"这张图为什么是这个高度"
    答得出来。

    单位口径**是净高不是层高**：`ceiling_height_m` 是地面完成面到天花完成面（三维里墙就起
    这么高），层高还要加楼板与地面做法。上游给的若是层高，减掉 `slab_thickness_m` 与地面做法
    再进这个字段——**换算在填包那一侧做，不在本仓做**（本仓不认识"层高"这个词）。
    """

    ceiling_height_m: float = 2.80
    slab_thickness_m: float = 0.12
    door_height_m: float = 2.05
    pass_height_m: float = 2.20
    window_sill_height_m: float = 0.90
    window_head_height_m: float = 2.10
    outer_opening_kind: GuessedOpeningKind = "window"
    """产出侧给不出类型（`kind == "unknown"`）时，外墙上的洞按什么算；内墙洞按
    `inner_opening_kind`。**这是退路不是主路**：产出侧给了 `kind` 的洞不经这两条。"""

    inner_opening_kind: GuessedOpeningKind = "door"


class FurnishingPlacement(_Contract):
    """一件家具摆在哪儿：位置用归一化平面坐标，尺寸用米。

    位置跟着几何走（归一化），尺寸跟着现实走（米）——两套单位并存是有意的：布置是
    平面上的决定，体量是产品的事实。`yaw_deg` 绕竖轴，0 度朝屏幕下方（+y），逆时针为正。
    """

    id: str
    category: str
    room: str
    center_x_ratio: float
    center_y_ratio: float
    width_m: float
    depth_m: float
    height_m: float
    yaw_deg: float = 0.0


class SurfaceMaterial(_Contract):
    """一种材质：底渲只用得上颜色，其余留给交互引擎与写实化那一步。

    底渲四路（几何/深度/线稿/遮罩）里颜色只决定"几何"那一路的分色；`roughness_ratio`
    与 `metallic_ratio` 本仓不消费，随场景包原样带出——它们是交互引擎和 realism-pass
    的输入，在这儿丢掉就得让上游再传一遍。
    """

    id: str
    base_color_hex: str
    roughness_ratio: float = 0.8
    metallic_ratio: float = 0.0


MaterialSurface = Literal["floor", "ceiling", "wall", "furnishing"]


class MaterialAssignment(_Contract):
    """哪个面用哪种材质：按 `surface` + 房间（或家具类别）指派，都不指定即全局兜底。"""

    surface: MaterialSurface
    material_id: str
    room: str | None = None
    category: str | None = None


CameraKind = Literal["bird", "room"]


class CameraSpec(_Contract):
    """一台相机。`bird` 俯瞰整户（`room` 留空），`room` 站在某间房里平视。

    相机是**输入不是产物**：换一个机位不该重编场景包（编场景包是几何活，摆相机是取景活）。
    底渲按 `camera_id` 取其中一台，一次出五路图。
    """

    id: str
    kind: CameraKind
    room: str | None = None
    eye_height_m: float = 1.55
    yaw_deg: float = 0.0
    pitch_deg: float = -30.0
    fov_deg: float = 55.0


class DesignPackage(_Contract):
    """**scene-compile 的输入：三维要的全部数据，一个包。**

    这就是"我们需要什么数据"那份清单的可执行形态。开发期由 `tests/fixtures/` 里的
    拟真包喂（同报告线三档考卷那套做法：卷子固定，跨天可比）；真派发时由上游填同一份包，
    本仓一行代码不改。缺哪一块就是上游哪一块没做出来，**不许在本仓编一个值补上**。
    """

    revision_id: str
    plan: FloorplanGeometry
    scale: PlanScale
    heights: HeightRules = Field(default_factory=HeightRules)
    furnishings: list[FurnishingPlacement] = Field(default_factory=list)
    materials: list[MaterialAssignment] = Field(default_factory=list)
    surface_materials: list[SurfaceMaterial] = Field(default_factory=list)
    cameras: list[CameraSpec] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 三、场景包：scene-compile 的产物，base-render 的输入
# ---------------------------------------------------------------------------

MeshSemantic = Literal["floor", "ceiling", "wall", "reveal", "furnishing"]

OpeningKindSource = Literal["upstream", "guessed"]


class SceneOpening(_Contract):
    """场景包里一个洞的身份：输入里的第几个、最终按什么种类起的体、这个种类是谁定的。

    `kind` 永远不是 `unknown`——起体必须有一个种类，上游没给就按档位猜，猜了记 `guessed`。
    `placed` 为 False 的洞没落到任何一道墙上（同直线上一段墙都没有），没起体、不计入
    `opening_count_by_kind`。
    """

    opening_index: int
    kind: OpeningKind
    kind_source: OpeningKindSource
    placed: bool = True


class Mesh(_Contract):
    """一块三角网格，米制右手系：x 向右、y 向里、z 向上。

    `semantic` 与 `room` 不是装饰——遮罩那一路按它们上色，交互引擎按它们分层。
    每块网格自带身份（`id`），底渲的遮罩索引表回指它，"图上这一块是什么"答得出来。
    """

    id: str
    semantic: MeshSemantic
    material_id: str
    room: str | None = None
    vertices: list[tuple[float, float, float]] = Field(default_factory=list)
    triangles: list[tuple[int, int, int]] = Field(default_factory=list)


class ScenePackage(_Contract):
    """场景包：一户人家的三维形态 + 材质 + 机位，**米制、确定性、不含图像**。

    自证数（`floor_area_sqm` 起四个）是本仓自己算给自己看的：编出来的地板面积对不上
    输入面积，就是尺子或几何错了，早于出图暴露。**不设死阈值**——门槛要有真跑数据才定
    （《纪律·阈值有数据才定》），今天先算出来、随包带出、在 CLI 上打出来。
    """

    revision_id: str
    unit: Literal["m"] = "m"
    meshes: list[Mesh] = Field(default_factory=list)
    materials: list[SurfaceMaterial] = Field(default_factory=list)
    cameras: list[CameraSpec] = Field(default_factory=list)
    bounds_min_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds_max_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    heights_source: Literal["upstream", "mock-default"] = "mock-default"
    """竖向那些数字是上游给的，还是吃了 :class:`HeightRules` 的常规住宅档位。

    **mock 与真值要在产出上分得开**（同报告线"draft 或已过期落点进正文、同页必须挂依据标注"
    那条口径）：一张图是按这户的实测层高起的，还是按常规档位起的，看图的人有权知道。
    判据＝上游填包时写没写 `heights` 这一段，不看值长什么样——值碰巧等于默认值不代表它是猜的。
    """

    scale_anchor_source: Literal["outline", "plan-box"] = "outline"
    """尺子拿哪块面积当锚：外轮廓围合面积（对的那个），还是退回了外接框。

    退回不是错，静默才是错——外接框是**外接**的，户型不是满铺矩形时它比套内建筑面积大，
    尺寸会整体偏小。外轮廓是 2026-08-31 才在产出侧补上的，老产物没有它，所以退路要留、
    退了要看得见。**随包带出而不是只打日志**：日志会丢，而下游拿到一份场景包时要能回答
    "这户的尺寸是按什么定的"。
    """

    metre_per_unit: float = 0.0
    """归一化 1.0 等于多少米（由 :class:`PlanScale` 反推）。换算只在编场景包时发生一次。"""

    floor_area_sqm: float = 0.0
    """所有地板网格加起来的面积——与输入套内面积的差距即 `area_match_ratio`。"""

    area_match_ratio: float = 0.0
    wall_segment_count: int = 0
    degenerate_wall_count: int = 0
    """上游给了几段零长墙线（起讫相同）——三维起体时跳过了它们。

    2D 那侧画一条零长的线等于没画，所以这件事只有三维会撞上。数目异常就是上游那一步坏了。
    """

    opening_count_by_kind: dict[str, int] = Field(default_factory=dict)
    """真做出来的洞按**最终**种类报数（上游给的与猜出来的合在一起数；猜了几个看下面两个字段）。"""

    openings: list[SceneOpening] = Field(default_factory=list)
    """每个洞的最终种类与来源，按输入次序。控制稿按它画门窗符号；老场景包没有这张表。"""

    guessed_opening_count: int = 0
    guessed_opening_indices: list[int] = Field(default_factory=list)
    """按档位猜的洞有几个、是哪几个（输入包 `plan.openings` 里的下标）。

    上游没给类型（`unknown`）的洞才会猜；猜了就要看得见——一张图上的门是上游认出来的
    还是我们按内墙猜的，看图的人有权知道（口径同 `heights_source`）。
    """

    triangle_count: int = 0


# ---------------------------------------------------------------------------
# 四、底渲的产物：遮罩索引表、取景自证数、五路图（纯库与 CLI 的返回值）
# ---------------------------------------------------------------------------
#
# 两个 activity 的请求/回执不在这里：那是编排与本仓之间的契约，归 activity_models
# （import-linter 把两份锁在同一层、互不可见）。这一节曾放过一版带档位（preview/final）的
# 请求模型，随用户裁决 2026-09-04（两档合一档）与 activity 实装（2026-09-05）删除。


class MaskEntry(_Contract):
    """遮罩索引表的一行：图上这个索引值是哪块网格、什么语义、哪间房。"""

    index: int
    mesh_id: str
    semantic: MeshSemantic
    room: str | None = None
    pixel_count: int = 0


class RoomViewCheck(_Contract):
    """``room`` 机位取景的自证数：这台机位最终站在哪儿、画面里目标房间占多少。

    数是按底渲自己的低分辨率深度/遮罩渲染量出来的（取景那一步的候选评估，见
    base_render ``resolve_camera_pose``），**不是最终那张图上的数**——两者分辨率不同，
    差在小数点后两位；要最终图上的数，从 ``mask_index`` 按房间加像素数即可。
    随产物带出的理由同其他自证数：一张室内图"是不是在拍这间房"要答得出来。
    """

    eye_m: tuple[float, float, float]
    yaw_deg: float
    min_depth_m: float
    """画面里最近的几何离镜头多远（米）。小于避墙距离就是站进了墙里或贴着墙。"""

    target_floor_ratio: float
    """目标房间**地板**像素占整幅的比例。"""

    target_room_ratio: float
    other_room_ratio: float
    """目标房间 / 其他房间的地板 + 天花像素各占整幅的比例。墙不归任何房间，不计入。"""

    dominance_ratio: float
    """目标房间在"有房间归属的像素"里占的比例：target ÷ (target + other)，两者都为 0 记 0。"""

    candidate_count: int
    """评估了几个候选位姿。上游显式给了 yaw 时不做候选评估，记 1（只量不判）。"""

    passed: bool
    """按取景判据（避墙距离、地板占比、主体占比）达标没有。自动取景选出来的必然为
    True（不达标就已经响亮失败）；上游显式给 yaw 的机位只量不判，False 也照渲。"""


class BaseRenderViews(BaseModel):
    """底渲一次的全部产物：五张图 + 遮罩索引表 + 自证数。

    不落桶、不签链接——落桶是 activity 那一层的事，本模型是纯库那条路的返回值
    （CLI 直接把它写到本地目录）。签名更是业务侧的事（"给谁看、看多久"）。
    """

    geometry_png: bytes
    depth_png: bytes
    line_png: bytes
    mask_png: bytes
    sketch_png: bytes
    """控制稿：给"线稿生图"控制通道画的那一路（与 ``line_png`` 同尺寸同编码，画法不同，
    分工见 base_render 模块 docstring）。"""

    width_px: int
    height_px: int
    camera_id: str
    mask_index: list[MaskEntry] = Field(default_factory=list)
    covered_pixel_ratio: float = 0.0
    """画面里被几何盖住的比例——整张几乎全空说明相机摆错了，失败要响亮。"""

    near_m: float = 0.0
    far_m: float = 0.0
    """深度图的两端（米）。深度是 16 位归一化存的，没有这两个数就还原不回米。"""

    room_view: RoomViewCheck | None = None
    """``room`` 机位的取景自证数；``bird`` 机位为 ``None``。"""
