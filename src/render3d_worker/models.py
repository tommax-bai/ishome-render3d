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

RenderTier = Literal["preview", "final"]
"""渲染两档：失效传播默认只重算 preview，final 由用户显式请求或交付节点触发。"""


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


class PlanOpening(_Contract):
    """墙线上的一个洞：坐标口径同 :class:`PlanWall`。

    **产出侧这一层不分门与窗**，只分洞在外墙还是内墙。三维绕不开这件事——墙上挖多高的
    洞取决于它是门还是窗，所以本仓按 :class:`HeightRules` 里写死的规则从
    `is_on_outer_wall` 推出 :data:`OpeningKind`，**推法是确定性的、写在一处**，
    产出侧补上门窗识别那一批之后改这一处即可（那时本字段升为上游直给）。
    """

    axis: PlanAxis
    position_ratio: float
    start_ratio: float
    end_ratio: float
    is_on_outer_wall: bool
    connects: list[str] = Field(default_factory=list)
    """这个洞两侧的房间名。三维用得上：门洞的两侧地面要连起来，遮罩才不会把一户切成孤岛。"""


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


OpeningKind = Literal["door", "window", "pass"]


class HeightRules(_Contract):
    """竖向那一维的全部数字，集中在一处，**没有一个散在代码里**。

    这些是常识档位不是这户人家的实测值（层高、门高、窗台高）。集中在包里的理由：
    换一户、换一个楼盘只换数据，代码不动；也让"这张图为什么是这个高度"答得出来。
    """

    ceiling_height_m: float = 2.80
    slab_thickness_m: float = 0.12
    door_height_m: float = 2.05
    pass_height_m: float = 2.20
    window_sill_height_m: float = 0.90
    window_head_height_m: float = 2.10
    outer_opening_kind: OpeningKind = "window"
    """外墙上的洞按什么算——产出侧不分门窗时的确定性推法（内墙洞按 `inner_opening_kind`）。"""

    inner_opening_kind: OpeningKind = "door"


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
    底渲按 `camera_id` 取其中一台，一次出四路图。
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
    triangle_count: int = 0


# ---------------------------------------------------------------------------
# 四、两个 activity 的出入参：都吃对象键（CLI 那条路吃同一份契约的本地 JSON）
# ---------------------------------------------------------------------------


class SceneCompileRequest(_Contract):
    """scene-compile 输入：**activity 吃对象键，不吃本地路径**。

    同解析那条 activity 的先例：包本体在私有桶里，activity 拿键去取。理由是编排里传
    不动一份包（几何 + 家具 + 材质动辄几百 KB），也不该让 workflow 的历史里躺着业务数据。
    CLI 那条路吃的是同一个 :class:`DesignPackage`，只是它直接读本地 JSON——**两条路
    同一份契约、同一段代码**，差别只在数据从哪儿来。

    `design_package_key` 的键模板今天**还没进 contracts 注册表**（那份表里只有报告册与
    用户上传件两行）。补登记的时点写死＝本仓落桶那一批，与 `scene_package_key`、底渲四路
    的键一起进表——在那之前 activity 是存根，键只在本模型里作为形态存在。
    """

    revision_id: str
    design_package_key: str


class BaseRenderRequest(_Contract):
    """base-render 输入：三维底渲（几何/深度/线稿/遮罩输出）。

    四路输出是**给下一步当条件图的**（realism-pass 在 imagegen 那仓），不是给人看的成品图，
    所以这一步既不需要 GPU 也不需要三维引擎：纯 numpy 软光栅，同一份场景包渲两次
    逐字节相同（同 render2d 母版那条口径）。
    """

    scene_package_key: str
    camera_id: str
    render_tier: RenderTier = "preview"
    width_px: int = 1024
    height_px: int = 768


class MaskEntry(_Contract):
    """遮罩索引表的一行：图上这个索引值是哪块网格、什么语义、哪间房。"""

    index: int
    mesh_id: str
    semantic: MeshSemantic
    room: str | None = None
    pixel_count: int = 0


class BaseRenderViews(BaseModel):
    """底渲一次的全部产物：四张图 + 遮罩索引表 + 自证数。

    不落桶、不签链接——落桶是 activity 那一层的事，本模型是纯库那条路的返回值
    （CLI 直接把它写到本地目录）。签名更是业务侧的事（"给谁看、看多久"）。
    """

    geometry_png: bytes
    depth_png: bytes
    line_png: bytes
    mask_png: bytes
    width_px: int
    height_px: int
    camera_id: str
    mask_index: list[MaskEntry] = Field(default_factory=list)
    covered_pixel_ratio: float = 0.0
    """画面里被几何盖住的比例——整张几乎全空说明相机摆错了，失败要响亮。"""

    near_m: float = 0.0
    far_m: float = 0.0
    """深度图的两端（米）。深度是 16 位归一化存的，没有这两个数就还原不回米。"""
