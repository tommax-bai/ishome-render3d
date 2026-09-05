# ishome-render3d

《是我的家》三维管线服务（`render3d-svc`）：独立部署的 Temporal worker，承接三维场景编译与底渲
（几何/深度/线稿/遮罩/控制稿五路输出）。

- **出处**：V1.4 裁决（2026-08-23，绘图能力物理拆分）——中控仓《架构对齐-设计Agent×技术架构.md》§三；
  绘图逻辑异质 → 独立仓库 + 独立服务，无 RPC、无 schema、无状态。
- **task queue**：`render3d-activities`（namespace `genpipe`；注册表：ishome-contracts `registries/task_queues.md`）。
- **本仓 activity**（注册名唯一真源：ishome-contracts `activities/registry.md`，只增不改）：

| 注册名 | 函数名 | 职责 |
|---|---|---|
| `scene-compile` | `compile_scene` | 输入包 → 三维场景包（米制网格 + 材质 + 机位） |
| `base-render` | `render_base` | 三维底渲（几何/深度/线稿/遮罩/控制稿五路输出） |

## 这条服务要什么数据（服务之间只有数据通信）

用户裁决 2026-08-31：**上下游是通信关系，不是调用关系**。三维的上游（定稿平面、三维资产库）
今天都不存在，本仓**不等它们**——把"三维需要什么"写成一份输入包契约（`models.DesignPackage`），
开发期喂 `tests/fixtures/` 里的拟真包，真派发时同一份包由上游填，本仓一行代码不改。

`DesignPackage` 的两半：

| 半 | 内容 | 今天从哪儿来 |
|---|---|---|
| 户型几何（归一化，无绝对尺寸） | `plan`：外轮廓、墙、洞、房间遮罩、参照系 `frame_*_px` | **已有真上游**：aipipe 的 `floorplan-geometry`（零模型调用）。本仓这一族模型是它的**逐字对面**——两个仓两种语言谁也不能 import 谁 |
| 三维要而二维没有的 | `scale`（建筑面积 + 得房率 → 尺子）、`heights`（层高/门高/窗台高）、`furnishings`（家具体块）、`materials`、`cameras` | 拟真包。上游是 DeepDesign 与三维资产库，**都还不存在** |

两条口径写死在契约里：**比例尺不许由模型给**（由面积反推，面积是上游真有的数）；
**缺哪一块就是上游哪一块没做出来，不许在本仓编一个值补上**（缺了就响亮失败）。

## 为什么这一步不需要 GPU 和三维引擎

底渲五路（几何/深度/线稿/遮罩/控制稿）**不是给人看的写实图，是给下一步当条件图的**——写实化是
`realism-pass`，在 imagegen 那个仓走生成模型。五路都是几何缓冲，纯 numpy 软光栅就能出，
而且能做到**同一份场景包渲两次逐字节相同**（同 render2d 母版那条口径）。所以本仓不引
pyrender / OpenGL / blender，也没有系统级渲染栈依赖。

架构文档 §16 待定项⑥（渲染算力：外部 API vs 自建 GPU 池）**不卡这一步**：它卡的是写实化那一步。

## 线稿与控制稿的分工（2026-09-05 起五路）

| 路 | 是什么 | 给谁用 | 画法 |
|---|---|---|---|
| `line.png` 线稿 | **几何事实边**：每一条网格边界、每一处深度断开 | 保真度尺子的输入 | 全画，**一个字节都不动** |
| `sketch.png` 控制稿 | **给"线稿生图"控制通道画的**：通道锁什么就画什么 | 写实化的控制条件 | 画地脚线、墙角、洞口轮廓、遮挡轮廓、揭顶下的墙顶墙厚、家具体块的棱；**不画**地面上的房间分界线、天花与墙的交线、天花分区线、共面接缝；**门一条斜线、窗一个十字** |

来路＝中控仓《评审/失效清单-控制图通路-2026-09-04》：构件级四条失效（地面分界线画成台阶、
天花交线画成灯槽、透视天花线读成斜顶、门窗不分）来源全是线稿画了不该画的边、没画该画的区别。
两路同尺寸同编码（单通道、黑底白线），细节与取舍理由在 `base_render.py` 模块 docstring。

室内机位同日起**不许对墙**：没给 yaw 的 `room` 机位按候选评估取景（退让方向 × 朝向，用本仓自己的
低分辨率深度/遮罩量最近深度、目标房间地板占比、主体占比），没有一个达标就响亮失败；上游显式给
yaw 的照旧听上游的。判据的数与取值理由在 `base_render.py` 的 `ROOM_VIEW_*` 常量上。

## 两条路，同一份契约

- **纯库 + CLI（今天能跑）**：`render3d --design design-package.json -o out/`，不碰 Temporal、
  不碰对象存储。import-linter 锁死 `cli` 看不见 `activities`——从它能看见那一层起，
  "本地渲一张图不需要起编排"就只是一句承诺而不是结构。
- **activity（已实装，2026-09-05）**：取键 → 调纯库 → 写桶 → 返回键与自证数，与 CLI 共用同一份
  纯库代码；怎么接、还欠什么见下节《接进编排》。

## 接进编排

两个 activity（`scene-compile` / `base-render`）已实装：取键 → 调纯库 → 写桶 → 返回键与自证数，请求收
camelCase 不透明字典、回执是 snake_case 字典（字段见 `activity_models.py` 模块 docstring；`base-render`
逐机位渲，一台失败整份 `failed`、已写进桶的照样列在 `renders` 里）。键形态
（`{prefix}/render3d/{revision_id}/…`，见 `object_store.py` 模块 docstring）是**草案，待 contracts
`registries/object_keys.md` 登记**，登记时若改形态只改那个模块的模板与 `tests/test_object_store.py`。
worker 起进程要 `ISHOME_OSS_ENDPOINT` / `ISHOME_OSS_BUCKET_PRIVATE` / `ISHOME_OSS_ACCESS_KEY_ID` /
`ISHOME_OSS_ACCESS_KEY_SECRET` 四个变量（凭证不入库：本机 `~/.ishome/oss-local.env`，服务器
`/opt/ishome/env/oss.env`），缺一即起不来。

## 常用命令

```bash
uv sync                 # 安装依赖与 dev 工具
uv run render3d --design tests/fixtures/design-package-full.json -o out/   # 本地出场景包与五路图
uv run ruff check .     # lint
uv run lint-imports     # import 方向契约（worker|cli → activities → scene_compile|base_render → mesh|raster → models）
uv run mypy             # strict 类型检查
uv run pytest           # 测试
uv run render3d-worker  # 起 worker（TEMPORAL_ADDRESS 默认 localhost:7233；私有桶四个 ISHOME_OSS_* 变量见《接进编排》）
```

新 clone 后执行一次：`git config core.hooksPath .githooks`（本地 pre-push 质量门）。
