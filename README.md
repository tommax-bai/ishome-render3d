# ishome-render3d

《是我的家》三维管线服务（`render3d-svc`）：独立部署的 Temporal worker，承接三维场景编译与底渲
（几何/深度/线稿/遮罩输出）。

- **出处**：V1.4 裁决（2026-08-23，绘图能力物理拆分）——中控仓《架构对齐-设计Agent×技术架构.md》§三；
  绘图逻辑异质 → 独立仓库 + 独立服务，无 RPC、无 schema、无状态。
- **task queue**：`render3d-activities`（namespace `genpipe`；注册表：ishome-contracts `registries/task_queues.md`）。
- **本仓 activity**（注册名唯一真源：ishome-contracts `activities/registry.md`，只增不改）：

| 注册名 | 函数名 | 职责 |
|---|---|---|
| `scene-compile` | `compile_scene` | 输入包 → 三维场景包（米制网格 + 材质 + 机位） |
| `base-render` | `render_base` | 三维底渲（几何/深度/线稿/遮罩四路输出） |

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

底渲四路（几何/深度/线稿/遮罩）**不是给人看的写实图，是给下一步当条件图的**——写实化是
`realism-pass`，在 imagegen 那个仓走生成模型。四路都是几何缓冲，纯 numpy 软光栅就能出，
而且能做到**同一份场景包渲两次逐字节相同**（同 render2d 母版那条口径）。所以本仓不引
pyrender / OpenGL / blender，也没有系统级渲染栈依赖。

架构文档 §16 待定项⑥（渲染算力：外部 API vs 自建 GPU 池）**不卡这一步**：它卡的是写实化那一步。

## 两条路，同一份契约

- **纯库 + CLI（今天能跑）**：`render3d --design design-package.json -o out/`，不碰 Temporal、
  不碰对象存储。import-linter 锁死 `cli` 看不见 `activities`——从它能看见那一层起，
  "本地渲一张图不需要起编排"就只是一句承诺而不是结构。
- **activity（存根，时点写死）**：接通那一批要做三件——①加只依赖 oss2 的落桶模块（形态照抄
  reportrender 的 `book_store`：只写不签，签名属业务侧）；②`design_package_key` /
  `scene_package_key` / 底渲四路的键模板进 contracts `registries/object_keys.md`，产物类型进
  `registries/artifacts.md`（今天两张表里都没有三维的位置）；③把两个函数体换成
  "取键 → 调纯库 → 写桶 → 返回键与自证数"。

## 常用命令

```bash
uv sync                 # 安装依赖与 dev 工具
uv run render3d --design tests/fixtures/design-package-full.json -o out/   # 本地出场景包与四路图
uv run ruff check .     # lint
uv run lint-imports     # import 方向契约（worker|cli → activities → scene_compile|base_render → mesh|raster → models）
uv run mypy             # strict 类型检查
uv run pytest           # 测试
uv run render3d-worker  # 起 worker（TEMPORAL_ADDRESS，默认 localhost:7233）
```

新 clone 后执行一次：`git config core.hooksPath .githooks`（本地 pre-push 质量门）。
