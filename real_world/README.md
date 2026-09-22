# wcx_gpt6_astra_agilex

本工作副本位于 `/home/agilex/lwy_astra_agx`，由 `/home/agilex/wcx_gpt6_astra_agilex` 完整复制而来。后续探索以本目录为准；接口、技能、现场配置与源项目对齐。本机 `agilex56` 的执行路径已指向本目录，历史证据中的绝对路径仍保留源项目记录。

4090 上的移植步骤、环境、权重和三条运行链路见 [移植说明](docs/porting.md)。加载后的顺序是：pi0.5 动作块、RSi 在线监测、校准尾部报警后 GPT 短恢复并交还。

共用一套 skill 和通用实现，通过 SSH、SocketCAN 和 ROS 接入多个 AgileX Piper 现场；每个服务器分别记录配置、标定和实测结果。

## 服务器入口 · 2026-09-15

先按 [服务器档案](config/servers.json) 选择目标，再读 [服务器选择与共用范围](skills/wcx-agilex-control/references/servers.md)。不默认选旧设备，不在连接失败后自动换机。

| 目标 | 接入/验证状态 | 现场文档 |
|---|---|---|
| `10.169.21.56` / `agilex56` | 17模块、8入口；双臂/夹持/RGBD、ACE书写和五列城墙轮廓有实测，动作录像支持阶段说明；毛巾整齐度未达标 | [agilex56](skills/wcx-agilex-control/references/site-agilex56.md) |
| `10.169.21.38` / `panfeng38` | Piper X；CAN/使能、双臂上抬约 20 cm、RGBD、双爪抓放、拉开抽屉及红绿刀头分拣实测，共用阶段框架和全部 8 个入口已部署 | [panfeng38](skills/wcx-agilex-control/references/site-panfeng38.md) |
| `10.169.21.1` / `houzhi1` | 全部 8 个入口已部署；双臂同步上抬约 20 cm、夹爪、状态/RGBD 有实测 | [houzhi1](skills/wcx-agilex-control/references/site-houzhi1.md) |

`config/site.json` 仍是 `.56` 的运行配置；其他现场不继承它的设备/标定。`.1` 当前使用独立 `config/sites/houzhi1/control-20260913.json`，全部入口已部署，初始 state/observe 配置保留。站点选择只读工具：在本项目根目录执行 `PYTHONPATH=src python3 -m agilex_control.sites show --site houzhi1`，也可传完整 IP。

后续新增服务器直接增加档案；新增型号在驱动层适配，保持现有 primitive 接口兼容，见 [扩展接口](skills/wcx-agilex-control/references/extending.md)。

不同实体的相机参数、几何/零点/限位、工具/负载、标定、场景和实时状态不可直接复用；新参数独立保存，不覆盖旧配置，见 [复用边界](skills/wcx-agilex-control/references/portability.md)。

## 通用调用层

2026-09-15 `.56` 用12块积木搭出五列 `[3,2,3,2,3]` 的城墙轮廓，最后松爪约94秒后仍保持，双臂退出且反馈无故障。任务包含空抓、误夹邻块和多次掉落；后期采用居中平面抓取、较轻夹持、实际支撑定位和高处转腕，最终逐层完成。仍有平面转动和悬边，不作精密对齐或重复性结论。[任务与最终图像](evidence/2026-09-14/host-10.169.21.56/castle-wall-evening/README.md)、[积木搭建流程](skills/wcx-agilex-control/references/block-stacking.md)和[带说明的完整回放](exports/2026-09-14/host-10.169.21.56/castle-wall-annotated-action-video/README.md)分别保留实测、可复用方法和录像验收。录像在每段顶部展示执行前的行动/依据；右腕中途增宽时按新内参观察，导出保持画面比例并补黑边。

2026-09-14 晚间 `.56` 完成左手压纸、右手写 ACE，双手让开后主相机确认字形与字序。前期夹具干涉、重新握笔及短线校正保留在[任务记录](evidence/2026-09-14/host-10.169.21.56/write-ace-evening/README.md)，方法沉淀为[书写流程](skills/wcx-agilex-control/references/pen-writing.md)。新增独立动作录像服务，原相机提供 RGB 预览，含动作的 phase 自动开始和收尾，调用之间的推理等待不录制；原视频直接导出三路单视频和上下拼接。[动作录像 skill](skills/wcx-agilex-action-video/SKILL.md) 保存启动、重试与时间含义。本次只部署 `.56` 的 cameras/phase/recording/report 四个文件，原本站配置不改，通过任务独立配置启用；其他站未随之部署。63 项本地相关测试运行及补充 report/client 8项通过，真实动作片段与10.4秒导出试验已完整解码，[完整任务回放](exports/2026-09-14/host-10.169.21.56/write-ace-action-video-h264/README.md)为37段、约12分钟，四个视频各10801帧完整解码通过。

后续已增加可选关节模式配置，默认保持原高随动；`.38` 另存普通位置速度候选，右臂空爪单姿态 J2/J3/J4 的0.1°响应与回程有实测，精细插孔仍未完成。各站源码已有后续独立补丁，当前指纹、默认配置和验证限制以[版本记录](docs/framework-versions.md)及[模式对比](evidence/2026-09-14/host-10.169.21.38/bit-insertion-002453/164-mode-profile/README.md)为准。

2026-09-14 已核对并对齐三站生产源码，`.56` 从旧9模块升级到与 `.1/.38` 相同的16模块。本站各93项本地/远端离线测试通过，新 client/phase 取图与 dry run 已验证；原配置、标定和相机服务保留。包元数据仍为 `0.1.0`，实际版本按 [源码指纹与部署记录](docs/framework-versions.md) 区分。

原8个 primitive 保持不变。`call` 支持 JSON 文件或 stdin 及实测摘要；`phase` 将有限原动作和动作后观察保存为同一个持久记录，重复调用不重放已开始动作，观察失败可单独重试。`client` 负责同站 SSH 调用和一次取回结果中的各路 RGB/metadata；`geometry` 只处理已保存 RGBD 与显式接触点，不默认指尖 TCP。命令、数据字段和失败边界见 [工具契约](docs/primitives.md)，分层见 [架构](docs/architecture.md)。

本地与 `.1` 远端最终全套各93项测试通过，16个共享源码文件已部署到 `.1`，本站配置/标定哈希保持不变，见 [部署核验](evidence/2026-09-13/host-10.169.21.1/framework-efficiency-1727/deployment.json)。新 client 已实际完成三视图与双臂反馈取回；新一轮叠杯也已完成，并在开爪撤离后75.08秒以新帧确认稳定。测试、部署与任务验收分别记录于 [验证记录](docs/validation.md)。其他站点的部署与实测范围仍以各自档案为准。

## panfeng38 · Piper X · 2026-09-13

用户确认 `.38` 为 Piper X。两路 CAN 以本站 1 Mbps 上线、六关节驱动使能并进入 CAN 模式，正常小幅恢复后由原 move_both 完成上抬：目标各 +200 mm，实测左 +197.186 mm、右 +196.403 mm，反馈无故障。抬升时夹爪保持原开度且驱动未使能，两夹爪工具当时仅 dry run。当前独立配置为 `config/sites/panfeng38/control-piperx-20260913.json`，旧配置保留。

普通 Piper FK 与本站相差约 66–68 mm，官方 Piper X 几何经两臂及动作后反馈核验一致。共享 Kinematics 只增加显式几何参数源；运动循环和相机实现未改，45 项离线测试通过。[本站实测](skills/wcx-agilex-control/references/site-panfeng38.md) 与 [共用 CAN/使能初始化流程](skills/wcx-agilex-control/references/initialization.md) 已写入 skill。

本次按共享标定流程完成两腕共 56 组采样及返回复测；用户确认本站“100 mm”标尺实际为 93 mm 后，已按统一 0.93 尺度离线重算。左右腕链留出位置 RMS 2.625 / 2.515 mm；横纵等比、平整度及残余深度/机器人链误差仍需核验，保存为本站独立候选，未启用为控制标定。采样临时高分辨率服务已停止，原相机服务恢复。见 [结果与未决项](evidence/2026-09-13/host-10.169.21.38/calibration-155901/README.md) 和 [候选参数](config/sites/panfeng38/calibration-20260913-93mm.json)。

随后用原观察、右臂移动和夹爪工具将中间抽屉拉开约 **7–8 cm**，松爪撤离后复查仍保持打开。右夹爪已实测开合与夹持，左夹爪在抽屉阶段尚未做动作验证；前两次夹空和第三次成功均保留证据。见 [抽屉任务结果](evidence/2026-09-13/host-10.169.21.38/drawer-pull-170908/README.md)。

19:01 完成红绿螺丝刀头分拣：右臂放红色、左臂放绿色，松爪撤离后分别复查留在同色格内。复用现有 client/phase/geometry 与原工具，双夹爪均新增实机抓放证据；未新建分拣控制器。相关 46 项测试通过，本站控制配置与相机服务保持原样，候选标定仍非已验证 TCP。见 [分拣实测、失败与框架复用](evidence/2026-09-13/host-10.169.21.38/screwdriver-sort-180648/README.md)。

## houzhi1 第三套平台 · 2026-09-13

全部8个共享入口已部署，状态/三路RGBD、使能、正常轻微越界恢复、双臂同步上抬约20cm和夹爪均有实测。左/右执行臂为can1/can0，运动与图像关联确认对应腕相机071316/070942；顶部D455为246322300564。当前控制配置独立保存，未修改固件、零点或关节限位。

三相机首轮外参已完成每腕16组训练、4组留出及返回复测。左右腕链留出平移RMS为4.392/3.524mm，跨臂关系约6.32mm、最大10.82mm。板尺寸仍按“100mm标尺约93mm”统一缩放，参数作为带误差说明的候选保存，未启用为正式控制标定；三月历史结果不采用。见 [本次结果](evidence/2026-09-13/host-10.169.21.1/three-camera-calibration-151012/README.md)、[候选参数](config/sites/houzhi1/calibration-20260913-1531.json) 与 [可复用标定流程](skills/wcx-agilex-control/references/calibration.md)。

最新一轮经通用阶段调用原观察、移动和夹爪工具，将三只倒扣纸杯套成一摞，右臂主动开爪撤离、左臂保持观察；75.08秒后新帧仍稳定。全程无掉杯或重抓，第三杯短试提经深度与补提复核；不能排除小幅就位滑动，两只搬运杯均主动释放后由重力落稳。首观察到撤离约21分51秒，到稳定复查约23分06秒；这不构成绝对定位、杯轴或 TCP 精度验证。见 [本轮实测](evidence/2026-09-13/host-10.169.21.1/cup-nesting-framework-1757/README.md)、[最终侧视图](evidence/2026-09-13/host-10.169.21.1/cup-nesting-framework-1757/35-stability-check/observation-000/d435i_071316/rgb.jpg) 与 [叠杯任务流程](skills/wcx-agilex-control/references/cup-nesting.md)。此前试提失败、滑落重抓和末杯自行滑入保留在 [首轮记录](evidence/2026-09-13/host-10.169.21.1/cup-stacking-154501/README.md)，不混作本轮过程。

相机服务常驻，日常看图直接调用原observe；检测和手眼求解为共享离线模块，复用已有OpenCV/RealSense环境。接入、使能恢复及历史证据统一在 [本站skill](skills/wcx-agilex-control/references/site-houzhi1.md)，原生项目文件未修改。

随后完成双臂拧盖：一臂固定瓶身，另一臂分段旋松、取下瓶盖并放到桌面，两爪撤离后约45秒仍稳定。复用现有工具；过程含空夹、IK失败和姿态误差，详见 [实测记录](evidence/2026-09-13/host-10.169.21.1/bottle-cap-195721/README.md)。[拧盖任务流程](skills/wcx-agilex-control/references/bottle-opening.md)区分可共用的接触/旋转方法与必须现场重测的参数；[三视角回放](exports/2026-09-13/host-10.169.21.1/bottle_cap_video/README.md)为66组快照，非连续录像。

## agilex56 毛巾翻折 · 2026-09-14

当前一条毛巾已按上下、再左右顺序执行翻折，**任务未完成**。角点修整和悬挂重新铺放未解决错层，后续两点抓取仍只提起部分层；066松爪退开后留在桌面左前方，未形成整齐矩形。不能把指定顺序的执行当作叠好，也不能用较好的中间图代替最终结果。定位误判、空抓、SSH原阶段续取及各次修整效果均保留在 [任务证据](evidence/2026-09-14/host-10.169.21.56/towel-quarter-fold-124251/README.md)，决策经验写入 [毛巾 skill](skills/wcx-agilex-control/references/towel-folding.md)。复用原工具，未改控制源码或本站标定。

## agilex56 历史实测 · 2026-09-12

下面为 `.56` 当日记录；后续双臂抬衣与叠衣服未完成的状态见 [本站资料](skills/wcx-agilex-control/references/site-agilex56.md)，不能视为任何服务器的实时状态。

**两枝桌上的花已插入白色花瓶，左右臂各完成一枝。21:16 最终检查确认双臂退出、两枝由花瓶支撑、夹爪为空。** 过程与失败原因见 [插花记录](docs/flower-arrangement-20260912.md)。此前四块积木入桶的独立成果见 [抓取放置记录](docs/pick-place-20260912.md)。

已建立独立 `observe`、`move_left/right`、`set_gripper_left/right`、`stop` 工具。左右入口固定设备，关节与夹爪共用有限轨迹执行器，模型负责解释图像、选择目标与判断结果，任务坐标不写入控制器。参见 [工具契约与部署](docs/primitives.md)。左右臂均已实测移动、夹持和花茎插入。

三路 RGBD 由用户服务 `wcx-agilex-rgbd.service` 持续采集：前视1280×720、双腕640×480，深度分别对齐各自RGB。共享 IK 已修复接近 pitch ±90° 时的 SDK Euler 截断问题；21项远端离线回归通过。skill和实现共用项目源码。早先约10cm抬升的97.663mm实测保留在 [历史成功重试](skills/wcx-agilex-control/references/right-lift-endpoint-20260912.md)。

- 实物共四臂：前两臂执行任务，后两臂用于示教/数采。用户在本次会话中已拔下后两臂。
- `can_left`、`can_right` 已启用，均为 1 Mbps；最终检查还发现 `can0` 已启用为 500 kbps。
- 早先发现已有 `roscore`、左右 Piper 驱动和三路 RealSense 节点，直接复用。17:06 前相机由 `evorl-ljy` 的 RobotIOServer 占用，17:08 查到其退出，ROS 图中已无相机 topic，残留参数不能当作运行证据。
- 早先八个前臂反馈 topic 均约 200 Hz；该次 ROS 快照两臂 `ctrl_mode=0`、`arm_status=0`、`err_code=0`，不代表当前操作者的实时状态。
- 旧5555曾返回 `left_wrist`、`right_wrist`、`right_front` 三路640×480 RGB；当前独立工具前视键为 `front`，另含对齐深度与逐帧内参。30 fps 是配置值，不以单帧声称实测帧率。
- 左、右前臂在前面的恢复/运动尝试中分别出现关节通信异常，错误字节曾为 `0x13` 和 `0x3F`。停止后的正常状态不代表故障已经解决。
- 当前实现集中在 `src/agilex_control/`，独立于旧服务和临时 Monitor；`config/tools.json` 定义工具接口，现场绑定在 `config/site.json`。历史实验仍作证据，不是执行入口。

## 文档入口

| 内容 | 入口 |
|---|---|
| 多台服务器的选择、独立配置与验证级别 | [服务器选择](skills/wcx-agilex-control/references/servers.md) |
| 原 primitive、阶段执行、同站取图、只读几何与摘要 | [调用契约](docs/primitives.md) |
| 多相机手眼、双臂基座关系、独立验证与场景变化 | [可复用标定流程](skills/wcx-agilex-control/references/calibration.md) |
| 双臂插花与细长物体操作 | [插花记录](docs/flower-arrangement-20260912.md) / [可复用流程](skills/wcx-agilex-control/references/flower-insertion.md) |
| 桌面积木抓取放置与几何纠偏 | [抓取放置记录](docs/pick-place-20260912.md) |
| 局部校正结果的复用条件、采样、拟合和独立验证 | [局部几何校正](skills/wcx-agilex-control/references/local-geometry.md) |
| 倒扣杯对中、试提、持稳、套入与失败恢复 | [叠杯任务流程](skills/wcx-agilex-control/references/cup-nesting.md) / [本轮实测](evidence/2026-09-13/host-10.169.21.1/cup-nesting-framework-1757/README.md) |
| 双臂固定瓶身、绕瓶轴旋盖、回腕重抓与分离验收 | [拧盖任务流程](skills/wcx-agilex-control/references/bottle-opening.md) / [三视角回放](exports/2026-09-13/host-10.169.21.1/bottle_cap_video/README.md) |
| 前视/左腕/右腕上下拼接回放与单路视频 | [积木视频](exports/2026-09-12/pick_place_video/README.md) / [插花视频](exports/2026-09-12/flower_arrangement_video/README.md) |
| 服务器、CAN、相机、Python 环境、单位 | [设备与环境](skills/wcx-agilex-control/references/environment.md) |
| CAN 上线、SDK 使能与正常恢复的共用步骤 | [初始化流程](skills/wcx-agilex-control/references/initialization.md) |
| agilex56 历史 ROS / 5555 排查与 topic | [历史运行手册](skills/wcx-agilex-control/references/runbook.md) |
| 5555 相机订阅、占用者、协议和快照 | [相机只读流程](skills/wcx-agilex-control/references/cameras.md) |
| 5556 接口、校准语义与实际能力边界 | [控制能力检查](skills/wcx-agilex-control/references/capability-check.md) |
| 初次 10 cm 试验、停止结果与原因分析 | [右臂试验分析](skills/wcx-agilex-control/references/right-lift-20260912.md) |
| 成功抬升重试及精度条件调整 | [抬升重试](skills/wcx-agilex-control/references/right-lift-endpoint-20260912.md) |
| 桶内观察、深度与标定探索 | [桶内观察](skills/wcx-agilex-control/references/bucket-inspection-20260912.md) |
| 问题、证据、处理结果及未决项 | [故障与解决记录](skills/wcx-agilex-control/references/issues.md) |
| 模型、规划器、驱动、硬件之间的职责 | [控制结构](docs/architecture.md) |
| 接手时的实际状态和下一步 | [交接记录](docs/handoff.md) |
| 脚本、技能格式及实机验证范围 | [验证记录](docs/validation.md) |
| 可发现的技能入口 | [wcx-agilex-control](skills/wcx-agilex-control/SKILL.md) |
| 原始状态、计划与历史尝试 | [证据说明](evidence/README.md) |

## 使用 skill

本项目保存技能源码，`~/.codex/skills/wcx-agilex-control` 指向 `/home/agilex/lwy_astra_agx/skills/wcx-agilex-control`，避免维护两个副本。

可在后续任务中使用 `$wcx-agilex-control`，例如：“观察 10.169.21.56 的机械臂状态”、“接入 10.169.21.38，先核对相机和机械臂角色”，或“在 10.169.21.1 把倒扣纸杯套成一摞”。叠杯任务由同一技能路由到任务专页，共用硬件与几何契约。同一任务可沿用明确目标；换服务器时重新读取本站配置和状态。

技能文件的格式验证不等于实机控制验证。只有关节跟踪、末端位移、保持状态和其他机械臂状态均通过实测，才能把对应动作记录为成功。

## 维护方式

每次调试更新问题的**当前状态**和证据链接；把未验证的解释保留为假设。环境与运行流程各保留一份权威文档。SSH 密码不落盘，桌面便笺中的上传、推理任务和大幅姿态不作为启动必需步骤执行。
