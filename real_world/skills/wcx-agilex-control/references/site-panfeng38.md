# panfeng38 · Piper X · 10.169.21.38

用户已于 2026-09-13 明确确认本站机械臂为 **Piper X**，不是普通 Piper。账户 `SENSETIME\panfeng1`，主机 `cn0614009042l`；连接和当前配置见 [servers.json](../../../config/servers.json)。不同型号的几何、固件语义和实体参数分别核验。

## 当前实测 · 2026-09-13

用户复原双臂及 12 块方块后，右 J6 原故障已消失，三路 RGBD 正常；现场解除方法未核实。92 阶段双臂示教状态字节已由 1 变为 2，随后 93 阶段各发送一次原 SDK 正常 CAN 模式，两臂均回读 `ctrl_mode=1`、全部驱动 `0x40`、无故障。没有执行整臂复位或重新使能。此前 83/84 阶段模式与结束示教记录命令未生效，不能把早先的发送成功写成模式恢复。

112 阶段按新反馈，仅将左 J2 从 −0.755° 小幅移到范围内：一次原 SDK 正常 `JointCtrl`，目标 1°、回读 0.970°，其他关节保持本次反馈值，未修改限位。随后重新执行双臂抓放并逐次观察；字母 A 已完成：第二轮 12 块全部进入两条斜边与四块横杠，205 / 210 完成顶部和横杠调整，211 / 213 静止新图确认布局稳定。双臂空手退开，夹爪左 59.4 / 右 59.5 mm，十二驱动均 0x40、ctrl_mode=1、无故障。方块姿态不完全一致，不代表毫米级定位或 TCP 已标定。旧场景的六块摆放不计入第二轮；当前实物、误抓/空夹、落点与结果以 [A 任务进度和证据](../../../evidence/2026-09-13/host-10.169.21.38/letter-a-195611/README.md) 及该目录 `progress.json` 为准。

### 此前摆 A 的故障记录

20:45 起，第一轮摆字母 A 任务因右腕故障暂停，**当时尚未完成**。已移动并释放六块，另有原位黄色参考块；排列仍需补齐和纠偏。右关节 6 驱动字节为 `0x22`（SDK 定义：电机过温、驱动错误、未使能），其余驱动正常；整臂 `arm_status=0/error_code=0` 不能覆盖这个驱动故障。回读时右 J6 电机 44°C、驱动 49°C，不代表故障瞬间峰值或故障已解除。右腕相机 `261322070474` 同时失联并从 USB 枚举消失。

两爪均为空；新动作已暂停，未清故障、重使能或改保护，已请求现场检查右腕与线缆。64 阶段腕部旋转未跟随；66 阶段在发送前被驱动检查拒绝，`commands=[]`。不能把该现象写成已确认的机械限位。再次操作前取完整新状态，确认物理问题排除、驱动故障消失和相机安装有效，再按原初始化流程恢复。见 [A 任务进度、失败与故障原始证据](../../../evidence/2026-09-13/host-10.169.21.38/letter-a-195611/README.md)。

### 此前刀头分拣

19:01 完成红绿螺丝刀头分拣：右臂将散放红色刀头放入已有红色刀头的格子，左臂将绿色刀头放入已有绿色刀头的格子。两次成功试提开度分别为 9.2 / 10.1 mm；松爪撤离后分别用近景复查，最终全景确认两件均已离开原桌面位置并留在对应格内。两臂反馈完整无故障，最终夹爪左 23.3 mm、右 23.9 mm，仍使能。见 [分拣过程、失败与最终证据](../../../evidence/2026-09-13/host-10.169.21.38/screwdriver-sort-180648/README.md)。这是记录时状态，下次重新观察。

本次复用 `client/report/geometry/phase` 和原 primitive，相关 46 项离线测试通过；共享源码部署快照及旧版均在证据中。活动 Piper X 控制配置和相机服务未改，只新增本站候选标定的字段格式适配副本，原数值和 `accepted_for_control:false` 保留。双爪现均有实机开合、试提、搬运与释放验证；stop 仍仅空闲返回。小件抓取出现多次空夹，依靠本帧视觉和实际接触纠偏后完成，不代表绝对定位或 TCP 已标定；左右接触参考分别建立，任务数值仅存证据，不作默认执行参数。

### 此前拉开抽屉

17:37 原工具复查确认：右臂已将中间抽屉拉开约 **7–8 cm**，松爪撤离后仍保持打开。前两次夹空后未拉动，第三次以 0.3 N·m 夹住把手（实测 16.1 mm），分段拉开并释放。当时右夹爪为 39.4 mm、驱动使能；左臂未动，左夹爪仍为 0 mm、驱动未使能。两臂关节反馈正常。见 [本次过程与证据](../../../evidence/2026-09-13/host-10.169.21.38/drawer-pull-170908/README.md)。

抽屉阶段 8 个共享 JSON/CLI 入口全部部署，右夹爪完成实机开合、夹持和释放验证，左夹爪当时仅 dry run。手眼候选仅辅助粗定位，后续采用图像与反馈修正；此次成功不代表绝对精度或 TCP 已标定。

### 此前双臂抬升与接入

已通过原 `move_both` 完成双臂同步上抬。目标相对小幅恢复前各 +200 mm，后续 observe 回读左臂 **+197.186 mm**、右臂 **+196.403 mm**；目标残差分别 −2.814 / −3.597 mm。两臂六关节驱动已使能、ctrl_mode=1、故障码为 0、反馈完整新鲜。夹爪保持原开度 0 / 79.7 mm，**夹爪驱动仍未使能**，不得与关节使能混为一谈。

抬升阶段 get_state/observe、move_both 和 move_left/right 已实机调用，两夹爪当时仅 dry run。后续标定采样以单臂运动和图像变化确认了下方左右腕相机绑定。手眼已得到带留出评估的候选，已按用户确认的 93 mm 标尺修正缩印尺度，残余一致性待核验，TCP 未标定。

[抬升结果](../../../evidence/2026-09-13/host-10.169.21.38/integration_1450/lift-outcome.json) · [动作后 RGBD/反馈](../../../evidence/2026-09-13/host-10.169.21.38/integration_1450/after-lift/observation.json) · [原工具结果](../../../evidence/2026-09-13/host-10.169.21.38/integration_1450/lift.result.json)。这些是记录时的状态，后续调用重新读取。

### 本站标定候选

使用 [共享标定流程](calibration.md) 完成每腕 22 组训练、6 组留出及返回复测，保留 56 组原始样本；使用本站实际关节和 Piper X FK。用户确认本站“100 mm”标尺实际为 93 mm，已用原数据按统一 0.93 等比尺度离线重算。采用 27.9 mm 格长、20.46 mm marker 后，左右腕链留出位置 RMS **2.625 / 2.515 mm**、最大 **4.221 / 4.431 mm**，双基座关系的留出组合 RMS **2.906 mm**。这些是局部内部一致性，不代表外部绝对精度。

93 mm 来源于用户在本会话对本站纸张的确认，没有采用 `.1` 的尺寸。横纵等比和平整度仍有假设；修正后左/右/固定相机深度比例质检约 1.0008 / 1.0099 / 1.0327，机器人拟合仍提出不同的额外尺度，残差原因未解决。未合并各臂独立尺度或更改深度比例。候选 `accepted_for_control:false`，当前运动配置未引用；内部残差不能替代物理点/TCP 验证。此次复算没有追加实机动作。

[独立候选参数](../../../config/sites/panfeng38/calibration-20260913-93mm.json) · [采样、质量、尺度问题与复算入口](../../../evidence/2026-09-13/host-10.169.21.38/calibration-155901/README.md)。完整数据本地/服务器各一份。检测用服务器现有 OpenCV 5.0.0；该构建缺少 calibrateHandEye，求解在本机现有 OpenCV 4.11 / SciPy 环境执行同一源码。

收尾时双臂返回看板区域（实际 Z 约 329.9 / 331.2 mm），反馈无故障，六关节仍使能，夹爪仍为 0 / 79.7 mm 且驱动未使能。采样用的临时固定相机 1280×720 服务已停止，原三路 640×480 服务恢复；[最终观察](../../../evidence/2026-09-13/host-10.169.21.38/calibration-155901/final-restored/observation.json)。这是当时状态，后续重新观察。

## 当前部署与调用

| 项目 | 本站值 |
|---|---|
| 共享项目 | `/home/SENSETIME/panfeng1/wcx_gpt6_astra_agilex` |
| CLI / 相机 Python | 上述项目 `.venv-observe/bin/python`，3.10.20；NumPy 2.2.6、SciPy 1.15.3、OpenCV 5.0.0、pyrealsense2 2.58.1.10581 |
| 当前运动配置 | [control-piperx-20260913.json](../../../config/sites/panfeng38/control-piperx-20260913.json) |
| SDK Python | `/home/SENSETIME/panfeng1/miniconda3/envs/piperaio/bin/python`，3.10.8，piper_sdk 0.6.2 |
| 运行目录 | `/tmp/wcx_agilex_panfeng38` |
| 相机服务 | 用户 systemd `wcx-agilex-panfeng38-rgbd.service`；已启动，未设开机自启 |
| 原生项目 | `/home/SENSETIME/panfeng1/ws/data/yuxiang/piper-aio-private`，未修改 |

```bash
PYTHONPATH=/home/SENSETIME/panfeng1/wcx_gpt6_astra_agilex/src \
/home/SENSETIME/panfeng1/wcx_gpt6_astra_agilex/.venv-observe/bin/python -m agilex_control \
  --config /home/SENSETIME/panfeng1/wcx_gpt6_astra_agilex/config/sites/panfeng38/control-piperx-20260913.json \
  call --request /本次新目录/request.json --output /本次新目录/result.json
```

例如 `{"primitive":"observe","arguments":{"output":"/本次新的绝对目录"}}`。后续直接复用相机服务；失败时检查 `cameras status` 和 `systemctl --user status wcx-agilex-panfeng38-rgbd.service`，确认占用后 `systemctl --user start ...`。管理入口仍是同一 CLI 的 `cameras serve/status/stop`，没有新增 ROS 观察后端。服务文件保存于本站 config 目录。

旧 [state 配置](../../../config/sites/panfeng38/state-20260913.json) 保留，按 can0/can1 被动读状态。旧 [observe 配置](../../../config/sites/panfeng38/observe-20260913.json) 是当时 CAN 未接入的相机专用记录；旧 `control-20260913.json` 使用不适合本站的普通 Piper FK，仅保留排查历史，不能用作当前运动配置。新配置恢复 observe 的完整臂状态，不覆盖这些历史文件。

## CAN、型号和独立参数

用户确认每根 USB-CAN 只接一只前方执行臂；不操作后部示教配置。原生 `piper_ws/scripts/can_config.sh` 的 USB 规则对应如下，枚举变化后按物理串号重新匹配：

| 逻辑标签 | 当前接口 | USB 路径 | 适配器序列号 | 固件 |
|---|---|---|---|---|
| right | can0 | 1-1:1.0 | 003900314148571320343133 | S-V1.8-9 |
| left | can1 | 1-2:1.0 | 003200274148571320343133 | S-V1.8-9 |

本次两路直接以本站原生记录的 **1 Mbps** 上线，未改接口名或持久网络规则。完整读反馈且无 CAN 错误，ERROR-ACTIVE 是正常 CAN 状态。本站单臂运动与对应腕图变化已核验；不能仅凭这张表推断其他平台角色。

逐电机只读回包验证两臂限位相同：J1 ±150°、J2 0～180°、J3 −170～0°、J4/J5 各 ±89°、J6 ±180°；查询的最大关节速度均为 0.3 rad/s。两夹爪的固件配置范围均为 0～100 mm；本次未验证全行程。没有复制 `.1` 的 J4/J5 或 70 mm 夹爪范围。

几何采用官方 pyAgxArm 提交 `e7aef17d54cac80cbaeb1b4110ab3d8f1337a95b` 的 Piper X modified-DH，出处及单位保存在当前配置中。普通 Piper SDK FK 在初始两臂上相差 65.811 / 67.656 mm；Piper X 模型位置差 0.00291 / 0.00506 mm、旋转差均小于 0.002°，抬升后位置差仍约 0.002–0.003 mm。这是内部模型一致性，非外部定位精度或标定。共享 Kinematics 仅增加显式 `mdh_mm_rad` 参数源，保留旧 `fk_file` 接口；通用运动循环、CAN 位置帧和相机代码未改。

## 初始化与本次恢复经验

操作顺序统一见 [CAN 初始化、使能与恢复](initialization.md)。本站差异与本次结果：

- 每臂各一次 `EnableArm(7, 2)` 后六关节均使能，夹爪仍未使能。首次检查同时要求夹爪使能而误报失败；完整回读后仅处理另一条尚未使能的臂，失败证据保留。
- SDK 0.6.2 的批量限位对象重复最后一个电机；本站参数来自逐关节匹配的新回包及保存的 0x473 原始字节，未修改 SDK。
- 正常小幅恢复使 J2/J3 回到本站范围，末端 Z 单调上升约 14 mm；之后原 move_both 执行 15 秒并驻留 3 秒，总目标按恢复前位姿计算。本次模式切换未改变夹爪开度。恢复、规划与实测分别保存于下方完整证据目录。

## 相机与深度

| 标签 | 型号 / SDK 序列号 | RGB / 对齐深度 |
|---|---|---|
| camera_l | D435 / 261622075832 | 640×480 @ 30 |
| camera_r | D435 / 261322070474 | 640×480 @ 30 |
| camera_f | D455 / 261822300695 | 640×480 @ 30 |

三站复用同一 cameras.py。逐帧保存 rgb.jpg、uint16 depth.npy、metadata.json（内参、深度比例、畸变、帧号与时间戳）；相机内对齐，相机之间无硬同步。深度比例来自本机 SDK，约 0.001 m/单位，0 为无效深度；非零比例不能替代测距准确度。USB 描述符序列号与 SDK 身份不同，配置使用已核验的 SDK 序列号。虚拟环境复用 lerobot 基础依赖，后者升级后需复核。

## 验证与证据

[本次完整目录](../../../evidence/2026-09-13/host-10.169.21.38/integration_1450/) 包含初始状态、逐电机原始参数、模型比较、使能、恢复、规划、实际轨迹和前后 RGBD。45 项离线测试通过，含原 42 项及 Piper X 实测反馈匹配、参数歧义/格式拒绝、原起点限位检查；[测试结果](../../../evidence/2026-09-13/host-10.169.21.38/integration_1450/local-tests.txt)。

[首次检查](../../../evidence/2026-09-13/host-10.169.21.38/initial-inspection.md) · [早期共享 RGBD](../../../evidence/2026-09-13/host-10.169.21.38/observe_shared_rgbd_004/README.md)。早期 ROS 截图与失败配对仅作证据，不是默认入口。
