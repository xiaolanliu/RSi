# 观察与移动 primitive

公开入口为 `get_state`、`observe`、`move_left`、`move_right`、`move_both`、`set_gripper_left`、`set_gripper_right`、`stop`。工具定义在 [tools.json](../config/tools.json)，Python 调用入口在 [primitives.py](../src/agilex_control/primitives.py)。目前是可通过 SSH 调用的 Python/JSON CLI 接口，未注册为 Codex 原生 MCP 工具。各站可用子集以 [servers.json](../config/servers.json) 与本站验证记录为准。

多服务器共用此契约和实现；先按 [servers.json](../config/servers.json) / [服务器选择](../skills/wcx-agilex-control/references/servers.md) 选择现场，再使用该现场的 Python、项目和 `--config`。本文后面的完整部署命令、运行目录与历史实测数值属于 `agilex56`；`config/site.json` 也仅属于该现场。`panfeng38` 为用户确认的 Piper X，已通过原 move_both 实测双臂上抬并复用原 observe 获取完整 RGBD/反馈，见 [本站调用方式](../skills/wcx-agilex-control/references/site-panfeng38.md)。`move_both` 和 `stop` 的作用域都是所选服务器。

左右臂的设备绑定不同，执行数学与协议相同。因此对 agent 暴露两个固定臂工具，内部共享一个执行器。物体识别、桶口选择、任务顺序、碰撞判断和完成判断由上层完成；控制器不包含桶的位置或任务流程。

| 工具 | 输入 | 返回 |
|---|---|---|
| `get_state` | `seconds`，默认 1 秒，范围 0.5–30 秒 | 被动采集本站 CAN 接口，返回时间戳、反馈和观察到的控制帧；不依赖相机或运动学，不发送指令 |
| `observe` | 服务器上新的输出目录 | 通过原有相机服务返回 RGB、对齐深度及逐相机参数；默认读取臂反馈，本站明确关闭该项时返回 states:null 和 missing_modalities:[arm_state] |
| `move_left` | 左基座中的 link6 XYZ/RPY，或 6 个关节角；时长、驻留时间、日志路径 | 实际执行状态、起终点、关节误差、XYZ 位移、完整遥测路径 |
| `move_right` | 同上，固定右臂 | 同上，固定右臂 |
| `move_both` | `targets: {left: 目标, right: 目标}`；每个目标格式与单臂相同，共用时长 | 每臂离线计划、实际反馈和共享时钟遥测；两臂全部规划成功后才打开动作通道 |
| `set_gripper_left` / `set_gripper_right` | `width_mm`、`effort_nm`（默认 1.0）、时长、驻留、日志路径 | 实际开度、驱动力矩反馈、状态位及遥测；支持 dry run |
| `stop` | 无参数 | 是否向当前运动调用发出停止信号；停止产生后续目标，保持原有使能，不自动撤回或失能 |

`dry_run:true` 只规划/校验，不执行。XYZ 单位 mm，RPY/关节角单位 deg；XYZ 对应所选臂的 base→link6，不是已标定的指尖。省略 RPY 时保持读取到的当前姿态。`duration_s` 是轨迹计划时长；调度延迟会延长实际时间，不追赶跳步。高随动模式的速度比例不能作为真实限速。

返回统一封装：`{primitive, ok, started_unix, finished_unix, data, error}`。`ok:true` 表示调用完成；运动的 `command_stream_completed` 表示完整目标流已发送并回读反馈，任务是否完成仍须结合实测误差与图像判断。原始遥测保存到独立文件，避免把几百条采样全部塞入 tool result。

夹爪测量可以轻微超出合法命令范围，例如闭爪反馈−0.07 mm。命令轨迹起点按该臂配置范围投影，原测量仍保留在 baseline 与 width_start_mm，width_command_start_mm 记录发送用起点；目标开度范围、驱动故障和有限值校验不变。这不是修改设备零点或扩大限位。

上述起点修复对应 b0a7ea9a…，目前仅部署 agilex56；其他站点应按 source_revision 核对后再依赖这项行为及新增字段。

## JSON CLI 与精简摘要

在所选服务器运行以下入口；`--request FILE` 与 `--request -` 分别读取文件和 stdin JSON，原文件调用保持兼容：

```bash
PYTHONPATH=src python -m agilex_control --config /absolute/site-config.json \
  call --request /absolute/request.json --output /absolute/new-result.json --summary
```

`--summary` 只精简 stdout，`call --output` 仍保存完整 primitive 结果；不提供 `--summary` 时 stdout 保持完整封装。摘要由 [report.py](../src/agilex_control/report.py) 生成，保留实测位姿/夹爪、可计算的目标误差、故障、观测时间、各相机图像路径与原始证据。`observation_error` 保留观察失败原因，`states_source` 区分动作后的观测和 primitive 反馈回退；缺失量保持空或标明原因。`task_success:null` 始终留给上层据图判断，不能把摘要 `ok` 当成物体操作成功。

## 有限阶段 phase

[phase.py](../src/agilex_control/phase.py) 的 `run_phase(config, request, output_dir, retry_observation=False)` 顺序调用已有 primitive，结束后按请求观察；它不增加运动循环、任务分支或后台调度。命令为：

```bash
PYTHONPATH=src python -m agilex_control --config /absolute/site-config.json \
  phase --request /absolute/phase-request.json --output /absolute/new-phase-dir --summary
```

请求为 `{"steps": [...], "observe_after": true, "context": {}}`：每个 step 必须包含 `primitive` 和 `arguments`，仅允许两个 move、move_both 和两个 gripper。arguments 沿用原工具契约，但不能传 `output`，每步的请求、结果和遥测路径由阶段派生；phase 拒绝 `dry_run:true`，规划继续用原单步 call。全部步骤先校验纯参数，实时状态与 IK 在各步执行前验证。`context` 仅作证据，不解释为动作。`observe_after` 默认 true；`steps: []` 可只获取一次观测，例如：

```json
{"steps": [], "observe_after": true, "context": {"purpose": "inspect current scene"}}
```

模型根据阶段验收选择下一阶段；不要把依赖尚未确认的持物或接触状态的动作一次排入同一列表。一个阶段与原单次动作共用现场动作锁，`stop` 覆盖当前步骤及后续步骤，不自动撤回、失能或复位。

输出目录是持久调用身份，`phase.json` 保存完整阶段记录，另存 `request.json`、每步的 request/result/telemetry 及每次 observation 的目录和结果。`action_status` 与 `observation.status` 分开：动作完成而观察失败，不等于动作没执行。总 `ok` 需要请求的动作与观察均完成，仍不代表任务验收。

同目录、同请求与同配置的再次调用只返回缓存、进行中状态或不确定记录，不续跑已开始的动作；请求或配置改变会拒绝复用该目录。目录已有其他证据而没有 phase 记录时也拒绝占用。动作前先记录 started；进程意外结束而没有终态时标为 `uncertain`，这不是硬件 exactly-once 保证。

若仅观察失败，在原请求和原目录上加 `--retry-observation`；它只补取观察，不重放动作，也不继续未执行步骤。已完成的观察会返回缓存。网络或进程状态不确定时先检查原身份的记录和现场，不能通过换一个输出目录自动重做动作。

## 本地同站调用与图像取回

[client.py](../src/agilex_control/client.py) 按现有服务器档案选择 SSH、远端 Python/项目/配置，不自行发现或切换目标；需要已可用的非交互 SSH 认证，可用 `--control-path` 复用已有连接。示例中的 request 是本地文件：

```bash
PYTHONPATH=src python -m agilex_control.client --site houzhi1 call \
  --request /absolute/observe-request.json \
  --remote-output /absolute/new-remote-phase-dir --output /absolute/new-local-dir
```

`call` 的实际移动/夹爪请求转换为单步骤 phase 并附带观察，`observe` 转为零步骤 phase；因此这些调用必须给出稳定的 `--remote-output`，动作参数中的输出路径由阶段管理。get_state、stop 和 dry run 沿用原 call；也可将 operation 设为 `phase` 直接发送阶段请求。相同本地目录不能换请求或站点。

结果中的各路 RGB 和 metadata 用一次归档传输取回；`--include-depth` 额外取回对齐深度，不另启相机或补采图像。`client-result.json` 的 `images[].local_rgb/local_metadata` 指向本地文件，`client_timing_s` 区分远端调用与文件传输耗时，并保留远端退出码；远端响应原文逐次保存。动作/观察与文件传输分别报告；取图失败或响应不确定时沿用原请求及远端目录重取，不能换身份重复动作。`--retry-observation` 与 phase 语义一致。

`transport_status/transport_error` 独立记录 SSH 调用异常，`delivery_status/delivery_error` 记录图像文件取回；两者不覆盖阶段的动作和观察结果。即使已经收到部分结果，SSH 异常仍使客户端 CLI 非零退出，不会被误记成完整成功。

## 只读几何入口

[geometry.py](../src/agilex_control/geometry.py) 只处理已保存 RGBD 和显式刚体参数，不连接硬件、不执行 move，也不设置默认 TCP：

```bash
PYTHONPATH=src python -m agilex_control.geometry \
  --request /absolute/geometry-request.json --output /absolute/new-geometry.json
```

`--request -` 可从 stdin 读取，`--output` 可省略，提供时不覆盖已有文件。三个 operation 为：

| operation | 请求字段 | 结果 |
|---|---|---|
| `observation_points` | `observation_dir`、`camera`、`target_arm`、`calibration`、`config`、`points: [{id, pixel}]`；可选 `radius`、`min_valid` | 各点相机/基座毫米坐标、深度统计、变换方向、逐帧参数和来源；单点失败独立返回，部分失败使总体 ok 为 false |
| `point_in_link6` | `point_base_mm`，以及 `observation_dir`、`config`、`arm`；或使用显式 `base_point_mm` 与 `actual_T_base_link6` | 从实测末端变换反算 `point_link6_mm`，返回变换和来源；不将这次接触关系认证为 TCP |
| `link6_target` | `task_point_base_mm`、`point_link6_mm`，以及 `rpy_deg` 或 `rotation_matrix` 二选一 | 由 `t = p - R @ p_local` 求 link6 XYZ；RPY 输入返回可交给原 move 的 `target`，仍需 dry run 与场景验证 |

固定相机使用该实体的外参；腕相机使用观测内实际关节的 FK 和自身手眼，跨臂时再组合基座变换。缺失身份、模式不符、无有效对齐深度或过期的腕部反馈会报错；历史数据可用于离线分析，但不会被标为实时。候选标定保留 warning，结果不会自动启用为控制参数。几何输出仍需交给原 move dry run，再结合当前场景判断。

`calibration` 和 `config` 接受文件路径或内嵌 JSON。`observation_points` 需要完整 `observation.json`、所选相机 metadata 和对齐 depth；client 默认取回的图像目录不是完整观测副本。

`link6_target` 的矩阵输入决定计算用朝向，但返回的 `target` 不含 RPY；执行时必须明确与该矩阵一致的朝向，不能省略姿态而让 move 保持一个不同的当前朝向。抓取中发生滑移时，反算的物体局部点同样失效。

基础几何需 NumPy；非零普通 Brown 畸变按需加载 OpenCV，非零 inverse Brown 按需加载 RealSense，腕部 FK 还需本站运动学环境。可通过本地 client 的 `geometry` operation 在已具备依赖的远端环境处理原始观测，JSON 内的文件路径指向实际执行端；它仍只读文件。

## 被动状态读取兼容入口

首选上述 `get_state`。原 [read_state.py](../skills/wcx-agilex-control/scripts/read_state.py) 仅保留命令行适配，直接调用共享 `primitives.state → CanBus → decode_state`，不再维护独立接收循环；读取使用内核接收时间戳，无 CAN 发送。部署时须保留项目目录布局，或通过 PYTHONPATH 提供同一包。

在所选服务器的项目目录内，以该站 Python 运行：

```bash
python skills/wcx-agilex-control/scripts/read_state.py --interfaces 已核验接口1 已核验接口2 --seconds 3 --output /absolute/new-state.json
```

`--interfaces` 必填，不再默认选择 `.56` 的接口名。输出直接是 get_state 的 `data`：`{captured_unix, states: {接口名: 反馈}, control_frames}`，不带 primitive 外层封装。旧 JSON 的 `interfaces`、采样统计和状态变化列表不再生成，已有历史文件不改写；原脚本和格式见 [归档说明](../evidence/2026-09-13/skill-consolidation/README.md)。

早期抬升规划脚本也已归档。所有当前规划统一使用原 move 工具的 `dry_run:true`，不维护单独的旧型号规划器。

## 一次部署，多次调用

叠衣服时，两爪抓住同一件布料，需要共享运动时钟。`move_both` 复用单臂执行器、全局动作锁和固定 CAN 路由；两臂在同一个 30 Hz tick 中依次写入，并非硬件原子同步。任一臂反馈异常或发送失败都停止后续双臂目标；第二臂发送失败不能撤销第一臂已经收到的当前 tick。遥测记录每臂发送时间，不能用两个并行单臂进程替代该工具。当前每次 move 仍为起终点平滑关节插值，分多次调用会在各端点停住，尚不支持穿过中间点的连续摆动轨迹。

2026-09-12 已实测双臂同时夹持并提起衣服。新增6项回归覆盖双臂共享时钟、主机阻塞、发送失败、故障前置、双臂停止和 CAN 路由；远端27项全部通过。本地21项通过、6项因缺少现场纯 FK 文件跳过。叠衣服任务尚未完成，这些结果仅验证工具行为。

服务器源码目录：`/home/agilex/lwy_astra_agx`。配置 [site.json](../config/site.json) 保存左右 CAN 映射、相机序列号、现有 SDK 关节边界、纯 FK 文件路径和运行目录。源码/配置不含 SSH 密码。

已有 `evorl-ljy` Python 提供 NumPy、SciPy、OpenCV 和 RealSense；无需为工具初始化机器人 SDK。更新时同步共享源码与必要的包元数据，现场配置、标定和运行目录保留；首次接入才按实际实体另建配置，不能批量覆盖 `config/`。使用以下入口：

```bash
PYTHONPATH=/home/agilex/lwy_astra_agx/src \
/home/agilex/miniconda3/envs/evorl-ljy/bin/python -m agilex_control \
  --config /home/agilex/lwy_astra_agx/config/site.json \
  call --request /absolute/request.json --output /absolute/result.json
```

观察请求示例：

```json
{"primitive":"observe","arguments":{"output":"/tmp/wcx_observation_001"}}
```

先进行当前位姿附近的离线规划示例；实际 XYZ 必须来自本次观测：

```json
{"primitive":"move_right","arguments":{"target":{"xyz_mm":[390.66,279.92,331.251]},"duration_s":5,"settle_s":2,"output":"/tmp/wcx_move_001.json","dry_run":true}}
```

执行同一计划时明确改为 `dry_run:false`，或传入规划返回的 `joint_deg`。上述坐标仅是本次历史示例，不能在新场景直接重放。每次调用使用新的观测/日志路径。

相机生命周期使用同一 CLI 的 `cameras serve/status/stop`。`serve` 是持续采集进程，三个相机分别由一个线程拥有；snapshot 通过仅本机 Unix socket `/tmp/wcx_agilex_control/cameras.sock` 读取。相机之间没有硬同步，结果分别保存时间戳。健康状态在 `camera_health.json`，运动默认要求其有效。启动前检查占用者；不要与其他相机服务争抢设备。

## 初始化与型号几何

CAN 接口上线、SDK 关节使能、CAN 模式和轻微越界恢复见 [共享初始化流程](../skills/wcx-agilex-control/references/initialization.md)。它们复用系统/SDK 入口，现有 8 个 JSON primitive 名称保持不变。使能与关节目标不是同一操作，夹爪状态独立验证。

运动学配置二选一：原有 `fk_file` 使用该 SDK 的普通 Piper offset=1 模型；`mdh_mm_rad` 为六行 `[d_mm, a_mm, alpha_rad, theta_offset_rad]`，直接复用同一个 FK/IK。不得同时配置或自动回退。`.38` 的 Piper X 几何另存带来源的新配置，对照双臂原始反馈与实测动作验证；缺省模型不适用于所有 Piper 型号。同一配置当前仍要求两臂几何/关节限位一致；混合型号需按臂扩展配置读取。

## 底层职责

`gripper_limits_mm` 可使用两臂共用的 `[min, max]`，也可使用 `{"left": [min, max], "right": [min, max]}`。按臂配置时必须提供对应臂范围，不回退到另一臂；dry run 和执行共用 `GripperTrajectory` 的校验与编码。范围来自本站固件查询，不能从旧现场复制。`.1` 实测左右夹爪分别为 70 / 100 mm；原 `.56` 的共用 70 mm 配置保持兼容。

三站共用原有 `cameras.py`，没有新增相机后端框架。`observe_arm_state` 默认 true；尚未接入臂反馈的站点明确设 false 并填写 `observe_arm_state_unavailable_reason`。预期应读取而实际读取失败时仍报错。

- [protocol.py](../src/agilex_control/protocol.py)：唯一的状态解码与关节帧编码；无硬件副作用。
- [can_bus.py](../src/agilex_control/can_bus.py)：读取左右 CAN，使用 Linux 内核接收时间戳判断新鲜度，记录接收队列丢帧；只向选定 CAN 写入白名单模式/关节帧。
- [motion.py](../src/agilex_control/motion.py)：有限轨迹、独占调用、按已发送 tick 推进、最终目标必发、设备故障与控制权监测、停止和遥测。
- [trajectory.py](../src/agilex_control/trajectory.py)：关节与夹爪的目标编码、起终点和反馈要求；由同一个 motion 执行循环调用。
- [cameras.py](../src/agilex_control/cameras.py)：相机拥有权、持续 RGBD、按需保存；调用方断开不会关闭采集。
- [kinematics.py](../src/agilex_control/kinematics.py)：纯 FK/IK，与执行分离。

关节帧是 `0x155/156/157` 的两两 signed int32 大端毫度；模式帧 `0x151=010164ad00000000`。关节轨迹不发送夹爪命令。夹爪轨迹仅发送 `0x159`：signed int32 开度（0.001 mm）、unsigned int16 力矩设定（0.001 N·m）、状态 1（使能并执行）、零点位 0。不清错误、不改零点，不发送后部配置。真实错误、过期反馈、驱动异常和控制权冲突会终止后续目标；不恢复用户明确移除的任意瞬时跟踪/路径精度中止条件。

## 实测范围与历史证据

实机结论按设备和版本记录，通用封装、文档与离线测试不扩大已有验证范围。

| 现场/阶段 | 已有证据 |
|---|---|
| `.56` 独立观察与右臂小幅上抬 | [实测摘要](../evidence/2026-09-12/independent_primitives/independent_result_summary.json) |
| `.56` 四块积木抓取放置 | [任务记录](pick-place-20260912.md)；含夹爪、RGBD、接触误判和重抓 |
| `.56` 双臂插花及近竖直 FK/IK 修复 | [任务与回归记录](flower-arrangement-20260912.md) |
| `.38` Piper X / `.1` Piper | [Piper X 档案](../skills/wcx-agilex-control/references/site-panfeng38.md) / [houzhi1 档案](../skills/wcx-agilex-control/references/site-houzhi1.md) |
| 各次离线与实机验证的边界 | [验证记录](validation.md) |
