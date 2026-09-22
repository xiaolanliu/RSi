# 10.169.21.1 机械臂可用性检查

首次检查：2026-09-13 13:21–13:24 +08:00；13:33 两路通信已建立；13:41 已复用原 get_state JSON/CLI 实测成功。主机 `60202263l`，Ubuntu 20.04.6 / Linux 5.15.0-67-generic。SSH 已登录；不保存认证信息。

本文保留首次接入时的检查结果；当时已部署 get_state，相机与运动尚未接入。后续状态与工具调用以 [本站接口说明](../skills/wcx-agilex-control/references/site-houzhi1.md) 为准。

本次复用 wcx-agilex-control 的被动发现流程。此主机与原来的 `agilex@10.169.21.56` 不同，旧设备的左右路由、相机序列号、标定和实机验证结果不能直接套用。

## 当前实测 · 已建立两路通信（13:30–13:33）

**当前已确认两路 Piper 反馈可读，尚未验证运动。** 两只 candleLight 适配器分别在 13:29:31、13:29:43 接入。现有无人占用的 can0、can1 最初为 DOWN；本次只把这两个网络接口配置为 1 Mbps 并启用，没有重命名接口、发送机器人控制帧或使能电机。

| 项目 | can0 | can1 |
|---|---|---|
| USB 路径 | 1-2:1.0 | 1-6:1.0 |
| 适配器序列号 | 004D00284148571420343133 | 0028003F4148570C20343133 |
| 历史序列号规则中的角色 | right，实物待核验 | left，实物待核验 |
| 接口状态 | UP，1 Mbps | UP，1 Mbps |
| 关节/位姿/状态反馈 | 两次 3 秒采集中约 200 Hz | 两次 3 秒采集中约 200 Hz |
| 驱动状态反馈 | 约 40 Hz | 约 40 Hz |
| 完整性 / 过期帧 | complete=true，stale=[] | complete=true，stale=[] |
| 控制模式 | ctrl_mode=0，待机 | ctrl_mode=0，待机 |
| 机械臂状态 / 错误码 | arm_status=0，error_code=0 | arm_status=0，error_code=0 |
| 关节限位 / 通信错误位 | 均未置位 | 均未置位 |
| 六个关节电机 | 全部未使能 | 全部未使能 |
| 驱动电压 | 23–24 V | 23 V |
| 夹爪反馈开度 | 99 mm | 69.79 mm |
| 夹爪状态 | 未使能，fault bits=0 | 未使能，fault bits=0 |
| 采样期间关节变化 | 六关节均为 0° | 六关节均为 0° |

当时被动读取工具来自本地技能的原文件（原读取器现已 [归档](../evidence/2026-09-13/skill-consolidation/read_state.py.txt)）：`scripts/read_state.py`、共享 `src/agilex_control/protocol.py` 及包初始化文件，复制到远端 `/tmp/wcx_agilex_discovery_20260913_1330`；逐文件 SHA-256 与本地一致。没有调用会初始化硬件的 SDK 实例或运动 primitive。

两次采集均未发现控制帧，内核 TX 计数保持 0。接口处于正常的 CAN 错误管理状态 `ERROR-ACTIVE`，bus-errors/bus-off 为 0。早期 RX dropped 计数非零，原因未确定；第二次 3 秒采集专门比较前后计数，两路均增加约 9,200 个接收包，rx_dropped/rx_errors/tx_packets/tx_errors 增量均为 0，反馈帧每组各 600 帧。

共享解码器的 `feedback_checks_pass=false` 不能解读成通信失败：它还要求 ctrl_mode=1 且所有驱动已使能；当前两臂均待机且未使能。通信可读不等于已经满足执行条件。

当前关节角（度）：
- can0：[-3.334, -2.663, 1.166, 6.337, 25.678, -18.496]
- can1：[0, -2.004, 1.511, 2.593, 20.464, -3.999]

**迁移配置前仍需核对：** 两路反馈的 J2 为负、J3 为正，超出原 10.169.21.56 配置的 J2/J3 软件范围；can0 夹爪反馈 99 mm 也超出旧配置 0–70 mm。尚未核实实物行程、零点和比例原因，不应直接裁剪反馈、回零或修改关节限位使动作通过。原相机序列号和左右实物角色也仍待核对。本次没有执行这些操作。

[完整上线证据与两次采样](../evidence/2026-09-13/host-10.169.21.1-connected-1333.json)。两路适配器及 Piper 反馈不代表已确定现场实物总臂数或示教拓扑。

## 首次检查 · 13:21–13:24（历史状态）

**首次检查时这台主机未检测到可读取反馈的 Piper 机械臂。** 当时没有 CAN 接口，也没有枚举出的 USB-CAN 适配器，无法验证机械臂通信或动作。这不证明现场没有机械臂，也不证明机械臂本体故障；最新已连接结果见上方。

| 检查项 | 实测结果 |
|---|---|
| 主机网络 | `enp0s31f6 = 10.169.21.1/24` |
| CAN 接口 | `ip -details -statistics link show type can` 无输出；sysfs 只有 lo、docker0、enp0s31f6 |
| USB-CAN | USB 只枚举出三台 RealSense、键盘、鼠标和根集线器；没有 gs_usb/candleLight 适配器 |
| 驱动 | 内核目录存在 `gs_usb.ko`；本次未加载 can、can_raw、gs_usb、slcan 模块 |
| 控制服务 | 进程匹配未发现 Piper、AgileX、RobotIO、ROS master/launch、LeRobot 或 RealSense 服务；未见 11311/5555/5556 监听 |
| 串口 | 未发现 ttyUSB、ttyACM 或 serial/by-id 设备 |
| 原有软件 | 存在 `~/piper_ros`、`~/catkin_ws`、`~/Evo-RL-real-world`、ROS Noetic |
| 独立 primitive | 当前用户家目录中未发现 `wcx_gpt6_astra_agilex` 部署 |
| 软件包文件 | evo-rl 中存在 piper_sdk 0.6.1 和 pyrealsense2 的包文件；未以导入测试宣称运行可用 |
| 内核日志 | 初次普通权限 journalctl 未读到系统日志；后续通过 sudo 只读 dmesg 成功，详见下方复查 |

本机旧版 ip 的 JSON 过滤输出为 `[{},{},{}]`，不能按数组长度推断 CAN 数量。文本输出和 sysfs 枚举一致表明当前没有 CAN 接口。

## 通电后复查 · 13:26–13:28

用户报告“现在应该连上了，通电了”后，13:26:31 和 13:28:14 两次枚举仍无 CAN 接口或 USB-CAN 适配器。25 秒 USB/网络内核事件监听未收到任何事件。

随后只读内核日志成功：当前匹配到的 USB 设备接入记录均在本次启动时的 13:11:27–13:11:30，涉及键盘、鼠标和三台 RealSense；未见 CAN 适配器接入记录，也未见本次接线对应的 USB 枚举失败记录。日志中存在启动时通用的 `usb: port power management may be unreliable` 提示，但没有证据将其认定为当前缺少适配器的原因。

结论仍是本主机未枚举到适配器，无法读取机械臂反馈；接线位置、USB 数据线、转接器/集线器及供电需要现场确认。没有执行模块加载、CAN 配置、模式切换、使能或动作。原始证据见 [13:28 复查 JSON](../evidence/2026-09-13/host-10.169.21.1-recheck-1328.json)。

## 相机枚举

仅确认 USB 枚举与设备节点，没有启动采集或验证 RGBD 帧。USB 描述符序列号未通过 RealSense SDK 另行核验，不能直接作为最终 site.json 绑定。

| 型号 | USB 路径 | USB 描述符序列号 |
|---|---|---|
| D435i | 2-1 | 252943061456 |
| D455 | 2-3 | 302623061751 |
| D435i | 2-4 | 252943060008 |

用户权限下的 fuser 检查未见 video 设备占用者；这不是对不可见进程的完整排除。三台相机的前视/左右腕安装角色未独立确认。

## 配置证据与冲突

远端 `~/Evo-RL-real-world/can_config_new.sh` 预期两个 1 Mbps 接口：
- USB `1-2:1.0` → `can_right`
- USB `1-10:1.0` → `can_left`

另一份 `setup_can_names.sh` 的历史注释与序列号规则为：
- USB `1-2`、序列号 `0028003F4148570C20343133` → `can_left`
- USB `1-10.3`、序列号 `004D00284148571420343133` → `can_right`

上述历史 USB 端口的左右定义不一致；首次检查时没有适配器，无法裁定。后来接入的序列号与 setup_can_names.sh 中的两条规则相符，可作为 can0=right、can1=left 的候选映射，但实物角色仍未确认。不能按接口枚举顺序或任选一份历史脚本确定左右。当前 `1-10` 枚举的是鼠标，左侧候选适配器位于 `1-6`，旧端口配置不能直接执行。配置中的双路预期也不能证明现场实物机械臂总数或示教/执行角色。

远端 `~/piper_ros/start_single_piper.launch` 默认 `auto_enable=true`，不适合作为本次被动检查入口。以上脚本均未执行；没有发送 CAN 指令、切换模式、使能、运动、夹爪操作或改写设备配置。

## 后续检查入口

后续接入、运动与标定进展统一维护在 [本站档案](../skills/wcx-agilex-control/references/site-houzhi1.md)。重复初始化按 [共享流程](../skills/wcx-agilex-control/references/initialization.md)，状态读取使用原 get_state；本页的早期范围疑问和角色候选不代表当前结论。

## 原始证据

[主机只读枚举 JSON](../evidence/2026-09-13/host-10.169.21.1-inventory.json)。记录时间、命令与输出、USB/sysfs、驱动和包文件发现；不包含密码或认证文件。
