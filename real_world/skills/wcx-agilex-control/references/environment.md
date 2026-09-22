# agilex56 的设备与环境

本页只描述 `10.169.21.56`。其他服务器先读 [服务器选择](servers.md) 和各自档案，不沿用本页路径、相机或 CAN 映射。

当前独立工具部署与运行状态以 [primitive契约](../../../docs/primitives.md) / [交接](../../../docs/handoff.md) 为准；以下PID、服务和ROS参数是历史环境快照。交互shell的ss是source ~/.bashrc别名，必须用/usr/bin/ss或/proc/net/tcp*查端口。

核验日期：2026-09-12。IP、USB 位置、进程和反馈状态均可能变化，操作前重新核对。

## 连接与软件

| 项目 | 实测配置 |
|---|---|
| SSH | `agilex@10.169.21.56`；认证由当前会话/用户提供，不保存密码 |
| 主机 | Ubuntu 20.04.6、Linux 5.15、x86_64，主机名 `agliex` |
| GPU | RTX 4060，约 8 GiB；NVIDIA 535.171.04 |
| ROS | ROS 1 Noetic，`/opt/ros/noetic` |
| 原厂 ROS 工作区 | `/home/agilex/cobot_magic/Piper_ros_private-ros-noetic` |
| 操作便笺 | `/home/agilex/Desktop/111.txt`，只提取当前任务需要的启动步骤 |
| Evo-RL | `/home/agilex/Evo-RL` |
| 当前工具与相机服务工作区 | `/home/agilex/lwy_astra_agx`（源项目 `/home/agilex/wcx_gpt6_astra_agilex`），用户服务 `wcx-agilex-rgbd.service`；前视1280×720、双腕640×480 RGBD |
| 历史 RobotIOServer 工作区 | `/home/agilex/evorl-ljy/evorl`；与上面的 Evo-RL 目录不同 |
| 老 SDK 源码 | `/home/agilex/piper_sdk-0.3.2`；目录名不代表当前进程加载的 SDK 版本 |
| 本次远端临时证据 | `/tmp/codex_robot_check_vyxmn1b5`；可能被系统清理，项目保留归档 |

### Python 不可混用

- `/home/agilex/miniconda3/envs/evo-rl/bin/python`：Python 3.10，已验证 `piper_sdk`、`numpy`、`cv2`、`pyrealsense2` 可用；本次没有 SciPy，缺 `rospkg`。
- `/home/agilex/miniconda3/envs/pi0/bin/python`：离线规划使用 NumPy/SciPy；通过指定文件加载已安装 SDK 的 FK，未用此环境连接机械臂。
- `/home/agilex/miniconda3/envs/aloha/bin/python`：Python 3.8.19，已验证 `piper_sdk`、`python-can`、`rospy`、`rospkg`、`tf.transformations` 可导入，适合该 ROS Noetic 工作区。
- `/home/agilex/miniconda3/envs/evorl-ljy/bin/python`：Python 3.11，当前独立 primitive 与 RGBD 服务使用；NumPy、SciPy、OpenCV、pyrealsense2 已实测。16:36 历史读包元数据确认 `piper-sdk 0.6.1`；工具只导入指定纯 FK 文件，不创建机器人 SDK 实例。
- `/usr/bin/python3`：Python 3.8.10，存在 ROS/NumPy，但本次未安装 `piper_sdk`、`python-can`。系统 Python 可以运行仅使用标准库的被动 CAN 读取脚本。
- shell 中的 `python3` 可能来自 Conda base。先检查实际可执行文件与模块位置，不盲目安装包、替换 SDK 或混用 Python 3.8/3.10 的二进制 ROS 扩展。

## 四臂与 CAN

用户确认：前两臂是执行臂，后两臂是数采/示教臂。后两臂在本次调试中被现场拔下；具体拔线位置、完整主从物理拓扑未独立核验。

| 原始接口名 | USB 物理端口 | 配置角色 | 当前接口名 | 波特率 |
|---|---|---|---|---|
| `can2` | `1-13:1.0` | 前左臂 | `can_left` | 1,000,000 |
| `can1` | `1-12:1.0` | 前右臂 | `can_right` | 1,000,000 |
| `can0` | `1-4:1.0` | 第三路，原配置为底盘/其他 | `can0` | 500,000 |

本次开始时三个接口均 DOWN。本任务启用并重命名了前两路，设置 `restart-ms=100`、`txqueuelen=1000`。后续发现第三路也已 UP；该变化不是本任务直接执行的。

USB-CAN 为 candleLight / `gs_usb`。不要按 `can0/1/2` 的枚举顺序猜左右；重新插拔后要以 USB 位置和配置重新确认。后部独立 `can_back_left/right` 在本次检查时没有枚举出来，不能从接口数量推断实物数量。

## 摄像头

| 角色 | 型号 | RealSense SDK 序列号 |
|---|---|---|
| 前视 | D455 | `239622301704` |
| 左腕 | D435I | `346522074444` |
| 右腕 | D435I | `346522074314` |

三路均已收到 640×480 彩色图像，配置 30 fps；本次单帧读取不测实际帧率。USB 描述中的序列号不一定等于 `pyrealsense2` 使用的序列号，以 SDK 枚举为准。最新相机由 RobotIOServer 占用：订阅 5555，不连接动作端口 5556。若另一次现场由 ROS 占用则读已有图像 topic；不重复开启原生 pipeline。详见 [相机只读流程](cameras.md)。

## SDK 与单位

本次两前臂查询到固件 `S-V1.8-2`。调用的实际 SDK 位于运行环境 `site-packages/piper_sdk`；不能仅凭源码目录名判断接口默认值。

| 接口/反馈 | 单位与含义 |
|---|---|
| `JointCtrl`、`0x2A5..0x2A7` | 整数单位 0.001°；不是弧度 |
| `EndPoseCtrl`、`0x2A2..0x2A4` | XYZ 为 0.001 mm，RPY 为 0.001° |
| ROS `JointState.position[:6]` | 弧度；该旧驱动使用近似换算常数，精确规划优先原始反馈 |
| ROS `JointState.position[6]` | 夹爪行程，米 |
| ROS `PoseStamped` / `PosCmd` XYZ | 米；`PosCmd` RPY 为弧度 |
| `0x2A1` byte 0 | 0 待机，1 CAN 控制；不是电机使能位 |
| `0x2A1` byte 1 | 0 正常，1 急停，5 关节通信异常 |
| `0x2A1` byte 6 / byte 7 | 分别为关节限位 / 关节通信错误位；每字节低六位对应关节 1–6 |
| `0x261..0x266` byte 5 bit 6 | 驱动使能；`0x40` 为仅使能位，其他位需另解码 |
| `MotionCtrl_2` 第四参数 | `0x00` 位置速度模式，`0xAD` MIT/高随动模式，`0xFF` 无效值、不设置该字段；不能盲目切换 |

`ConnectPort(piper_init=False)` 在本次使用的 Evo-RL SDK 中可避免自动初始化查询。纯被动读取直接使用 SocketCAN，无 SDK 初始化副作用。

FK 文件：`/home/agilex/miniconda3/envs/evo-rl/lib/python3.10/site-packages/piper_sdk/kinematics/piper_fk.py`。`C_PiperForwardKinematics(1)` 与控制器位置反馈吻合到约 0.003–0.01 mm；这是内部模型一致性，不能替代实物标定和碰撞检查。

2026-09-12 读回的右臂关节限位为 J1 ±150°、J2 0–180°、J3 −170–0°、J4 ±100°、J5 ±70°、J6 ±180°；速度原始值均 300，加速度原始值均 500。离线规划额外把 J6 收窄到 ±120°。动作前仍需重新读回实际限制。
