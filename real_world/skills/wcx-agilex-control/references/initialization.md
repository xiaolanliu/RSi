# CAN 上线、使能与运动准备

用于首次接入、主机/机械臂重启、USB 重插、用户要求使能或控制恢复。先选定服务器并读取本站档案；沿用当前会话授权。单纯 observe/get_state 不隐式使能或切换模式。流程共用，型号、固件、串号、拓扑、参数和当前状态来自本站。

## 1. 确认设备和 CAN 身份

核对 hostname、账户、项目、Python/SDK、机械臂具体型号及固件。Piper、Piper H/L/X 不能仅凭系列名当成同一几何。固件版本相同也不能证明型号相同。

逐个检查接口，将 sysfs USB 路径和适配器序列号与本站档案对照：

```bash
ip -details -statistics link show type can
readlink -f /sys/class/net/已选接口/device
```

从该 USB 设备的父目录读取 `serial`。接口枚举名会变；不要按 can0/can1 或接口数量猜左右、执行/示教角色。确认每条总线实际连接的臂和反馈 ID。总线拓扑、候选绑定及实体核验程度见本站档案。

确认无其他 CAN 写入者：检查原生 ROS/SDK/服务进程，再用共享 get_state 观察控制帧。未观察到控制帧不证明绝无写入者，尤其 SDK 可以关闭本地 CAN 回环。原生 ROS 与独立写入器不能同时控制同一总线；不运行含自动使能/主从重配置的启动脚本来试连通。

## 2. 只将网络接口上线

仅对身份已确认、当前 DOWN 的接口，使用**本站已核验**的波特率：

```bash
sudo ip link set dev 已选接口 type can bitrate 本站波特率
sudo ip link set dev 已选接口 up
ip -details -statistics link show dev 已选接口
```

已经 UP 且参数正确则保留。不要为了重复初始化先 down/up；USB 重插或 BUS-OFF 后先保存状态、查占用/接线/电源，再决定恢复。需要关 CAN 时先结束本任务的目标流并确认保持状态；`ip link ... down` 只断通信，不保证停机/失能，不能作为急停。接口改名、系统网络规则和自动启动均不是动作前必需步骤。

`UP`、正确 bitrate、完整新鲜反馈及错误计数才是通信证据；CAN 的 `ERROR-ACTIVE` 是正常可通信状态，不等于有故障。历史 RX dropped 与本次 socket 接收队列丢帧分开看。调用原 get_state；驱动未使能或 ctrl_mode=0 会使 `feedback_checks_pass=false`，不应误判为 CAN 不通。

## 3. 读取本站参数，不修改配置

使用本站已经审查的 SDK，`C_PiperInterface_V2(已选接口)` 后 `ConnectPort(piper_init=False)`，跳过隐式初始化。关闭端口不会自动失能。SDK 的连接默认行为和查询/设置字段须按安装版本核对。

已在 `.1`/`.38` 审查的只读入口：

| 入口 | 用途与检查 |
|---|---|
| `SearchPiperFirmwareVersion()` / `GetPiperFirmwareVersion()` | 固件查询 0x4AF；记录真实版本 |
| `SearchMotorMaxAngleSpdAccLimit(n, 1)` / `GetCurrentMotorAngleLimitMaxVel()` | 逐关节 1–6 查询 0x472；等待新的、motor_num 匹配的 0x473，再复制数值；角度单位 0.1°、速度单位 0.001 rad/s |
| `ArmParamEnquiryAndConfig(param_enquiry=4, param_setting=0, data_feedback_0x48x=0, end_load_param_setting_effective=0, set_end_load=3)` | 0x477 只查询夹爪参数；读取 0x47E，保存范围与回包时间 |

批量查询对象可能复用缓存；逐关节核验回包 ID、时间戳并复制原始字节，不能直接将未核验的批量对象写入配置。所有查询也会增加 TX 计数，不等于发了位置目标。`GripperTeachingPendantParamConfig` 是设置命令，不可替代查询。

新参数另存版本。逐臂核对范围、几何和坐标语义；用当前关节角计算 FK，与设备末端反馈比较 XYZ **及旋转**。模型不一致先查型号/固件/安装/协议，不改零点、放宽误差或拟合常量偏移来强行通过。模型/反馈一致性不等于外部定位或 TCP 标定；本站模型来源与比较结果保存在现场档案。

## 4. 关节驱动使能

使能使用本站原 SDK，现有 8 个 JSON primitive 不包含 enable；move 不自动使能。共享的调用顺序为：

1. 用原 observe/get_state 保存完整新鲜、无故障的当前状态和场景。取得本站 `runtime_dir/motion.lock` 的同一 `fcntl.flock(LOCK_EX | LOCK_NB)` 独占锁，覆盖检查、发送、回读全过程；锁已有持有者则不并行接管。锁只能协调本工具，不能约束外部控制器。
2. 对选定臂调用 `EnableArm(motor_num=7, enable_flag=2)`（0x471）。不发送位置目标、复位、零点/限位或示教配置。已使能的臂不重复发送。
3. 原 get_state 回读，六关节 `driver_status_bytes=[64,64,64,64,64,64]` 且无错误、缺失或过期反馈，才记录关节已使能。包括 EnablePiper() 即时布尔值在内的调用返回，都不代表设备执行成功。
4. 无故障但使能尚未齐全时才有限重试；本流程最多每臂 5 次、每次之间读取 1 秒完整状态，成功即停；出现故障、控制冲突或异常先保存完整结果，停止追加命令。不要无限循环 enable。

夹爪单独记录 `gripper_enabled`、故障位、开度和物体接触。夹爪是否随该命令使能存在现场差异，见下方实测；不能因夹爪未使能而反复向关节发送 enable 或宣称关节失败。需要夹爪动作时使用原 set_gripper 工具，它发送带执行使能位的开度目标；该调用是实际夹爪动作，不能当无动作初始化。

## 5. CAN 控制模式与轻微越界恢复

CAN 上线、驱动使能、CAN 控制模式、限位内起点是四项独立状态。仅初始化关节使能后 ctrl_mode 可能仍为 0。

在当前动作授权内，复用已核验 SDK 的正常模式入口：

```python
piper.MotionCtrl_2(
    ctrl_mode=1, move_mode=1, move_spd_rate_ctrl=10,
    is_mit_mode=0, installation_pos=0,
)
```

这是正常 MOVE J/位置速度模式；`installation_pos=0` 为不设置安装位置。10% 是本次初始化用的设备速度比例，不能推广为高随动模式下的实际限速。回读 ctrl_mode 和驱动状态，观察关节、夹爪及物体接触。切换模式可能触发缓存的夹爪目标，即使没有新发夹爪帧；不能用某次开度未变来排除该副作用。

仅有少量关节轻微越界、持续无设备故障且模型/路径已核验时，可在当前动作授权内用 SDK `JointCtrl` 的正常小幅目标恢复到范围内，再重新 observe 和规划。不要求固定准备姿态，不修改或绕过共享轨迹范围检查，也不自动回零。恢复目标从**本次反馈**产生，保留其他关节，按本次场景检查路径；历史证据里的目标不是通用复原坐标。首次单发未动时先回读模式与全部状态，再决定是否短时重复同一目标。

## 6. 调用原工具并保存结果

按本站服务信息启动或复用原 cameras serve；相机服务生命周期与 CAN/驱动初始化分开，不自动使能机械臂。默认仍调用原 observe、move_left/right、move_both、set_gripper_left/right、stop。所有依赖模型的 dry run 通过后，依据当前任务执行并回读验证。`stop` 停止本工具后续目标，不撤回、失能或复位。

结果至少分开写：接口/bitrate、型号/固件及来源、各臂使能、夹爪使能、模式、当前范围与故障、FK一致性、动作实测位移、图像证据和剩余缺项。不能把部署完成、dry run 或状态通过写成任务成功。

本站实测与路径：[Piper X · panfeng38](site-panfeng38.md)；[houzhi1 的使能、恢复与双臂上抬](site-houzhi1.md#sdk-使能)。
