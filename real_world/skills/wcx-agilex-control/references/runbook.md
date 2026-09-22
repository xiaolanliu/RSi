# agilex56 历史 ROS / 5555 排查

日期：2026-09-12。连接、三路 RGBD、右臂独立运动与夹持已实测；早期通信异常保留为历史未决问题，不能据此覆盖最新验证，也不能抹去历史故障。见 [本轮抓取](../../../docs/pick-place-20260912.md) 与 [问题记录](issues.md)。

当前首选已部署的独立 primitive，见 [工具契约](../../../docs/primitives.md)。本页保留 ROS/旧5555/5556 排查参考，不是当前动作执行入口。权限以会话最新指令为准。

相机优先复用 `wcx-agilex-rgbd.service`，调用 `observe` 获取 RGBD 与当前反馈；按选定臂调用 move 和 set_gripper，固定路由、共用执行器。不要因旧 PID 消失或 ROS 无图像 topic 就启动第二个相机拥有者。

## CAN 与当前工具入口

各站 CAN 上线、SDK 参数查询、使能和恢复统一按 [初始化流程](initialization.md)。`.56` 的接口身份与旧系统脚本说明见 [环境档案](environment.md)；本页不再提供第二套初始化步骤。

状态读取首选已有 `get_state`。保留的 `scripts/read_state.py` 是它的薄命令行适配，需要一同部署 `src/agilex_control`，接口及 JSON 格式见 [工具契约](../../../docs/primitives.md#被动状态读取兼容入口)。

## ROS：先发现，再启动

便笺 `/home/agilex/Desktop/111.txt` 的相关步骤是 `roscore`、CAN 配置、加载 catkin 环境、`start_ms_piper.launch` 和 `multi_camera.launch`。便笺还含历史模型推理、文件上传和大幅关节姿态，与读取前臂 topic 无关，不执行它们，也不把便笺全文复制进项目。

```bash
source /opt/ros/noetic/setup.bash
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash
rosnode list
rostopic list
```

如果 ROS master 和 Piper 节点已运行，直接复用。本次发现已有节点，未重复启动。

如果后续任务允许启动设备、没有其他占用者且节点未运行，分别在受管理终端中启动：

```bash
# 终端 1
source /opt/ros/noetic/setup.bash
roscore
```

```bash
# 终端 2：确保 python3 使用有 SDK 和 ROS 依赖的 Python 3.8 环境
source /home/agilex/miniconda3/etc/profile.d/conda.sh
conda activate aloha
source /opt/ros/noetic/setup.bash
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=false
```

便笺原命令为 `auto_enable:=true`。当前节点源码显示，自动使能和 `/enable_flag` 回调都会额外操作夹爪；只读阶段使用 false。现场已运行 true 的实例时，不擅自终止或重启它；先查发布者、订阅者和真实硬件状态。

只有后续任务允许启动相机、设备无人占用且服务未运行时，才使用配置好的环境启动 `roslaunch realsense2_camera multi_camera.launch`。若现场仍由旧 RobotIOServer 拥有设备，可只读 5555；当前本轮由独立 RGBD 服务拥有，使用 observe。检查 RealSense SDK 序列号，不同时用第二个 pipeline 占用设备。

## 已核验 topic

| 方向 | 左侧 | 右侧 | 类型 / 单位 |
|---|---|---|---|
| 反馈 | `/puppet/joint_left` | `/puppet/joint_right` | `sensor_msgs/JointState`；前六项 rad，末项 gripper m |
| 反馈 | `/puppet/arm_status_left` | `/puppet/arm_status_right` | `piper_msgs/PiperStatusMsg` |
| 反馈 | `/puppet/end_pose_left` | `/puppet/end_pose_right` | `geometry_msgs/PoseStamped`；m、四元数 |
| 反馈 | `/puppet/end_pose_euler_left` | `/puppet/end_pose_euler_right` | `piper_msgs/PosCmd`；m、rad |
| 关节命令 | `/master/joint_left` | `/master/joint_right` | `sensor_msgs/JointState`；旧驱动回调也发送夹爪控制 |
| 位姿命令 | `/puppet/pos_cmd_left` | `/puppet/pos_cmd_right` | `piper_msgs/PosCmd`；旧驱动回调使用 50% 速度 |

**`/enable_flag` 是两条前臂共享的命令 topic**，会同时触发使能/失能和夹爪指令。读取反馈不需要发布它。`/master/joint_*` 在 `mode=1` 下是从臂命令入口，名称中的 master 不代表它只控制后臂。

```bash
rostopic info /puppet/joint_left
rostopic hz /puppet/joint_left
rostopic echo -n 1 /puppet/arm_status_left
rostopic echo -n 1 /puppet/end_pose_euler_right
```

发布前先用 `rostopic info` / ROS graph 确认现有命令发布者。2026-09-12 本次快照中没有 `/master/joint_*`、`pos_cmd`、`/enable_flag` 发布者；驱动订阅这些入口。

旧驱动关节回调写死 `MotionCtrl_2(..., 100)`，且同时发夹爪命令，不能通过在 topic 中添加不存在的速度字段得到低速保障。未修改并核验回调前，不把它包装成“仅抬臂、夹爪保持”的执行工具。

## 当前规划与历史追溯

规划统一使用原 move_left/right/move_both 的 `dry_run:true`，契约与调用示例见 [primitive](../../../docs/primitives.md)。这与执行使用同一份型号配置、FK/IK 和范围检查。

早期 `plan_lift.py` 及独立 CAN 读取循环已 [原样归档](../../../evidence/2026-09-13/skill-consolidation/README.md)，不再作为执行入口。历史 20 mm、8 秒等参数仅描述当时实验，实机结论见 [抬升重试](right-lift-endpoint-20260912.md)；不能直接重放到当前姿态或其他型号。
