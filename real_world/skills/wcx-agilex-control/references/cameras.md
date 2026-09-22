# 不干扰现场的相机读取

当前默认入口已是独立 `observe`：`wcx-agilex-rgbd.service` 持续采集前视 1280×720、双腕 640×480 RGBD，每路深度对齐到自己的 RGB。见[工具契约](../../../docs/primitives.md)和[抓取记录](../../../docs/pick-place-20260912.md)。以下 5555/PID/ROS 信息是历史快照；新任务先查实际拥有者与健康状态。

## ROS与独立相机服务切换时的读取

2026-09-14 12:40，本站数采源码的主视角订阅为 `/camera_f/color/image_raw`，曾注册发布者 `/camera_f/realsense2_camera_manager`。话题注册不能证明仍在产帧：本次ROS读取未取得消息，随后master不可达、发布者连接拒绝，相关进程已退出。读取应同时核验实际进程、消息时间与设备占用，给整个读取进程设置总超时，不能只限定wait_for_message。

若现有服务在运行则复用它，不争用设备。仅在已核实相机空闲且会话授权采集时，按所需相机从本站配置选取序列号/分辨率，使用独立临时配置与运行目录复用原 cameras serve/capture/stop；不覆盖活动配置，单次取图后关闭本次拥有的服务并验证退出。证据见[主视角读取](../../../evidence/2026-09-14/host-10.169.21.56/front-view-123703/README.md)。

## 导出已有观测视频

本轮相机采集只缓存最新帧，observe按需保存RGBD，未保存连续录像。积木任务保存52组、插花任务保存37组三路快照，可各自按原始时间排序导出回放，不能声称恢复了未记录的运动。离线工具为 [export_observation_video.py](../../../scripts/export_observation_video.py)，只读本地文件，不连接相机或CAN；依赖Pillow与FFmpeg（也可由imageio-ffmpeg提供），用 `--font` 指定支持中文的字体文件。若Python环境没有imageio-ffmpeg但已有FFmpeg可执行文件，可用 `--ffmpeg /absolute/path/to/ffmpeg` 直接复用。

输入为包含各次观测目录的路径，输出为新目录。默认上前视、中左腕、下右腕；另外导出三路单独MP4。每组展示1.25秒、最后一组4秒，52组共67.75秒，24fps。每路显示自身原始采样时间，保留完整视场并用黑边维持比例；标注“快照等间隔回放·非连续录像”。manifest记录观测与播放时长的映射，不承诺跨相机硬同步。

积木视频见 [导出说明](../../../exports/2026-09-12/pick_place_video/README.md)。插花使用相同脚本和布局，37组共49秒，原始时间20:10:08–21:16:04，见 [插花视频](../../../exports/2026-09-12/flower_arrangement_video/README.md)。拼接版960×1800、H.264/yuv420p、无音频，三路与拼接文件均已解码核验，另检查开头/中段/结尾画面。录制连续视频属于未来采集需求，不能用本轮快照代替。

最新新增：桶任务将 5555 的 14 个关节标量与图片一起保存；原服务退出后取得独立 D455 RGBD 一帧，具体占用检查、内参、标定与适用条件见 [桶内观察](bucket-inspection-20260912.md)。5555 本身仍为 RGB，无深度。以下按时间保留历史读取验证。

北京时间 **2026-09-12 16:07:56**，用户请求当前观测，沿用同一只读脚本成功收到序号 47207 的三路 640×480 RGB 图像后退出。快照保存在项目 `evidence/2026-09-12/observations/160756/`，见 [证据索引](../../../evidence/README.md)。该次未连接 5556 或改动现场进程。后续右臂试验前后亦复用此读取器，最新图像时间 16:34:53、序号 95562，保存在 `evidence/2026-09-12/right_lift_via_5556/after/`；动作阶段与其授权另见 [试验记录](right-lift-20260912.md)。以下保留首次协议核验结果。

## 2026-09-12 实测结果

用户确认 `5555` 是相机端口、`5556` 是动作端口。机械臂正在被其他人使用，本次只从已有服务订阅一条观测后退出。

| 观测键 | 角色 | SDK 序列号 | 成功收到的数据 |
|---|---|---|---|
| `left_wrist` | 左腕 D435I | `346522074444` | 640×480×3，uint8 RGB |
| `right_wrist` | 右腕 D435I | `346522074314` | 640×480×3，uint8 RGB |
| `right_front` | 前视 D455 | `239622301704` | 640×480×3，uint8 RGB |

前视挂在右臂配置下，因此键名是 `right_front`。已目视核对三张 JPEG；前视可见桌面、装有方块的桶和散落方块。

采样 UTC `2026-09-12T07:46:09.008577+00:00`（北京时间 15:46:09），序号 8087，消息大小 2,765,513 字节。封装时间与读取时间相差约 6 ms，仅表示消息封装年龄，不是曝光到显示延迟。配置为 30 fps，但只收一帧，未测实际帧率或多相机同步。另有 14 个非图像字段，本次没有把它们解释为动作任务。

快照与元数据见 [证据目录](../../../evidence/README.md)。未连接 5556 或 5557，未访问相机设备、发送机器人命令或更改现场服务。

## 为什么先查占用者

此前曾有三路 ROS 相机；此次最新图中没有相机 topic/publisher，只有残留的相机参数。设备由 `evorl-ljy` 的 `scripts/robot_io_server.py` 使用。不要因 `rostopic list` 无相机就启动相机节点。

实测进程 PID 19216（动态值，不能复用来终止进程）：

- 可执行文件：`/home/agilex/miniconda3/envs/evorl-ljy/bin/python3.11`。
- 工作目录：`/home/agilex/evorl-ljy/evorl`。
- 入口：`scripts/robot_io_server.py`，转入 `lerobot.scripts.robot_io_server.main`。
- 设备映射：USB `2-1` / video0–5 为 D435I，`2-3` / video6–11 为 D455，`2-5` / video12–17 为 D435I。这些枚举也可能变化。

只读检查进程命令行、`/proc/PID/fd`、USB sysfs、监听端口及 ROS 图即可辨认占用者，无须额外初始化摄像头。

## 历史 RobotIOServer 服务协议

源码根目录：`/home/agilex/evorl-ljy/evorl/src/lerobot/`。

| 端口 | Socket | 实现与用途 |
|---|---|---|
| 5555 | 服务端 PUB → 读取端 SUB | `robots/robot_io/server.py` 发布观测；本次唯一连接端口 |
| 5556 | 服务端 PULL ← 客户端 PUSH | 动作通道；相机读取禁止连接 |
| 5557 | 服务端 REP | 元数据；本次没有访问 |

启动配置见 `scripts/robot_io_server.py`，实际进程配置观测地址 `tcp://127.0.0.1:5555`、频率 30。`server.py` 发布单条二进制消息，`serialization.py` 使用 pickle 最高协议。`robots/robot_io/client.py` 的 `RobotIOClient.connect()` 同时建立观测 SUB 和动作 PUSH，所以相机只读不调用这个客户端。

`cameras/realsense/configuration_realsense.py` 默认 `ColorMode.RGB`，本次参数未覆盖它。保存 JPEG 时转 RGB→BGR；后续换服务时要重验。消息格式、键名和颜色约定均为当前实现，不是所有 AgileX 的通用协议。

## 可复用读取命令

将本技能脚本复制到远端临时目录后，在服务器运行：

```bash
/home/agilex/miniconda3/envs/evorl-ljy/bin/python /path/to/read_camera_5555.py \
  --timeout 5 --output-dir /tmp/agilex_camera_snapshot
```

脚本只接受 `tcp://HOST:5555`；默认 loopback，避免额外开放网络端口。SUB 配置 `CONFLATE=1`、接收队列 1、`LINGER=0`、消息上限 32 MiB，最多等 5 秒、接收一次后立即关闭。它会收到整个已有观测包，只保存图像和必要元数据。

限制反序列化只允许当前协议必要的 NumPy 全局项，不能把它当作任意不可信 pickle 的通用沙箱；仅用于已确认的现场服务。出现不支持的类型应检查协议，不能直接解除限制。

无消息时排查现有进程与观测绑定地址。纯只读任务到此停止，不重启服务、不重配相机、不启动额外采集或 ROS 节点。持续高频预览不是本次验证范围。
