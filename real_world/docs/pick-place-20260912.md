# 桌面积木入桶 · 2026-09-12

本轮已用右臂将桌面可见的四块散落积木逐一放入桶内。**19:36:41 的最终 RGBD 观测确认可见桌面没有剩余散落积木，夹爪已空并抬离桶口。** 验收结合每块的提起、桶内上方图像、开爪反馈与最终全景；桶内原本已有积木，不靠最终单帧区分每块身份。

[最终全景](../evidence/2026-09-12/pick_place/observe_final/front/rgb.jpg) · [最后释放后的桶内图](../evidence/2026-09-12/pick_place/observe_cyan_released_at_bin/right_wrist/rgb.jpg) · [结构化结果](../evidence/2026-09-12/pick_place/result_summary.json)

## 已完成结果与证据

| 项目 | 可确认结果 | 证据与边界 |
|---|---|---|
| 第一块深蓝积木 | 提起后夹爪实际开度 27.37 mm，反馈力矩 −1.043 N·m，图像可见积木随夹爪离开桌面；后续在桶内释放 | [提起图](../evidence/2026-09-12/pick_place/observe_blue_lift/front/rgb.jpg)、[提起状态](../evidence/2026-09-12/pick_place/observe_blue_lift/state.json)、[释放后图](../evidence/2026-09-12/pick_place/observe_blue_released/front/rgb.jpg)、[释放后状态](../evidence/2026-09-12/pick_place/observe_blue_released/state.json)。释放后开度 69.44 mm；桶内原本已有积木，结论综合搬运过程、开爪反馈和前后图像，不靠桶内单帧区分每块身份 |
| 红小方块 | 提起开度26.95 mm；一次夹取成功并入桶 | [提起](../evidence/2026-09-12/pick_place/observe_red_cube_lift/right_wrist/rgb.jpg) · [桶内上方](../evidence/2026-09-12/pick_place/observe_red_cube_over_bin/right_wrist/rgb.jpg) · [释放后](../evidence/2026-09-12/pick_place/observe_red_cube_released/front/rgb.jpg) |
| 红长条 | 提起开度27.23 mm；一次夹取成功并入桶 | [提起](../evidence/2026-09-12/pick_place/observe_red_long_lift/right_wrist/rgb.jpg) · [桶内上方](../evidence/2026-09-12/pick_place/observe_red_long_over_bin/right_wrist/rgb.jpg) · [释放后](../evidence/2026-09-12/pick_place/observe_red_long_released/front/rgb.jpg) |
| 浅蓝方块 | 首抓推移物体；重新定位后开度27.16 mm，提起并入桶 | [重抓提起](../evidence/2026-09-12/pick_place/observe_cyan_retry_lift/right_wrist/rgb.jpg) · [桶内上方](../evidence/2026-09-12/pick_place/observe_cyan_over_bin/right_wrist/rgb.jpg) · [释放后](../evidence/2026-09-12/pick_place/observe_cyan_released_at_bin/right_wrist/rgb.jpg) |
| 左臂能力 | 固定路由与共享执行分支已离线验证 | 本稿不新增左臂实机移动成功声明 |

`observe_blue_grasp`、`observe_grasp2` 等目录名只是过程标签，不能按名称认定抓取成功。早期闭爪接近 0 mm 且目标被推移，并未建立成功抓取；本轮明确的成功证据是后续 `observe_blue_lift`。

## 可复用工具与相机生命周期

`set_gripper_left` / `set_gripper_right` 已纳入统一 primitive。两个入口固定对应前臂，共享 `motion.py` 的有限轨迹执行与独占锁；`trajectory.py` 区分关节和夹爪的编码及反馈要求。返回同一 `{primitive, ok, started_unix, finished_unix, data, error}`，物体识别、任务顺序和抓取结果判断仍由上层负责。契约以 [primitives.md](primitives.md) 与 [tools.json](../config/tools.json) 为准。

夹爪输入是开度 mm 和驱动力矩设定 N·m，默认 1.0 N·m；这是既有驱动默认值，不是经木积木力学标定的推荐范围或指尖牛顿力。`0x159` 仅向选定臂发送，格式为大端 `>iHBB`：开度乘 1000、力矩乘 1000、状态 1、零点位 0。状态 1 含使能并执行，不清错误或改零点。反馈 `0x2A8` 的力矩为 signed int16；FOC 位 0–5 为故障，位 6 使能，位 7 回零标记，0x40/0xC0 都可无故障。夹住物体时实际开度大于关闭目标是可能的正常接触，仍需提起图像验证。

三相机 RGBD 由用户 systemd 临时服务 `wcx-agilex-rgbd.service` 承载同一个 `cameras serve` 入口，避免进程寿命依附单次工具会话。PID 328311 是已退出的历史进程，不能用作当前状态。接手时读取实际服务状态及相机健康记录，复用现有拥有者；临时服务不等同于已安装开机自启服务。最终只读检查确认服务 active/running、三相机 ready，且没有遗留 motion call 进程，见 [运行状态](../evidence/2026-09-12/pick_place/final_runtime.json)。

前视 D455 `239622301704` 已改为 **1280×720**；左右腕默认仍为 640×480，配置帧率为 30 fps，不能以单帧证据宣称实际帧率或跨相机硬同步。配置见 [site.json](../config/site.json)，1280×720 实测见 [提起时前视元数据](../evidence/2026-09-12/pick_place/observe_blue_lift/front/metadata.json)：fx=642.00494、fy=641.31512、cx=647.33154、cy=370.47998。

相机内参、图像尺寸和深度比例必须取自对应观测。旧 640×480 ROI、像素坐标或内参不能直接用于新图，也不能简单把所有坐标乘二。历史 `right_front` 与独立工具 `front` 都指前视 D455，但文件布局和元数据不同。旧 5555 RGB 订阅仅作兼容入口；当前独立观测通过本机 Unix socket 读取 RGBD。

## 固定支架与真实夹指的误认

早期被跟踪的尖锐棕黑色结构连着三角筋板，是**固定支架**，不是可动夹指或抓取中心。固定特征可用于检查刚体相对移动，却不能由此标定夹持中心。后续根据开闭前后形状变化识别了真正的黑色可动指垫；仅看单张图中尖锐、深度稳定的边缘容易误判。

具体修正见 [离线特征记录](../experiments/gripper_features_offline.json)。其中固定支架在 `observe_vertical_closed` 的约 (335,260)，真实闭合夹指可见区域约 x=374–378、y=262–267；这些像素只属于该幅 640×480 图像，不是可复用坐标。抓取验证要看可动指垫相对目标的关系，并确认物体确实随提起动作离开桌面。

腕图中的近距离黑色指垫可能失焦、遮挡或没有有效深度，边缘还会混入后方桌面。`observe_grasp2` 的相邻样点从约 557/566 mm 跳到 623 mm，较远值不能解释为指尖；不要跨深度断层插值出精确 TCP。link6、URDF 沿局部 Z 的 135.8 mm 指关节框架、固定支架和现场指尖是不同对象。本轮没有建立通用的指尖 TCP 或相机绝对外参标定。

## 低位跟踪平台与接触线索

此前三次近桌面动作状态正常、六驱动使能，但实际关节停留在带偏差位置：

| 过程日志 | J2 目标 / 实际（°） | J5 实际减目标（°） |
|---|---:|---:|
| `blue_align_v2.result.json` | 121.642 / 119.435 | −3.379 |
| `blue_lower.result.json` | 125.446 / 122.920 | −6.206 |
| `blue_insert.result.json` | 127.211 / 122.984 | −8.519 |

上述原始动作日志已从远端归档到本地 `evidence/2026-09-12/pick_place/`，原始完整压缩包为 [pick_place-raw.tgz](../evidence/2026-09-12/pick_place-raw.tgz)。`blue_lower` 和 `blue_insert` 在终点仍持续发送约 3 秒，J2 分别保持 122.920° 和 122.984°；指令间隔中位约 33.8 ms、最大约 35.4 ms，已保存的控制帧均来自本机，未见另一发送者。

**后续实际 J2 达到 126.394°，排除了“约 123° 固定硬限位”的解释。** 见 [重新对准后的状态](../evidence/2026-09-12/pick_place/observe_blue_true_align/state.json)。改变为接近垂直姿态并重新定位后跟踪改善；不能因此证明所有早期偏差都出自同一原因。

后续 `observe_blue_side_position` 的前视与腕图显示真正可动夹指与蓝块顶部重叠、存在下压接触线索，并且物体被推移。其 link6 实际 z=122.091 mm，所请求 z=85 mm 未达到；这与接触阻碍相符。**接触力、确切接触点，以及它是否解释前述全部 J2/J5 偏差仍未测量。** 故“normal 状态证明自由空间”“只需不断降低目标”和“接触是唯一根因”均不成立。对应过程解释已写入 [特征记录](../experiments/gripper_features_offline.json)。

此前被动快照右 J2 电流约 0.827 A、J5 约 2.012 A，速度均为零；未知实际限流与负载参数，单帧不足以认定堵转。实际安装 SDK 0.6.1 的高速帧 `0x251–0x256` 解析为大端 `>hhi`，电流与速度都是 signed int16，分别乘 0.001 A、0.001 rad/s；消息文档的电流 uint16 注释与 parser 不一致。原始位置缩放未单独认证，应保留 raw。

`0x151=010164ad00000000` 是官方支持的 MOVE J 高随动配合 JointCtrl；高随动忽略速度百分比。已查官方资料未提供明确的最低目标发送率或看门狗阈值，不能将 30 Hz 定为根因。也不应把单电机 MOVE M 的要求套到此路径。[官方双臂说明](https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/double_piper.MD)

本轮诊断没有发送限位查询或切换模式。需要保留的通用经验是：综合实际关节/位姿、夹爪状态和真实接触图像解释未到位；保留已有故障与控制权检查，不重新加入用户已撤销的任意数值中止阈值。

## 浅蓝方块重抓与复用边界

首抓 `cyan_grasp` 后实际开度为0 mm、力矩约−0.038 N·m，不能认为抓住。抬起图中物体被夹爪遮挡，主动让出视野后确认它仍在桌面，且从约(844,464)推移到(884,410)并转动。新图重新定位后，沿用同一个 move/set_gripper 工具，以任务参数修正预抓取和下降位置，第二次提起图和27.16 mm夹持反馈确认成功。

真实夹指末端的深度仍有空洞；模板最高分曾落在银色相机外壳，已排除。邻近较高工具表面的488–489 mm深度也不能代替指尖。首抓失败与偏心合爪推移相符，但没有据此确定唯一几何原因。细节及带来源的即时/保存快照见 [浅蓝重抓分析](../experiments/cyan_retry_geometry.json)。

局部视觉平移映射见 [hd_local_translation_model.json](../experiments/hd_local_translation_model.json)，其适用范围是同姿态附近的可见特征，不是通用手眼标定。红小块和红长条复用后一次夹取成功，浅蓝外推目标则需要重观测；不能把四块最终成功等同于无需纠偏的通用抓取能力。

## 最终运行与验证

最终右臂 link6 为 `[428.393,194.804,374.951]` mm，RPY `[179.062,58.983,-151.114]`°；右爪69.44 mm、左爪0.49 mm，双臂反馈正常、无错误码。未向左臂发送本轮动作目标。运动调用已退出，保留驱动现有使能与独立相机服务；不自动复位、归零或失能。

15项离线测试通过，Ruff E9/F 检查与 skill 格式验证通过；前视1280×720、双腕640×480均由实机观测验证。README、工具契约、架构、交接、验证与skill引用已按本轮结果同步。个人skill仍指向同一项目源码，任务坐标和每步请求保存在evidence/experiments，工具实现没有固化桶位或物体坐标。
