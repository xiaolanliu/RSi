# agilex56 · 10.169.21.56

连接字段与部署位置见 [服务器档案](../../../config/servers.json)，主机名为 `agliex`。当前工作副本为 `/home/agilex/lwy_astra_agx`；2026-09-13 已重新登录并确认源项目 `/home/agilex/wcx_gpt6_astra_agilex` 和其中的技能目录存在。这次未复测运动。详细环境以 [environment.md](environment.md) 为准，以下实测均属于本站。

## 使用独立 primitive

2026-09-15 完成五列 `[3,2,3,2,3]` 的积木城墙轮廓；最后松爪约94.24秒后新RGBD仍保持。双爪为空，左臂已抬高让出空间，右臂停在墙外，反馈完整且故障码0；这只是该时刻记录。各块仍有转动和悬边，未作抗扰/重复性验收。大量早期掉落及最终成功都保留在[城墙证据](../../../evidence/2026-09-14/host-10.169.21.56/castle-wall-evening/README.md)，流程见[积木搭建](block-stacking.md)。现场已从纸笔换为积木，任务坐标必须重新观察。

该任务将右腕346522074314从640×480增宽到848×480，原相机服务空闲停止后，用独立配置 `/home/agilex/lwy_astra_agx/runs/2026-09-14/castle-wall-evening/camera-wide-right.json` 启动 `wcx-agilex-rgbd-wide.service`，runtime仍为 `/tmp/wcx_agilex_control`。主相机1280×720、左腕640×480保持原样。实际动作配置为同目录 `control-recording-wide-right.json`，本地任务registry指向它。原config/site.json哈希未变，17模块指纹为6c7f42c5…；读取每帧内参，不套用旧主点。`wcx-agilex-recording.service`保持运行且片段已结束。新任务应独立指定录像输出，不能把本任务场景、录像目录或几何候选当作永久默认值。

2026-09-14 晚间重新接入时，CAN 原枚举为 can2/can1，经本站 USB 路径和适配器序列号核验后恢复为 can_left/can_right，1Mbps、UP、ERROR-ACTIVE；can0 保持 DOWN。两臂驱动分别使能一次，随后在新 ACE 任务授权下用正常位置指令切入控制并处理 J2/J3 的轻微初始越界。相机服务已重新启动，另增加 `wcx-agilex-recording.service` 按动作阶段录制真实 RGB 视频；本次17模块指纹与独立配置启用方法见[版本记录](../../../docs/framework-versions.md)和[录像 skill](../../wcx-agilex-action-video/SKILL.md)。[启动证据](../../../evidence/2026-09-14/host-10.169.21.56/services-enable-evening/README.md)与[书写任务证据](../../../evidence/2026-09-14/host-10.169.21.56/write-ace-evening/README.md)分开保存。ACE 已在纸上完成，左手解除压纸、右手持笔停在纸外后主图仍清楚可辨；前期夹具干涉和换抓保留为失败证据，[书写流程](pen-writing.md)记录复用边界。该书写阶段现场为纸和笔；后续场景以最新任务记录与实时观察为准。

2026-09-14 午间，在重新布置的场景和用户新任务授权下，当前毛巾按上下、左右顺序执行翻折，任务仍未完成。角点、按压及悬挂重新铺放未解决错层，后续两点提起发现只抓到部分层。066释放并退出后，毛巾在桌面左前方但未形成整齐矩形；两爪为空、反馈完整、故障码0。没有新建控制器或使用灰色固定端头的错误拟合作为正式 TCP。详见 [毛巾记录](towel-folding.md) 与 [本次证据](../../../evidence/2026-09-14/host-10.169.21.56/towel-quarter-fold-124251/README.md)。

同日凌晨的两条毛巾任务仍未完成：当时中央毛巾已放回桌面、两臂退开，右侧第二条未操作。交叉抓取期间出现疑似腕相机/附件接触与未受控臂偏移，已停止并请求现场检查，具体原因仍未获现场确认；午间动作不解释或覆盖该历史事件。随后仅修复夹爪负开度反馈导致合法开爪失败的问题，源码指纹为 b0a7ea9a13f94d12…，不再与下述对齐快照完全相同。

2026-09-14 已将本站生产源码对齐到 `.1/.38` 的16模块框架（指纹 `e5a665a90b469d16…`），包括 `phase/client/report/geometry` 和共享标定模块。本站全部8个工具已在服务器档案声明；本地与本站 staging 各93项测试通过，新 client 的三路 RGBD/状态取回及5个执行入口 dry run 通过，未执行实机动作。原现场配置和12个参数/实验 JSON 哈希不变，相机服务未重启。版本、回退和旧标定格式边界见 [部署说明](../../../docs/framework-versions.md)。

现场配置是 [config/site.json](../../../config/site.json)，左右执行臂为 `can_left`/`can_right`，三路 SDK 相机身份也保存在该配置中。本机共四臂，前两臂执行、后两臂示教/数采；用户曾拔下后两臂，不是永久拓扑保证。

服务器项目 `/home/agilex/lwy_astra_agx`，执行 Python `/home/agilex/miniconda3/envs/evorl-ljy/bin/python`。按 [primitive 契约](../../../docs/primitives.md) 的显式 `--config` 入口调用。相机运行目录 `/tmp/wcx_agilex_control`，历史由用户服务 `wcx-agilex-rgbd.service` 提供；使用前读取实际服务和健康状态。

旧 `robot_io_server.sh` 包含过时腕相机序列号和四 CAN 重配，不直接执行。旧 5555 只有 RGB 不说明硬件没有深度；若已有服务占用相机，只按 [相机只读流程](cameras.md) 订阅，不实例化同时连接动作端口的 RobotIOClient。交互 shell 的 `ss` 是 source 别名，用 `/usr/bin/ss` 或 `/proc/net/tcp*` 检查端口。

## 实测与条件性经验

- 2026-09-12 独立观察取得三路 RGBD；右臂目标上抬 20 mm、实测 17.896 mm，左臂和两夹爪开度未变。右爪空中开合 0 → 69.51 → 0 mm 有实体图像。左臂后续插花时完成运动与夹持验证。
- 19:36 桌面四块积木全部由右臂放入桶；最终前视确认桌面清空、夹爪为空。浅蓝首抓推移物体，重观测后重抓成功。详见 [抓取放置记录](../../../docs/pick-place-20260912.md)，该阶段15项离线回归通过。
- 21:16 左右臂各插入一枝花，双臂退出后两枝仍由白色花瓶支撑。滑茎重抓、花瓶移动、观察臂撤离和花头接触纠正见 [插花记录](../../../docs/flower-arrangement-20260912.md)。该阶段21项远端离线回归通过。
- 叠衣服时已验证双臂共享轨迹时钟夹持并抬衣，27项远端离线回归通过；叠衣服本身未完成。示范数据位置、悬空摆折与用户纠正见 [cloth-folding.md](cloth-folding.md)。
- 更早的 5556 路径完成约 10 cm 抬升，实测97.663 mm；短时保持和腕漂移边界见 [right-lift-endpoint-20260912.md](right-lift-endpoint-20260912.md)。历史失败执行器仅作证据，不能作为默认入口。桶位和姿态后来由现场改变，不能把已有姿态归因于未执行过的完整任务。

## 局部几何与任务资料

前视从640×480变为1280×720后，需读取每份观测自身内参；双腕保持640×480。历史局部矩阵属于本站右臂和指定相机，详见 [local-geometry.md](local-geometry.md)，权威数值在 [模型文件](../../../experiments/hd_local_translation_model.json)。现有四点映射没有独立留出精度验证，跨会话复用未验证；不能将3.7/3.9 mm拟合残差当作绝对手眼/TCP精度，也不能把右臂矩阵给左臂或其他服务器。

先通过开闭变化识别真实活动手指：本站曾把尖黑固定支架误作指尖；腕图近处黑指失焦，深度可能来自后方桌面。低位目标继续下降而实体停住时，先检查接触、卸载并重新张开对准；不能用越来越低的目标补偿受阻。夹取后提起，结合非零开度、力矩和前视/腕图确认物体随动，再搬运释放并复查。

按任务读取 [flower-insertion.md](flower-insertion.md)、[bucket-inspection-20260912.md](bucket-inspection-20260912.md)、[故障记录](issues.md) 或 [运行手册](runbook.md)。导出历史回放按 [离线导出说明](cameras.md#导出已有观测视频)，保留时间戳；离散快照不能插值伪造为连续实测录像。
