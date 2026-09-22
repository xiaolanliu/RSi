# 多服务器入口与历史交接

2026-09-13 共用一套技能，先按 [服务器选择](../skills/wcx-agilex-control/references/servers.md) 确定 `.56`、`.38` 或 `.1`，部署与实测范围以各站档案为准。`.56` 后续还有双臂抬衣及叠衣服未完成记录，见 [本站档案](../skills/wcx-agilex-control/references/site-agilex56.md)。

`.1` 三相机已完成首轮外参采样与求解（每腕16训练+4留出）；参数仍为统一0.93打印尺度的候选，跨臂留出平移一致性RMS约6.32mm、最大约10.82mm，尚无外部绝对计量或TCP验证。标定阶段结束时两臂返回看板位置，此后已执行套杯，不能把看板位当成当前状态。见 [标定记录](../evidence/2026-09-13/host-10.169.21.1/three-camera-calibration-151012/README.md) 和 [复用与重标定条件](../skills/wcx-agilex-control/references/calibration.md)。

## agilex56 当前毛巾结果 · 2026-09-14 午间

用户重新指定当前一条毛巾先上下、再左右对折。`towel-quarter-fold-124251` 已执行两折并进行了角点、按压、悬挂及两点修整，但抓层不全和错位未解决，**任务未完成**。066双臂松爪退开后，毛巾在桌面左前方，没有形成整齐矩形。056/057为后续两点尝试前的中间状态，不能用来代替最终图。托盘内原有卷毛巾不是本次成果，凌晨两条毛巾目标也未完成。详见 [本轮证据](../evidence/2026-09-14/host-10.169.21.56/towel-quarter-fold-124251/README.md) 与 [毛巾技能边界](../skills/wcx-agilex-control/references/towel-folding.md)。

本轮使用原 client/phase/observe/move/gripper，未改控制源码或配置。固定灰色端头误认为夹持点的 PnP 仅留作反例，未写入控制标定；031回传中断通过原请求、原远端目录取回 journal，未重放动作。相机服务仍运行；最终位姿、故障位及后续稳定复查在本轮证据中，下次先读新状态。凌晨疑似附件接触的现场原因仍未确认，本轮没有跨臂夹取时把另一臂留在工作区。

## houzhi1 双臂拧盖任务交接 · 2026-09-13

最新 `bottle-cap-195721` 已完成：左臂固定瓶身，右臂分段旋松并取下瓶盖，随后将瓶盖放在旁边桌面、释放瓶身并撤开双爪。最终新帧在双臂脱离后44.97秒仍显示瓶身直立、瓶口敞开、瓶盖独立放置；反馈完整、故障位均为0。见 [本轮证据与局限](../evidence/2026-09-13/host-10.169.21.1/bottle-cap-195721/README.md) 和 [结果](../evidence/2026-09-13/host-10.169.21.1/bottle-cap-195721/outcome.json)。任务使用现有 client/phase/geometry 和原 primitive，未新写控制工具。初期空夹、IK失败、接触点估计及第77阶段J6约12°未到位均留有记录，不能把此次成功当作精密TCP或通用拧盖能力已验证。当前姿态已不同于历史套杯结束位，下次任务从新观察开始。

拧盖方法已沉淀在原 skill 的 [任务专页](../skills/wcx-agilex-control/references/bottle-opening.md)，硬件和几何仍引用共享实现。该页尚未经另一瓶型或现场复测。[上前、中左腕、下右腕回放](../exports/2026-09-13/host-10.169.21.1/bottle_cap_video/README.md)由66组三视角快照组成，85.25秒，保留逐路时间戳，不是连续录像。

## houzhi1 套杯任务交接 · 2026-09-13

最新 `cup-nesting-framework-1757` 已完成三只倒扣纸杯套叠，主动开爪撤离后75.08秒的新帧仍稳定；右臂执行、左臂保持观察，两臂最终反馈完整、无故障。23阶段全部 completed，24条原执行 primitive；没有掉杯或重抓，但第三杯第一次短试提需深度和补提复核，不能排除小幅就位滑动。两只搬运杯均在主动释放后由重力落稳，见 [本轮结果](../evidence/2026-09-13/host-10.169.21.1/cup-nesting-framework-1757/outcome.json) 与 [最终侧视图](../evidence/2026-09-13/host-10.169.21.1/cup-nesting-framework-1757/35-stability-check/observation-000/d435i_071316/rgb.jpg)。

首轮的试提失败、搬运滑落、重抓及末杯自行滑入另保留在 [历史证据](../evidence/2026-09-13/host-10.169.21.1/cup-stacking-154501/README.md)。两轮均不能视为绝对几何/TCP验证；本轮白色包边选点的法向复查不一致，未验证精确杯轴或扶正。下次任务重新读取 RGBD/反馈，不重放接触偏移、杯坐标和力度。

叠杯流程已整理在原 skill 的 [任务专页](../skills/wcx-agilex-control/references/cup-nesting.md)，几何换算仍引用 [共用几何](../skills/wcx-agilex-control/references/local-geometry.md)。接触、试提、持稳搬运、套入、释放和撤离后稳定分别验收；滑移后局部点和方向一起失效，取消依赖它们的后续目标。外参继续保持候选 `accepted_for_control=false`，TCP 未计量验证。

## 通用框架交接

2026-09-14 `.56` 已补齐并对齐到同一16模块源码。新 client/phase 的 RGBD与状态链路和 dry run 已通过，本轮未发实机动作；原配置和相机服务保留。三站代码版本、技能副本归属与恢复路径见 [框架版本](framework-versions.md)。后续 `.56` 日常观察可直接使用同一 client，不再另写 SCP/SSH 观察封装；旧现场标定格式仍有明确边界。

共享源码新增有限 `phase`、同站传输 `client`、只读 `geometry` 和实测 `report`，仍委托原8个 primitive。`call`/`phase` 接收 JSON 文件或 stdin；阶段输出目录绑定请求和配置，动作完成但观察失败时只补观察，通信不确定时不换目录重放。几何参数必须来自对应实体和当前观测，任务专页不再复制硬件契约。完整调用与文件布局见 [接口文档](primitives.md)，验证状态见 [validation](validation.md)。

本地与 `.1` 远端最终全套93项测试均通过，16个共享源码文件已部署到 `.1`，原源码已备份；本站配置和标定哈希保持不变，见 [部署记录](../evidence/2026-09-13/host-10.169.21.1/framework-efficiency-1727/deployment.json)。新 client 的观察链路和上述叠杯任务已分别实测，详见 [验证范围](validation.md)。接手仍先核对目标现场版本和新状态。

本轮从首观察到撤离约21分51秒，到稳定复查约23分06秒；client 阶段调用累计208.43秒，三视图/元数据传回累计1.7545秒。完整阶段记录与图像已在本地，原始深度仍在远端对应阶段目录；时间统计和计量局限以本轮结果为准，不据此宣称任意杯型的速度或成功率。

## agilex56 插花阶段交接 · 2026-09-12 21:16

以下姿态、PID、服务和授权表述均是该阶段的历史记录，实际操作使用当前会话目标与现场状态。

两枝桌面的花已插入白色花瓶，左右臂各完成一枝。最终图确认两条花茎入瓶、花瓶直立、双爪为空，双臂退出后两枝仍由花瓶支撑。先读 [插花记录](flower-arrangement-20260912.md)、[结构化结果](../evidence/2026-09-12/flower_arrangement/result_summary.json) 和 [工具契约](primitives.md)。此前四块积木入桶的结果保留在 [积木记录](pick-place-20260912.md)。

### 工具与最终快照

主要入口为 observe、move_left/right、set_gripper_left/right、stop。左右固定设备，共享有限执行器；实现集中在 src/agilex_control，现场绑定在 config。服务器 `/home/agilex/lwy_astra_agx`，Python `/home/agilex/miniconda3/envs/evorl-ljy/bin/python`。不重放证据中的任务坐标。

- 相机用户服务 `wcx-agilex-rgbd.service` 为 active/running，PID537936；PID只是最终检查的快照。前视1280×720，双腕640×480，分别对齐RGB深度并携带逐帧内参。
- IPC为 `/tmp/wcx_agilex_control/cameras.sock`，健康记录为 `camera_health.json`。新任务读取实际健康情况，旧5555/5556不是本轮执行链。
- 左 link6 XYZ `[196.001,3.788,309.118]` mm，右 `[196.223,4.619,310.569]` mm；左爪69.79 mm，右爪69.44 mm。双臂错误码0、反馈完整。左右移动、夹持和各插入一枝已有实物验证。
- 无遗留 motion call 进程。保留驱动使能和相机服务，没有失能、归零或复位。[运行快照](../evidence/2026-09-12/flower_arrangement/final_runtime.json)

### 当时实现与经验

共享运动学已修复近pitch±90°时SDK Euler截断引起的求解异常：`Kinematics.transform(q)`直接组合安装SDK的modified-DH矩阵，目标/残差使用原始旋转。不要从显示RPY重建精确旋转。SDK私有DH接口是依赖，升级后复核。21项远端离线回归通过，修改文件Ruff E9/F通过；没有放宽关节边界或精度标准，也没有增加专用动作执行器。

细长物体操作读 [插入技能](../skills/wcx-agilex-control/references/flower-insertion.md)：抓后重测茎轴、分段翻转、检查相对滑动、侧视对瓶口、插入后宽开退出。首抓0.3 N·m设定下滑转约60°，重抓1.2 N·m对这两枝有效，不代表通用夹持力标定。细茎和黑指的非零深度也可能是背景。

花瓶在首抓失败放回过程中被带近，已按新位置完成任务。线缆/夹具牵带是怀疑，具体接触点未测得。观察臂也曾靠近活动工具，右支架接触过已有花头；随后撤远、解除接触并调整。不能把设备无错误码当作没有物理接触。两枝已插好，下一任务从新观测开始。

没有新增完整手眼/TCP标定。link6不是指尖，历史外参和右臂局部平移映射都带适用边界；左右、新姿态与新安装不能直接共用数值。校正流程见 [局部几何](../skills/wcx-agilex-control/references/local-geometry.md)。

用户已有动作授权持续有效，最新停止指令优先；不因历史文档重复询问已确认的现场环境。保留真实故障、过期反馈、驱动异常、控制权和现有SDK边界检查，不恢复已移除的任意瞬时精度、步长、速度中止阈值。

### 归档

远端本轮请求、结果、遥测与RGBD已归档并解压至 `evidence/2026-09-12/flower_arrangement/`；[原始包](../evidence/2026-09-12/flower_arrangement-raw.tgz)。只有离散快照，无连续视频。几何分析与结果摘要另存证据目录。

个人技能符号链接仍指向项目技能源码；根AGENTS.md指向CLAUDE.md。SSH密码未写入文件。远端shell的ss是source别名，检查端口用/usr/bin/ss。
