# houzhi1 · 第三套平台 · 10.169.21.1

用户明确将此设备称为第三套平台。连接与部署字段见 [servers.json](../../../config/servers.json)，主机名应为 `60202263l`，账户为 `SENSETIME\houzhi1`。当前实测为两路 Piper 反馈，实物总臂数和示教拓扑尚未确认。

## 接入与状态记录 · 2026-09-13

已将三只倒扣纸杯套叠成一摞，右爪张开并撤离后由顶部与左腕图像确认稳定；两臂故障码为0。本次用9月13日外参候选初定位、当前RGBD局部纠偏和已有 `move_right` / `set_gripper_right` / `observe` 完成，未为套杯改写控制工具或现场参数。过程包含多次试提失败及第三杯搬运滑落；重抓后成功入位，末次套入中杯子提前脱离夹持并自行落稳，不能称为全程精确受控放置，也不构成绝对标定/TCP验证。见 [套杯记录与最终图像](../../../evidence/2026-09-13/host-10.169.21.1/cup-stacking-154501/README.md) 和 [可复用接触/套入几何](local-geometry.md#有外参时的夹持与套入几何)。

15:31 完成三相机首轮外参：每腕20组停稳采样，16组拟合、4组预留验证，另有返回看板位置复测。单臂运动与图像变化确认 can1/left 对应腕相机243722071316，can0/right 对应243322070942；顶部为246322300564。共享Park候选采用统一0.93打印尺度，左/右腕链留出平移RMS为4.392/3.524mm，顶部到左/右基座为3.356/3.592mm，跨臂基座关系为约6.32mm（最大10.82mm）。精确板尺寸、外部绝对定位及TCP未验证，保持候选、未覆盖控制配置。见 [参数包](../../../config/sites/houzhi1/calibration-20260913-1531.json)、[采样与误差](../../../evidence/2026-09-13/host-10.169.21.1/three-camera-calibration-151012/README.md) 和 [可复用流程](calibration.md)。最终两臂已返回看板位置，故障0，夹爪仍开；后续动作仍先取新观测。

14:56 已完成用户要求的双臂同步上抬。先用本机 SDK 的正常 MOVE J 小幅恢复至限位内，再由原 move_both 共享时钟执行 15 秒轨迹、驻留 3 秒；相对恢复前，后续 observe 回读左臂 +197.022 mm、右臂 +197.546 mm，目标为各 +200 mm。两臂 ctrl_mode=1、六关节使能、故障码为 0；夹爪原开度已恢复。未改运动源码、固件、零点或限位。见 [结果](../../../evidence/2026-09-13/host-10.169.21.1/recover-and-lift-145027/outcome.json)、[move_both 结果与实测误差](../../../evidence/2026-09-13/host-10.169.21.1/recover-and-lift-145027/lift.result.json)、[动作后图像与反馈](../../../evidence/2026-09-13/host-10.169.21.1/recover-and-lift-145027/after-lift/observation.json)。

14:45 用户要求使能，复用本机 SDK 向两路各发送一次 EnableArm(7, 2)，两臂六个关节驱动及夹爪均回报已使能，完整新鲜反馈中故障码为 0。ctrl_mode 仍为 0，J2/J3 仍超出本站范围，因此上抬 20 cm 尚未执行。见下方使能步骤与 [实测结果](../../../evidence/2026-09-13/host-10.169.21.1/enable-both-144515/result.json)。

14:42 用户明确要求双臂同时抬升 20 cm。用原 observe 获取当前 RGBD/完整反馈，再以两臂各自当前 link6 XYZ 的 Z+200 mm、保持姿态调用 move_both dry run。规划因当前 J2/J3 超出本站范围而拒绝，两臂仍为 ctrl_mode=0、六驱动未使能；未发送动作、未修改参数。须先恢复限位内正常待机姿态，再重新观察和规划。见 [本次请求与报错](../../../evidence/2026-09-13/host-10.169.21.1/both-lift-20cm-144236/plan.result.json)。

14:20 全部 8 个共享 JSON/CLI 入口已部署：get_state、observe、move_left、move_right、move_both、set_gripper_left、set_gripper_right、stop。状态与 RGBD 实机调用成功，移动/夹爪完成 dry run，stop 验证空闲返回；尚未执行运动。当时两臂未使能且 J2/J3 超出固件范围，实物左右绑定也待确认，不能把接口部署当作动作成功。

当前使用 [control-20260913.json](../../../config/sites/houzhi1/control-20260913.json)，原 state/observe 配置保留。10 个共享模块共约 50 KB，复用一个运动循环；唯一运动适配为 GripperTrajectory 按臂读取夹爪范围，原共用数组配置保持兼容。42 项离线测试全部通过，远端逐项调用与本地源码/配置哈希一致，验证阶段 CAN TX 前后均为每路 14 帧（此前只读参数查询产生），没有动作帧。

CLI 使用本机已有 `openpi` Python 3.11.14（NumPy 1.26.4 / SciPy 1.16.3）；原相机服务继续使用 `evo-rl` Python 与 RealSense，通过已有 Unix socket 复用，无新增依赖安装或服务框架。

[远端验证与实机阻塞项](../../../evidence/2026-09-13/host-10.169.21.1/integration_1415/verification.json) / [部署摘要](../../../evidence/2026-09-13/host-10.169.21.1/integration_1415/deployment-summary.json) / [42 项离线结果](../../../evidence/2026-09-13/host-10.169.21.1/integration_1415/local-tests.txt)。

## 早期共享读取接入

2026-09-13 13:41，直接复用 `.56` 的 `get_state` JSON/CLI 入口，实机返回 `ok=true`，两路反馈完整、新鲜，故障码为 0；关节及夹爪保持未使能，CAN TX 计数保持 0。

本地与 `agilex@10.169.21.56` 远端核对了 `__init__.py`、`__main__.py`、`primitives.py`、`can_bus.py`、`protocol.py` 的 SHA-256，五个文件完全一致。本站部署这五个原文件，沿用同一实现。

| 字段 | 本站值 |
|---|---|
| 项目 | `/home/SENSETIME/houzhi1/wcx_gpt6_astra_agilex` |
| CLI Python | `/home/SENSETIME/houzhi1/miniconda3/envs/openpi/bin/python`，实测 3.11.14；相机守护进程仍用 evo-rl 3.10.20 |
| 当前配置 | [control-20260913.json](../../../config/sites/houzhi1/control-20260913.json)；初始 [观察配置](../../../config/sites/houzhi1/observe-20260913.json) 和 [纯状态配置](../../../config/sites/houzhi1/state-20260913.json) 保留 |
| 独立运行目录 | `/tmp/wcx_agilex_houzhi1` |
| 已部署 | 全部 8 个入口；双臂同步上抬、两爪开度恢复、状态/RGBD 有实测；stop 仅空闲及离线验证 |
| 标定 | 三路厂家内参逐帧保存；9月13日外参候选已采样并作留出验证，打印尺寸暂估，未启用为正式控制参数；三月数据不采用 |

13:55 已部署原有 cameras.py，实测三路 RGBD 与完整机械臂反馈；14:20 完成全部入口部署、原有 FK/IK 和按臂夹爪范围校验。

## 常规取图与深度

相机常驻服务为 `wcx-agilex-houzhi1-rgbd.service`，使用本站 observe 配置。后续同一现场看图直接调用已有 observe，再取回新输出目录；observe 已包含状态和相机元数据，不重复枚举设备、部署代码或另取一次状态。只有服务失败、身份/安装变化时重新诊断。

```json
{"primitive":"observe","arguments":{"output":"/absolute/new-observation-directory"}}
```

调用命令沿用下方 JSON/CLI，选择当前 control 配置；旧 observe 配置仍可按总线标签取状态。13:55:50 实测服务端完整调用 0.582 秒，14:20 部署后为 0.508 秒，不含网络传输和模型看图时间；重启服务需另计预热。

| 视图标签 | 实测 SDK 序列号 | RGB / 对齐深度尺寸 | 本帧非零深度像素比例 |
|---|---|---|---|
| d455，全景 | 246322300564 | 1280×720 | 77.5% |
| d435i_070942，腕部近景 | 243322070942 | 640×480 | 88.7% |
| d435i_071316，腕部近景 | 243722071316 | 640×480 | 85.1% |

三路均保存 rgb.jpg、uint16 depth.npy 和逐帧 metadata.json（内参、畸变、深度比例、时间戳）。深度比例约 0.001 m/单位；0 值无深度。非零像素比例不代表测距准确度，相机之间没有硬同步。USB 描述符与 SDK 序列号不同，配置使用 SDK 身份。早期观察尚未确认左右；后续运动关联已核验左 can1/腕071316、右 can0/腕070942，见下方本机差异。

[本次图像、深度与参数](../../../evidence/2026-09-13/host-10.169.21.1/observations/observe_135550/images/observation.json) / [深度文件核验](../../../evidence/2026-09-13/host-10.169.21.1/observations/observe_135550/depth-validation.json)。

## 使用现成接口

本地项目根目录查看目标：`PYTHONPATH=src python3 -m agilex_control.sites show --site houzhi1`。选择器只显示档案，不自动连接 SSH 或操作硬件。

SSH 登录本站并核对主机名、CAN 适配器身份后，创建本次的新请求文件：

```json
{"primitive":"get_state","arguments":{"seconds":3}}
```

在服务器执行下列入口，请求和结果换成本次的新绝对路径：

```bash
PYTHONPATH=/home/SENSETIME/houzhi1/wcx_gpt6_astra_agilex/src \
/home/SENSETIME/houzhi1/miniconda3/envs/openpi/bin/python -m agilex_control \
  --config /home/SENSETIME/houzhi1/wcx_gpt6_astra_agilex/config/sites/houzhi1/control-20260913.json \
  call --request /absolute/request.json --output /absolute/result.json
```

当前配置返回 `data.states.left`、`data.states.right`，对应 can1 / can0；串号规则与本次单臂运动、左右腕图像变化一致。原 state/observe 配置仍按 can0 / can1 返回。完整工具参数见 [契约](../../../docs/primitives.md)，本站范围如下。

## SDK 使能

操作顺序、锁、查询、使能回读与重试上限统一见 [初始化流程](initialization.md)。本站使用原生 `piper_sdk.C_PiperInterface_V2`，SDK Python 为 `/home/SENSETIME/houzhi1/miniconda3/envs/evo-rl/bin/python`。

14:45 两臂各一次 `EnableArm(7, 2)` 后，六关节及夹爪均已使能；ctrl_mode 仍为 0，J2/J3 仍轻微越界。这是本站本次实测，夹爪行为不能推广到其他固件/型号。

## CAN 模式与轻微越界恢复

按 [共享恢复流程](initialization.md#5-can-控制模式与轻微越界恢复)，本站已验证 CAN / MOVE J / 10% 位置速度模式。首次单发目标没有关节位移；回读模式后短时重复同一小幅目标成功，随后调用原 move_both 完成上抬。目标来自当时反馈，仅留在证据中，未修改零点或限位。

模式切换时没有发送 0x159，两爪仍闭合到约 0.2 mm；已用原 set_gripper_left/right 恢复到约 69.4 / 98.5 mm。当时夹爪为空。此记录是共享流程中“模式切换可触发缓存目标”的实测依据。

## 本机差异与后续接入

- can0：USB `1-2:1.0`，序列号 `004D00284148571420343133`；右执行臂，运动/图像关联已核验。
- can1：USB `1-6:1.0`，序列号 `0028003F4148570C20343133`；左执行臂，运动/图像关联已核验。USB 重插后重新核验绑定。
- 两路均为 1 Mbps，关节反馈约 200 Hz；14:56 驱动及夹爪已使能，ctrl_mode=1，反馈检查通过。更早的 false 包含了模式/使能条件，不能当作通信失败。
- 三台 RealSense 的 RGBD、逐帧内参和实体绑定已核验；腕手眼及顶部到双基座外参候选、留出误差见上，不能当作完整精密手眼/TCP计量验证。
- 本机固件均为 `S-V1.8-2`。通过 SDK 只读查询实得六关节范围为 ±150、0～180、−170～0、±100、±70、±180 度；J6 与 `.56` 软件配置不同。早期静止状态两臂 J2 约 −2°、J3 为正角，确实超出本站范围；14:54 已通过正常关节控制恢复，未放宽范围或修改零点。
- 固件夹爪范围 can1 为 0～70 mm、can0 为 0～100 mm；因此 can0 的 99 mm 反馈不再是未知范围。查询 `0x477` 时设置字段全为无效，不修改示教器或行程配置。
- 本机 SDK 的 DH offset=1 与当前反馈 XYZ 误差左 0.00268 mm、右 0.00360 mm；offset=0 相差约 10 mm。共享 Kinematics 无需修改；此结果仅为内部模型/反馈一致性，不是外部定位精度或 TCP 标定。

## 已找到的相机标定数据

2026-09-13 15:02 当前桌面放置 ChArUco 打印板。用户报告名义 100 mm 实测约 93 mm；0.93 仅为暂估缩放比例，横纵等比性、精确格长/marker 长度和平整度尚未核验，未写入正式标定配置。原 observe 图像用本机 OpenCV 的 DICT_5X5_100 检出 15 个标记（ID 0～16 中缺 11、13），说明此帧图案可读，不等于标定精度通过。正式使用时按实体实测格长和 marker 长度建模，不能沿用未缩放名义尺寸。见 [图像与检测记录](../../../evidence/2026-09-13/host-10.169.21.1/board-check-150208/board-assessment.json)。

三路当前内参、畸变、深度比例保存于 observe 元数据。本站 `/home/SENSETIME/houzhi1/calib_result/` 另有 2026-03-18 两份历史手眼外参；运行日志将 17:28:35 文件关联到 `/puppet/end_pose_left`，17:45:31 文件关联到 `/puppet/end_pose_right`，实际模式均为 `eye_to_hand`。这比当前 launch 默认值可靠。

用户随后指出三月标定应该不对，本次按不适用当前现场处理，不采用，也不继续扩大排查。原始文件未修改；相机序列号、当前安装一致性、原始样本和误差仍缺失。详见 [原文件、日志与内参索引](../../../evidence/2026-09-13/host-10.169.21.1/calibration-discovery/README.md)。本站已有历史候选不等于经过当前现场复核的手眼/TCP 标定。

双臂任务流程和通用算法按 [复用边界](portability.md) 使用；相机内外参、基座变换、TCP、夹具与负载等物理参数独立保存。新缺项只影响依赖它的功能，状态读取可先使用。

## 证据

- [2026-09-13 双臂拧盖实测](../../../evidence/2026-09-13/host-10.169.21.1/bottle-cap-195721/README.md)：瓶盖分离并落桌，双爪撤离后约45秒复查稳定；候选外参、接触点估计及J6未到位的局限保留。方法见 [拧盖任务流程](bottle-opening.md)，实测参数不作为下一次默认。

- [设备发现和早期反馈](../../../docs/host-10.169.21.1-check-20260913.md)。
- [共享 get_state 实机调用与 CAN 计数](../../../evidence/2026-09-13/host-10.169.21.1/skill-reuse/get-state-verification.json)。
- [源文件对比与部署清单](../../../evidence/2026-09-13/host-10.169.21.1/skill-reuse/source-and-deployment.json)。
- [本地验证](../../../evidence/2026-09-13/host-10.169.21.1/skill-reuse/local-validation.json)：现有 5 项选站回归、skill 格式和链接检查通过；.56/.38 原档案及 .56 配置、标定与改动前一致。
