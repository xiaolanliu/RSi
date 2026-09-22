# 验证范围

## agilex56 毛巾任务与夹爪边界修复 · 2026-09-14

毛巾目标未完成：第一条未折整齐、未移左；第二条未操作。35号阶段左臂接近时，未受控右臂位移约47 mm，图像提示附件可能接触。已保存完整状态、分离两臂、放下毛巾并退开，等待本次附件检查；无错误码不能排除物理接触。见[任务记录](../evidence/2026-09-14/host-10.169.21.56/towel-folding-0025/README.md)与[视频学习](../evidence/2026-09-14/host-10.169.21.56/towel-folding-0025/video-learning.md)。

修复合法开爪请求因−0.07 mm反馈而在首帧失败的问题，只投影命令起点、保留原测量；只部署到本站。当前源码版本为 `b0a7ea9a13f94d12f533bbd52ca2168f1de388a21b3b6a1d6b48f52202a06498`，本地与远端16模块一致。两处各96项离线测试通过、无跳过，其中远端是隔离staging；本站生产旧测试目录首次运行实际为30项中1项导入失败，不记成96项通过。详见[版本记录](framework-versions.md)。测试和工具执行成功均不代表毛巾任务成功。

下节是此前93测试与三站源码对齐的历史快照；本次夹爪修复后本站版本已单独前进。

## agilex56 框架对齐 · 2026-09-14

本站生产源码已与本地、`.1`、`.38` 的16模块版本逐文件对齐。本地及本站隔离 staging 各93项离线测试全部通过，无跳过。新版 client/phase 实际取得三路 RGBD 和完整双臂反馈，move_left/right/both 与双夹爪共5个入口仅做 dry run，CAN TX 计数不变；本轮没有实机运动复测。相机服务持续运行，原12个参数/实验 JSON 哈希不变。版本、原始日志、配置格式边界及备份见 [对齐记录](framework-versions.md)。

## houzhi1 拧盖技能与快照回放 · 2026-09-13

[双臂拧盖任务页](../skills/wcx-agilex-control/references/bottle-opening.md)已接入原 skill 路由，复用原工具与几何契约；skill-creator 格式检查通过。独立只读检查覆盖换现场、左右角色互换、倾斜瓶轴及历史第62阶段的未证实分离，未执行实机动作。检查后补充了关节插值段内接触中心/瓶轴偏离的核验要求；这是流程文档，尚无另一瓶型或现场的实机复测。

[本轮视频](../exports/2026-09-13/host-10.169.21.1/bottle_cap_video/README.md)复用原导出脚本，将66组完整观测按上前视、中左腕、下右腕排列；85.25秒、960×1800、H.264/24fps，另保留单路视频。四个MP4全量解码通过，起始、旋盖、盖口分离和最终帧用于画面复查；[验证记录](../exports/2026-09-13/host-10.169.21.1/bottle_cap_video/validation.json)保留尺寸、时长、源相机映射及逐帧索引。这是带原时间戳的等间隔快照回放，未生成未记录的运动，不能视作三相机硬同步或连续录像。

## 通用调用与几何：测试、部署与实机验证 · 2026-09-13

本地与 `.1` 远端 `staging-final` 的完整离线回归均为 **93 项通过，无跳过**，见 [本地最终日志](../evidence/2026-09-13/host-10.169.21.1/framework-efficiency-1727/local-tests-final.log) 与 [远端最终日志](../evidence/2026-09-13/host-10.169.21.1/framework-efficiency-1727/remote-tests-final.log)。覆盖原 primitive、阶段身份/观察重试/停止/崩溃、客户端传输与错误分离、实测摘要，以及几何的刚体方向、单位、深度、畸变、实体身份、同臂/跨臂、实际关节 FK、RPY、CLI 和输出不覆盖。测试使用硬件替身或纯数据计算，不访问 CAN 或相机。

共享 `src` 的16个文件已部署到 `.1`；原源码保留为远端 `source-before.tgz`，全部 houzhi1 配置及标定的 SHA-256 前后相同，见已取回的 [部署核验](../evidence/2026-09-13/host-10.169.21.1/framework-efficiency-1727/deployment.json)。[本轮源码校验表](../evidence/2026-09-13/host-10.169.21.1/framework-efficiency-1727/source-sha256-final.json)独立于各实体参数保存。

新 client 的一次实际 observe 已成功取得三路图像/元数据及双臂完整反馈：客户端总计0.671秒，远端调用0.6002秒，文件取回0.0709秒；远端 phase 内部用时0.5282秒。这是单次观察链路实测，不是后续时延保证。主 CLI、client 和 geometry 的本地 `--help` 也已通过，接口见 [调用契约](primitives.md)。新框架的叠杯动作与任务验收见下一节。

## houzhi1 通用框架叠杯实测 · 2026-09-13

`cup-nesting-framework-1757` 已将三只倒扣纸杯套成一摞，主动开爪撤离后75.08秒的新帧仍显示稳定。右臂执行、左臂保持观察；23个阶段全部 completed，共调用24条原执行 primitive（20条 move_right、4条 set_gripper_right），其中3个阶段各组合2条原 move。任务验收依据 [结构化结果](../evidence/2026-09-13/host-10.169.21.1/cup-nesting-framework-1757/outcome.json) 和 [稳定复查侧视图](../evidence/2026-09-13/host-10.169.21.1/cup-nesting-framework-1757/35-stability-check/observation-000/d435i_071316/rgb.jpg)，不只依据阶段状态。

本轮没有掉杯或重抓；第三杯第一次短试提的视觉不明确，口缘深度变化约8.2mm，再补10mm提起验证后搬运。不能排除夹持中的小幅就位滑动，两只被搬运的杯子均在主动开爪后依靠重力落稳。白色包边的局部圆弧拟合虽有低残差，转腕后的法向复查仍不一致，不能据此记为精确扶正或杯轴计量成功。外参保持 `accepted_for_control=false`，TCP 仍未计量验证。

| 本轮计时项 | 实测 |
|---|---|
| 首观察至开爪撤离 | 1311.14秒，约21分51秒 |
| 首观察至稳定复查 | 1386.22秒，约23分06秒 |
| client 阶段调用累计 | 208.43秒 |
| 三视图/元数据传回累计 | 1.7545秒；每次中位0.074秒，最大0.0913秒 |

图像、调用结果及 [完整阶段记录](../evidence/2026-09-13/host-10.169.21.1/cup-nesting-framework-1757/remote-records) 已在本地，原始对齐深度保留于对应远端阶段目录。全流程时长与客户端调用累计分开记录；场景和已积累经验也有变化，不能由两轮总时长估计框架的独立加速比例。

## 叠杯任务分层 · 2026-09-13

叠杯任务整理为原技能的 [cup-nesting.md](../skills/wcx-agilex-control/references/cup-nesting.md)，通用接触点/刚体换算保留在 local-geometry，未增加独立硬件技能。skill-creator 格式校验通过，修改的三个技能文件共24个本地引用均可解析；这是文档/路由验证，没有发出硬件调用。

此前 `.1` 三只倒扣纸杯最终成摞、撤离后复查稳定已有 [实物证据](../evidence/2026-09-13/host-10.169.21.1/cup-stacking-154501/README.md)。过程含试提失败、搬运滑落和末次自行滑入，不构成精准受控放置、TCP 或绝对几何验证。后续通用框架升级及再次实机执行分别记录，不能沿用此次历史结果作为新版本通过证明。

## Skill 收敛验证 · 2026-09-13

旧抬升规划器已归档，读取 CLI 委托共享 get_state/CAN 接收实现；初始化步骤只维护在 [initialization.md](../skills/wcx-agilex-control/references/initialization.md)。本次完整离线回归 **46 项通过，无跳过**，使用证据中保存的纯 FK 文件，无硬件访问；Ruff F、skill 格式和本地引用另行检查。原 45 项中退役 1 项旧规划器测试，增加 2 项读取回归；站点隔离测试改用明确 fixture，不再硬编码 .38 的早期部署状态。

在项目根目录、具有 NumPy/SciPy 的 Python 环境中运行：

```bash
PYTHONPATH=src AGILEX_FK_FILE="$PWD/evidence/2026-09-13/host-10.169.21.38/integration_1450/legacy-piper-fk.py" python -m unittest discover -s tests -v
```

`.38` 同步后的相关 37 项离线测试通过；运动学所需的 `.56` 测试夹具不在该站部署目录，由上述本地全量测试覆盖。

[改动、兼容性与验证清单](../evidence/2026-09-13/skill-consolidation/README.md) · [测试日志](../evidence/2026-09-13/skill-consolidation/local-tests.txt)。以下各节保留对应阶段的实测范围，不能作为当前设备状态。

## houzhi1 三相机标定 · 2026-09-13

沿用原运动/观察接口，每腕20组真实停稳采样（16训练+4留出），最终返回看板位置复测。三相机均有原始RGBD、逐帧参数、角点/PnP、实际关节/FK和遥测。左/右腕链留出平移RMS4.392/3.524mm，跨臂6.32mm、最大10.82mm；精确实体尺寸与绝对精度未验证，未启用为正式控制参数。[完整结果](../evidence/2026-09-13/host-10.169.21.1/three-camera-calibration-151012/README.md)。

共享 `calibration_board.py` 与 `calibration_hand_eye.py` 是离线模块，没有新增相机采集或机器人执行器。检测在OpenCV4.11/4.13验证自定义marker IDs、归一化射线真值、默认现场图像逐值回归；求解验证了三种算法真值/尺度、变换方向、留出隔离、退化、已知手眼复用、非有限输入及不覆盖旧输出。见 [检测证据](../evidence/2026-09-13/host-10.169.21.1/calibration-analysis/integrated-image-regression.json)、[自定义ID验证](../evidence/2026-09-13/host-10.169.21.1/calibration-analysis/custom-ids-test/verification.json) 和 [求解验证](../evidence/2026-09-13/host-10.169.21.1/calibration-analysis/synthetic-verification.json)。技能格式校验通过；这些验证不扩大实机精度结论。

## panfeng38 早期仅 RGBD 接入 · 2026-09-13

复用原有 cameras.py（哈希与 .1 部署一致），仅新增本站独立配置及 observe 中显式缺少臂反馈的支持。三路 RGBD 实测成功，见 [证据](../evidence/2026-09-13/host-10.169.21.38/observe_shared_rgbd_004/README.md)。新增 ROS 适配器和专用 launch 已删除。原生项目与 .56/.1 远端程序未修改。

该阶段本地回归共39项，33项通过、6项因缺少现场纯FK模型跳过。新增7项覆盖默认 RGBD+反馈、显式缺少臂反馈、错误不降级/不回退、证据不覆盖、主机身份及配置类型；不访问硬件。本站仅增项目隔离环境中的 RealSense 依赖，读取后相机服务已关闭，当时 CAN 保持 DOWN、TX=0。后续 CAN/使能、Piper X 几何及上抬实测见 [本站档案](../skills/wcx-agilex-control/references/site-panfeng38.md)，勿将早期结果当作当前状态。


## 早期多服务器技能与扩展入口 · 2026-09-13（后续部署见上）

本轮在本地共用技能中增加明确选站、两份现场档案和扩展/复用边界；没有修改服务器程序、启动设备或验证新动作。新增 [sites.py](../src/agilex_control/sites.py) 只解析本地档案，不负责硬件路由执行。

- `test_sites.py` 的5项离线测试通过：ID/IP一致、域用户名完整传递、未知/歧义目标拒绝、配置不继承及副本隔离、不修改选择器即可增加第三个现场。
- 在已有临时验证环境运行完整测试：共32项，26项通过，6项因本地缺少现场纯FK模型跳过。首次使用缺依赖的系统/打包Python未能加载SciPy/PyYAML，改用已有验证环境后通过；未修改机器人服务器环境。
- skill格式校验、修改代码的Ruff E9/F检查通过；站点CLI的list/show与缺失/未知目标退出行为通过。
- `agilex56` 的 `config/site.json`、已有标定和原有控制接口保持原样；`.38` 的 `primitive_deployment`、`calibration` 保持 `null`，不填假设参数。
- 本次结论是技能路由/文档与离线选择器有效，不代表 `.38` 已能执行共享primitive，也不代表已适配其他型号或任意执行臂数量。

## agilex56 历史验证 · 2026-09-12

| 层级 | 结果 | 范围与限制 |
|---|---|---|
| 现场相机只读 | 通过 | 现有 5555 服务收到三路 640×480 图像并保存；接收后退出，未连接动作端口 |
| 相机脚本本地回归 | 通过 | NumPy pickle 协议 4/5；拒绝非允许全局项；拒绝 5556/5557；只有一次 SUB/recv；超时清理 |
| 状态解码 / 规划输入 | 通过 | 验证 `0x13` 通信位、过期反馈、故障及越界输入拒绝；没有访问硬件 |
| 技能格式与文件完整性 | 通过 | `quick_validate.py` 通过；Markdown 本地链接、Python 语法、AGENTS 与个人技能符号链接通过 |
| 当前 5556 接口能力检查 | 只读核验通过 | 服务端监听、动作路由、单位及校准分支已核对；未连接动作端口，见 [检查记录](../skills/wcx-agilex-control/references/capability-check.md) |
| 初次 5556 右臂 10 cm 试验 | **未通过，已停止** | 99 次 ZMQ 发送，98 套 CAN 本机回显；起步跟踪偏差中止，无通信错误，见 [分析记录](../skills/wcx-agilex-control/references/right-lift-20260912.md) |
| 离线抬升 | 十段完整 FK/IK 通过 | 当时的独立规划脚本（现已 [归档](../evidence/2026-09-13/skill-consolidation/README.md)）曾在 pi0 环境生成十段各 10 mm 的完整计划，并离线回放试验目标/反馈；这不代表实际抬升通过 |
| 右臂抬升重试 | **约 10 cm 已完成** | 实测 +97.663 mm，结束 XYZ 稳定，腕部角度微漂移边界见 [重试记录](../skills/wcx-agilex-control/references/right-lift-endpoint-20260912.md) |
| 独立工具观察与右臂上抬 | **通过** | 三路RGBD；右+17.896mm、211套目标，左和夹爪未变；夹爪接管前已在桶口附近 |
| 左臂独立工具 | **实机通过** | 插花任务完成移动、夹持、翻转、插入与退出；与右臂各完成一枝 |
| 双臂插花 | **完成** | 两枝由花瓶支撑，双臂退出；包含滑茎重抓及接触纠正，见 [记录](flower-arrangement-20260912.md) |
| 原始旋转 FK/IK | **21项远端离线回归通过** | 包括新增6项运动学回归；实际 SDK DH 模型，近 pitch ±90° 和普通姿态；[日志](../evidence/2026-09-12/flower_arrangement/kinematics_regression.txt) |
| 独立 D455 RGBD | 单帧获取通过 | 原服务退出、设备无占用后采集并释放；不代表 5555 提供深度 |

早期版本的 6 项本地回归覆盖只读相机、状态及旧规划器；当前测试位于 [test_readonly_tools.py](../tests/test_readonly_tools.py)，依赖 NumPy 与 SciPy，不打开网络或 CAN。运行方式：

```bash
python -m unittest discover -s wcx_gpt6_astra_agilex/tests -v
```

本轮使用临时虚拟环境 `/tmp/agilex-skill-validation-20260912`，未修改服务器环境。相机脚本在现场首次成功执行后仅调整了 NumPy 兼容命名空间的选取方式，改动由本地反序列化测试覆盖；没有为验证该改动再次订阅现场图像。

技能源码保存在项目中；`~/.codex/skills/wcx-agilex-control` 指向它。格式验证与测试结果不提供实机执行保证。

后续 10 cm 试验前，还在本地检查了临时执行器的动作字段、有限数值约束、分段连续性、保持时序和状态中止条件；实测暴露出保持漂移判据不足。失败执行器已从实验入口撤下，仅保留 `.py.txt` 证据，没有部署为技能执行脚本。

成功重试另做了执行器的端点、右关节字段和实际故障检测离线冒烟检查，并有两阶段实机记录；历史失败执行器仍不作默认入口。

积木任务阶段统一实现共有15项本地回归通过，覆盖只读协议、primitive 路由/终点/部分发送失败/并发隔离/固定臂/相机锁，以及夹爪独立帧、接触时非零开度、带符号力矩和 0xC0 状态。前视 1280×720 与双腕 640×480 的配置已通过实机 RGBD 观测验证。规范见[工具契约](primitives.md)，抓取放置的实物证据见[本轮记录](pick-place-20260912.md)。技能格式、源码语法和本地链接另行核验。

插花阶段修复共享运动学，增加 [test_kinematics.py](../tests/test_kinematics.py)，在服务器 evorl-ljy 环境的实际 SDK FK 文件下21项离线测试通过。新测试不访问 CAN。修改后的两文件 Ruff E9/F 通过；格式化后重新同步源码，不更改逻辑。花茎滑转与 SDK Euler 截断分别记录，不能将数学修复当作消除物理滑移的证明。
