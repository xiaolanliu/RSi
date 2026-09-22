# Agent-in-loop 仿真链路

默认由 pi05 执行动作。每完成一个原生控制步，RSI 读取真实观测；报警后清空
VLA 尚未执行的 chunk，向恢复策略提供成功示范、任务、当前三视角、近三秒
观测及三项 OOD 证据。恢复策略只输出一段短动作，执行结束即从最新观测
重新调用 pi05。当前配置 `recovery_mode="mock"`，不调用 GPT。

## 目录与调用关系

```text
rsi_loop/
  cli.py             配置、资源、独立 worker 启动与操作入口
  controller.py      单一动作所有者、chunk 中断、恢复和交回 VLA
  simulator.py       RoboDojo 原生观测、逐步动作、成功判定、录像
  vla.py             官方 pi05_base 的 JAX 推理与绝对目标后处理
  monitor.py         因果 RGB/Wan48、量纲转换、25→30 Hz 时钟适配
  recovery.py        上下文、Responses 请求、结果/动作校验、mock
  demonstration.py   GPT-Policy 抽帧、示范缓存、成功轨迹筛选
  kinematics.py      机器人本体 FK、IK 与动作速度界限
  contracts.py       state/action/observation/恢复计划的显式接口
  workers.py         同一个 Python 环境下的进程隔离与应答检查
configs/loop.toml               闭环参数；默认禁止 GPT 调用
configs/sources.lock.json       第三方源码的固定提交
configs/sim_normalization.json  仿真归一化来源与 SHA256
resources.local.json           本机资源路径，忽略 Git
outputs/<run>/                 逐步证据、完整 action chunk、视频和结果
```

模型仅加载一次。RSI 的 `OnlineMonitor` 不再通过训练脚本加载权重。
OpenPI 使用推理模块组合相同 transforms，不导入 LeRobot 数据集或训练管线。
JAX VLA 与 IsaacSim 分进程，在同一个 conda 环境内运行，避免 Kit 的依赖搜索
路径污染 VLA。默认 GPU 0 放 pi05，GPU 1 放仿真与 Wan，RSI 小头在 CPU。

## 安装与资源准备

建议 Linux、NVIDIA GPU；当前机器有两张 24 GB 显卡。源码和权重都较大，
额外预留约 80 GB 以上环境/缓存空间；仿真场景资产另计。需要系统
`git`、`ffmpeg`、`ffprobe`、可用的 NVIDIA Vulkan ICD。

```bash
conda create -n rsi python=3.11 pip -y
conda activate rsi
python tools/setup_sources.py --assets /absolute/path/to/RoboDojo/Assets
cp configs/resources.example.json resources.local.json
# 编辑 resources.local.json，填写真实资源位置。
bash tools/install_rsi.sh
rsi-loop doctor
rsi-loop doctor --hashes  # 一次性读取全部 pi05 参数，核对原始权重身份。
```

`tools/install_rsi.sh` 默认安装到 `$HOME/.conda/envs/rsi`；若上面的 conda
选择其他前缀，执行前设置 `RSI_CONDA_PREFIX="$CONDA_PREFIX"`。
使用镜像可设置 `RSI_PYPI_INDEX`。脚本不管理或读取 API key。
所有后续命令在仓库根目录、已激活的 `rsi` 环境执行。

外部资源：

* **pi05_base**：`gs://openpi-assets/checkpoints/pi05_base`，必须保留完整
  `params/` Orbax 结构；不能只取一个 `ocdbt` 文件。原始 base 权重不是
  RoboDojo 60,000 步微调 checkpoint，不能宣称达到后者的基准成功率。
  `configs/pi05_base_identity.json` 保存已对照官方对象 CRC32C/MD5 校验的
  20 个参数文件的 SHA256；`doctor --hashes` 可在离线机器重新核对。
* **RoboDojo**：固定源码与原生场景、Eval_Layout、机器人 USD/URDF、材质。
  资产来源为 [官方数据仓库](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo)。
  可以链接完整已下载的 `Assets`。首次新机器安装按固定版本 RoboDojo 的
  README 下载资产；不要给缺失对象塞占位物体。
* **Wan VAE**：与 [USAGE.md](USAGE.md) 相同的 Wan2.2 权重和哈希。
* **tokenizer**：OpenPI 缓存中的 `big_vision/paligemma_tokenizer.model`；首次可由
  OpenPI 下载。`openpi_cache` 是缓存根目录。
* **GPT-Policy**：固定提交的 `FfmpegVideoExtractor`，直接用于带 PTS 的视频
  抽帧。这里没有调用其 Codex/Claude 登录流程或硬件驱动。
* **Kit 扩展**：离线机器可配置 `nvidia_extensions`；普通联网安装可省略该键。

统一依赖约束见 `requirements-simulation.txt` 与 `requirements-integration.txt`。
`requirements-rsi.lock.txt` 记录这台 Linux/Python 3.11 机器实测的完整版本，
安装脚本以它作为约束；源码 editable 包使用 `configs/sources.lock.json` 固定。
IsaacSim 固定 torch 2.7.0 / numpy 1.26.0；所选 IsaacLab 原来固定 Starlette
0.49.1，与 IsaacSim 的 FastAPI 0.115.7 冲突。源码安装脚本只把 IsaacLab
这条约束改为 `>=0.40,<0.46`，不修改控制、动力学或成功条件。
最后必须运行 `uv pip check --python "$CONDA_PREFIX/bin/python"`。

## 运行与产物

```bash
# 原生仿真 + 真实 pi05/Wan/RSI；报警后 mock 持位三步再交回 VLA。
rsi-loop evaluate --config configs/loop.toml --output outputs/integration_01 --max-steps 150

# pi05 独立执行，仍记录 RSI，但禁用接管。用于采集候选成功 demo。
rsi-loop evaluate --config configs/loop.toml --mode disabled --output outputs/vla_candidate_01

# 只有完整、原生判定成功、没有接管的轨迹才能自动成为成功示范。
rsi-loop promote-demo --run outputs/vla_candidate_01 --output outputs/demos/task_01

# 查看将发送给 GPT 的实际请求；只写文件，API 调用数为零。
rsi-loop prepare-context --run outputs/integration_01 --step 50 \
  --output outputs/context/request.json
# 如已有早期成功示教视频，加 --demo /absolute/path/to/success.mp4

# 独立测试真实仿真交接：在第 32 步注入一次测试报警，原始分数仍保存。
python tools/check_sim_handoff.py --output outputs/native_handoff_test

# 为已录制回合生成本地网页：三路视频、三项分数、关节与真实夹爪曲线。
python tools/report_sim_run.py --run outputs/vla_candidate_01
# 打开 outputs/vla_candidate_01/index.html
```

修改 `configs/loop.toml` 的 `task`、`eval_seed`（布局组）和 `layout_id`（组内
布局）。每个 worker 只执行一个新 episode，不回滚仿真；下一条轨迹创建新
输出目录。`max_steps` 小于原生时限时属于预算截断，不能记为完整失败评测。

每次输出包含：

| 文件 | 用途 |
|---|---|
| `events.jsonl` | 已确认执行的控制步：来源、命令、风险、交接、丢弃动作数、各环节耗时 |
| `commands_requested.jsonl` | 执行前请求；缺少对应 ACK 的末行不能当成已执行动作 |
| `loop_summary.json` | VLA 次数、接管次数、各来源执行步数、结束原因 |
| `native_outcome.json` | 原生成功/结束、是否可计入成功率、原生控制时限 |
| `sensors.mp4` | 三相机拼接录像；每个原生动作 ACK 后记录一帧 |
| `observations/*.npz` | 原生 RGB、机器人 state、末端姿态和时间 |
| `vla/chunk_*.npz` | 模型原输出和夹爪饱和后的绝对动作；不会再次叠加 state |
| `vla/vla_metadata.json` | 原始 base 路径、归一化身份、动作语义 |
| `native_metadata.json` | 本体、dt、注册后的原生成功条件 |
| `*.log` | 各 worker 的完整错误与初始化记录 |

`mock` 只测试控制交接，不能证明能修复掉落/抓取失败。没有成功 demo 时，
配置中保留空入口；不能用失败轨迹、预算截断或 mock 轨迹冒充成功示教。
`check_sim_handoff.py` 使用真实模型、真实仿真与动作 ACK，但覆盖报警位，
保留 `natural_alarm` 和三项原始分数；它只验收中断/交回与新观测推理。
`control_test.json` 明确标记测试来源，示范筛选器拒绝这类轨迹。

## 动作与报警的严格含义

仿真观测/action 布局均为 `L6,gL,R6,gR`，关节弧度、夹爪归一化开度。
RoboDojo 原生 `state` 的夹爪分量来自上一条控制目标；pi05 保持这个训练接口。
适配器另外读取物理夹爪关节，保存 `measured_gripper_openings`；RSI 使用这个
实测值换算行程，恢复上下文同时提供目标与实测开度。不能用目标闭合判断
夹爪已经闭合。渲染使用上游建议的零延迟同步设置，再获取该步相机观测。
pi05 内部对十二个关节预测相对当前 state 的增量，夹爪是绝对值。
`Unnormalize → AbsoluteActions → AlohaOutputs(adapt_to_pi=False)` 后返回
实际绝对目标；执行端不能再加一次 state。

默认对原始 base 使用官方 `arx_x5_sim` 正常示范统计适配仿真量纲。
这是明确的跨域实验配置：它没有对 base 权重进行仿真微调，不等同于
原始 base 的某项已验证部署，也不等同于 RoboDojo 榜单 checkpoint。
原 base 的 `arx` 夹爪 state/action 数值范围不同，直接把它们当成同一
0～1 开度会产生错误；因此两种归一化来源不能静默互换。

RSI 仍使用发布的三项判据和正常校准，无新增第四项判据或失败标签调参。
25 Hz 原生仿真没有改速；监测接口按时间戳用过去观测做零阶保持，生成
30 Hz 的因果 RGB/state 流。任何生成时刻都不读取未来观测。
双指行程映射 `2*(0.044-(-0.010))=0.108 m` 来自仿真 X5 配置，是相对行程
近似，不是已经实测对齐的真实夹爪接触面宽度。

**这些适配并不能恢复实机折衣数据上的 OOD 校准保证。** 仿真相机、物体、
本体和任务均有分布变化，可能从任务早期就报警。应另外收集正常成功
仿真轨迹，按 episode 分开正常训练/校准/验证，再建立仿真专用参考分布；
不能把当前失败 rollout 当成正常数据在线吸收。冻结头与新校准版本须
分别保存。当前链路测试和异常检测准确率评测是两个验收项目。

## GPT 恢复协议与下一步启用

成功 demo 最多抽取八帧，保留真实视频时间戳；默认均匀抽样，不产生额外
LLM 选帧费用。上下文另附当前三相机、近三秒最多三张顶视历史图、机器人
state、末端位姿、任务指令和 OOD 三项证据。视频中的示范是参考，不是
当前环境或未来的视觉观测。

恢复 JSON 包含诊断、`recover/unable`、持续步数和两臂相对末端位移/旋转、
目标夹爪开度。上限为 25 个原生控制步、每臂 5 cm / 0.35 rad；解析成功后
做本体 FK 与仿真机器人末端观测核对、逐点 IK、限位和 1.5 rad/s 指令变化
率检查。无法求解或超限则结束该次评测，不执行部分未知动作。
这些是恢复指令检查，不是绕障或真机安全认证。

执行恢复时仍记录 RSI，但不递归请求 GPT。恢复完成后重新获取 pi05 chunk。
默认冷却 25 步且连续五步风险解除才重新武装，持续同一次报警只请求一次。
超时/拒答/畸形 JSON/陈旧 episode-step 均停止本次运行；请求和动作不自动
重试。仿真在等待推理时不推进物理时间，wall-clock 延迟单独记录；这不是
硬件异步控制器。

后续用户授权真实 GPT 测试后，才执行以下配置：

1. 将成功示教路径写入 `[gpt].demo_mp4`，可用 `system_prompt_file` 覆盖任务
   SYSTEM_PROMPT，选择支持图像和 structured outputs 的 `model`。
2. 设置专用 `RSI_SIM_OPENAI_API_KEY` 环境变量，不放进仓库/输出/命令日志，
   不复用 Codex/IDE 登录凭据。
3. `[gpt].enabled=true`，运行 `--mode live --allow-live-gpt`。两个显式开关
   必须同时满足；mock 和 prepare-context 不会访问 API。

请求使用 [Responses 图像输入](https://developers.openai.com/api/docs/guides/images-vision)
与 [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)，
`store=false`，只读专用 key，关闭自动重试。当前阶段没有进行真实 GPT 调用。

## 验证与后续修改

```bash
python tools/run_unit_tests.py
python -m pytest -q tests/test_control_loop.py
rsi-verify --device cpu --output outputs/verification_conda_cpu.json
rsi-verify --device cuda:0 --output outputs/verification_conda_cuda.json
```

控制器测试验证 chunk 中途报警、过期动作清理、交回后新观测、持续报警限次、
清除后重新触发、超时/陈旧响应、动作校验、时钟因果性、失败视频拒绝晋升。
HTTP 传输测试仅向本机测试服务器发送请求，无外部 GPT 调用。
源码调整后更新 `artifacts.json` 的源码清单；三份权重和独立 goldens 不得
为了通过测试而重写。原离线教师仍可按 `docs/REPRODUCE.md` 复现。

## 本机验证记录（2026-09-22）

Python 3.11.14 的 `rsi` 环境通过依赖检查。36 项模型机制测试、15 项闭环
测试通过；全部 13,548 帧在 CPU 与 CUDA 上分别通过独立历史结果校验，
没有修改权重、golden、报警阈值或比较容差。

原始 pi05 的 20 个参数文件通过官方对象校验；真实三相机、pi05/Wan/RSI、
原生逐步执行、动作 chunk、物理夹爪记录和 MP4 已贯通。GPT-Policy 抽帧器
从实际录制视频读取了八个真实 PTS 帧；该接口测试视频不是成功示范。
本体 FK 在初始与运动后的姿态均与原生末端观测对齐；1 mm/25 步的 IK
接口测试通过，但它不是经 GPT 决策或实际执行的恢复效果评估。

`outputs/vla_candidate_02` 为原始 base 的首个完整搭塔回合：1,050 个控制步、
105 次 VLA 推理、原生任务失败、自然报警 0 次；风险分位数最高约 0.145。
这条轨迹被成功示范筛选器拒绝。不能把“通路能运行”解读为 base 的仿真
操作能力或冻结 RSI 对新任务的检测能力已达标。后续成功示范和正常校准
应单独收集；GPT 的真实恢复效果仍须在用户授权调用后验证。

`outputs/native_handoff_test` 完成 60 步原生交接验收：第 32 步测试报警清空
八条未执行的 VLA 动作，mock 持位三步，第 35 步用新观测重新调用 pi05。
真实推理时刻为 `0,10,20,30,35,45,55`，动作 ACK 连续且逐步对应。
这个测试保留了自然报警位，测试结果 `passed=true` 只代表控制交接通过。
两个目录内的 `index.html` 可直接查看视频与曲线。真实 GPT API 调用数为零。

`outputs/fold_candidate_01` 为原始 base 的完整折衣回合：500 步、50 次 VLA
推理、原生任务失败、自然报警 0 次，最高风险分位数约 0.496。该回合也不
进入示范库，网页为同目录 `index.html`。复现时将配置 `task` 改为
`fold_clothes`，其余仍为布局组 0、布局 0。

可移植的完整验证摘要见 [integration_20260922.json](../provenance/integration_20260922.json)。
这两个单布局实验不是总体成功率评测；目前收集到的合格成功示范数量为零。
