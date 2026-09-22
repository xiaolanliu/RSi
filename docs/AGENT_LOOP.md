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
  context.py         在线/离线共用的因果历史采样与观测读取
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
rsi-loop doctor --hashes  # 核对原始 pi05 参数，以及已配置的可选采集权重。
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

# 完整本地协议测试：视频抽帧 → HTTP 替身 → 结构化回复 → IK → 原生执行。
# 该视频只作为协议测试材料，不声明其任务成功。
python tools/check_sim_handoff.py --output outputs/structured_recovery_test \
  --video-fixture outputs/native_handoff_test/sensors.mp4

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
每次请求之前保存不含凭据的 `recovery/request_<step>.json`，失败/超时也保留。
`prepare-context` 优先原样导出该快照；没有快照时，用同一历史采样与缩图
函数重建上下文，并读取运行配置中的示范路径与 SYSTEM_PROMPT。
历史严格限制在当前时刻之前三秒，并记录实测夹爪开度。

## 项目内 HahaModel 中转

`configs/fold_clothes_haha.toml` 使用项目自己的 provider 配置。它由
`rsi_loop.providers` 读取，不会写入 `~/.codex/config.toml`，也不修改 IDE
登录、本机转发服务或全局代理环境变量。

```toml
model_provider = "haha"

[model_providers.haha]
name = "HahaModel"
base_url = "https://hahamodel.com/v1"
wire_api = "responses"
env_key = "HAHA_API_KEY"
requires_openai_auth = false
use_environment_proxy = false
# proxy_url = "http://127.0.0.1:YOUR_PORT"
```

此处 `requires_openai_auth=false` 表示不读取 OpenAI/Codex 登录凭据；HTTP
请求仍使用指定 `HAHA_API_KEY` 的 Bearer 认证。密钥取自同名环境变量，
或显式配置的 `gpt.credential_file=".env.local"` 中的同名条目。文件必须
为 0600 权限且已被 Git 忽略；文件内容不会导入全局进程环境，也不会写入
请求快照。不要把真实密钥写入 TOML、脚本、提交说明或命令行参数。

复制模板为 `configs/fold_clothes_live.local.toml`，填写该中转实际支持的
视觉/Responses 模型，设置 `[gpt].enabled=true`。模板限制每回合最多一个
恢复请求和一个干预，默认最多输出 1600 tokens。网络错误、超时、HTTP
错误均不自动重试。`proxy_url` 仅作用于项目 HTTP 客户端；未配置时，
`use_environment_proxy=false` 使用项目内直连。

```bash
# 自然 OOD 路径：只有冻结检测器真正报警时才请求 GPT。
rsi-loop evaluate --config configs/fold_clothes_live.local.toml \
  --allow-live-gpt --output outputs/live_natural_01

# 单独验证真实 GPT 接管：第 175 步增加一次明确标记的测试触发。
# 保留 natural_alarm 与原始三项证据，不修改模型阈值。
python tools/check_live_handoff.py --config configs/fold_clothes_live.local.toml \
  --at-step 175 --allow-live-gpt --output outputs/live_handoff_01
```

第二条命令会产生真实 API 费用，但它只测试诊断、动作执行和交回 VLA，
不能用来证明自然 OOD 检出率。此类轨迹带有 `control_test.json`，不得晋升
为 VLA-only 成功示范。每次请求保存不含认证信息的 `api_request_*`、
`api_response_*` 和 `api_attempt_*`，记录模型、耗时、usage 与 HTTP 状态；
即使动作因越界、无法求解 IK 或 `unable` 被拒绝，模型回复仍保留。
Responses 接口依据 [OpenAI Docs 的结构化输出](https://developers.openai.com/api/docs/guides/structured-outputs)
和 [多图像输入](https://developers.openai.com/api/docs/guides/images-vision) 组织；
第三方中转的可用模型、兼容性和收费必须以其实际响应为准。

可离线查看实际多图像请求：

```bash
python tools/report_context.py --request outputs/live_handoff_01/recovery/api_request_000001.json \
  --output outputs/live_handoff_01/request.html
```

2026-09-22 本机当前验收：27 项本地测试通过；原始 base 完整运行 500 步、
50 次推理，原生任务失败、自然 OOD 报警 0 次，最大风险分位数约 0.225。
两条成功采集轨迹的最大风险分位数约 0.609、0.551；这些结果说明原始
实机校准分数尚不能直接用于该仿真任务，不能通过观察失败后降低阈值来
声称检测改善。

中转域名从本机直连超时；现有 18080 端口拒绝 CONNECT，未改动该服务。
使用真实示范与现场观测尝试过一次 Responses 请求，90 秒后连接失败，
没有模型回复，也没有 GPT 动作执行，模型可用性尚未核实。
凭据不在仓库中。完整实测结果见
[provider_validation_20260922.json](../provenance/provider_validation_20260922.json)。
实际 GPT 接管验收需要先提供本机可用的项目级网络路径。

## 独立采集仿真成功示范

默认 `configs/loop.toml` 继续使用指定的原始 `pi05_base`。由于原始 base 在
当前搭塔/折衣回合均未成功，另外提供 `configs/collect_demo.toml`，仅用于
以 RoboDojo 官方 59,999 步微调 pi05 自行 rollout 收集示范。它不会替换
主闭环权重，也不能启用恢复接管。两种 checkpoint 的身份与用途都写入
`vla_metadata.json`；自动晋升的示范会保存模型来源。
对应的官方配置为
[`pi05_base_aloha_full_sim_arx-x5_seed_0`](https://github.com/XPolicyLab/XPolicyLab/blob/a768326d05f68d421542636e6467bea7ef6bbe42/policy/Pi_05/openpi/src/openpi/training/config.py)：
从原始 base 初始化、训练 60,000 步；部署读取编号 59,999 的 checkpoint。

```bash
python tools/fetch_checkpoint.py --kind demo \
  --output external/checkpoints/robodojo_pi05_sim_59999 --workers 4 --segments 4
# resources.local.json 增加：
# "pi05_demo": "external/checkpoints/robodojo_pi05_sim_59999"
rsi-loop evaluate --config configs/collect_demo.toml --output outputs/demo_candidate_01
rsi-loop promote-demo --run outputs/demo_candidate_01 --output outputs/demos/fold_clothes_01
```

下载清单在 `configs/pi05_demo_identity.json`，约 12.44 GB，只取推理参数与
归一化，不取训练优化器。每个文件核对公开仓库给出的 SHA256，支持 `.part`
断点续传；`--segments 4` 并发读取精确字节范围，完整文件仍必须通过 SHA256。
禁止覆盖身份不符的现有 checkpoint。主模型也可使用
`fetch_checkpoint.py --kind base --output /your/pi05_base` 下载其已固定的参数；
完整仿真仍使用 `setup_sources.py` 下载的 `arx_x5_sim` 归一化。

## 验证与后续修改

```bash
python tools/run_unit_tests.py
python -m pytest -q tests/test_control_loop.py tests/test_checkpoint_download.py tests/test_provider.py
rsi-verify --device cpu --output outputs/verification_conda_cpu.json
rsi-verify --device cuda:0 --output outputs/verification_conda_cuda.json
```

控制器测试验证 chunk 中途报警、过期动作清理、交回后新观测、持续报警限次、
清除后重新触发、超时/陈旧响应、动作校验、时钟因果性、失败视频拒绝晋升。
HTTP 传输测试仅向本机测试服务器发送请求，无外部 GPT 调用。
源码调整后更新 `artifacts.json` 的源码清单；三份权重和独立 goldens 不得
为了通过测试而重写。原离线教师仍可按 `docs/REPRODUCE.md` 复现。

## 本机验证记录（2026-09-22）

Python 3.11.14 的 `rsi` 环境通过依赖检查。36 项模型机制测试、19 项闭环
测试通过；全部 13,548 帧在 CPU 与 CUDA 上分别通过独立历史结果校验，
没有修改权重、golden、报警阈值或比较容差。
另外两项本地 HTTP 下载测试验证中断后字节续传，以及拒绝错误 Content-Range。

原始 pi05 的 20 个参数文件通过官方对象校验；真实三相机、pi05/Wan/RSI、
原生逐步执行、动作 chunk、物理夹爪记录和 MP4 已贯通。GPT-Policy 抽帧器
从实际录制视频读取了八个真实 PTS 帧；该接口测试视频不是成功示范。
本体 FK 在初始与运动后的姿态均与原生末端观测对齐；独立的
`outputs/structured_recovery_test` 还完成了视频抽帧、14 张上下文图像、
本地 HTTP 结构化回复、IK、两臂 1 mm 上移的十步原生执行。第 32 步触发
测试报警，第 42 步交回 VLA 并重新推理。保存的请求快照与 HTTP 实际
收到的内容一致。该回复来自本地固定替身，只验证协议与执行，不代表 GPT
恢复能力；所用视频明确标记为测试视频，不能进入成功示范库。
两臂实际末端竖直位移均约 0.987 mm，完整记录见
[context_handoff_20260922.json](../provenance/context_handoff_20260922.json)。

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
这两个原始 base 单布局实验不是总体成功率评测，也没有生成合格成功示范。

独立采集模型在 `outputs/tuned_demo_candidate_01` 完成折衣任务：布局组 0、
布局 0，286 个原生控制步（11.44 秒），29 次 pi05 推理，原生成功；
全部动作来自 VLA，恢复接管和外部 GPT 调用均为零。视频已晋升至
`outputs/demos/fold_clothes_01/success.mp4`，同目录保留模型来源与成功结果。
这条视频是仿真判据下的成功参考，最终衣物仍有明显褶皱，不等同于用户的
早期人工成功示教，也不是原始 base 的成功记录。

已用该视频加原始 base 的折衣失败现场验证上下文：8 张成功示范帧、
3 张因果历史帧、当前三视角，共 14 张图像，未发送到 GPT。
记录见 [sim_demonstration_20260922.json](../provenance/sim_demonstration_20260922.json)。

```bash
python tools/report_sim_run.py --run outputs/tuned_demo_candidate_01
rsi-loop prepare-context --run outputs/fold_candidate_01 --step 175 \
  --demo outputs/demos/fold_clothes_01/success.mp4 \
  --output outputs/context/fold_failure_with_success_demo.json
```

上述两个运行目录是本机实测输出；新机器先执行对应 rollout 再使用其路径。
第 175 步只用于检查请求格式，未被当成故障起点标签或报警校准依据。

仓库附有这条约 3.1 MB 的 [示范视频及来源](../examples/demonstrations/README.md)，
克隆后可以直接用于参考输入。另一个布局的
`outputs/tuned_demo_candidate_02` 也在 315 步（12.60 秒）、32 次 VLA 推理后
通过原生成功条件，没有接管。最终关键帧中袖子与下摆向内折叠，较示范 01
平整，因此 `configs/fold_clothes_context.toml` 默认引用
`examples/demonstrations/sim_fold_clothes_02/success.mp4`。
第二条记录见 [sim_demonstration_layout1_20260922.json](../provenance/sim_demonstration_layout1_20260922.json)。
复现第二条时，复制 `configs/collect_demo.toml` 为本机配置，仅将
`layout_id` 改成 `1`，再通过 `evaluate --config` 指定该文件并使用新输出目录。
两次采集成功不代表总体成功率，冻结 RSI 的仿真检测能力仍需独立评估与校准。

该参考配置的任务设为折衣、控制模型为原始 base，GPT 仍默认关闭。用户提供早期人工
成功示教后，只需替换 `gpt.demo_mp4` 与可选 `system_prompt_file`。
