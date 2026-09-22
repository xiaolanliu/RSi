# 移植到 4090

计算放在 4090 上：pi0.5、Wan2.2 VAE、RSi 监测。机械臂仍由 Piper 侧执行，单位不要在这边重算第二遍。本机 8GB 4060 不再跑完整策略。

移植后先跑文末的链路检查。它会加载真正的 `fold_all.pt`，在官方失败样例上走到报警，再进入 GPT 接管并交还 pi0.5。这一步不连机器人，也不调用外网。

## 链路

每一拍安静时走本地策略；监测始终在旁边看。报警只在 RSi 自己的校准尾部成立时出现，然后 GPT 只做一段短恢复。

1. **pi0.5**。`ChunkedPi05` 拿着当前 action chunk，用完才向 `127.0.0.1:8088` 要下一块。关节弧度、夹爪米，和 OpenPI 请求一致。GPT 交还后丢掉旧 chunk，下一拍按恢复后的现场重新推理。
2. **RSi 在线监测**。`OnlineMonitor` 每拍读 `state14 + visual48`。`state14` 是左 1–6、右 1–6、左夹爪、右夹爪，关节弧度、夹爪米。`visual48` 来自官方 Wan2.2 VAE：RGB、letterbox 256、当前帧加前 4 帧、平均成 48 维。布局必须是 `joints12_grippers2`。
3. **安全分位报警**。V11 权重在正常校准轨迹的尾部秩 `tail <= alarm_tail_fraction`（此包为 0.05）时把 `alarm` 设为真。`risk_percentile = 1 - tail`。本循环不再设第二道阈值，直接使用这个 `alarm`。
4. **GPT 接管**。报警且冷却为 0 时，`GptTakeover` 把示范 MP4 和报警记录交给 `decide`。模型只能返回 1–2 步、`hand_back: true` 的 phase，目标用 `joint_deg`、`xyz_mm` 或 `gripper_mm`。然后主导权回到 pi0.5，监测 `reset()`，默认冷却 30 帧，避免同一报警立刻再抢走控制。

冷却期间监测仍在步进，但不会再次接管。GPT 不收到 RSi 的 14 维弧度向量，避免把它当成关节目标。

现场一拍的观测这样组：

```python
from agilex_control.rsi_observe import rsi_observation

observation = rsi_observation(robot_state_deg_mm, rgb_uint8, wan_encoder)
observation["frame"] = frame_index
observation["image"] = "当前主相机画面路径"
event = loop.tick(observation)
```

`robot_state_deg_mm` 用 PiperAdapter 已经换成度/毫米的 `RobotState`。夹爪不要再做 `deg2rad`。官方 npz 里的 `states` 已经是弧度/米，回放时直接送进 `tick`，不要再乘一次。

GPT 返回的 phase 交给现有的 `agilex_control` phase 执行。这个循环只决定谁拥有这一拍，不直接写 CAN。

## 要拷走的东西

| 内容 | 位置 | 说明 |
| --- | --- | --- |
| 本分支 | `real_world` | 控制代码在 `real_world/`，RSi 本体在仓库根目录 |
| RSi 权重 | `models/v11/fold_all.pt` | 约 9.3 MB，监测头 |
| Wan2.2 VAE | `weights/Wan2.2_VAE.pth` | 2.7 GB，git 忽略，必须单独拷到仓库根的 `weights/` |
| pi0.5 | `clothesfoldingv2` 与 checkpoint | 服务仍由 `openpi` 环境在 8088 提供 |
| 示范视频 | 任务自己的 MP4 | 报警时给 GPT 的上文，不是 RSi 的输入 |

VAE 官方文件来自 `Wan-AI/Wan2.2-TI2V-5B` 的 `Wan2.2_VAE.pth`。SHA256 必须是：

```text
20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36
```

```bash
sha256sum weights/Wan2.2_VAE.pth
```

不要换成别的 Wan 版本。RSi 的视觉特征只认这一份。

## 4090 环境

RSi 发布说明的 CUDA 基线是 Python 3.10、PyTorch 2.8.0+cu128、驱动 550 或更新。4090 按这个装。本机 4060 的驱动是 535，只能用 cu121，不要把那个环境原样拷过去。

两个环境分开，不要把 JAX / OpenPI 装进 `rsi`：

```bash
conda create -n rsi python=3.10 -y
conda activate rsi
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e . --no-deps
python -m pip install "av>=13,<14" "Pillow>=10" "einops>=0.8,<0.9" numpy
python -m pip install -e real_world --no-deps
```

pi0.5 继续用原来的 `openpi` 环境，按 `clothesfoldingv2/deploy/README.md` 启动。机器人侧仍是 `evo-rl`。

4090 是 24 GB，但 `serve_local_policy.py` 会把 JAX 预分配夹在显存的 85%–90%。24 GB 的 88% 大约 21 GB，剩下的空当小于 Wan fp32 编码测到的 3091 MiB。移植时在 4090 那份服务脚本里把比例改成大约 **0.50**（约 12 GB 给 pi0.5），并去掉 0.85–0.90 的强制夹紧，给 VAE 留出 4 GB 以上。监测头只有 9 MB，放 CPU 即可。

```bash
# 仅在 4090 的策略服务进程里
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.50
```

若脚本仍把比例改回 0.88，VAE 加载会被显存检查拒绝。

## 在 4090 上接上真实调用

链路检查里的 `infer` / `decide` 是离线替身，用来证明调用顺序。上机时换成下面两个真实入口。

pi0.5，chunk 用完再请求服务：

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from agilex_control.agent_in_loop import ChunkedPi05, GptTakeover, RsiOodHead, AgentInLoop

policy = WebsocketClientPolicy(host="127.0.0.1", port=8088)

def infer_chunk(observation):
    result = policy.infer(observation["openpi_request"])
    return list(result["actions"])

vla = ChunkedPi05(infer_chunk)
```

`openpi_request` 仍由 clothesfoldingv2 的 `encode_openpi_request` 生成：关节 `deg2rad`，夹爪毫米乘 `1e-3`。策略输出的弧度/米要经 `action_postprocessor` 变回度/毫米，再下发 Piper。

GPT 用环境变量里的 key，不要把 key 写进仓库：

```python
import os
from agilex_control.local_loop import openai_decide

def decide(messages):
    decision, _payload = openai_decide(
        os.environ["OPENAI_API_KEY"],
        os.environ.get("OPENAI_MODEL", "gpt-4o"),
        messages,
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    return decision

gpt = GptTakeover(decide, demo_path="demos/fold.mp4")
loop = AgentInLoop(
    vla,
    RsiOodHead("models/v11/fold_all.pt", ".", device="cpu"),
    gpt,
)
```

Wan 编码在 4090 上用 CUDA。pi0.5 已经按上一节把显存让出来之后：

```python
from agilex_control.rsi_observe import CausalWanVisual

wan = CausalWanVisual(device="cuda")
```

每条任务开始时调用 `loop.ood.reset()` 和 `wan.reset()`。

## 单位

| 位置 | 关节 | 夹爪 |
| --- | --- | --- |
| Piper SDK | 0.001 度整数 | 0.001 mm 整数 |
| RobotState / GPT phase | 度 | 毫米 |
| pi0.5 与 RSi `state14` | 弧度 | 米 |

180° 对应 π rad，70 mm 对应 0.07 m。OpenPI 请求末尾补的 6 个 `-10000` 不要放进 RSi 的 14 维。

## 移植后的检查

在仓库根目录：

```bash
git clone -b real_world https://github.com/xiaolanliu/RSi.git
cd RSi
conda activate rsi
python -m pip install -e real_world --no-deps
cd real_world
export PYTHONPATH=src
python -m unittest tests.test_agent_in_loop
python -m agilex_control.agent_in_loop --demo demos/fold.mp4
```

第二条会读仓库根目录的 `examples/episodes/pant_fail_ep000000.npz` 和 `models/v11/fold_all.pt`。通过时退出码为 0，并且：

- `chain_ok` 为 true
- `vla_commands` 大于 0，且报警前的最后一次调用是 `vla`
- `first_alarm_frame` 为 303
- `gpt_commands` 为 1，`handback_frames` 含 303
- `final_owner` 为 `vla`

这证明正确加载时会先走本地策略，再由 RSi 在线监测在安全分位之外报警，然后 GPT 接管一次并交还。它不证明 4090 上的 websocket 和 API key 已经接通。那两项在服务和 key 就绪后，用上面的 `infer_chunk` 与 `decide` 替换检查里的替身。
