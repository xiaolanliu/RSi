# 使用操作说明

以下命令都从仓库根目录执行，先按 `ENVIRONMENT.md` 安装环境。

## 1. 输入契约

| 项目 | 必须满足 |
|---|---|
| 时间 | 每条 episode 内按时间顺序连续 30 FPS；每次 `step` 恰好一帧 |
| state | `float32[14]`，12 个关节为弧度，2 个夹爪宽度为米，全部有限 |
| 夹爪位置 | 历史布局 `L6,R6,gL,gR`，或机器人布局 `L6,gL,R6,gR`，由 `state_layout` 显式说明 |
| visual | 当前帧及此前四帧，经相同 Wan2.2 VAE 生成的 `float32[48]`；不是图像、分类概率或其他编码器向量 |
| episode | 每次任务开始/重启调用 `reset()`；一个对象的缓存只属于一条机械臂轨迹 |
| 归一化 | 传给 `step()` 的是原始量；模型内部归一化。只有历史规范化样例才使用 `step_normalized()` |

布局示意：

```text
joints12_grippers2: [Lq0 Lq1 Lq2 Lq3 Lq4 Lq5 Rq0 Rq1 Rq2 Rq3 Rq4 Rq5 gL gR]
left7_right7:       [Lq0 Lq1 Lq2 Lq3 Lq4 Lq5 gL Rq0 Rq1 Rq2 Rq3 Rq4 Rq5 gR]
```

夹爪不是第七个旋转关节。14 维整体不能全部乘 180/π。部署系统若返回度/毫米，先只转换对应维度，再调用模型。

## 2. 在线 API

```python
from agent_closed_loop.monitor import OnlineMonitor

monitor = OnlineMonitor("models/v11/fold_all.pt", device="cpu", fps=30)
monitor.reset()
# 遍历你自己的同步观测流。state14/visual48 必须遵守上面的输入契约。
for state14, visual48 in observations:
    result = monitor.step(state14, visual48, state_layout="left7_right7")
    print(result["frame_index"], result["confirmed_phase"],
          result["risk_score"], result["alarm"])
```

`observations` 由调用方提供；示例不是相机或控制驱动。可先用真实 fixture 构造它：

```python
import numpy as np
from agent_closed_loop.monitor import OnlineMonitor
data = np.load("examples/episodes/pant_fail_ep000003.npz")
monitor = OnlineMonitor("models/v11/fold_all.pt")
for state, visual in zip(data["states"], data["visual"]):
    out = monitor.step(state, visual, state_layout="joints12_grippers2")
print(out["confirmed_phase"], out["alarm"])
```

输出字段：

| 字段 | 解释 |
|---|---|
| `frame_index` | 从 0 开始的当前帧 |
| `phase_probs` | P1–P5 五个阶段槽位的软概率；未赋予人工技能名称 |
| `confirmed_phase` | 1–5，单向、相邻推进的阶段记忆 |
| `accepted_phase` | 正常时等于确认阶段；报警时为 0，表示暂停接受阶段 |
| `ood_signals` | 三个原始证据：`projection_distance`、`action_unit_repetition`、`line_energy` |
| `ood_percentiles` | 三项各自在正常训练经验分布中的百分位（0–1），未乘重复权重 |
| `frame_risk_score` | 三项加权尾部强度的最大值，尚未平滑 |
| `risk_score` | 10 帧均值后再取 5 帧最小值的持续分数 |
| `risk_percentile` | 相对于正常校准轨迹最大分数的分位数；不是故障概率 |
| `alarm` | 校准尾部秩 `p <= 0.05` 的最终布尔判定 |
| `threshold` | 此权重导出的原始分数参考边界约 4.7099385 |
| `debug` | 仅 `debug=True` 返回，含 latent128 和三项证据，用于诊断 |

模型不执行急停、回退或新动作；调用方决定如何使用报警。阶段在报警时暂停，报警解除后继续原有记忆。在线推理不会往正常参考库追加当前观测，也不会更新网络权重。

## 3. 回放自己的记录

保存一条 episode 为无对象数组的 NPZ：

```python
import numpy as np
np.savez_compressed("episode.npz",
    states=np.asarray(states, dtype=np.float32),       # [T,14]
    visual=np.asarray(visual48, dtype=np.float32),     # [T,48]
    timestamps=np.arange(len(states), dtype=np.float64)/30)
```

```bash
rsi-replay --input episode.npz --state-layout left7_right7 \
  --checkpoint models/v11/fold_all.pt --device cpu --output outputs/my_episode
```

若 NPZ 包含 `normalized_input[T,76]`，CLI 使用规范化接口；不要同时用这个字段和另一套 state/visual 输入。`timestamps` 可省略，默认从 0 秒、30 FPS 开始；若提供，必须递增且相邻约 1/30 秒。丢帧/变帧率需要上游处理并重新评估时间窗假设，CLI 不会静默跳帧。程序检查 NaN、shape、帧数和采样间隔。

每次回放生成 `predictions.npz`、`predictions.csv`、`summary.json`、`index.html`。HTML 是单文件图表，无需互联网；在同目录保留其他文件即可下载 CSV/NPZ。它展示阶段和风险，没有伪造缺失的原始视频。

## 4. 原始离线 epoch 2000 分段

```bash
rsi-replay --mode offline \
  --checkpoint models/offline_2000/checkpoint_epoch_2000.pt \
  --input examples/episodes/offline_pants_0820_ep000003.npz \
  --expected examples/expected/offline_pants_0820_ep000003.npz \
  --device cpu --output outputs/offline_2000
```

离线接口接收完整 raw `states[T,14] + visual[T,48]`。它内部构造 `state[t+1]-state[t]` 重建目标，末帧屏蔽动作损失，`sample_latents=False`。输出 `phase_probs=segment_masks.T`，`boundaries` 是 one-based 边界，最后为 `T+1`。尽管主编码器使用因果 TCN，边界在整条时间轴归一化，因此这仍是离线模型，不能逐帧调用冒充在线阶段推断。

如需直接读取自有 LeRobot 数据，`compile.data.LeRobotEpisodeDataset` 支持 v2 的 `episode_*.parquet` 和 v3 的共享 `file-*.parquet`。原始 split 列的默认顺序是 `state.joints`（12）再 `state.gripper_w`（2）。`python -m compile.subtasks --help` 和 `python -m compile.evaluate --help` 提供原始完整离线导出入口。

## 5. 从 RGB 提取 visual48（可选）

缓存样例已附 visual48，不需要以下下载。接新相机数据时，需要相同的冻结 Wan2.2 VAE。发布仓库内置训练时使用的 `third_party/wan22_vae` 代码及 Apache-2.0 许可证；VAE 大权重由上游单独取得：

```bash
mkdir -p weights
curl -L --fail --retry 3 \
  https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/resolve/main/Wan2.2_VAE.pth \
  -o weights/Wan2.2_VAE.pth
sha256sum weights/Wan2.2_VAE.pth
```

必须与本项目实际使用的字节哈希一致：

```text
20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36
```

上游 `main` 可能变化，哈希不符时不要直接替换提取器，也不要用另一个 Wan 版本产生特征。VAE 文件没有塞入本项目 Git。

批量缓存 LeRobot 视频：

```bash
python -m agent_closed_loop.cache_wan_visual \
  --data-root /PATH/TO/LEROBOT_DATASET \
  --vae weights/Wan2.2_VAE.pth --sieve-root . \
  --video-key global_image --size 256 --frame-stride 1 \
  --vae-batch-size 8 --device cuda:0 --output-dir cache/my_dataset/global_image
```

视频应有匹配的 LeRobot 元数据与逐帧 state；自动解析共享 MP4 的 episode 起始时间。输出 `episode_000000.npz` 的 `features[T,48]`、`encoded_indices`、`frame_indices` 和 manifest。已有缓存会跳过，因此更换 VAE/参数时使用新目录。

精确预处理是：RGB uint8 → 等比例缩放到 256×256 并居中 letterbox → `pixel/127.5-1` → 当前及此前四帧（开头复制首帧补足 5 帧）→ VAE 编码 → 对 latent 的时间、高度、宽度求均值，保留 48 通道。CUDA 使用 bfloat16 VAE，池化输出 float32。每个窗口独立编码；不要改成对完整视频一次编码或把 VAE 的内部跨窗缓存一直累积。

在线使用时维护自己的 5 帧 RGB deque，并复用 `letterbox_frame` 和同一个 `Wan2_2_VAE`，按 `visual_features.encode_episode_features` 的 `flush_pending` 计算路径处理当前窗口。必须保证窗口结束于当前观测、相机与关节时序匹配。`frame_stride>1` 的缓存代码会在稀疏特征间插值并使用后续特征，仅适合离线预览；在线及正式复现固定为 1。

## 6. action chunk 与动力学接口

`command_logs.py` 支持解析真实 chunk/tick 日志、统一单位、丢弃过期前缀和对比同一未来时刻的规划；`CausalCommandEvidence` 生成因果跟踪/限幅诊断。`command_dynamics.py` 是待用正常真实命令数据拟合的响应模型接口。详见 `V11_RESEARCH_NOTES_CN.md`。

这两者目前不被 `OnlineMonitor.step()` 调用。不能把未来预测 chunk 当成已发生的实测运动，不能把 60 Hz 插值 tick 当成 30 Hz 策略步，也不能把 recorder 中 `action==state` 的列当成独立命令监督。
