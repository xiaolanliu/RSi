# π0.5 独立批量采集

此入口只使用 π0.5 控制目标双臂；不加载 Wan/RSI，不调用 GPT，也不读取 API
密钥。RoboDojo 的原生场景、辅助机器人行为、25 Hz 控制频率、任务成功条件
和任务时限保持不变。每次推理生成 50 步，执行前 10 步后从最新观测重新推理。

本机发现 54 个任务配置（包含 `_random` 变体），每任务 30 条有效实验，目标
1620 条。默认原始 `pi05_base`；使用 `--checkpoint-kind demo` 则明确标记为
RoboDojo 微调 π0.5 采集，不能混称为 base 的成功率。实验不承诺固定的成功/
失败比例，也不根据结果选择性保留数据。

```bash
conda activate rsi
cd /path/to/RSi
python -m rsi_loop.collection prepare --output outputs/pi05_all_tasks_30 \
  --checkpoint-kind base --trials 30
python -m rsi_loop.collection run --output outputs/pi05_all_tasks_30
```

`prepare` 固定任务、权重身份、资源路径和采集代码 SHA256。布局按 group 0、1、2
交错取样，正常情况下每组 10 个不同布局。每个回合使用独立且记录在案的
π0.5 随机种子。布局编号严格遵循原生 SeedManager 的文件排序位置，不能把
文件名数字后缀直接当成 layout_id。原始布局文件的 SHA256 在回合配置中保存。
可用 `--tasks fold_clothes align_blocks` 限定一个新实验计划的任务范围。

π0.5 权重常驻 GPU 0，每回合重置 RNG 和输出目录；每条轨迹新建 GPU 1 上的
仿真 worker，避免跨任务状态污染。各任务轮流运行，以尽早得到任务覆盖。
每回合完整执行到原生结束，不用统一的较短 `max_steps` 截断长任务。

## 长期运行、进度与恢复

在仓库根目录用当前 rsi Python 启动后台进程，例如：

```bash
nohup python -u -m rsi_loop.collection run --output outputs/pi05_all_tasks_30 \
  > outputs/pi05_all_tasks_30/runner.log 2>&1 < /dev/null &
```

打开输出目录的 `index.html` 查看总数、每任务成功/失败/错误计数、当前回合和
步数、剩余空间，以及最近录像。页面每 30 秒刷新，`status.json` 每 15 秒更新。
`dataset_index.json` 保存所有尝试；只有 `label=success/failure` 计入有效实验。

```bash
# 在当前回合完成后停止；不会删数据。
touch outputs/pi05_all_tasks_30/STOP
# 确认 status.json 的 status=stopped 后，删除停止标记并续跑。
rm outputs/pi05_all_tasks_30/STOP
python -m rsi_loop.collection run --output outputs/pi05_all_tasks_30
```

同一目录有排他锁，禁止重复启动。SIGTERM/SIGINT 也请求回合结束后停止。
崩溃后已经提交的有效结果不重复采集，未提交尝试保留为 interrupted。
运行错误和物理不稳定结果不计作任务失败；后续使用新的原生布局补足配额。
一个任务连续三次运行错误会暂停该任务，其余任务继续；页面保留欠缺数量，
绝不将该情况报告成 30 条完成。需要排查错误后建立明确版本的新补采计划。

可用空间低于 50 GiB 时停止启动新回合、释放 VLA worker，并等待空间恢复。
不会自动删除成功或失败样本。可以在 `prepare` 时设置 `--minimum-free-gib`。
采集过程中不要修改 `rsi_loop` 源码；续跑会校验源代码身份。

## 数据与验收

`episodes/<task>/attempt_<id>/` 保留：

| 文件 | 内容 |
|---|---|
| `observations/*.npz` | 初始帧及每次动作后观测；state14、物理夹爪开度、末端位姿、时间、任务指令、每路 RGB SHA256 |
| `rgb/cam_*.mkv` | 三路原生 640×480 RGB；libx264rgb、CRF=0，无损、原生 25 FPS |
| `sensors.mp4` | 便于观看的三视角有损拼接预览；训练时读取上面的无损视频 |
| `vla/chunk_*.npz` | 完整 50 步 predicted/commanded action chunk，以及其对应的观测步和 state |
| `events.jsonl` | 已 ACK 的实际执行命令、来源、逐步耗时；未运行 OOD，不提供伪造风险分数 |
| `commands_requested.jsonl` | 执行前请求；必须匹配 ACK 才能当作已执行动作 |
| `native_outcome.json` | 原生成功/失败、结束、有效性、控制步数与原生时限 |
| `run_config.json`、`vla/vla_metadata.json` | 任务/布局/随机种子、明确的 checkpoint 与归一化身份 |

所有有效回合必须满足：完整原生结束、物理状态有效、零接管、仅 VLA 命令、
连续动作 ACK、T 个动作对应 T+1 个观测。随后解码三路视频的**全部帧**，逐帧
验证保存前 RGB 的 SHA256；视频缺帧、损坏或像素变化都不会被标记为有效数据。

使用统一接口恢复单个原生观测（兼容原有包含 RGB 的 NPZ）：

```python
from rsi_loop.context import read_observation
obs = read_observation("outputs/pi05_all_tasks_30/episodes/align_blocks/attempt_000", 50)
```

此随机读取接口从视频开始解码到目标帧，适用于上下文检查；训练整条轨迹时
应顺序解码各视频，与 NPZ 按 step 对齐，避免逐帧重复随机读取。无需从有损
预览恢复训练图像。可用 `tools/report_sim_run.py --run <episode_dir>` 生成关节/
夹爪曲线页面；独立采集的页面会明确显示未运行 OOD。

采集结束前的成功率只是进度统计。后续模型拟合必须按 episode 划分正常训练、
校准、验证集合；任务失败样本不能进入正常分布拟合。
