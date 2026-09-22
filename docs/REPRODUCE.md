# 复现与重新训练

## 1. 三个层次的复现

| 层次 | 仓库已提供 | 额外需要 | 本次发布验证范围 |
|---|---|---|---|
| 发布权重推理 | 3 份 checkpoint、8 条在线真实样例、1 条离线真实样例、历史独立预测 | Python 环境 | 全轨迹逐帧一致性、CPU/CUDA；见 `validation/` |
| 自有数据部署 | 在线 API、LeRobot loader、Wan 特征提取代码、输入规范 | 同步 state/RGB、匹配 VAE 或已有 visual48 | 样例输入路径验证；没有替你验证新的相机/机械臂分布 |
| 全量从头训练 | 模型/训练源代码、配置、episode 划分、路径映射模板、准备与校准入口 | 原始正常/OOD 数据及视觉缓存，GPU/时间 | 本次未重新跑 2000 轮；给出可执行流程和小规模准备/拟合检查 |

下载 checkpoint 并重现输出，与从随机初始化获得完全相同参数是不同目标。发布模型保持原始字节；改变硬件、DDP world size、随机采样顺序、数据路径元信息或 CUDA 实现可能改变重训结果。原离线 checkpoint 没有优化器状态，其 `--resume-checkpoint` 是恢复模型参数继续训练，不是逐位恢复当时优化器轨迹。

## 2. 发布验收

```bash
rsi-verify --hashes-only
python tools/run_unit_tests.py
rsi-verify --device cpu --output outputs/verification_cpu.json
rsi-verify --device cuda:0 --output outputs/verification_cuda.json
```

校验内容：

1. checkpoint、配置、样例和发布代码 SHA256；在线 encoder2000 参数与 V11 内对应张量完全相同。
2. 每帧三项原始证据、瞬时分数、持续分数和五阶段概率。浮点容差 `atol=8e-5, rtol=3e-4`。
3. 所有布尔报警、确认阶段、接受阶段、离线硬边界逐项完全相同。
4. 在线只有三项 OOD，禁止旧 dropout disagreement 参与；阶段不回溯/跳级。
5. 独立前缀、重置、两种14维排列等价性。

在线样例包括一个接近报警边界的正常验证 episode、3 个原始 OOD、4 个 pant_fail，总计12384帧。前4条以完整规范化76维输入保存，避免反归一化重建输入引入数值漂移；pant_fail 保存原始 state14 和 visual48。离线样例1164帧。

Golden 来自发布前独立缓存的完整轨迹评估和原离线报告，不是新回放脚本自己生成自己比较。`examples/manifest.json` 记录每个文件、帧数、split 和预期首报时间。样例不包含全部训练数据和原始相机视频。

离线历史报告使用 GPU cuDNN TF32；CPU FP32 软概率与其最多相差0.00114918，硬边界均为 `[238,423,699,899,1165]`。因此离线 NPZ 额外保留通过**原 COMPILE 代码 + 原始 parquet/视觉缓存**独立计算的 `phase_probs_cpu`、`boundaries_cpu`，验收按设备选择，不扩大原有容差。出处见 `provenance/offline_cpu_reference.json`。在线CPU/GPU共用同一套历史参考，无此分流。

| pant_fail | 首次报警帧（0-based） | 秒 | 总报警帧 |
|---|---:|---:|---:|
| EP0 | 303 | 10.10 | 1278 |
| EP1 | 545 | 18.17 | 1089 |
| EP2 | 188 | 6.27 | 436 |
| EP3 | 261 | 8.70 | 292 |

这些时间仅是算法首报时间，无人工故障起点标签；短暂首报不代表已稳定介入。原完整评估记录在 `provenance/v11_evaluation.json`。

## 3. 全量数据与目录映射

`configs/online_data_index.json` 枚举2799条原始 episode 的身份、长度、FPS和split：正常 train2246、calibration275、validation275、OOD3。前者的原始教师训练/验证划分在 `configs/offline_split.json`。4条外部 pant_fail 不参加正常参考拟合和校准。

先取得原始 LeRobot 数据与相同 causal visual48 缓存。没有公开下载原始机械臂数据的链接，本仓库不假设它们能从 Hugging Face 自动获取。所有正常 dataset 名称及实际路径模板在 `configs/dataset_paths.example.json`。

```bash
cp configs/dataset_paths.example.json dataset_paths.json
# 编辑 dataset_paths.json：把每个 data_root 和 visual_cache_dir 换成本机路径
```

示例条目：

```json
{
  "data_fold_pant_0820_lerobot": {
    "data_root": "/datasets/data_fold_pant_0820_lerobot",
    "visual_cache_dir": "/datasets/visual/data_fold_pant_0820_lerobot/global_image",
    "state_layout": "joints12_grippers2",
    "state_columns": ["state.joints", "state.gripper_w"]
  }
}
```

`source_root` 在 index/split 中保留历史字符串作为 episode 身份，**不是运行时强制访问路径**。新工具以 dataset basename 查上述映射。不要更换历史身份后继续宣称用了原始split。缓存每个 episode 要有 `features[T,48]`，长度匹配，提取时 `frame_stride=1`。

## 4. 重新准备在线训练输入

以下输出路径均为新目录，不覆盖发布模型。需要 full 依赖与 GPU：

```bash
python -m rsi_tools.prepare_training \
  --paths dataset_paths.json --index configs/online_data_index.json \
  --teacher models/offline_2000/checkpoint_epoch_2000.pt \
  --device cuda:0 --output outputs/retrain/source

python -m agent_closed_loop.prepare_ood_v4 \
  --source outputs/retrain/source --teacher-split configs/offline_split.json \
  --output outputs/retrain/data
```

第一步读取真实 state/visual，正常 episode 用冻结离线2000教师生成五阶段软目标；OOD目标置零、不做教师监督。第二步构造 `state14 + backward_delta14 + visual48 + teacher5` 共81列；只在正常训练帧拟合 mean/scale，然后标准化前76列，保留源文件hash、split及长度。分割/归一化都不能按帧随机混到测试集里。

输出 `data/frames.npy`、`normalization.json`、`manifest.json`。正式数据总帧3215346，正常训练帧2579110。任何数量、长度不匹配都应先检查数据版本。不要用所附9条样例代替正常训练分布；训练入口会拒绝非2246条的正式编码器/阶段训练。

## 5. 在线128维编码器重训2000轮

```bash
# 可先预检显存与梯度；这个输出仅是 smoke，不代表训练完成
python -m agent_closed_loop.train_ood_v4 \
  --data outputs/retrain/data --output outputs/retrain/preflight \
  --dim 128 --device cuda:0 --epochs 2000 --batch-size 64 \
  --target-frames 256 --workers 4 --lr 0.0003 --seed 29 --preflight-steps 8

# 正式训练，每轮每个训练目标帧恰好监督一次
python -m agent_closed_loop.train_ood_v4 \
  --data outputs/retrain/data --output outputs/retrain/encoder \
  --dim 128 --device cuda:0 --epochs 2000 --batch-size 64 \
  --target-frames 256 --workers 4 --lr 0.0003 --seed 29
```

配置必须显式指定 batch=64，历史脚本的默认值32不是本次发布训练值。训练使用149帧左上下文、256帧目标，BF16、AdamW、固定seed29、最终2000轮，无验证集早停。中断后使用完全相同参数加 `--resume`；脚本检查代码、协议、数据manifest指纹。

最终 `checkpoint_epoch_2000.pt` 是新的编码器，hash可能与发布模型不同。原始 encoder checkpoint 也已随仓库提供；若只做推理，完全无需重训。不能把它塞进不同归一化/教师缓存后跳过 provenance 检查。

## 6. 正常参考特征和阶段头

```bash
python -m rsi_tools.refit --encode-only \
  --data outputs/retrain/data \
  --encoder outputs/retrain/encoder/checkpoint_epoch_2000.pt \
  --source outputs/retrain/encoded --device cuda:0

python -m agent_closed_loop.train_duration_subtask \
  --source outputs/retrain/encoded --output outputs/retrain/stage \
  --epochs 60 --batch-size 16 --duration-weight 0.05 \
  --prior-strength 2 --device cuda:0
```

`refit --encode-only` 调用原正常参考特征准备函数，冻结编码器、生成 latent、正常标准化参数、LINe等统计，不训练旧版弱监督风险网络。阶段头完整遍历正常轨迹，seed41、默认长度200，输出 `checkpoint_epoch_0060.pt`。

## 7. 重新拟合三项分布及最终V11报警

```bash
python -m rsi_tools.refit \
  --data outputs/retrain/data \
  --encoder outputs/retrain/encoder/checkpoint_epoch_2000.pt \
  --source outputs/retrain/encoded \
  --stage outputs/retrain/stage/checkpoint_epoch_0060.pt \
  --template models/v11/fold_all.pt \
  --output outputs/retrain/v11 --device cuda:0
```

顺序为：检查编码器/阶段/缓存身份 → 装配新编码器/阶段和正常化 → 正常PCA记忆/三项参考 → 90帧近期动作单元分布、稳定LINe路由 → 重复强度乘1.5 → 正常校准轨迹峰值重新计算边界。所有中间路径显式传入，不需要复制过去的V7/V8/V10目录。

此命令固定使用发布版阶段确认参数 `evidence_frames=15`、`confirmation_threshold=0.77`，其历史选择来源是正常校准；不会重新做超参数搜索。它保留V11的10帧均值/5帧持续规则。对于不同任务/机械臂，固定先验是否合适必须另行验证。原 `calibrate_duration_subtask.py` 还依赖历史V6基线记录，不是本发布流程必须调用的入口。

最终包 `outputs/retrain/v11/fold_all.pt`，拟合评估见同目录 `weighted_fit/evaluation.json`。此训练流程不使用4条 pant_fail 的信息来设门限；部署后可用它们及新收集的独立数据评估。替换 `rsi-replay --checkpoint ...` 可检查新模型，但不要要求重训模型自动通过旧权重的 exact golden。更改模型的研究结果需独立记录。

## 8. 原始离线教师重新训练或继续训练

从路径映射生成教师所需 manifest（正常 dataset，沿用原始顺序）：

```bash
python tools/make_offline_manifest.py --paths dataset_paths.json \
  --output outputs/offline_manifest.json
```

用原始模型结构做一次新的离线训练：

该离线 manifest 入口要求原历史 `state.joints`（12关节）与 `state.gripper_w`（2夹爪）的排列；它会拒绝在线接口中可接受的其他排列，避免悄悄错用统计量。

```bash
torchrun --standalone --nproc-per-node=2 -m compile.train \
  --dataset-manifest outputs/offline_manifest.json --output-dir outputs/offline_retrain \
  --max-segments 5 --num-codes 16 --hidden-dim 64 --embedding-dim 32 \
  --state-hidden-dim 32 --sequence-encoder tcn --fusion-mode attention \
  --causal-sequence-encoder --policy-uses-fused-features --visual-input-scale 1 \
  --tcn-kernel-size 5 --temporal-stride 1 --derive-actions-from-state \
  --adaptive-poisson-rate --kl-weight 1 --boundary-kl-weight 0.01 \
  --segment-balance-weight 0.05 --segment-balance-min-ratio 0.5 \
  --segment-balance-max-ratio 1.8 --kl-warmup-epochs 5 \
  --batch-size 8 --epochs 2000 --learning-rate 0.0001 --seed 0 \
  --report-epochs 2000 --validation-interval 100
```

这里给出保留架构下的重训配方；离线历史checkpoint未保存全部启动/优化器状态，不能保证上述新训练逐位生成原始权重。精确报告复现应加载随仓库提供的离线权重。若继续训练，添加 `--resume-checkpoint models/offline_2000/checkpoint_epoch_2000.pt --epoch-offset 2000`，并把 `--epochs` 改为额外训练轮数；注意优化器重新创建。单GPU可用 `python -m compile.train`，但改变了分布式优化路径。

## 9. 原始资产与发布代码的区别

- 三个权重文件未改写；原始代码导入hash在 `provenance/source_import_sha256.json`。
- 发布新增 `rsi_tools/`、CLI、文档、真实样例、验收；`cache_wan_visual` 新增本仓库 third_party 查找；两个拟合脚本新增显式路径参数。模型前向、阶段递推和OOD公式保持不变。
- 历史 `reports/*`、大型 `runs/*`、旧V5–V10 bundle及完整命令日志未整体上传。相应旧研究脚本仍保留源码，用其 `--help`/源代码检查额外资产需求；不要把这些旧入口当作首次复现入口。
- `provenance/*.json` 中旧机器路径保留作出处；可迁移的新数据访问路径来自你自己的映射文件。发布模型加载只依赖checkpoint本身。

## 10. 可选网页检查

预生成的两个便携报告已通过真实 Chromium 检查。若修改网页模板，可以在安装 Node.js 后执行：

```bash
npm install --no-save playwright
npx playwright install chromium
# 一个终端：
python -m http.server 8000 --bind 127.0.0.1
# 另一个终端（同一仓库根目录）：
node tools/verify_report.cjs
```

检查 slider、EP0第303帧报警、CSV/NPZ资源和手机布局，结果写入 `validation/browser.json`。修改模板后先用 `rsi-replay` 重新生成报告。Node/浏览器不是模型推理的依赖。
