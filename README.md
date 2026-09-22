# RSi — 在线阶段识别与三信号 OOD 监测

这是 `Agent_closed_loop` 当前 **V11** 的独立复现仓库，同时保留原始离线 CompILE **epoch 2000** 和在线编码器 **epoch 2000** 权重。克隆后即可在 CPU 上回放真实轨迹、核对逐帧结果，不需要原开发机器、COMPILE/SIEVE 目录、机器人、URDF 或下载视觉大模型。

RSi provides a frozen V11 streaming subtask/OOD monitor, both original epoch-2000 checkpoints, real trajectory fixtures, independent golden outputs, and reproducible CLI examples. Start with the commands below; detailed documentation is in Chinese with explicit commands and data contracts.

## 1. 十分钟内开始复现

推荐 Linux + Python 3.10。在仓库根目录执行：

```bash
git clone https://github.com/xiaolanliu/RSi.git
cd RSi
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-full.txt
python -m pip install -e . --no-deps

# 校验模型、样例和代码指纹
rsi-verify --hashes-only

# 回放整条 pant_fail EP0；逐帧对比历史独立计算结果
rsi-replay --input examples/episodes/pant_fail_ep000000.npz \
  --expected examples/expected/pant_fail_ep000000.npz \
  --output outputs/pant_fail_ep0 --device cpu

# 全部 8 条在线轨迹 + 1 条离线轨迹，检查概率、分数、报警、阶段和重置
rsi-verify --device cpu --output outputs/verification_cpu.json
python tools/run_unit_tests.py
```

EP0 应输出 `frames=2018`、`first_alarm_frame=303`（10.10 秒）、`alarm_frames=1278`、`reference_check=passed`。首次报警并不是人工标注的故障起点。

打开 `outputs/pant_fail_ep0/index.html` 可拖动时间轴，查看阶段概率、持续风险和报警；全部数据已嵌入页面，没有 CDN。也可运行 `python -m http.server 8000 --bind 127.0.0.1`，访问 `http://127.0.0.1:8000/outputs/pant_fail_ep0/`。NPZ、CSV、摘要与 HTML 同目录生成。完整验收共 **13,548 帧**，其中在线 **12,384 帧**、离线 **1,164 帧**。

仓库也附有已生成的 [EP0 便携回放页面](reports/pant_fail_ep0/index.html) 和 [离线2000轮回放页面](reports/offline_2000/index.html)，克隆后可直接打开；GitHub 文件预览不会执行 HTML。

## 2. 三份权重，含义不同

| 文件 | 用途 | 是否可直接在线报警 |
|---|---|---|
| [`models/v11/fold_all.pt`](models/v11/fold_all.pt) | 当前完整部署包：编码器、阶段头、三项 OOD 参考分布、归一化、校准参数 | 是 |
| [`models/encoder_2000/checkpoint_epoch_2000.pt`](models/encoder_2000/checkpoint_epoch_2000.pt) | 正常数据蒸馏 2000 轮的 128 维因果编码器；参数与 V11 内编码器逐项相同 | 需阶段头与 OOD 配置 |
| [`models/offline_2000/checkpoint_epoch_2000.pt`](models/offline_2000/checkpoint_epoch_2000.pt) | 原始离线 CompILE 教师；完整 episode 分成五个阶段 | 离线分段模型 |

权重保持原始字节，直接存储在 Git 中，不是 LFS 指针。SHA256、大小和样例指纹见 [`artifacts.json`](artifacts.json)。这里不包含 pi0.5 动作生成模型权重；RSi 是监测器。

## 3. 阅读顺序

1. [环境安装与排错](docs/ENVIRONMENT.md)：CPU/CUDA、依赖版本、磁盘与可选视觉依赖。
2. [详细使用说明](docs/USAGE.md)：在线 API、两种关节排列、NPZ 格式、离线推理和视觉输入。
3. [架构与 OOD 计算](docs/ARCHITECTURE.md)：数据流、三项证据、经验分位数、报警与阶段冻结。
4. [复现、重新训练与校准](docs/REPRODUCE.md)：验收标准、原始数据需求、完整训练步骤与限制。
5. [给其他 agent 的操作指引](AGENTS.md)：修改入口、应运行的检查、常见错误。
6. [V11 真实 action 研究记录](docs/V11_RESEARCH_NOTES_CN.md)：命令单位、执行队列和后续动力学接入。

## 4. 当前能力与边界

- 正式报警只有 **投影距离、动作重复、LINe 正常阶段支持不足** 三项。重复尾部强度权重是 1.5；阈值来自正常校准集。
- 输入是连续 30 FPS 的 `state14 + causal visual48`。阶段默认长度先验为 200 帧，阶段只相邻推进，报警时暂停阶段更新。
- `action_unit_repetition` 描述实测动作轨迹的重复；目前 **真实 action chunk / sent 命令尚未进入正式报警**。动力学和命令诊断代码保留为研究接口。
- 提供的真实样例足以完整复现发布权重的推理结果；2796 条正常训练/校准/验证轨迹、3 条原 OOD、原始相机视频及约 2.7 GB Wan VAE 不随仓库发布。全量重训需要另行取得原始数据；见复现说明。
- 历史正常验证集有 12/275 条、140/314751 帧报警。四条 pant_fail 均有报警，但已被用于诊断设计，不能当作新独立泛化评估。分数和分位数不是失败概率。

`provenance/` 保存原始训练/评估出处，`validation/` 保存本次发布验收结果。带有旧机器路径的 provenance 字段只是实验身份记录；发布推理不读取这些路径。历史所有网页及大型中间缓存没有整体复制；本仓库通过回放命令生成可独立打开的新页面。

第三方 Wan 代码及许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
