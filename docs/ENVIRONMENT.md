# 环境、依赖与安装

## 验证基线

原训练环境：Ubuntu 22.04、Python 3.10.12、PyTorch 2.8.0+cu128、NumPy 1.26.4，RTX 4090 D（24 GB），驱动 550.90.07。其他直接依赖固定在 `requirements-full.txt`：pandas 2.2.3、PyArrow 20.0.0、PyAV 13.1.0、Pillow 12.3.0、einops 0.8.1、SciPy 1.15.3、Matplotlib 3.9.4。

本次发布的实际测试结果见 `validation/`。这些版本是复现基线；不能由依赖范围推断其他平台已经测试通过。

`requirements-lock-cuda.txt` 记录本次新建隔离环境的完整依赖版本；可用 `python -m pip install -r requirements-lock-cuda.txt --extra-index-url https://download.pytorch.org/whl/cu128` 安装，然后单独安装本仓库。该新环境使用 CUDA wheel 分别执行 CPU/GPU 验证，没有继承其他项目的 site-packages。纯 CPU wheel 的安装流程由仓库 CI 再次检查。

## CPU：推荐的首次验收方式

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-full.txt
python -m pip install -e . --no-deps
python -c 'import torch,numpy; print(torch.__version__,numpy.__version__)'
rsi-verify --hashes-only
```

先装 CPU PyTorch 可避免为 CPU 回放下载整套 CUDA 库。在线核心实际只需 NumPy/PyTorch；若只接自己的视觉特征，可使用 `requirements-core.txt`。离线 CompILE 的加载器需要 pandas/PyArrow，边界可视化模块需要 Matplotlib，因此完整验收使用 full 依赖。

如果系统缺少 `ensurepip`，安装对应的 `python3.10-venv` 系统包，或使用已安装的 uv：

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements-full.txt
uv pip install --python .venv/bin/python -e . --no-deps
source .venv/bin/activate
```

## CUDA：推理或训练

在独立环境安装同一 PyTorch 的 CUDA 12.8 wheel：

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-full.txt
python -m pip install -e . --no-deps
python -c 'import torch; print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))'
rsi-verify --device cuda:0 --output outputs/verification_cuda.json
```

若从 CPU wheel 切换，应在新环境安装，或明确卸载旧 torch 后安装所需版本。wheel 自带运行库，仍需要可用 NVIDIA 驱动；系统 `nvcc` 并不是回放必要条件。在线 FP32 推理显式关闭 matmul/cuDNN TF32，保持逐步与批量计算的一致性；请不要擅自转换成 FP16/BF16 或开启 TF32。训练编码器/阶段头使用 BF16 autocast，这是另一条计算路径。

## 资源与范围

- 仓库模型和缓存样例只有几十 MB；CPU 回放无需 GPU。PyTorch/CUDA 安装体积另计。建议至少 4 GB RAM 进行回放，2 个计算线程即可；CLI 默认设为 2。
- 全量训练有 3,215,346 帧；准备后的 `frames.npy` 约 1.04 GB，128-D latent 约 1.65 GB，另有教师缓存、证据、视频和检查点。建议为中间产物预留至少 15 GB，原始视频另计。
- 发布编码器训练 batch size=64、target_frames=256、left context=149，使用 CUDA。原离线训练使用两卡 DDP。降低 batch size 可以降低显存需求，但会改变优化路径，不应称为精确重训。
- 原始 RGB → visual48 需要独立约 2.7 GB Wan2.2 VAE 和视频解码依赖；这是可选步骤，缓存样例不需要它。见 `USAGE.md`。

## 常见错误

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: agent_closed_loop` | 激活正确环境，在仓库根目录执行 `python -m pip install -e .` |
| `No module named pandas / matplotlib` | 安装 full 依赖；离线工具的导入需要它们 |
| `Could not locate SIEVE` | 本发布版内置 `third_party/wan22_vae`；使用本仓库安装的包，并传有效 `--vae`；不要误导入旧项目 |
| CUDA 不可用 | 先用 `--device cpu` 完成验收；训练脚本需要 CUDA wheel 和 GPU |
| 权重哈希不一致 | 检查是否误用了其他版本、下载截断、LFS 指针或自行重存的 checkpoint；不要直接更新 manifest 掩盖问题 |
| 概率略有差异 | 在线使用固定版本与 FP32 配置；验收容差是 `atol=8e-5, rtol=3e-4`，报警/阶段/边界必须完全相同。离线 CPU 使用独立原代码CPU参考，GPU使用原TF32报告参考，见 `REPRODUCE.md` |
| 新数据一开始大量报警 | 首先核对单位、排列、视觉预处理和时间间隔，再分析分布变化；不要直接降低阈值 |
| 出现旧机器的绝对路径 | provenance JSON 和原权重可能记录历史身份；发布 replay 不需要这些路径。研究脚本请使用 `REPRODUCE.md` 的显式路径流程 |
| 下载失败或 CONNECT 403 | 检查本机代理/镜像配置；仓库不设置系统网络代理，也不依赖开发机代理 |
