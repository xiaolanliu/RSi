#!/usr/bin/env bash
set -euo pipefail
rsi_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$rsi_root"
rsi_conda=${CONDA_EXE:-conda}
rsi_prefix=${RSI_CONDA_PREFIX:-"$HOME/.conda/envs/rsi"}
if [[ ! -x "$rsi_prefix/bin/python" ]]; then
  "$rsi_conda" create --prefix "$rsi_prefix" python=3.11 pip -y
fi
rsi_python="$rsi_prefix/bin/python"
rsi_index=${RSI_PYPI_INDEX:-https://pypi.org/simple}
command -v uv >/dev/null || "$rsi_python" -m pip install uv
rsi_uv=$(command -v uv || echo "$rsi_prefix/bin/uv")
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-180}
"$rsi_uv" pip install --python "$rsi_python" --default-index "$rsi_index" \
  --index https://pypi.nvidia.com --index https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match -c requirements-rsi.lock.txt 'isaacsim[all,extscache]==5.1.0.0'
"$rsi_uv" pip install --python "$rsi_python" --default-index "$rsi_index" \
  -r requirements-integration.txt -c requirements-simulation.txt -c requirements-rsi.lock.txt
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.0.0+gd17b54ce
"$rsi_uv" pip install --python "$rsi_python" --default-index "$rsi_index" --no-build-isolation \
  -c requirements-simulation.txt -c requirements-integration.txt -c requirements-rsi.lock.txt \
  -e external/robodojo/third_party/IsaacLab/source/isaaclab \
  -e external/robodojo/third_party/IsaacLab/source/isaaclab_assets \
  -e external/robodojo/third_party/IsaacLab/source/isaaclab_tasks \
  -e external/robodojo/third_party/IsaacLab/source/isaaclab_rl \
  -e external/robodojo/third_party/IsaacLab/source/isaaclab_mimic \
  -e external/robodojo/third_party/IsaacLab/source/isaaclab_contrib \
  -e 'external/robodojo/third_party/curobo[cu12]'
"$rsi_uv" pip install --python "$rsi_python" -e . --no-deps
"$rsi_uv" pip check --python "$rsi_python"
"$rsi_python" -m rsi_loop.cli doctor
