# Third-party components

`third_party/wan22_vae/vae2_2.py` and its initializer were copied without computation changes from the VAE implementation used by the original experiment. The source credits **The Alibaba Wan Team Authors (2024–2025)** and originates from [Wan2.2](https://github.com/Wan-Video/Wan2.2). The accompanying [Apache License 2.0](third_party/wan22_vae/LICENSE.txt) is retained. The local integration uses this component only as a frozen observation encoder.

The optional VAE checkpoint comes from [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B). It is not stored in this repository. The exact experiment checksum and download instructions are in [the usage guide](docs/USAGE.md).

The offline `compile/` model is the local project's CompILE-inspired PyTorch implementation and includes modifications for continuous robotic trajectories and visual features. It should not be described as the original paper authors' released code. The offline segmentation idea is discussed in [CompILE, ICML 2019](https://proceedings.mlr.press/v97/kipf19a.html). V11's causal online stage updates and OOD calibration are subsequent project components, not claims from that paper.

The simulation integration downloads independent source trees into ignored `external/` directories. Exact revisions are recorded in [configs/sources.lock.json](configs/sources.lock.json). These trees and their assets are not redistributed as part of RSi:

- [OpenPI](https://github.com/Physical-Intelligence/openpi), Apache-2.0: original pi05 inference and transforms. Its model weights remain external and subject to their own terms.
- [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo), MIT: native simulation, observations, actions and task success checks, with its pinned XPolicyLab, IsaacLab and cuRobo submodules. Retain each upstream license when downloading them. NVIDIA IsaacSim and simulation assets are separate dependencies under their respective terms.
- [GPT-Policy](https://github.com/cheng-haha/GPT-Policy): the pinned revision states that source is published for review/evaluation and that no redistribution/commercial license has been selected. RSi imports its video extractor from the user's external checkout for evaluation; no GPT-Policy source is copied into the tracked RSi package.

The single documented IsaacLab dependency patch changes its Starlette version constraint to match IsaacSim's FastAPI dependency. It does not change physics or task criteria. RSi's control owner, IPC, recovery protocol and robot kinematics adapter are local implementations.
