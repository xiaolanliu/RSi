# Third-party components

`third_party/wan22_vae/vae2_2.py` and its initializer were copied without computation changes from the VAE implementation used by the original experiment. The source credits **The Alibaba Wan Team Authors (2024–2025)** and originates from [Wan2.2](https://github.com/Wan-Video/Wan2.2). The accompanying [Apache License 2.0](third_party/wan22_vae/LICENSE.txt) is retained. The local integration uses this component only as a frozen observation encoder.

The optional VAE checkpoint comes from [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B). It is not stored in this repository. The exact experiment checksum and download instructions are in [the usage guide](docs/USAGE.md).

The offline `compile/` model is the local project's CompILE-inspired PyTorch implementation and includes modifications for continuous robotic trajectories and visual features. It should not be described as the original paper authors' released code. The offline segmentation idea is discussed in [CompILE, ICML 2019](https://proceedings.mlr.press/v97/kipf19a.html). V11's causal online stage updates and OOD calibration are subsequent project components, not claims from that paper.
