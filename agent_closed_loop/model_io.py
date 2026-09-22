"""Inference-only checkpoint loading: no training/data/report imports."""
from pathlib import Path

import torch


def load(path):
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def inference_precision():
    # Keep the frozen CPU/CUDA online decision boundary reproducible.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
