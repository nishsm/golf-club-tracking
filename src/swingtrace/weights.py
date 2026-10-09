"""Find the model weights: a local path if given, otherwise download from the Hugging Face Hub."""

from __future__ import annotations

import os
from pathlib import Path

HF_REPO = os.environ.get("SWINGTRACE_HF_REPO", "nishsm/golf-club-detection-swingtrace")
HF_FILE = "swingtrace-yolo11m.pt"
LOCAL_CANDIDATES = ["weights/best.pt", "weights/swingtrace-yolo11m.pt"]


def resolve_weights(weights: str | None = None) -> str:
    if weights:
        if not Path(weights).exists():
            raise SystemExit(f"weights not found: {weights}")
        return weights
    for c in LOCAL_CANDIDATES:
        if Path(c).exists():
            return c
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise SystemExit("pip install huggingface_hub, or pass --weights path/to/best.pt") from e
    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
