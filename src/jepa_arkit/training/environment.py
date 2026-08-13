from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from jepa_arkit.io import dump_json, stable_hash


def environment_report() -> dict[str, object]:
    cuda = torch.cuda.is_available()
    gpu: dict[str, object] | None = None
    if cuda:
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "capability": [properties.major, properties.minor],
        }
    git_revision = "unavailable"
    try:
        git_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    value: dict[str, object] = {
        "created_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda,
        "gpu": gpu,
        "numpy": np.__version__,
        "git_revision": git_revision,
        "process_id": os.getpid(),
    }
    value["environment_hash"] = f"sha256:{stable_hash(value)}"
    return value


def write_environment_report(path: str | Path) -> dict[str, object]:
    report = environment_report()
    dump_json(path, report)
    return report
