"""Shared helpers for the deep-learning-systems practical notebooks.

Design goals: tiny, dependency-light, and *loud* — every helper logs to stdout so the
GitHub Actions run log and the executed notebook both record what happened.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import platform
import random
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

# NOTE: do not force a backend here. Inside a Jupyter kernel (local or CI) the inline backend is already
# headless-safe and is what embeds figures into the executed notebook.
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:                              # pragma: no cover
    torch = nn = F = None
    HAS_TORCH = False

__all__ = ["np", "plt", "torch", "nn", "F", "HAS_TORCH", "log", "seed_everything", "env_report",
           "show", "check", "timer", "METRICS", "save_metrics", "FIG_DIR", "set_plot_style",
           "count_params", "DEVICE"]

# ----------------------------------------------------------------------------- logging
log = logging.getLogger("dlsys")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(logging.INFO)
    log.propagate = False

FIG_DIR = Path(os.environ.get("NB_FIG_DIR", "figures"))
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS: dict = {}
DEVICE = "cpu"   # the course is designed to run on free CPU-only runners


def set_plot_style():
    plt.rcParams.update({"figure.dpi": 100, "figure.figsize": (6.5, 4), "axes.grid": True,
                         "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10})


set_plot_style()


def seed_everything(seed: int = 42) -> int:
    """Seed python, numpy and torch. (Fixed seed != bit-identical results across hardware.)"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    log.info("seed set to %d", seed)
    return seed


def _mem_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
    except Exception:
        return None


def env_report():
    """Print the reproducibility record recommended by the wiki's research standards."""
    import scipy, sklearn  # noqa
    info = {
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu_count": os.cpu_count(), "ram_gb": _mem_gb(),
        "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "matplotlib": matplotlib.__version__, "torch": torch.__version__ if HAS_TORCH else None,
        "cuda_available": bool(HAS_TORCH and torch.cuda.is_available()),
        "github_actions": bool(os.environ.get("GITHUB_ACTIONS")),
        "run_id": os.environ.get("GITHUB_RUN_ID"), "commit": os.environ.get("GITHUB_SHA"),
    }
    for k, v in info.items():
        print(f"{k:>16}: {v}")
    METRICS["environment"] = info
    return info


def show(name: str):
    """Save the current figure to figures/<name>.png and display it."""
    path = FIG_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    log.info("figure saved -> %s", path)
    plt.show()


def check(cond, msg: str):
    """Assertion that logs a green tick, so passing checks are visible in the output."""
    if not bool(cond):
        log.error("CHECK FAILED: %s", msg)
        raise AssertionError(msg)
    print(f"  ✅ {msg}")


@contextlib.contextmanager
def timer(label: str):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    METRICS.setdefault("timings_s", {})[label] = round(dt, 3)
    log.info("%s took %.2fs", label, dt)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def save_metrics(num: str, metrics: dict, path: str = "metrics.json"):
    metrics = dict(metrics)
    metrics["notebook"] = num

    def _clean(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=_clean)
    log.info("metrics saved -> %s", path)
