"""Configuration loading, device selection, and run control.

PATH CONVENTION (important):
    This file lives at src/utils/config.py.
    _ROOT resolves 3 levels up:  config.py → utils/ → src/ → <project_root>/

    The project uses PYTHONPATH=src so all imports are bare:
        from utils.config import ...   ← works when PYTHONPATH=src
        from dynamics import ...       ← works when PYTHONPATH=src

    Do NOT change these to relative imports (from .config import ...) unless
    you also restructure src/ as a proper package and update every import site.
    The bare-import convention is intentional and load-bearing for the test
    suite and main.py.
"""

import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import yaml

# Project root: src/utils/config.py → .parent = utils/ → .parent = src/ → .parent = root/
_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_device(device_str: str) -> str:
    """Resolve a device string to a concrete torch device name.

    "auto" → "cuda" if torch.cuda.is_available() else "cpu"
    "cuda" / "cpu" → passed through unchanged.
    """
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def get_device() -> torch.device:
    """Return the auto-selected torch.device.

    Prefer reading device from cfg["device"] (set by load_scenario) in new
    code. This function is kept for backward compatibility.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path) -> dict:
    """Load a YAML file. Relative paths resolve from project root."""
    p = Path(path)
    if not p.is_absolute():
        p = _ROOT / p
    with open(p) as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates base)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_scenario(scenario_path) -> tuple:
    """Load scenario config merged with defaults.

    Device is resolved once here so callers never need to call get_device().

    Returns:
        (cfg, dyn_cfg) where:
          cfg["device"] is a concrete string ("cuda" or "cpu")
          dyn_cfg is the dynamics sub-config dict
    """
    cfg = load_config(scenario_path)
    defaults = load_config("configs/defaults.yaml")
    merged = deep_merge(defaults, cfg)
    merged["device"] = resolve_device(merged.get("device", "auto"))
    dyn_cfg = load_config(merged["dynamics"])
    return merged, dyn_cfg


# ── Skip / run context manager ───────────────────────────────────────

class _SkipWith(Exception):
    pass


@contextmanager
def skip_run(flag: str, label: str):
    """Context manager to conditionally skip or run a code block.

    Usage:
        with skip_run("run", "My Experiment") as check, check():
            do_stuff()

        with skip_run("skip", "My Experiment") as check, check():
            do_stuff()   # never executed
    """
    @contextmanager
    def check():
        if flag == "skip":
            sys.stderr.write(f"\x1b[90m  skip │ {label}\x1b[0m\n")
            raise _SkipWith()
        else:
            sys.stdout.write(f"\x1b[1;32m  run  │ {label}\x1b[0m\n")
            yield

    try:
        yield check
    except _SkipWith:
        pass
