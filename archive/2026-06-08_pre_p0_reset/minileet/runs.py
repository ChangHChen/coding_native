from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = Path(os.environ.get("MINILEET_RUNS_ROOT", PROJECT_ROOT / "runs"))


def timestamped_run_dir(root: Path, prefix: str = "run", seed: int | None = None) -> Path:
    root = _resolve_root(root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{prefix}_{stamp}"
    if seed is not None:
        suffix += f"_seed{seed}"
    candidate = root / suffix
    index = 1
    while candidate.exists():
        candidate = root / f"{suffix}_{index:02d}"
        index += 1
    return candidate


def create_timestamped_run_dir(root: Path, prefix: str = "run", seed: int | None = None) -> Path:
    root = _resolve_root(root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{prefix}_{stamp}"
    if seed is not None:
        suffix += f"_seed{seed}"
    for index in range(1000):
        name = suffix if index == 0 else f"{suffix}_{index:02d}"
        candidate = root / name
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"could not create unique run directory under {root}")


def ensure_new_run_dir(run_dir: Path) -> Path:
    run_dir = _resolve_root(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory already exists and is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_root(kind: str, dataset: str, experiment: str, root: Path = RUNS_ROOT) -> Path:
    if kind not in {"train", "eval", "analysis", "smoke", "cache_build", "sweep"}:
        raise ValueError(f"unknown run kind {kind}")
    return _resolve_root(root) / kind / _slug(dataset) / _slug(experiment)


def infer_dataset_slug(*values: object) -> str:
    stems = " ".join(Path(str(value)).stem.lower() for value in values if value)
    text = stems
    if "semantic_hard" in text and "search_v3" in text:
        return "semantic_hard_search_v3"
    if "semantic_hard" in text and "search_v2" in text:
        return "semantic_hard_search_v2"
    if "semantic_hard" in text:
        return "semantic_hard_v1"
    if "scaled" in text and "50x" in text:
        return "scaled_50x"
    if text.startswith("std_") or "_std_" in text or "standard" in text:
        return "standard"
    if "hard_split" in text or text.startswith("hard_") or "_hard_" in text:
        return "hard_split"
    return "custom"


def _slug(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value.strip().lower())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    if not cleaned:
        raise ValueError("run path component cannot be empty")
    return cleaned


def _resolve_root(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
