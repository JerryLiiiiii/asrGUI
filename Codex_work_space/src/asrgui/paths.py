from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspacePaths:
    project_root: Path
    data_root: Path
    runtime_root: Path
    models_root: Path
    cache_root: Path
    tmp_root: Path
    logs_root: Path
    results_root: Path

    def ensure(self) -> None:
        for path in (
            self.data_root,
            self.runtime_root,
            self.models_root,
            self.cache_root,
            self.tmp_root,
            self.logs_root,
            self.results_root,
        ):
            path.mkdir(parents=True, exist_ok=True)


def find_project_root() -> Path:
    override = os.environ.get("ASRGUI_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    if getattr(sys, "frozen", False):
        executable = Path(sys.executable).resolve()
        for parent in executable.parents:
            if parent.name == "AppFile":
                return parent.parent
            if (parent / "AppFile").is_dir():
                return parent
        return Path.cwd().resolve()
    return Path(__file__).resolve().parents[3]


def find_resource_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root).resolve()
    return Path(__file__).resolve().parent / "resources"


def workspace_paths() -> WorkspacePaths:
    project_root = find_project_root()
    data_root = Path(
        os.environ.get("ASRGUI_DATA_ROOT", project_root / "AppFile")
    ).expanduser().resolve()
    runtime_root = data_root / "runtime"
    models_root = Path(
        os.environ.get("ASRGUI_MODELS_ROOT", runtime_root / "models")
    ).expanduser().resolve()
    return WorkspacePaths(
        project_root=project_root,
        data_root=data_root,
        runtime_root=runtime_root,
        models_root=models_root,
        cache_root=runtime_root / "cache",
        tmp_root=runtime_root / "tmp",
        logs_root=runtime_root / "logs",
        results_root=runtime_root / "m0",
    )


def configure_process_environment() -> WorkspacePaths:
    """Pin every known framework/cache path inside the development data root."""
    paths = workspace_paths()
    paths.ensure()

    cache_paths = {
        "HF_HOME": paths.cache_root / "huggingface",
        "MODELSCOPE_CACHE": paths.cache_root / "modelscope",
        "TORCH_HOME": paths.cache_root / "torch",
        "XDG_CACHE_HOME": paths.cache_root / "xdg",
        "TMPDIR": paths.tmp_root,
        "PYTHONPYCACHEPREFIX": paths.cache_root / "pycache",
    }
    for name, path in cache_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(min(4, os.cpu_count() or 4)))
    return paths
