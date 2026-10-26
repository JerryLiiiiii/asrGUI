from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "asr": {
        "alias": "paraformer-zh",
        "directory": "paraformer-zh",
        "source": (
            "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-"
            "vocab8404-pytorch"
        ),
        "revision": "master@2026-10-25",
        "license": "Apache-2.0",
        "required": ["configuration.json", "config.yaml", "model.pt"],
    },
    "vad": {
        "alias": "fsmn-vad",
        "directory": "fsmn-vad",
        "source": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "revision": "master@2026-10-25",
        "license": "Apache-2.0",
        "required": ["configuration.json", "config.yaml", "model.pt"],
    },
    "punctuation": {
        "alias": "ct-punc",
        "directory": "ct-punc",
        "source": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        "revision": "master@2026-10-25",
        "license": "Apache-2.0",
        "required": ["configuration.json", "config.yaml", "model.pt"],
    },
    "speaker": {
        "alias": "cam++",
        "directory": "campplus",
        "source": "iic/speech_campplus_sv_zh-cn_16k-common",
        "revision": "master@2026-10-25",
        "license": "Apache-2.0",
        "required": ["configuration.json", "config.yaml", "campplus_cn_common.bin"],
    },
}


@dataclass(frozen=True)
class ModelPaths:
    asr: Path
    vad: Path
    punctuation: Path
    speaker: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model_manifest(models_root: Path) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for role, spec in MODEL_SPECS.items():
        model_dir = models_root / spec["directory"]
        missing = [name for name in spec["required"] if not (model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{model_dir} missing required files: {missing}")

        files = []
        for path in sorted(p for p in model_dir.rglob("*") if p.is_file()):
            if ".msc" in path.parts:
                continue
            files.append(
                {
                    "path": path.relative_to(model_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        models[role] = {
            **spec,
            "local_directory": spec["directory"],
            "files": files,
            "size_bytes": sum(item["size_bytes"] for item in files),
        }
    return {
        "schema_version": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "models": models,
    }


def write_model_manifest(models_root: Path, output: Path) -> dict[str, Any]:
    manifest = build_model_manifest(models_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(output)
    return manifest


def model_paths(models_root: Path) -> ModelPaths:
    paths = {
        role: models_root / spec["directory"] for role, spec in MODEL_SPECS.items()
    }
    for role, path in paths.items():
        missing = [
            name
            for name in MODEL_SPECS[role]["required"]
            if not (path / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Model {role} at {path} is missing {missing}")
    return ModelPaths(**paths)


def model_metadata() -> dict[str, dict[str, Any]]:
    return {
        role: {
            "id": spec["alias"],
            "source": spec["source"],
            "revision": spec["revision"],
            "license": spec["license"],
        }
        for role, spec in MODEL_SPECS.items()
    }
