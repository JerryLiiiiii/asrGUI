from __future__ import annotations

import logging
import os
import resource
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from asrgui.m0.models import model_metadata, model_paths
from asrgui.m0.transcript import jsonable, normalize_model_result, write_json_atomic


ProgressCallback = Callable[[dict[str, Any]], None]


def _emit(callback: ProgressCallback | None, **event: Any) -> None:
    if callback:
        callback(event)


def load_funasr_model(
    models_root: Path,
    *,
    device: str = "cpu",
    ncpu: int = 4,
    progress: ProgressCallback | None = None,
) -> tuple[Any, float]:
    started = time.perf_counter()
    _emit(progress, stage="model_loading", message="正在加载本地 FunASR 模型")

    import torch

    torch.set_num_threads(max(1, ncpu))
    from funasr import AutoModel

    paths = model_paths(models_root)
    model = AutoModel(
        model=str(paths.asr),
        vad_model=str(paths.vad),
        punc_model=str(paths.punctuation),
        spk_model=str(paths.speaker),
        device=device,
        ncpu=max(1, ncpu),
        disable_update=True,
        disable_pbar=True,
        log_level="WARNING",
        vad_kwargs={"max_single_segment_time": 30_000},
        spk_kwargs={"cb_kwargs": {"merge_thr": 0.78}},
    )
    elapsed = time.perf_counter() - started
    _emit(
        progress,
        stage="model_loaded",
        message="本地模型加载完成",
        elapsed_seconds=round(elapsed, 3),
    )
    return model, elapsed


def run_transcription(
    *,
    source_audio: Path,
    analysis_audio: Path,
    source_probe: dict[str, Any],
    source_sha256: str,
    models_root: Path,
    output_dir: Path,
    device: str = "cpu",
    ncpu: int = 4,
    batch_size_s: int = 60,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model, load_seconds = load_funasr_model(
        models_root, device=device, ncpu=ncpu, progress=progress
    )

    effective_settings = {
        "preset": "balanced",
        "language": "zh",
        "timestamp_granularity": "token",
        "device": device,
        "ncpu": ncpu,
        "batch_size_s": batch_size_s,
        "vad": {"max_single_segment_ms": 30_000},
        "speaker": {"min_count": 1, "max_count": 10},
    }
    _emit(progress, stage="inference", percent=0, message="正在执行 VAD/ASR/标点/说话人")
    started = time.perf_counter()
    result = model.generate(
        input=str(analysis_audio),
        batch_size_s=batch_size_s,
        sentence_timestamp=True,
        return_spk_res=True,
        return_spk_center=True,
        disable_pbar=True,
        merge_vad=False,
        use_itn=True,
    )
    inference_seconds = time.perf_counter() - started
    if not result:
        raise RuntimeError("FunASR returned no result")
    raw_result = result[0]
    write_json_atomic(output_dir / "model_result.json", raw_result)

    meeting_id = f"mtg_{uuid.uuid4().hex}"
    transcript_id = f"tr_{uuid.uuid4().hex}"
    audio_payload = {
        "source_file": source_audio.name,
        "analysis_file": analysis_audio.relative_to(output_dir).as_posix(),
        "duration_ms": source_probe["duration_ms"],
        "sha256": source_sha256,
        "sample_rate": 16000,
        "channels": 1,
    }
    normalized = normalize_model_result(
        raw_result,
        transcript_id=transcript_id,
        meeting_id=meeting_id,
        audio=audio_payload,
        models=model_metadata(),
        effective_settings=effective_settings,
    )
    transcript_path = output_dir / "transcript" / "raw.json"
    write_json_atomic(transcript_path, normalized)

    speaker_payload = {
        "model": model_metadata()["speaker"],
        "speaker_ids": [item["id"] for item in normalized["speakers"]],
        "centers": jsonable(raw_result.get("spk_embedding_center")),
    }
    write_json_atomic(output_dir / "speakers.json", speaker_payload)
    _emit(
        progress,
        stage="completed",
        percent=100,
        message="转写与说话人聚类完成",
        artifact=str(transcript_path),
    )

    metadata = {
        "model_load_seconds": round(load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "total_seconds": round(load_seconds + inference_seconds, 3),
        "audio_duration_seconds": round(source_probe["duration_ms"] / 1000, 3),
        "realtime_factor": round(
            inference_seconds / max(source_probe["duration_ms"] / 1000, 0.001), 4
        ),
        "process_id": os.getpid(),
        "peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_rss_bytes": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin"
            else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "result": str(transcript_path),
    }
    write_json_atomic(output_dir / "run.json", metadata)
    return {"transcript": normalized, "run": metadata, "raw_result": raw_result}
