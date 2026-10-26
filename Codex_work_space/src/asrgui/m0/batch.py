from __future__ import annotations

import json
import math
import resource
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from asrgui.m0.engine import ProgressCallback, _emit, load_funasr_model
from asrgui.m0.models import model_metadata
from asrgui.m0.transcript import jsonable, normalize_model_result, write_json_atomic


CancelCheck = Callable[[], bool]


def _chunk_path(checkpoint_dir: Path, index: int) -> Path:
    return checkpoint_dir / f"chunk_{index:04d}.json"


def _read_cached_chunk(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    payload.setdefault("sentence_info", [])
    payload.setdefault("timestamp", [])
    payload.setdefault("text", "")
    if not isinstance(payload["sentence_info"], list):
        return None
    return payload


def _load_or_transcribe_chunk(
    model: Any,
    audio_path: Path,
    checkpoint_dir: Path,
    index: int,
    chunk_seconds: int,
    batch_size_s: int,
) -> tuple[dict[str, Any], bool]:
    path = _chunk_path(checkpoint_dir, index)
    cached = _read_cached_chunk(path)
    if cached is not None:
        return cached, True

    with sf.SoundFile(str(audio_path)) as stream:
        if stream.samplerate != 16000 or stream.channels != 1:
            raise ValueError("Analysis audio must be mono 16 kHz PCM")
        stream.seek(index * chunk_seconds * stream.samplerate)
        audio = stream.read(
            frames=chunk_seconds * stream.samplerate,
            dtype="float32",
            always_2d=False,
        )
    if not len(audio):
        payload = {"text": "", "timestamp": [], "sentence_info": []}
    else:
        started = time.perf_counter()
        results = model.generate(
            input=audio,
            fs=16000,
            batch_size_s=batch_size_s,
            sentence_timestamp=True,
            return_spk_res=True,
            return_spk_center=True,
            disable_pbar=True,
            merge_vad=False,
            use_itn=True,
        )
        if not results:
            raise RuntimeError(f"FunASR returned no result for chunk {index}")
        payload = jsonable(results[0])
        payload.setdefault("sentence_info", [])
        payload.setdefault("timestamp", [])
        payload.setdefault("text", "")
        payload["_chunk_offset_ms"] = index * chunk_seconds * 1000
        payload["_chunk_samples"] = len(audio)
        payload["_processing_seconds"] = round(time.perf_counter() - started, 3)
    write_json_atomic(path, payload)
    return payload, False


def _cluster_speaker_centers(
    chunks: list[dict[str, Any]], merge_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers: list[list[float]] = []
    center_keys: list[tuple[int, int]] = []
    for chunk_index, chunk in enumerate(chunks):
        for speaker_index, center in enumerate(chunk.get("spk_embedding_center") or []):
            centers.append(center)
            center_keys.append((chunk_index, speaker_index))
    if not centers:
        raise RuntimeError("No speaker embeddings returned by FunASR")

    matrix = np.asarray(centers, dtype=np.float32)
    if len(matrix) == 1:
        labels = np.zeros(1, dtype=int)
    else:
        from sklearn.cluster import AgglomerativeClustering

        labels = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=1 - merge_threshold,
        ).fit_predict(matrix)

    canonical: dict[int, int] = {}
    for label in labels:
        canonical.setdefault(int(label), len(canonical))
    labels = np.asarray([canonical[int(label)] for label in labels], dtype=int)

    global_centers = np.stack(
        [matrix[labels == label].mean(axis=0) for label in range(labels.max() + 1)]
    )
    max_local_speakers = max(
        len(chunk.get("spk_embedding_center") or []) for chunk in chunks
    )
    mapping = np.zeros((len(chunks), max_local_speakers), dtype=int)
    for label, (chunk_index, speaker_index) in zip(labels, center_keys):
        mapping[chunk_index, speaker_index] = int(label)
    return labels, mapping, global_centers


def _merge_chunks(
    chunks: list[dict[str, Any]],
    speaker_mapping: np.ndarray,
    *,
    analysis_audio: Path,
    source_audio: Path,
    source_probe: dict[str, Any],
    source_sha256: str,
    effective_settings: dict[str, Any],
    meeting_id: str,
    transcript_id: str,
) -> dict[str, Any]:
    sentence_info = []
    text_parts = []
    timestamps = []
    for chunk_index, chunk in enumerate(chunks):
        offset = chunk_index * int(effective_settings["chunk_seconds"]) * 1000
        text_parts.append(str(chunk.get("text") or ""))
        for sentence in chunk.get("sentence_info") or []:
            item = dict(sentence)
            item["start"] = int(item.get("start", 0)) + offset
            item["end"] = int(item.get("end", item.get("start", 0))) + offset
            item["timestamp"] = [
                [int(pair[0]) + offset, int(pair[1]) + offset]
                for pair in item.get("timestamp") or []
                if isinstance(pair, (list, tuple)) and len(pair) >= 2
            ]
            local_speaker = int(item.get("spk", 0))
            if local_speaker >= speaker_mapping.shape[1]:
                raise ValueError(
                    f"Chunk {chunk_index} references unknown speaker {local_speaker}"
                )
            item["spk"] = int(speaker_mapping[chunk_index, local_speaker])
            sentence_info.append(item)
        for pair in chunk.get("timestamp") or []:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                timestamps.append([int(pair[0]) + offset, int(pair[1]) + offset])

    model_result = {
        "text": "".join(text_parts),
        "timestamp": timestamps,
        "sentence_info": sentence_info,
    }
    return normalize_model_result(
        model_result,
        transcript_id=transcript_id,
        meeting_id=meeting_id,
        audio={
            "source_file": source_audio.name,
            "analysis_file": "audio/asr_16k.wav",
            "duration_ms": source_probe["duration_ms"],
            "sha256": source_sha256,
            "sample_rate": 16000,
            "channels": 1,
        },
        models=model_metadata(),
        effective_settings=effective_settings,
    )


def run_batch_transcription(
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
    chunk_seconds: int = 300,
    speaker_merge_threshold: float = 0.78,
    should_cancel: CancelCheck | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if chunk_seconds < 30:
        raise ValueError("chunk_seconds must be at least 30")
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    duration_ms = int(source_probe["duration_ms"])
    chunk_count = max(1, math.ceil(duration_ms / (chunk_seconds * 1000)))
    previous_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    all_cached = all(
        _read_cached_chunk(_chunk_path(checkpoint_dir, index)) is not None
        for index in range(chunk_count)
    )
    pipeline_started = time.perf_counter()
    if all_cached:
        model = None
        load_seconds = 0.0
    else:
        model, load_seconds = load_funasr_model(
            models_root, device=device, ncpu=ncpu, progress=progress
        )
    processing_started = time.perf_counter()
    chunks: list[dict[str, Any]] = []
    resumed_chunks = 0
    for index in range(chunk_count):
        if should_cancel and should_cancel():
            state = "cancelled"
            break
        _emit(
            progress,
            stage="chunk",
            chunk_index=index + 1,
            chunk_count=chunk_count,
            percent=round(index / chunk_count * 100),
            message=f"正在处理分块 {index + 1}/{chunk_count}",
        )
        chunk, resumed = _load_or_transcribe_chunk(
            model,
            analysis_audio,
            checkpoint_dir,
            index,
            chunk_seconds,
            batch_size_s,
        )
        chunks.append(chunk)
        resumed_chunks += int(resumed)
        write_json_atomic(
            output_dir / "batch_checkpoint.json",
            {
                "schema_version": 1,
                "state": "running",
                "completed_chunks": index + 1,
                "chunk_count": chunk_count,
                "chunk_seconds": chunk_seconds,
                "resumed_chunks": resumed_chunks,
            },
        )
        _emit(
            progress,
            stage="chunk_completed",
            chunk_index=index + 1,
            chunk_count=chunk_count,
            percent=round((index + 1) / chunk_count * 100),
            resumed=resumed,
            message=f"分块 {index + 1}/{chunk_count} 已提交",
        )
    else:
        state = "completed"

    if state != "completed":
        cancelled_inference_seconds = time.perf_counter() - processing_started
        cancelled_chunk_seconds = sum(
            float(chunk.get("_processing_seconds", 0) or 0) for chunk in chunks
        )
        current_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json_atomic(
            output_dir / "batch_checkpoint.json",
            {
                "schema_version": 1,
                "state": "cancelled",
                "completed_chunks": len(chunks),
                "chunk_count": chunk_count,
                "chunk_seconds": chunk_seconds,
                "resumed_chunks": resumed_chunks,
            },
        )
        write_json_atomic(
            output_dir / "cancel.json",
            {
                "state": "cancelled",
                "completed_chunks": len(chunks),
                "chunk_count": chunk_count,
                "session_inference_seconds": round(cancelled_inference_seconds, 3),
                "chunk_processing_seconds": round(cancelled_chunk_seconds, 3),
                "peak_rss_bytes": (
                    current_peak if sys.platform == "darwin" else current_peak * 1024
                ),
            },
        )
        _emit(
            progress,
            stage="cancelled",
            percent=round(len(chunks) / chunk_count * 100),
            message="已在安全分块边界取消",
        )
        return {"state": state, "completed_chunks": len(chunks)}

    _, speaker_mapping, global_centers = _cluster_speaker_centers(
        chunks, speaker_merge_threshold
    )
    effective_settings = {
        "preset": "balanced",
        "language": "zh",
        "timestamp_granularity": "token",
        "device": device,
        "ncpu": ncpu,
        "batch_size_s": batch_size_s,
        "chunk_seconds": chunk_seconds,
        "speaker": {
            "cluster": "campplus-agglomerative-average",
            "cross_chunk_merge_threshold": speaker_merge_threshold,
        },
    }
    meeting_id = f"mtg_{uuid.uuid4().hex}"
    transcript_id = f"tr_{uuid.uuid4().hex}"
    normalized = _merge_chunks(
        chunks,
        speaker_mapping,
        analysis_audio=analysis_audio,
        source_audio=source_audio,
        source_probe=source_probe,
        source_sha256=source_sha256,
        effective_settings=effective_settings,
        meeting_id=meeting_id,
        transcript_id=transcript_id,
    )
    transcript_path = output_dir / "transcript" / "raw.json"
    write_json_atomic(transcript_path, normalized)
    write_json_atomic(
        output_dir / "speakers.json",
        {
            "model": model_metadata()["speaker"],
            "speaker_ids": [item["id"] for item in normalized["speakers"]],
            "centers": jsonable(global_centers),
        },
    )

    inference_seconds = time.perf_counter() - processing_started
    chunk_processing_seconds = sum(
        float(chunk.get("_processing_seconds", 0) or 0) for chunk in chunks
    )
    session_seconds = time.perf_counter() - pipeline_started
    current_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = current_peak if sys.platform == "darwin" else current_peak * 1024
    run_metadata = {
        "state": state,
        "model_load_seconds": round(load_seconds, 3),
        "session_inference_seconds": round(inference_seconds, 3),
        "chunk_processing_seconds": round(chunk_processing_seconds, 3),
        "session_total_seconds": round(session_seconds, 3),
        "cold_total_seconds": round(load_seconds + chunk_processing_seconds, 3),
        "audio_duration_seconds": round(duration_ms / 1000, 3),
        "chunk_seconds": chunk_seconds,
        "chunk_count": chunk_count,
        "resumed_chunks": resumed_chunks,
        "realtime_factor": round(
            chunk_processing_seconds / max(duration_ms / 1000, 0.001), 4
        ),
        "peak_rss_bytes": peak_bytes,
        "peak_rss_before_inference_bytes": (
            previous_peak if sys.platform == "darwin" else previous_peak * 1024
        ),
        "speaker_count": len(normalized["speakers"]),
        "segment_count": len(normalized["segments"]),
        "token_count": sum(len(item["tokens"]) for item in normalized["segments"]),
        "result": str(transcript_path),
    }
    write_json_atomic(output_dir / "run.json", run_metadata)
    write_json_atomic(
        output_dir / "batch_checkpoint.json",
        {
            "schema_version": 1,
            "state": "completed",
            "completed_chunks": chunk_count,
            "chunk_count": chunk_count,
            "chunk_seconds": chunk_seconds,
            "resumed_chunks": resumed_chunks,
        },
    )
    _emit(
        progress,
        stage="completed",
        percent=100,
        message="长音频分块转写与合并完成",
        artifact=str(transcript_path),
    )
    return {"state": state, "transcript": normalized, "run": run_metadata}
