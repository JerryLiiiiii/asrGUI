from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from asrgui.m0.models import model_metadata, model_paths


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Cannot compare a zero speaker embedding")
    return float(np.dot(left, right) / (left_norm * right_norm))


def extract_segment_embeddings(
    *,
    analysis_audio: Path,
    transcript: dict[str, Any],
    models_root: Path,
    max_per_speaker: int = 3,
    min_duration_ms: int = 1200,
    device: str = "cpu",
    ncpu: int = 4,
) -> dict[str, Any]:
    selected = []
    counts: dict[str, int] = {}
    for segment in transcript["segments"]:
        speaker_id = segment["speaker_id"]
        if counts.get(speaker_id, 0) >= max_per_speaker:
            continue
        duration = segment["end_ms"] - segment["start_ms"]
        if duration < min_duration_ms:
            continue
        selected.append(segment)
        counts[speaker_id] = counts.get(speaker_id, 0) + 1
    if not selected:
        raise RuntimeError("No sufficiently long confirmed segments for embedding check")

    audio = []
    with sf.SoundFile(str(analysis_audio)) as stream:
        if stream.samplerate != 16000 or stream.channels != 1:
            raise ValueError("Analysis audio must be mono 16 kHz PCM")
        for segment in selected:
            start = round(segment["start_ms"] * stream.samplerate / 1000)
            frames = round(
                (segment["end_ms"] - segment["start_ms"]) * stream.samplerate / 1000
            )
            stream.seek(start)
            audio.append(stream.read(frames, dtype="float32", always_2d=False))

    from funasr import AutoModel

    started = time.perf_counter()
    model = AutoModel(
        model=str(model_paths(models_root).speaker),
        device=device,
        ncpu=max(1, ncpu),
        disable_update=True,
        disable_pbar=True,
        log_level="ERROR",
    )
    results = model.generate(input=audio, fs=16000, disable_pbar=True)
    embeddings = [
        item["spk_embedding"].detach().cpu().numpy().reshape(-1) for item in results
    ]
    if len(embeddings) != len(selected):
        raise RuntimeError(
            f"Expected {len(selected)} embeddings, received {len(embeddings)}"
        )

    similarities = []
    for left_index, left in enumerate(selected):
        for right_index in range(left_index + 1, len(selected)):
            right = selected[right_index]
            similarities.append(
                {
                    "left_speaker": left["speaker_id"],
                    "right_speaker": right["speaker_id"],
                    "same_speaker": left["speaker_id"] == right["speaker_id"],
                    "cosine": round(
                        cosine_similarity(embeddings[left_index], embeddings[right_index]),
                        6,
                    ),
                }
            )
    same = [item["cosine"] for item in similarities if item["same_speaker"]]
    different = [item["cosine"] for item in similarities if not item["same_speaker"]]
    return {
        "model": model_metadata()["speaker"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "segments": [
            {
                "segment_id": segment["id"],
                "speaker_id": segment["speaker_id"],
                "start_ms": segment["start_ms"],
                "end_ms": segment["end_ms"],
                "embedding_dimension": int(embedding.shape[0]),
            }
            for segment, embedding in zip(selected, embeddings, strict=True)
        ],
        "similarities": similarities,
        "summary": {
            "same_speaker_mean_cosine": round(float(np.mean(same)), 6) if same else None,
            "different_speaker_mean_cosine": (
                round(float(np.mean(different)), 6) if different else None
            ),
            "same_speaker_min_cosine": round(min(same), 6) if same else None,
            "different_speaker_max_cosine": (
                round(max(different), 6) if different else None
            ),
        },
    }
