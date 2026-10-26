from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


SURFACE_TOKEN_RE = re.compile(
    r"\s+"
    r"|[A-Za-z]+(?:['’\-][A-Za-z]+)*"
    r"|\d+(?:[.,:/\-]\d+)*"
    r"|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
    r"|.",
    re.DOTALL,
)
PUNCTUATION_RE = re.compile(r"^[\s\W_]+$", re.UNICODE)


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    return repr(value)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def surface_tokens(text: str) -> list[dict[str, Any]]:
    tokens = []
    for match in SURFACE_TOKEN_RE.finditer(text):
        token = match.group(0)
        tokens.append(
            {
                "text": token,
                "char_start": match.start(),
                "char_end": match.end(),
                "clickable": not bool(PUNCTUATION_RE.fullmatch(token)),
            }
        )
    return tokens


def _valid_timestamp(item: Any) -> tuple[int, int] | None:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None
    try:
        start, end = int(round(float(item[0]))), int(round(float(item[1])))
    except (TypeError, ValueError):
        return None
    if start < 0 or end < start:
        return None
    return start, end


def _interpolate_offsets(
    start_ms: int,
    end_ms: int,
    start_offset: int,
    end_offset: int,
    total_units: int,
) -> tuple[int, int]:
    duration = max(0, end_ms - start_ms)
    token_start = start_ms + round(duration * start_offset / max(total_units, 1))
    token_end = start_ms + round(duration * end_offset / max(total_units, 1))
    return token_start, max(token_start, token_end)


def timestamped_tokens(
    text: str,
    raw_timestamps: Iterable[Any] | None,
    start_ms: int,
    end_ms: int,
) -> tuple[list[dict[str, Any]], str]:
    tokens = surface_tokens(text)
    timestamps = [
        parsed
        for item in (raw_timestamps or [])
        if (parsed := _valid_timestamp(item)) is not None
    ]
    timestamp_char_count = sum(len(token["text"]) for token in tokens if token["clickable"])
    exact_alignment = len(timestamps) == timestamp_char_count
    char_cursor = 0
    previous_end = start_ms
    for token in tokens:
        if not token["clickable"]:
            token["start_ms"] = previous_end
            token["end_ms"] = previous_end
            token["time_precision"] = "segment"
            continue
        count = len(token["text"])
        char_end = char_cursor + count
        if exact_alignment:
            selected = timestamps[char_cursor:char_end]
            token["start_ms"] = selected[0][0]
            token["end_ms"] = selected[-1][1]
            token["time_precision"] = "token"
        else:
            token["start_ms"], token["end_ms"] = _interpolate_offsets(
                start_ms,
                end_ms,
                char_cursor,
                char_end,
                timestamp_char_count,
            )
            token["time_precision"] = "interpolated" if end_ms > start_ms else "segment"
        token["start_ms"] = max(previous_end, min(token["start_ms"], end_ms))
        token["end_ms"] = max(token["start_ms"], min(token["end_ms"], end_ms))
        if token["end_ms"] == token["start_ms"] and end_ms > token["start_ms"]:
            token["end_ms"] = min(end_ms, token["start_ms"] + 1)
        previous_end = token["end_ms"]
        char_cursor = char_end

    precision = "token" if timestamps and exact_alignment else (
        "interpolated" if timestamps or end_ms > start_ms else "segment"
    )

    for token in tokens:
        token.pop("char_start")
        token.pop("char_end")
    return tokens, precision


def normalize_model_result(
    model_result: dict[str, Any],
    *,
    transcript_id: str,
    meeting_id: str,
    audio: dict[str, Any],
    models: dict[str, Any],
    effective_settings: dict[str, Any],
) -> dict[str, Any]:
    sentence_info = model_result.get("sentence_info") or []
    segments = []
    dropped_segments = 0
    speaker_ids: set[int] = set()
    precisions = {"token": 0, "interpolated": 0, "segment": 0}
    token_id = 0

    for segment_index, sentence in enumerate(sentence_info, start=1):
        text = str(sentence.get("text") or sentence.get("sentence") or "")
        start_ms = max(0, int(round(sentence.get("start", 0))))
        end_ms = max(start_ms, int(round(sentence.get("end", start_ms))))
        spk = int(sentence.get("spk", 0))
        speaker_ids.add(spk)
        tokens, precision = timestamped_tokens(
            text, sentence.get("timestamp"), start_ms, end_ms
        )
        if not any(token["clickable"] for token in tokens):
            dropped_segments += 1
            continue
        for token in tokens:
            token_id += 1
            token["id"] = f"tok_{token_id:06d}"
        precisions[precision] += 1
        segments.append(
            {
                "id": f"seg_{segment_index:04d}",
                "speaker_id": f"spk_{spk + 1}",
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": text,
                "tokens": tokens,
            }
        )

    speakers = [
        {
            "id": f"spk_{speaker + 1}",
            "temporary_label": f"说话人 {speaker + 1}",
            "display_name": None,
            "profile_id": None,
            "match_confidence": None,
        }
        for speaker in sorted(speaker_ids)
    ]
    return {
        "schema_version": "1.0",
        "transcript_id": transcript_id,
        "meeting_id": meeting_id,
        "kind": "raw",
        "created_at": datetime.now().astimezone().isoformat(),
        "software": {"app_version": "0.1.0", "pipeline_version": "m0-prototype"},
        "audio": audio,
        "models": models,
        "effective_settings": effective_settings,
        "speakers": speakers,
        "segments": segments,
        "diagnostics": {
            "sentence_count": len(segments),
            "token_precision_segments": precisions,
            "dropped_non_text_segments": dropped_segments,
            "model_text": model_result.get("text"),
            "model_timestamp_count": len(model_result.get("timestamp") or []),
        },
    }
