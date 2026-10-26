from __future__ import annotations

from collections import Counter
from typing import Any


class TranscriptValidationError(ValueError):
    pass


def validate_transcript(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    speaker_ids = {item.get("id") for item in data.get("speakers") or []}
    segments = data.get("segments") or []
    if not segments:
        errors.append("transcript has no segments")

    segment_ids = Counter()
    token_ids = Counter()
    precision = Counter()
    speaker_segments = Counter()
    last_segment_start = -1
    duration_ms = int((data.get("audio") or {}).get("duration_ms") or 0)

    for segment in segments:
        segment_id = segment.get("id")
        segment_ids[segment_id] += 1
        start_ms = segment.get("start_ms")
        end_ms = segment.get("end_ms")
        if not isinstance(start_ms, int) or not isinstance(end_ms, int):
            errors.append(f"{segment_id}: start/end must be integer milliseconds")
            continue
        if start_ms < 0 or end_ms < start_ms:
            errors.append(f"{segment_id}: invalid interval [{start_ms}, {end_ms})")
        if start_ms < last_segment_start:
            errors.append(f"{segment_id}: segments are not chronological")
        last_segment_start = start_ms
        if duration_ms and end_ms > duration_ms + 250:
            errors.append(f"{segment_id}: end exceeds audio duration")

        speaker_id = segment.get("speaker_id")
        if speaker_id not in speaker_ids:
            errors.append(f"{segment_id}: unknown speaker {speaker_id}")
        speaker_segments[speaker_id] += 1

        tokens = segment.get("tokens") or []
        if "".join(str(token.get("text", "")) for token in tokens) != segment.get("text"):
            errors.append(f"{segment_id}: token text does not reconstruct segment text")
        clickable_count = 0
        last_token_start = start_ms
        for token in tokens:
            token_id = token.get("id")
            token_ids[token_id] += 1
            token_start = token.get("start_ms")
            token_end = token.get("end_ms")
            token_precision = token.get("time_precision")
            precision[token_precision] += 1
            if not isinstance(token_start, int) or not isinstance(token_end, int):
                errors.append(f"{token_id}: start/end must be integer milliseconds")
                continue
            if token_start < start_ms or token_end > end_ms or token_end < token_start:
                errors.append(
                    f"{token_id}: {token_start}-{token_end} outside segment "
                    f"{start_ms}-{end_ms}"
                )
            if token_start < last_token_start:
                errors.append(f"{token_id}: token time is not monotonic")
            last_token_start = token_start
            if token_precision not in {"token", "interpolated", "segment"}:
                errors.append(f"{token_id}: invalid time_precision {token_precision}")
            if token.get("clickable"):
                clickable_count += 1
                if token_end <= token_start:
                    errors.append(f"{token_id}: clickable token has no duration")
        if clickable_count == 0:
            errors.append(f"{segment_id}: no clickable tokens")

    for segment_id, count in segment_ids.items():
        if count != 1:
            errors.append(f"duplicate segment id {segment_id}: {count}")
    for token_id, count in token_ids.items():
        if count != 1:
            errors.append(f"duplicate token id {token_id}: {count}")

    if errors:
        preview = "\n".join(errors[:20])
        raise TranscriptValidationError(
            f"{len(errors)} transcript validation error(s):\n{preview}"
        )
    return {
        "speaker_count": len(speaker_ids),
        "segment_count": len(segments),
        "token_count": sum(len(segment["tokens"]) for segment in segments),
        "clickable_token_count": sum(
            int(token["clickable"])
            for segment in segments
            for token in segment["tokens"]
        ),
        "precision_counts": dict(sorted(precision.items())),
        "speaker_segment_counts": dict(sorted(speaker_segments.items())),
    }
