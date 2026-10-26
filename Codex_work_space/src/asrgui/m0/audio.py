from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import soundfile as sf

from asrgui.paths import find_resource_root


class AudioToolError(RuntimeError):
    pass


def find_ffmpeg(name: str) -> Path:
    bundled = find_resource_root() / "ffmpeg" / name
    if bundled.is_file():
        return bundled
    system = shutil.which(name)
    if system:
        return Path(system)
    raise AudioToolError(f"{name} was not found in the app resources or PATH")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_media(path: Path) -> dict[str, Any]:
    command = [
        str(find_ffmpeg("ffprobe")),
        "-v",
        "error",
        "-show_entries",
        (
            "format=format_name,duration,size,bit_rate:"
            "stream=index,codec_name,codec_type,sample_rate,channels,channel_layout"
        ),
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, encoding="utf-8"
    )
    if result.returncode != 0:
        raise AudioToolError(result.stderr.strip() or f"ffprobe failed for {path}")
    data = json.loads(result.stdout)
    streams = [
        stream for stream in data.get("streams", []) if stream.get("codec_type") == "audio"
    ]
    if not streams:
        raise AudioToolError(f"No audio stream found in {path}")
    media_format = data.get("format", {})
    stream = streams[0]
    return {
        "format_name": media_format.get("format_name"),
        "duration_ms": round(float(media_format.get("duration", 0)) * 1000),
        "size_bytes": int(media_format.get("size", path.stat().st_size)),
        "bit_rate": int(media_format.get("bit_rate", 0) or 0),
        "codec_name": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate", 0) or 0),
        "channels": int(stream.get("channels", 0) or 0),
        "channel_layout": stream.get("channel_layout"),
    }


def transcode_to_asr_wav(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(find_ffmpeg("ffmpeg")),
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-map_metadata",
        "-1",
        "-f",
        "wav",
        str(destination),
    ]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, encoding="utf-8"
    )
    if result.returncode != 0:
        raise AudioToolError(result.stderr.strip() or f"ffmpeg failed for {source}")
    info = sf.info(destination)
    return {
        "path": destination,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration_ms": round(info.frames / info.samplerate * 1000),
        "subtype": info.subtype,
        "format": info.format,
    }


def prepare_audio(source: Path, meeting_dir: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    audio_dir = meeting_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    original = audio_dir / f"original{source.suffix.lower()}"
    if source != original:
        shutil.copy2(source, original)
    analysis = audio_dir / "asr_16k.wav"
    source_probe = probe_media(original)
    analysis_info = transcode_to_asr_wav(original, analysis)
    return {
        "source": original,
        "analysis": analysis,
        "source_probe": source_probe,
        "analysis_info": analysis_info,
        "sha256": sha256_file(original),
    }
