from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, TextIO

from asrgui.m0.audio import find_ffmpeg, prepare_audio
from asrgui.m0.models import model_paths
from asrgui.m0.transcript import write_json_atomic
from asrgui.paths import (
    configure_process_environment,
    find_resource_root,
    workspace_paths,
)


PROTOCOL_STDOUT: TextIO = sys.stdout
EMIT_LOCK = threading.Lock()


def emit(payload: dict[str, Any]) -> None:
    with EMIT_LOCK:
        PROTOCOL_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
        PROTOCOL_STDOUT.flush()


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "audio"


class Worker:
    def __init__(self) -> None:
        self.paths = configure_process_environment()
        self.worker_thread: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self.active_job_id: str | None = None

    def handle(self, request: dict[str, Any]) -> None:
        job_id = str(request.get("id") or "job")
        request_type = request.get("type")
        payload = request.get("payload") or {}
        if request_type == "cancel":
            self.cancel_event.set()
            emit(
                {
                    "id": self.active_job_id or job_id,
                    "event": "cancel_requested",
                    "message": "将在当前安全阶段结束后取消",
                }
            )
            return
        if request_type == "shutdown":
            self.cancel_event.set()
            emit({"id": job_id, "event": "shutdown_ack"})
            raise SystemExit(0)
        if request_type == "smoke":
            emit({"id": job_id, "event": "progress", "stage": "smoke", "percent": 10})
            emit({"id": job_id, "event": "completed", "result": self.smoke()})
            return
        if request_type == "transcribe":
            if self.worker_thread and self.worker_thread.is_alive():
                emit(
                    {
                        "id": job_id,
                        "event": "failed",
                        "stage": "queue",
                        "code": "WORKER_BUSY",
                        "message": "worker 当前已有任务",
                    }
                )
                return
            self.cancel_event.clear()
            self.active_job_id = job_id
            self.worker_thread = threading.Thread(
                target=self._transcribe,
                args=(job_id, payload),
                name=f"asr-{job_id}",
                daemon=True,
            )
            self.worker_thread.start()
            return
        emit(
            {
                "id": job_id,
                "event": "failed",
                "stage": "protocol",
                "code": "UNKNOWN_REQUEST",
                "message": f"未知请求类型: {request_type}",
            }
        )

    def smoke(self) -> dict[str, Any]:
        resource_root = find_resource_root()
        checks: dict[str, Any] = {
            "python_executable": sys.executable,
            "frozen": bool(getattr(sys, "frozen", False)),
            "worker_pid": os.getpid(),
            "resource_root": str(resource_root),
            "data_root": str(self.paths.data_root),
        }
        try:
            checks["models"] = {
                key: str(path) for key, path in vars(model_paths(self.paths.models_root)).items()
            }
        except FileNotFoundError as exc:
            checks["models_error"] = str(exc)
        try:
            checks["ffmpeg"] = str(find_ffmpeg("ffmpeg"))
            checks["ffprobe"] = str(find_ffmpeg("ffprobe"))
        except RuntimeError as exc:
            checks["ffmpeg_error"] = str(exc)
        return checks

    def _transcribe(self, job_id: str, payload: dict[str, Any]) -> None:
        try:
            from asrgui.m0.batch import run_batch_transcription

            source = Path(payload["input"]).expanduser().resolve()
            suffix = str(payload.get("output") or "").strip()
            output = (
                Path(suffix).expanduser().resolve()
                if suffix
                else self.paths.results_root / f"{_safe_name(source.stem)}__worker"
            )

            def progress(event: dict[str, Any]) -> None:
                emit({"id": job_id, "event": "progress", **event})

            emit(
                {
                    "id": job_id,
                    "event": "progress",
                    "stage": "importing",
                    "percent": 0,
                    "message": "正在复制并规范化音频",
                }
            )
            prepared = prepare_audio(source, output)
            write_json_atomic(
                output / "input.json",
                {
                    "source": str(source),
                    "source_probe": prepared["source_probe"],
                    "analysis_info": prepared["analysis_info"],
                    "sha256": prepared["sha256"],
                },
            )
            result = run_batch_transcription(
                source_audio=prepared["source"],
                analysis_audio=prepared["analysis"],
                source_probe=prepared["source_probe"],
                source_sha256=prepared["sha256"],
                models_root=self.paths.models_root,
                output_dir=output,
                chunk_seconds=int(payload.get("chunk_seconds", 300)),
                should_cancel=self.cancel_event.is_set,
                progress=progress,
            )
            if result["state"] == "cancelled":
                emit(
                    {
                        "id": job_id,
                        "event": "cancelled",
                        "result": result,
                    }
                )
            else:
                emit({"id": job_id, "event": "completed", "result": result.get("run", result)})
        except Exception as exc:
            emit(
                {
                    "id": job_id,
                    "event": "failed",
                    "stage": "worker",
                    "code": type(exc).__name__,
                    "message": str(exc),
                    "detail": traceback.format_exc(),
                }
            )
        finally:
            self.active_job_id = None


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(prog="asr-worker").parse_args(argv)
    console_stdout = sys.stdout
    globals()["PROTOCOL_STDOUT"] = console_stdout
    sys.stdout = sys.stderr
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    worker = Worker()
    emit(
        {
            "id": "system",
            "event": "ready",
            "pid": os.getpid(),
            "frozen": bool(getattr(sys, "frozen", False)),
        }
    )
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            worker.handle(request)
        except SystemExit:
            raise
        except Exception as exc:
            emit(
                {
                    "id": "system",
                    "event": "failed",
                    "stage": "protocol",
                    "code": type(exc).__name__,
                    "message": str(exc),
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
