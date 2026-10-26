from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from asrgui.m0.audio import prepare_audio, probe_media, sha256_file
from asrgui.m0.models import model_paths, write_model_manifest
from asrgui.m0.transcript import write_json_atomic
from asrgui.paths import configure_process_environment, workspace_paths


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "audio"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="asrgui-m0")
    parser.add_argument(
        "--models-lock",
        type=Path,
        help="model lock path (default: Codex_work_space/models.lock.json)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("paths", help="print resolved paths as JSON")
    subparsers.add_parser("check-models", help="check local model directories")

    hash_parser = subparsers.add_parser("hash-models", help="generate model SHA-256 lock")
    hash_parser.add_argument("--output", type=Path)

    probe_parser = subparsers.add_parser("probe", help="probe a media file")
    probe_parser.add_argument("input", type=Path)

    transcribe_parser = subparsers.add_parser("transcribe", help="run local ASR")
    transcribe_parser.add_argument("input", type=Path)
    transcribe_parser.add_argument("--output", type=Path)
    transcribe_parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    transcribe_parser.add_argument("--ncpu", type=int, default=4)
    transcribe_parser.add_argument("--batch-size-s", type=int, default=60)

    batch_parser = subparsers.add_parser(
        "transcribe-batch", help="run checkpointed long-audio ASR"
    )
    batch_parser.add_argument("input", type=Path)
    batch_parser.add_argument("--output", type=Path)
    batch_parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    batch_parser.add_argument("--ncpu", type=int, default=4)
    batch_parser.add_argument("--batch-size-s", type=int, default=60)
    batch_parser.add_argument("--chunk-seconds", type=int, default=300)
    batch_parser.add_argument("--cancel-after-chunks", type=int)
    batch_parser.add_argument("--resume", action="store_true")

    speaker_parser = subparsers.add_parser(
        "check-speakers", help="extract CAM++ embeddings from transcript segments"
    )
    speaker_parser.add_argument("run_dir", type=Path)
    speaker_parser.add_argument("--max-per-speaker", type=int, default=3)

    validate_parser = subparsers.add_parser(
        "validate", help="validate a normalized transcript"
    )
    validate_parser.add_argument("path", type=Path)
    return parser


def _default_lock_path() -> Path:
    return workspace_paths().project_root / "Codex_work_space" / "models.lock.json"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = configure_process_environment()

    if args.command == "paths":
        print(json.dumps({key: str(value) for key, value in vars(paths).items()}, indent=2))
        return 0
    if args.command == "check-models":
        resolved = model_paths(paths.models_root)
        print(json.dumps({key: str(value) for key, value in vars(resolved).items()}, indent=2))
        return 0
    if args.command == "hash-models":
        output = args.output or args.models_lock or _default_lock_path()
        manifest = write_model_manifest(paths.models_root, output)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "models": {
                        role: {
                            "file_count": len(item["files"]),
                            "size_bytes": item["size_bytes"],
                        }
                        for role, item in manifest["models"].items()
                    },
                },
                indent=2,
            )
        )
        return 0
    if args.command == "probe":
        source = args.input.expanduser().resolve()
        payload = probe_media(source)
        payload["sha256"] = sha256_file(source)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command == "transcribe":
        from asrgui.m0.engine import run_transcription

        source = args.input.expanduser().resolve()
        output = (
            args.output
            or paths.results_root / f"{_safe_name(source.stem)}__{sha256_file(source)[:8]}"
        ).expanduser().resolve()
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

        def progress(event: dict) -> None:
            print(json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True)

        result = run_transcription(
            source_audio=prepared["source"],
            analysis_audio=prepared["analysis"],
            source_probe=prepared["source_probe"],
            source_sha256=prepared["sha256"],
            models_root=paths.models_root,
            output_dir=output,
            device=args.device,
            ncpu=args.ncpu,
            batch_size_s=args.batch_size_s,
            progress=progress,
        )
        print(json.dumps(result["run"], ensure_ascii=False, indent=2))
        return 0
    if args.command == "transcribe-batch":
        from asrgui.m0.batch import run_batch_transcription

        source = args.input.expanduser().resolve()
        output = (
            args.output
            or paths.results_root
            / f"{_safe_name(source.stem)}__{sha256_file(source)[:8]}__batch"
        ).expanduser().resolve()
        if output.exists() and not args.resume and any(output.glob("checkpoints/*.json")):
            raise SystemExit(
                f"{output} has existing checkpoints; pass --resume to continue them"
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

        def progress(event: dict) -> None:
            print(json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True)

        def should_cancel() -> bool:
            if args.cancel_after_chunks is None:
                return False
            checkpoint = output / "batch_checkpoint.json"
            if not checkpoint.is_file():
                return False
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            return int(state.get("completed_chunks", 0)) >= args.cancel_after_chunks

        result = run_batch_transcription(
            source_audio=prepared["source"],
            analysis_audio=prepared["analysis"],
            source_probe=prepared["source_probe"],
            source_sha256=prepared["sha256"],
            models_root=paths.models_root,
            output_dir=output,
            device=args.device,
            ncpu=args.ncpu,
            batch_size_s=args.batch_size_s,
            chunk_seconds=args.chunk_seconds,
            should_cancel=should_cancel,
            progress=progress,
        )
        print(json.dumps(result.get("run", result), ensure_ascii=False, indent=2))
        return 0
    if args.command == "check-speakers":
        from asrgui.m0.speaker import extract_segment_embeddings

        run_dir = args.run_dir.expanduser().resolve()
        transcript = json.loads(
            (run_dir / "transcript" / "raw.json").read_text(encoding="utf-8")
        )
        analysis = run_dir / "audio" / "asr_16k.wav"
        result = extract_segment_embeddings(
            analysis_audio=analysis,
            transcript=transcript,
            models_root=paths.models_root,
            max_per_speaker=args.max_per_speaker,
        )
        write_json_atomic(run_dir / "speaker_check.json", result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate":
        from asrgui.m0.validation import validate_transcript

        path = args.path.expanduser().resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        summary = validate_transcript(data)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
