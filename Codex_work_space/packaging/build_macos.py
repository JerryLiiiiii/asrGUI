from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE.parent
APP_NAME = "asrGUI.app"


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=check, text=True)


def disk_free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sign-identity",
        default="-",
        help="codesign identity; default '-' is ad-hoc for the technical prototype",
    )
    parser.add_argument("--deploy", action="store_true")
    args = parser.parse_args()

    if sys.platform != "darwin":
        raise SystemExit("M0 macOS packaging must run on macOS")
    if shutil.which("codesign") is None:
        raise SystemExit("codesign is required")
    if disk_free_bytes(PROJECT_ROOT) < 8 * 1024**3:
        raise SystemExit("at least 8 GiB free disk is required for the M0 build")

    dist = WORKSPACE / "dist"
    build = WORKSPACE / "build"
    pyinstaller_cache = build / "pyinstaller-cache"
    pyinstaller_tmp = build / "tmp"
    pyinstaller_cache.mkdir(parents=True, exist_ok=True)
    pyinstaller_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["PYINSTALLER_CONFIG_DIR"] = str(pyinstaller_cache)
    os.environ["TMPDIR"] = str(pyinstaller_tmp)
    run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            str(dist),
            "--workpath",
            str(build / "pyinstaller"),
            str(WORKSPACE / "packaging" / "asrGUI.spec"),
        ]
    )

    app = dist / APP_NAME
    main_binary = app / "Contents" / "MacOS" / "asrGUI"
    worker_binary = app / "Contents" / "MacOS" / "asr-worker"
    if not main_binary.is_file():
        raise SystemExit(f"PyInstaller did not create {main_binary}")

    funasr_spec = importlib.util.find_spec("funasr")
    if funasr_spec is None or not funasr_spec.submodule_search_locations:
        raise SystemExit("FunASR package location could not be resolved")
    funasr_source = Path(next(iter(funasr_spec.submodule_search_locations)))
    funasr_destination = app / "Contents" / "Frameworks" / "funasr"
    shutil.copytree(
        funasr_source,
        funasr_destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    shutil.copy2(main_binary, worker_binary)
    worker_binary.chmod(0o755)

    run(
        [
            "codesign",
            "--force",
            "--deep",
            "--sign",
            args.sign_identity,
            str(app),
        ]
    )
    run(["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app)])

    build_info = {
        "built_at": datetime.now(UTC).isoformat(),
        "app": str(app),
        "sign_identity": args.sign_identity,
        "bundle_identifier": "com.jerryliiiiii.asrgui",
        "python": sys.version,
        "app_size_bytes": sum(path.stat().st_size for path in app.rglob("*") if path.is_file()),
        "funasr_source_bundled": True,
    }
    info_path = dist / "build-info.json"
    info_path.write_text(
        json.dumps(build_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(build_info, ensure_ascii=False, indent=2))

    if args.deploy:
        destination = PROJECT_ROOT / "AppFile" / "app" / APP_NAME
        if destination.exists():
            raise SystemExit(f"refusing to overwrite existing app: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        run(["ditto", str(app), str(destination)])
        print(f"deployed: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
