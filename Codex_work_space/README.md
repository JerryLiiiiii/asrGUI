# asrGUI M0 workspace

Status: M0 technical validation is complete. The current speaker clustering
prototype over-segments the 4-person long sample into 8 groups; M0 validates the
interface only, and accuracy correction is an M3 entry requirement.

M0 只验证离线 ASR、时间戳、说话人、长音频和最小打包技术路线。产品和里程碑边界以
`Assets/doc/DESIGN_v0.1.md`、`Assets/doc/ROADMAP_v0.1.md` 为准。

## Environment

- Interpreter: `/opt/homebrew/bin/python3.11` (3.11.17)
- Virtual environment: `Codex_work_space/.venv`
- Data root: `AppFile`
- Local models: `AppFile/runtime/models`
- Experiment results: `AppFile/runtime/m0`

All framework caches are redirected below `AppFile/runtime/cache` by
`asrgui.paths.configure_process_environment`.

## Commands

```bash
Codex_work_space/.venv/bin/python -m asrgui.m0.cli paths
Codex_work_space/.venv/bin/python -m asrgui.m0.cli check-models
Codex_work_space/.venv/bin/python -m asrgui.m0.cli hash-models
Codex_work_space/.venv/bin/python -m asrgui.m0.cli transcribe \
  Assets/audio/reference_rec_5min.mp3 --ncpu 4 --batch-size-s 60
Codex_work_space/.venv/bin/python -m asrgui.m0.cli transcribe-batch \
  Assets/audio/reference_rec_30min.m4a --ncpu 4 --batch-size-s 60 \
  --chunk-seconds 300
Codex_work_space/.venv/bin/python -m asrgui.m0.cli validate \
  AppFile/runtime/m0/reference_rec_30min_clean/transcript/raw.json
```

The CLI copies source audio into the experiment meeting directory, generates a
16 kHz mono signed 16-bit WAV, runs local models with update checks disabled,
and writes `model_result.json`, `transcript/raw.json`, `speakers.json`, and
`run.json`.

## Offline verification

The models must be addressed by local directory only. A real offline run sets
invalid proxy endpoints plus framework offline flags so any accidental hub call
fails immediately:

```bash
env MODELSCOPE_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 \
  Codex_work_space/.venv/bin/python -m asrgui.m0.cli transcribe ...
```

Do not treat "no network configured" as an offline test.

## Tests and packaging

```bash
Codex_work_space/.venv/bin/python -m pytest -q Codex_work_space/tests
Codex_work_space/.venv/bin/python Codex_work_space/packaging/build_macos.py \
  --sign-identity -
```

`-` requests an ad-hoc signature for technical verification. A Personal Team
identity must be passed explicitly for signed builds, for example:

```bash
Codex_work_space/.venv/bin/python Codex_work_space/packaging/build_macos.py \
  --sign-identity "Apple Development: Name (TEAMID)"
```

The signing process needs normal access to the login keychain. The build keeps
PyInstaller caches and temporary files below `Codex_work_space/build`; it does
not write them to the user Library.

The M0 result is recorded in
`Codex_work_space/docs/M0_REPORT.md`.
