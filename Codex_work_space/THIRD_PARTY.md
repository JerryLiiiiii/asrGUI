# M0 third-party baseline

This file records the M0 candidates. It is not legal advice and does not replace
the complete license texts distributed by each upstream project.

| Component | Locked version / snapshot | License | M0 condition |
| --- | --- | --- | --- |
| CPython | 3.11.17 | PSF-2.0 | Bundled runtime must retain Python notices. |
| FunASR | 1.4.16 | MIT | Include upstream copyright and license. |
| ModelScope | 1.40.2 | Apache-2.0 | Local model paths are used; include license and notices. |
| PyTorch | 2.11.0 | BSD-3-Clause | Include upstream copyright and license. |
| torchaudio | 2.11.0 | BSD-2-Clause | Include upstream copyright and license. |
| PySide6 / Qt | 6.12.0 | LGPL-3.0-only / GPL / commercial | Current app dynamically links Qt and does not modify Qt. Keep replacement possible and distribute LGPL notices/source offer as applicable. |
| FFmpeg + ffprobe | 9.0 Apple Silicon static build | GPL-3.0-or-later (binary enables GPL) | Personal local use only in M0. Any later distribution requires GPL notices and corresponding source/offer. |
| Paraformer SeACo ASR | ModelScope `master` snapshot 2026-10-25 | Apache-2.0 | Local snapshot is pinned by `models.lock.json`; include model notice. |
| FSMN-VAD | ModelScope `master` snapshot 2026-10-25 | Apache-2.0 | Local snapshot is pinned by `models.lock.json`; include model notice. |
| CT-Punc | ModelScope `master` snapshot 2026-10-25 | Apache-2.0 | Local snapshot is pinned by `models.lock.json`; include model notice. |
| CAM++ speaker model | ModelScope `master` snapshot 2026-10-25 | Apache-2.0 | Local snapshot is pinned by `models.lock.json`; include model notice. |
| PyInstaller | 6.22.3 | GPL-2.0-or-later with bootloader exception | Built applications may be distributed under the project's chosen terms. |

## Verified model sources

- ASR: `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`
- VAD: `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`
- Punctuation: `iic/punc_ct-transformer_cn-en-common-vocab471067-large`
- Speaker: `iic/speech_campplus_sv_zh-cn_16k-common`

The source hubs expose a mutable `master` branch. The effective M0 pin is the
content SHA-256 in `models.lock.json`, not the branch name. A future release
must ship the verified local snapshot and must not download it at first launch.

## Static FFmpeg source

- Distribution page: <https://www.osxexperts.net/>
- Archives: `ffmpeg9arm.zip`, `ffprobe9arm.zip`
- Archive SHA-256:
  - `ffmpeg9arm.zip`: `d0c06c5c68ce48af3143b262f7a9118a7c9f67de1e237fcc24ffb14df9c67af9`
  - `ffprobe9arm.zip`: `0c94fbdd8917022f28115eca512196cf4648732bc9e5db9ec8896c7e519d02aa`
- Binary SHA-256:
  - `ffmpeg`: `591260c945d0eef150e3bf82b0ef988bd36a9cecc18ff05d6679617159f0a95e`
  - `ffprobe`: `e11c17e8200b3ee4c4c186d245e2b4053f01d56957336c1817fca0b997469106`

The binaries link only Apple system frameworks/libraries. They were ad-hoc
signed by the upstream build and must be re-signed as nested code when producing
the local `.app`.

## Packaged FunASR compatibility note

FunASR 1.4.16 scans model modules during import and calls
`inspect.getsourcelines()` to store diagnostic source locations. PyInstaller
does not retain Python source in the frozen archive. A project-owned runtime
hook supplies empty source lines only when that diagnostic lookup fails. The
build also includes FunASR's Python source files because `torch.jit.script` in
the Paraformer CIF predictor requires source text. Neither measure changes model
code, weights, audio, timestamps, or inference results.
