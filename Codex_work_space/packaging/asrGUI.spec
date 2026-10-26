# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


packaging_dir = Path(SPECPATH).resolve()
workspace = packaging_dir.parent
project_root = workspace.parent
model_root = project_root / "AppFile" / "runtime" / "models"
ffmpeg_root = workspace / "vendor" / "ffmpeg"

datas = [
    (str(model_root / "paraformer-zh"), "models/paraformer-zh"),
    (str(model_root / "fsmn-vad"), "models/fsmn-vad"),
    (str(model_root / "ct-punc"), "models/ct-punc"),
    (str(model_root / "campplus"), "models/campplus"),
    (str(ffmpeg_root / "ffmpeg"), "ffmpeg"),
    (str(ffmpeg_root / "ffprobe"), "ffmpeg"),
    (str(workspace / "models.lock.json"), "."),
    (str(workspace / "THIRD_PARTY.md"), "licenses"),
]
datas += collect_data_files("funasr")

hiddenimports = collect_submodules("funasr")

a = Analysis(
    [str(workspace / "src/asrgui/m0/packaged_main.py")],
    pathex=[str(workspace / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(packaging_dir / "pyi_rth_funasr.py")],
    excludes=[
        "tkinter",
        "matplotlib",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtPdf",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="asrGUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    argv_emulation=False,
    target_arch="arm64",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="asrGUI",
)

app = BUNDLE(
    coll,
    name="asrGUI.app",
    icon=None,
    bundle_identifier="com.jerryliiiiii.asrgui",
    version="0.1.0",
    info_plist={
        "CFBundleDisplayName": "asrGUI",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "14.0",
        "NSHighResolutionCapable": True,
        "NSPrincipalClass": "NSApplication",
    },
)
