# PyInstaller spec — onedir build of Medical Visit Assistant.
# Build: uv run pyinstaller packaging/mva.spec --noconfirm --clean
# The GigaAM checkpoint is NOT bundled: ModelManager downloads and verifies it on first run.

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent
SRC = ROOT / "src" / "mva"
IS_WIN = sys.platform == "win32"

datas = [
    (str(SRC / "resources"), "mva/resources"),
    (str(SRC / "clinical" / "prompts" / "clinical_extraction_v1.md"), "mva/clinical/prompts"),
    (str(SRC / "clinical" / "prompts" / "medical_formatter_v1.md"), "mva/clinical/prompts"),
    (str(ROOT / "THIRD_PARTY_LICENSES.md"), "."),
    (str(ROOT / "PRIVACY.md"), "."),
]
datas += collect_data_files("silero_vad", includes=["data/silero_vad.jit"])
datas += collect_data_files("hydra")

# hydra.utils.instantiate() imports classes by dotted path from the checkpoint config.
hiddenimports = collect_submodules("gigaam", filter=lambda name: "vad_utils" not in name and "onnx_utils" not in name)
hiddenimports += collect_submodules("hydra") + collect_submodules("omegaconf")
hiddenimports += ["keyring.backends.Windows", "keyring.backends.fail", "sentencepiece", "soxr"]

binaries = []
# App-local MSVC runtime (redistributable) so the app runs without a separate VC++ install.
crt_dir = os.environ.get("MVA_VC_CRT_DIR")
if IS_WIN and crt_dir and Path(crt_dir).is_dir():
    binaries += [(str(p), ".") for p in Path(crt_dir).glob("*.dll")]

excludes = [
    "tkinter",
    "matplotlib",
    "IPython",
    "pytest",
    "onnx",
    "onnxruntime",
    "pyannote",
    "transformers",
    "tensorboard",
    "torch.utils.tensorboard",
    "PySide6.QtWebEngineCore",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.Qt3DCore",
    "PySide6.QtMultimedia",
]

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MedicalVisitAssistant",
    console=False,
    icon=str(ROOT / "packaging" / "mva.ico"),
    version=str(ROOT / "packaging" / "version_info.txt") if IS_WIN else None,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="MedicalVisitAssistant")
