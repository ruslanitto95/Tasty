# Packaging

## Tool choice
**PyInstaller 6 (onedir)** was chosen after an actual build: it has maintained hooks for PySide6, PyTorch (pyinstaller-hooks-contrib), sounddevice/soundfile DLLs, and the frozen build passes `--self-test` (download → load GigaAM → transcribe WAV) on Linux and Windows. Nuitka was not prototyped: compiling PyTorch with Nuitka is slow and its torch support is less mature; with PyInstaller working end-to-end there was no measurable benefit to justify it (see TECH_DECISIONS.md).

- `onedir`, windowed (`console=False`), no UPX (antivirus false positives).
- GigaAM is **not** bundled (~450 MB); `ModelManager` downloads it once on first run.
- Bundled: app resources (manifest, lexicon, prompts, synthetic test WAV), Silero VAD JIT model, hydra/omegaconf/gigaam modules (instantiated by name), Qt Russian translations, app-local MSVC runtime from Visual Studio's redistributable folder.
- Excluded: onnx/onnxruntime, pyannote, transformers, tensorboard, unused Qt modules.
- FFmpeg is **not** needed: audio is passed to GigaAM as tensors; files are read with libsndfile.

## Build (Windows)
```powershell
uv sync --frozen
$env:MVA_VC_CRT_DIR = "<VS>\VC\Redist\MSVC\<ver>\x64\Microsoft.VC143.CRT"
uv run pyinstaller packaging/mva.spec --noconfirm --clean
& "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" /DAppVersion=0.1.0 installer\mva.iss
# → dist\MedicalVisitAssistant-0.1.0-Setup.exe
```

## Installer (Inno Setup 6)
- Per-user install to `%LOCALAPPDATA%\Programs\MedicalVisitAssistant`, no admin rights, Start-menu (+ optional desktop) shortcut, Russian UI.
- Uninstall removes the program, `temp` and `logs`; settings and the downloaded model are kept unless the user agrees to delete them (silent uninstall keeps them).
- Removes the optional autostart registry value.

## Clean-machine status
CI (`windows-2022` runner) installs and runs the installed app with a fresh profile and a PATH without Python, verifying first-run download, WAV and microphone transcription, GUI start and uninstall. A pristine consumer Windows 10/11 VM (no Visual Studio runtimes, sleep/resume, USB hot-plug) has **not** been tested yet; the app-local MSVC runtime is included specifically for machines without the VC++ redistributable.
