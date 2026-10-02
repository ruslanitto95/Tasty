# CI helper: plays the bundled synthetic dialogue into the VB-CABLE playback device in a
# loop, so whatever records from "CABLE Output" hears speech regardless of timing.
param([string]$Workspace)
Set-Location $Workspace
uv run python -c @"
import time, sounddevice as sd, soundfile as sf
x, r = sf.read('src/mva/resources/test_audio/self_test_ru.wav', dtype='float32')
dev = [i for i, d in enumerate(sd.query_devices()) if 'CABLE Input' in d['name'] and d['max_output_channels'] > 0][0]
for _ in range(12):
    sd.play(x, r, device=dev, blocking=True)
    time.sleep(1.0)
"@
