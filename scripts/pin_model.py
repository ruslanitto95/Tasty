"""Release-engineering helper: download GigaAM files once and pin size + SHA-256.

Writes src/mva/resources/model_manifest.json. Run only when deliberately updating the
model; the application itself downloads and verifies against this manifest.

    uv run python scripts/pin_model.py
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "src" / "mva" / "resources" / "model_manifest.json"
CDN = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM"
MODELS = {"v3_e2e_rnnt": "GigaAM-v3 e2e RNNT (punctuation + normalization)"}


def gigaam_commit() -> str:
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    match = re.search(r"salute-developers/GigaAM\?rev=[0-9a-f]{40}#([0-9a-f]{40})", lock)
    if not match:
        raise SystemExit("GigaAM commit not found in uv.lock")
    return match.group(1)


def fetch(url: str, dest: Path) -> tuple[int, str, str]:
    digest = hashlib.sha256()
    size = 0
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as resp:
        resp.raise_for_status()
        etag = resp.headers.get("etag", "").strip('"')
        with dest.open("wb") as out:
            for chunk in resp.iter_bytes(1 << 20):
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    return size, digest.hexdigest(), etag


def main() -> int:
    commit = gigaam_commit()
    entries = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, display in MODELS.items():
            files = []
            etags = []
            for fname in (f"{name}.ckpt", f"{name}_tokenizer.model"):
                size, sha, etag = fetch(f"{CDN}/{fname}", Path(tmp) / fname)
                files.append({"name": fname, "url": f"{CDN}/{fname}", "size": size, "sha256": sha})
                etags.append(etag)
                print(f"pinned {fname}: {size} bytes", file=sys.stderr)
            entries.append(
                {
                    "model_name": name,
                    "display_name": display,
                    "gigaam_commit": commit,
                    "revision": f"{name}@cdn-etag:{etags[0]}",
                    "files": files,
                }
            )
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"models": entries}, indent=2) + "\n", encoding="utf-8")
    subprocess.run(["git", "diff", "--stat", str(MANIFEST)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
