#!/usr/bin/env python3
"""Reassemble and verify RF-DETR artifacts split to fit GitHub file limits."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "rf-detr-nano-3class"
manifest = json.loads((PARTS / "artifact_parts.json").read_text())
for output_name, meta in manifest.items():
    output = PARTS / output_name
    with output.open("wb") as dst:
        for part_name in meta["parts"]:
            with (PARTS / part_name).open("rb") as src:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    if output.stat().st_size != meta["size_bytes"] or digest != meta["sha256"]:
        output.unlink(missing_ok=True)
        raise SystemExit(f"Verification failed: {output_name}")
    print(f"Verified {output_name}: {digest}")
