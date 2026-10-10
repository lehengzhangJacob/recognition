#!/usr/bin/env python3
"""Make a symlink-based copy of the existing 602-class OIV7 train set plus pouch labels."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "vending_machine_datasets/pen_notebook_full_oiv7_train_v9"
POUCH = ROOT / "vending_machine_datasets/pencil_pouch"
OUT = ROOT / "vending_machine_datasets/pen_notebook_pouch_full_oiv7_train_v10"


def link_tree(source: Path, destination: Path) -> int:
    count = 0
    for item in sorted(source.rglob("*")):
        if not item.is_file():
            continue
        target = destination / item.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(item.resolve())
        count += 1
    return count


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserving existing dataset: {OUT}")
    image_counts, label_counts = {}, {}
    for split, pouch_split in (("train", "train"), ("val", "val")):
        image_counts[split] = link_tree(BASE / "images" / split, OUT / "images" / split)
        label_counts[split] = link_tree(BASE / "labels" / split, OUT / "labels" / split)
        pouch_images = sorted((POUCH / "images" / pouch_split).glob("*.jpg"))
        for image in pouch_images:
            stem = "pouch_" + image.stem
            (OUT / "images" / split / (stem + ".jpg")).symlink_to(image.resolve())
            source_label = POUCH / "labels" / pouch_split / (image.stem + ".txt")
            rows = []
            for line in source_label.read_text().splitlines():
                fields = line.split()
                if len(fields) != 5 or fields[0] != "0":
                    raise ValueError(f"Expected class-0 YOLO label: {source_label}: {line}")
                fields[0] = "377"  # OIV7 Pencil case; zero-based class index.
                rows.append(" ".join(fields))
            (OUT / "labels" / split / (stem + ".txt")).write_text("\n".join(rows) + "\n")
        image_counts[split] += len(pouch_images)
        label_counts[split] += len(pouch_images)

    cfg = yaml.safe_load((BASE / "data.yaml").read_text())
    cfg["path"] = str(OUT)
    (OUT / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    manifest = {
        "source_oiv7_602_dataset": str(BASE),
        "source_pencil_pouch_dataset": str(POUCH),
        "output": str(OUT),
        "classes": {"Pen": 376, "Pencil case": 377, "Notebook": 601},
        "images": image_counts,
        "label_files": label_counts,
        "pouch": {"train": len(list((POUCH / "images/train").glob("*.jpg"))),
                  "validation": len(list((POUCH / "images/val").glob("*.jpg")))},
        "notes": [
            "Existing OIV7 dataset files are symlinked; original data is unchanged.",
            "Pouch YOLO class 0 is remapped to the retained OIV7 Pencil case class 377.",
            "Pouch photos are partially labeled: training loss must supervise only class 377 for pouch_ images.",
            "Pouch validation images come from the same photographed pouch/session as the training images.",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
