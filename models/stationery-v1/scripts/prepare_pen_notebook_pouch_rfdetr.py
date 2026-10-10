#!/usr/bin/env python3
"""Build a separate three-class COCO dataset for RF-DETR Nano."""
from __future__ import annotations

import json
import os
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "vending_machine_datasets/pen_notebook_rfdetr_2class"
POUCH = ROOT / "vending_machine_datasets/pencil_pouch"
OUT = ROOT / "vending_machine_datasets/pen_notebook_pouch_rfdetr_3class"


def attach(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    # Symlink images so the experimental dataset does not duplicate the large
    # source photos or alter either source dataset.
    target.symlink_to(source.resolve())


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserving existing dataset: {OUT}")
    splits = {"train": [], "valid": [], "test": []}
    hashes = {split: set() for split in splits}
    for split in splits:
        src_dir = BASE / split
        src_coco = json.loads((src_dir / "_annotations.coco.json").read_text())
        dst_dir = OUT / split
        dst_dir.mkdir(parents=True)
        images = []
        annotations = []
        for record in src_coco["images"]:
            name = "base_" + record["file_name"]
            attach(src_dir / record["file_name"], dst_dir / name)
            item = dict(record, file_name=name)
            item["id"] = len(images) + 1
            images.append(item)
            for ann in (a for a in src_coco["annotations"] if a["image_id"] == record["id"]):
                annotations.append(dict(ann, id=len(annotations) + 1, image_id=item["id"]))

        # Add the user's reviewed pouch images to train and validation. They
        # come from one photographed pouch/session, so these val numbers only
        # measure held-out frames/scenes of that same item.
        if split in ("train", "valid"):
            pouch_split = "train" if split == "train" else "val"
            img_dir = POUCH / "images" / pouch_split
            label_dir = POUCH / "labels" / pouch_split
            for source in sorted(img_dir.glob("*.jpg")):
                name = "pouch_" + source.name
                attach(source, dst_dir / name)
                with Image.open(source) as im:
                    width, height = im.size
                    im.verify()
                image_id = len(images) + 1
                images.append({"id": image_id, "file_name": name, "width": width, "height": height})
                label = label_dir / (source.stem + ".txt")
                if not label.is_file():
                    raise FileNotFoundError(label)
                for line in label.read_text().splitlines():
                    fields = list(map(float, line.split()))
                    if len(fields) != 5 or int(fields[0]) != 0:
                        raise ValueError(f"Expected one class-0 YOLO box: {label}: {line}")
                    _, cx, cy, w, h = fields
                    if min(cx, cy, w, h) < 0 or max(cx, cy, w, h) > 1 or w <= 0 or h <= 0:
                        raise ValueError(f"Invalid YOLO box: {label}: {line}")
                    x1, y1 = max(0, cx - w / 2) * width, max(0, cy - h / 2) * height
                    x2, y2 = min(1, cx + w / 2) * width, min(1, cy + h / 2) * height
                    annotations.append({
                        "id": len(annotations) + 1, "image_id": image_id,
                        "category_id": 3, "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
                    })
        payload = {
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": 1, "name": "pen", "supercategory": "stationery"},
                {"id": 2, "name": "paper_notebook", "supercategory": "stationery"},
                {"id": 3, "name": "pencil_pouch", "supercategory": "stationery"},
            ],
        }
        (dst_dir / "_annotations.coco.json").write_text(json.dumps(payload, indent=2))
        splits[split] = {"images": len(images), "boxes": len(annotations)}

    manifest = {
        "classes": {1: "pen", 2: "paper_notebook", 3: "pencil_pouch"},
        "counts": splits,
        "source": [str(BASE), str(POUCH)],
        "notes": [
            "Separate experiment dataset; source datasets were not modified.",
            "Pouch validation uses 27 held-out photos from the same photographed pouch/session as training.",
            "Pouch test images are unavailable; report pouch validation separately from the original pen/notebook test split.",
            "This RF-DETR model is a dedicated three-class detector, initialized from official COCO-pretrained RF-DETR Nano weights.",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
