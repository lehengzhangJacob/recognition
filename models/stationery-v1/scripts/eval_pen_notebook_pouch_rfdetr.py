#!/usr/bin/env python3
"""Evaluate the three-class RF-DETR checkpoint using IoU-0.5 box matches."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from rfdetr import RFDETRNano

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "vending_machine_datasets/pen_notebook_pouch_rfdetr_3class"
NAMES = {0: "pen", 1: "paper_notebook", 2: "pencil_pouch"}
GT_TO_MODEL = {1: 0, 2: 1, 3: 2}


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-9)


def evaluate(rows, class_id, threshold):
    tp = fp = targets = 0
    for row in rows:
        gt = row["gt"].get(class_id, [])
        targets += len(gt)
        preds = sorted((p for p in row["pred"] if p["class"] == class_id and p["score"] >= threshold),
                       key=lambda p: p["score"], reverse=True)
        used = set()
        for pred in preds:
            overlaps = [(iou(pred["box"], box), idx) for idx, box in enumerate(gt) if idx not in used]
            overlap, idx = max(overlaps, default=(0.0, -1))
            if overlap >= 0.5:
                tp += 1
                used.add(idx)
            else:
                fp += 1
    fn = targets - tp
    return {"targets": targets, "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(targets, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1), "threshold": threshold}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, nargs="?", default=ROOT / "runs/pen_notebook_pouch_rfdetr_nano_v1/checkpoint_best_total.pth")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/pen_notebook_pouch_rfdetr_nano_v1/evaluation_iou50.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = RFDETRNano(pretrain_weights=str(args.checkpoint), num_classes=3)
    model.inference(compile=False, dtype=torch.float16)
    datasets = {}
    for split in ("valid", "test"):
        folder = DATA / split
        coco = json.loads((folder / "_annotations.coco.json").read_text())
        gt = defaultdict(lambda: defaultdict(list))
        for ann in coco["annotations"]:
            x, y, w, h = ann["bbox"]
            gt[ann["image_id"]][GT_TO_MODEL[ann["category_id"]]].append([x, y, x + w, y + h])
        rows = []
        images = coco["images"]
        for start in range(0, len(images), 8):
            batch = images[start:start + 8]
            results = model.predict([str(folder / row["file_name"]) for row in batch], threshold=0.01)
            if not isinstance(results, list):
                results = [results]
            for image, result in zip(batch, results):
                detections = [{"class": int(c), "score": float(s), "box": list(map(float, b))}
                              for b, s, c in zip(result.xyxy, result.confidence, result.class_id)]
                rows.append({"image": image["file_name"], "gt": gt[image["id"]], "pred": detections})
        datasets[split] = rows

    val = datasets["valid"]
    groups = {
        "pen_notebook_validation_base_images": [r for r in val if r["image"].startswith("base_")],
        "pouch_validation_same_item_session": [r for r in val if r["image"].startswith("pouch_")],
    }
    threshold_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]
    payload = {
        "checkpoint": str(args.checkpoint),
        "classes": NAMES,
        "metric": "greedy per-class box match, IoU >= 0.50",
        "validation": {},
        "test_original_pen_notebook_set": {},
        "threshold_curves": {},
        "scope_note": "Pouch validation is 27 held-out photos from the same photographed pouch/session; there is no independent pouch test set. The original pen/notebook test split has no pouch positive labels.",
    }
    recommended_thresholds = {0: 0.30, 1: 0.45, 2: 0.30}
    for key, rows in groups.items():
        payload["validation"][key] = {NAMES[c]: evaluate(rows, c, recommended_thresholds[c]) for c in NAMES}
        payload["threshold_curves"][key] = {
            NAMES[c]: [evaluate(rows, c, t) for t in threshold_grid] for c in NAMES
        }
    payload["test_original_pen_notebook_set"] = {
        NAMES[c]: evaluate(datasets["test"], c, recommended_thresholds[c]) for c in NAMES
    }
    payload["test_original_pen_notebook_set"]["pencil_pouch_threshold_curve"] = [
        evaluate(datasets["test"], 2, t) for t in threshold_grid
    ]
    for cls_name, cls_id, rows in [
        ("pen", 0, groups["pen_notebook_validation_base_images"]),
        ("paper_notebook", 1, groups["pen_notebook_validation_base_images"]),
        ("pencil_pouch", 2, groups["pouch_validation_same_item_session"]),
    ]:
        best = max((evaluate(rows, cls_id, t) for t in threshold_grid), key=lambda m: m["f1"])
        payload.setdefault("best_validation_f1_thresholds", {})[cls_name] = best
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "threshold_curves"}, indent=2))


if __name__ == "__main__":
    main()
