#!/usr/bin/env python3
"""Compare the prior and pouch-adapted full OIV7 YOLO models."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA3 = ROOT / "vending_machine_datasets/pen_notebook_pouch_rfdetr_3class"
BASE2 = ROOT / "vending_machine_datasets/pen_notebook_rfdetr_2class"
MODELS = {
    "prior_full_oiv7_602": ROOT / "runs/pen_notebook_full_oiv7_v9/weights/best.pt",
    "pouch_finetuned_full_oiv7_602": ROOT / "runs/pen_notebook_pouch_full_oiv7_v10/weights/best.pt",
}
CLASS_IDS = {"pen": 376, "pencil_pouch": 377, "paper_notebook": 601}
THRESHOLDS = {"pen": 0.30, "pencil_pouch": 0.30, "paper_notebook": 0.45}


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-9)


def read_rows(folder, prefix, from_three_class):
    coco = json.loads((folder / "_annotations.coco.json").read_text())
    by_image = defaultdict(lambda: defaultdict(list))
    # Three-class dataset COCO ids are pen=1, notebook=2, pouch=3. The old
    # two-class dataset contains only pen=1 and notebook=2.
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        category = ann["category_id"] - 1
        by_image[ann["image_id"]][category].append([x, y, x + w, y + h])
    output = []
    for image in coco["images"]:
        if prefix is not None and not image["file_name"].startswith(prefix):
            continue
        output.append({"path": folder / image["file_name"], "name": image["file_name"],
                       "gt": by_image[image["id"]]})
    return output


def match(rows, cls_name, threshold):
    cls = {"pen": 0, "paper_notebook": 1, "pencil_pouch": 2}[cls_name]
    tp = fp = targets = images_with_tp = 0
    for row in rows:
        gt = row["gt"].get(cls, [])
        targets += len(gt)
        preds = sorted((p for p in row["pred"] if p["name"] == cls_name and p["score"] >= threshold),
                       key=lambda p: p["score"], reverse=True)
        used = set()
        local_tp = 0
        for pred in preds:
            ov = [(iou(pred["box"], target), idx) for idx, target in enumerate(gt) if idx not in used]
            best, idx = max(ov, default=(0.0, -1))
            if best >= 0.5:
                tp += 1
                local_tp += 1
                used.add(idx)
            else:
                fp += 1
        images_with_tp += bool(local_tp)
    return {"targets": targets, "tp": tp, "fp": fp, "fn": targets - tp,
            "images_with_tp": images_with_tp,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(targets, 1),
            "threshold": threshold, "iou_threshold": 0.5}


def main():
    torch.set_num_threads(4)
    val_base = read_rows(DATA3 / "valid", "base_", True)
    val_pouch = read_rows(DATA3 / "valid", "pouch_", True)
    test = read_rows(BASE2 / "test", None, False)
    groups = {"pen_notebook_validation_base_images": val_base,
              "pouch_validation_same_item_session": val_pouch,
              "original_pen_notebook_test_split": test}
    all_paths = list(dict.fromkeys(row["path"] for rows in groups.values() for row in rows))
    path_index = {path: i for i, path in enumerate(all_paths)}
    payload = {
        "metric": "greedy per-class match at IoU >= 0.50",
        "class_ids": CLASS_IDS,
        "thresholds": THRESHOLDS,
        "scope_note": "Pouch validation is the 27 same-item/same-session held-out images. The original test split contains 65 images, 94 Pen boxes, and no pouch or notebook positives.",
        "models": {},
    }
    for name, checkpoint in MODELS.items():
        model = YOLO(str(checkpoint))
        output = {}
        predictions = list(model.predict([str(p) for p in all_paths], imgsz=640, conf=0.01,
                                         iou=0.5, device=0, batch=8, rect=False, verbose=False))
        prediction_rows = {}
        for path, result in zip(all_paths, predictions):
            detections = []
            for box in result.boxes:
                cls_id = int(box.cls[0])
                name_by_id = {376: "pen", 377: "pencil_pouch", 601: "paper_notebook"}
                if cls_id not in name_by_id:
                    continue
                detections.append({"name": name_by_id[cls_id], "score": float(box.conf[0]),
                                   "box": list(map(float, box.xyxy[0].tolist()))})
            prediction_rows[path] = detections
        for group_name, group_rows in groups.items():
            rows = []
            for row in group_rows:
                rows.append({**row, "pred": prediction_rows[row["path"]]})
            metrics = {cls_name: match(rows, cls_name, THRESHOLDS[cls_name])
                       for cls_name in CLASS_IDS}
            if group_name == "pouch_validation_same_item_session":
                metrics["pencil_pouch_threshold_curve"] = [
                    match(rows, "pencil_pouch", t) for t in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60)
                ]
            if group_name == "original_pen_notebook_test_split":
                metrics["pencil_pouch_threshold_curve"] = [
                    match(rows, "pencil_pouch", t) for t in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60)
                ]
            output[group_name] = metrics
        payload["models"][name] = output
        print(name, json.dumps(output, ensure_ascii=False), flush=True)
    target = ROOT / "runs/pen_notebook_pouch_full_oiv7_v10/yolo_comparison.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print("saved", target)


if __name__ == "__main__":
    main()
