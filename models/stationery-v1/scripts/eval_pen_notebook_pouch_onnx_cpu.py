#!/usr/bin/env python3
"""Evaluate exported YOLO OIV7 and RF-DETR ONNX models on CPU."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "vending_machine_datasets/pen_notebook_pouch_rfdetr_3class"
YOLO_ONNX = ROOT / "runs/pen_notebook_pouch_full_oiv7_v10/weights/best.onnx"
RF_ONNX = ROOT / "runs/pen_notebook_pouch_rfdetr_nano_v1/onnx_export/rfdetr-nano-3class.onnx"
# Eval id : (full OIV7 YOLO id, dedicated RF-DETR id, label)
CLASS_INFO = {0: (376, 0, "pen"), 1: (601, 1, "paper_notebook"), 2: (377, 2, "pencil_pouch")}
THRESHOLDS = {"pen": 0.30, "paper_notebook": 0.45, "pencil_pouch": 0.30}


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-9)


def load_gt(folder):
    coco = json.loads((folder / "_annotations.coco.json").read_text())
    gt = defaultdict(lambda: defaultdict(list))
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        gt[ann["image_id"]][ann["category_id"] - 1].append([x, y, x + w, y + h])
    return coco, gt


def prep_rf(image, h_model, w_model):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w_model, h_model), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    mean = np.array([.485, .456, .406], dtype=np.float32)[None, :, None, None]
    std = np.array([.229, .224, .225], dtype=np.float32)[None, :, None, None]
    return np.ascontiguousarray((x - mean) / std)


def match(rows, class_id, name, threshold):
    tp = fp = targets = 0
    for row in rows:
        gt = row["gt"].get(class_id, [])
        targets += len(gt)
        preds = sorted((p for p in row["pred"] if p["name"] == name and p["score"] >= threshold),
                       key=lambda p: p["score"], reverse=True)
        used = set()
        for pred in preds:
            overlap, index = max(((iou(pred["box"], box), i) for i, box in enumerate(gt) if i not in used),
                                 default=(0.0, -1))
            if overlap >= .5:
                tp += 1
                used.add(index)
            else:
                fp += 1
    return {"targets": targets, "tp": tp, "fp": fp, "fn": targets - tp,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(targets, 1),
            "threshold": threshold, "iou": .5}


def main():
    cv2.setNumThreads(0)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    yolo = ort.InferenceSession(str(YOLO_ONNX), sess_options=options, providers=["CPUExecutionProvider"])
    rf = ort.InferenceSession(str(RF_ONNX), sess_options=options, providers=["CPUExecutionProvider"])
    rf_h, rf_w = rf.get_inputs()[0].shape[-2:]
    groups = {}
    for split in ("valid", "test"):
        folder = DATA / split
        coco, gt = load_gt(folder)
        rows = []
        for image_record in coco["images"]:
            name = image_record["file_name"]
            if split == "valid" and name.startswith("pouch_"):
                group = "pouch_validation_same_item_session"
            elif split == "valid":
                group = "pen_notebook_validation_base_images"
            else:
                group = "original_pen_notebook_test_split"
            image = cv2.imread(str(folder / name))
            if image is None:
                raise FileNotFoundError(folder / name)
            ih, iw = image.shape[:2]

            # Full 602-class YOLO output, retaining only the three target channels.
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            size = int(yolo.get_inputs()[0].shape[-1])
            scale = min(size / ih, size / iw)
            nw, nh = int(round(iw * scale)), int(round(ih * scale))
            resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((size, size, 3), 114, dtype=np.uint8)
            left, top = (size - nw) // 2, (size - nh) // 2
            canvas[top:top + nh, left:left + nw] = resized
            yolo_input = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
            yout = yolo.run(None, {yolo.get_inputs()[0].name: yolo_input})[0][0]
            # Exported raw YOLO output is [channels, anchors] with xywh then class scores.
            if yout.shape[0] < yout.shape[1]:
                yout = yout.T
            yboxes, yscores = yout[:, :4], yout[:, 4:]
            ypred = []
            selected_class_ids = [info[0] for info in CLASS_INFO.values()]
            selected_scores = yscores[:, selected_class_ids]
            selected_local = selected_scores.argmax(axis=1)
            selected_conf = selected_scores[np.arange(len(yscores)), selected_local]
            for local_id, (_, _, cls_name) in enumerate(CLASS_INFO.values()):
                scores = selected_conf
                keep = np.flatnonzero((selected_local == local_id) & (scores >= .01))
                # Per-class greedy NMS on the resized network canvas.
                order = keep[np.argsort(scores[keep])[::-1]]
                retained = []
                for idx in order:
                    box = [yboxes[idx, 0] - yboxes[idx, 2] / 2, yboxes[idx, 1] - yboxes[idx, 3] / 2,
                           yboxes[idx, 0] + yboxes[idx, 2] / 2, yboxes[idx, 1] + yboxes[idx, 3] / 2]
                    if all(iou(box, other) <= .5 for other in retained):
                        retained.append(box)
                        px = [(box[0] - left) / scale, (box[1] - top) / scale,
                              (box[2] - left) / scale, (box[3] - top) / scale]
                        ypred.append({"name": cls_name, "score": float(scores[idx]),
                                      "box": [max(0, min(iw, px[0])), max(0, min(ih, px[1])),
                                              max(0, min(iw, px[2])), max(0, min(ih, px[3]))]})

            # RF-DETR ONNX returns normalized cxcywh boxes and class logits.
            rx = prep_rf(image, rf_h, rf_w)
            boxes, logits = rf.run(None, {rf.get_inputs()[0].name: rx})
            scores = 1 / (1 + np.exp(-np.clip(logits[0, :, :3], -88, 88)))
            rpred = []
            for q in range(len(boxes[0])):
                cx, cy, bw, bh = boxes[0, q]
                xyxy = [(cx - bw / 2) * iw, (cy - bh / 2) * ih,
                        (cx + bw / 2) * iw, (cy + bh / 2) * ih]
                for cls_id, (_, rf_class, cls_name) in CLASS_INFO.items():
                    score = float(scores[q, rf_class])
                    if score >= .01:
                        rpred.append({"name": cls_name, "score": score, "box": xyxy})
            row = {"group": group, "name": name, "gt": gt[image_record["id"]],
                   "predictions": {"yolo": ypred, "rfdetr": rpred}}
            rows.append(row)
            groups.setdefault(group, []).append(row)

    result = {"models": {}, "thresholds": THRESHOLDS,
              "scope": "CPU ONNX runtime outputs; box match IoU >= 0.5. Pouch validation is same item/session; old test has no pouch positives.",
              "cpu": {}}
    for model_name in ("yolo", "rfdetr"):
        result["models"][model_name] = {}
        for group_name, rows in groups.items():
            converted = [{"gt": r["gt"], "pred": r["predictions"][model_name]} for r in rows]
            result["models"][model_name][group_name] = {
                name: match(converted, cls_id, name, THRESHOLDS[name])
                for cls_id, (_, _, name) in CLASS_INFO.items()
            }
    output = ROOT / "runs/pen_notebook_pouch_full_oiv7_v10/onnx_cpu_evaluation.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("saved", output)


if __name__ == "__main__":
    main()
