#!/usr/bin/env python3
"""Evaluate forced image-level top-1 among pen, notebook, pouch on CPU ONNX."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from eval_pen_notebook_pouch_onnx_cpu import DATA, RF_ONNX, YOLO_ONNX, prep_rf

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("pen", "paper_notebook", "pencil_pouch")
YOLO_IDS = (376, 601, 377)


def yolo_scores(session: ort.InferenceSession, image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    size = int(session.get_inputs()[0].shape[-1])
    scale = min(size / height, size / width)
    new_width, new_height = round(width * scale), round(height * scale)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - new_height) // 2, (size - new_width) // 2
    canvas[top:top + new_height, left:left + new_width] = resized
    network_input = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255)
    output = session.run(None, {session.get_inputs()[0].name: network_input})[0][0]
    if output.shape[0] < output.shape[1]:
        output = output.T
    return output[:, np.array(YOLO_IDS) + 4].max(axis=0)


def rf_scores(session: ort.InferenceSession, image: np.ndarray) -> np.ndarray:
    height, width = session.get_inputs()[0].shape[-2:]
    network_input = prep_rf(image, height, width)
    _, logits = session.run(None, {session.get_inputs()[0].name: network_input})
    return (1 / (1 + np.exp(-np.clip(logits[0, :, :3], -88, 88)))).max(axis=0)


def summarize(rows: list[dict]) -> dict:
    single = [row for row in rows if len(row["truth"]) == 1]
    mixed = [row for row in rows if len(row["truth"]) > 1]
    empty = [row for row in rows if not row["truth"]]
    result = {
        "single_label_images": len(single),
        "mixed_label_images": len(mixed),
        "empty_label_images": len(empty),
        "models": {},
    }
    for model in ("yolo", "rfdetr"):
        correct = sum(row["predictions"][model]["top1"] == row["truth"][0] for row in single)
        confusion = {name: {other: 0 for other in NAMES} for name in NAMES}
        for row in single:
            confusion[row["truth"][0]][row["predictions"][model]["top1"]] += 1
        result["models"][model] = {
            "correct": correct,
            "total": len(single),
            "accuracy": correct / len(single) if single else None,
            "confusion": confusion,
            "mixed_top1_matches_any_label": sum(row["predictions"][model]["top1"] in row["truth"] for row in mixed),
            "empty_forced_choice": Counter(row["predictions"][model]["top1"] for row in empty),
            "wrong_images": [
                {"file": row["file"], "truth": row["truth"][0], **row["predictions"][model]}
                for row in single if row["predictions"][model]["top1"] != row["truth"][0]
            ],
        }
        result["models"][model]["empty_forced_choice"] = dict(result["models"][model]["empty_forced_choice"])
    return result


def main() -> None:
    cv2.setNumThreads(0)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    sessions = {
        "yolo": ort.InferenceSession(str(YOLO_ONNX), sess_options=options, providers=["CPUExecutionProvider"]),
        "rfdetr": ort.InferenceSession(str(RF_ONNX), sess_options=options, providers=["CPUExecutionProvider"]),
    }
    result = {"rule": "For each image, take each target class's maximum raw model score across all candidates, then choose the class with the highest score. No confidence threshold or IoU requirement.",
              "class_order": list(NAMES), "datasets": {}, "rows": []}
    for split in ("valid", "test"):
        folder = DATA / split
        coco = json.loads((folder / "_annotations.coco.json").read_text())
        ground_truth: dict[int, set[int]] = defaultdict(set)
        for annotation in coco["annotations"]:
            ground_truth[annotation["image_id"]].add(annotation["category_id"] - 1)
        split_rows = []
        for image_record in coco["images"]:
            path = folder / image_record["file_name"]
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            scores = {"yolo": yolo_scores(sessions["yolo"], image),
                      "rfdetr": rf_scores(sessions["rfdetr"], image)}
            row = {
                "split": split,
                "file": image_record["file_name"],
                "truth": [NAMES[index] for index in sorted(ground_truth[image_record["id"]])],
                "predictions": {model: {"top1": NAMES[int(np.argmax(values))],
                                        "scores": {name: float(score) for name, score in zip(NAMES, values)}}
                                for model, values in scores.items()},
            }
            split_rows.append(row)
        result["rows"].extend(split_rows)
        if split == "valid":
            base = [row for row in split_rows if not row["file"].startswith("pouch_")]
            pouch = [row for row in split_rows if row["file"].startswith("pouch_")]
            result["datasets"]["validation_base"] = summarize(base)
            result["datasets"]["validation_pouch_same_item_session"] = summarize(pouch)
            result["datasets"]["validation_all"] = summarize(split_rows)
        else:
            result["datasets"]["original_test"] = summarize(split_rows)
    output = ROOT / "runs/pen_notebook_pouch_full_oiv7_v10/onnx_cpu_top1_3class.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["datasets"], ensure_ascii=False, indent=2))
    print("saved", output)


if __name__ == "__main__":
    main()
