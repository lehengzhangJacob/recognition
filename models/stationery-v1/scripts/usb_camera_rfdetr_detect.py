#!/usr/bin/env python3
"""USB camera, CPU RF-DETR: draw every pen/notebook/pouch detection above threshold."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

NAMES = ("pen", "notebook", "pencil_pouch")
COLORS = ((50, 220, 50), (255, 180, 40), (60, 80, 255))  # BGR
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_MODEL = PROJECT_ROOT / "rf-detr-nano-3class/model.onnx"
LOCAL_MODEL = PROJECT_ROOT / "runs/pen_notebook_pouch_rfdetr_nano_v1/onnx_export/rfdetr-nano-3class.onnx"
DEFAULT_MODEL = PUBLISHED_MODEL if PUBLISHED_MODEL.is_file() else LOCAL_MODEL


def prepare(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    tensor = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    mean = np.array([.485, .456, .406], dtype=np.float32)[None, :, None, None]
    std = np.array([.229, .224, .225], dtype=np.float32)[None, :, None, None]
    return np.ascontiguousarray((tensor - mean) / std)


def box_pixels(cxcywh: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = cxcywh
    x1, y1 = int((cx - bw / 2) * width), int((cy - bh / 2) * height)
    x2, y2 = int((cx + bw / 2) * width), int((cy + bh / 2) * height)
    return max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)


def iou(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1)


def detect(session: ort.InferenceSession, frame: np.ndarray, thresholds: tuple[float, ...]) -> list[tuple[int, float, tuple[int, ...]]]:
    frame_h, frame_w = frame.shape[:2]
    model_h, model_w = session.get_inputs()[0].shape[-2:]
    tensor = prepare(frame, model_h, model_w)
    boxes, logits = session.run(None, {session.get_inputs()[0].name: tensor})
    scores = 1 / (1 + np.exp(-np.clip(logits[0, :, :3], -88, 88)))
    candidates = []
    for query, class_scores in enumerate(scores):
        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id])
        if score < thresholds[class_id]:
            continue
        box = box_pixels(boxes[0, query], frame_w, frame_h)
        if box[2] > box[0] and box[3] > box[1]:
            candidates.append((class_id, score, box))
    kept: list[tuple[int, float, tuple[int, ...]]] = []
    for candidate in sorted(candidates, key=lambda item: item[1], reverse=True):
        class_id, _, box = candidate
        if all(class_id != previous_class or iou(box, previous_box) <= .5
               for previous_class, _, previous_box in kept):
            kept.append(candidate)
    return kept


def draw(frame: np.ndarray, predictions: list[tuple[int, float, tuple[int, ...]]]) -> np.ndarray:
    result = frame.copy()
    for class_id, score, (x1, y1, x2, y2) in predictions:
        color = COLORS[class_id]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        cv2.putText(result, f"{NAMES[class_id]} {score:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="RF-DETR 3-class ONNX path")
    parser.add_argument("--camera", default="0", help="camera index (0, 1, ...) or /dev/videoN")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--pen-conf", type=float, default=.30)
    parser.add_argument("--notebook-conf", type=float, default=.45)
    parser.add_argument("--pouch-conf", type=float, default=.30)
    parser.add_argument("--image", type=Path, help="optional still image for checking the setup")
    parser.add_argument("--save", type=Path, help="save the annotated still image")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"model not found: {args.model}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=["CPUExecutionProvider"])
    thresholds = (args.pen_conf, args.notebook_conf, args.pouch_conf)
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            parser.error(f"image not readable: {args.image}")
        predictions = detect(session, frame, thresholds)
        print([(NAMES[class_id], round(score, 3), box) for class_id, score, box in predictions])
        if args.save:
            cv2.imwrite(str(args.save), draw(frame, predictions))
        return
    camera = int(args.camera) if args.camera.isdecimal() else args.camera
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        parser.error(f"camera cannot be opened: {args.camera}; try 0, 1, or /dev/videoN")
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    print(f"Camera resolution: {int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("Press q or Esc to quit; press s to save a frame")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("camera frame could not be read")
            view = draw(frame, detect(session, frame, thresholds))
            cv2.imshow("RF-DETR: pen / notebook / pencil pouch", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                cv2.imwrite("camera_detection.jpg", view)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
