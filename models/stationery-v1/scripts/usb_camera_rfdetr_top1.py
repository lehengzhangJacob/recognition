#!/usr/bin/env python3
"""USB camera, CPU RF-DETR: force one result from the highest of three class scores."""
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


def classify(session: ort.InferenceSession, frame: np.ndarray) -> tuple[int, float, tuple[int, int, int, int], list[float]]:
    frame_h, frame_w = frame.shape[:2]
    model_h, model_w = session.get_inputs()[0].shape[-2:]
    tensor = prepare(frame, model_h, model_w)
    boxes, logits = session.run(None, {session.get_inputs()[0].name: tensor})
    scores = 1 / (1 + np.exp(-np.clip(logits[0, :, :3], -88, 88)))
    # Exactly the same forced top-1 rule as onnx_cpu_top1_3class.json:
    # maximum over queries for each class, then maximum over the three classes.
    query_id, class_id = np.unravel_index(int(np.argmax(scores)), scores.shape)
    cx, cy, bw, bh = boxes[0, query_id]
    box = (max(0, int((cx - bw / 2) * frame_w)),
           max(0, int((cy - bh / 2) * frame_h)),
           min(frame_w - 1, int((cx + bw / 2) * frame_w)),
           min(frame_h - 1, int((cy + bh / 2) * frame_h)))
    return int(class_id), float(scores[query_id, class_id]), box, scores.max(axis=0).tolist()


def draw(frame: np.ndarray, prediction: tuple[int, float, tuple[int, ...], list[float]]) -> np.ndarray:
    class_id, score, (x1, y1, x2, y2), class_scores = prediction
    result = frame.copy()
    color = COLORS[class_id]
    if x2 > x1 and y2 > y1:
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
    cv2.putText(result, f"TOP 1: {NAMES[class_id]} {score:.2f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, .85, color, 2, cv2.LINE_AA)
    cv2.putText(result, "  ".join(f"{name}: {value:.2f}" for name, value in zip(NAMES, class_scores)),
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="RF-DETR 3-class ONNX path")
    parser.add_argument("--camera", default="0", help="camera index (0, 1, ...) or /dev/videoN")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--image", type=Path, help="optional still image for checking the setup")
    parser.add_argument("--save", type=Path, help="save the annotated still image")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"model not found: {args.model}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=["CPUExecutionProvider"])
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            parser.error(f"image not readable: {args.image}")
        prediction = classify(session, frame)
        print(NAMES[prediction[0]], round(prediction[1], 3),
              {name: round(score, 3) for name, score in zip(NAMES, prediction[3])})
        if args.save:
            cv2.imwrite(str(args.save), draw(frame, prediction))
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
            view = draw(frame, classify(session, frame))
            cv2.imshow("RF-DETR: forced top-1 of three classes", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                cv2.imwrite("camera_top1.jpg", view)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
