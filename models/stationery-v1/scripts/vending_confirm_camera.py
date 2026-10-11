#!/usr/bin/env python3
"""Confirm one of three products from a camera and write one SQLite row per dispense."""
from __future__ import annotations

import argparse
import sqlite3
import uuid
from collections import Counter, deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_MODEL = ROOT / "rf-detr-nano-3class/model.onnx"
LOCAL_MODEL = ROOT / "runs/pen_notebook_pouch_rfdetr_nano_v1/onnx_export/rfdetr-nano-3class.onnx"
DEFAULT_MODEL = PUBLISHED_MODEL if PUBLISHED_MODEL.is_file() else LOCAL_MODEL
NAMES = ("pen", "notebook", "pencil_pouch")
COLORS = ((50, 220, 50), (255, 180, 40), (60, 80, 255))


@dataclass(frozen=True)
class Candidate:
    class_id: int
    score: float
    margin: float
    box: tuple[int, int, int, int]
    accepted: bool


def parse_roi(value: str) -> tuple[float, float, float, float]:
    try:
        roi = tuple(float(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("ROI must be x1,y1,x2,y2 in 0..1") from error
    if len(roi) != 4 or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        raise argparse.ArgumentTypeError("ROI must be x1,y1,x2,y2 in 0..1")
    return roi


def prepare(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    tensor = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    mean = np.array([.485, .456, .406], dtype=np.float32)[None, :, None, None]
    std = np.array([.229, .224, .225], dtype=np.float32)[None, :, None, None]
    return np.ascontiguousarray((tensor - mean) / std)


def infer(session: ort.InferenceSession, frame: np.ndarray,
          roi: tuple[float, float, float, float], thresholds: tuple[float, float, float],
          min_margin: float) -> Candidate | None:
    frame_h, frame_w = frame.shape[:2]
    model_h, model_w = session.get_inputs()[0].shape[-2:]
    boxes, logits = session.run(None, {session.get_inputs()[0].name: prepare(frame, model_h, model_w)})
    scores = 1 / (1 + np.exp(-np.clip(logits[0, :, :3], -88, 88)))
    boxes = boxes[0]
    # A candidate must have its center inside the visible confirmation region.
    valid = ((boxes[:, 0] >= roi[0]) & (boxes[:, 0] <= roi[2]) &
             (boxes[:, 1] >= roi[1]) & (boxes[:, 1] <= roi[3]) &
             (boxes[:, 2] > 0) & (boxes[:, 3] > 0))
    if not np.any(valid):
        return None
    valid_scores = np.where(valid[:, None], scores, -1.0)
    query_id, class_id = np.unravel_index(int(np.argmax(valid_scores)), valid_scores.shape)
    score = float(valid_scores[query_id, class_id])
    if score < 0:
        return None
    class_max = valid_scores.max(axis=0)
    second = float(np.partition(class_max, -2)[-2])
    margin = score - second
    cx, cy, bw, bh = boxes[query_id]
    box = (max(0, int((cx - bw / 2) * frame_w)),
           max(0, int((cy - bh / 2) * frame_h)),
           min(frame_w - 1, int((cx + bw / 2) * frame_w)),
           min(frame_h - 1, int((cy + bh / 2) * frame_h)))
    accepted = (box[2] > box[0] and box[3] > box[1] and
                score >= thresholds[class_id] and margin >= min_margin)
    return Candidate(int(class_id), score, margin, box, accepted)


def changed_fraction(frame: np.ndarray, baseline: np.ndarray,
                     box: tuple[int, int, int, int], pixel_delta: int) -> float:
    """Fraction of changed pixels within the detected box since arming."""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return 0.0
    current = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    previous = cv2.cvtColor(baseline[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    current = cv2.GaussianBlur(current, (5, 5), 0)
    previous = cv2.GaussianBlur(previous, (5, 5), 0)
    return float(np.mean(cv2.absdiff(current, previous) >= pixel_delta))


def open_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(path)
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("""
        CREATE TABLE IF NOT EXISTS confirmations (
            dispense_id TEXT PRIMARY KEY,
            confirmed_at_utc TEXT NOT NULL,
            product_type TEXT NOT NULL CHECK (product_type IN ('pen', 'notebook', 'pencil_pouch')),
            confidence REAL NOT NULL,
            score_margin REAL NOT NULL,
            box_x1 INTEGER NOT NULL,
            box_y1 INTEGER NOT NULL,
            box_x2 INTEGER NOT NULL,
            box_y2 INTEGER NOT NULL,
            model_version TEXT NOT NULL,
            camera TEXT NOT NULL
        )
    """)
    database.commit()
    return database


class ConfirmController:
    def __init__(self, database: sqlite3.Connection, camera: str, stable_hits: int = 5,
                 window: int = 7, clear_frames: int = 8):
        if not (1 <= stable_hits <= window and clear_frames >= 1):
            raise ValueError("invalid confirmation window")
        self.database = database
        self.camera = camera
        self.stable_hits = stable_hits
        self.window = window
        self.clear_frames = clear_frames
        self.history: deque[int | None] = deque(maxlen=window)
        self.state = "IDLE"
        self.dispense_id: str | None = None
        self.clear_count = 0
        self.last_confirmation = ""

    def arm(self, dispense_id: str | None = None) -> bool:
        if self.state != "IDLE":
            return False
        self.dispense_id = dispense_id or uuid.uuid4().hex
        self.history.clear()
        self.clear_count = 0
        self.state = "ARMED"
        return True

    def step(self, candidate: Candidate | None) -> bool:
        accepted = candidate if candidate is not None and candidate.accepted else None
        if self.state == "ARMED":
            self.history.append(accepted.class_id if accepted else None)
            if accepted is None:
                return False
            hits = Counter(self.history)[accepted.class_id]
            if hits < self.stable_hits:
                return False
            assert self.dispense_id is not None
            with self.database:
                cursor = self.database.execute("""
                    INSERT OR IGNORE INTO confirmations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (self.dispense_id, datetime.now(timezone.utc).isoformat(),
                      NAMES[accepted.class_id], accepted.score, accepted.margin,
                      *accepted.box, "rf-detr-nano-3class-v1", self.camera))
            self.state = "CONFIRMED"
            self.clear_count = 0
            self.last_confirmation = f"{NAMES[accepted.class_id]} {accepted.score:.2f} / {self.dispense_id[:8]}"
            return cursor.rowcount == 1
        if self.state == "CONFIRMED":
            self.clear_count = self.clear_count + 1 if accepted is None else 0
            if self.clear_count >= self.clear_frames:
                self.state = "IDLE"
                self.dispense_id = None
                self.history.clear()
        return False


def draw(frame: np.ndarray, roi: tuple[float, float, float, float],
         candidate: Candidate | None, controller: ConfirmController,
         change: float) -> np.ndarray:
    view = frame.copy()
    height, width = view.shape[:2]
    cv2.rectangle(view, (int(roi[0] * width), int(roi[1] * height)),
                  (int(roi[2] * width), int(roi[3] * height)), (180, 180, 180), 2)
    if candidate is not None:
        x1, y1, x2, y2 = candidate.box
        color = COLORS[candidate.class_id] if candidate.accepted else (130, 130, 130)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
            cv2.putText(view, f"{NAMES[candidate.class_id]} {candidate.score:.2f} margin {candidate.margin:.2f} change {change:.0%}",
                        (x1, max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
    cv2.rectangle(view, (0, 0), (width, 100), (0, 0, 0), -1)
    cv2.putText(view, f"{controller.state} | SPACE: arm | Q: quit", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    if controller.state == "ARMED":
        cv2.putText(view, f"Stable frames: {len(controller.history)}/{controller.window}; need {controller.stable_hits} matching",
                    (12, 59), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
    elif controller.state == "CONFIRMED":
        cv2.putText(view, f"SAVED: {controller.last_confirmation} | clear view to reset",
                    (12, 59), cv2.FONT_HERSHEY_SIMPLEX, .6, (60, 220, 60), 2)
    else:
        cv2.putText(view, "Remove previous item, then press SPACE to confirm next dispense",
                    (12, 59), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
    return view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--camera", default="0", help="camera index, e.g. 0 for laptop front camera")
    parser.add_argument("--db", type=Path, default=ROOT / "runtime/confirmations.sqlite3")
    parser.add_argument("--list", action="store_true", help="show the latest database records without opening the camera")
    parser.add_argument("--roi", type=parse_roi, default=(.15, .15, .85, .85),
                        help="normalized confirmation area x1,y1,x2,y2")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--pen-conf", type=float, default=.30)
    parser.add_argument("--notebook-conf", type=float, default=.45)
    parser.add_argument("--pouch-conf", type=float, default=.30)
    parser.add_argument("--min-margin", type=float, default=.15)
    parser.add_argument("--stable-hits", type=int, default=5)
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--clear-frames", type=int, default=8)
    parser.add_argument("--min-change", type=float, default=.08,
                        help="minimum changed-pixel fraction within the box since arming")
    parser.add_argument("--pixel-delta", type=int, default=25,
                        help="grayscale pixel difference considered a change")
    args = parser.parse_args()
    if args.list:
        if not args.db.is_file():
            parser.error(f"database not found: {args.db}")
        with sqlite3.connect(args.db) as database:
            rows = database.execute("""
                SELECT dispense_id, confirmed_at_utc, product_type, confidence, score_margin
                FROM confirmations ORDER BY confirmed_at_utc DESC LIMIT 30
            """).fetchall()
        for row in rows:
            print(f"{row[1]}  {row[2]:<13} score={row[3]:.3f} margin={row[4]:.3f} id={row[0]}")
        if not rows:
            print("No confirmations yet")
        return
    if not args.model.is_file():
        parser.error(f"model not found: {args.model}; reconstruct it or use --model")
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=["CPUExecutionProvider"])
    database = open_database(args.db)
    controller = ConfirmController(database, args.camera, args.stable_hits, args.window, args.clear_frames)
    camera = int(args.camera) if args.camera.isdecimal() else args.camera
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        database.close()
        parser.error(f"camera cannot be opened: {args.camera}; try --camera 1")
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    print(f"Camera: {args.camera}, resolution: {int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"Database: {args.db.resolve()}")
    print("Press SPACE to arm one confirmation, Q or Esc to quit")
    thresholds = (args.pen_conf, args.notebook_conf, args.pouch_conf)
    baseline: np.ndarray | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("camera frame could not be read")
            candidate = infer(session, frame, args.roi, thresholds, args.min_margin)
            change = 0.0
            if candidate is not None and baseline is not None:
                change = changed_fraction(frame, baseline, candidate.box, args.pixel_delta)
                if change < args.min_change:
                    candidate = replace(candidate, accepted=False)
            if controller.step(candidate):
                print(f"CONFIRMED: {controller.last_confirmation}")
            if controller.state == "IDLE":
                baseline = None
            cv2.imshow("Vending product confirmation", draw(frame, args.roi, candidate, controller, change))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == 32 and controller.arm():
                baseline = frame.copy()
                print(f"ARMED: {controller.dispense_id}")
    finally:
        capture.release()
        database.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
