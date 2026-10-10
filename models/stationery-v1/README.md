# Pen / notebook / pencil pouch detectors — v1

Training artifacts produced 2026-10-10. Dataset images are not included; the source photos remain in the local dataset directories. Evaluation JSON files contain aggregate counts and evaluation scope.

## Models

| Model | Initialization and output classes | Files |
|---|---|---|
| YOLOv8n OIV7, 602 classes | Fine-tuned from the prior full OIV7 pen/notebook checkpoint. Keeps the original 601 class outputs and adds `paper_notebook` as class 601; pouch uses OIV7 `Pencil case` class 377 and pen remains class 376. | `yolov8n-oiv7-602/best.pt`, `best.onnx` |
| RF-DETR Nano, 3 classes | Fine-tuned from official COCO-pretrained RF-DETR Nano. Outputs only `pen`, `paper_notebook`, `pencil_pouch`; does not retain COCO-80 or OIV7 classes. | `rf-detr-nano-3class/checkpoint_best_total.pth` and `model.onnx` (reassemble from the `.part-*` files with `scripts/assemble_rf_detr_artifacts.py`) |

The RF-DETR checkpoint and ONNX file are split into 70 MB parts to fit GitHub file limits. Run `python scripts/assemble_rf_detr_artifacts.py` from this version directory to reconstruct them and verify SHA-256 hashes. For CPU inference, use the ONNX files with ONNX Runtime's CPU execution provider. YOLO ONNX emits raw model outputs and requires YOLO box decoding/NMS; the RF-DETR ONNX emits normalized boxes and logits. The evaluation script in `scripts/eval_pen_notebook_pouch_onnx_cpu.py` shows preprocessing, output decoding, class selection and matching. Its dataset paths refer to the original local project layout and must be adjusted if you reproduce evaluation elsewhere.

## Evaluation snapshot

All counts below use greedy per-class box matching at IoU ≥ 0.50 and confidence thresholds pen 0.30, notebook 0.45, pouch 0.30. Refer to the JSON files for the complete metrics.

| Split / class | YOLO 602 (TP / FP / FN) | RF-DETR 3-class (TP / FP / FN) |
|---|---:|---:|
| Same-session pouch validation, pouch (27 targets) | 23 / 2 / 4 | 26 / 0 / 1 |
| Base validation, pen (48 targets) | 33 / 6 / 15 | 43 / 9 / 5 |
| Base validation, notebook (23 targets) | 18 / 4 / 5 | 19 / 0 / 4 |
| Original test split, pen (94 targets) | 58 / 31 / 36 | 84 / 27 / 10 |
| Original test split, pouch (0 targets) | 10 false detections | 3 false detections |

The 27 pouch validation photos are held out from training, but show the same physical pouch in the same photo session. These results measure performance on that photographed item/session and do not establish generalization to other pouch designs, lighting or settings. The original test split has no pouch positives, so its pouch counts only indicate false detections there. Pen and notebook evaluation remains based on the pre-existing splits; compare model results with these dataset and threshold limits in mind.


## Forced three-way image classification

For each image, take the highest raw model score for each of the three target classes across all candidate boxes, then choose the class with the highest score. No confidence threshold or box overlap requirement is applied. This measures the requested closed-set, image-level decision, assuming the image contains one of the three categories.

| Dataset | YOLO correct / total | RF-DETR correct / total |
|---|---:|---:|
| Validation, one label per image (44 pen, 19 notebook, 27 pouch) | 78 / 90 (86.7%) | 89 / 90 (98.9%) |
| Original test, pen-only images | 55 / 62 (88.7%) | 60 / 62 (96.8%) |

Four validation images contain both a pen and a notebook, so they are excluded from the strict one-label denominator; both models chose one of the two annotated classes on all four. Three original test images have no annotated target; forced classification still assigns a category to them. On the single-label validation images, RF-DETR classified 44/44 pen, 18/19 notebook and 27/27 pouch. The one error was a notebook classified as pen. The pouch examples remain the same physical item and photo session, and the separate test set has no notebook or pouch positives. These figures do not establish ≥95% accuracy on new objects, backgrounds or real video. See `onnx_cpu_top1_3class.json` for image-level scores and `scripts/eval_pen_notebook_pouch_top1.py` for the exact procedure.

## Training provenance

- YOLOv8n OIV7 602: initialized from the preceding full OIV7 pen/notebook run (`pen_notebook_full_oiv7_v9`), then fine-tuned for 30 epochs at 640px with the pouch annotations mapped to OIV7 `Pencil case`. See `yolov8n-oiv7-602/training_args.yaml` and the evaluation JSON.
- RF-DETR Nano: official COCO pretrained initialization, 25 epochs, 576px resolution, effective batch 16 (batch 4, gradient accumulation 4), three output classes. See `rf-detr-nano-3class/experiment.json` and `evaluation_iou50.json`.
- Source dataset counts: the two-class pen/notebook set had 479 train, 67 validation, and 65 test images; 108 pouch images were added to training and 27 to validation. The images and labels are not included in this repository.


## USB / Type-C camera on CPU

These two standalone Python programs use the RF-DETR three-class ONNX file with ONNX Runtime's CPU provider. A Type-C UVC camera is opened through OpenCV just like any other USB camera. Install dependencies with `python -m pip install numpy opencv-python onnxruntime`. From this version directory, first run `python scripts/assemble_rf_detr_artifacts.py` to reconstruct `rf-detr-nano-3class/model.onnx` from the checked-in parts.

- Standard pattern/object detection: `python scripts/usb_camera_rfdetr_detect.py --camera 0` — shows every qualifying box for pen, notebook and pencil pouch. Default class thresholds are 0.30, 0.45 and 0.30. Set `--pen-conf`, `--notebook-conf` or `--pouch-conf` to adjust them.
- Forced relative top-1: `python scripts/usb_camera_rfdetr_top1.py --camera 0` — selects the highest score among the three classes for the whole frame and draws its box. It always picks a class, including on frames with no target. This is the same rule used in the 98.9% image-level validation experiment.

Both request 1280×720 capture by default; use `--width` and `--height` to change it. Camera indices differ across computers, so try `--camera 1` or a Linux device path such as `--camera /dev/video2` if camera 0 is not the Type-C device. `q` or Esc quits; `s` saves an annotated frame. `--model PATH` accepts an ONNX file elsewhere. To test without a camera, add `--image PATH --save annotated.jpg`. Camera capture needs a local desktop session that can display an OpenCV window. Actual frame rate depends on the computer CPU.

## Reproduction scripts

The scripts used to prepare datasets, fine-tune and evaluate are under `scripts/`. They expect the original project's dataset directories and Python environments. The model files are ready-to-run inference artifacts; training scripts require the corresponding frameworks, pretrained source weights and local datasets.
