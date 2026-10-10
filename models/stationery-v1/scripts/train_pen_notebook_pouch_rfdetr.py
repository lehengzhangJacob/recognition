#!/usr/bin/env python3
"""Train a dedicated Pen/Notebook/Pencil-pouch RF-DETR Nano model."""
import argparse
import json
from pathlib import Path

import torch
from rfdetr import RFDETRNano

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=ROOT / 'vending_machine_datasets/pen_notebook_pouch_rfdetr_3class')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/pen_notebook_pouch_rfdetr_nano_v1')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--resolution', type=int, default=576)
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserving existing run: {args.output}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; set the documented local NVIDIA library path before training.')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True)
    (args.output / 'experiment.json').write_text(json.dumps({
        'model': 'RF-DETR Nano official COCO pretrained weights',
        'classes': ['pen', 'paper_notebook', 'pencil_pouch'],
        'dataset': str(args.dataset), 'epochs': args.epochs,
        'resolution': args.resolution, 'batch_size': args.batch,
        'gradient_accumulation': 4, 'learning_rate': 1e-4,
        'environment': 'pen-notebook-rfdetr',
        'notes': 'Dedicated three-class output; does not retain the original COCO 80 classes or OIV7 601 classes. Pouch validation is same-item/same-session.',
    }, indent=2))
    model = RFDETRNano()
    model.train(dataset_dir=str(args.dataset), output_dir=str(args.output),
                epochs=args.epochs, resolution=args.resolution,
                batch_size=args.batch, grad_accum_steps=4, lr=1e-4,
                early_stopping=True, early_stopping_patience=8)


if __name__ == '__main__':
    main()
