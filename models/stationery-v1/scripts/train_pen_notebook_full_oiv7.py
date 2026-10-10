#!/usr/bin/env python3
"""Full 602-class fine-tuning with partial-label supervision and old-model distillation."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

import train_pen_notebook_joint_oiv7 as joint

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'vending_machine_datasets/pen_notebook_colabeled'
DATA = ROOT / 'vending_machine_datasets/pen_notebook_full_oiv7_train'


class PartialLabelLoss(v8DetectionLoss):
    # Only the annotated classes are supervised as positives/negatives.
    # Other objects in these photos have not been comprehensively annotated.
    def __init__(self, model, teacher, distill_weight, include_pencil_pouch=False):
        super().__init__(model)
        self.teacher = teacher
        self.distill_weight = distill_weight
        self.include_pencil_pouch = include_pencil_pouch
        self.class_weights = torch.zeros(1, 1, 602, device=self.device)
        self.class_weights[..., 376] = 1
        self.class_weights[..., 601] = 1
        self.reviewed_negatives = {p.stem for p in (ROOT / 'vending_machine_datasets/notebook_negatives/train').glob('*.jpg')}
        review = DATA / 'reviewed_no_notebook.json'
        self.reviewed_broad_negatives = set(json.loads(review.read_text())['ids']) if review.exists() else set()

    def loss(self, preds, batch):
        self.class_weights = torch.zeros(batch['img'].shape[0], 1, 602, device=self.device)
        for k, path in enumerate(batch['im_file']):
            stem = Path(path).stem
            if self.include_pencil_pouch and stem.startswith('pouch_'):
                # These photos are exhaustively annotated only for Pencil case.
                # Other unannotated classes must remain unknown, not negatives.
                self.class_weights[k, :, 377] = 1
                continue
            if stem.startswith('oiv_replay_'):
                present = batch['cls'][batch['batch_idx'] == k].view(-1).long().unique()
                self.class_weights[k, :, present] = 1
                if stem[len('oiv_replay_'):].split('_hardrepeat')[0] in self.reviewed_broad_negatives:
                    self.class_weights[k, :, 601] = 1
                continue  # Other labels in these photos have not been verified.
            self.class_weights[k, :, 376] = 1
            if not stem.startswith('replay_') or stem[len('replay_'):] in self.reviewed_negatives:
                self.class_weights[k, :, 601] = 1
        values, logged = super().loss(preds, batch)
        with torch.no_grad():
            teacher_raw = self.teacher(batch['img'])[1]
        # KL on original class scores, except Pen which is deliberately adapted.
        # Sample every fourth anchor to control memory, retain all original classes.
        student = preds['scores'][:, :601, ::4].float()
        teacher = teacher_raw['scores'][:, :, ::4].float()
        student = torch.cat((student[:, :376], student[:, 377:]), dim=1)
        teacher = torch.cat((teacher[:, :376], teacher[:, 377:]), dim=1)
        probs = teacher.sigmoid()
        entropy = F.binary_cross_entropy_with_logits(teacher, probs, reduction='none')
        kl = (F.binary_cross_entropy_with_logits(student, probs, reduction='none') - entropy).sum(1).mean()
        # Keep generic features near the original representation while allowing updates.
        features = sum((s.float() - t.float()).square().mean() /
                       (t.float().square().mean().detach() + 1e-4)
                       for s, t in zip(preds['feats'], teacher_raw['feats'])) / 3
        # Uniform anchors dilute the few peaks that contain old objects. Preserve
        # each old class's strongest locations explicitly, including its logits.
        old_student = preds['scores'][:, :601].float()
        old_teacher = teacher_raw['scores'].float()
        ids = torch.arange(601, device=self.device)
        ids = ids[ids != 376]
        old_student, old_teacher = old_student[:, ids], old_teacher[:, ids]
        top_values, indices = old_teacher.topk(10, dim=2)
        selected_student = old_student.gather(2, indices)
        # Absent classes vastly outnumber positive old-class locations. Giving
        # every absent peak a baseline weight dilutes the actual old objects.
        weights = top_values.sigmoid() * (top_values.sigmoid() > 0.05)
        peak_loss = (F.smooth_l1_loss(selected_student, top_values, reduction='none') * weights).sum() / weights.sum().clamp_min(1)
        # Class-independent DFL box distributions also affect all old categories.
        mask = old_teacher.sigmoid().amax(1) > 0.1
        sb = preds['boxes'].float().reshape(-1, 4, 16, mask.shape[1])
        tb = teacher_raw['boxes'].float().reshape_as(sb)
        box_kl = F.kl_div(sb.log_softmax(2), tb.softmax(2), reduction='none').sum(2).mean(1)
        box_loss = (box_kl * mask).sum() / mask.sum().clamp_min(1)
        extra = self.distill_weight * (0.1 * kl + peak_loss + box_loss + 0.05 * features)
        values[1] = values[1] + extra * batch['img'].shape[0]
        logged['cls_loss'] = logged['cls_loss'] + extra.detach()
        return values, logged


def prepare(all_replay=False, broad_replay=False, hard_negative_repeats=1):
    if DATA.exists():
        raise FileExistsError(f'Dataset already exists: {DATA}; preserve the previous training input.')
    counts = {}
    for split in ('train', 'val'):
        imgdir, labdir = DATA / 'images' / split, DATA / 'labels' / split
        imgdir.mkdir(parents=True)
        labdir.mkdir(parents=True)
        counts[split] = 0
        for image in sorted((SOURCE / 'images' / split).glob('*.jpg')):
            shutil.copy2(image, imgdir / image.name)
            shutil.copy2(SOURCE / 'labels' / split / (image.stem + '.txt'), labdir / (image.stem + '.txt'))
            counts[split] += 1
            if split == 'train' and image.name.startswith('pen_'):
                name = image.stem + '_repeat2'
                shutil.copy2(image, imgdir / (name + '.jpg'))
                shutil.copy2(SOURCE / 'labels' / split / (image.stem + '.txt'), labdir / (name + '.txt'))
                counts[split] += 1
    # Public Pen/Pencil case images provide a different background distribution.
    # Pen labels are supervised; Pencil case labels contribute box supervision,
    # and its class response is preserved by teacher distillation.
    replay = ROOT / ('vending_machine_datasets/images/train' if all_replay else 'vending_machine_datasets/notebook_negatives/train')
    replay_count = 0
    for image in sorted(replay.glob('*.jpg')):
        rows = []
        source_rows = (ROOT / 'vending_machine_datasets/labels/train' / (image.stem + '.txt')).read_text().splitlines()
        if any(line.split()[0] == '1' for line in source_rows):
            continue  # Historical notebook labels were not reviewed for this run.
        for line in source_rows:
            fields = line.split()
            assert fields[0] in ('0', '2')
            fields[0] = '376' if fields[0] == '0' else '377'
            rows.append(' '.join(fields))
        name = 'replay_' + image.stem
        shutil.copy2(image, DATA / 'images/train' / (name + '.jpg'))
        (DATA / 'labels/train' / (name + '.txt')).write_text('\n'.join(rows) + '\n')
        counts['train'] += 1
        replay_count += 1
    cfg = yaml.safe_load((SOURCE / 'data.yaml').read_text())
    broad_count = 0
    if broad_replay:
        folder = ROOT / 'vending_machine_datasets/oiv7_replay'
        audit = json.loads((folder / 'manifest.json').read_text())
        assert not audit['failures']
        if (folder / 'reviewed_no_notebook.json').exists():
            shutil.copy2(folder / 'reviewed_no_notebook.json', DATA / 'reviewed_no_notebook.json')
        reviewed=set(json.loads((DATA/'reviewed_no_notebook.json').read_text())['ids']) if (DATA/'reviewed_no_notebook.json').exists() else set()
        for image in sorted((folder / 'images').glob('*.jpg')):
            name = 'oiv_replay_' + image.stem
            shutil.copy2(image, DATA / 'images/train' / (name + '.jpg'))
            shutil.copy2(folder / 'labels' / (image.stem + '.txt'), DATA / 'labels/train' / (name + '.txt'))
            broad_count += 1
            if image.stem in reviewed:
                for repeat in range(2,hard_negative_repeats+1):
                    extra=name+'_hardrepeat'+str(repeat)
                    shutil.copy2(image,DATA/'images/train'/(extra+'.jpg'))
                    shutil.copy2(folder/'labels'/(image.stem+'.txt'),DATA/'labels/train'/(extra+'.txt'))
                    broad_count += 1
        counts['train'] += broad_count
    cfg['path'] = str(DATA)
    (DATA / 'data.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (DATA / 'manifest.json').write_text(json.dumps({'source': str(SOURCE), 'counts': counts,
                                                  'base': str(joint.BASE), 'pen_repeats': 2,
                                                  'replay_images': replay_count,
                                                  'broad_replay_images': broad_count,
                                                  'hard_negative_repeats':hard_negative_repeats}, indent=2))
    return counts


def main():
    global DATA
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='pen_notebook_full_oiv7_v1')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--distill-weight', type=float, default=1.0)
    parser.add_argument('--reuse-data', action='store_true')
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--all-replay', action='store_true')
    parser.add_argument('--freeze-bn', action='store_true')
    parser.add_argument('--freeze-old-rows', action='store_true')
    parser.add_argument('--lr0', type=float, default=0.0005)
    parser.add_argument('--warmup-bias-lr', type=float, default=0.0005)
    parser.add_argument('--broad-replay', action='store_true')
    parser.add_argument('--hard-negative-repeats',type=int,default=1)
    parser.add_argument('--include-pencil-pouch', action='store_true',
                        help='Supervise OIV7 class 377 on pouch_* photos only.')
    parser.add_argument('--init-checkpoint', type=Path, default=None,
                        help='Continue from an existing 602-class checkpoint.')
    args = parser.parse_args()
    DATA = args.data_dir.resolve()
    torch.set_num_threads(4)
    if not args.reuse_data:
        if args.hard_negative_repeats<1:parser.error('hard-negative-repeats must be >=1')
        print('prepared:', prepare(args.all_replay, args.broad_replay,args.hard_negative_repeats), flush=True)
    initial = args.init_checkpoint.resolve() if args.init_checkpoint else joint.expand_base(book_init=True, pen_start=None)
    model = YOLO(str(initial))

    def start(trainer):
        # Original OIV7 checkpoint uses SiLU here. Ultralytics YAML reconstruction
        # otherwise substitutes the current architecture default during loading.
        trainer.model.model[9].cv1.act = torch.nn.SiLU(inplace=True)
        if trainer.ema:
            trainer.ema.ema.model[9].cv1.act = torch.nn.SiLU(inplace=True)
        teacher = YOLO(str(joint.BASE)).model.to(trainer.device).eval()
        teacher.requires_grad_(False)
        trainer.model.criterion = PartialLabelLoss(trainer.model, teacher, args.distill_weight,
                                                   include_pencil_pouch=args.include_pencil_pouch)
        if args.freeze_bn:
            joint.freeze_batch_norm(trainer)
        if args.freeze_old_rows:
            def mask_old(gradient):
                mask = torch.zeros_like(gradient)
                mask[376] = gradient[376]
                mask[601] = gradient[601]
                if args.include_pencil_pouch:
                    mask[377] = gradient[377]
                return mask
            for branch in trainer.model.model[-1].cv3:
                branch[-1].weight.register_hook(mask_old)
                branch[-1].bias.register_hook(mask_old)
        params = {n: p.numel() for n, p in trainer.model.named_parameters() if p.requires_grad}
        (trainer.save_dir / 'method.json').write_text(json.dumps({
            'base': str(joint.BASE), 'pen_start': None, 'trainable_parameters': sum(params.values()),
            'trainable_names': list(params), 'distill_weight': args.distill_weight,
            'supervised_classes': [376, 601] + ([377] if args.include_pencil_pouch else []), 'backbone_and_box_trainable': True,
            'supervised_classes_scope': 'Custom photos: Pen/notebook; broad replay: classes actually annotated per image; notebook negatives only manually reviewed images',
            'broad_replay': args.broad_replay,
            'hard_negative_repeats':args.hard_negative_repeats,
            'distillation': 'top-10 logits per old class, DFL boxes at teacher-confident anchors, feature and background KL',
            'peak_loss_normalization': 'Only teacher probability > 0.05, weighted by its probability',
            'freeze_bn': args.freeze_bn, 'freeze_old_final_classifier_rows': args.freeze_old_rows,
            'include_pencil_pouch': args.include_pencil_pouch,
            'init_checkpoint': str(initial),
            'old_class_retention_scope': 'teacher outputs on custom photos and public training images; not all-class validation'
        }, indent=2))
        for source in (Path(__file__).resolve(), ROOT / 'scripts/train_pen_notebook_joint_oiv7.py'):
            shutil.copy2(source, trainer.save_dir / source.name)
        assert any(n.startswith('model.0.') for n in params)
        assert any(n.startswith('model.22.cv2.') for n in params)

    model.add_callback('on_train_start', start)
    if args.freeze_bn:
        model.add_callback('on_train_batch_start', joint.freeze_batch_norm)
    model.train(data=str(DATA / 'data.yaml'), imgsz=args.imgsz, epochs=args.epochs,
                batch=args.batch, device=0, workers=4, optimizer='AdamW', lr0=args.lr0,
                warmup_epochs=1, warmup_bias_lr=args.warmup_bias_lr,
                lrf=0.1, weight_decay=0 if args.freeze_old_rows else 0.0005, patience=15, mosaic=0, close_mosaic=0,
                mixup=0, scale=0.25, translate=0.1, degrees=10, hsv_h=0.01,
                hsv_s=0.3, hsv_v=0.3, amp=False, project=str(ROOT / 'runs'),
                name=args.name, exist_ok=False, seed=0, plots=False, save_period=10)


if __name__ == '__main__':
    main()
