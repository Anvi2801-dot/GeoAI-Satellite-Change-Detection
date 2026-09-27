"""
Evaluate trained weights on the LEVIR-CD test split and save visual comparisons.

    python evaluate.py --data data/raw/LEVIR-CD --weights models/siamese_unet.pth --save-vis 20

Prints P / R / F1 / IoU / OA (change class, accumulated over all 2048 test patches)
and writes runs/test_metrics.csv. With --save-vis N, writes N panels to runs/vis/:
    [ image A | image B | ground truth | prediction (TP white, FP red, FN green) ]
"""
import argparse
import csv
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import LEVIRCDDataset
from src.models.network import SiameseUNet
from src.training.metrics import ChangeMetrics
from train import pick_device


def panel(a, b, gt, pred):
    a = (a.transpose(1, 2, 0) * 255).astype(np.uint8)
    b = (b.transpose(1, 2, 0) * 255).astype(np.uint8)
    g = np.repeat((gt[0] * 255).astype(np.uint8)[..., None], 3, axis=2)
    err = np.zeros_like(a)
    gt, pred = gt[0] > 0.5, pred[0]
    err[pred & gt] = (255, 255, 255)   # TP
    err[pred & ~gt] = (255, 0, 0)      # FP
    err[~pred & gt] = (0, 255, 0)      # FN
    sep = np.full((a.shape[0], 4, 3), 128, np.uint8)
    out = np.concatenate([a, sep, b, sep, g, sep, err], axis=1)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw/LEVIR-CD")
    p.add_argument("--split", default="test")
    p.add_argument("--weights", default="models/siamese_unet.pth")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save-vis", type=int, default=0, help="number of panels to save (changed patches first)")
    args = p.parse_args()

    device = pick_device()
    model = SiameseUNet(3, 1).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    ds = LEVIRCDDataset(args.data, args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    metrics = ChangeMetrics(args.threshold)

    vis_dir = os.path.join("runs", "vis")
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)
    saved = 0

    for i, (a, b, m) in enumerate(loader):
        logits = model(a.to(device), b.to(device)).cpu()
        metrics.update(logits, m)
        if saved < args.save_vis:
            pred = (torch.sigmoid(logits) > args.threshold).numpy()
            for j in range(a.size(0)):
                if saved >= args.save_vis:
                    break
                if m[j].sum() == 0:      # only show patches that actually contain change
                    continue
                idx = i * args.batch_size + j
                cv2.imwrite(os.path.join(vis_dir, f"{idx:05d}.png"),
                            panel(a[j].numpy(), b[j].numpy(), m[j].numpy(), pred[j]))
                saved += 1

    r = metrics.compute()
    print(f"[{args.split}]  P {r['precision']:.4f}  R {r['recall']:.4f}  F1 {r['f1']:.4f}  "
          f"IoU {r['iou']:.4f}  OA {r['oa']:.4f}")
    os.makedirs("runs", exist_ok=True)
    with open(os.path.join("runs", f"{args.split}_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", *r.keys()])
        w.writerow([args.split, *(f"{v:.5f}" for v in r.values())])
    if saved:
        print(f"[INFO] saved {saved} panels to {vis_dir}/")


if __name__ == "__main__":
    main()
