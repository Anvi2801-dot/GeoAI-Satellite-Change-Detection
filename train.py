"""
Train the Siamese UNet (src/models/network.py) on LEVIR-CD.

    python train.py --data data/raw/LEVIR-CD --epochs 100 --batch-size 16

Outputs
    models/siamese_unet.pth   best-val-F1 weights (plain state_dict, loaded by main.py)
    models/last.pt            full checkpoint for --resume
    runs/train_log.csv        per-epoch loss and val metrics
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import LEVIRCDDataset
from src.models.network import SiameseUNet
from src.training.losses import BCEDiceLoss
from src.training.metrics import ChangeMetrics


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw/LEVIR-CD")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--crops-per-image", type=int, default=16,
                   help="random crops per 1024px train image per epoch (16 = 7120 patches, same as the standard split)")
    p.add_argument("--pos-weight", type=float, default=None,
                   help="BCE weight on change pixels, e.g. 5. Default: none (Dice handles imbalance)")
    p.add_argument("--prior", type=float, default=0.05, help="initial change probability for output bias")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="models/siamese_unet.pth")
    p.add_argument("--resume", default=None, help="path to models/last.pt")
    p.add_argument("--overfit", type=int, default=0,
                   help="debug: train and validate on the first N train patches only")
    return p.parse_args()


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    metrics = ChangeMetrics()
    total_loss, n = 0.0, 0
    for a, b, m in loader:
        a, b, m = a.to(device), b.to(device), m.to(device)
        logits = model(a, b)
        total_loss += criterion(logits, m).item() * a.size(0)
        n += a.size(0)
        metrics.update(logits, m)
    return total_loss / max(n, 1), metrics.compute()


def main():
    args = get_args()
    seed_everything(args.seed)
    device = pick_device()
    use_amp = args.amp and device.type == "cuda"
    print(f"[INFO] device={device}  amp={use_amp}")

    # ---------------- data ----------------
    train_ds = LEVIRCDDataset(args.data, "train", args.patch_size, args.crops_per_image)
    if args.overfit:
        # Fixed patches, no augmentation: loss should go to ~0 and F1 to ~1
        fixed = LEVIRCDDataset(args.data, "train", args.patch_size, augment=False)
        fixed.train = False  # use deterministic grid tiles
        train_ds = val_ds = Subset(fixed, range(min(args.overfit, len(fixed))))
    else:
        val_ds = LEVIRCDDataset(args.data, "val", args.patch_size)
    print(f"[INFO] train patches/epoch={len(train_ds)}  val patches={len(val_ds)}")

    pin = device.type == "cuda"
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin, drop_last=not args.overfit,
                              worker_init_fn=seed_worker, generator=g,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin)

    # ---------------- model / optim ----------------
    model = SiameseUNet(in_channels=3, out_channels=1).to(device)
    # Prior-probability bias init (RetinaNet trick): start by predicting ~5% change,
    # the LEVIR-CD change fraction, instead of 50%. Speeds up and stabilises early training.
    torch.nn.init.constant_(model.outc.bias, float(np.log(args.prior / (1 - args.prior))))
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[INFO] SiameseUNet params: {n_params:.2f} M")

    criterion = BCEDiceLoss(pos_weight=args.pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.05)
    # torch.amp.GradScaler exists from torch 2.3; requirements.txt pins 2.0.1
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch, best_f1 = 0, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        if ck["scheduler"]["total_steps"] != scheduler.total_steps:
            raise SystemExit("[ERROR] --resume needs the same --epochs / --batch-size as the original run "
                             "(the one-cycle LR schedule is fixed at start). To train longer, start a new run.")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_f1 = ck["epoch"] + 1, ck["best_f1"]
        print(f"[INFO] resumed from epoch {start_epoch}, best F1 {best_f1:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    last_path = os.path.join(os.path.dirname(args.out) or ".", "last.pt")
    os.makedirs("runs", exist_ok=True)
    log_path = os.path.join("runs", "train_log.csv")
    if start_epoch == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "lr", "train_loss", "val_loss",
                                    "precision", "recall", "f1", "iou", "oa", "sec"])

    # ---------------- loop ----------------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for a, b, m in train_loader:
            a, b, m = a.to(device, non_blocking=True), b.to(device, non_blocking=True), m.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(a, b)
                loss = criterion(logits.float(), m)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run_loss += loss.item() * a.size(0)
            n += a.size(0)

        train_loss = run_loss / max(n, 1)
        val_loss, mt = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        tag = ""
        if mt["f1"] > best_f1:
            best_f1 = mt["f1"]
            torch.save(model.state_dict(), args.out)  # format expected by src/spark/inference.py
            tag = "  *best*"
        print(f"Ep {epoch+1:3d}/{args.epochs}  lr {lr:.2e}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"P {mt['precision']:.4f} R {mt['recall']:.4f} F1 {mt['f1']:.4f} IoU {mt['iou']:.4f}  {dt:.0f}s{tag}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{lr:.3e}", f"{train_loss:.5f}", f"{val_loss:.5f}",
                                    *(f"{mt[k]:.5f}" for k in ("precision", "recall", "f1", "iou", "oa")),
                                    f"{dt:.1f}"])
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_f1": best_f1}, last_path)

    print(f"[DONE] best val F1 = {best_f1:.4f}  weights -> {args.out}")


if __name__ == "__main__":
    main()
