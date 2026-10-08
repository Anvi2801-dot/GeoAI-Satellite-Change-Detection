import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import csv
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.models.dataset import LEVIRCDDataset
from src.models.network import SiameseUNet
from src.training.losses import BCEDiceLoss
from src.training.metrics import ChangeMetrics


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw/LEVIR-CD")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--crops-per-image", type=int, default=16)
    p.add_argument("--pos-weight", type=float, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="models/siamese_unet.pth")
    p.add_argument("--resume", default=None)
    p.add_argument("--overfit", type=int, default=0)
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
    p_mins, p_maxs, p_means = [], [], []

    for a, b, m in loader:
        a, b, m = a.to(device), b.to(device), m.to(device)
        logits = model(a, b)
        probs = torch.sigmoid(logits)
        
        p_mins.append(probs.min().item())
        p_maxs.append(probs.max().item())
        p_means.append(probs.mean().item())

        total_loss += criterion(logits, m).item() * a.size(0)
        n += a.size(0)
        metrics.update(logits, m)

    prob_stats = {
        "min": np.mean(p_mins),
        "max": np.mean(p_maxs),
        "mean": np.mean(p_means)
    }
    return total_loss / max(n, 1), metrics.compute(), prob_stats


def main():
    args = get_args()
    seed_everything(args.seed)
    device = pick_device()
    use_amp = args.amp and device.type == "cuda"
    print(f"[INFO] Device={device} | AMP={use_amp}")

    # Data Loading
    train_ds = LEVIRCDDataset(args.data, "train", args.patch_size, args.crops_per_image)
    if args.overfit:
        fixed = LEVIRCDDataset(args.data, "train", args.patch_size, augment=False)
        fixed.train = False
        train_ds = val_ds = Subset(fixed, range(min(args.overfit, len(fixed))))
    else:
        val_ds = LEVIRCDDataset(args.data, "val", args.patch_size)

    print(f"[INFO] Train Patches={len(train_ds)} | Val Patches={len(val_ds)}")

    pin = device.type == "cuda"
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin, drop_last=not args.overfit,
                              worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin)

    # Model & Optimization
    model = SiameseUNet(in_channels=3, out_channels=1).to(device)
    
    # Standard zero-bias initialization for stable gradient propagation
    torch.nn.init.zeros_(model.outc.bias)
    torch.nn.init.xavier_uniform_(model.outc.weight)

    criterion = BCEDiceLoss(pos_weight=args.pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch, best_f1 = 0, -1.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

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
        val_loss, mt, p_stats = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        tag = ""
        # Only save state dict if model demonstrates clear prediction dynamics (Max Prob > 0.70)
        if mt["f1"] > best_f1 and p_stats["max"] > 0.50:
            best_f1 = mt["f1"]
            torch.save(model.state_dict(), args.out)
            tag = "  *best*"

        print(f"Ep {epoch+1:2d}/{args.epochs} | Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"F1: {mt['f1']:.4f} | Prob Min/Max/Mean: [{p_stats['min']:.3f} / {p_stats['max']:.3f} / {p_stats['mean']:.3f}]{tag}")


if __name__ == "__main__":
    main()