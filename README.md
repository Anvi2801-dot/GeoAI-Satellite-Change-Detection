# GeoAI Satellite Change Detection

Siamese UNet for bitemporal building-change detection on LEVIR-CD, with distributed
inference and area metrics via PySpark + Apache Sedona.

## 1. Dataset

Download LEVIR-CD (637 pairs, 1024×1024, 0.5 m/px) from the official page
(https://justchenhao.github.io/LEVIR/) and arrange it as:

```
data/raw/LEVIR-CD/
├── train/  A/  B/  label/     445 pairs
├── val/    A/  B/  label/      64 pairs
└── test/   A/  B/  label/     128 pairs
```

No pre-cropping needed. Training takes random 256×256 crops (16 per image per epoch
= 7120 patches); val/test are split into a fixed 4×4 grid (1024 / 2048 patches),
matching the split used by BIT, ChangeFormer and SNUNet.

## 2. Train

```bash
pip install torch torchvision opencv-python numpy
python train.py --data data/raw/LEVIR-CD --epochs 100 --batch-size 16 --amp
```

| Setting | Default |
|---|---|
| Model | `src/models/network.py` SiameseUNet, 7.70 M params |
| Loss | BCE + batch-level Dice (class imbalance, ~5% change pixels) |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4, one-cycle schedule |
| Augmentation | joint flips / 90° rotations, A↔B swap, per-date brightness/contrast |
| Output bias init | log(0.05/0.95), the change-pixel prior |
| Model selection | best val F1 |

Outputs:
- `models/siamese_unet.pth`: best weights (plain `state_dict`, used directly by `main.py`)
- `models/last.pt`: full checkpoint; continue an interrupted run with `--resume models/last.pt` (same `--epochs` / `--batch-size`)
- `runs/train_log.csv`: loss and P / R / F1 / IoU / OA per epoch

Useful flags: `--batch-size 8` if GPU memory is short, `--pos-weight 5` to push recall,
`--overfit 8` to sanity-check the pipeline (F1 should reach ~1.0 on 8 patches).
`--amp` only applies on CUDA. On Apple Silicon the script uses MPS automatically.

## 3. Evaluate

```bash
python evaluate.py --data data/raw/LEVIR-CD --weights models/siamese_unet.pth --save-vis 20
```

Prints change-class P / R / F1 / IoU / OA accumulated over all test pixels, writes
`runs/test_metrics.csv`, and saves panels to `runs/vis/`:
`A | B | ground truth | prediction` (white = TP, red = FP, green = FN).

Reference test F1 on LEVIR-CD: FC-Siam-diff 86.3, BIT 89.3, ChangeFormer 90.4.

## 4. Distributed inference + area metrics

```bash
python main.py
```

Loads `models/siamese_unet.pth`, runs inference over `data/raw/LEVIR-CD/train` with
Spark `mapPartitions`, converts changed pixels to m² / hectares (0.5 m/px), and
exports `data/processed/metrics_summary.csv` for the Power BI dashboard.
