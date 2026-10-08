import os
import torch
import numpy as np
from torch.utils.data import DataLoader

# Imports matching your repository structure
from src.models.network import SiameseUNet  # or your exact model class name in network.py
from src.models.dataset import LEVIRCDDataset # or your exact dataset class name in dataset.py


def evaluate_metrics(model_path, data_dir, split="val", batch_size=8, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model checkpoint from: {model_path}")
    print(f"Using device: {device}")

    # 1. Instantiate model and load trained weights
    model = SiameseUNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Initialize dataset and dataloader
    # Ensure data_dir points to your LEVIR-CD root (e.g., 'data/LEVIR-CD')
    dataset = LEVIRCDDataset(root_dir=data_dir, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 3. Accumulate Confusion Matrix across all batches
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0

    print(f"\nEvaluating on '{split}' split ({len(dataset)} samples)...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Adjust key names if your dataset returns a tuple (t1, t2, mask) or dict
            if isinstance(batch, (list, tuple)):
                t1, t2, masks = batch
            else:
                t1, t2, masks = batch['t1'], batch['t2'], batch['mask']

            t1 = t1.to(device)
            t2 = t2.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(t1, t2)

            # Apply sigmoid if model outputs logits
            probs = torch.sigmoid(outputs) if outputs.max() > 1.0 or outputs.min() < 0.0 else outputs
            preds = (probs > 0.5).float()

            # Flatten tensors for fast pixel-wise confusion matrix calculation
            preds_flat = preds.view(-1).cpu().numpy().astype(np.uint8)
            masks_flat = masks.view(-1).cpu().numpy().astype(np.uint8)

            total_tp += np.sum((preds_flat == 1) & (masks_flat == 1))
            total_fp += np.sum((preds_flat == 1) & (masks_flat == 0))
            total_tn += np.sum((preds_flat == 0) & (masks_flat == 0))
            total_fn += np.sum((preds_flat == 0) & (masks_flat == 1))

    # 4. Compute Table metrics
    eps = 1e-7
    precision = (total_tp / (total_tp + total_fp + eps)) * 100
    recall = (total_tp / (total_tp + total_fn + eps)) * 100
    f1 = (2 * precision * recall) / (precision + recall + eps)
    iou = (total_tp / (total_tp + total_fp + total_fn + eps)) * 100
    oa = ((total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + eps)) * 100

    print("\n=======================================================")
    print(f"   TABLE VII METRICS FOR RESEARCH PAPER ({split.upper()})")
    print("=======================================================")
    print(f" Precision : {precision:.2f}%")
    print(f" Recall    : {recall:.2f}%")
    print(f" F1-Score  : {f1 / 100:.4f}  ({f1:.2f}%)")
    print(f" IoU       : {iou:.2f}%")
    print(f" OA        : {oa:.2f}%")
    print("=======================================================\n")

    return precision, recall, f1, iou, oa


if __name__ == "__main__":
    MODEL_PATH = "models/siamese_unet.pth"
    DATA_DIR = "data/raw/LEVIR-CD"  # Updated to match your exact folder structure

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {MODEL_PATH}. Check file location.")

    # Run for Validation split (best epoch 25)
    evaluate_metrics(MODEL_PATH, DATA_DIR, split="val")

    # Run for Test split
    # evaluate_metrics(MODEL_PATH, DATA_DIR, split="test")