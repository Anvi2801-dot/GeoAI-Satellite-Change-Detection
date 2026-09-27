import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    Weighted BCE + soft Dice for single-channel logits.
    Only ~5% of LEVIR-CD pixels are 'change', so plain BCE drifts toward
    predicting no-change; Dice optimises overlap directly (as in SNUNet-CD).
    """
    def __init__(self, pos_weight=None, bce_weight=1.0, dice_weight=1.0, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))
        else:
            self.pos_weight = None

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        # Dice over the whole batch, not per image: many LEVIR patches contain no
        # change, and per-image Dice on an empty patch stays ~1 unless every pixel is ~0.
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum()
        union = prob.sum() + target.sum()
        dice = 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)
        return self.bce_weight * bce + self.dice_weight * dice
