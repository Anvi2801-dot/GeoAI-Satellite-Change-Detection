import torch


class ChangeMetrics:
    """
    Accumulates TP/FP/FN/TN over the whole split (not averaged per image),
    which is how LEVIR-CD results are reported in BIT, ChangeFormer, SNUNet.
    Precision / Recall / F1 / IoU are for the 'change' class.
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0

    @torch.no_grad()
    def update(self, logits, target):
        pred = (torch.sigmoid(logits) > self.threshold)
        gt = target > 0.5
        self.tp += int((pred & gt).sum())
        self.fp += int((pred & ~gt).sum())
        self.fn += int((~pred & gt).sum())
        self.tn += int((~pred & ~gt).sum())

    def compute(self):
        eps = 1e-10
        p = self.tp / (self.tp + self.fp + eps)
        r = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * p * r / (p + r + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        oa = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + eps)
        return {"precision": p, "recall": r, "f1": f1, "iou": iou, "oa": oa}
