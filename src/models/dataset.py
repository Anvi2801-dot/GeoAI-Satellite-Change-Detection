import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class LEVIRCDDataset(Dataset):
    """
    Dataset loader for LEVIR-CD Bitemporal Building Change Detection.
    """
    def __init__(self, root_dir: str, split: str = "train", patch_size: int = 256, crops_per_image: int = 16, augment: bool = True):
        self.root_dir = root_dir
        self.split = split
        self.patch_size = patch_size
        self.crops_per_image = crops_per_image
        self.augment = augment and (split == "train")

        self.dir_a = os.path.join(root_dir, split, "A")
        self.dir_b = os.path.join(root_dir, split, "B")
        self.dir_label = os.path.join(root_dir, split, "label")

        if not os.path.exists(self.dir_a):
            raise FileNotFoundError(f"Directory not found: {self.dir_a}")

        self.filenames = sorted([f for f in os.listdir(self.dir_a) if f.lower().endswith(('.png', '.tif', '.jpg'))])
        
        # Build patches index
        self.samples = []
        for fname in self.filenames:
            if split == "train" and self.crops_per_image > 1:
                for _ in range(self.crops_per_image):
                    self.samples.append(fname)
            else:
                self.samples.append(fname)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname = self.samples[idx]
        path_a = os.path.join(self.dir_a, fname)
        path_b = os.path.join(self.dir_b, fname)
        path_l = os.path.join(self.dir_label, fname)

        img_a = cv2.cvtColor(cv2.imread(path_a), cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(cv2.imread(path_b), cv2.COLOR_BGR2RGB)
        
        if os.path.exists(path_l):
            label = cv2.imread(path_l, cv2.IMREAD_GRAYSCALE)
            label = (label > 128).astype(np.float32)
        else:
            label = np.zeros((img_a.shape[0], img_a.shape[1]), dtype=np.float32)

        h, w, _ = img_a.shape

        # Random Cropping to patch_size during training
        if h > self.patch_size or w > self.patch_size:
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            
            img_a = img_a[top:top+self.patch_size, left:left+self.patch_size]
            img_b = img_b[top:top+self.patch_size, left:left+self.patch_size]
            label = label[top:top+self.patch_size, left:left+self.patch_size]

        # Data Augmentations (Horizontal & Vertical Flips)
        if self.augment:
            if random.random() > 0.5:
                img_a = np.fliplr(img_a).copy()
                img_b = np.fliplr(img_b).copy()
                label = np.fliplr(label).copy()
            if random.random() > 0.5:
                img_a = np.flipud(img_a).copy()
                img_b = np.flipud(img_b).copy()
                label = np.flipud(label).copy()

        # Convert to float tensors [C, H, W] in range [0.0, 1.0]
        tensor_a = torch.from_numpy(img_a).permute(2, 0, 1).float() / 255.0
        tensor_b = torch.from_numpy(img_b).permute(2, 0, 1).float() / 255.0
        tensor_l = torch.from_numpy(label).unsqueeze(0).float()

        return tensor_a, tensor_b, tensor_l