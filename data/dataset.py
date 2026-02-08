import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SplitImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = sorted(os.listdir(root_dir))

        self.transform = transforms.Compose([
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ])

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")

        img = self.transform(img)

        _, h, w = img.shape
        w_half = w // 2

        real = img[:, :, :w_half]
        corrupted = img[:, :, w_half:]

        return corrupted, real

    def __len__(self):
        return len(self.images)