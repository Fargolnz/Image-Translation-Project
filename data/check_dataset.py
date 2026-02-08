import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import SplitImageDataset

dataset = SplitImageDataset("data/facades/train")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

corrupted, real = next(iter(loader))

# denormalize
corrupted = (corrupted + 1) / 2
real = (real + 1) / 2

corrupted = corrupted.squeeze().permute(1, 2, 0).numpy()
real = real.squeeze().permute(1, 2, 0).numpy()

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(corrupted)
plt.title("Input (Corrupted)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(real)
plt.title("Ground Truth")
plt.axis("off")

plt.tight_layout()
plt.show()
