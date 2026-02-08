import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.utils import make_grid
from models.generator import Generator
from data.dataset import InpaintingDataset

OUTPUT_DIR = "results/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def denormalize(tensor):
    return (tensor + 1) / 2

device = "cuda" if torch.cuda.is_available() else "cpu"

gen = Generator().to(device)
gen.load_state_dict(torch.load("generator.pth", map_location=device))
gen.eval()

test_dataset = InpaintingDataset("data/facades/test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


with torch.no_grad():
    for idx, (corrupted, real) in enumerate(test_loader):
        corrupted = corrupted.to(device)
        real = real.to(device)

        fake = gen(corrupted)

        # denormalize
        corrupted_img = denormalize(corrupted)
        fake_img = denormalize(fake)
        real_img = denormalize(real)

        save_image(
            corrupted_img,
            f"{OUTPUT_DIR}/{idx}_input.png"
        )
        save_image(
            fake_img,
            f"{OUTPUT_DIR}/{idx}_output.png"
        )
        save_image(
            real_img,
            f"{OUTPUT_DIR}/{idx}_target.png"
        )

        if idx == 9:  # فقط 10 تصویر برای گزارش
            break


"""
with torch.no_grad():
    for idx, (corrupted, real) in enumerate(test_loader):
        corrupted = corrupted.to(device)
        real = real.to(device)

        fake = gen(corrupted)

        grid = make_grid(
            torch.cat([
                denormalize(corrupted),
                denormalize(fake),
                denormalize(real)
            ], dim=0),
            nrow=3
        )

        save_image(grid, f"{OUTPUT_DIR}/{idx}_comparison.png")

        if idx == 4:  # 5 تصویر مقایسه‌ای کافیه
            break
"""