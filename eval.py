import torch
import numpy as np
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from models.generator import Generator
from data.dataset import InpaintingDataset


def tensor_to_image(tensor):
    image = tensor.detach().cpu().numpy()
    image = (image + 1) / 2  # [-1,1] → [0,1]
    image = np.clip(image, 0, 1)
    image = np.transpose(image, (1, 2, 0))  # C,H,W → H,W,C
    return image


device = "cuda" if torch.cuda.is_available() else "cpu"

gen = Generator().to(device)
gen.load_state_dict(torch.load("generator.pth", map_location=device))
gen.eval()


test_dataset = InpaintingDataset("data/facades/test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


psnr_scores = []
ssim_scores = []

with torch.no_grad():
    for corrupted, real in test_loader:
        corrupted = corrupted.to(device)
        real = real.to(device)

        fake = gen(corrupted)

        fake_img = tensor_to_image(fake[0])
        real_img = tensor_to_image(real[0])

        psnr_value = psnr(real_img, fake_img, data_range=1.0)
        ssim_value = ssim(
            real_img,
            fake_img,
            data_range=1.0,
            channel_axis=2
        )

        psnr_scores.append(psnr_value)
        ssim_scores.append(ssim_value)


print("Average PSNR:", np.mean(psnr_scores))
print("Average SSIM:", np.mean(ssim_scores))