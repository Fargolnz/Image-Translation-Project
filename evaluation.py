import os
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from models.generator import Generator
from data.dataset import SplitImageDataset

# -------------------------------------------------
# Settings
# -------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# Load Latest Checkpoint
# -------------------------------------------------
checkpoint_files = glob.glob("checkpoints/*.pth")

if len(checkpoint_files) == 0:
    raise RuntimeError("No checkpoint found. Train the model first.")

latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
print("Loading checkpoint:", latest_checkpoint)

checkpoint = torch.load(
    latest_checkpoint,
    map_location=DEVICE,
    weights_only=False)

current_epoch = checkpoint["epoch"] + 1

gen = Generator().to(DEVICE)
gen.load_state_dict(checkpoint["generator"])
gen.eval()

print("Model loaded successfully ✅")

# -------------------------------------------------
# Dataset
# -------------------------------------------------
test_dataset = SplitImageDataset("data/edges2shoes/val")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def denormalize(tensor):
    return (tensor + 1) / 2

def tensor_to_image(tensor):
    image = denormalize(tensor)
    image = image.detach().cpu().numpy()
    image = np.clip(image, 0, 1)
    image = np.transpose(image, (1, 2, 0))
    return image

# -------------------------------------------------
# Evaluation Function
# -------------------------------------------------
def evaluate_model(model, dataloader, device):

    model.eval()

    psnr_scores = []
    ssim_scores = []

    with torch.no_grad():
        for corrupted, real in dataloader:

            corrupted = corrupted.to(device)
            real = real.to(device)

            fake = model(corrupted)

            fake_img = tensor_to_image(fake[0])
            real_img = tensor_to_image(real[0])

            psnr_value = psnr(real_img, fake_img, data_range=1.0)
            ssim_value = ssim(real_img, fake_img, data_range=1.0, channel_axis=2)

            psnr_scores.append(psnr_value)
            ssim_scores.append(ssim_value)

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)

    print("\n========== Evaluation Results ==========")
    print("Average PSNR:", avg_psnr)
    print("Average SSIM:", avg_ssim)
    print("========================================")

    model.train()

    return avg_psnr, avg_ssim


# -------------------------------------------------
# Save Sample Comparisons Function
# -------------------------------------------------
def save_sample_images(model, dataloader, device, output_dir, epoch, max_images=10):

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for idx, (corrupted, real) in enumerate(dataloader):

            corrupted = corrupted.to(device)
            real = real.to(device)

            fake = model(corrupted)

            corrupted_img = denormalize(corrupted)
            fake_img = denormalize(fake)
            real_img = denormalize(real)

            comparison = make_grid(
                [corrupted_img[0], fake_img[0], real_img[0]],
                nrow=3
            )

            save_image(
                comparison,
                f"{output_dir}/comparison_e{epoch}_{idx}.png"
            )

            if idx + 1 == max_images:
                break

    model.train()

# -------------------------------------------------
# Evaluation + Save Results
# -------------------------------------------------
psnr_scores = []
ssim_scores = []

with torch.no_grad():
    for idx, (corrupted, real) in enumerate(test_loader):

        corrupted = corrupted.to(DEVICE)
        real = real.to(DEVICE)

        fake = gen(corrupted)

        # ----- Metrics -----
        fake_img = tensor_to_image(fake[0])
        real_img = tensor_to_image(real[0])

        psnr_value = psnr(real_img, fake_img, data_range=1.0)
        ssim_value = ssim(real_img, fake_img, data_range=1.0, channel_axis=2)

        psnr_scores.append(psnr_value)
        ssim_scores.append(ssim_value)

        # ----- Save Comparison Image -----
        corrupted_img = denormalize(corrupted)
        fake_img_t = denormalize(fake)
        real_img_t = denormalize(real)

        comparison = make_grid(
            [corrupted_img[0], fake_img_t[0], real_img_t[0]],
            nrow=3
        )

        save_image(
            comparison,
            f"{OUTPUT_DIR}/comparison_e{current_epoch}_{idx}.png"
        )

        # 10 images
        if idx == 9:
            break

# -------------------------------------------------
# Print Final Metrics
# -------------------------------------------------
print("\n========== Evaluation Results ==========")
print("Average PSNR:", np.mean(psnr_scores))
print("Average SSIM:", np.mean(ssim_scores))
print("Images saved in:", OUTPUT_DIR)
print("========================================")