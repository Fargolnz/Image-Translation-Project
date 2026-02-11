import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from data.dataset import SplitImageDataset
from models.generator import Generator
from models.discriminator import Discriminator
from utils import generator_loss, discriminator_loss

# ----------------- Settings -----------------
EPOCHS = 22
BATCH_SIZE = 2
LR = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# ----------------- Folders -----------------
os.makedirs("outputs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# ----------------- Dataset -----------------
train_dataset = SplitImageDataset("data/facades/train")
val_dataset = SplitImageDataset("data/facades/val")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
fixed_input, _ = next(iter(val_loader))
fixed_input = fixed_input.to(DEVICE)

# ----------------- Models -----------------
gen = Generator().to(DEVICE)
disc = Discriminator().to(DEVICE)

opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=5e-5, betas=(0.5, 0.999))

# ----------------- Auto Resume -----------------
start_epoch = 0
checkpoint_files = glob.glob("checkpoints/*.pth")

if len(checkpoint_files) > 0:
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    print("Loading checkpoint:", latest_checkpoint)

    checkpoint = torch.load(
        latest_checkpoint,
        map_location=DEVICE,
        weights_only=False)

    gen.load_state_dict(checkpoint["generator"])
    disc.load_state_dict(checkpoint["discriminator"])
    opt_gen.load_state_dict(checkpoint["opt_g"])
    opt_disc.load_state_dict(checkpoint["opt_d"])

    start_epoch = checkpoint["epoch"] + 1
    print("Resuming from epoch:", start_epoch)
else:
    print("No checkpoint found. Training from scratch.")

# ----------------- Training -----------------
for epoch in range(start_epoch, EPOCHS):

    gen.train()
    disc.train()

    epoch_g_loss = 0
    epoch_d_loss = 0

    for idx, (corrupted, real) in enumerate(train_loader):

        corrupted = corrupted.to(DEVICE)
        real = real.to(DEVICE)

        # ----------------------
        # Train Discriminator
        # ----------------------
        fake = gen(corrupted)

        d_loss = discriminator_loss(
            disc, real, fake.detach(), corrupted
        )

        if idx % 2 == 0:
            opt_disc.zero_grad()
            d_loss.backward()
            opt_disc.step()

        # ----------------------
        # Train Generator
        # ----------------------
        fake = gen(corrupted)

        g_loss = generator_loss(
            disc, fake, real, corrupted
        )

        opt_gen.zero_grad()
        g_loss.backward()
        opt_gen.step()

        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()

        if idx % 50 == 0:
            print(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Batch [{idx}/{len(train_loader)}] "
                f"D Loss: {d_loss.item():.4f} "
                f"G Loss: {g_loss.item():.4f}"
            )
    # -------------------------------------------------
    # Average Loss
    # -------------------------------------------------
    avg_g = epoch_g_loss / len(train_loader)
    avg_d = epoch_d_loss / len(train_loader)

    print(
        f"\nEpoch [{epoch+1}/{EPOCHS}] Completed | "
        f"Avg D Loss: {avg_d:.4f} | "
        f"Avg G Loss: {avg_g:.4f}\n"
    )

    # -------------------------------------------------
    # Save Sample Output Every 5 Epochs
    # -------------------------------------------------
    if (epoch + 1) % 5 == 0:
        gen.eval()
        with torch.no_grad():
            sample_fake = gen(fixed_input)

        save_image(
            (sample_fake + 1) / 2,
            f"outputs/epoch_{epoch+1}.png"
        )
        print(f"Saved sample output for epoch {epoch+1}")

    # -------------------------------------------------
    # Save Checkpoint (Each Epoch)
    # -------------------------------------------------
    torch.save({
        "epoch": epoch,
        "generator": gen.state_dict(),
        "discriminator": disc.state_dict(),
        "opt_g": opt_gen.state_dict(),
        "opt_d": opt_disc.state_dict()
    }, f"checkpoints/epoch_{epoch+1}.pth")

print("Training Finished Successfully ✅")