import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import SplitImageDataset
from models.generator import Generator
from models.discriminator import Discriminator
from utils import generator_loss, discriminator_loss

EPOCHS = 100
BATCH_SIZE = 8

dataset = SplitImageDataset("data/facades/train")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
gen = Generator().to(device)
disc = Discriminator().to(device)

opt_gen = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

for epoch in range(EPOCHS):
    for idx, (corrupted, real) in enumerate(loader):
        corrupted = corrupted.to(device)
        real = real.to(device)

        # ============================
        # Train Discriminator
        # ============================
        fake = gen(corrupted)

        disc_loss = discriminator_loss(
            disc, real, fake, corrupted
        )

        opt_disc.zero_grad()
        disc_loss.backward()
        opt_disc.step()

        # ============================
        # Train Generator
        # ============================
        fake = gen(corrupted)

        gen_loss = generator_loss(
            disc, fake, real, corrupted
        )

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        if idx % 50 == 0:
            print(
                f"Epoch [{epoch}/{EPOCHS}] "
                f"Batch [{idx}/{len(loader)}] "
                f"D Loss: {disc_loss.item():.4f} "
                f"G Loss: {gen_loss.item():.4f}"
            )