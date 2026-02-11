import torch
import torch.nn as nn

bce = nn.BCEWithLogitsLoss()
l1 = nn.L1Loss()

LAMBDA_L1 = 30

def discriminator_loss(disc, real_img, fake_img, condition):
    real_pred = disc(condition, real_img)
    fake_pred = disc(condition, fake_img.detach())

    real_loss = bce(real_pred, torch.ones_like(real_pred) * 0.9)
    fake_loss = bce(fake_pred, torch.zeros_like(fake_pred))

    return (real_loss + fake_loss) / 2


def generator_loss(disc, fake_img, real_img, condition):
    fake_pred = disc(condition, fake_img)

    adv_loss = bce(fake_pred, torch.ones_like(fake_pred))
    l1_loss = l1(fake_img, real_img)

    return adv_loss * 2 + LAMBDA_L1 * l1_loss