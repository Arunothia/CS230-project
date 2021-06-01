import random, torch, os, numpy as np
import config
from torchvision.utils import save_image
import torch_utils


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("===> Saving Checkpoint")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("===> Loading Checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Reset learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def val(gen_F, gen_P, disc_F, disc_P, mse, L1, val_loader, epoch, folder):
    losses = torch_utils.AverageMeter('Loss', length=10)
    progress = torch_utils.ProgressMeter(
        len(val_loader),
        [losses],
        prefix='Val: ')
    flute, piano = next(iter(val_loader))
    flute, piano = flute.to(config.DEVICE), piano.to(config.DEVICE)
    gen_F.eval(), gen_P.eval(), disc_F.eval(), disc_P.eval()
    with torch.no_grad():
        piano_fake = gen_F(piano)
        flute_fake = gen_P(flute)
        save_image(flute, folder + f"/flute_{epoch}.png")
        save_image(piano, folder + f"/piano_{epoch}.png")
        save_image(flute_fake, folder + f"/fake_flute{epoch}.png")
        save_image(piano_fake, folder + f"/fake_piano{epoch}.png")

        # Adversarial Loss for both generators
        D_P_fake = disc_P(piano_fake)
        D_F_fake = disc_F(flute_fake)
        loss_G_F = mse(D_F_fake, torch.ones_like(D_F_fake))
        loss_G_P = mse(D_P_fake, torch.ones_like(D_P_fake))

        # Cycle Loss
        cycle_piano = gen_P(flute_fake)
        cycle_flute = gen_F(piano_fake)
        cycle_piano_loss = L1(piano, cycle_piano)
        cycle_flute_loss = L1(flute, cycle_flute)

        # Identity Loss
        identity_flute = gen_F(flute)
        identity_piano = gen_P(piano)
        identity_piano_loss = L1(piano, identity_piano)
        identity_flute_loss = L1(flute, identity_flute)

        # Overall Generator Loss
        G_loss =  (
            loss_G_F +
            loss_G_P +
            cycle_flute_loss * config.LAMBDA_CYCLE +
            cycle_piano_loss * config.LAMBDA_CYCLE +
            identity_flute_loss * config.LAMBDA_IDENTITY +
            identity_piano_loss * config.LAMBDA_IDENTITY
        )

        losses.update(G_loss, 1)
        progress.display(epoch)

    gen_F.train(), gen_P.train(), disc_F.train(), disc_P.train()