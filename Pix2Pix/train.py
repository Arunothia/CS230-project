import torch
from dataset import PianoFluteDataset
import sys
from utils import save_checkpoint, load_checkpoint, save_some_examples
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator

def train_fn(disc, gen, loader, opt_disc, opt_gen, l1, bce, g_scaler, d_scaler):
    loop = tqdm(loader, leave=True)

    for idx, (x,y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_fake = disc(x, y_fake.detach())
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1
        
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()


def main():
    disc = Discriminator(in_channels=1).to(config.DEVICE)
    gen = Generator(in_channels=1).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_Loss = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)
    
    train_dataset = PianoFluteDataset(root_piano=config.PIANO_TRAIN_DIR, root_flute=config.FLUTE_TRAIN_DIR, transform=config.transforms, isTrain=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size = config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    val_dataset = PianoFluteDataset(root_piano=config.PIANO_TRAIN_DIR, root_flute=config.FLUTE_TRAIN_DIR, transform=config.transforms, isTrain=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size = 1,
        shuffle=False
    )

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_Loss, BCE, g_scaler, d_scaler)

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
        
        save_some_examples(gen, val_loader, epoch, folder="evaluation")

if __name__ == "__main__":
    main()