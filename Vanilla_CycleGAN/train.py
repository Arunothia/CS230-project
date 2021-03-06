import torch
from dataset import PianoFluteDataset
import sys
from utils import save_checkpoint, load_checkpoint, val, AverageMeter, ProgressMeter, draw_result
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator

def train_fn(epoch, disc_P, disc_F, gen_F, gen_P, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler):
  loop = tqdm(loader, leave=True)
  disc_loss = AverageMeter('Discriminator_Loss', ':.4e')
  gen_loss = AverageMeter('Generator_Loss', ':.4e')
  progress = ProgressMeter(
    len(loop),
    [disc_loss, gen_loss],
    prefix='Train: ')
  
  Generator_Loss = []
  Discriminator_Loss = []

  for idx, (flute, piano) in enumerate(loop):
    piano = piano.to(config.DEVICE)
    flute = flute.to(config.DEVICE)

    # Train Discriminators P and F
    with torch.cuda.amp.autocast():
      fake_piano = gen_P(flute)
      D_P_real = disc_P(piano)
      D_P_fake = disc_P(fake_piano.detach())
      D_P_real_loss = mse(D_P_real, torch.ones_like(D_P_real))
      D_P_fake_loss = mse(D_P_fake, torch.zeros_like(D_P_real))
      Disc_piano_loss = D_P_real_loss + D_P_fake_loss

      fake_flute = gen_F(piano)
      D_F_real = disc_F(flute)
      D_F_fake = disc_F(fake_flute.detach())
      D_F_real_loss = mse(D_F_real, torch.ones_like(D_F_real))
      D_F_fake_loss = mse(D_F_fake, torch.zeros_like(D_F_real))
      Disc_flute_loss = D_F_real_loss + D_F_fake_loss

      # Overall discriminator loss

      D_loss = (Disc_flute_loss + Disc_piano_loss)/2
      with torch.no_grad():
        disc_loss.update(D_loss, 1)
        Discriminator_Loss.append(D_loss)

      opt_disc.zero_grad()
      d_scaler.scale(D_loss).backward()
      d_scaler.step(opt_disc)
      d_scaler.update()
    
    # Train Generators P and F
    with torch.cuda.amp.autocast():

      # Adversarial Loss for both generators
      D_P_fake = disc_P(fake_piano)
      D_F_fake = disc_F(fake_flute)
      loss_G_F = mse(D_F_fake, torch.ones_like(D_F_fake))
      loss_G_P = mse(D_P_fake, torch.ones_like(D_P_fake))

      # Cycle Loss
      cycle_piano = gen_P(fake_flute)
      cycle_flute = gen_F(fake_piano)
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

      opt_gen.zero_grad()
      g_scaler.scale(G_loss).backward()
      g_scaler.step(opt_gen)
      g_scaler.update()

      with torch.no_grad():
        gen_loss.update(G_loss, 1)
        Generator_Loss.append(G_loss)


      if idx % 200 == 0:
        with torch.no_grad():
          progress.display(idx)
        save_image(fake_piano, config.SAVED_IMAGES_DIR+f"piano_{epoch}_{idx}.png")
        save_image(fake_flute, config.SAVED_IMAGES_DIR+f"flute_{epoch}_{idx}.png")
        val_dataset = PianoFluteDataset(root_piano=config.PIANO_TRAIN_DIR, root_flute=config.FLUTE_TRAIN_DIR, transform=config.transforms, isTrain=False)

        val_loader = DataLoader(
            val_dataset,
            batch_size = config.BATCH_SIZE,
            shuffle=False
        )
        val(gen_F, gen_P, disc_F, disc_P, mse, L1, val_loader, idx, epoch, folder=config.SAVED_IMAGES_DIR)
  
  return (Discriminator_Loss, Generator_Loss)

def main():
  disc_P = Discriminator(in_channels=1).to(config.DEVICE)
  disc_F = Discriminator(in_channels=1).to(config.DEVICE)
  gen_P = Generator(img_channels=1, num_residuals=9).to(config.DEVICE)
  gen_F = Generator(img_channels=1, num_residuals=9).to(config.DEVICE)

  opt_disc = optim.Adam(
    list(disc_P.parameters()) + list(disc_F.parameters()),
    lr = config.LEARNING_RATE,
    betas=(0.5, 0.999),
  )

  opt_gen = optim.Adam(
    list(gen_P.parameters()) + list(gen_F.parameters()),
    lr = config.LEARNING_RATE,
    betas=(0.5, 0.999),
  )

  L1 = nn.L1Loss() # For cycle consistency and identity loss
  mse = nn.MSELoss()

  if config.LOAD_MODEL:
    load_checkpoint(
      config.CHECKPOINT_GEN_P, gen_P, opt_gen, config.LEARNING_RATE,
    )
    load_checkpoint(
      config.CHECKPOINT_GEN_F, gen_F, opt_gen, config.LEARNING_RATE,
    )
    load_checkpoint(
      config.CHECKPOINT_CRITIC_P, disc_P, opt_disc, config.LEARNING_RATE,
    )
    load_checkpoint(
      config.CHECKPOINT_CRITIC_F, disc_F, opt_disc, config.LEARNING_RATE,
    )

  dataset = PianoFluteDataset(
    root_piano=config.PIANO_TRAIN_DIR, root_flute=config.FLUTE_TRAIN_DIR, transform=config.transforms
  )

  loader = DataLoader(
    dataset,
    batch_size = config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    pin_memory=True
  )

  g_scaler = torch.cuda.amp.GradScaler()
  d_scaler = torch.cuda.amp.GradScaler()

  D_Loss, G_Loss = [], []

  for epoch in range(config.NUM_EPOCHS):
    print("=======================> Epoch " + str(epoch) + " =======================================>")
    d_Loss, g_Loss = train_fn(epoch, disc_P, disc_F, gen_F, gen_P, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

    if config.SAVE_MODEL and epoch % 5 == 0:
      save_checkpoint(gen_P, opt_gen, filename=config.CHECKPOINT_GEN_P)
      save_checkpoint(gen_F, opt_gen, filename=config.CHECKPOINT_GEN_F)
      save_checkpoint(disc_P, opt_disc, filename=config.CHECKPOINT_CRITIC_P)
      save_checkpoint(disc_F, opt_disc, filename=config.CHECKPOINT_CRITIC_F)

    D_Loss.extend(d_Loss), G_Loss.extend(g_Loss)
    
  
  draw_result(range(config.NUM_EPOCHS*len(loader)), D_Loss, G_Loss, "Training_Loss_Curve")

if __name__ == "__main__":
  main()

