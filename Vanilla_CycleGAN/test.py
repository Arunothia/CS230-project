import torch
from dataset import PianoFluteDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
import numpy as np

def pad_zeros(image):
  # resizing to 336 x 336 when original size is 336 x 250
  return np.pad(image, ((0, 0), (0, 86)), 'constant')

def getPair(file):
    flute_path = config.PIANO_TEST_DIR + file
    piano_path = config.FLUTE_TEST_DIR + file

    flute_img = np.repeat(np.expand_dims(pad_zeros(np.load(flute_path)), axis=-1), 1, axis=-1)
    piano_img = np.repeat(np.expand_dims(pad_zeros(np.load(piano_path)), axis=-1), 1, axis=-1)

    flute_img = np.exp(flute_img)
    piano_img = np.exp(piano_img)
    
    augmentations = config.transform(image=flute_img, image0=piano_img)
    flute_img = augmentations["image"]
    piano_img = augmentations["image0"]
    
    return flute_img, piano_img

def test():
  pass  


def main():
  gen_P = Generator(img_channels=1, num_residuals=9).to(config.DEVICE)
  gen_F = Generator(img_channels=1, num_residuals=9).to(config.DEVICE)

  opt_gen = optim.Adam(
    list(gen_P.parameters()) + list(gen_F.parameters()),
    lr = config.LEARNING_RATE,
    betas=(0.5, 0.999),
  )

  load_checkpoint(
    config.CHECKPOINT_GEN_P, gen_P, opt_gen, config.LEARNING_RATE,
  )
  load_checkpoint(
    config.CHECKPOINT_GEN_F, gen_F, opt_gen, config.LEARNING_RATE,
  )

  for file in ["andersen41-14_chunk6.npy", "bor_ps2_chunk12.npy", "mz_330_2_chunk16.npy"]:
    flute_img, piano_img = getPair(file)
    fake_flute, fake_piano = gen_F(piano_img), gen_P(flute_img)
    save_image(fake_piano, config.SAVED_IMAGES_DIR+f"fake_piano_{file}.png")
    save_image(fake_flute, config.SAVED_IMAGES_DIR+f"fake_piano_{file}.png")

if __name__ == "__main__":
  main()