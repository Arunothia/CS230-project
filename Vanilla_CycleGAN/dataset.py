from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np
from random import sample

def pad_zeros(image):
  # resizing to 336 x 336 when original size is 336 x 250
  return np.pad(image, ((0, 0), (0, 86)), 'constant')

class PianoFluteDataset(Dataset):
    def __init__(self, root_piano, root_flute, transform=None, isTrain=True):
        self.root_piano = root_piano
        self.root_flute = root_flute
        self.transform = transform

        if isTrain:
            self.flute_images = os.listdir(root_flute)[0: 100]
        else:
            self.flute_images = os.listdir(root_flute)[101:121]

        self.piano_images = self.flute_images 
        self.length_dataset = max(len(self.piano_images), len(self.flute_images))

        self.piano_dataset_length = len(self.piano_images)
        self.flute_dataset_length = len(self.flute_images)
    
    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        flute_image = self.flute_images[index % self.flute_dataset_length]
        piano_image = self.piano_images[index % self.piano_dataset_length]

        flute_path = os.path.join(self.root_flute, flute_image)
        piano_path = os.path.join(self.root_piano, piano_image)

        flute_img = np.repeat(np.expand_dims(pad_zeros(np.load(flute_path)), axis=-1), 1, axis=-1)
        piano_img = np.repeat(np.expand_dims(pad_zeros(np.load(piano_path)), axis=-1), 1, axis=-1)

        #flute_img = np.exp(flute_img)
        #piano_img = np.exp(piano_img)

        if self.transform:
            augmentations = self.transform(image=flute_img, image0=piano_img)
            flute_img = augmentations["image"]
            piano_img = augmentations["image0"]
        
        return flute_img, piano_img