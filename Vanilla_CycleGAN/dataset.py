from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np

class PianoFluteDataset(Dataset):
    def __init__(self, root_piano, root_flute, transform=None):
        self.root_piano = root_piano
        self.root_flute = root_flute
        self.transform = transform

        self.piano_images = os.listdir(root_piano)
        self.flute_images = os.listdir(root_flute)
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

        flute_img = np.array(Image.open(flute_path).Convert("RGB"))
        piano_img = np.array(Image.open(piano_path).Convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=flute_img, image0=piano_img)
            flute_img = augmentations["image"]
            piano_img = augmentations["image0"]
        
        return flute_img, piano_img