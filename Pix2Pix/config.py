import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIANO_TRAIN_DIR = "../../dataset/processedData/trainSet/flute/cqtChunks/"
FLUTE_TRAIN_DIR = "../../dataset/processedData/trainSet/piano/cqtChunks/"
SAVED_IMAGES_DIR = "../../dataset/Pix2Pix_evaluation/" 
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
L1_LAMBDA = 100
NUM_WORKERS = 2
NUM_EPOCHS = 50
IMAGE_SIZE = 336
CHANNELS_IMG = 1
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN = "genp.pth.tar"
CHECKPOINT_CRITIC = "criticp.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=336, height=336),
        #A.HorizontalFlip(p=0.5),
        #A.Normalize(mean=0.5, std=0.5, max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)