import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import args

# Arguments
opt = args.get_setup_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = opt.input_data_train_path
VAL_DIR = opt.input_data_val_path
PIANO_TRAIN_DIR = opt.input_train_piano_path
FLUTE_TRAIN_DIR = opt.input_train_flute_path
PIANO_TEST_DIR = opt.input_test_piano_path
FLUTE_TEST_DIR = opt.input_test_flute_path
SAVED_IMAGES_DIR = opt.output_path
BATCH_SIZE = opt.batch_size
LEARNING_RATE = opt.lr
LAMBDA_IDENTITY = opt.lambda_identity
LAMBDA_CYCLE = opt.lambda_cycle
NUM_WORKERS = opt.num_workers
NUM_EPOCHS = opt.num_epochs
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_P = "genp.pth.tar"
CHECKPOINT_GEN_F = "genf.pth.tar"
CHECKPOINT_CRITIC_P = "criticp.pth.tar"
CHECKPOINT_CRITIC_F = "criticf.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=336, height=336),
        #A.HorizontalFlip(p=0.5),
        #A.Normalize(mean=0.5, std=0.5, max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)