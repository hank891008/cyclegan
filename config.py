import torch
import torchvision.transforms as transforms

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "result/train/"
VAL_DIR = "result/val/"
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
LAMBDA_IDENTITY = 1
LAMBDA_CYCLE = 10
IMG_SIZE = 256
NUM_WORKERS = 0
EPOCHS = 400
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_xy = "checkpoint/gen_xy.pth"
CHECKPOINT_GEN_yx = "checkpoint/gen_yx.pth"
CHECKPOINT_CRITIC_x = "checkpoint/critic_x.pth"
CHECKPOINT_CRITIC_y = "checkpoint/critic_y.pth"

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
