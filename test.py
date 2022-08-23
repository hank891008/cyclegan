import torch
from dataset import dataset
from utils import *
from torch.utils.data import DataLoader
import config
from qqdm import qqdm
from torchvision.utils import save_image
from generator_model import Generator

init_seeds()
mkdir('result/')
gen_xy = Generator(img_channels=3).to(config.DEVICE)
gen_yx = Generator(img_channels=3).to(config.DEVICE)
if config.LOAD_MODEL:
    load_checkpoint(config.CHECKPOINT_GEN_xy, gen_xy)
    load_checkpoint(config.CHECKPOINT_GEN_yx, gen_yx)

x_dataset = dataset("inputs/testA/", config.transform)
y_dataset = dataset("inputs/testB/", config.transform)
x_loader = DataLoader(dataset=x_dataset, batch_size=1, num_workers=config.NUM_WORKERS, pin_memory=True)
y_loader = DataLoader(dataset=y_dataset, batch_size=1, num_workers=config.NUM_WORKERS, pin_memory=True)

progressbar = qqdm(x_loader)
with torch.no_grad():
    for i, x in enumerate(progressbar):
            x = x.to(config.DEVICE)
            save_image((torch.cat((x, gen_xy(x))) + 1) / 2, f'result/fake_zebra{i}.png')
progressbar = qqdm(y_loader)
with torch.no_grad():
    for i, y in enumerate(progressbar):
            y = y.to(config.DEVICE)
            save_image((torch.cat((y, gen_yx(y))) + 1) / 2, f'result/fake_horse{i}.png')