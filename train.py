import torch
from dataset import dataset
from utils import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from qqdm import qqdm
from torchvision.utils import save_image, make_grid
from discriminator_model import Discriminator
from generator_model import Generator

init_seeds()
gen_xy = Generator(img_channels=3).to(config.DEVICE)
gen_yx = Generator(img_channels=3).to(config.DEVICE)
disc_x = Discriminator(in_channels=3).to(config.DEVICE)
disc_y = Discriminator(in_channels=3).to(config.DEVICE)
g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()
optimizer_disc = optim.Adam(
    list(disc_x.parameters()) + list(disc_y.parameters()),
    lr=config.LEARNING_RATE,
    betas=(0.5, 0.999),
)
optimizer_disc.param_groups[0]['capture_step'] = True
optimizer_gen = optim.Adam(
    list(gen_xy.parameters()) + list(gen_yx.parameters()),
    lr=config.LEARNING_RATE,
    betas=(0.5, 0.999),
)
optimizer_gen.param_groups[0]['capture_step'] = True
L1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()
if config.LOAD_MODEL:
    load_checkpoint(config.CHECKPOINT_GEN_xy, gen_xy, optimizer_gen, config.LEARNING_RATE)
    load_checkpoint(config.CHECKPOINT_GEN_yx, gen_yx, optimizer_gen, config.LEARNING_RATE)
    load_checkpoint(config.CHECKPOINT_CRITIC_x, disc_x, optimizer_disc, config.LEARNING_RATE)
    load_checkpoint(config.CHECKPOINT_CRITIC_y, disc_y, optimizer_disc, config.LEARNING_RATE)

x_dataset = dataset("inputs/trainA/", config.transform)
y_dataset = dataset("inputs/trainB/", config.transform)
x_loader = DataLoader(dataset=x_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
y_loader = DataLoader(dataset=y_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
mkdir('outputs/')
for epoch in range(config.EPOCHS):
    progressbar = qqdm(x_loader)
    for idx, x in enumerate(progressbar):
        y = next(iter(y_loader))
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        with torch.cuda.amp.autocast():
            # train discriminator
            fake_x = gen_yx(y)
            disc_x_real = disc_x(x)
            disc_x_fake = disc_x(fake_x.detach())
            disc_x_loss = mse_loss(disc_x_real, torch.ones_like(disc_x_real)) + mse_loss(disc_x_fake, torch.zeros_like(disc_x_fake))
            
            fake_y = gen_xy(x)
            disc_y_real = disc_y(y)
            disc_y_fake = disc_y(fake_y.detach())
            disc_y_loss = mse_loss(disc_y_real, torch.ones_like(disc_y_real)) + mse_loss(disc_y_fake, torch.zeros_like(disc_y_fake))
        
            disc_loss = (disc_x_loss + disc_y_loss) / 2
            
        optimizer_disc.zero_grad()
        d_scaler.scale(disc_loss).backward()
        d_scaler.step(optimizer_disc)
        d_scaler.update()
        
        # train generator
        with torch.cuda.amp.autocast():
            disc_x_fake = disc_x(fake_x)
            disc_y_fake = disc_y(fake_y)
            adversarial_loss = mse_loss(disc_x_fake, torch.ones_like(disc_x_fake)) + mse_loss(disc_y_fake, torch.ones_like(disc_y_fake))
            
            cycle_xyx = gen_yx(fake_y)
            cycle_yxy = gen_xy(fake_x)
            cycle_loss = L1_loss(cycle_xyx, x) + L1_loss(cycle_yxy, y)
            
            if config.LAMBDA_IDENTITY > 0:
                identity_x = gen_yx(x)
                identity_y = gen_xy(y)
                identity_loss = L1_loss(x, identity_x) + L1_loss(y, identity_y)
            
            gen_loss = adversarial_loss + config.LAMBDA_IDENTITY * identity_loss + config.LAMBDA_CYCLE * cycle_loss
            
        optimizer_gen.zero_grad()
        g_scaler.scale(gen_loss).backward()
        g_scaler.step(optimizer_gen)
        g_scaler.update()
        progressbar.set_infos({
            'generator_loss': f'{gen_loss.item():.4f}',
            'discriminator_loss': f'{disc_loss.item():.4f}',
            'Eopch': f'{epoch}', 
        })
        if idx % 20 == 0:
            fake_y_and_x = make_grid(torch.cat((fake_y, x), dim=0), nrow=config.BATCH_SIZE)
            fake_x_and_y = make_grid(torch.cat((fake_x, y), dim=0), nrow=config.BATCH_SIZE)
            save_image((fake_y_and_x + 1) / 2, "outputs/fake_y_and_x_idx_{}.png".format(idx))
            save_image((fake_x_and_y + 1) / 2, "outputs/fake_x_and_y_idx_{}.png".format(idx))
    if config.SAVE_MODEL:
        save_checkpoint(config.CHECKPOINT_GEN_xy, gen_xy, optimizer_gen)
        save_checkpoint(config.CHECKPOINT_GEN_yx, gen_yx, optimizer_gen)
        save_checkpoint(config.CHECKPOINT_CRITIC_x, disc_x, optimizer_disc)
        save_checkpoint(config.CHECKPOINT_CRITIC_y, disc_y, optimizer_disc)