import argparse
import os
import numpy as np
import math
import time

from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from fid_evaluation import fid_score
from log_writer import write_log

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=50, help="size of the batches")
parser.add_argument("--g_lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--g_b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--g_b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--d_lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--d_b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--d_b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--evaluate_interval", type=int, default=100, help="interval between generator evaluation using FID")
parser.add_argument("--save_interval", type=int, default=500, help="interval between generator saving")
parser.add_argument("--outdir", type=str, help="Training output directory")
parser.add_argument("--dataset_folder_dir", type=str, help="Image dataset folder directory")
parser.add_argument("--dataset_dir", type=str, help="Image dataset directory")
opt = parser.parse_args()

msg = f"""
====================
  Training Options:
====================

Epoch: {opt.n_epochs}
Dataset directory: {opt.dataset_folder_dir}
Output Directory: {opt.outdir}
Image Size: {opt.img_size} x {opt.img_size}
Evaluation Interval: {opt.evaluate_interval}
"""
log_dir = os.path.join(opt.outdir, "log.txt")

print(msg)
write_log(msg="\n"+msg, dir=log_dir)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Input: opt.latent_dim x 1 x 1
        self.model = nn.Sequential(
            # layer 1: (latent_dim) -> (ngf*8) x 4 x 4
            nn.ConvTranspose2d(opt.latent_dim, 1024, 4, 1, 0, bias=False), # Changed 1024 to 1024 (ngf*8 for 128x128)
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # layer 2: (ngf*8) x 4 x 4 -> (ngf*4) x 8 x 8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False), # Changed 512 to 512 (ngf*4 for 128x128)
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # layer 3: (ngf*4) x 8 x 8 -> (ngf*2) x 16 x 16
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), # Changed 256 to 256 (ngf*2 for 128x128)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # layer 4: (ngf*2) x 16 x 16 -> (ngf) x 32 x 32
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), # Changed 128 to 128 (ngf for 128x128)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # layer 5: (ngf) x 32 x 32 -> (ngf/2) x 64 x 64
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), # Added for 128x128 images
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # layer 6: (ngf/2) x 64 x 64 -> (channels) x 128 x 128
            nn.ConvTranspose2d(64, opt.channels, 4, 2, 1, bias=False), # Added for 128x128 images
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), opt.latent_dim, 1, 1)
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Input: (channels) x 128 x 128
        self.model = nn.Sequential(
            # layer 1: (channels) x 128 x 128 -> (ndf/2) x 64 x 64
            nn.Conv2d(opt.channels, 64, 4, 2, 1, bias=False), # Changed 64 to 64 (ndf/2 for 128x128)
            nn.LeakyReLU(0.2, inplace=True),
            # layer 2: (ndf/2) x 64 x 64 -> (ndf) x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), # Changed 128 to 128 (ndf for 128x128)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # layer 3: (ndf) x 32 x 32 -> (ndf*2) x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), # Changed 256 to 256 (ndf*2 for 128x128)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # layer 4: (ndf*2) x 16 x 16 -> (ndf*4) x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), # Changed 512 to 512 (ndf*4 for 128x128)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # layer 5: (ndf*4) x 8 x 8 -> (ndf*8) x 4 x 4
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False), # Added for 128x128 images
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # layer 6: (ndf*8) x 4 x 4 -> 1 x 1 x 1
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False), # Final layer to output 1x1
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        return out.view(-1, 1)


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

msg = f"""
===========================
    Generator Structure
===========================

{generator}
"""
print(msg)
write_log(msg="\n"+msg, dir=log_dir)

msg = f"""
===============================
    Discriminator Structure
===============================

{discriminator}
"""
print(msg)
write_log(msg="\n"+msg, dir=log_dir)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

msg = f"""
Configuring dataset from {opt.dataset_folder_dir}
"""
print(msg)
write_log(msg="\n"+msg, dir=log_dir)

# Configure image data loader
transform = transforms.Compose([
    transforms.Resize(opt.img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

imagenet_dataset = datasets.ImageFolder(opt.dataset_folder_dir, transform=transform)
dataloader = DataLoader(
    imagenet_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    persistent_workers=False
)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.g_lr, betas=(opt.g_b1, opt.g_b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.d_lr, betas=(opt.d_b1, opt.d_b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

os.makedirs(opt.outdir, exist_ok=True)

total_kimg_to_train = ((len(dataloader) * opt.batch_size * opt.n_epochs) // 1000)

msg = f"Training for {opt.n_epochs} Epoch / {total_kimg_to_train} kimg"
print(msg)
write_log(msg="\n"+msg, dir=log_dir)

last_interval_time = time.time() 
last_kimg_reported = 0 
total_images_processed = 0

for epoch in range(1, opt.n_epochs + 1):
    for i, (imgs, _) in enumerate(dataloader, 1):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        total_images_processed += opt.batch_size
        current_kimg = total_images_processed // 1000

        if current_kimg >= last_kimg_reported + 4:
            time_taken = time.time() - last_interval_time
            last_kimg_reported = current_kimg 
            
            msg = ("[Time/4 kimg : %.2f sec] [Epoch %d/%d] [kimg %d/%d] [D loss: %f] [G loss: %f]"
                % (
                    time_taken,
                    epoch,
                    opt.n_epochs,
                    current_kimg,
                    total_kimg_to_train,
                    d_loss.item(),
                    g_loss.item()
                )
            )
            print(msg)
            write_log(msg="\n"+msg, dir=log_dir)
            
            last_interval_time = time.time()


        # Saving Generator
        if current_kimg % opt.save_interval == 0:
            out = os.path.join(opt.outdir, f"generator_{current_kimg}.pt")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            z = torch.randn(1, opt.latent_dim, 1, 1).to(device)
            traced_gen = torch.jit.trace(generator.to(device), z)
            traced_gen.save(out)


        # Evaluation
        if current_kimg % opt.evaluate_interval == 0:
            # FID evaluation
            eval_start_time = time.time()
            dataset_total_images = len(dataloader) * opt.batch_size
            fid_n_samples = min(50_000, dataset_total_images)
            if fid_n_samples < opt.batch_size: 
                fid_n_samples = opt.batch_size

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            score = fid_score(
                kimg=current_kimg,
                dir=opt.outdir,
                generator=generator,
                latent_dim=opt.latent_dim,
                real_dir=opt.dataset_dir,
                device=device,
                n_samples=fid_n_samples,
                batch_size=opt.batch_size
            )

            time_taken = time.time() - eval_start_time

            msg = f"Network evaluated at {current_kimg} kimg with FID: {score}. Time taken: {time_taken:.2f} seconds"
            print(msg)
            write_log(msg="\n"+msg, dir=log_dir)
            