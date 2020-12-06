import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
parser.add_argument("--name", type=str, default="gan", help="experiment name")

opt = parser.parse_args()
print(opt)

os.makedirs("images/"+opt.name, exist_ok=True)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.num_classes, opt.num_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model  = nn.Sequential(
            *block(opt.latent_dim + opt.num_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )


    def forward(self, z, labels):
        # Concatenate label embedding and image
        gen_input = torch.cat((z, self.label_emb(labels)), 1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_emb = nn.Embedding(opt.num_classes, opt.num_classes)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape))+opt.num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        # Concatenate label embedding and image
        d_in = torch.cat((img_flat, self.label_emb(labels)), 1)
        validity = self.model(d_in)

        return validity

# Loss function
gan_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    gan_loss.cuda()
print("----Generator----")
print(generator)
print("----Discriminator----")
print(discriminator)

# Configure data loader
os.makedirs("./data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Tensorboard
os.makedirs('runs', exist_ok=True)
writer = SummaryWriter('runs/'+ opt.name)

def sample_image(n_row, batches_done, writer):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = torch.tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim)), dtype=torch.float32).cuda()
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = torch.tensor(labels, dtype=torch.long).cuda()
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/"+opt.name+"/%d.png" % batches_done, nrow=n_row, normalize=True)
    img_grid = make_grid(gen_imgs.data[:25])
    writer.add_image('cgan', img_grid, global_step=batches_done)
# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):


        # Adversrial ground truths
        valid = torch.ones(imgs.size(0), 1, dtype=torch.float32, requires_grad=False).cuda()
        fake = torch.zeros(imgs.size(0), 1, dtype=torch.float32, requires_grad=False).cuda()

        # Configure input
        real_img = torch.tensor(imgs, dtype=torch.float32).cuda()
        real_labels = torch.tensor(labels, dtype=torch.long).cuda()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = torch.randn(imgs.size(0), opt.latent_dim, dtype=torch.float32).cuda()
        gen_labels = torch.tensor(np.random.randint(0, opt.num_classes, size=imgs.size(0)), dtype=torch.long).cuda()

        # Generator a batch of images
        gen_imgs = generator(z, gen_labels)

        d_real_loss = gan_loss(discriminator(real_img, real_labels), valid)
        d_fake_loss = gan_loss(discriminator(gen_imgs.detach(), gen_labels), fake)  # detach avoid to update G

        d_loss = (d_real_loss + d_fake_loss) / 2 # slow down the rate at which D learns relative to G by dividing 2

        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Loss measures generator's ability to fool the discriminator
        g_loss = gan_loss(discriminator(gen_imgs, gen_labels), valid) # 实际G的损失函数用的是log(D(G(z))
        # g_loss = torch.mean(torch.log(1. - discriminator(gen_imgs)))

        g_loss.backward()
        optimizer_G.step()

        print(
            "[Epoch: %d/%d] [Batch: %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done, writer=writer)
            writer.add_scalar('D loss', d_loss.item(), batches_done)
            writer.add_scalar('G loss', g_loss.item(), batches_done)