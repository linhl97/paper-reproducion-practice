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
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
parser.add_argument("--name", type=str, default="gan", help="experiment name")

opt = parser.parse_args()
print(opt)

os.makedirs("images/"+opt.name, exist_ok=True)

ngf = 64
ndf = 64
nc = 1

ratio = opt.img_size / 64
c_exp = math.floor(3 + math.log(ratio, 2))

cuda = True if torch.cuda.is_available() else False

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        net = []
        net.append(nn.ConvTranspose2d(opt.latent_dim, ngf*2**c_exp, 4, 1, 0, bias=False))
        net.append(nn.BatchNorm2d(ngf*2**c_exp))
        net.append(nn.ReLU(True))

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, 2, 1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.ReLU(True))
            return layers

        for i in range(c_exp, 0, -1):
            net.extend(block(ngf*2**i, ngf*2**(i-1)))

        net.append(nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False))
        net.append(nn.Tanh())

        self.model = nn.Sequential(
            *net
        )


    def forward(self, input):
        img = self.model(input)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        net = []
        net.append(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        net.append(nn.LeakyReLU(0.2, inplace=True))

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        for i in range(c_exp):
            net.extend(block(ndf*2**i, ndf*2**(i+1)))

        net.append(nn.Conv2d(ndf*2**c_exp, 1, 4, 1, 0, bias=False))
        net.append(nn.Sigmoid())

        self.model = nn.Sequential(
            *net
        )

    def forward(self, input):
        validity = self.model(input)

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

generator.apply(weights_init)
discriminator.apply(weights_init)

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
    z = torch.tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim, 1, 1)), dtype=torch.float32).cuda()
    # Get labels ranging from 0 to n_classes for n rows
    gen_imgs = generator(z)
    save_image(gen_imgs.data, "images/"+opt.name+"/%d.png" % batches_done, nrow=n_row, normalize=True)
    img_grid = make_grid(gen_imgs.data[:25])
    writer.add_image('dcgan', img_grid, global_step=batches_done)
# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):


        # Adversrial ground truths
        valid = torch.ones(imgs.size(0), 1, dtype=torch.float32, requires_grad=False).cuda()
        fake = torch.zeros(imgs.size(0), 1, dtype=torch.float32, requires_grad=False).cuda()

        # Configure input
        real_img = torch.tensor(imgs, dtype=torch.float32).cuda()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = torch.randn(imgs.size(0), opt.latent_dim, 1, 1, dtype=torch.float32).cuda()

        # Generator a batch of images
        gen_imgs = generator(z)

        d_real_loss = gan_loss(discriminator(real_img), valid)
        d_fake_loss = gan_loss(discriminator(gen_imgs.detach()), fake)  # detach avoid to update G

        d_loss = (d_real_loss + d_fake_loss) / 2 # slow down the rate at which D learns relative to G by dividing 2

        d_loss.backward()

        for name, parms in discriminator.named_parameters():
            writer.add_histogram('D_'+name, parms.grad, epoch * len(dataloader) + i)

        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Loss measures generator's ability to fool the discriminator
        g_loss = gan_loss(discriminator(gen_imgs), valid) # 实际G的损失函数用的是log(D(G(z))
        # g_loss = torch.mean(torch.log(1. - discriminator(gen_imgs)))

        g_loss.backward()

        for name, parms in generator.named_parameters():
            writer.add_histogram('G_'+name, parms.grad, epoch * len(dataloader) + i)

        optimizer_G.step()

        print(
            "[Epoch: %d/%d] [Batch: %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=5, batches_done=batches_done, writer=writer)
            writer.add_scalar('D loss', d_loss.item(), batches_done)
            writer.add_scalar('G loss', g_loss.item(), batches_done)
