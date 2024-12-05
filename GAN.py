# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 09:37:20 2024

@author: fadhi
"""

from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Define transformations
my_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Config class to store hyperparameters
class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 3e-4
    zDIm = 128
    imageDim = 28 * 28 * 1
    batchSize = 32
    numEpochs = 100
    logStep = 625

# Load dataset
dataset = datasets.MNIST(root="dataset/", transform=my_transforms, download=True)
loader = DataLoader(dataset, batch_size=Config.batchSize, shuffle=True)

# Generator Class
class Generator(nn.Module):
    def __init__(self, zDIm, imageDim, hiddenDim=512):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(zDIm, hiddenDim),
            nn.LeakyReLU(0.2),
            nn.Linear(hiddenDim, imageDim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

# Discriminator Class
class Discriminator(nn.Module):
    def __init__(self, inFeats, hiddenDim=512):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(inFeats, hiddenDim),
            nn.LeakyReLU(0.2),
            nn.Linear(hiddenDim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

# Initialize models
discriminator = Discriminator(Config.imageDim).to(Config.device)
generator = Generator(Config.zDIm, Config.imageDim).to(Config.device)
fixed_noise = torch.randn((Config.batchSize, Config.zDIm)).to(Config.device)

# Optimizers
discriminator_opt = optim.Adam(discriminator.parameters(), lr=Config.lr)
generator_opt = optim.Adam(generator.parameters(), lr=Config.lr)

# Loss function
criterion = nn.BCELoss()

# TensorBoard SummaryWriters
Fake_logs = SummaryWriter(f"C:/Users/fadhi/OneDrive/Desktop/GAN_Practical/Logs/Fake")
Right_logs = SummaryWriter(f"C:/Users/fadhi/OneDrive/Desktop/GAN_Practical/Logs/Real")

# Training loop
step = 0
for epoch in range(Config.numEpochs):
    print('-'*80)
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(Config.device)
        batchSize = real.shape[0]

        # Train Discriminator
        noise = torch.randn(batchSize, Config.zDIm).to(Config.device)
        fake = generator(noise)
        
        discReal = discriminator(real).view(-1)
        lossDreal = criterion(discReal, torch.ones_like(discReal))
        
        discFake = discriminator(fake.detach()).view(-1)
        lossDfake = criterion(discFake, torch.zeros_like(discFake))
        
        lossD = (lossDreal + lossDfake) / 2
        discriminator_opt.zero_grad()
        lossD.backward()
        discriminator_opt.step()

        # Train Generator
        output = discriminator(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        generator_opt.zero_grad()
        lossG.backward()
        generator_opt.step()

        if batch_idx % Config.logStep == 0:
            print(f"Epoch [{epoch}/{Config.numEpochs}] Batch {batch_idx}/{len(loader)} \n Loss Disc: {lossD:.4f}, Loss Gen: {lossG:.4f}")
            with torch.no_grad():
                fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                imgGridFake = torchvision.utils.make_grid(fake, normalize=True)
                imgGridReal = torchvision.utils.make_grid(data, normalize=True)
                Fake_logs.add_image("Mnist Fake", imgGridFake, global_step=step)
                Right_logs.add_image("Mnist Real", imgGridReal, global_step=step)
                Fake_logs.flush()
                Right_logs.flush()
                step += 1

Right_logs.close()
Fake_logs.close()
