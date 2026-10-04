import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path

EPOCHS = 10
BATCH_SIZE = 16
LR = 0.0002
IMAGE_SIZE = 64

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# Fixed Discriminator with Adaptive Global Pooling
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # Collapses spatial dimensions to 1x1
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)  # Output tensor shape matches batch_size exactly

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset_path = Path("dataset_frames")
if dataset_path.exists():
    dataset = ImageFolder(root=str(dataset_path), transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.999))

    print(f"Starting GAN Training on {device}...")
    for epoch in range(EPOCHS):
        for i, (real_images, _) in enumerate(dataloader):
            b_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Train Discriminator
            netD.zero_grad()
            label_real = torch.ones(b_size, device=device)
            output_real = netD(real_images)
            lossD_real = criterion(output_real, label_real)
            
            noise = torch.randn(b_size, 100, 1, 1, device=device)
            fake_images = netG(noise)
            label_fake = torch.zeros(b_size, device=device)
            output_fake = netD(fake_images.detach())
            lossD_fake = criterion(output_fake, label_fake)
            
            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()

            # Train Generator
            netG.zero_grad()
            output = netD(fake_images)
            lossG = criterion(output, label_real)
            lossG.backward()
            optimizerG.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}")

    torch.save(netG.state_dict(), "gan_generator.pth")
    print("GAN Training completed! Saved weights as 'gan_generator.pth'")