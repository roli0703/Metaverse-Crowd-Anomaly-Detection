import torch
import torch.nn as nn
from torchvision.utils import save_image

# 1. Generator Model Definition
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

# 2. Load Weights and Generate Synthetic Images
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)

# Load saved PGGAN weights
netG.load_state_dict(torch.load("gan_generator.pth", map_location=device))
netG.eval()

# Generate 16 synthetic frames
noise = torch.randn(16, 100, 1, 1, device=device)
with torch.no_grad():
    fake_images = netG(noise)

# Save grid image output
save_image(fake_images, "synthetic_frames_output.png", normalize=True)
print("Success! Synthetic frames saved as 'synthetic_frames_output.png'.")