import os
import glob
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

IMAGE_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 10
LR = 0.0002
LAMBDA_L1 = 100.0  # Reconstruction weight for temporal frame prediction

# 1. Dataset for Paired Temporal Frames (I_t -> I_{t+1})
class SequentialFramesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.frame_pairs = []
        
        # Traverse each subfolder containing extracted 10 FPS sequences
        for video_dir in Path(root_dir).glob("*/*"):
            if video_dir.is_dir():
                frames = sorted(list(video_dir.glob("*.jpg")))
                for i in range(len(frames) - 1):
                    self.frame_pairs.append((frames[i], frames[i + 1]))

    def __len__(self):
        return len(self.frame_pairs)

    def __getitem__(self, idx):
        past_path, future_path = self.frame_pairs[idx]
        past_img = Image.open(past_path).convert("RGB")
        future_img = Image.open(future_path).convert("RGB")

        if self.transform:
            past_img = self.transform(past_img)
            future_img = self.transform(future_img)

        return past_img, future_img

# 2. Paper Phase II: Encoder-Decoder Generator Architecture
class EncoderDecoderGenerator(nn.Module):
    def __init__(self):
        super(EncoderDecoderGenerator, self).__init__()
        # Encoder: Compresses spatial features of current frame I_t
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True))

        # Decoder: Reconstructs next predicted frame I_{t+1}
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d1 = self.dec1(e4)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        out = self.dec4(d3)
        return out

# 3. Paper Phase II: Discriminator (Conditioned on Frame Transition)
class FrameDiscriminator(nn.Module):
    def __init__(self):
        super(FrameDiscriminator, self).__init__()
        # Takes both current frame (3 channels) + predicted/real future frame (3 channels) = 6 channels
        self.net = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, current_frame, future_frame):
        combined = torch.cat([current_frame, future_frame], dim=1)
        return self.net(combined).view(-1)

# Execution Pipeline
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = SequentialFramesDataset("dataset_frames", transform=transform)
    if len(dataset) == 0:
        print("No frame pairs found! Ensure dataset_frames folder contains extracted images.")
        exit()

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = EncoderDecoderGenerator().to(device)
    discriminator = FrameDiscriminator().to(device)

    criterion_bce = nn.BCELoss()
    criterion_l1 = nn.L1Loss()

    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

    print(f"Starting Paper's Exact Frame-Prediction GAN Training on {device} ({len(dataset)} frame pairs)...")

    for epoch in range(EPOCHS):
        total_loss_g, total_loss_d = 0.0, 0.0
        for i, (current_frame, real_future_frame) in enumerate(dataloader):
            b_size = current_frame.size(0)
            current_frame = current_frame.to(device)
            real_future_frame = real_future_frame.to(device)

            real_labels = torch.ones(b_size, device=device)
            fake_labels = torch.zeros(b_size, device=device)

            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            pred_real = discriminator(current_frame, real_future_frame)
            loss_d_real = criterion_bce(pred_real, real_labels)

            fake_future_frame = generator(current_frame)
            pred_fake = discriminator(current_frame, fake_future_frame.detach())
            loss_d_fake = criterion_bce(pred_fake, fake_labels)

            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()
            pred_fake_for_g = discriminator(current_frame, fake_future_frame)
            loss_g_adv = criterion_bce(pred_fake_for_g, real_labels)
            loss_g_l1 = criterion_l1(fake_future_frame, real_future_frame) * LAMBDA_L1

            loss_g = loss_g_adv + loss_g_l1
            loss_g.backward()
            optimizer_G.step()

            total_loss_g += loss_g.item()
            total_loss_d += loss_d.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss G: {total_loss_g/len(dataloader):.4f}, Loss D: {total_loss_d/len(dataloader):.4f}")

    torch.save(generator.state_dict(), "paper_future_frame_generator.pth")
    print("Training finished! Generator saved as 'paper_future_frame_generator.pth'.")