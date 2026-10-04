import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed_fall_dataset"
MODEL_SAVE_PATH = BASE_DIR / "vit_fall_detection_model.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ViT input transforms (224x224)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class FallDataset(Dataset):
    def __init__(self, split_dir, transform=None):
        self.samples = []
        self.transform = transform
        
        # Label 0: FALL, Label 1: NORMAL
        fall_imgs = [(p, 0) for p in (split_dir / "fall").glob("*.*") if p.suffix.lower() in ['.jpg', '.png', '.jpeg']]
        normal_imgs = [(p, 1) for p in (split_dir / "normal").glob("*.*") if p.suffix.lower() in ['.jpg', '.png', '.jpeg']]
        
        self.samples = fall_imgs + normal_imgs
        print(f"Loaded {len(fall_imgs)} falls and {len(normal_imgs)} normal images from {split_dir.name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def get_vit_model(num_classes=2):
    weights = models.ViT_B_16_Weights.DEFAULT
    model = models.vit_b_16(weights=weights)
    
    # Backbone freeze: sirf classification head train hogi (fast execution)
    for param in model.parameters():
        param.requires_grad = False
        
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

def train():
    train_dataset = FallDataset(DATA_DIR / "train", transform=train_transform)
    val_dataset = FallDataset(DATA_DIR / "val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = get_vit_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.heads.head.parameters(), lr=1e-3, weight_decay=1e-4)

    # Set explicitly to 2 epochs
    epochs = 1
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if i % 10 == 0 or i == len(train_loader):
                print(f"Epoch [{epoch+1}/{epochs}] Step [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")

        acc = correct / total * 100
        print(f"\n--- Epoch [{epoch+1}/{epochs}] Summary: Loss = {total_loss/len(train_loader):.4f} | Accuracy = {acc:.2f}% ---\n")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved successfully to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()