import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ---------------------------------------------------------
#  Dataset Class (same as your ResNet50 implementation)
# ---------------------------------------------------------
class BrainTumorDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = int(self.data.iloc[idx, 1])
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


# ---------------------------------------------------------
#  Custom Multi-Branch DenseNet201 Feature Extractor
# ---------------------------------------------------------
class DenseNet201_MultiBranch(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        base = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)

        self.initial = nn.Sequential(
            base.features.conv0,
            base.features.norm0,
            base.features.relu0,
            base.features.pool0
        )

        self.block1 = nn.Sequential(
            base.features.denseblock1,
            base.features.transition1
        )

        self.block2 = nn.Sequential(
            base.features.denseblock2,
            base.features.transition2
        )

        self.block3 = nn.Sequential(
            base.features.denseblock3,
            base.features.transition3
        )

        self.block4 = base.features.denseblock4
        self.norm5 = base.features.norm5

        # ---------------------------------------------------------
        #   🔍 Auto-detect output channel sizes using dummy input
        # ---------------------------------------------------------
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)

            d0 = self.initial(dummy)
            d1 = self.block1(d0)      # after DenseBlock1
            d2 = self.block2(d1)      # after DenseBlock2
            d3 = self.block3(d2)      # after DenseBlock3  ← branch1
            d4 = self.block4(d3)      # after DenseBlock4
            d4 = self.norm5(d4)       # final output        ← branch2 & branch3

            c3 = d3.shape[1]   # channels for branch 1
            c4 = d4.shape[1]   # channels for branches 2 & 3

        print(f"[DEBUG] Auto-detected channel sizes: c3={c3}, c4={c4}")

        # ---------------------------------------------------------
        # Branch Heads (now correct!)
        # ---------------------------------------------------------

        self.branch1_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(c3, 512)
        )

        self.branch2_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(c4, 512)
        )

        self.branch3_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(c4, 512)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.initial(x)

        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x4 = self.norm5(x4)

        f1 = self.branch1_fc(x3)
        f2 = self.branch2_fc(x4)
        f3 = self.branch3_fc(x4)

        fused = torch.cat([f1, f2, f3], dim=1)
        out = self.classifier(fused)
        return out


# ---------------------------------------------------------
#  Training Script
# ---------------------------------------------------------
def main():
    # Paths
    train_img_dir = r"C:\Users\MSI INFINITE S3\Downloads\brisc_2\classification_task\train\data"
    train_csv = r"C:\Users\MSI INFINITE S3\Downloads\brisc_2\classification_task\train\labels.csv"
    val_img_dir = r"C:\Users\MSI INFINITE S3\Downloads\brisc_2\classification_task\validate\data"
    val_csv = r"C:\Users\MSI INFINITE S3\Downloads\brisc_2\classification_task\validate\labels.csv"
    save_path = "best_densenet201_multibranch_brain_tumor.pth"

    # Hyperparameters
    num_epochs = 32
    batch_size = 8
    learning_rate = 1e-5
    num_classes = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Datasets & Loaders
    train_dataset = BrainTumorDataset(train_csv, train_img_dir, transform=train_transform)
    val_dataset = BrainTumorDataset(val_csv, val_img_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model
    model = DenseNet201_MultiBranch(num_classes=num_classes).to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0

    # ---------------------------------------------------------
    # Training Loop
    # ---------------------------------------------------------
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_loss = running_loss / total

        # ---------------- Validation ----------------
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved best model with Val Acc = {val_acc:.2f}%")

    print("\nTraining complete.")
    print("Best model saved at:", save_path)


if __name__ == "__main__":
    main()
