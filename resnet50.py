import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim

# --- Custom Dataset class ---
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


def main():
    # --- Paths ---
    train_img_dir = r"C:\Users\MSI INFINITE S3\Downloads\brisc_2\classification_task\train\data"
    train_csv = r"C:\Users\MSI INFINITE S3\Downloads\brisc_2\classification_task\train\labels.csv"
    val_img_dir = r"C:\Users\MSI INFINITE S3\Downloads\brisc_2\classification_task\validate\data"
    val_csv = r"C:\Users\MSI INFINITE S3\Downloads\brisc_2\classification_task\validate\labels.csv"
    save_path = "best_resnet50_brain_tumor.pth"

    # --- Hyperparameters ---
    num_epochs = 28
    batch_size = 8
    learning_rate = 5e-5
    num_classes = 4

    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ Using GPU: {gpu_name}")
    else:
        print("⚠️ GPU not found. Using CPU instead — training will be slower.")

    # --- Transforms ---
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

    # --- Datasets & Loaders ---
    train_dataset = BrainTumorDataset(train_csv, train_img_dir, transform=train_transform)
    val_dataset = BrainTumorDataset(val_csv, val_img_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # --- Model setup ---
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Fine-tune all layers
    for param in model.parameters():
        param.requires_grad = True

    # Replace the final FC layer to match 4 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # --- Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training and Validation Loop ---
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        model.train()
        running_loss = 0.0
        total, correct = 0, 0

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

        # --- Validation ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # --- Save best model ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved new best model with Val Acc: {val_acc:.2f}%")

    print(f"\n✅ Training complete. Best model saved to: {save_path}")


if __name__ == "__main__":
    main()
