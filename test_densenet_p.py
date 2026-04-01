import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import seaborn as sns
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Dataset
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
# Custom Multi-Branch DenseNet201  (same as training)
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

        # ---- Auto-detect channels ----
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            d0 = self.initial(dummy)
            d1 = self.block1(d0)
            d2 = self.block2(d1)
            d3 = self.block3(d2)  # branch 1
            d4 = self.block4(d3)
            d4 = self.norm5(d4)   # branch 2 & 3

            c3 = d3.shape[1]
            c4 = d4.shape[1]

        # Branch heads
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

        # Final fused classifier
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
# Testing Script
# ---------------------------------------------------------
def main():

    # ---- Paths ----
    test_img_dir = r"C:\Users\MSI INFINITE S3\Downloads\brisc_2\classification_task\test\data"
    test_csv = r"C:\Users\MSI INFINITE S3\Downloads\brisc_2\classification_task\test\labels.csv"

    #test_img_dir =  r"C:\Users\MSI INFINITE S3\Downloads\bangladesh_data\data"
    #test_csv = r"C:\Users\MSI INFINITE S3\Downloads\bangladesh_data\labels.csv"


    model_path = "best_densenet201_multibranch_brain_tumor.pth"

    num_classes = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # ---- Transforms ----
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ---- Dataset & Loader ----
    test_dataset = BrainTumorDataset(test_csv, test_img_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # ---- Load model ----
    model = DenseNet201_MultiBranch(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Loaded model:", model_path)

    # ---- Inference ----
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ---- Metrics ----
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro")
    rec = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    print("\n===== TEST RESULTS =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # ---- Confusion Matrix ----
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        linewidths=0.5, cbar=True, square=True
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix – DenseNet201 Multi-Branch Brain Tumor")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
