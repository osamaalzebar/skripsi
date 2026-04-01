import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ------------------------------------------------------------
# Import custom models
# ------------------------------------------------------------
from train_inceptionV3_origin_pp import InceptionV3ConcatHead


# ------------------------------------------------------------
# Custom DenseNet201 Multi-Branch (same as training)
# ------------------------------------------------------------
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

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            d3 = self.block3(self.block2(self.block1(self.initial(dummy))))
            d4 = self.norm5(self.block4(d3))
            c3, c4 = d3.shape[1], d4.shape[1]

        self.branch1_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(c3, 512)
        )

        self.branch2_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(c4, 512)
        )

        self.branch3_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(c4, 512)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.block1(x)
        x = self.block2(x)
        x3 = self.block3(x)
        x4 = self.norm5(self.block4(x3))

        f1 = self.branch1_fc(x3)
        f2 = self.branch2_fc(x4)
        f3 = self.branch3_fc(x4)

        return self.classifier(torch.cat([f1, f2, f3], dim=1))


# ------------------------------------------------------------
# Dataset (shared)
# ------------------------------------------------------------
class EnsembleDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir

        self.tf_incv3 = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.tf_224 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = int(self.data.iloc[idx, 1])

        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        return (
            self.tf_incv3(img),   # InceptionV3
            self.tf_224(img),     # ResNet50
            self.tf_224(img),     # DenseNet
            label
        )


# ------------------------------------------------------------
# Ensemble Evaluation (3-Model Soft Voting)
# ------------------------------------------------------------
@torch.no_grad()
def evaluate_ensemble(
    img_dir,
    csv_path,
    incv3_ckpt,
    resnet_ckpt,
    densenet_ckpt,
    batch_size=8
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    loader = DataLoader(
        EnsembleDataset(csv_path, img_dir),
        batch_size=batch_size,
        shuffle=False
    )

    # ---------- Load Models ----------
    incv3 = InceptionV3ConcatHead(num_classes=4).to(device)
    ckpt = torch.load(incv3_ckpt, map_location=device)
    incv3.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    incv3.eval()

    resnet = models.resnet50(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 4)
    resnet.load_state_dict(torch.load(resnet_ckpt, map_location=device))
    resnet.to(device).eval()

    densenet = DenseNet201_MultiBranch(num_classes=4).to(device)
    densenet.load_state_dict(torch.load(densenet_ckpt, map_location=device))
    densenet.eval()

    y_true, y_pred = [], []

    # ---------- Inference ----------
    for x_inc, x_res, x_den, labels in loader:
        x_inc = x_inc.to(device)
        x_res = x_res.to(device)
        x_den = x_den.to(device)

        p1 = F.softmax(incv3(x_inc), dim=1)
        p2 = F.softmax(resnet(x_res), dim=1)
        p3 = F.softmax(densenet(x_den), dim=1)

        probs = (p1 + p2 + p3) / 3.0
        preds = torch.argmax(probs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

    # ---------- Metrics ----------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n📊 3-MODEL ENSEMBLE (SOFT VOTING)")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    # ---------- Confusion Matrix ----------
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap="Blues", linewidths=1,
        linecolor="black", square=True
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix – 3-Model Soft Voting Ensemble")
    plt.tight_layout()
    plt.savefig("confusion_matrix_ensemble_3models.png", dpi=300)
    plt.close()

    print("✅ Confusion matrix saved as confusion_matrix_ensemble_3models.png")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    evaluate_ensemble(
        img_dir=r"C:\Users\MSI INFINITE S3\Downloads\brisc_2\classification_task\test\data",
        csv_path= r"C:\Users\MSI INFINITE S3\Downloads\brisc_2\classification_task\test\labels.csv",
        incv3_ckpt=r"C:\Users\MSI INFINITE S3\PycharmProjects\origin\best_incv3_concat.pth",
        resnet_ckpt=r"best_resnet50_brain_tumor.pth",
        densenet_ckpt=r"best_densenet201_multibranch_brain_tumor.pth",
        batch_size=8
    )
