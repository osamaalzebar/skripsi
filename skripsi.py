import os
import csv
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ------------------------------------------------------------
# Import your InceptionV3 model
# ------------------------------------------------------------
from train_inceptionV3_origin_pp import InceptionV3ConcatHead

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
NUM_CLASSES = 4
CLASS_NAMES = ["Class 0", "Class 1", "Class 2", "Class 3"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ------------------------------------------------------------
# Dataset (shared for both models)
# ------------------------------------------------------------
class EnsembleDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir

        self.transform_incv3 = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

        self.transform_resnet = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = int(self.data.iloc[idx, 1])

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        return (
            self.transform_incv3(image),
            self.transform_resnet(image),
            label
        )

# ------------------------------------------------------------
# Confusion Matrix Plot
# ------------------------------------------------------------
def plot_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=1,
        linecolor="black",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix – Soft Voting Ensemble")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Confusion matrix saved to: {out_path}")

# ------------------------------------------------------------
# Ensemble Evaluation (Soft Voting)
# ------------------------------------------------------------
@torch.no_grad()
def evaluate_ensemble(
    img_dir,
    csv_path,
    incv3_ckpt,
    resnet_ckpt,
    batch_size=8,
    num_workers=4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = EnsembleDataset(csv_path, img_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # -------- Load InceptionV3 --------
    incv3 = InceptionV3ConcatHead(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(incv3_ckpt, map_location=device)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    incv3.load_state_dict(ckpt, strict=False)
    incv3.eval()

    # -------- Load ResNet50 --------
    resnet = models.resnet50(weights=None)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, NUM_CLASSES)
    resnet.load_state_dict(torch.load(resnet_ckpt, map_location=device))
    resnet = resnet.to(device)
    resnet.eval()

    y_true, y_pred = [], []

    # -------- Inference --------
    for x_incv3, x_resnet, labels in loader:
        x_incv3 = x_incv3.to(device)
        x_resnet = x_resnet.to(device)

        logits_incv3 = incv3(x_incv3)
        logits_resnet = resnet(x_resnet)

        probs_incv3 = F.softmax(logits_incv3, dim=1)
        probs_resnet = F.softmax(logits_resnet, dim=1)

        # ===== Soft Voting (Average Probabilities) =====
        ensemble_probs = (probs_incv3 + probs_resnet) / 2
        preds = torch.argmax(ensemble_probs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

    # ------------------------------------------------
    # Metrics
    # ------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n📊 ENSEMBLE RESULTS (Soft Voting)")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    # ------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, CLASS_NAMES, "confusion_matrix_ensemble.png")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    evaluate_ensemble(
        img_dir=r"C:\Users\MSI INFINITE S3\Downloads\brisc_2\classification_task\test\data",
        csv_path= r"C:\Users\MSI INFINITE S3\Downloads\brisc_2\classification_task\test\labels.csv",
        incv3_ckpt=r"C:\Users\MSI INFINITE S3\PycharmProjects\origin\best_incv3_concat.pth",
        resnet_ckpt=r"best_resnet50_brain_tumor.pth",
        batch_size=8,
        num_workers=4
    )
