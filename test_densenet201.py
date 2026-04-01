import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn as nn

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# --------------------- Dataset Class ---------------------
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


# --------------------- Main Testing Script ---------------------
def main():

    # ---------------- Paths ----------------
    test_img_dir = r"C:\Users\MSI INFINITE S3\Downloads\bangladesh_data\data"
    test_csv = r"C:\Users\MSI INFINITE S3\Downloads\bangladesh_data\labels.csv"

    #test_img_dir = r"C:\Users\MSI INFINITE S3\Downloads\brisc_2\classification_task\test\data"
    #test_csv = r"C:\Users\MSI INFINITE S3\Downloads\brisc_2\classification_task\test\labels.csv"


    model_path = "best_densenet201_brain_tumor.pth"
    output_conf_matrix = "confusion_matrix.png"

    num_classes = 4

    # ---------------- Device ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # ---------------- Transforms ----------------
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ---------------- DataLoader ----------------
    test_dataset = BrainTumorDataset(test_csv, test_img_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # ---------------- Load Model ----------------
    model = models.densenet201(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # ---------------- Inference ----------------
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    # ---------------- Metrics ----------------
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n===== Test Results =====")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # ---------------- Confusion Matrix ----------------
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=[f"Class {i}" for i in range(num_classes)],
                yticklabels=[f"Class {i}" for i in range(num_classes)])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_conf_matrix)
    plt.close()

    print(f"\n📁 Confusion matrix saved to: {output_conf_matrix}")


if __name__ == "__main__":
    main()
