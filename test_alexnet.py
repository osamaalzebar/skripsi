import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from torchvision import models, transforms
import torch.nn as nn


# ------------------------------------------------------------
# Dataset Class (same as your training code)
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Evaluation Script
# ------------------------------------------------------------
def main():

    # --- UPDATE THESE PATHS ---
    #test_img_dir = r"C:\Users\MSI INFINITE S3\Downloads\brisc_2\classification_task\test\data"
    #test_csv = r"C:\Users\MSI INFINITE S3\Downloads\brisc_2\classification_task\test\labels.csv"

    test_img_dir = r"C:\Users\MSI INFINITE S3\Downloads\bangladesh_data\data"
    test_csv = r"C:\Users\MSI INFINITE S3\Downloads\bangladesh_data\labels.csv"

    model_path = "best_alexnet_brain_tumor.pth"
    save_confusion_matrix_path = "alexnet_confusion_matrix.png"

    num_classes = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- Same transforms as training/validation ---
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # --- Load Dataset ---
    test_dataset = BrainTumorDataset(test_csv, test_img_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    # --- Load AlexNet model ---
    model = models.alexnet(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    print("\n========== TEST RESULTS ==========")
    print(f"Accuracy:   {acc:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")

    # --------------------------------------------------------
    # Confusion Matrix Plot
    # --------------------------------------------------------
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        square=True,
        linewidths=0.7,
        linecolor="black",
        cbar=False
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("AlexNet Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_confusion_matrix_path, dpi=300)
    plt.close()

    print(f"\n📊 Confusion matrix saved to: {save_confusion_matrix_path}")


if __name__ == "__main__":
    main()
