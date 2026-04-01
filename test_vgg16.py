import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


# --- Dataset Class (same as training) ---
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


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=1, linecolor='black')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def main():

    # --- Paths ---
    test_img_dir = r"C:\Users\MSI INFINITE S3\Downloads\brisc_2\classification_task\test\data"
    test_csv = r"C:\Users\MSI INFINITE S3\Downloads\brisc_2\classification_task\test\labels.csv"

    #test_img_dir = r"C:\Users\MSI INFINITE S3\Downloads\bangladesh_data\data"
    #test_csv = r"C:\Users\MSI INFINITE S3\Downloads\bangladesh_data\labels.csv"


    model_path = r"C:\Users\MSI INFINITE S3\PycharmProjects\clean\best_vgg16_brain_tumor.pth"

    class_names = ["Class 0", "Class 1", "Class 2", "Class 3"]  # <-- replace with actual names if available
    num_classes = 4

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- Test Transform ---
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # --- Dataset & Loader ---
    test_dataset = BrainTumorDataset(test_csv, test_img_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    # --- Load Model ---
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("Model loaded successfully!")

    all_labels = []
    all_preds = []

    # --- Testing Loop ---
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # --- Metrics ---
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print("\n===== TEST METRICS =====")
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:\n", cm)

    plot_confusion_matrix(cm, class_names)


if __name__ == "__main__":
    main()
