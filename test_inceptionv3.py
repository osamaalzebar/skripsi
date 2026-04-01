import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt


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


# --- Confusion Matrix Plot ---
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=1,
        linecolor="black"
    )
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


    model_path = "best_inceptionv3_brain_tumor.pth"

    class_names = ["Class 0", "Class 1", "Class 2", "Class 3"]  # Replace with your real class names if available
    num_classes = 4

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- Test Transform ---
    test_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # --- Dataset & DataLoader ---
    test_dataset = BrainTumorDataset(test_csv, test_img_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    # --- Load Model ---
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)

    # Enable aux logits so the architecture matches the saved model
    model.aux_logits = True

    # Replace main FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Replace auxiliary FC layer
    if model.aux_logits:
        in_features_aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(in_features_aux, num_classes)

    # Now load weights successfully
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Disable aux output for inference
    model.aux_logits = False

    model.to(device)
    model.eval()

    print("Model loaded successfully!")

    all_labels = []
    all_preds = []

    # --- Inference Loop ---
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            # Inception returns only main output during eval
            if isinstance(outputs, tuple):
                outputs = outputs[0]

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
