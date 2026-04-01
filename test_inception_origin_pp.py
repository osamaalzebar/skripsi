import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ===== Import your trained model =====
from train_inceptionV3_origin import InceptionV3ConcatHead

# ===== ImageNet normalization =====
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

NUM_CLASSES = 4
CLASS_NAMES = ["Class 0", "Class 1", "Class 2", "Class 3"]  # optional rename


# ================= Dataset =================
class TestInceptionV3(Dataset):
    """
    CSV must contain:
      - Image_path
      - label in {0,1,2,3}
    """

    def __init__(self, image_root, csv_path):
        self.samples = []

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            ip = reader.fieldnames[0].strip()
            lb = reader.fieldnames[1].strip()

            for row in reader:
                img_path = Path(image_root) / row[ip].strip()
                label = int(row[lb])  # already 0–3
                self.samples.append((img_path, label))

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, torch.tensor(label, dtype=torch.long)


# ================= Confusion Matrix Plot =================
def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="InceptionV3 Confusion Matrix"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Grid lines (cell borders)
    ax.set_xticks(np.arange(cm.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Cell values
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Confusion matrix saved to: {out_path}")


# ================= Evaluation =================
@torch.no_grad()
def evaluate_inceptionv3(
    image_root,
    csv_path,
    ckpt_path,
    batch_size=8,
    num_workers=4,
    device=None
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = TestInceptionV3(image_root, csv_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model = InceptionV3ConcatHead(num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    y_true, y_pred = [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = torch.argmax(F.softmax(logits, dim=1), dim=1)

        y_true.extend(y.numpy())
        y_pred.extend(preds.cpu().numpy())

    # ===== Metrics =====
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n📊 InceptionV3 Test Results")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    # ===== Confusion Matrix =====
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, CLASS_NAMES, "confusion_matrix_inceptionv3.png")


# ================= Main =================
if __name__ == "__main__":
    evaluate_inceptionv3(
        image_root= r"C:\Users\MSI INFINITE S3\Downloads\Bangladesh_data\data",
        csv_path= r"C:\Users\MSI INFINITE S3\Downloads\Bangladesh_data\labels.csv",
        ckpt_path=r"C:\Users\MSI INFINITE S3\PycharmProjects\origin\best_incv3_concat.pth",
        batch_size=8,
        num_workers=4
    )
