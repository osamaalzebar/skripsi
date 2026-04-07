import os
import shutil
import random
import csv

# ====== USER INPUT ======
base_dir = r"C:\path\to\your\dataset"  # <-- change this

classes = {
    "meningioma": 0,
    "glioma": 1,
    "pituitary": 2,
    "no_tumor": 3
}

VAL_PER_CLASS = 95

# ====== CREATE OUTPUT STRUCTURE ======
train_data_dir = os.path.join(base_dir, "train", "data")
val_data_dir = os.path.join(base_dir, "validate", "data")

os.makedirs(train_data_dir, exist_ok=True)
os.makedirs(val_data_dir, exist_ok=True)

train_csv_path = os.path.join(base_dir, "train", "labels.csv")
val_csv_path = os.path.join(base_dir, "validate", "labels.csv")

# ====== OPEN CSV FILES ======
train_csv = open(train_csv_path, mode='w', newline='')
val_csv = open(val_csv_path, mode='w', newline='')

train_writer = csv.writer(train_csv)
val_writer = csv.writer(val_csv)

# Write headers
train_writer.writerow(["image", "label"])
val_writer.writerow(["image", "label"])

# ====== PROCESS EACH CLASS ======
for class_name, label in classes.items():
    class_dir = os.path.join(base_dir, class_name)

    images = os.listdir(class_dir)
    random.shuffle(images)

    val_images = images[:VAL_PER_CLASS]
    train_images = images[VAL_PER_CLASS:]

    # ---- VALIDATION SET ----
    for img in val_images:
        src = os.path.join(class_dir, img)
        dst = os.path.join(val_data_dir, img)

        shutil.move(src, dst)
        val_writer.writerow([img, label])

    # ---- TRAIN SET ----
    for img in train_images:
        src = os.path.join(class_dir, img)
        dst = os.path.join(train_data_dir, img)

        shutil.move(src, dst)
        train_writer.writerow([img, label])

# ====== CLOSE FILES ======
train_csv.close()
val_csv.close()

print("✅ Done! Dataset split into train and validate successfully.")