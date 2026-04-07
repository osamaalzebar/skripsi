import os
import shutil
import csv

# ====== USER INPUT ======
base_dir = r"C:\path\to\your\dataset"  # <-- change this

classes = {
    "meningioma": 0,
    "glioma": 1,
    "pituitary": 2,
    "no_tumor": 3
}

# ====== CREATE OUTPUT ======
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)

csv_path = os.path.join(base_dir, "labels.csv")

# ====== OPEN CSV ======
with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["image", "label"])

    # ====== PROCESS EACH CLASS ======
    for class_name, label in classes.items():
        class_dir = os.path.join(base_dir, class_name)

        if not os.path.exists(class_dir):
            print(f"⚠️ Skipping missing folder: {class_name}")
            continue

        for img in os.listdir(class_dir):
            src = os.path.join(class_dir, img)

            # Avoid overwriting if same filename exists
            dst = os.path.join(data_dir, img)
            if os.path.exists(dst):
                name, ext = os.path.splitext(img)
                counter = 1
                while os.path.exists(dst):
                    new_name = f"{name}_{counter}{ext}"
                    dst = os.path.join(data_dir, new_name)
                    counter += 1
                img = os.path.basename(dst)

            # Move image
            shutil.move(src, dst)

            # Write to CSV
            writer.writerow([img, label])

print("✅ Done! All images moved and labels.csv created.")