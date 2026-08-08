import os
import shutil
import random

random.seed(42)

DATA_DIR = "chest_xray"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

classes = ["NORMAL", "PNEUMONIA"]
VAL_SPLIT = 0.15  # move 15% of train images into val

for cls in classes:
    train_class_dir = os.path.join(TRAIN_DIR, cls)
    val_class_dir = os.path.join(VAL_DIR, cls)
    os.makedirs(val_class_dir, exist_ok=True)

    images = os.listdir(train_class_dir)
    random.shuffle(images)

    n_val = int(len(images) * VAL_SPLIT)
    val_images = images[:n_val]

    for img in val_images:
        src = os.path.join(train_class_dir, img)
        dst = os.path.join(val_class_dir, img)
        shutil.move(src, dst)

    print(f"{cls}: moved {len(val_images)} images to val/")

print("Done. New split created.")