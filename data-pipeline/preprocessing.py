import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Path to your dataset (adjust if your folder structure differs)
DATA_DIR = "chest_xray"

IMG_SIZE = 224
BATCH_SIZE = 32

# Transforms for TRAINING data — includes augmentation
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])  # ImageNet stats, needed for transfer learning later
])

# Transforms for TEST/VAL data — NO augmentation, just resize + normalize
eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

def get_dataloaders():
    train_data = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transforms)
    val_data   = datasets.ImageFolder(f"{DATA_DIR}/val", transform=eval_transforms)
    test_data  = datasets.ImageFolder(f"{DATA_DIR}/test", transform=eval_transforms)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    print("Classes:", train_data.classes)  # should print ['NORMAL', 'PNEUMONIA']
    print("Train size:", len(train_data))
    print("Val size:", len(val_data))
    print("Test size:", len(test_data))

    return train_loader, val_loader, test_loader, train_data.classes

def show_batch(loader, classes):
    images, labels = next(iter(loader))
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    for i in range(5):
        img = images[i].permute(1, 2, 0).numpy()
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # undo normalization for display
        img = img.clip(0, 1)
        axes[i].imshow(img)
        axes[i].set_title(classes[labels[i]])
        axes[i].axis("off")
    plt.show()

if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    show_batch(train_loader, classes)