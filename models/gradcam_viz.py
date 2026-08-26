import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data-pipeline"))

import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from preprocessing import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_densenet(weights_path):
    model = models.densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def unnormalize(img_tensor):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    return np.clip(img, 0, 1)

def generate_gradcam_grid(model, loader, classes, num_images=6, save_path="gradcam_results.png"):
    # DenseNet121's last conv block — the standard target layer for Grad-CAM on this architecture
    target_layers = [model.features.denseblock4.denselayer16.conv2]
    cam = GradCAM(model=model, target_layers=target_layers)

    images, labels = next(iter(loader))
    images, labels = images[:num_images], labels[:num_images]

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 3, 6))

    for i in range(num_images):
        input_tensor = images[i].unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred_class = output.argmax(dim=1).item()

        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

        rgb_img = unnormalize(images[i])
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        true_label = classes[labels[i]]
        pred_label = classes[pred_class]
        color = "green" if true_label == pred_label else "red"

        axes[0, i].imshow(rgb_img)
        axes[0, i].set_title(f"True: {true_label}", fontsize=10)
        axes[0, i].axis("off")

        axes[1, i].imshow(cam_image)
        axes[1, i].set_title(f"Pred: {pred_label}", fontsize=10, color=color)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Grad-CAM grid saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    _, _, test_loader, classes = get_dataloaders()
    model = load_densenet("densenet_model.pt")
    generate_gradcam_grid(model, test_loader, classes)