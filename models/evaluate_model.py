import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data-pipeline"))

import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt

from preprocessing import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_densenet(weights_path):
    model = models.densenet121(weights=None)  # no need to redownload pretrained weights
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_predictions_and_probs(model, loader):
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # probability of PNEUMONIA class
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def compute_auc_roc(labels, probs, classes, save_path="roc_curve.png"):
    auc = roc_auc_score(labels, probs)
    fpr, tpr, _ = roc_curve(labels, probs)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {classes[1]} vs {classes[0]}")
    plt.legend()
    plt.savefig(save_path)
    print(f"ROC curve saved to {save_path}")
    plt.show()

    return auc

def sensitivity_specificity(labels, preds):
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)  # recall for positive class (PNEUMONIA)
    specificity = tn / (tn + fp)  # recall for negative class (NORMAL)
    return sensitivity, specificity, cm

if __name__ == "__main__":
    _, _, test_loader, classes = get_dataloaders()
    print("Classes:", classes)

    model = load_densenet("densenet_model.pt")

    labels, preds, probs = get_predictions_and_probs(model, test_loader)

    print("\n--- Classification Report ---")
    print(classification_report(labels, preds, target_names=classes))

    sensitivity, specificity, cm = sensitivity_specificity(labels, preds)
    print(f"Sensitivity (Recall for {classes[1]}): {sensitivity:.4f}")
    print(f"Specificity (Recall for {classes[0]}): {specificity:.4f}")
    print("Confusion Matrix:\n", cm)

    auc = compute_auc_roc(labels, probs, classes)
    print(f"\nAUC-ROC: {auc:.4f}")