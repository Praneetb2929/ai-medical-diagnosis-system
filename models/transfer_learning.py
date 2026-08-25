import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data-pipeline"))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def build_model():
    # Load pretrained DenseNet121
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    # Freeze early layers (keep pretrained features, don't retrain from scratch)
    for param in model.parameters():
        param.requires_grad = False

    # Replace final classifier layer for our 2-class problem
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)

    return model.to(device)

def train_model(model, train_loader, val_loader, class_weights, epochs=5):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))  # handles class imbalance
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)  # only train the new layer

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

    return model

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

def full_report(model, test_loader, classes):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n--- Test Set Results ---")
    print(classification_report(all_labels, all_preds, target_names=classes))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()

    # Compute class weights to fix the imbalance problem from baseline
    import numpy as np
    from collections import Counter
    train_labels = [label for _, label in train_loader.dataset.samples]
    counts = Counter(train_labels)
    total = sum(counts.values())
    class_weights = torch.tensor([total / counts[0], total / counts[1]], dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()
    print("Class weights (NORMAL, PNEUMONIA):", class_weights)

    model = build_model()
    model = train_model(model, train_loader, val_loader, class_weights, epochs=5)

    full_report(model, test_loader, classes)

    torch.save(model.state_dict(), "densenet_model.pt")
    print("\nModel saved as densenet_model.pt")