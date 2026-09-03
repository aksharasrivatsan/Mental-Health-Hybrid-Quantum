from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vgg16, VGG16_Weights

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("mfcc_images")
MODEL_PATH = Path("models/vgg16_best.pth")

BATCH_SIZE = 16
NUM_WORKERS = 2
NUM_CLASSES = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Image transformations
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# Load test dataset
# ============================================================

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


# ============================================================
# Load VGG16
# ============================================================

model = vgg16(weights=None)

model.classifier[6] = nn.Linear(
    model.classifier[6].in_features,
    NUM_CLASSES,
)

model = model.to(DEVICE)


# ============================================================
# Load trained checkpoint
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()


# ============================================================
# Generate predictions
# ============================================================

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE, non_blocking=True)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())


# ============================================================
# Classification report
# ============================================================

print()
print("=" * 70)
print("VGG16 CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes,
        digits=4,
    )
)


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
)

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# Save confusion matrix figure
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=test_dataset.classes,
)

fig, ax = plt.subplots(figsize=(10, 8))

disp.plot(
    ax=ax,
    xticks_rotation=45,
)

plt.title("VGG16 Confusion Matrix")
plt.tight_layout()

output_path = Path("models/vgg16_confusion_matrix.png")

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Final accuracy
# ============================================================

correct = sum(
    prediction == label
    for prediction, label in zip(
        all_predictions,
        all_labels,
    )
)

accuracy = correct / len(all_labels)

print()
print("=" * 70)
print("FINAL VGG16 EVALUATION")
print("=" * 70)

print(f"Test samples: {len(all_labels)}")
print(f"Correct predictions: {correct}")
print(f"Test accuracy: {accuracy * 100:.2f}%")
print(f"Confusion matrix saved to: {output_path}")

print("=" * 70)
