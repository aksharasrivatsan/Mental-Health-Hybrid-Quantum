from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vgg16, VGG16_Weights


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("mfcc_images")
MODEL_DIR = Path("models")

BATCH_SIZE = 16
NUM_WORKERS = 2
NUM_CLASSES = 8

NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "vgg16_best.pth"


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
# Load datasets
# ============================================================

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=transform,
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "validation",
    transform=transform,
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=transform,
)


# ============================================================
# Data loaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
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

weights = VGG16_Weights.DEFAULT

model = vgg16(weights=weights)

# Replace the original 1000-class ImageNet classifier
# with an 8-class emotion classifier.
model.classifier[6] = nn.Linear(
    model.classifier[6].in_features,
    NUM_CLASSES,
)

model = model.to(DEVICE)


# ============================================================
# Training configuration
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)


# ============================================================
# Training function
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# Validation function
# ============================================================

def evaluate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# Print configuration
# ============================================================

print("=" * 60)
print("VGG16 BASELINE TRAINING")
print("=" * 60)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"Training samples:   {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples:       {len(test_dataset)}")

print(f"Classes: {train_dataset.classes}")
print(f"Number of classes: {len(train_dataset.classes)}")

print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Weight decay: {WEIGHT_DECAY}")

print(f"Best model: {BEST_MODEL_PATH}")

print("=" * 60)


# ============================================================
# Main training loop
# ============================================================

best_val_loss = float("inf")

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
}


for epoch in range(NUM_EPOCHS):

    print()
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )
    print("-" * 60)

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
    )

    val_loss, val_accuracy = evaluate(
        model,
        val_loader,
        criterion,
    )

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    print(f"Train Loss:     {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Val Loss:       {val_loss:.4f}")
    print(f"Val Accuracy:   {val_accuracy * 100:.2f}%")
    print(f"Learning Rate:  {current_lr:.7f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "classes": train_dataset.classes,
            },
            BEST_MODEL_PATH,
        )

        print("✓ Best model saved")


# ============================================================
# Load best model
# ============================================================

print()
print("=" * 60)
print("LOADING BEST MODEL")
print("=" * 60)

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(checkpoint["model_state_dict"])

print(f"Best epoch: {checkpoint['epoch']}")
print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
print(
    f"Best validation accuracy: "
    f"{checkpoint['val_accuracy'] * 100:.2f}%"
)


# ============================================================
# Final test evaluation
# ============================================================

test_loss, test_accuracy = evaluate(
    model,
    test_loader,
    criterion,
)

print()
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("=" * 60)
print("VGG16 BASELINE TRAINING COMPLETE")
print("=" * 60)
