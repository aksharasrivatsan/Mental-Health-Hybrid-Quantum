from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vgg16, VGG16_Weights

import pennylane as qml


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("mfcc_images")
MODEL_DIR = Path("models")

BATCH_SIZE = 8
NUM_WORKERS = 2
NUM_CLASSES = 8

N_QUBITS = 4
N_Q_LAYERS = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)


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
# Datasets
# ============================================================

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=transform,
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "validation",
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


# ============================================================
# VGG16 feature extractor
# ============================================================

weights = VGG16_Weights.DEFAULT

vgg = vgg16(weights=weights)

feature_extractor = vgg.features

for parameter in feature_extractor.parameters():
    parameter.requires_grad = False

feature_extractor = feature_extractor.to(DEVICE)
feature_extractor.eval()


# ============================================================
# Quantum circuit
# ============================================================

quantum_device = qml.device(
    "default.qubit",
    wires=N_QUBITS,
)


@qml.qnode(
    quantum_device,
    interface="torch",
    diff_method="backprop",
)
def quantum_circuit(inputs, weights):

    for qubit in range(N_QUBITS):
        qml.RY(
            inputs[qubit],
            wires=qubit,
        )

    qml.StronglyEntanglingLayers(
        weights,
        wires=range(N_QUBITS),
    )

    return [
        qml.expval(qml.PauliZ(qubit))
        for qubit in range(N_QUBITS)
    ]


weight_shapes = {
    "weights": (
        N_Q_LAYERS,
        N_QUBITS,
        3,
    ),
}


quantum_layer = qml.qnn.TorchLayer(
    quantum_circuit,
    weight_shapes,
)


# ============================================================
# Hybrid VGG16 + Quantum Model
# ============================================================

class HybridVGGQuantum(nn.Module):

    def __init__(self):
        super().__init__()

        # Frozen VGG16 convolutional feature extractor
        self.feature_extractor = feature_extractor

        # Convert VGG feature maps to 512 values
        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        # Classical dimensionality reduction:
        # 512 -> 64 -> 4 qubits
        self.classical_reduction = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, N_QUBITS),
            nn.Tanh(),
        )

        # Quantum layer
        self.quantum_layer = quantum_layer

        # Quantum outputs -> 8 emotion classes
        self.classifier = nn.Linear(
            N_QUBITS,
            NUM_CLASSES,
        )

    def forward(self, x):

        # ----------------------------------------------------
        # VGG16 feature extraction
        # ----------------------------------------------------

        # VGG16 feature extractor is frozen.
        with torch.no_grad():
            x = self.feature_extractor(x)

        # ----------------------------------------------------
        # Global average pooling
        # ----------------------------------------------------

        x = self.pool(x)

        # ----------------------------------------------------
        # Flatten
        # ----------------------------------------------------

        x = torch.flatten(
            x,
            start_dim=1,
        )

        # ----------------------------------------------------
        # Classical dimensionality reduction
        # ----------------------------------------------------

        x = self.classical_reduction(x)

        # ----------------------------------------------------
        # Convert values from [-1, 1] to approximately
        # [-pi, pi] for quantum angle encoding
        # ----------------------------------------------------

        x = x * torch.pi

        # ----------------------------------------------------
        # Quantum layer
        #
        # PennyLane default.qubit is a CPU simulator.
        # The current QNode is configured for one 4-value
        # input, so process each sample separately.
        # ----------------------------------------------------

        quantum_outputs = []

        for sample in x:

            sample_cpu = sample.cpu()

            quantum_output = self.quantum_layer(
                sample_cpu
            )

            quantum_outputs.append(
                quantum_output
            )

        # Combine individual quantum outputs into a batch
        x = torch.stack(
            quantum_outputs
        )

        # Move quantum outputs back to GPU
        x = x.to(DEVICE)

        # ----------------------------------------------------
        # Final classifier
        # ----------------------------------------------------

        x = self.classifier(x)

        return x


# ============================================================
# Create model
# ============================================================

model = HybridVGGQuantum().to(DEVICE)


# ============================================================
# Loss and optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(
        lambda parameter: parameter.requires_grad,
        model.parameters(),
    ),
    lr=0.001,
    weight_decay=0.0001,
)


# ============================================================
# Configuration information
# ============================================================

print("=" * 70)
print("HYBRID VGG16 + QUANTUM MODEL")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

print(f"Training samples:   {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

print(
    f"Classes: "
    f"{train_dataset.classes}"
)

print(
    f"Number of classes: "
    f"{len(train_dataset.classes)}"
)

print(f"Batch size: {BATCH_SIZE}")
print(f"Qubits: {N_QUBITS}")
print(f"Quantum layers: {N_Q_LAYERS}")

print("=" * 70)


# ============================================================
# Single-batch architecture test
# ============================================================

print()
print("Running single-batch architecture test...")

images, labels = next(
    iter(train_loader)
)

images = images.to(
    DEVICE,
    non_blocking=True,
)

labels = labels.to(
    DEVICE,
    non_blocking=True,
)


# Forward pass
outputs = model(images)


# Loss
loss = criterion(
    outputs,
    labels,
)


# Backward pass
loss.backward()


# ============================================================
# Test results
# ============================================================

print()
print(f"Input shape:  {images.shape}")
print(f"Output shape: {outputs.shape}")
print(f"Loss:         {loss.item():.6f}")

print()
print("Forward pass: successful")
print("Backward pass: successful")

print()
print("Hybrid architecture test complete.")
print("=" * 70)
