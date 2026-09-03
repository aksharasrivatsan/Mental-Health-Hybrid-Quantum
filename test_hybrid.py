import torch
import pennylane as qml
from torch import nn


# ============================================================
# Simple quantum layer
# ============================================================

N_QUBITS = 4

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS,
)


@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):

    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

    qml.StronglyEntanglingLayers(
        weights,
        wires=range(N_QUBITS),
    )

    return [
        qml.expval(qml.PauliZ(i))
        for i in range(N_QUBITS)
    ]


# ============================================================
# Quantum layer configuration
# ============================================================

weight_shapes = {
    "weights": (2, N_QUBITS, 3),
}

quantum_layer = qml.qnn.TorchLayer(
    quantum_circuit,
    weight_shapes,
)


# ============================================================
# Test hybrid model
# ============================================================

model = nn.Sequential(
    nn.Linear(N_QUBITS, N_QUBITS),
    quantum_layer,
    nn.Linear(N_QUBITS, 8),
)


# ============================================================
# Test input
# ============================================================

x = torch.randn(
    4,
    N_QUBITS,
    requires_grad=True,
)

labels = torch.tensor(
    [0, 1, 2, 3],
)


# ============================================================
# Forward pass
# ============================================================

outputs = model(x)

print("Hybrid PyTorch + PennyLane test")
print("================================")
print(f"Input shape:  {x.shape}")
print(f"Output shape: {outputs.shape}")


# ============================================================
# Backward pass
# ============================================================

criterion = nn.CrossEntropyLoss()

loss = criterion(
    outputs,
    labels,
)

loss.backward()

print(f"Loss:         {loss.item():.6f}")
print("Backward pass successful")

print()
print("Hybrid quantum-classical layer is working!")
