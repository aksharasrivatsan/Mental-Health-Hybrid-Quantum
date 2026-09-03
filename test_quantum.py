import pennylane as qml

dev = qml.device("default.qubit", wires=2)


@qml.qnode(dev, interface="autograd")
def circuit(x):
    qml.RY(x[0], wires=0)
    qml.RY(x[1], wires=1)
    qml.CNOT(wires=[0, 1])

    return qml.expval(qml.PauliZ(0))


x = qml.numpy.array(
    [0.2, 0.4],
    requires_grad=True,
)

output = circuit(x)
gradient = qml.grad(circuit)(x)

print("PennyLane gradient test")
print("=======================")
print(f"Output: {output}")
print(f"Gradient: {gradient}")
print("Gradient calculation successful")
