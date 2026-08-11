import tensorflow as tf
from hybrid_quantum_tf import QuantumCircuitLayer

model = tf.keras.models.load_model(
    "vgg16_quantum_accuracy.h5",
    custom_objects={"_QuantumCircuitLayer": QuantumCircuitLayer}
)
model.summary()
print("Model loaded successfully.")
