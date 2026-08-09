import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DEFAULT_DATA_DIR = Path("mfcc_images")


def add_common_training_args(parser, *, default_epochs, default_learning_rate, default_plot_path):
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=default_learning_rate)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--quantum-depth", type=int, default=2)
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--plot-path", type=Path, default=Path(default_plot_path))
    return parser


def load_image_datasets(data_dir, batch_size):
    train_dir = data_dir / "train"
    validation_dir = data_dir / "validation"

    if not train_dir.exists() or not validation_dir.exists():
        raise FileNotFoundError(
            f"Expected dataset folders at {train_dir} and {validation_dir}. "
            "Pass a different root with --data-dir."
        )

    import tensorflow as tf

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="int",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        validation_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="int",
    )

    return train_ds, val_ds


class QuantumCircuitLayer:
    """Keras layer wrapper for a small PennyLane variational circuit."""

    def __new__(cls, *args, **kwargs):
        import tensorflow as tf
        import pennylane as qml
        from tensorflow.keras import layers

        class _QuantumCircuitLayer(layers.Layer):
            def __init__(self, n_qubits=4, quantum_depth=2, **layer_kwargs):
                super().__init__(**layer_kwargs)
                self.n_qubits = n_qubits
                self.quantum_depth = quantum_depth
                self.device = qml.device("default.qubit", wires=n_qubits)

                @qml.qnode(self.device, interface="tf", diff_method="backprop")
                def circuit(inputs, weights):
                    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                    return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

                self.circuit = circuit

            def build(self, input_shape):
                self.quantum_weights = self.add_weight(
                    name="quantum_weights",
                    shape=(self.quantum_depth, self.n_qubits, 3),
                    initializer="random_uniform",
                    trainable=True,
                )

            def call(self, inputs):
                inputs = tf.cast(inputs, tf.float64)
                weights = tf.cast(self.quantum_weights.value, tf.float64)

                @tf.autograph.experimental.do_not_convert
                def run_circuit(sample):
                    values = self.circuit(sample, weights)
                    return tf.cast(tf.stack(values), tf.float32)

                return tf.map_fn(run_circuit, inputs, fn_output_signature=tf.float32)

            def compute_output_shape(self, input_shape):
                return (input_shape[0], self.n_qubits)

            def get_config(self):
                config = super().get_config()
                config.update({
                    "n_qubits": self.n_qubits,
                    "quantum_depth": self.quantum_depth,
                })
                return config

        return _QuantumCircuitLayer(*args, **kwargs)


def build_hybrid_vgg_quantum_model(
    *,
    num_classes,
    n_qubits,
    quantum_depth,
    fine_tune_block5,
    learning_rate,
    weights,
):
    import math
    import tensorflow as tf
    from tensorflow.keras import applications, layers, models

    if n_qubits < 1:
        raise ValueError("--n-qubits must be at least 1.")

    vgg_weights = None if weights == "none" else weights

    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="mfcc_image")
    x = layers.Lambda(applications.vgg16.preprocess_input, name="vgg16_preprocess")(inputs)

    base_model = applications.VGG16(
        weights=vgg_weights,
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base_model.trainable = fine_tune_block5

    for layer in base_model.layers:
        layer.trainable = fine_tune_block5 and "block5_" in layer.name

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="vgg16_feature_pool")(x)
    x = layers.Dense(128, activation="relu", name="mfcc_feature_projection")(x)
    x = layers.BatchNormalization(name="projection_norm")(x)
    x = layers.Dropout(0.3, name="projection_dropout")(x)

    # Compress the full MFCC/VGG feature vector into rotation angles for the qubits.
    x = layers.Dense(n_qubits, activation="tanh", name="qubit_angle_features")(x)
    x = layers.Lambda(lambda values: values * math.pi, name="scale_angles_to_radians")(x)
    x = QuantumCircuitLayer(
        n_qubits=n_qubits,
        quantum_depth=quantum_depth,
        name="mfcc_quantum_layer",
    )(x)

    x = layers.Dense(32, activation="relu", name="post_quantum_dense")(x)
    x = layers.Dropout(0.2, name="post_quantum_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_classifier")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="mfcc_vgg16_quantum_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        run_eagerly=True,
    )
    return model


def save_accuracy_plot(history, plot_path, title):
    import matplotlib.pyplot as plt

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.show()


def run_training(args, *, fine_tune_block5, title):
    try:
        train_ds, val_ds = load_image_datasets(args.data_dir, args.batch_size)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error

    model = build_hybrid_vgg_quantum_model(
        num_classes=len(train_ds.class_names),
        n_qubits=args.n_qubits,
        quantum_depth=args.quantum_depth,
        fine_tune_block5=fine_tune_block5,
        learning_rate=args.learning_rate,
        weights=args.weights,
    )

    model.summary()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
    model_path = args.plot_path.with_suffix(".h5")
    model.save(model_path)
    print(f"Saved trained model to: {model_path}")
    save_accuracy_plot(history, args.plot_path, title)
    return history
