'''import argparse
import os
from pathlib import Path
import pandas as pd
import tensorflow as tf

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
    test_dir = data_dir/"test"

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

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="int",
    )

    return train_ds, val_ds, test_ds

class QuantumCircuitLayer:
    """Keras layer wrapper for a small PennyLane variational circuit."""

    def __new__(cls, *args, **kwargs):
        import tensorflow as tf
        import pennylane as qml
        keras=tf.keras
        from keras import layers

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
                weights = tf.cast(self.quantum_weights, tf.float64)

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
    keras=tf.keras
    from keras import applications, layers, models

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
        train_ds, val_ds,test_ds = load_image_datasets(args.data_dir, args.batch_size)
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
    import tensorflow as tf

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.2,patience=2,min_lr=1e-7,verbose=1)

    checkpoint = tf.keras.callbacks.ModelCheckpoint("best_finetuned_quantum.keras",monitor="val_accuracy",save_best_only=True,verbose=1)

    history = model.fit(train_ds,validation_data=val_ds,epochs=args.epochs,callbacks=[early_stop,reduce_lr,checkpoint])

    save_accuracy_plot(history, args.plot_path, title)

    metrics=evaluate_model(model,test_ds,test_ds.class_names,)
    

    save_results(accuracy=metrics["accuracy"],precision=metrics["precision"],recall=metrics["recall"],
                 f1=metrics["f1"],epochs=args.epochs,learning_rate=args.learning_rate,n_qubits=args.n_qubits,
                quantum_depth=args.quantum_depth)

    for layer in model.layers:
        print(layer.name, layer.output.shape)

    return history

def evaluate_model(model, test_ds, class_names):
    """
    Evaluate the trained model on the test dataset.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix,
        ConfusionMatrixDisplay,
    )

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(
        test_ds,
        verbose=1,
    )

    print(f"\nTest Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f}")

    y_true = []
    y_pred = []

    for images, labels in test_ds:

        predictions = model.predict(images, verbose=0)

        predicted_labels = np.argmax(predictions, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted_labels)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    print("\nOverall Metrics")
    print("-" * 40)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    print("\nClassification Report")
    print("-" * 40)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    disp.plot(
        cmap="Blues",
        values_format="d",
        ax=plt.gca(),
        colorbar=False,
    )

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()

    print("\nConfusion matrix saved as confusion_matrix.png")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def save_results(
    accuracy,
    precision,
    recall,
    f1,
    epochs,
    learning_rate,
    n_qubits,
    quantum_depth,
    csv_path="experiment_results.csv",
):

    result = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Epochs": epochs,
        "Learning Rate": learning_rate,
        "Qubits": n_qubits,
        "Quantum Depth": quantum_depth,
    }

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    else:
        df = pd.DataFrame([result])

    df.to_csv(csv_path, index=False)

    print(f"\nResults saved to {csv_path}")'''


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]        = "3"   
os.environ["TF_ENABLE_ONEDNN_OPTS"]       = "0"   
os.environ["PYTHONWARNINGS"]              = "ignore"

# Must be set BEFORE importing tensorflow
import tensorflow as tf
tf.get_logger().setLevel("ERROR")                  
tf.autograph.set_verbosity(0)                      

import warnings
warnings.filterwarnings("ignore")                  

import argparse
from pathlib import Path
import pandas as pd
import tensorflow as tf

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
DEFAULT_DATA_DIR = Path("mfcc_images")


def add_common_training_args(parser, *, default_epochs, default_learning_rate, default_plot_path):
    parser.add_argument("--data-dir",       type=Path,  default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs",         type=int,   default=default_epochs)
    parser.add_argument("--batch-size",     type=int,   default=BATCH_SIZE)
    parser.add_argument("--learning-rate",  type=float, default=default_learning_rate)
    parser.add_argument("--n-qubits",       type=int,   default=4)
    parser.add_argument("--quantum-depth",  type=int,   default=2)
    parser.add_argument("--weights",        choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--plot-path",      type=Path,  default=Path(default_plot_path))
    return parser


def load_image_datasets(data_dir, batch_size):
    train_dir = data_dir / "train"
    val_dir   = data_dir / "validation"
    test_dir  = data_dir / "test"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected dataset folders at {train_dir} and {val_dir}. "
            "Pass a different root with --data-dir."
        )

    normalization = tf.keras.layers.Rescaling(1.0 / 255)

    def prepare(ds):
        return ds.map(
            lambda x, y: (normalization(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)

    # ── Load raw datasets first to capture class_names ──────────
    raw_train = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMG_SIZE, batch_size=batch_size, label_mode="int"
    )
    raw_val = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=IMG_SIZE, batch_size=batch_size, label_mode="int"
    )
    raw_test = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=IMG_SIZE, batch_size=batch_size, label_mode="int"
    )

    # Save class_names before prefetch loses them
    class_names = raw_train.class_names

    train_ds = prepare(raw_train)
    val_ds   = prepare(raw_val)
    test_ds  = prepare(raw_test)

    # Attach class_names back to each dataset manually
    train_ds.class_names = class_names
    val_ds.class_names   = class_names
    test_ds.class_names  = class_names

    return train_ds, val_ds, test_ds


class QuantumCircuitLayer:
    def __new__(cls, *args, **kwargs):
        import pennylane as qml
        from keras import layers

        class _QuantumCircuitLayer(layers.Layer):
            def __init__(self, n_qubits=4, quantum_depth=2, **layer_kwargs):
                super().__init__(**layer_kwargs)
                self.n_qubits      = n_qubits
                self.quantum_depth = quantum_depth

                # Use lightning.qubit or default.qubit with tf interface
                self.device = qml.device("default.qubit", wires=n_qubits)

                @qml.qnode(
                    self.device,
                    interface="tf",
                    diff_method="backprop",
                )
                def circuit(inputs, weights):
                    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                    return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

                self.circuit = circuit

            def build(self, input_shape):
                self.quantum_weights = self.add_weight(
                    name="quantum_weights",
                    shape=(self.quantum_depth, self.n_qubits, 3),
                    initializer="random_uniform",
                    trainable=True,
                    dtype=tf.float32,
                )
            def call(self, inputs):
                inputs_64  = tf.cast(inputs,               tf.float64)
                weights_64 = tf.cast(self.quantum_weights, tf.float64)

                @tf.autograph.experimental.do_not_convert
                def run_circuit(sample):
                    results = self.circuit(sample, weights_64)
                    # Results are already float64 — just stack and cast to float32
                    return tf.cast(tf.stack(results), tf.float32)

                return tf.map_fn(run_circuit,inputs_64,fn_output_signature=tf.float32,)

            

            def compute_output_shape(self, input_shape):
                return (input_shape[0], self.n_qubits)

            def get_config(self):
                cfg = super().get_config()
                cfg.update({
                    "n_qubits":      self.n_qubits,
                    "quantum_depth": self.quantum_depth,
                })
                return cfg

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
    from keras import applications, layers, models

    if n_qubits < 1:
        raise ValueError("--n-qubits must be at least 1.")

    vgg_weights = None if weights == "none" else weights

    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="mfcc_image")

    # VGG16 expects pixel values in [0,255] — rescale back from [0,1]
    x = layers.Rescaling(255.0, name="rescale_for_vgg")(inputs)
    x = layers.Lambda(
        applications.vgg16.preprocess_input,
        name="vgg16_preprocess",
    )(x)

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

    x = layers.Dense(128, activation="relu",  name="mfcc_feature_projection")(x)
    x = layers.BatchNormalization(name="projection_norm")(x)
    x = layers.Dropout(0.3, name="projection_dropout")(x)

    # Compress to n_qubits rotation angles in [-π, π]
    x = layers.Dense(n_qubits, activation="tanh", name="qubit_angle_features")(x)
    x = layers.Lambda(
        lambda v: v * math.pi,
        name="scale_angles_to_radians",
    )(x)

    # ── Quantum layer ────────────────────────────────────────────────────
    # Explicitly cast to float32 before entering the layer so Keras graph
    # dtype stays consistent; the layer itself casts to float64 internally.
    x = layers.Lambda(
        lambda v: tf.cast(v, tf.float32),
        name="ensure_float32_before_quantum",
    )(x)
    x = QuantumCircuitLayer(
        n_qubits=n_qubits,
        quantum_depth=quantum_depth,
        name="mfcc_quantum_layer",
    )(x)
    # Cast output back to float32 explicitly
    x = layers.Lambda(
        lambda v: tf.cast(v, tf.float32),
        name="ensure_float32_after_quantum",
    )(x)

    x       = layers.Dense(32, activation="relu", name="post_quantum_dense")(x)
    x       = layers.Dropout(0.2, name="post_quantum_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_classifier")(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="mfcc_vgg16_quantum_classifier",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        run_eagerly=True,           # required for PennyLane + TF
    )
    return model


def save_accuracy_plot(history, plot_path, title):
    import matplotlib.pyplot as plt

    acc     = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(acc)),     acc,     label="Training Accuracy")
    plt.plot(range(len(val_acc)), val_acc, label="Validation Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.show()


def run_training(args, *, fine_tune_block5, title):
    try:
        train_ds, val_ds, test_ds = load_image_datasets(args.data_dir, args.batch_size)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error

    model = build_hybrid_vgg_quantum_model(
        num_classes      = len(train_ds.class_names),
        n_qubits         = args.n_qubits,
        quantum_depth    = args.quantum_depth,
        fine_tune_block5 = fine_tune_block5,
        learning_rate    = args.learning_rate,
        weights          = args.weights,
    )
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7, verbose=1
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "best_finetuned_quantum.keras",
        monitor="val_accuracy", save_best_only=True, verbose=1,
    )

    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = args.epochs,
        callbacks       = [early_stop, reduce_lr, checkpoint],
    )

    save_accuracy_plot(history, args.plot_path, title)

    metrics = evaluate_model(model, test_ds, test_ds.class_names)

    save_results(
        accuracy      = metrics["accuracy"],
        precision     = metrics["precision"],
        recall        = metrics["recall"],
        f1            = metrics["f1"],
        epochs        = args.epochs,
        learning_rate = args.learning_rate,
        n_qubits      = args.n_qubits,
        quantum_depth = args.quantum_depth,
    )

    for layer in model.layers:
        print(layer.name, layer.output.shape)

    return history


def evaluate_model(model, test_ds, class_names):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, classification_report,
        confusion_matrix, ConfusionMatrixDisplay,
    )

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f}")

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall    = recall_score(   y_true, y_pred, average="macro", zero_division=0)
    f1        = f1_score(       y_true, y_pred, average="macro", zero_division=0)

    print("\nOverall Metrics")
    print("-" * 40)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    print("\nClassification Report")
    print("-" * 40)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))

    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
        cmap="Blues", values_format="d", ax=ax, colorbar=False
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()
    print("\nConfusion matrix saved as confusion_matrix.png")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def save_results(
    accuracy, precision, recall, f1,
    epochs, learning_rate, n_qubits, quantum_depth,
    csv_path="experiment_results.csv",
):
    result = {
        "Accuracy":      accuracy,
        "Precision":     precision,
        "Recall":        recall,
        "F1-Score":      f1,
        "Epochs":        epochs,
        "Learning Rate": learning_rate,
        "Qubits":        n_qubits,
        "Quantum Depth": quantum_depth,
    }

    if os.path.exists(csv_path):
        df = pd.concat([pd.read_csv(csv_path), pd.DataFrame([result])], ignore_index=True)
    else:
        df = pd.DataFrame([result])

    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")