from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras import applications, layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


DATA_DIR = Path("mfcc_images")
MODEL_DIR = Path("models")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

NUM_EPOCHS = 30
LEARNING_RATE = 0.00001

MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "vgg16_fine_tuned_best.keras"

ACCURACY_PLOT_PATH = Path("vgg16_fine_tuned_accuracy.png")
LOSS_PLOT_PATH = Path("vgg16_fine_tuned_loss.png")
CONFUSION_MATRIX_PATH = Path("vgg16_fine_tuned_confusion_matrix.png")
FINE_TUNED_BASE_WEIGHTS_PATH = (
    MODEL_DIR / "vgg16_block5_fine_tuned.weights.h5"
)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "validation",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)
class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

print(f"Number of classes: {num_classes}")
base_model = applications.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(*IMG_SIZE, 3),
    name="vgg16_fine_tuned_base",
)
base_model.trainable = False
for layer in base_model.layers:
    if layer.name.startswith("block5_"):
        layer.trainable = True
inputs = layers.Input(
    shape=(*IMG_SIZE, 3),
    name="mfcc_image",
)

x = layers.Lambda(
    applications.vgg16.preprocess_input,
    name="vgg16_preprocess",
)(inputs)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D(
    name="vgg16_feature_pool"
)(x)

x = layers.Dense(
    128,
    activation="relu",
    name="vgg16_feature_projection",
)(x)

x = layers.Dropout(
    0.3,
    name="vgg16_dropout",
)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax",
    name="emotion_classifier",
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="vgg16_fine_tuned_classifier",
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=NUM_EPOCHS,
    callbacks=[checkpoint],
)
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy",
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy",
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("VGG16 Fine-Tuning Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    ACCURACY_PLOT_PATH,
    bbox_inches="tight",
)

plt.close()
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss",
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VGG16 Fine-Tuning Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    LOSS_PLOT_PATH,
    bbox_inches="tight",
)

plt.close()
model = tf.keras.models.load_model(
    BEST_MODEL_PATH,
)
test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1,
)

print()
print("=" * 60)
print("VGG16 FINAL TEST RESULTS")
print("=" * 60)
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print("=" * 60)
y_true = []
y_pred = []
for images, labels in test_ds:
    predictions = model.predict(
        images,
        verbose=0,
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1,
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)
cm = confusion_matrix(
    y_true,
    y_pred,
)
plt.figure(figsize=(10, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

disp.plot(
    cmap="Blues",
    xticks_rotation=45,
)

plt.title(
    "VGG16 Fine-Tuned Test Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()
model.get_layer(

    "vgg16_fine_tuned_base"

).save_weights(

    FINE_TUNED_BASE_WEIGHTS_PATH
)
print(f"Fine-tuned VGG16 weights: {FINE_TUNED_BASE_WEIGHTS_PATH}")

print()
print("=" * 60)
print("VGG16 FINE-TUNING COMPLETE")
print("=" * 60)

print(f"Best model: {BEST_MODEL_PATH}")
print(f"Accuracy plot: {ACCURACY_PLOT_PATH}")
print(f"Loss plot: {LOSS_PLOT_PATH}")
print(f"Confusion matrix: {CONFUSION_MATRIX_PATH}")

print()
print(
    f"Final VGG16 test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print("=" * 60)


