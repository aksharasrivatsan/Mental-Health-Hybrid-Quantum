import tensorflow as tf
from tensorflow.keras import layers, models, applications
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "/content/mfcc_images"

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR + "/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR + "/validation",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True

for layer in base_model.layers:
    if 'block5_' in layer.name:
        layer.trainable = True
    else:
        layer.trainable = Falseṣ

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(6, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training the Purely Classical Fine-Tuned VGG16 (Block 5 Awake)...")
history = model.fit(train_ds, validation_data=val_ds, epochs=30)

%matplotlib inline
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_range = range(len(acc))

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, acc, label='Fine-Tuned Training Accuracy')
plt.plot(epochs_range, val_acc, label='Fine-Tuned Validation Accuracy')
plt.title('VGG16 (Block 5 Fine-Tuning) Classical Results')
plt.legend()
plt.grid(True)
plt.show()
