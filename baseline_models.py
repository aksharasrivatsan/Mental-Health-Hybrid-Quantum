import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import applications
import pickle

IMG_SIZE = (224, 224)

def extract_vgg16_features(data_dir, batch_size=32):
    """Extract VGG16 features from MFCC images (no quantum layer)"""
    from hybrid_quantum_tf import load_image_datasets
    
    train_ds, val_ds = load_image_datasets(data_dir, batch_size)
    
    # Load pre-trained VGG16 (just feature extractor)
    vgg_model = applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    vgg_model.trainable = False
    
    # Extract features and labels
    train_features, train_labels = [], []
    val_features, val_labels = [], []
    
    print("Extracting VGG16 features from training data...")
    for images, labels in train_ds:
        features = vgg_model.predict(images, verbose=0)
        features = features.reshape(features.shape[0], -1)  # Flatten
        train_features.append(features)
        train_labels.append(labels.numpy())
    
    print("Extracting VGG16 features from validation data...")
    for images, labels in val_ds:
        features = vgg_model.predict(images, verbose=0)
        features = features.reshape(features.shape[0], -1)
        val_features.append(features)
        val_labels.append(labels.numpy())
    
    X_train = np.vstack(train_features)
    y_train = np.hstack(train_labels)
    X_val = np.vstack(val_features)
    y_val = np.hstack(val_labels)
    
    return X_train, y_train, X_val, y_val


def train_pure_vgg16(data_dir, epochs=30):
    """Train pure VGG16 (without quantum layer)"""
    from hybrid_quantum_tf import load_image_datasets
    
    train_ds, val_ds = load_image_datasets(data_dir, batch_size=32)
    num_classes = len(train_ds.class_names)
    
    # Build pure VGG16
    base_model = applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False
    
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Training Pure VGG16...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)
    return model, history


def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest on VGG16 features"""
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=1)
    rf.fit(X_train, y_train)
    
    train_acc = rf.score(X_train, y_train)
    val_acc = rf.score(X_val, y_val)
    print(f"RF - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    return rf


def train_svm(X_train, y_train, X_val, y_val):
    """Train SVM on VGG16 features"""
    print("Training SVM...")
    
    # Standardize features (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    svm = SVC(kernel='rbf', C=10, verbose=1)
    svm.fit(X_train_scaled, y_train)
    
    train_acc = svm.score(X_train_scaled, y_train)
    val_acc = svm.score(X_val_scaled, y_val)
    print(f"SVM - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    return svm, scaler


def save_baseline_models(vgg16_model, rf_model, svm_model, scaler, save_dir="baseline_models"):
    """Save trained baseline models"""
    Path(save_dir).mkdir(exist_ok=True)
    
    vgg16_model.save(f"{save_dir}/vgg16_pure.h5")
    with open(f"{save_dir}/random_forest.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    with open(f"{save_dir}/svm.pkl", "wb") as f:
        pickle.dump(svm_model, f)
    with open(f"{save_dir}/svm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"Baseline models saved to {save_dir}/")


if __name__ == "__main__":
    # USAGE:
    # python baseline_models.py
    
    # Extract VGG16 features
    X_train, y_train, X_val, y_val = extract_vgg16_features(
        data_dir=Path("mfcc_images")
    )
    
    # Train pure VGG16
    vgg16_model, _ = train_pure_vgg16(data_dir=Path("mfcc_images"))
    
    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train, X_val, y_val)
    
    # Train SVM
    svm_model, scaler = train_svm(X_train, y_train, X_val, y_val)
    
    # Save all models
    save_baseline_models(vgg16_model, rf_model, svm_model, scaler)
