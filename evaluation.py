import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_curve, auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from pathlib import Path
import pickle
import tensorflow as tf
from hybrid_quantum_tf import load_image_datasets
from hybrid_quantum_tf import QuantumCircuitLayer

EMOTION_MAP = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happiness",
    4: "Neutral",
    5: "Sadness",
}

def load_test_set(data_dir, batch_size=32):
    """Load test set from data directory"""
    test_dir = Path(data_dir) / "test"
    
    if not test_dir.exists():
        print(f"WARNING: Test directory not found at {test_dir}")
        print("Using validation set as test set (not ideal)")
        _, test_ds = load_image_datasets(data_dir, batch_size)
        return test_ds
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode="int",
    )
    return test_ds


def evaluate_model(model, test_ds, model_name="Model"):
    """Evaluate model on test set and return metrics"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print('='*60)
    
    y_true, y_pred = [], []
    y_pred_proba = []
    
    # Get predictions on test set
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_pred_proba.append(predictions)
        y_pred.append(np.argmax(predictions, axis=1))
        y_true.append(labels.numpy())
    
    y_true = np.hstack(y_true)
    y_pred = np.hstack(y_pred)
    y_pred_proba = np.vstack(y_pred_proba)
    
    # Overall accuracy
    accuracy = np.mean(y_true == y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    print("\nPer-Class Metrics:")
    print(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 48)
    for i, emotion in EMOTION_MAP.items():
        print(f"{emotion:<12} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f}")
    
    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    print("-" * 48)
    print(f"{'Macro Avg':<12} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f}")
    
    return {
        'accuracy': accuracy,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_f1': macro_f1,
    }


def plot_confusion_matrix(y_true, y_pred, model_name, save_dir="results"):
    """Plot and save confusion matrix"""
    Path(save_dir).mkdir(exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=[EMOTION_MAP[i] for i in range(6)],
        yticklabels=[EMOTION_MAP[i] for i in range(6)],
        cbar_kws={'label': 'Count'}
    )
    plt.title(f"Confusion Matrix: {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    save_path = Path(save_dir) / f"cm_{model_name.replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


def plot_roc_curves(y_true, y_pred_proba, model_name, save_dir="results"):
    """Plot ROC curves for each emotion"""
    Path(save_dir).mkdir(exist_ok=True)
    
    y_true_bin = label_binarize(y_true, classes=range(6))
    
    plt.figure(figsize=(12, 8))
    
    for i in range(6):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{EMOTION_MAP[i]} (AUC = {roc_auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves: {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = Path(save_dir) / f"roc_{model_name.replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


def compare_models(results_dict, save_dir="results"):
    """Create comparison table of all models"""
    Path(save_dir).mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print('='*80)
    print(f"{'Model':<25} {'Accuracy':<15} {'Macro F1':<15}")
    print("-" * 80)
    
    comparison_data = []
    for model_name, results in results_dict.items():
        print(f"{model_name:<25} {results['accuracy']:<15.4f} {results['macro_f1']:<15.4f}")
        comparison_data.append({
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'Macro F1': results['macro_f1']
        })
    
    # Save as CSV
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    csv_path = Path(save_dir) / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison to: {csv_path}")
    
    return df


if __name__ == "__main__":
    # USAGE:
    # python evaluation.py
    
    from hybrid_quantum_tf import build_hybrid_vgg_quantum_model
    
    results = {}
    data_dir = Path("mfcc_images")
    
    # Load test set
    test_ds = load_test_set(data_dir)
    
    # 1. Evaluate Hybrid Quantum Model (Frozen)
    print("\nLoading Hybrid Quantum Model (Frozen VGG16)...")
    # Assume model was trained and saved
    # For now, we'll need to train it fresh
    # model_hybrid = load_model("hybrid_quantum_model.h5")
    
    # 2. Evaluate Pure VGG16
    print("\nLoading Pure VGG16 Model...")
    vgg16_model = tf.keras.models.load_model("baseline_models/vgg16_pure.h5")
    results['Pure VGG16'] = evaluate_model(vgg16_model, test_ds, "Pure VGG16")
    plot_confusion_matrix(
        results['Pure VGG16']['y_true'],
        results['Pure VGG16']['y_pred'],
        "Pure VGG16"
    )
    plot_roc_curves(
        results['Pure VGG16']['y_true'],
        results['Pure VGG16']['y_pred_proba'],
        "Pure VGG16"
    )
    
    # Compare models
    compare_models(results)

 import pickle
    with open("results/results.pkl", "wb") as f:
        pickle.dump(results, f)
