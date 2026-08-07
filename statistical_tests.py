import numpy as np
from scipy import stats

def mcnemar_test(y_pred_model1, y_pred_model2, y_true, model1_name="Model 1", model2_name="Model 2"):
    """
    McNemar's test: Tests if two models have significantly different error rates.
    Used when predictions are paired (same test set).
    """
    print(f"\n{'='*70}")
    print(f"McNemar's Test: {model1_name} vs {model2_name}")
    print('='*70)
    
    # Correct/incorrect predictions
    model1_correct = (y_pred_model1 == y_true)
    model2_correct = (y_pred_model2 == y_true)
    
    # Contingency table
    # n01: model1 wrong, model2 correct
    # n10: model1 correct, model2 wrong
    n01 = np.sum((~model1_correct) & model2_correct)
    n10 = np.sum(model1_correct & (~model2_correct))
    
    print(f"Model 1 Accuracy: {np.mean(model1_correct):.4f}")
    print(f"Model 2 Accuracy: {np.mean(model2_correct):.4f}")
    
    # McNemar's statistic
    if (n01 + n10) == 0:
        print("No disagreements between models - they perform identically")
        return None
    
    chi2 = ((n01 - n10) ** 2) / (n01 + n10)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    print(f"\nn01 (M1 wrong, M2 correct): {n01}")
    print(f"n10 (M1 correct, M2 wrong): {n10}")
    print(f"Chi-squared statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Significant at α=0.05? {'YES' if p_value < 0.05 else 'NO'}")
    
    return {'chi2': chi2, 'p_value': p_value, 'significant': p_value < 0.05}


def paired_t_test(accuracies_model1, accuracies_model2, model1_name="Model 1", model2_name="Model 2"):
    """
    Paired t-test: Compare average accuracy across multiple runs/folds.
    """
    print(f"\n{'='*70}")
    print(f"Paired T-Test: {model1_name} vs {model2_name}")
    print('='*70)
    
    mean1 = np.mean(accuracies_model1)
    mean2 = np.mean(accuracies_model2)
    
    print(f"Model 1 Mean Accuracy: {mean1:.4f}")
    print(f"Model 2 Mean Accuracy: {mean2:.4f}")
    print(f"Difference: {abs(mean1 - mean2):.4f}")
    
    t_stat, p_value = stats.ttest_ind(accuracies_model1, accuracies_model2)
    
    print(f"\nT-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Significant at α=0.05? {'YES' if p_value < 0.05 else 'NO'}")
    
    return {'t_stat': t_stat, 'p_value': p_value, 'significant': p_value < 0.05}


def effect_size(accuracy_diff, baseline_accuracy):
    """
    Calculate effect size (Cohen's d-like metric) for accuracy improvements.
    Shows if improvement is practically meaningful.
    """
    # Rough approximation: Cohen's h for proportions
    p1 = baseline_accuracy
    p2 = baseline_accuracy + accuracy_diff
    
    # Clamp to valid probability range
    p2 = np.clip(p2, 0, 1)
    
    cohens_h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
    
    print(f"\nEffect Size (Cohen's h): {cohens_h:.4f}")
    if abs(cohens_h) < 0.2:
        print("→ Small effect (improvement is modest)")
    elif abs(cohens_h) < 0.5:
        print("→ Medium effect (improvement is notable)")
    else:
        print("→ Large effect (improvement is substantial)")
    
    return cohens_h


def summarize_significance(model_accuracies_dict, baseline_model="Pure VGG16"):
    """
    Comprehensive significance summary for all models vs baseline.
    """
    print(f"\n{'='*80}")
    print("SIGNIFICANCE SUMMARY")
    print('='*80)
    
    baseline_acc = model_accuracies_dict[baseline_model]
    
    for model_name, accuracy in model_accuracies_dict.items():
        if model_name == baseline_model:
            continue
        
        diff = accuracy - baseline_acc
        print(f"\n{model_name} vs {baseline_model}:")
        print(f"  Accuracy: {accuracy:.4f} vs {baseline_acc:.4f}")
        print(f"  Improvement: {diff:+.4f} ({100*diff/baseline_acc:+.2f}%)")
        
        if diff > 0:
            effect_size(diff, baseline_acc)
            print("  ✓ Better than baseline")
        elif diff < 0:
            print("  ✗ Worse than baseline")
        else:
            print("  ≈ Same as baseline")


if __name__ == "__main__":
    # USAGE EXAMPLE:
    # python statistical_tests.py
    
    # Example: Compare two models
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred_model1 = np.array([0, 1, 2, 0, 1, 2, 0, 1, 1, 0])  # 1 error
    y_pred_model2 = np.array([0, 1, 2, 0, 1, 1, 0, 1, 2, 0])  # 1 error
    
    mcnemar_test(y_pred_model1, y_pred_model2, y_true, 
                 "Hybrid Quantum", "Pure VGG16")
    
    # Example: Compare across multiple runs
    accuracies_model1 = np.array([0.75, 0.76, 0.74, 0.77, 0.75])
    accuracies_model2 = np.array([0.72, 0.71, 0.73, 0.71, 0.72])
    
    paired_t_test(accuracies_model1, accuracies_model2,
                  "Hybrid Quantum", "Pure VGG16")
