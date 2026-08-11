from statistical_tests import mcnemar_test
import pickle

# You'll need results['Hybrid Quantum'] and results['Pure VGG16'] from evaluation.py
# Easiest: modify evaluation.py's __main__ to pickle `results` at the end, e.g.:
# with open("results/results.pkl", "wb") as f: pickle.dump(results, f)

with open("results/results.pkl", "rb") as f:
    results = pickle.load(f)

mcnemar_test(
    results['Hybrid Quantum']['y_pred'],
    results['Pure VGG16']['y_pred'],
    results['Pure VGG16']['y_true'],
    "Hybrid Quantum", "Pure VGG16"
)
