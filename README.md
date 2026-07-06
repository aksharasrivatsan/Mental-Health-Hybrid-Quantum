# Cloud-Enabled Hybrid Quantum–Classical Framework for Speech-Based Mental Health Analysis

## 🚀 Project Goal
To develop a high-efficiency, "commodity" diagnostic tool that outperforms classical VGGNet models using Hybrid Quantum circuits.

## 📊 Research Gaps & Solutions
| Problem (VGGNet) | Quantum Solution (Our "Filler") |
| :--- | :--- |
| Image Scaling Loss | Amplitude Encoding (No resizing) |
| Parameter Heaviness | Variational Quantum Circuits (90% fewer params) |
| Low Nuance Detection | High-Dimensional Hilbert Space Mapping |

## 🛠️ Tech Stack
* **Classical:** Python, Scikit-learn, VGGNet (Baseline)
* **Quantum:** PennyLane / Qiskit (IBM Quantum Cloud)
* **Collaboration:** GitHub, Overleaf, Google Colab

## Run the Hybrid TensorFlow + Quantum Models

Activate the local environment:

```bash
source .venv/bin/activate
```

Prepare MFCC image folders in this layout:

```text
mfcc_images/
  train/
    Anger/
    Disgust/
    Fear/
    Happiness/
    Neutral/
    Sadness/
  validation/
    Anger/
    Disgust/
    Fear/
    Happiness/
    Neutral/
    Sadness/
```

Run the frozen VGG16 hybrid model:

```bash
python VGGnet_for_journal.py --data-dir mfcc_images --n-qubits 4 --quantum-depth 2
```

Run the block-5 fine-tuned hybrid model:

```bash
python VGG16_fine_tuned.py --data-dir mfcc_images --n-qubits 4 --quantum-depth 2
```

The TensorFlow feature extractor projects the MFCC/VGG feature vector into qubit
rotation angles, then the PennyLane quantum layer returns qubit expectation
values for the final TensorFlow classifier.
