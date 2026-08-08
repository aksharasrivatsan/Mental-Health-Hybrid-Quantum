import os
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

# CREMA-D Emotion Label Map
EMOTION_MAP = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happiness",
    4: "Neutral",
    5: "Sadness",
}

# Configuration
SR = 16000
TARGET_DURATION = 3.0
TARGET_SAMPLES = int(SR * TARGET_DURATION)
N_MFCC = 40
TARGET_RMS = 0.1
OUTPUT_DIR = "/content/mfcc_images"

# Load CREMA-D Dataset
def load_crema_d_splits():
    print("Loading CREMA-D dataset...")

    full_ds = load_dataset("silky1708/CREMA-D", split="train")

    # myleslinder/crema-d only ships a single "train" split, so we
    # carve out train/validation/test splits ourselves (80/10/10)
    split_1 = full_ds.train_test_split(test_size=0.2, seed=42)
    train_ds = split_1["train"]
    split_2 = split_1["test"].train_test_split(test_size=0.5, seed=42)
    val_ds = split_2["train"]
    test_ds = split_2["test"]

    print("Loaded Train / Validation / Test splits successfully.")
    return train_ds, val_ds, test_ds

# Standardize Audio Length
def standardize_audio(audio, target_samples=TARGET_SAMPLES):
    if len(audio) > target_samples:
        audio = audio[:target_samples]
    else:
        pad = target_samples - len(audio)
        audio = np.pad(
            audio,
            (pad // 2, pad - pad // 2),
            mode="constant"
        )

    return audio

# Normalize RMS Energy
def normalize_rms(audio, target_rms=TARGET_RMS):
    rms = np.sqrt(np.mean(audio ** 2))

    if rms > 1e-6:
        audio = audio * (target_rms / rms)

    return audio

# Extract MFCC Image
def create_mfcc_image(audio):
    # Convert to float32
    audio = audio.astype(np.float32)

    # Scale if needed
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / 32768.

    audio, _ = librosa.effects.trim(audio, top_db=20)

    # Standardize duration
    audio = standardize_audio(audio)

    # RMS normalization
    audio = normalize_rms(audio)

    # MFCC extraction
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=512,
        hop_length=128
    )

    # Remove MFCC coefficient 0
    mfcc = mfcc[1:, :]

    # Delta features
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Stack all features
    combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
    combined = np.sign(combined) * np.log1p(np.abs(combined))

    # Row-wise normalization
    combined_norm = np.zeros_like(combined)

    for i in range(combined.shape[0]):
        row = combined[i]

        # Contrast stretching
        low = np.percentile(row, 5)
        high = np.percentile(row, 95)

        row = np.clip(row, low, high)

        # Normalize row
        combined_norm[i] = (
            (row - row.min()) /
            (row.max() - row.min() + 1e-9)
        )

    # Convert to grayscale image
    as_uint8 = (combined_norm * 255).astype(np.uint8)
    gray_img = PILImage.fromarray(as_uint8, mode="L")

    # Resize to CNN-friendly size
    resized_img = gray_img.resize((224, 224), PILImage.BICUBIC)

    # Convert to RGB using inferno colormap
    resized_array = np.array(resized_img) / 255.0

    cmap = plt.get_cmap("inferno")
    colored = cmap(resized_array)
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)

    final_img = PILImage.fromarray(colored_rgb, mode="RGB")

    return final_img

# Save Dataset Split Images
def process_split(dataset, split_name):
    split_dir = Path(OUTPUT_DIR) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    emotion_counts = {emotion: 0 for emotion in EMOTION_MAP.values()}

    total_samples = 0

    print(f"\nProcessing {split_name} split...")

    for idx, sample in enumerate(tqdm(dataset)):
        try:
            audio = np.array(sample["audio"]["array"])
            label = int(sample["label"])

            emotion_name = EMOTION_MAP[label]
            emotion_dir = split_dir / emotion_name
            emotion_dir.mkdir(parents=True, exist_ok=True)

            # Generate MFCC image
            mfcc_img = create_mfcc_image(audio)

            # File name
            file_name = f"{split_name}_{idx:05d}.png"
            save_path = emotion_dir / file_name

            # Save image
            mfcc_img.save(save_path)

            emotion_counts[emotion_name] += 1
            total_samples += 1

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")

    print(f"\nFinished {split_name} split")
    print(f"Total images saved: {total_samples}")

    print("\nClass distribution:")
    for emotion, count in emotion_counts.items():
        print(f"{emotion:12s}: {count}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_ds, val_ds, test_ds = load_crema_d_splits()

    process_split(train_ds, "train")
    process_split(val_ds, "validation")
    process_split(test_ds, "test")

    print("\nAll MFCC images generated successfully.")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
