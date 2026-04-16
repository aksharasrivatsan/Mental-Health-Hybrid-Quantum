'''import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from tqdm import tqdm
from pathlib import Path

# ─────────────────────────────────────────────
# CREMA-D Emotion label map (tfds uses integers)
# ─────────────────────────────────────────────
EMOTION_MAP = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happiness",
    4: "Neutral",
    5: "Sadness",
}

SR              = 16000   # CREMA-D via tfds is 16kHz
TARGET_DURATION = 3.0
N_MFCC          = 40
TARGET_RMS      = 0.1
OUTPUT_DIR      = "/content/mfcc_images"  # change if needed


# ─────────────────────────────────────────────
# Load CREMA-D splits from tfds
# ─────────────────────────────────────────────
def load_crema_d_splits():
    """
    Loads CREMA-D with pre-defined speaker-independent splits.
    Returns: train, validation, and test datasets.
    """
    print("Loading CREMA-D dataset...")

    train_ds, val_ds, test_ds = tfds.load(
        'crema_d',
        split=['train', 'validation', 'test'],
        with_info=False
    )

    print(f"Loaded Train / Val / Test splits successfully.")
    return train_ds, val_ds, test_ds


# ─────────────────────────────────────────────
# Audio processing functions
# ─────────────────────────────────────────────
def standardize_duration(audio: np.ndarray, sr: int, target_duration: float) -> np.ndarray:
    """Trim or centre-pad audio to exactly target_duration seconds."""
    target_samples  = int(target_duration * sr)
    current_samples = len(audio)

    if current_samples > target_samples:
        audio = audio[:target_samples]
    elif current_samples < target_samples:
        pad_total = target_samples - current_samples
        pad_left  = pad_total // 2
        pad_right = pad_total - pad_left
        audio = np.pad(audio, (pad_left, pad_right), mode="constant", constant_values=0)

    return audio


def rms_normalize(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Scale audio so its RMS amplitude equals target_rms."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-9:
        return audio  # skip near-silent files
    return audio * (target_rms / rms)


def generate_mfcc_image(audio, sr, n_mfcc, output_path, image_size=(224, 224)):
    
    mfcc        = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta  = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    combined    = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

    vmin, vmax    = combined.min(), combined.max()
    combined_norm = (combined - vmin) / (vmax - vmin + 1e-9)

    # Resize numpy array directly to 224x224 BEFORE applying colormap
    # combined_norm is (120, 94) → resize to (224, 224)
    # Step 1: resize the normalized array itself to 224x224
    norm_img     = PILImage.fromarray((combined_norm * 255).astype(np.uint8), mode="L")
    norm_resized = norm_img.resize(image_size, PILImage.NEAREST)
    norm_array   = np.array(norm_resized) / 255.0  # back to [0,1], shape (224, 224)

    # Step 2: apply colormap to the full 224x224 array
    cmap        = plt.get_cmap("inferno")
    colored     = cmap(norm_array)                          # (224, 224, 4)
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)  # (224, 224, 3)

    # Step 3: save directly
    img = PILImage.fromarray(colored_rgb, mode="RGB")
    img.save(output_path)
    
    
    
# ─────────────────────────────────────────────
# Main preprocessing — reads directly from tfds
# ─────────────────────────────────────────────
def preprocess_split(ds, split_name: str, output_dir: str) -> None:
    """
    Process one split (train / val / test) from the tfds dataset.
    Saves MFCC images into output_dir/<split_name>/<emotion>/<filename>.png
    """
    split_path = Path(output_dir) / split_name

    # Create per-emotion subdirectories
    for emotion in EMOTION_MAP.values():
        (split_path / emotion).mkdir(parents=True, exist_ok=True)

    processed, skipped = 0, 0

    for i, sample in enumerate(tqdm(ds, desc=f"Processing {split_name}")):
        try:
            # Extract audio and label from tfds sample
            audio = sample["audio"].numpy().astype(np.float32)
            label = int(sample["label"].numpy())

            # Scale from int16 range to [-1, 1] if needed
            if audio.max() > 1.0:
                audio = audio / 32768.0

            # Standardize duration
            audio = standardize_duration(audio, SR, TARGET_DURATION)

            # RMS normalize
            audio = rms_normalize(audio, TARGET_RMS)

            # Get emotion folder
            emotion  = EMOTION_MAP.get(label, "Unknown")
            out_path = str(split_path / emotion / f"{split_name}_{i}.png")

            # Generate and save MFCC image
            generate_mfcc_image(audio, SR, N_MFCC, out_path)

            processed += 1

        except Exception as exc:
            print(f"\n[SKIP] sample {i} in {split_name}: {exc}")
            skipped += 1

    print(f"{split_name} → Processed: {processed}  |  Skipped: {skipped}\n")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_crema_d_splits()

    preprocess_split(train_ds, "train",      OUTPUT_DIR)
    preprocess_split(val_ds,   "validation", OUTPUT_DIR)
    preprocess_split(test_ds,  "test",       OUTPUT_DIR)

    print(f"All splits done! Images saved to: {OUTPUT_DIR}")'''

'''# ── TEST CELL — run this in VS Code ────────────────────
import numpy as np
import librosa
import matplotlib.pyplot as plt
from PIL import Image
import os

train_ds, val_ds, test_ds = load_crema_d_splits()

for sample in train_ds.take(1):
    audio = sample["audio"].numpy().astype(np.float32)

print(f"1. Raw audio - min: {audio.min():.2f}, max: {audio.max():.2f}")

# Scale
if audio.max() > 1.0:
    audio = audio / 32768.0
print(f"2. Scaled audio - min: {audio.min():.4f}, max: {audio.max():.4f}")

# Standardize
target_samples = 16000 * 3
if len(audio) > target_samples:
    audio = audio[:target_samples]
else:
    pad = target_samples - len(audio)
    audio = np.pad(audio, (pad//2, pad - pad//2), mode="constant")
print(f"3. Audio shape after padding: {audio.shape}")

# MFCC
mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
print(f"4. MFCC shape: {mfcc.shape}")
print(f"   MFCC min: {mfcc.min():.4f}, max: {mfcc.max():.4f}")

# Stack
mfcc_delta  = librosa.feature.delta(mfcc)
mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
combined    = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
print(f"5. Combined shape: {combined.shape}")
print(f"   Combined min: {combined.min():.4f}, max: {combined.max():.4f}")

# Normalize
vmin, vmax    = combined.min(), combined.max()
combined_norm = (combined - vmin) / (vmax - vmin + 1e-9)
print(f"6. Normalized min: {combined_norm.min():.4f}, max: {combined_norm.max():.4f}")
print(f"   Unique values in normalized: {len(np.unique(combined_norm.round(2)))}")

# Convert to uint8
as_uint8 = (combined_norm * 255).astype(np.uint8)
print(f"7. uint8 min: {as_uint8.min()}, max: {as_uint8.max()}")
print(f"   Unique uint8 values: {len(np.unique(as_uint8))}")

# PIL grayscale image
norm_img = Image.fromarray(as_uint8, mode="L")
print(f"8. PIL image size: {norm_img.size}, mode: {norm_img.mode}")

# Save grayscale BEFORE resize — open and check this
norm_img.save("debug_grayscale_before_resize.png")
print("   Saved: debug_grayscale_before_resize.png")

# Resize
norm_resized = norm_img.resize((224, 224), Image.NEAREST)
norm_array   = np.array(norm_resized) / 255.0
print(f"9. After resize - min: {norm_array.min():.4f}, max: {norm_array.max():.4f}")
print(f"   Unique values after resize: {len(np.unique(norm_array.round(2)))}")

# Save resized grayscale — open and check this too
norm_resized.save("debug_grayscale_after_resize.png")
print("   Saved: debug_grayscale_after_resize.png")

# Colormap
cmap        = plt.get_cmap("inferno")
colored     = cmap(norm_array)
colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
print(f"10. colored_rgb shape: {colored_rgb.shape}")
print(f"    colored_rgb min: {colored_rgb.min()}, max: {colored_rgb.max()}")
print(f"    unique colors: {len(np.unique(colored_rgb.reshape(-1,3), axis=0))}")

# Save final
img = Image.fromarray(colored_rgb, mode="RGB")
img.save("debug_final_mfcc.png")
print("    Saved: debug_final_mfcc.png")

import os
os.startfile(os.path.abspath("debug_grayscale_before_resize.png"))
os.startfile(os.path.abspath("debug_grayscale_after_resize.png"))
os.startfile(os.path.abspath("debug_final_mfcc.png"))


train_ds, val_ds, test_ds = load_crema_d_splits()

# =========================
# Get one audio sample
# =========================
for sample in train_ds.take(1):
    audio = sample["audio"].numpy().astype(np.float32)

print("=" * 60)
print("Original Audio")
print("=" * 60)
print(f"Audio shape: {audio.shape}")
print(f"Audio min: {audio.min():.4f}")
print(f"Audio max: {audio.max():.4f}")
print(f"Audio RMS: {np.sqrt(np.mean(audio**2)):.4f}")

# =========================
# Scale audio if needed
# =========================
if np.max(np.abs(audio)) > 1.0:
    audio = audio / 32768.0

print("\nAfter Scaling")
print(f"Scaled min: {audio.min():.4f}")
print(f"Scaled max: {audio.max():.4f}")

# =========================
# Standardize audio length
# 3 seconds at 16kHz
# =========================
target_samples = 16000 * 3

if len(audio) > target_samples:
    audio = audio[:target_samples]
else:
    pad = target_samples - len(audio)
    audio = np.pad(audio, (pad // 2, pad - pad // 2), mode="constant")

print("\nAfter Padding / Truncation")
print(f"Audio shape: {audio.shape}")
print(f"RMS after padding: {np.sqrt(np.mean(audio**2)):.4f}")

# =========================
# MFCC Extraction
# =========================
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=16000,
    n_mfcc=40,
    n_fft=1024,
    hop_length=256
)

print("\nOriginal MFCC Info")
print(f"MFCC shape: {mfcc.shape}")
print(f"MFCC min: {mfcc.min():.4f}")
print(f"MFCC max: {mfcc.max():.4f}")

# =========================
# Remove MFCC coefficient 0
# It usually dominates because it stores energy
# =========================
mfcc = mfcc[1:, :]

print("\nAfter Removing MFCC[0]")
print(f"MFCC shape: {mfcc.shape}")
print(f"MFCC min: {mfcc.min():.4f}")
print(f"MFCC max: {mfcc.max():.4f}")

# =========================
# Delta Features
# =========================
mfcc_delta = librosa.feature.delta(mfcc)
mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

print("\nCombined Feature Info")
print(f"Combined shape: {combined.shape}")
print(f"Combined min: {combined.min():.4f}")
print(f"Combined max: {combined.max():.4f}")
print(f"Combined range: {combined.max() - combined.min():.4f}")

# =========================
# Row-wise normalization
# =========================
combined_norm = np.zeros_like(combined)

print("\nRow-wise Range Check")
for i in range(combined.shape[0]):
    row = combined[i]
    row_min = row.min()
    row_max = row.max()

    combined_norm[i] = (row - row_min) / (row_max - row_min + 1e-9)

    print(
        f"Row {i:02d} -> min: {row_min:8.3f}, "
        f"max: {row_max:8.3f}, "
        f"range: {(row_max - row_min):8.3f}"
    )

print("\nNormalized Feature Info")
print(f"Normalized min: {combined_norm.min():.4f}")
print(f"Normalized max: {combined_norm.max():.4f}")

# =========================
# Plot raw MFCC
# =========================
plt.figure(figsize=(12, 5))
plt.imshow(
    mfcc,
    aspect='auto',
    origin='lower',
    cmap='inferno',
    vmin=np.percentile(mfcc, 5),
    vmax=np.percentile(mfcc, 95)
)
plt.colorbar()
plt.title("MFCC Without Coefficient 0")
plt.xlabel("Time Frames")
plt.ylabel("MFCC Coefficients")
plt.tight_layout()
plt.savefig("raw_mfcc_without_0.png")
plt.show()

# =========================
# Plot combined raw features
# =========================
plt.figure(figsize=(12, 8))
plt.imshow(
    combined,
    aspect='auto',
    origin='lower',
    cmap='inferno',
    vmin=np.percentile(combined, 5),
    vmax=np.percentile(combined, 95)
)
plt.colorbar()
plt.title("Combined MFCC + Delta + Delta2")
plt.xlabel("Time Frames")
plt.ylabel("Feature Channels")
plt.tight_layout()
plt.savefig("combined_raw.png")
plt.show()

# =========================
# Plot normalized combined features
# =========================
plt.figure(figsize=(12, 8))
plt.imshow(
    combined_norm,
    aspect='auto',
    origin='lower',
    cmap='inferno'
)
plt.colorbar()
plt.title("Normalized Combined Features")
plt.xlabel("Time Frames")
plt.ylabel("Feature Channels")
plt.tight_layout()
plt.savefig("combined_normalized.png")
plt.show()

# =========================
# Convert normalized data to grayscale image
# =========================
as_uint8 = (combined_norm * 255).astype(np.uint8)

gray_img = PILImage.fromarray(as_uint8, mode="L")
gray_img.save("debug_grayscale_before_resize.png")

print("\nSaved: debug_grayscale_before_resize.png")

# =========================
# Resize image
# =========================
resized_img = gray_img.resize((224, 224), PILImage.BICUBIC)
resized_img.save("debug_grayscale_after_resize.png")

print("Saved: debug_grayscale_after_resize.png")

# =========================
# Convert grayscale to inferno RGB
# =========================
resized_array = np.array(resized_img) / 255.0

cmap = plt.get_cmap("inferno")
colored = cmap(resized_array)
colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)

final_img = PILImage.fromarray(colored_rgb, mode="RGB")
final_img.save("debug_final_mfcc.png")

print("Saved: debug_final_mfcc.png")

# =========================
# Final image statistics
# =========================
print("\nFinal Image Info")
print(f"Resized grayscale min: {resized_array.min():.4f}")
print(f"Resized grayscale max: {resized_array.max():.4f}")
print(f"Unique grayscale values: {len(np.unique(resized_array.round(2)))}")
print(f"Unique RGB colors: {len(np.unique(colored_rgb.reshape(-1, 3), axis=0))}")

# =========================
# Open all saved images
# =========================
os.startfile(os.path.abspath("raw_mfcc_without_0.png"))
os.startfile(os.path.abspath("combined_raw.png"))
os.startfile(os.path.abspath("combined_normalized.png"))
os.startfile(os.path.abspath("debug_grayscale_before_resize.png"))
os.startfile(os.path.abspath("debug_grayscale_after_resize.png"))
os.startfile(os.path.abspath("debug_final_mfcc.png"))

'''

import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from tqdm import tqdm
from pathlib import Path

# =========================================================
# CREMA-D Emotion Label Map
# =========================================================
EMOTION_MAP = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happiness",
    4: "Neutral",
    5: "Sadness",
}

# =========================================================
# Configuration
# =========================================================
SR = 16000
TARGET_DURATION = 3.0
TARGET_SAMPLES = int(SR * TARGET_DURATION)
N_MFCC = 40
TARGET_RMS = 0.1
OUTPUT_DIR = "/content/mfcc_images"

# =========================================================
# Load CREMA-D Dataset
# =========================================================
def load_crema_d_splits():
    print("Loading CREMA-D dataset...")

    train_ds, val_ds, test_ds = tfds.load(
        "crema_d",
        split=["train", "validation", "test"],
        with_info=False
    )

    print("Loaded Train / Validation / Test splits successfully.")
    return train_ds, val_ds, test_ds

# =========================================================
# Standardize Audio Length
# =========================================================
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

# =========================================================
# Normalize RMS Energy
# =========================================================
def normalize_rms(audio, target_rms=TARGET_RMS):
    rms = np.sqrt(np.mean(audio ** 2))

    if rms > 1e-6:
        audio = audio * (target_rms / rms)

    return audio

# =========================================================
# Extract MFCC Image
# =========================================================
def create_mfcc_image(audio):
    # Convert to float32
    audio = audio.astype(np.float32)

    # Scale if needed
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / 32768.0

    # Standardize duration
    audio, _ = librosa.effects.trim(audio, top_db=20)
    audio = standardize_audio(audio)

    # RMS normalization
    audio = normalize_rms(audio)

    # MFCC extraction
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=1024,
        hop_length=256
    )

    # Remove MFCC coefficient 0
    mfcc = mfcc[1:, :]

    # Delta features
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Stack all features
    combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

    # Row-wise normalization
    combined_norm = np.zeros_like(combined)

    for i in range(combined.shape[0]):
        row = combined[i]
        row_min = row.min()
        row_max = row.max()

        combined_norm[i] = (
            (row - row_min) /
            (row_max - row_min + 1e-9)
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

# =========================================================
# Save Dataset Split Images
# =========================================================
def process_split(dataset, split_name):
    split_dir = Path(OUTPUT_DIR) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    emotion_counts = {emotion: 0 for emotion in EMOTION_MAP.values()}

    total_samples = 0

    print(f"\nProcessing {split_name} split...")

    for idx, sample in enumerate(tqdm(dataset)):
        try:
            audio = sample["audio"].numpy()
            label = int(sample["label"].numpy())

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

# =========================================================
# Main Function
# =========================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_ds, val_ds, test_ds = load_crema_d_splits()

    process_split(train_ds, "train")
    process_split(val_ds, "validation")
    process_split(test_ds, "test")

    print("\nAll MFCC images generated successfully.")
    print(f"Saved to: {OUTPUT_DIR}")

# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    main()
