import argparse
import os
import random
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

try:
    PROJECT_DIR = Path(__file__).resolve().parent
except NameError:
    PROJECT_DIR = Path.cwd().resolve()

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))

import librosa
import matplotlib
import numpy as np
from datasets import Audio, load_dataset
from PIL import Image as PILImage
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CREMA_DATASET = "silky1708/CREMA-D"
TESS_KAGGLE_DATASET = "ejlok1/toronto-emotional-speech-set-tess"
RAVDESS_KAGGLE_DATASET = "uwrfkaggler/ravdess-emotional-speech-audio"
TESS_ARCHIVE = "toronto-emotional-speech-set-tess.zip"

EMOTION_MAP = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happiness",
    4: "Neutral",
    5: "Sadness",
    6: "Calm",
    7: "Surprise",
}

CREMA_LABEL_MAP = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happiness",
    4: "Neutral",
    5: "Sadness",
}

TESS_ROOT_DIRS = {
    "OAF_angry",
    "OAF_disgust",
    "OAF_Fear",
    "OAF_happy",
    "OAF_neutral",
    "OAF_Pleasant_surprise",
    "OAF_Sad",
    "YAF_angry",
    "YAF_disgust",
    "YAF_fear",
    "YAF_happy",
    "YAF_neutral",
    "YAF_pleasant_surprised",
    "YAF_sad",
}

RAVDESS_LABEL_MAP = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happiness",
    "04": "Sadness",
    "05": "Anger",
    "06": "Fear",
    "07": "Disgust",
    "08": "Surprise",
}

SR = 16000
TARGET_DURATION = 3.0
TARGET_SAMPLES = int(SR * TARGET_DURATION)
N_MFCC = 40
TARGET_RMS = 0.1
DEFAULT_RAW_DIR = PROJECT_DIR / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "mfcc_images"
AUDIO_EXTENSIONS = {".wav"}


@dataclass(frozen=True)
class AudioRecord:
    path: Path | None
    emotion: str
    source: str
    audio_record: dict | None = None


def run_command(command):
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def load_dataset_with_retry(*args, retries=10, initial_wait=10, **kwargs):
    wait_seconds = initial_wait

    for attempt in range(1, retries + 1):
        try:
            return load_dataset(*args, **kwargs)
        except Exception as exc:
            message = str(exc)
            is_rate_limited = "429" in message or "Too Many Requests" in message

            if not is_rate_limited or attempt == retries:
                raise

            print(
                "Rate limited while loading dataset. "
                f"Waiting {wait_seconds}s before retry {attempt + 1}/{retries}..."
            )
            time.sleep(wait_seconds)
            wait_seconds *= 2


def download_kaggle_dataset(dataset_name, output_dir, *, unzip=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    command = ["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(output_dir)]

    if unzip:
        command.append("--unzip")

    run_command(command)


def archive_branches(zip_path):
    with zipfile.ZipFile(zip_path) as archive:
        branches = set()
        for name in archive.namelist():
            parts = Path(name).parts
            if len(parts) >= 2:
                branches.add(f"{parts[0]}/{parts[1]}")
        return sorted(branches)


def safe_extract_member(archive, member, destination):
    parts = Path(member.filename).parts

    if member.is_dir() or len(parts) < 3 or parts[1] not in TESS_ROOT_DIRS:
        return None

    target_path = destination.joinpath(*parts[1:])
    target_path.relative_to(destination)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with archive.open(member) as source, target_path.open("wb") as target:
        shutil.copyfileobj(source, target)

    return target_path


def extract_tess_roots(tess_dir):
    zip_path = tess_dir / TESS_ARCHIVE

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Expected TESS archive at {zip_path}. "
            "Run with --download-kaggle or download it first."
        )

    print("\nTESS archive branches:")
    for branch in archive_branches(zip_path):
        print(f"  {branch}")

    extracted = 0
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if safe_extract_member(archive, member, tess_dir):
                extracted += 1

    print(f"\nExtracted {extracted} TESS files into flattened root folders at {tess_dir}")


def tess_emotion_from_folder(folder_name):
    normalized = folder_name.lower()

    if "pleasant_surprise" in normalized or "pleasant_surprised" in normalized:
        return "Surprise"
    if "angry" in normalized:
        return "Anger"
    if "disgust" in normalized:
        return "Disgust"
    if "fear" in normalized:
        return "Fear"
    if "happy" in normalized:
        return "Happiness"
    if "neutral" in normalized:
        return "Neutral"
    if "sad" in normalized:
        return "Sadness"

    return None


def collect_tess_records(tess_dir):
    records = []
    
    if not tess_dir.exists():
        return records

    # Iterates over all subdirectories inside tess_dir dynamically
    for folder in sorted(tess_dir.iterdir()):
        if folder.is_dir():
            emotion = tess_emotion_from_folder(folder.name)
            if emotion is None:
                continue

            for audio_path in sorted(folder.rglob("*")):
                if audio_path.is_file() and audio_path.suffix.lower() in AUDIO_EXTENSIONS:
                    records.append(AudioRecord(audio_path, emotion, "TESS"))

    return records


def ravdess_emotion_from_path(audio_path):
    parts = audio_path.stem.split("-")

    if len(parts) != 7:
        return None

    modality, vocal_channel, emotion_code = parts[0], parts[1], parts[2]

    if modality != "03" or vocal_channel != "01":
        return None

    return RAVDESS_LABEL_MAP.get(emotion_code)


def collect_ravdess_records(ravdess_dir):
    records = []

    for audio_path in sorted(ravdess_dir.rglob("*")):
        if not audio_path.is_file() or audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        emotion = ravdess_emotion_from_path(audio_path)
        if emotion:
            records.append(AudioRecord(audio_path, emotion, "RAVDESS"))

    return records


def collect_crema_records(crema_dir):
    records = []
    # CREMA-D audio filename format: 1001_DFA_ANG_XX.wav
    crema_emotion_map = {
        "ANG": "Anger",
        "DIS": "Disgust",
        "FEA": "Fear",
        "HAP": "Happiness",
        "NEU": "Neutral",
        "SAD": "Sadness"
    }
    
    for audio_path in sorted(crema_dir.rglob("*.wav")):
        parts = audio_path.stem.split("_")
        if len(parts) >= 3:
            emotion_code = parts[2]
            emotion = crema_emotion_map.get(emotion_code)
            if emotion:
                records.append(AudioRecord(audio_path, emotion, "CREMA-D"))
                
    return records


def split_records(records, *, validation_size, test_size, seed):
    rng = random.Random(seed)
    by_emotion = {emotion: [] for emotion in EMOTION_MAP.values()}

    for record in records:
        by_emotion.setdefault(record.emotion, []).append(record)

    splits = {"train": [], "validation": [], "test": []}

    for emotion, emotion_records in sorted(by_emotion.items()):
        rng.shuffle(emotion_records)
        total = len(emotion_records)

        test_count = round(total * test_size)
        validation_count = round(total * validation_size)

        splits["test"].extend(emotion_records[:test_count])
        splits["validation"].extend(emotion_records[test_count:test_count + validation_count])
        splits["train"].extend(emotion_records[test_count + validation_count:])

    for split_records_ in splits.values():
        rng.shuffle(split_records_)

    return splits


def load_audio_sample(record):
    if record.audio_record is not None:
        audio_path = record.audio_record.get("path")
        if audio_path:
            audio, _ = librosa.load(audio_path, sr=SR, mono=True)
            return audio

    if record.path is not None:
        audio, _ = librosa.load(record.path, sr=SR, mono=True)
        return audio

    raise ValueError("Audio record did not include a readable path.")


def standardize_audio(audio, target_samples=TARGET_SAMPLES):
    if len(audio) > target_samples:
        return audio[:target_samples]

    pad = target_samples - len(audio)
    return np.pad(audio, (pad // 2, pad - pad // 2), mode="constant")


def normalize_rms(audio, target_rms=TARGET_RMS):
    rms = np.sqrt(np.mean(audio ** 2))

    if rms > 1e-6:
        audio = audio * (target_rms / rms)

    return audio


def create_mfcc_image(audio):
    audio = audio.astype(np.float32)

    if np.max(np.abs(audio)) > 1.0:
        audio = audio / 32768.0

    audio, _ = librosa.effects.trim(audio, top_db=20)
    audio = standardize_audio(audio)
    audio = normalize_rms(audio)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=512,
        hop_length=128,
    )

    mfcc = mfcc[1:, :]
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
    combined = np.sign(combined) * np.log1p(np.abs(combined))
    combined_norm = np.zeros_like(combined)

    for idx in range(combined.shape[0]):
        row = combined[idx]
        low = np.percentile(row, 5)
        high = np.percentile(row, 95)
        row = np.clip(row, low, high)
        combined_norm[idx] = (row - row.min()) / (row.max() - row.min() + 1e-9)

    as_uint8 = (combined_norm * 255).astype(np.uint8)
    gray_img = PILImage.fromarray(as_uint8, mode="L")
    resized_img = gray_img.resize((224, 224), PILImage.BICUBIC)

    resized_array = np.array(resized_img) / 255.0
    cmap = plt.get_cmap("inferno")
    colored = cmap(resized_array)
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)

    return PILImage.fromarray(colored_rgb, mode="RGB")


def process_split(records, split_name, output_dir):
    split_dir = Path(output_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    emotion_counts = {emotion: 0 for emotion in EMOTION_MAP.values()}
    source_counts = {}

    print(f"\nProcessing {split_name} split...")

    for idx, record in enumerate(tqdm(records)):
        try:
            emotion_dir = split_dir / record.emotion
            emotion_dir.mkdir(parents=True, exist_ok=True)

            source_counts[record.source] = source_counts.get(record.source, 0) + 1
            emotion_counts[record.emotion] += 1

            file_name = f"{split_name}_{record.source.lower().replace('-', '')}_{idx:05d}.png"
            save_path = emotion_dir / file_name

            if save_path.exists():
                continue

            audio = load_audio_sample(record)
            mfcc_img = create_mfcc_image(audio)
            mfcc_img.save(save_path)
        except Exception as exc:
            print(f"Error processing {record.source} sample {idx}: {exc}")

    print(f"\nFinished {split_name} split")
    print(f"Total records: {len(records)}")

    print("\nClass distribution:")
    for emotion, count in emotion_counts.items():
        print(f"{emotion:12s}: {count}")

    print("\nSource distribution:")
    for source, count in sorted(source_counts.items()):
        print(f"{source:12s}: {count}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate MFCC image folders from CREMA-D, flattened TESS roots, "
            "and RAVDESS speech .wav files."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--download-kaggle", action="store_true")
    parser.add_argument("--extract-tess", action="store_true")
    parser.add_argument("--skip-crema", action="store_true")
    parser.add_argument("--skip-tess", action="store_true")
    parser.add_argument("--skip-ravdess", action="store_true")
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args, _ = build_arg_parser().parse_known_args()

    crema_dir = args.raw_dir/ "crema"
    tess_dir = args.raw_dir / "tess"
    ravdess_dir = args.raw_dir / "ravdess"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []

    # 1. Collect CREMA-D from local folder
    crema_records = collect_crema_records(crema_dir)
    print(f"Collected {len(crema_records)} CREMA-D .wav files.")
    records.extend(crema_records)

    # 2. Collect TESS from local folder
    tess_records = collect_tess_records(tess_dir)
    print(f"Collected {len(tess_records)} TESS .wav files.")
    records.extend(tess_records)

    # 3. Collect RAVDESS from local folder
    ravdess_records = collect_ravdess_records(ravdess_dir)
    print(f"Collected {len(ravdess_records)} RAVDESS .wav files.")
    records.extend(ravdess_records)

    if not records:
        raise RuntimeError("No audio records were collected. Check file paths.")

    # Split dataset (80% Train, 10% Val, 10% Test)
    splits = split_records(
        records,
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    # Process audio into MFCC image sets
    for split_name in ("train", "validation", "test"):
        process_split(splits[split_name], split_name, args.output_dir)

    print("\nAll MFCC images generated successfully!")


if __name__ == "__main__":
    main()





#output
"""
Collected 7442 CREMA-D .wav files.
Collected 2800 TESS .wav files.
Collected 2880 RAVDESS .wav files.

Processing train split...
  0%|          | 0/10494 [00:00<?, ?it/s]/tmp/ipykernel_1088/2406337677.py:367: DeprecationWarning: 'mode' parameter is deprecated and will be removed in Pillow 13 (2026-10-15)
  gray_img = PILImage.fromarray(as_uint8, mode="L")
/tmp/ipykernel_1088/2406337677.py:375: DeprecationWarning: 'mode' parameter is deprecated and will be removed in Pillow 13 (2026-10-15)
  return PILImage.fromarray(colored_rgb, mode="RGB")
100%|██████████| 10494/10494 [12:07<00:00, 14.43it/s]

Finished train split
Total records: 10494

Class distribution:
Anger       : 1643
Disgust     : 1643
Fear        : 1643
Happiness   : 1643
Neutral     : 1343
Sadness     : 1643
Calm        : 308
Surprise    : 628

Source distribution:
CREMA-D     : 5974
RAVDESS     : 2290
TESS        : 2230

Processing validation split...
100%|██████████| 1314/1314 [01:27<00:00, 15.00it/s]

Finished validation split
Total records: 1314

Class distribution:
Anger       : 206
Disgust     : 206
Fear        : 206
Happiness   : 206
Neutral     : 168
Sadness     : 206
Calm        : 38
Surprise    : 78

Source distribution:
CREMA-D     : 728
RAVDESS     : 295
TESS        : 291

Processing test split...
100%|██████████| 1314/1314 [01:28<00:00, 14.92it/s]

Finished test split
Total records: 1314

Class distribution:
Anger       : 206
Disgust     : 206
Fear        : 206
Happiness   : 206
Neutral     : 168
Sadness     : 206
Calm        : 38
Surprise    : 78

Source distribution:
CREMA-D     : 740
RAVDESS     : 295
TESS        : 279

All MFCC images generated successfully!
"""
