# Run these before running data_loader_new_CREMA_TESS_RAVDESS.py
#------------Terminal commands-----------------
# 1. Ensure Kaggle directory exists
mkdir -p ~/.kaggle

# 2. Download CREMA-D from Kaggle
kaggle datasets download -d ejlok1/cremad -p ./data/raw/crema --unzip

# 3. Download TESS and clean parent directory structure
kaggle datasets download -d manikantagade/tess-dataset -p ./data/raw/tess --unzip
mv ./data/raw/tess/tess\ raven/* ./data/raw/tess/ 2>/dev/null || mv ./data/raw/tess/TESS*/* ./data/raw/tess/ 2>/dev/null || true
rm -rf ./data/raw/tess/tess\ raven ./data/raw/tess/TESS*

# 4. Download RAVDESS (.wav audio files only)
kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio -p ./data/raw/ravdess --unzip


#------------Colab commands-----------------
# 1. Ensure Kaggle directory exists
!mkdir -p ~/.kaggle

# 2. Download CREMA-D from Kaggle (bypasses HF rate limits)
!kaggle datasets download -d ejlok1/cremad -p ./data/raw/crema --unzip

# 3. Download TESS and clean parent directory structure
!kaggle datasets download -d manikantagade/tess-dataset -p ./data/raw/tess --unzip
!mv ./data/raw/tess/tess\ raven/* ./data/raw/tess/ 2>/dev/null || mv ./data/raw/tess/TESS*/* ./data/raw/tess/ 2>/dev/null || true
!rm -rf ./data/raw/tess/tess\ raven ./data/raw/tess/TESS*

# 4. Download RAVDESS (.wav audio files only)
!kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio -p ./data/raw/ravdess --unzip
