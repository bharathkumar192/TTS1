#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Pipelines fail if any command fails, not just the last one.
set -o pipefail

# --- Configuration ---
export HUGGING_FACE_HUB_TOKEN=""
PROJECT_DIR="xtts_sangeeta_finetune"
BASE_MODEL_DIR="${PROJECT_DIR}/xtts_v2_base_model"
DATASET_DOWNLOAD_DIR="${PROJECT_DIR}/sangeeta_dataset_download" # Raw download
DATASET_PROCESSED_DIR="${PROJECT_DIR}/sangeeta_processed"      # Processed data
TRAINING_OUTPUT_DIR="${PROJECT_DIR}/run/training"             # Training runs/checkpoints will go here

# Hugging Face details
USER_DATASET_ID="bharathkumar1922001/Sangeeta-Hindi-VoiceData-1x"
BASE_MODEL_REPO="coqui/XTTS-v2"
TARGET_HF_MODEL_REPO="bharathkumar1922001/TTS-1x" # Your target repo for uploading results

# --- Check for Hugging Face Token ---
if [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo "Error: HUGGING_FACE_HUB_TOKEN environment variable is not set."
  echo "Please set it before running the script: export HUGGING_FACE_HUB_TOKEN='your_token_here'"
  exit 1
fi
echo "✓ Hugging Face token found."

# --- Setup Directories ---
echo ">>> Setting up directories..."
mkdir -p "$PROJECT_DIR"
mkdir -p "$BASE_MODEL_DIR"
mkdir -p "$DATASET_DOWNLOAD_DIR"
mkdir -p "$DATASET_PROCESSED_DIR"
mkdir -p "$TRAINING_OUTPUT_DIR"
echo "✓ Directories created."

# --- Install Dependencies ---
echo ">>> Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q coqui-tts datasets torch torchaudio huggingface_hub deepspeed soundfile pandas transformers accelerate
echo "✓ Dependencies installed."

# --- Log in to Hugging Face ---
echo ">>> Logging in to Hugging Face Hub..."
huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN" --add-to-git-credential
echo "✓ Logged in to Hugging Face Hub."

# --- Download Base XTTS-v2 Model ---
echo ">>> Downloading base XTTS-v2 model from ${BASE_MODEL_REPO}..."
# Use python for slightly more robust download + rename logic
python <<EOF
import os
from huggingface_hub import hf_hub_download

repo_id = "${BASE_MODEL_REPO}"
local_dir = "${BASE_MODEL_DIR}"
filenames = [
    "config.json",
    "model.pth",
    "dvae.pth",
    "mel_stats.pth",
    "speakers_xtts.pth",
    "vocab.json"
    # Handle vocab.json potentially being Vocab.json
]

# Download main files
for filename in filenames:
    try:
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, local_dir_use_symlinks=False)
    except Exception as e:
        print(f"Warning: Could not download {filename}: {e}")

# Specifically handle Vocab.json vs vocab.json
vocab_target_path = os.path.join(local_dir, "vocab.json")
vocab_source_filename = "Vocab.json"
vocab_source_path = os.path.join(local_dir, vocab_source_filename)

if not os.path.exists(vocab_target_path):
    print(f"Attempting to download {vocab_source_filename}...")
    try:
        hf_hub_download(repo_id=repo_id, filename=vocab_source_filename, local_dir=local_dir, local_dir_use_symlinks=False)
        if os.path.exists(vocab_source_path):
            print(f"Renaming {vocab_source_filename} to vocab.json")
            os.rename(vocab_source_path, vocab_target_path)
        else:
             print(f"Downloaded {vocab_source_filename}, but file not found at expected location {vocab_source_path} after download.")
    except Exception as e:
        print(f"Error downloading {vocab_source_filename}, checking for vocab.json directly: {e}")
        try:
            hf_hub_download(repo_id=repo_id, filename="vocab.json", local_dir=local_dir, local_dir_use_symlinks=False)
        except Exception as e2:
            print(f"Error downloading vocab.json as well: {e2}")
            print("CRITICAL: Could not obtain vocab file.")
            # Optionally exit here: exit(1)

if not os.path.exists(vocab_target_path):
     print("CRITICAL: vocab.json could not be downloaded or created. Please check manually.")
     # Optionally exit here: exit(1)

print("Base model download attempt finished.")
EOF
echo "✓ Base model files downloaded to ${BASE_MODEL_DIR}."

# --- Download User Dataset ---
echo ">>> Downloading user dataset ${USER_DATASET_ID}..."
# Using huggingface_hub download which is simpler than snapshot_download for basic cases
huggingface-hub download --repo-type dataset "$USER_DATASET_ID" --local-dir "$DATASET_DOWNLOAD_DIR" --local-dir-use-symlinks False
echo "✓ User dataset downloaded to ${DATASET_DOWNLOAD_DIR}."

# --- Run Data Preprocessing ---
echo ">>> Running data preprocessing script..."
# Create the preprocess_data.py file dynamically or ensure it exists
# Assuming preprocess_data.py is in the same directory as this shell script
python preprocess_data.py --raw_data_dir "$DATASET_DOWNLOAD_DIR" --processed_data_dir "$DATASET_PROCESSED_DIR"
echo "✓ Data preprocessing finished. Processed data in ${DATASET_PROCESSED_DIR}."

# --- Run Training ---
echo ">>> Starting model training..."
# Assuming train_model.py is in the same directory
# Pass necessary paths as arguments or rely on the script finding them relative to PROJECT_DIR
# Set CUDA_VISIBLE_DEVICES - adjust "0" if you want to use a different GPU
CUDA_VISIBLE_DEVICES="0" python train_model.py \
    --base_model_dir "$BASE_MODEL_DIR" \
    --processed_data_dir "$DATASET_PROCESSED_DIR" \
    --output_dir "$TRAINING_OUTPUT_DIR"

# Find the specific run directory created by the trainer
# This assumes the trainer creates a directory starting with RUN_NAME inside TRAINING_OUTPUT_DIR
# Modify RUN_NAME if it's different in train_model.py
LATEST_RUN_DIR=$(ls -td "${TRAINING_OUTPUT_DIR}"/GPT_XTTS_Sangeeta_Hindi_FT*/ | head -1)

if [ -z "$LATEST_RUN_DIR" ]; then
  echo "Error: Could not find the latest training run directory in ${TRAINING_OUTPUT_DIR}"
  exit 1
fi
echo "✓ Training finished. Output files are in ${LATEST_RUN_DIR}"

# --- Push Model to Hugging Face Hub ---
echo ">>> Uploading trained model artifacts to ${TARGET_HF_MODEL_REPO}..."
python <<EOF
from huggingface_hub import HfApi, upload_folder
import os

local_folder_path = "${LATEST_RUN_DIR}"
target_repo_id = "${TARGET_HF_MODEL_REPO}"
commit_msg = "Upload fine-tuned Sangeeta Hindi XTTS model"

print(f"Uploading contents of {local_folder_path} to {target_repo_id}...")

try:
    # Ensure repository exists, create if not (private=False by default)
    api = HfApi()
    api.create_repo(repo_id=target_repo_id, repo_type='model', exist_ok=True)
    print(f"Repository {target_repo_id} ensured.")

    # Upload the folder contents
    upload_folder(
        folder_path=local_folder_path,
        repo_id=target_repo_id,
        repo_type='model',
        commit_message=commit_msg,
        # Set ignore_patterns if you want to exclude certain files/folders
        # ignore_patterns=["*.log", "tensorboard/*"]
    )
    print("✓ Upload successful!")
except Exception as e:
    print(f"Error uploading to Hugging Face Hub: {e}")
    # Optionally exit here: exit(1)

EOF

echo ">>> All steps completed! <<<"