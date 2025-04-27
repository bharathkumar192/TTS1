#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Pipelines fail if any command fails, not just the last one.
set -o pipefail

# --- Configuration ---
# Ensure HUGGING_FACE_HUB_TOKEN is set in your environment before running
# export HUGGING_FACE_HUB_TOKEN=""

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
echo "✓ Hugging Face token found in environment."

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
pip install  --upgrade pip
pip install  coqui-tts datasets torch torchaudio huggingface_hub deepspeed soundfile pandas transformers accelerate hf_xet
echo "✓ Dependencies installed."

# --- Log in to Hugging Face ---
echo ">>> Logging in to Hugging Face Hub (using environment token)..."
# Login happens automatically if HUGGING_FACE_HUB_TOKEN is set for library functions
# Explicit login for CLI commands if needed
huggingface-cli whoami 
# > /dev/null 2>&1 || huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN" --add-to-git-credential
echo "✓ Logged in to Hugging Face Hub."

# --- Download Base XTTS-v2 Model ---
# Check if model files already exist to potentially skip download
CONFIG_PATH_CHECK="${BASE_MODEL_DIR}/config.json"
MODEL_PATH_CHECK="${BASE_MODEL_DIR}/model.pth"
SPEAKERS_PATH_CHECK="${BASE_MODEL_DIR}/speakers_xtts.pth"

if [ -f "$CONFIG_PATH_CHECK" ] && [ -f "$MODEL_PATH_CHECK" ] && [ -f "$SPEAKERS_PATH_CHECK" ]; then
    echo "✓ Base model files appear to exist in ${BASE_MODEL_DIR}. Skipping download."
else
    echo ">>> Downloading base XTTS-v2 model from ${BASE_MODEL_REPO}..."
    # Use python for slightly more robust download + rename logic
    python <<EOF
import os
from huggingface_hub import hf_hub_download

repo_id = "${BASE_MODEL_REPO}"
local_dir = "${BASE_MODEL_DIR}"
# List essential files
filenames = [
    "config.json", "model.pth", "dvae.pth",
    "mel_stats.pth", "speakers_xtts.pth", "vocab.json"
]
missing_files = False
for filename in filenames:
    target_path = os.path.join(local_dir, filename)
    if not os.path.exists(target_path):
        print(f"--- Downloading {filename}...")
        try:
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, local_dir_use_symlinks=False, resume_download=True)
        except Exception as e:
             # Handle potential Vocab.json case specifically
            if filename == "vocab.json":
                 print(f"Warning: Could not download vocab.json: {e}. Trying Vocab.json...")
                 try:
                      hf_hub_download(repo_id=repo_id, filename="Vocab.json", local_dir=local_dir, local_dir_use_symlinks=False, resume_download=True)
                      source_path = os.path.join(local_dir, "Vocab.json")
                      if os.path.exists(source_path):
                           os.rename(source_path, target_path)
                           print("--- Renamed Vocab.json to vocab.json")
                      else:
                           print(f"CRITICAL: Failed to download {filename} or Vocab.json.")
                           missing_files = True
                 except Exception as e2:
                      print(f"CRITICAL: Failed to download {filename} and Vocab.json: {e2}")
                      missing_files = True
            else:
                 print(f"CRITICAL: Failed to download essential file {filename}: {e}")
                 missing_files = True
    else:
        print(f"--- Skipping {filename}, already exists.")

if missing_files:
    print("Error: Essential base model files failed to download.")
    exit(1)

print("Base model download/check finished.")
EOF
    echo "✓ Base model files downloaded to ${BASE_MODEL_DIR}."
fi

# --- Download User Dataset ---
DATASET_MARKER_PATH="${DATASET_DOWNLOAD_DIR}/data"
if [ -d "$DATASET_MARKER_PATH" ] && [ "$(ls -A "$DATASET_MARKER_PATH")" ]; then
     echo "✓ User dataset appears to exist in ${DATASET_DOWNLOAD_DIR}. Skipping download."
else
    echo ">>> Downloading user dataset ${USER_DATASET_ID}..."
    huggingface-cli download --repo-type dataset "$USER_DATASET_ID" --local-dir "$DATASET_DOWNLOAD_DIR" --local-dir-use-symlinks False --resume-download
    echo "✓ User dataset downloaded to ${DATASET_DOWNLOAD_DIR}."
fi

# --- Check and Run Data Preprocessing ---
METADATA_PATH="${DATASET_PROCESSED_DIR}/metadata.csv"
WAVS_PATH="${DATASET_PROCESSED_DIR}/wavs"

echo ">>> Checking for existing preprocessed data..."
if [ -f "$METADATA_PATH" ] && [ -d "$WAVS_PATH" ] && [ "$(ls -A "$WAVS_PATH")" ]; then
    echo "✓ Preprocessed data (metadata.csv and non-empty wavs dir) found in ${DATASET_PROCESSED_DIR}. Skipping preprocessing."
else
    echo "--- Preprocessed data not found or incomplete. Running preprocessing script..."
    # Assuming preprocess_data.py is in the same directory as this shell script
    python preprocess_data.py --raw_data_dir "$DATASET_DOWNLOAD_DIR" --processed_data_dir "$DATASET_PROCESSED_DIR"
    echo "✓ Data preprocessing finished. Processed data in ${DATASET_PROCESSED_DIR}."
fi

# --- Run Training ---
echo ">>> Starting model training..."
# Assuming train_model.py is in the same directory
# Pass necessary paths as arguments
# Set CUDA_VISIBLE_DEVICES - adjust "0" if you want to use a different GPU
CUDA_VISIBLE_DEVICES="0" python train_model.py \
    --base_model_dir "$BASE_MODEL_DIR" \
    --processed_data_dir "$DATASET_PROCESSED_DIR" \
    --output_dir "$TRAINING_OUTPUT_DIR"

# Find the specific run directory created by the trainer
TRAINING_RUN_NAME="GPT_XTTS_Sangeeta_Hindi_FT" # Ensure this matches train_model.py
LATEST_RUN_DIR=$(ls -td "${TRAINING_OUTPUT_DIR}/${TRAINING_RUN_NAME}"*/ | head -1)

if [ -z "$LATEST_RUN_DIR" ]; then
  echo "Error: Could not find the latest training run directory matching pattern '${TRAINING_OUTPUT_DIR}/${TRAINING_RUN_NAME}*/'"
  # Attempt generic find if specific name failed
  LATEST_RUN_DIR=$(ls -td "${TRAINING_OUTPUT_DIR}"/*/ | head -1)
  if [ -z "$LATEST_RUN_DIR" ]; then
     echo "Error: Could not find any training run directory in ${TRAINING_OUTPUT_DIR}"
     exit 1
  else
     echo "Warning: Found a generic run directory: ${LATEST_RUN_DIR}. Proceeding with upload."
  fi
fi

# Clean potential trailing slash from LATEST_RUN_DIR
LATEST_RUN_DIR=$(echo "$LATEST_RUN_DIR" | sed 's:/*$::')

echo "✓ Training finished. Output files are in ${LATEST_RUN_DIR}"

# --- Ensure all required files exist in the output directory ---
echo ">>> Verifying and preparing model artifacts for upload..."
# Make sure we have all the required files in the latest run directory
python <<EOF
import os
import shutil
import glob

source_dir = "${BASE_MODEL_DIR}"
target_dir = "${LATEST_RUN_DIR}"

# Files that should be present in the upload
required_files = [
    "config.json",       # Should be generated during training
    "model.pth",         # Rename from best_model_*.pth if needed
    "dvae.pth",          # Copy from base model
    "mel_stats.pth",     # Copy from base model 
    "speakers_xtts.pth", # Copy from base model
    "vocab.json"         # Copy from base model
]

# First check for best model to use as model.pth
best_model_files = glob.glob(os.path.join(target_dir, "best_model*.pth"))
if best_model_files:
    best_model = max(best_model_files, key=os.path.getmtime)
    model_path = os.path.join(target_dir, "model.pth")
    if not os.path.exists(model_path):
        print(f"Copying {os.path.basename(best_model)} to model.pth")
        shutil.copy2(best_model, model_path)
    else:
        print("model.pth already exists")
else:
    print("Warning: No best_model*.pth file found")

# Copy any missing files from base model directory
for file in required_files:
    target_path = os.path.join(target_dir, file)
    if not os.path.exists(target_path):
        source_path = os.path.join(source_dir, file)
        if os.path.exists(source_path):
            print(f"Copying {file} from base model to output directory")
            shutil.copy2(source_path, target_path)
        else:
            print(f"Warning: {file} not found in base model or output directory")
    else:
        print(f"File {file} already exists in output directory")

# Verify all required files exist
missing = False
for file in required_files:
    file_path = os.path.join(target_dir, file)
    if not os.path.exists(file_path):
        print(f"ERROR: Required file {file} missing from {target_dir}")
        missing = True
    else:
        print(f"✓ Found {file}")

if missing:
    print("Warning: Some required files are missing!")
else:
    print("✓ All required files present for model upload")
EOF

# --- Push Model to Hugging Face Hub ---
echo ">>> Uploading trained model artifacts from ${LATEST_RUN_DIR} to ${TARGET_HF_MODEL_REPO}..."
python <<EOF
from huggingface_hub import HfApi, upload_folder
import os

local_folder_path = "${LATEST_RUN_DIR}"
target_repo_id = "${TARGET_HF_MODEL_REPO}"
commit_msg = "Upload fine-tuned Sangeeta Hindi XTTS model artifacts"

print(f"Uploading contents of {local_folder_path} to {target_repo_id}...")

if not os.path.isdir(local_folder_path):
    print(f"Error: Local folder to upload does not exist: {local_folder_path}")
    exit(1)

try:
    # Ensure repository exists, create if not
    api = HfApi()
    api.create_repo(repo_id=target_repo_id, repo_type='model', exist_ok=True)
    print(f"Repository {target_repo_id} ensured.")

    # Upload the folder contents
    upload_folder(
        folder_path=local_folder_path,
        repo_id=target_repo_id,
        repo_type='model',
        commit_message=commit_msg,
        ignore_patterns=["*.log", "tensorboard/*"]  # Ignore logs and tensorboard files
    )
    print("✓ Upload successful!")
except Exception as e:
    print(f"Error uploading to Hugging Face Hub: {e}")
    # exit(1) # Decide if upload failure should stop the script
EOF

echo ">>> All steps completed! <<<"