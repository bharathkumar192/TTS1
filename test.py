import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from huggingface_hub import hf_hub_download

# --- 1. Define Paths ---
MODEL_DIR = "/content/xtts_model_local" # Choose a local directory
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_model.pth") # Assuming you use best_model.pth
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.json")
SPEAKERS_PATH = os.path.join(MODEL_DIR, "speakers_xtts.pth")
SPEAKER_WAV_PATH = "/content/sangeeta_reference.wav"
OUTPUT_WAV_PATH = "/content/output_sangeeta_local.wav"

# Your fine-tuned model repo and the base repo
FT_MODEL_REPO = "bharathkumar1922001/TTS-1x"
BASE_MODEL_REPO = "coqui/XTTS-v2"

# --- 2. Download necessary files ---
os.makedirs(MODEL_DIR, exist_ok=True)
print("Downloading files...")
# Download from your fine-tuned repo
hf_hub_download(repo_id=FT_MODEL_REPO, filename="config.json", local_dir=MODEL_DIR, local_dir_use_symlinks=False)
hf_hub_download(repo_id=FT_MODEL_REPO, filename="best_model.pth", local_dir=MODEL_DIR, local_dir_use_symlinks=False)
hf_hub_download(repo_id=BASE_MODEL_REPO, filename="vocab.json", local_dir=MODEL_DIR, local_dir_use_symlinks=False)
hf_hub_download(repo_id=BASE_MODEL_REPO, filename="speakers_xtts.pth", local_dir=MODEL_DIR, local_dir_use_symlinks=False)
print("Downloads complete.")