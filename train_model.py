import os
import argparse

# Set TORCH_HOME cache directory *before* importing torch or TTS
# This can prevent issues in environments with restricted home directories
cache_dir = os.path.join(os.getcwd(), '.cache', 'torch')
os.makedirs(cache_dir, exist_ok=True)
os.environ['TORCH_HOME'] = cache_dir
os.environ['HF_HOME'] = os.path.join(os.getcwd(), '.cache', 'huggingface') # Also set HF cache


from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import  XttsAudioConfig

# from TTS.utils.manage import ModelManager # Not needed if paths are passed directly


def train(base_model_dir, processed_data_dir, output_dir):
    """
    Sets up and runs the XTTS fine-tuning process.
    """
    print("Starting training script setup...")
    print(f"Base model directory: {base_model_dir}")
    print(f"Processed data directory: {processed_data_dir}")
    print(f"Training output directory: {output_dir}")

    # Logging parameters
    RUN_NAME = "GPT_XTTS_Sangeeta_Hindi_FT"
    PROJECT_NAME = "XTTS_Finetuning"
    DASHBOARD_LOGGER = "tensorboard"  # Or "wandb"
    LOGGER_URI = None

    # Training Parameters (adjust based on your GPU capability)
    OPTIMIZER_WD_ONLY_ON_WEIGHTS = True # Keep True for single GPU
    START_WITH_EVAL = True
    # Lower batch size if facing memory issues, increase GRAD_ACUMM_STEPS accordingly
    BATCH_SIZE = 3
    GRAD_ACUMM_STEPS = 84 # Effective batch size = 3 * 84 = 252
    EPOCHS = 100 # Adjust number of training epochs
    LR = 5e-6

    # Dataset Configuration
    LANGUAGE = "hi"
    METADATA_FILE_PATH = os.path.join(processed_data_dir, "metadata.csv")
    WAVS_DIR_PATH = os.path.join(processed_data_dir, "wavs")
    # Ensure reference speaker wav exists
    SPEAKER_REFERENCE_WAV = os.path.join(WAVS_DIR_PATH, "000001.wav") # Example, ensure this exists
    if not os.path.exists(SPEAKER_REFERENCE_WAV):
         # Try finding *any* wav file if the default doesn't exist
         found_wav = False
         for fname in os.listdir(WAVS_DIR_PATH):
              if fname.lower().endswith(".wav"):
                   SPEAKER_REFERENCE_WAV = os.path.join(WAVS_DIR_PATH, fname)
                   print(f"Warning: Default reference {SPEAKER_REFERENCE_WAV} not found. Using first found wav: {fname}")
                   found_wav = True
                   break
         if not found_wav:
              raise FileNotFoundError(f"No reference speaker .wav file found in {WAVS_DIR_PATH}. Cannot proceed.")


    config_dataset = BaseDatasetConfig(
        formatter="ljspeech", # Using ljspeech structure with relative paths in metadata
        dataset_name="Sangeeta_Hindi",
        path=processed_data_dir, # Base path where metadata.csv is located
        meta_file_train=METADATA_FILE_PATH,
        language=LANGUAGE,
    )
    DATASETS_CONFIG_LIST = [config_dataset]

    # --- Model File Paths (relative to base_model_dir) ---
    DVAE_CHECKPOINT = os.path.join(base_model_dir, "dvae.pth")
    MEL_NORM_FILE = os.path.join(base_model_dir, "mel_stats.pth")
    TOKENIZER_FILE = os.path.join(base_model_dir, "vocab.json")
    XTTS_CHECKPOINT = os.path.join(base_model_dir, "model.pth")
    CONFIG_FILE = os.path.join(base_model_dir, "config.json")
    SPEAKERS_FILE = os.path.join(base_model_dir, "speakers_xtts.pth")

    # --- Check if essential model files exist ---
    essential_files = [DVAE_CHECKPOINT, MEL_NORM_FILE, TOKENIZER_FILE, XTTS_CHECKPOINT, CONFIG_FILE, SPEAKERS_FILE]
    for f_path in essential_files:
        if not os.path.exists(f_path):
            raise FileNotFoundError(f"Essential base model file not found: {f_path}. Please ensure downloads were successful.")
        print(f"Found: {f_path}")

    # --- Model and Training Config ---
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs @ 22050 (adjust if needed for 24k?) -> 144000 for 6s @ 24k
        min_conditioning_length=72000,   # 3 secs @ 24k
        # Let's try adjusting conditioning lengths for 24kHz
        # max_conditioning_length=144000,
        # min_conditioning_length=72000,
        # Keep original values for now, model might internally handle this based on config sample rate
        max_wav_length=576000,           # ~24 seconds @ 24kHz (adjust based on your longest sample)
        max_text_length=400,             # Max text characters
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT, # Base model to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        # speakers_file=SPEAKERS_FILE,     # Provide path to speakers file
        # speaker_embedding_channels=1024, # From base config inspection
        # use_speaker_embedding=True,      # Important for fine-tuning speaker characteristics
        # GPT specific tokens from XTTS v2 config
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        # Other GPT params (usually okay to keep defaults matching base model)
        gpt_n_model_channels=1024,
        gpt_n_heads=16,
        gpt_layers=20,
        kv_cache=False, # Training specific
    )

    audio_config = XttsAudioConfig(
        sample_rate=24000,          # Input audio data is 24kHz
        dvae_sample_rate=24000,     # Assume DVAE in v2 works with 24kHz
        output_sample_rate=24000    # Standard XTTS v2 output rate
    )

    config = GPTTrainerConfig(
        output_path=output_dir, # Use the directory passed from shell script
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="Fine-tuning XTTS-v2 GPT encoder for Sangeeta Hindi voice.",
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=4,       # Adjust based on CPU cores
        eval_split_max_size=256,
        eval_split_size=0.01,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=5000,             # Save checkpoint frequency
        save_n_checkpoints=2,       # Keep last 2 checkpoints + best_model
        save_checkpoints=True,
        save_best_after=10000,      # Start saving best model after 10k steps
        print_eval=True,
        epochs=EPOCHS,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=LR,
        lr_scheduler="MultiStepLR",
        # Adjust milestones based on total expected steps (samples/eff_batch_size * epochs)
        # Example: 14558 samples / 252 eff_batch_size = ~58 steps/epoch
        # 100 epochs * 58 steps/epoch = ~5800 total steps (approx)
        # Milestones should be within this range
        lr_scheduler_params={"milestones": [2000, 4000], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {
                "text": "नमस्ते, यह एक परीक्षण वाक्य है।",
                "speaker_wav": [SPEAKER_REFERENCE_WAV], # List format
                "language": LANGUAGE,
            },
            {
                "text": "मुझे यह आवाज़ बहुत पसंद है।",
                "speaker_wav": [SPEAKER_REFERENCE_WAV],
                "language": LANGUAGE,
            },
        ],
        # xtts_config_path=CONFIG_FILE, # Provide path to base model config
        speakers_file_path=SPEAKERS_FILE, # Provide path to speakers file
    )

    # Init the model trainer from config
    print("Initializing GPTTrainer...")
    model = GPTTrainer.init_from_config(config)
    print("✓ GPTTrainer initialized.")

    # Load training samples
    print("Loading training and evaluation samples...")
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    print(f"✓ Loaded {len(train_samples)} training samples and {len(eval_samples)} evaluation samples.")

    # Init the PyTorch Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        TrainerArgs(
            restore_path=None, # Base model loaded internally by GPTTrainer
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
            # deepspeed_config_path= "path/to/deepspeed_config.json" # Add if using deepspeed
        ),
        config,
        output_path=output_dir, # Use the specific output path for this run
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    print("✓ Trainer initialized.")

    # Start training 🚀
    print("Starting training fitting process...")
    trainer.fit()
    print("✓ Training fitting process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Fine-tune XTTS model.")
    parser.add_argument("--base_model_dir", required=True, help="Directory containing the base XTTS model files.")
    parser.add_argument("--processed_data_dir", required=True, help="Directory containing the processed wavs and metadata.")
    parser.add_argument("--output_dir", required=True, help="Directory to save training outputs (checkpoints, logs).")
    args = parser.parse_args()

    train(args.base_model_dir, args.processed_data_dir, args.output_dir)