import os
import argparse
import json
# Set TORCH_HOME cache directory *before* importing torch or TTS
cache_dir = os.path.join(os.getcwd(), '.cache', 'torch')
os.makedirs(cache_dir, exist_ok=True)
os.environ['TORCH_HOME'] = cache_dir
os.environ['HF_HOME'] = os.path.join(os.getcwd(), '.cache', 'huggingface') # Also set HF cache

from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig

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
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None
    
    # Training Parameters (adjusted based on feedback)
    OPTIMIZER_WD_ONLY_ON_WEIGHTS = True
    START_WITH_EVAL = False
    BATCH_SIZE = 8
    # Reduced from 64 to 16 as recommended for better training dynamics
    GRAD_ACUMM_STEPS = 16  # Effective batch size = 8 * 16 = 128 (down from 512)
    EPOCHS = 15
    # Reduced from 5e-6 to 1e-6 as recommended for better convergence
    LR = 1e-6  
    
    # Dataset Configuration
    LANGUAGE = "hi"
    METADATA_FILE_PATH = "metadata.csv"
    WAVS_DIR_PATH = os.path.join(processed_data_dir, "wavs")
    
    # Find reference speaker wav
    SPEAKER_REFERENCE_WAV = os.path.join(WAVS_DIR_PATH, "1.wav")  # First try "1.wav"
    if not os.path.exists(SPEAKER_REFERENCE_WAV):
        found_wav = False
        for fname in os.listdir(WAVS_DIR_PATH):
            if fname.lower().endswith(".wav"):
                SPEAKER_REFERENCE_WAV = os.path.join(WAVS_DIR_PATH, fname)
                print(f"Using speaker reference WAV: {fname}")
                found_wav = True
                break
        if not found_wav:
            raise FileNotFoundError(f"No reference speaker .wav file found in {WAVS_DIR_PATH}. Cannot proceed.")
    
    config_dataset = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name="Sangeeta_Hindi",
        path=processed_data_dir,
        meta_file_train=METADATA_FILE_PATH,
        language=LANGUAGE,
    )
    DATASETS_CONFIG_LIST = [config_dataset]
    
    # --- Model File Paths ---
    DVAE_CHECKPOINT = os.path.join(base_model_dir, "dvae.pth")
    MEL_NORM_FILE = os.path.join(base_model_dir, "mel_stats.pth")
    TOKENIZER_FILE = os.path.join(base_model_dir, "vocab.json")
    XTTS_CHECKPOINT = os.path.join(base_model_dir, "model.pth")
    CONFIG_FILE = os.path.join(base_model_dir, "config.json")
    SPEAKERS_FILE = os.path.join(base_model_dir, "speakers_xtts.pth")
    
    # --- Check essential model files ---
    essential_files = [DVAE_CHECKPOINT, MEL_NORM_FILE, TOKENIZER_FILE, 
                      XTTS_CHECKPOINT, CONFIG_FILE, SPEAKERS_FILE]
    for f_path in essential_files:
        if not os.path.exists(f_path):
            raise FileNotFoundError(f"Essential base model file not found: {f_path}")
        print(f"Found: {f_path}")
    
    # Load the original config to check GPT layers
    base_config = {}
    try:
        with open(CONFIG_FILE, 'r') as f:
            base_config = json.load(f)
        gpt_layers = base_config.get("model_args", {}).get("gpt_layers", 30)
        print(f"Using {gpt_layers} GPT layers from base model config")
    except Exception as e:
        print(f"Warning: Could not determine GPT layers from config, defaulting to 30: {e}")
        gpt_layers = 30
    
    # --- Model and Training Config ---
    # CRITICAL: Use the same sample rates as base model: internal 22050, output 24000
    model_args = GPTArgs(
        # Original conditioning lengths matching base model expectation
        # max_conditioning_length=264600,  # ~12 seconds @ 22050Hz
        # min_conditioning_length=66150,   # ~3 seconds @ 22050Hz
        min_conditioning_length = 66150,
        max_conditioning_length = 132300,
        max_wav_length=576000,           # ~24 seconds @ 24kHz
        max_text_length=400,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        speakers_file=SPEAKERS_FILE,  # Added to support speaker handling
        # IMPORTANT: Use the SAME number of GPT layers as the base model
        gpt_layers=gpt_layers,
        # GPT specific tokens from XTTS v2 config
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        # CRITICAL: Enable perceiver resampler for better handling of conditioning
        gpt_use_perceiver_resampler=True,
        # Other GPT model architecture params
        gpt_n_model_channels=1024,
        gpt_n_heads=16,
        kv_cache=False,  # No KV cache during training
    )
    
    # Set audio config with correct sample rates - internal 22050, output 24000
    audio_config = XttsAudioConfig(
        sample_rate=22050,          # Internal processing rate
        dvae_sample_rate=22050,     # DVAE expects this rate
        output_sample_rate=24000    # Final output rate
    )
    
    config = GPTTrainerConfig(
        output_path=output_dir,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="Fine-tuning XTTS-v2 GPT encoder for Sangeeta Hindi voice.",
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=4,
        eval_split_max_size=256,
        eval_split_size=0.01,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=5000,
        save_n_checkpoints=2,
        save_checkpoints=True,
        save_best_after=10000,
        print_eval=True,
        epochs=EPOCHS,
        # CRITICAL: Enable mixed precision with bf16 for better performance & stability
        mixed_precision=True,
        precision="bf16",  # Use bf16 (better than fp16) if your GPU supports it
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=LR,
        lr_scheduler="MultiStepLR",
        # Adjusted milestones based on effective batch size
        lr_scheduler_params={"milestones": [5000, 10000], "gamma": 0.5},
        # Add this to make loss comparison clearer between train and eval logs
        weighted_loss_attrs={"loss_text_ce": 0.01, "loss_mel_ce": 1.0},
        # Add this to track mel-CE (spectral quality) as target metric for best model
        target_loss="loss_mel_ce",
        test_sentences=[
            {
                "text": "नमस्ते, यह एक परीक्षण वाक्य है।",
                "speaker_wav": [SPEAKER_REFERENCE_WAV],
                "language": LANGUAGE,
            },
            {
                "text": "मुझे यह आवाज़ बहुत पसंद है।",
                "speaker_wav": [SPEAKER_REFERENCE_WAV],
                "language": LANGUAGE,
            },
        ],
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
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=output_dir,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    print("✓ Trainer initialized.")
    
    # Start training
    print("Starting training fitting process...")
    trainer.fit()
    print("✓ Training fitting process completed.")
    
    # After training, copy necessary files for model distribution
    print("Ensuring all necessary files are in the output directory for model distribution...")
    final_output_dir = os.path.join(output_dir, config.run_name + "*")
    try:
        import glob
        import shutil
        
        run_dirs = glob.glob(final_output_dir)
        if run_dirs:
            latest_run_dir = max(run_dirs, key=os.path.getmtime)
            print(f"Latest run directory: {latest_run_dir}")
            
            # Copy essential files from base model if they don't exist in the run directory
            for file_name in ["dvae.pth", "mel_stats.pth", "vocab.json", "speakers_xtts.pth"]:
                target_path = os.path.join(latest_run_dir, file_name)
                if not os.path.exists(target_path):
                    source_path = os.path.join(base_model_dir, file_name)
                    if os.path.exists(source_path):
                        shutil.copy2(source_path, target_path)
                        print(f"Copied {file_name} to run directory")
                    else:
                        print(f"Warning: Could not find {file_name} in base model directory")
            
            # Rename best model to model.pth if it exists
            best_model_files = glob.glob(os.path.join(latest_run_dir, "best_model*.pth"))
            if best_model_files:
                best_model = max(best_model_files, key=os.path.getmtime)
                target_path = os.path.join(latest_run_dir, "model.pth")
                if not os.path.exists(target_path):
                    shutil.copy2(best_model, target_path)
                    print(f"Created model.pth from {os.path.basename(best_model)}")
            
            print("File preparation complete!")
        else:
            print(f"Warning: No run directories found matching {final_output_dir}")
    except Exception as e:
        print(f"Warning: Error during file preparation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Fine-tune XTTS model.")
    parser.add_argument("--base_model_dir", required=True, help="Directory containing the base XTTS model files.")
    parser.add_argument("--processed_data_dir", required=True, help="Directory containing the processed wavs and metadata.")
    parser.add_argument("--output_dir", required=True, help="Directory to save training outputs (checkpoints, logs).")
    args = parser.parse_args()
    train(args.base_model_dir, args.processed_data_dir, args.output_dir)