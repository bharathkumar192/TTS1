import os
import soundfile as sf
from datasets import load_dataset, Audio
import argparse

def preprocess(raw_data_dir, processed_data_dir):
    """
    Loads the dataset, saves audio at original 24kHz,
    and creates metadata.csv.
    """
    print(f"Starting preprocessing...")
    print(f"Raw data expected in: {raw_data_dir}")
    print(f"Processed data will be saved to: {processed_data_dir}")

    # Define paths within the processed directory
    audio_output_dir = os.path.join(processed_data_dir, "wavs")
    metadata_file_path = os.path.join(processed_data_dir, "metadata.csv")

    # Create Directories
    os.makedirs(audio_output_dir, exist_ok=True)

    # --- Load Dataset ---
    # Try loading directly from the downloaded directory
    try:
        print(f"Loading dataset from local directory: {raw_data_dir}")
        # Cast audio column on the fly to ensure it's interpreted correctly
        dataset = load_dataset(raw_data_dir, split="train").cast_column("audio", Audio(sampling_rate=24000))
        print(f"Dataset loaded successfully. Number of samples: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset from {raw_data_dir}: {e}")
        print("Please ensure the dataset was downloaded correctly.")
        return # Exit preprocessing if dataset fails to load

    # --- Process and Save ---
    print(f"Processing and saving data...")
    metadata = []
    expected_sr = 24000
    skipped_count = 0

    for i, item in enumerate(dataset):
        try:
            sentence = item['hindi_sentence']
            audio_data = item['audio']['array']
            actual_sr = item['audio']['sampling_rate']

            # Basic check for sample rate consistency
            if actual_sr != expected_sr:
                print(f"Warning: Item {i} has sample rate {actual_sr} Hz, expected {expected_sr} Hz. Skipping this item.")
                skipped_count += 1
                continue

            # Format filename
            file_id = item.get('sentence_id', f"{i:06d}")
            output_filename = f"{file_id}.wav"
            output_path = os.path.join(audio_output_dir, output_filename)

            # Save audio file AT ITS ORIGINAL (expected 24kHz) RATE
            sf.write(output_path, audio_data, actual_sr)
            cleaned_sentence = sentence.replace("|", " ")
            metadata.append(f"{file_id}|{cleaned_sentence}|Sangeeta") # Speaker name 'Sangeeta'

        except Exception as e:
            print(f"Error processing item {i}: {e}")
            skipped_count += 1
            continue

    # --- Write Metadata File ---
    if not metadata:
        print("Error: No valid metadata generated. Please check dataset and processing logic.")
        return

    print(f"Writing metadata for {len(metadata)} samples to {metadata_file_path}...")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} items due to errors or sample rate mismatch.")

    with open(metadata_file_path, 'w', encoding='utf-8') as f:
        for line in metadata:
            f.write(line + '\n')

    print("Data preparation complete (using original 24kHz sample rate).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Sangeeta Hindi dataset for XTTS training.")
    parser.add_argument("--raw_data_dir", required=True, help="Directory where the raw dataset was downloaded.")
    parser.add_argument("--processed_data_dir", required=True, help="Directory to save the processed wavs and metadata.")
    args = parser.parse_args()

    preprocess(args.raw_data_dir, args.processed_data_dir)