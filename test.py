import os
import argparse
import torch
import torchaudio
import numpy as np
from IPython.display import Audio, display
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

def test_xtts_model(model_path, output_dir, speaker_wav, text_list=None, language="hi"):
    print("Starting XTTS model testing...")
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Speaker reference: {speaker_wav}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize a list of Hindi test sentences if none provided
    if text_list is None:
        text_list = [
            "नमस्ते, मैं आपकी फाइन-ट्यून्ड हिंदी वॉइस हूँ।",
            "मुझे आशा है कि आप मेरी आवाज़ पसंद करेंगे।",
            "यह मॉडल संगीता की आवाज़ पर फाइन-ट्यून किया गया है।",
            "क्या आप मुझे सुन सकते हैं? मैं हिंदी में बात कर रही हूँ।",
            "मैं एक्सटीटीएस मॉडल से जेनरेट की गई आवाज़ हूँ।",
        ]
    
    # Check if running in a notebook environment
    in_notebook = 'ipykernel' in sys.modules
    
    # Load the XTTS model
    print("Loading model...")
    config = XttsConfig()
    config.load_json(os.path.join(model_path, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config, 
        checkpoint_path=os.path.join(model_path, "model.pth"),
        vocab_path=os.path.join(model_path, "vocab.json"),
        eval=True
    )
    
    # Use CUDA if available
    if torch.cuda.is_available():
        model.cuda()
    
    print("Model loaded successfully.")
    
    # Process speaker reference
    print(f"Processing speaker reference: {speaker_wav}")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[speaker_wav]
    )
    
    # Synthesize each test sentence
    results = []
    for i, text in enumerate(text_list):
        print(f"\nSynthesizing text {i+1}/{len(text_list)}:")
        print(f"'{text}'")
        
        # Generate speech
        output_dict = model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.7  # Adjust if needed
        )
        
        # Save and play the generated audio
        output_path = os.path.join(output_dir, f"sample_{i+1}.wav")
        torchaudio.save(
            output_path,
            torch.tensor(output_dict["wav"]).unsqueeze(0),
            24000  # XTTS output sample rate
        )
        print(f"Saved to: {output_path}")
        
        # Play audio if in notebook
        if in_notebook:
            display(Audio(output_dict["wav"], rate=24000))
        
        results.append({
            "text": text,
            "output_path": output_path,
            "duration": len(output_dict["wav"]) / 24000
        })
    
    # Print summary
    print("\n--- Synthesis Summary ---")
    total_duration = sum(r["duration"] for r in results)
    print(f"Generated {len(results)} audio samples")
    print(f"Total audio duration: {total_duration:.2f} seconds")
    print(f"Output directory: {output_dir}")
    
    # Return results for further processing if needed
    return results

if __name__ == "__main__":
    import sys
    
    parser = argparse.ArgumentParser(description="Test a fine-tuned XTTS model")
    parser.add_argument("--model_path", required=True, help="Path to the model directory")
    parser.add_argument("--output_dir", default="./samples", help="Directory to save generated audio")
    parser.add_argument("--speaker_wav", required=True, help="Path to reference speaker WAV file")
    parser.add_argument("--text_file", help="Path to a file containing text samples to synthesize")
    parser.add_argument("--language", default="hi", help="Language code (default: hi)")
    
    args = parser.parse_args()
    
    # Load text samples from file if provided
    text_list = None
    if args.text_file and os.path.exists(args.text_file):
        with open(args.text_file, "r", encoding="utf-8") as f:
            text_list = [line.strip() for line in f.readlines() if line.strip()]
    
    # Run the test
    test_xtts_model(
        model_path=args.model_path,
        output_dir=args.output_dir,
        speaker_wav=args.speaker_wav,
        text_list=text_list,
        language=args.language
    )