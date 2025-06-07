import os
import argparse
from diffusers import StableDiffusionPipeline
from huggingface_hub import snapshot_download

def download_model(model_id, output_dir, use_safetensors=True, revision=None):
    """
    Download and save a model from Hugging Face Hub.
    
    Args:
        model_id (str): The Hugging Face model ID (e.g., "runwayml/stable-diffusion-v1-5")
        output_dir (str): Directory to save the model
        use_safetensors (bool): Whether to use safetensors format
        revision (str, optional): The specific model version to use
    """
    print(f"Downloading model: {model_id}")
    print(f"Output directory: {output_dir}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Method 1: Using StableDiffusionPipeline to download and save
    try:
        print("Downloading using StableDiffusionPipeline...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            revision=revision,
            use_safetensors=use_safetensors
        )
        pipeline.save_pretrained(output_dir)
        print(f"Model successfully downloaded and saved to {output_dir}")
        return True
    except Exception as e:
        print(f"Error downloading with StableDiffusionPipeline: {e}")
        
        # Method 2: Fallback to snapshot_download
        try:
            print("Falling back to snapshot_download...")
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                local_dir=output_dir,
                local_dir_use_symlinks=False
            )
            print(f"Model successfully downloaded and saved to {output_dir}")
            return True
        except Exception as e2:
            print(f"Error downloading with snapshot_download: {e2}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Download and save a model from Hugging Face Hub")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="The Hugging Face model ID")
    parser.add_argument("--output_dir", type=str, default="./downloaded_model",
                        help="Directory to save the model")
    parser.add_argument("--use_safetensors", action="store_true", default=True,
                        help="Whether to use safetensors format")
    parser.add_argument("--revision", type=str, default=None,
                        help="The specific model version to use")
    
    args = parser.parse_args()
    
    success = download_model(args.model_id, args.output_dir, args.use_safetensors, args.revision)
    
    if success:
        print("Model download completed successfully!")
    else:
        print("Failed to download model.")

if __name__ == "__main__":
    main() 