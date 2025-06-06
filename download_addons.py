#!/usr/bin/env python
import os
import argparse
import requests
import tqdm
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from diffusers import ControlNetModel, AutoencoderKL
from transformers import CLIPVisionModelWithProjection, CLIPTextModel, CLIPTokenizer


def download_file(url, dest_folder, filename=None):
    """Download a file from a URL to a destination folder"""
    if filename is None:
        filename = os.path.basename(url)
    
    os.makedirs(dest_folder, exist_ok=True)
    filepath = os.path.join(dest_folder, filename)
    
    # Skip if file already exists
    if os.path.exists(filepath):
        print(f"File already exists: {filepath}")
        return filepath
    
    # Download file
    print(f"Downloading {url} to {filepath}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f, tqdm.tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
    
    return filepath


def download_controlnet_models(output_dir):
    """Download ControlNet models from Huggingface"""
    controlnet_models = {
        "canny": "lllyasviel/sd-controlnet-canny",
        "depth": "lllyasviel/sd-controlnet-depth",
        "pose": "lllyasviel/sd-controlnet-openpose",
        "seg": "lllyasviel/sd-controlnet-seg"
    }
    
    base_dir = os.path.join(output_dir, "controlnet")
    os.makedirs(base_dir, exist_ok=True)
    
    for name, repo_id in controlnet_models.items():
        print(f"Downloading ControlNet model: {name}")
        output_path = os.path.join(base_dir, f"control_sd15_{name}")
        
        try:
            # Download the model
            ControlNetModel.from_pretrained(repo_id, torch_dtype=torch.float16).save_pretrained(output_path)
            print(f"Downloaded {name} ControlNet to {output_path}")
        except Exception as e:
            print(f"Error downloading {name} ControlNet: {e}")


def download_instantid(output_dir):
    """Download InstantID models from Huggingface"""
    base_dir = os.path.join(output_dir, "instantid")
    os.makedirs(base_dir, exist_ok=True)
    
    try:
        # Download ControlNet
        print("Downloading InstantID ControlNet")
        controlnet_path = os.path.join(base_dir, "ControlNetModel")
        snapshot_download(
            repo_id="InstantID/InstantID", 
            allow_patterns=["ControlNetModel/*"], 
            local_dir=base_dir
        )
        
        # Download adapter
        print("Downloading InstantID adapter")
        hf_hub_download(
            repo_id="InstantID/InstantID", 
            filename="ip-adapter-instantid.bin",
            local_dir=base_dir
        )
        
        print(f"Downloaded InstantID to {base_dir}")
    except Exception as e:
        print(f"Error downloading InstantID: {e}")


def download_ip_adapter(output_dir):
    """Download IP-Adapter models from Huggingface"""
    base_dir = os.path.join(output_dir, "ip-adapter")
    os.makedirs(base_dir, exist_ok=True)
    
    try:
        # Download adapter
        print("Downloading IP-Adapter")
        hf_hub_download(
            repo_id="h94/IP-Adapter", 
            filename="models/ip-adapter_sd15.bin",
            local_dir=base_dir
        )
        
        # Download CLIP vision model
        print("Downloading CLIP vision model")
        vision_path = os.path.join(base_dir, "clip-vit-large-patch14")
        CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14", 
            torch_dtype=torch.float16
        ).save_pretrained(vision_path)
        
        print(f"Downloaded IP-Adapter to {base_dir}")
    except Exception as e:
        print(f"Error downloading IP-Adapter: {e}")


def download_insightface(output_dir):
    """Download InsightFace models"""
    base_dir = os.path.join(output_dir, "insightface")
    os.makedirs(base_dir, exist_ok=True)
    
    try:
        # Download Buffalo L model
        print("Downloading InsightFace Buffalo L model")
        # InsightFace will automatically download its models on first use
        # Create placeholder to indicate we should use this
        with open(os.path.join(base_dir, ".download_on_use"), "w") as f:
            f.write("Models will be downloaded when first used")
        
        print(f"InsightFace will download models on first use to {base_dir}")
    except Exception as e:
        print(f"Error setting up InsightFace: {e}")


def download_face_restoration(output_dir):
    """Download face restoration models"""
    base_dir = os.path.join(output_dir, "face_restoration")
    codeformer_dir = os.path.join(output_dir, "codeformer")
    gfpgan_dir = os.path.join(output_dir, "gfpgan")
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(codeformer_dir, exist_ok=True)
    os.makedirs(gfpgan_dir, exist_ok=True)
    
    try:
        # Download CodeFormer
        print("Downloading CodeFormer model")
        hf_hub_download(
            repo_id="sczhou/codeformer", 
            filename="codeformer.pth",
            local_dir=codeformer_dir
        )
        
        # Download face detection model
        print("Downloading face detection model")
        hf_hub_download(
            repo_id="sczhou/codeformer", 
            filename="detection_Resnet50_Final.pth",
            local_dir=codeformer_dir
        )
        
        # Download face parsing model
        print("Downloading face parsing model")
        hf_hub_download(
            repo_id="sczhou/codeformer", 
            filename="parsing_parsenet.pth",
            local_dir=codeformer_dir
        )
        
        # Copy the same detection models to GFPGAN directory
        for filename in ["detection_Resnet50_Final.pth", "parsing_parsenet.pth"]:
            src = os.path.join(codeformer_dir, filename)
            dst = os.path.join(gfpgan_dir, filename)
            if os.path.exists(src) and not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)
        
        # Download GFPGAN
        print("Downloading GFPGAN model")
        hf_hub_download(
            repo_id="TencentARC/GFPGAN", 
            filename="GFPGANv1.4.pth",
            local_dir=gfpgan_dir
        )
        
        print(f"Downloaded face restoration models to {codeformer_dir} and {gfpgan_dir}")
    except Exception as e:
        print(f"Error downloading face restoration models: {e}")


def download_reactor(output_dir):
    """Download ReActor face swapping model"""
    base_dir = os.path.join(output_dir, "reactor")
    os.makedirs(base_dir, exist_ok=True)
    
    try:
        # Download ReActor model
        print("Downloading ReActor model")
        hf_hub_download(
            repo_id="JiahuiYu/ReActor", 
            filename="reactor.pth",
            local_dir=base_dir
        )
        
        print(f"Downloaded ReActor to {base_dir}")
    except Exception as e:
        print(f"Error downloading ReActor: {e}")


def download_textual_inversion(output_dir):
    """Download example textual inversion embeddings"""
    base_dir = os.path.join(output_dir, "textual_inversion")
    os.makedirs(base_dir, exist_ok=True)
    
    # List of popular textual inversion embeddings
    embeddings = [
        {"repo_id": "sd-concepts-library/cat-toy", "filename": "learned_embeds.bin", "name": "cat-toy.bin"},
        {"repo_id": "sd-concepts-library/midjourney-style", "filename": "learned_embeds.bin", "name": "midjourney-style.bin"},
        {"repo_id": "sd-concepts-library/ghibli-style", "filename": "learned_embeds.bin", "name": "ghibli-style.bin"},
        {"repo_id": "sd-concepts-library/disco-diffusion-style", "filename": "learned_embeds.bin", "name": "disco-diffusion-style.bin"},
        {"repo_id": "sd-concepts-library/arcane-style", "filename": "learned_embeds.bin", "name": "arcane-style.bin"},
    ]
    
    for embedding in embeddings:
        try:
            print(f"Downloading textual inversion embedding: {embedding['name']}")
            hf_hub_download(
                repo_id=embedding["repo_id"], 
                filename=embedding["filename"],
                local_dir=base_dir,
                local_filename=embedding["name"]
            )
        except Exception as e:
            print(f"Error downloading {embedding['name']}: {e}")
    
    print(f"Downloaded textual inversion embeddings to {base_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download addon models for Realistic Vision API")
    parser.add_argument("--output_dir", type=str, default="./addons", help="Output directory for downloaded models")
    parser.add_argument("--addons", type=str, nargs="+", 
                        choices=["all", "controlnet", "instantid", "ip-adapter", "insightface", 
                                 "face-restoration", "reactor", "textual-inversion"],
                        default=["all"], 
                        help="Which addons to download")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download selected addons
    addons = args.addons
    if "all" in addons:
        addons = ["controlnet", "instantid", "ip-adapter", "insightface", 
                  "face-restoration", "reactor", "textual-inversion"]
    
    for addon in addons:
        if addon == "controlnet":
            download_controlnet_models(args.output_dir)
        elif addon == "instantid":
            download_instantid(args.output_dir)
        elif addon == "ip-adapter":
            download_ip_adapter(args.output_dir)
        elif addon == "insightface":
            download_insightface(args.output_dir)
        elif addon == "face-restoration":
            download_face_restoration(args.output_dir)
        elif addon == "reactor":
            download_reactor(args.output_dir)
        elif addon == "textual-inversion":
            download_textual_inversion(args.output_dir)
    
    print("All requested addons have been downloaded.")


if __name__ == "__main__":
    main() 