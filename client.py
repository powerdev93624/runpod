import requests
import argparse
import json
import time
import os
from PIL import Image
import io

class RealisticVisionClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def health_check(self):
        """Check if the API server is running"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
        
    def generate_image(self, prompt, negative_prompt=None, width=768, height=768, 
                      steps=30, guidance_scale=7.5, seed=None):
        """Generate an image using the Realistic Vision API"""
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed
        }
        
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        print(f"Generating image with prompt: {prompt}")
        print("This may take a minute...")
        
        start_time = time.time()
        response = requests.post(f"{self.base_url}/generate", json=payload)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
            
        result = response.json()
        print(f"Image generated in {result['generation_time']:.2f} seconds")
        
        # Download the image
        image_url = result["image_url"]
        image_response = requests.get(f"{self.base_url}{image_url}")
        
        if image_response.status_code == 200:
            # Save the image
            filename = f"generated_{int(time.time())}.png"
            with open(filename, "wb") as f:
                f.write(image_response.content)
            print(f"Image saved as {filename}")
            
            # Open the image
            try:
                image = Image.open(io.BytesIO(image_response.content))
                image.show()
            except Exception as e:
                print(f"Could not display image: {e}")
                
            return filename
        else:
            print(f"Error downloading image: {image_response.status_code}")
            return None

def main():
    parser = argparse.ArgumentParser(description="Realistic Vision V6.0 B1 API Client")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="API server URL")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, help="Negative prompt")
    parser.add_argument("--width", type=int, default=768, help="Image width")
    parser.add_argument("--height", type=int, default=768, help="Image height")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    args = parser.parse_args()
    
    client = RealisticVisionClient(args.url)
    
    # Check if the server is running
    try:
        health = client.health_check()
        print(f"Server status: {health['status']}")
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return
    
    # Generate the image
    client.generate_image(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed
    )

if __name__ == "__main__":
    main() 