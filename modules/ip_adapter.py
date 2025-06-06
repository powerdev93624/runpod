import os
import torch
from PIL import Image
import numpy as np

class IPAdapter:
    def __init__(self, device="cuda", model_path=None):
        """Initialize IP-Adapter
        
        Args:
            device (str): Device to run on ('cuda' or 'cpu')
            model_path (str): Path to model directory
        """
        self.device = device
        self.model_path = model_path
        self.model = None
        self.image_encoder = None
        
        # Initialize model
        self.initialize()
    
    def initialize(self):
        """Initialize IP-Adapter model"""
        try:
            from transformers import CLIPVisionModelWithProjection
            from diffusers.utils import load_image
            
            # Load the IP-Adapter models
            if self.model_path:
                ip_ckpt = os.path.join(self.model_path, "ip-adapter_sd15.bin")
                image_encoder_path = os.path.join(self.model_path, "clip-vit-large-patch14")
            else:
                ip_ckpt = "ip-adapter_sd15.bin"
                image_encoder_path = "clip-vit-large-patch14"
            
            # Load the image encoder
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                image_encoder_path,
                torch_dtype=torch.float16
            ).to(self.device)
            
            # The actual model is loaded as part of the pipeline during generate
            self.ip_ckpt = ip_ckpt
            print("IP-Adapter initialized successfully")
            
        except Exception as e:
            print(f"Error initializing IP-Adapter: {e}")
            print("IP-Adapter will not be available.")
    
    def preprocess_image(self, image):
        """Preprocess image for IP-Adapter
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        from torchvision import transforms
        
        # Define preprocessing for the image
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], 
                                [0.26862954, 0.26130258, 0.27577711]),
        ])
        
        image_tensor = preprocess(image).unsqueeze(0).to(self.device, torch.float16)
        return image_tensor
    
    def generate(self, reference_image, prompt, negative_prompt=None, 
                 scale=0.8, pipe=None, **kwargs):
        """Generate an image using IP-Adapter
        
        Args:
            reference_image (PIL.Image): Reference image for IP-Adapter
            prompt (str): Text prompt
            negative_prompt (str, optional): Negative prompt
            scale (float): Scale for IP-Adapter conditioning
            pipe (StableDiffusionPipeline): Base pipeline to use
            **kwargs: Additional arguments for the pipeline
            
        Returns:
            PIL.Image: Generated image
        """
        if self.image_encoder is None or pipe is None:
            print("IP-Adapter not initialized or pipeline not provided")
            return None
        
        try:
            from diffusers.utils import load_image
            from ip_adapter import IPAdapterPlus
            
            # Preprocess the reference image
            image_tensor = self.preprocess_image(reference_image)
            
            # Get image embeddings
            with torch.no_grad():
                image_embeds = self.image_encoder(image_tensor).image_embeds
            
            # Create the IP-Adapter
            ip_adapter = IPAdapterPlus(
                pipe, 
                self.ip_ckpt, 
                image_embeds, 
                device=self.device
            )
            
            # Generate the image
            output = ip_adapter.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                scale=scale,
                **kwargs
            )
            
            return output
            
        except Exception as e:
            print(f"Error generating with IP-Adapter: {e}")
            print("Falling back to standard generation without IP-Adapter")
            
            # Fallback to standard generation
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                **kwargs
            ).images[0]
            
            return output 