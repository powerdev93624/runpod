import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import numpy as np
import random
import os

class RealisticVisionGenerator:
    def __init__(self):
        """
        Initialize the Realistic Vision image generator with V5.1 as a fallback
        """
        print("Loading Realistic Vision V5.1 model (fallback)...")
        
        # Model ID for Realistic Vision v5.1 (more stable fallback)
        model_id = "SG161222/Realistic_Vision_V5.1"
        
        # Load the pipeline with the recommended scheduler
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Set the scheduler to DPM++ SDE Karras (recommended for Realistic Vision)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.device = "cuda"
            self.pipe = self.pipe.to(self.device)
            
            # Try to enable xformers for memory efficiency, but don't fail if not available
            try:
                import xformers
                print("xformers is installed. Enabling memory efficient attention.")
                self.pipe.enable_xformers_memory_efficient_attention()
            except ImportError:
                print("xformers is not installed. Using default attention mechanism.")
                print("To enable xformers for better performance, install it with:")
                print("pip install xformers")
                
                # If xformers is not available, try using PyTorch's native memory-efficient attention
                if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                    print("Using PyTorch's native memory-efficient attention instead.")
                    # No explicit call needed as it will be used automatically
        else:
            self.device = "cpu"
            print("Warning: Running on CPU, which will be very slow!")
        
        print(f"Model loaded successfully on {self.device}")

    def generate(self, prompt, negative_prompt=None, width=768, height=768, 
                num_inference_steps=30, guidance_scale=7.5, seed=None, output_path=None):
        """
        Generate an image using the Realistic Vision model
        
        Args:
            prompt (str): The prompt to generate an image from
            negative_prompt (str, optional): Things to avoid in the image
            width (int): Width of the generated image
            height (int): Height of the generated image
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Guidance scale for generation
            seed (int, optional): Random seed for reproducibility
            output_path (str, optional): Path to save the generated image
            
        Returns:
            PIL.Image: The generated image
        """
        # Set a random seed if not provided
        if seed is None:
            seed = random.randint(0, 2147483647)
        
        # Set the generator for reproducibility
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Default negative prompt for Realistic Vision v5.1
        if negative_prompt is None:
            negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
        
        # Enhance prompt with recommended settings
        enhanced_prompt = f"RAW photo, {prompt}, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
        
        # Generate the image
        with torch.no_grad():
            image = self.pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
        
        # Save the image if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path)
        
        return image 