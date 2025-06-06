import os
import torch
from PIL import Image
import numpy as np

class HyperDreamBooth:
    def __init__(self, device="cuda", model_path=None):
        """Initialize HyperDreamBooth
        
        Args:
            device (str): Device to run on ('cuda' or 'cpu')
            model_path (str): Path to model directory
        """
        self.device = device
        self.model_path = model_path
        self.subjects = {}
        
        # Initialize model
        self.initialize()
    
    def initialize(self):
        """Initialize HyperDreamBooth model"""
        try:
            # Scan for available subject embeddings
            if self.model_path and os.path.exists(self.model_path):
                for subject_folder in os.listdir(self.model_path):
                    subject_path = os.path.join(self.model_path, subject_folder)
                    if os.path.isdir(subject_path) and os.path.exists(os.path.join(subject_path, "subject_embedding.pt")):
                        self.subjects[subject_folder] = subject_path
            
            print(f"HyperDreamBooth initialized with {len(self.subjects)} subjects")
            
        except Exception as e:
            print(f"Error initializing HyperDreamBooth: {e}")
            print("HyperDreamBooth will not be available.")
    
    def get_available_subjects(self):
        """Get list of available subjects
        
        Returns:
            list: List of available subject names
        """
        return list(self.subjects.keys())
    
    def generate(self, subject_name, prompt, negative_prompt=None, 
                 strength=0.8, pipe=None, **kwargs):
        """Generate an image using HyperDreamBooth
        
        Args:
            subject_name (str): Name of the subject to use
            prompt (str): Text prompt
            negative_prompt (str, optional): Negative prompt
            strength (float): Strength of subject conditioning
            pipe (StableDiffusionPipeline): Base pipeline to use
            **kwargs: Additional arguments for the pipeline
            
        Returns:
            PIL.Image: Generated image
        """
        if subject_name not in self.subjects or pipe is None:
            print(f"Subject '{subject_name}' not found or pipeline not provided")
            
            # Fallback to standard generation
            if pipe is not None:
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    **kwargs
                ).images[0]
                return output
            
            return None
        
        try:
            from diffusers import StableDiffusionPipeline
            from transformers import CLIPTextModel, CLIPTokenizer
            
            subject_path = self.subjects[subject_name]
            embedding_path = os.path.join(subject_path, "subject_embedding.pt")
            
            # Load the subject embedding
            subject_embedding = torch.load(embedding_path, map_location=self.device)
            
            # Get the text encoder and tokenizer
            text_encoder = pipe.text_encoder
            tokenizer = pipe.tokenizer
            
            # Tokenize the prompt
            # Replace [subject] with the actual subject token in the prompt
            subject_token = f"<{subject_name}>"
            if subject_token not in prompt:
                # Add subject token if not in prompt
                prompt = f"{subject_token} {prompt}"
            
            # Tokenize the prompt
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            
            # Encode the prompt
            with torch.no_grad():
                prompt_embeds = text_encoder(text_inputs.input_ids)[0]
            
            # Apply the subject embedding
            # Find the subject token index
            subject_token_id = tokenizer.convert_tokens_to_ids(subject_token)
            if subject_token_id in text_inputs.input_ids[0]:
                subject_idx = torch.where(text_inputs.input_ids[0] == subject_token_id)[0][0]
                # Replace the embedding at this index
                prompt_embeds[0, subject_idx] = subject_embedding * strength + prompt_embeds[0, subject_idx] * (1 - strength)
            
            # Generate the image with the modified embedding
            with torch.no_grad():
                output = pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt=negative_prompt,
                    **kwargs
                ).images[0]
            
            return output
            
        except Exception as e:
            print(f"Error generating with HyperDreamBooth: {e}")
            print("Falling back to standard generation without HyperDreamBooth")
            
            # Fallback to standard generation
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                **kwargs
            ).images[0]
            
            return output 