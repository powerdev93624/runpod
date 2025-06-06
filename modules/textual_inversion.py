import os
import torch
import re
from diffusers import StableDiffusionPipeline

class TextualInversionLoader:
    def __init__(self, pipe=None, model_path=None):
        """Initialize Textual Inversion loader
        
        Args:
            pipe (StableDiffusionPipeline): Base pipeline to use
            model_path (str): Path to embeddings directory
        """
        self.pipe = pipe
        self.model_path = model_path
        self.embeddings = {}
        self.token_map = {}
        
        # Initialize
        if pipe is not None:
            self.initialize()
    
    def initialize(self):
        """Initialize textual inversion embeddings"""
        try:
            # Scan for available embeddings
            if self.model_path and os.path.exists(self.model_path):
                print(f"Scanning for textual inversion embeddings in {self.model_path}")
                for filename in os.listdir(self.model_path):
                    if filename.endswith('.pt') or filename.endswith('.bin') or filename.endswith('.safetensors'):
                        name = os.path.splitext(filename)[0]
                        path = os.path.join(self.model_path, filename)
                        
                        # Store path to embedding
                        self.embeddings[name] = path
                        
                        # Add token mapping
                        token = f"<{name}>"
                        self.token_map[name] = token
                
                print(f"Found {len(self.embeddings)} textual inversion embeddings")
            
        except Exception as e:
            print(f"Error initializing Textual Inversion: {e}")
            print("Textual Inversion will not be available.")
    
    def load_embeddings(self):
        """Load all available embeddings into the pipeline"""
        if self.pipe is None or not self.embeddings:
            print("No pipeline or embeddings available")
            return
        
        try:
            # Load each embedding
            for name, path in self.embeddings.items():
                token = self.token_map[name]
                try:
                    self.pipe.load_textual_inversion(path, token=token)
                    print(f"Loaded embedding {name} with token {token}")
                except Exception as e:
                    print(f"Error loading embedding {name}: {e}")
            
        except Exception as e:
            print(f"Error loading embeddings: {e}")
    
    def get_available_embeddings(self):
        """Get list of available embeddings
        
        Returns:
            dict: Dict mapping embedding names to tokens
        """
        return self.token_map
    
    def process_prompt(self, prompt):
        """Process prompt to add available embedding tokens
        
        Args:
            prompt (str): Original prompt
            
        Returns:
            str: Processed prompt with embedding tokens
        """
        # Look for embedding names in the prompt
        for name, token in self.token_map.items():
            # Look for the name with various delimiters
            name_patterns = [
                f"\\b{re.escape(name)}\\b",  # word boundary
                f"\\[{re.escape(name)}\\]",  # [name]
                f"\\({re.escape(name)}\\)",  # (name)
            ]
            
            for pattern in name_patterns:
                # Replace with the actual token
                prompt = re.sub(pattern, token, prompt)
        
        return prompt 