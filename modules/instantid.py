import os
import torch
import numpy as np
from PIL import Image
import insightface
from insightface.app import FaceAnalysis

class InstantIDModel:
    def __init__(self, device="cuda", model_path=None):
        """Initialize InstantID model
        
        Args:
            device (str): Device to run on ('cuda' or 'cpu')
            model_path (str): Path to model directory
        """
        self.device = device
        self.model_path = model_path
        self.face_analyzer = None
        self.instantid_adapter = None
        
        # Initialize model
        self.initialize()
    
    def initialize(self):
        """Initialize InstantID model"""
        try:
            # Initialize face analyzer
            self.face_analyzer = FaceAnalysis(
                name="buffalo_l", 
                root=os.path.join(self.model_path, "insightface") if self.model_path else None,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] 
                if self.device == "cuda" else ['CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0 if self.device == "cuda" else -1)
            
            # Initialize InstantID adapter
            from diffusers import StableDiffusionPipeline
            
            # Define model paths
            if self.model_path:
                instantid_ckpt = os.path.join(self.model_path, "ip-adapter-instantid.bin")
                controlnet_ckpt = os.path.join(self.model_path, "ControlNetModel")
            else:
                instantid_ckpt = "ip-adapter-instantid.bin"
                controlnet_ckpt = "ControlNetModel"
            
            # We'll load the actual adapter during generation
            self.instantid_ckpt = instantid_ckpt
            self.controlnet_ckpt = controlnet_ckpt
            
            print("InstantID initialized successfully")
            
        except Exception as e:
            print(f"Error initializing InstantID: {e}")
            print("InstantID will not be available.")
    
    def detect_faces(self, image):
        """Detect faces in an image
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            list: List of detected faces
        """
        if self.face_analyzer is None:
            return []
            
        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image
            
        # Detect faces
        faces = self.face_analyzer.get(img)
        return faces
    
    def get_face_embedding(self, face_image):
        """Get face embedding from image
        
        Args:
            face_image (PIL.Image): Face image
            
        Returns:
            tuple: Face embedding, face bbox
        """
        # Detect face
        faces = self.detect_faces(face_image)
        
        if not faces:
            print("No face detected in face image")
            return None, None
        
        # Use the first detected face
        face = faces[0]
        return face.embedding, face.bbox
    
    def generate(self, face_image, prompt, negative_prompt=None, strength=0.8, pipe=None, **kwargs):
        """Generate an image using InstantID
        
        Args:
            face_image (PIL.Image): Face image for identity
            prompt (str): Text prompt
            negative_prompt (str, optional): Negative prompt
            strength (float): Strength of identity conditioning
            pipe (StableDiffusionPipeline): Base pipeline to use
            **kwargs: Additional arguments for the pipeline
            
        Returns:
            PIL.Image: Generated image
        """
        if self.face_analyzer is None or pipe is None:
            print("InstantID not initialized or pipeline not provided")
            return None
        
        try:
            from diffusers import ControlNetModel
            from instantid.pipeline_stable_diffusion_instantid import StableDiffusionInstantIDPipeline
            
            # Get face embedding
            face_embedding, face_bbox = self.get_face_embedding(face_image)
            
            if face_embedding is None:
                print("Failed to get face embedding. Falling back to standard generation.")
                # Fallback to standard generation
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    **kwargs
                ).images[0]
                
                return output
            
            # Load ControlNet
            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_ckpt, 
                torch_dtype=torch.float16
            )
            
            # Create InstantID pipeline
            instantid_pipe = StableDiffusionInstantIDPipeline.from_pretrained(
                pipe, 
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            # Load InstantID adapter
            instantid_pipe.load_ip_adapter_instantid(self.instantid_ckpt)
            
            # Generate image
            image = instantid_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                ip_adapter_scale=strength,
                face_embedding=face_embedding,
                face_image=face_image,
                **kwargs
            ).images[0]
            
            return image
            
        except Exception as e:
            print(f"Error generating with InstantID: {e}")
            print("Falling back to standard generation without InstantID")
            
            # Fallback to standard generation
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                **kwargs
            ).images[0]
            
            return output 