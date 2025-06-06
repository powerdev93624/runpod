import os
import torch
import numpy as np
from PIL import Image
import cv2
import insightface
from insightface.app import FaceAnalysis

class ReActorSwapper:
    def __init__(self, device="cuda", model_path=None):
        """Initialize ReActor face swapper
        
        Args:
            device (str): Device to run on ('cuda' or 'cpu')
            model_path (str): Path to model directory
        """
        self.device = device
        self.model_path = model_path
        self.model = None
        self.face_analyzer = None
        
        # Initialize models
        self.initialize()
    
    def initialize(self):
        """Initialize ReActor model and face analyzer"""
        try:
            # Initialize face analyzer
            self.face_analyzer = FaceAnalysis(
                name="buffalo_l", 
                root=os.path.join(self.model_path, "../insightface") if self.model_path else None,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] 
                if self.device == "cuda" else ['CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0 if self.device == "cuda" else -1)
            
            # Load ReActor model
            from insightface.model_zoo.inswapper import INSwapper
            
            # Define model path
            if self.model_path and os.path.exists(self.model_path):
                model_path = os.path.join(self.model_path, "reactor.pth")
            else:
                model_path = "reactor.pth"
            
            if os.path.exists(model_path):
                self.model = INSwapper(model_path, self.device == "cuda")
                print(f"ReActor model loaded from {model_path}")
            else:
                print(f"ReActor model not found at {model_path}")
                
        except Exception as e:
            print(f"Error initializing ReActor: {e}")
            print("ReActor face swapping will not be available.")
    
    def detect_faces(self, image):
        """Detect faces in an image
        
        Args:
            image (PIL.Image or numpy.ndarray): Input image
            
        Returns:
            list: List of detected faces
        """
        if self.face_analyzer is None:
            return []
            
        # Convert PIL to numpy/BGR for insightface
        if isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image
            
        # Detect faces
        faces = self.face_analyzer.get(img)
        return faces
    
    def swap_face(self, source_image, target_image, face_index=0, swap_strength=0.8):
        """Swap face from source image to target image
        
        Args:
            source_image (PIL.Image): Source face image
            target_image (PIL.Image): Target image to apply face to
            face_index (int): Index of face to swap in target (if multiple)
            swap_strength (float): Strength of face swapping effect
            
        Returns:
            PIL.Image: Image with swapped face
        """
        if self.model is None or self.face_analyzer is None:
            print("ReActor not initialized. Returning original image.")
            return target_image
            
        # Convert PIL to numpy/BGR for insightface
        if isinstance(source_image, Image.Image):
            source_img = cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR)
        else:
            source_img = source_image
            
        if isinstance(target_image, Image.Image):
            target_img = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)
        else:
            target_img = target_image
        
        # Detect faces in source and target
        source_faces = self.detect_faces(source_img)
        target_faces = self.detect_faces(target_img)
        
        if not source_faces:
            print("No face detected in source image")
            return target_image
            
        if not target_faces:
            print("No face detected in target image")
            return target_image
        
        # Use first face from source
        source_face = source_faces[0]
        
        # Select face from target based on index
        if face_index >= len(target_faces):
            print(f"Face index {face_index} out of range. Using face 0.")
            face_index = 0
        
        # Swap the face
        try:
            # Get target faces to swap
            target_face = target_faces[face_index]
            
            # Perform face swapping
            result_img = self.model.get(target_img, target_face, source_face, swap_strength)
            
            # Convert back to PIL
            return Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            
        except Exception as e:
            print(f"Error swapping face: {e}")
            return target_image 