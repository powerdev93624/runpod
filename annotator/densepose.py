import cv2
import numpy as np
from PIL import Image
import torch
from .util import HWC3, resize_image, pil_to_cv2, cv2_to_pil

class DenseposeDetector:
    def __init__(self):
        """Initialize DensePose detector"""
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self):
        """Load DensePose model if not already loaded"""
        if self.model is not None:
            return
        
        try:
            from controlnet_aux import DenseposeDetector as DenseposeLegacy
            self.model = DenseposeLegacy.from_pretrained("lllyasviel/ControlNet")
            
        except ImportError:
            print("Failed to load DensePose. Make sure you have controlnet_aux installed.")
            raise
    
    def detect(self, image, resolution=512):
        """
        Generate DensePose map for human body
        
        Args:
            image (PIL.Image or numpy.ndarray): Input image
            resolution (int): Resolution for resizing the image
            
        Returns:
            PIL.Image: DensePose map
        """
        if isinstance(image, Image.Image):
            img = image
        else:
            img = cv2_to_pil(HWC3(image))
        
        # Load model if needed
        self.load_model()
        
        # Process image with DensePose
        densepose_image = self.model(img)
        
        # Convert to RGB
        if isinstance(densepose_image, np.ndarray):
            densepose_image = HWC3(densepose_image)
            densepose_image = cv2_to_pil(densepose_image)
        
        return densepose_image 