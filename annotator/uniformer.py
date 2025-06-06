import cv2
import numpy as np
from PIL import Image
import torch
from .util import HWC3, resize_image, pil_to_cv2, cv2_to_pil

class UniformerDetector:
    def __init__(self):
        """Initialize Uniformer segmentation detector"""
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self):
        """Load Uniformer model if not already loaded"""
        if self.model is not None:
            return
        
        try:
            from controlnet_aux import UniformerDetector as UniformerLegacy
            self.model = UniformerLegacy.from_pretrained("lllyasviel/ControlNet")
            
        except ImportError:
            print("Failed to load Uniformer. Make sure you have controlnet_aux installed.")
            raise
    
    def detect(self, image, resolution=512):
        """
        Generate semantic segmentation map using Uniformer
        
        Args:
            image (PIL.Image or numpy.ndarray): Input image
            resolution (int): Resolution for resizing the image
            
        Returns:
            PIL.Image: Segmentation map
        """
        if isinstance(image, Image.Image):
            img = image
        else:
            img = cv2_to_pil(HWC3(image))
        
        # Load model if needed
        self.load_model()
        
        # Process image with Uniformer
        seg_image = self.model(img)
        
        # Convert to RGB
        if isinstance(seg_image, np.ndarray):
            seg_image = HWC3(seg_image)
            seg_image = cv2_to_pil(seg_image)
        
        return seg_image 