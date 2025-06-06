import cv2
import numpy as np
from PIL import Image
import torch
from .util import HWC3, resize_image, pil_to_cv2, cv2_to_pil

class MidasDetector:
    def __init__(self):
        """Initialize MiDaS depth detector"""
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self):
        """Load MiDaS model if not already loaded"""
        if self.model is not None:
            return
        
        try:
            import torch
            from transformers import DPTForDepthEstimation, DPTImageProcessor
            
            self.processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
            self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
            self.model.to(self.device)
            
        except ImportError:
            print("Failed to load MiDaS. Make sure you have transformers installed.")
            raise
    
    def detect(self, image, resolution=512):
        """
        Generate depth map using MiDaS
        
        Args:
            image (PIL.Image or numpy.ndarray): Input image
            resolution (int): Resolution for resizing the image
            
        Returns:
            PIL.Image: Depth map
        """
        if isinstance(image, Image.Image):
            img = image
        else:
            img = cv2_to_pil(HWC3(image))
        
        # Load model if needed
        self.load_model()
        
        # Process image with MiDaS
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            depth_map = outputs.predicted_depth
        
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(resolution, resolution),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        # Normalize depth map
        depth_min = torch.min(depth_map)
        depth_max = torch.max(depth_map)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        depth_map = (depth_map * 255).cpu().numpy().astype(np.uint8)
        
        # Convert to RGB
        depth_map = HWC3(depth_map)
        
        # Convert to PIL
        return cv2_to_pil(depth_map) 