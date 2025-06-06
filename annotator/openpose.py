import cv2
import numpy as np
from PIL import Image
import torch
from .util import HWC3, resize_image, pil_to_cv2, cv2_to_pil

class OpenposeDetector:
    def __init__(self):
        """Initialize OpenPose detector"""
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self):
        """Load OpenPose model if not already loaded"""
        if self.model is not None:
            return
        
        try:
            from controlnet_aux import OpenposeDetector as OpenposeLegacy
            self.model = OpenposeLegacy.from_pretrained("lllyasviel/ControlNet")
            
        except ImportError:
            print("Failed to load OpenPose. Make sure you have controlnet_aux installed.")
            raise
    
    def detect(self, image, resolution=512):
        """
        Generate pose keypoints using OpenPose
        
        Args:
            image (PIL.Image or numpy.ndarray): Input image
            resolution (int): Resolution for resizing the image
            
        Returns:
            PIL.Image: Pose keypoints visualization
        """
        if isinstance(image, Image.Image):
            img = image
        else:
            img = cv2_to_pil(HWC3(image))
        
        # Load model if needed
        self.load_model()
        
        # Process image with OpenPose
        pose_image = self.model(img)
        
        # Convert to RGB
        if isinstance(pose_image, np.ndarray):
            pose_image = HWC3(pose_image)
            pose_image = cv2_to_pil(pose_image)
        
        return pose_image 