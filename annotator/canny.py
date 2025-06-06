import cv2
import numpy as np
from PIL import Image
from .util import HWC3, resize_image, pil_to_cv2, cv2_to_pil

class CannyDetector:
    def __init__(self):
        """Initialize Canny edge detector"""
        pass
    
    def detect(self, image, low_threshold=100, high_threshold=200, resolution=512):
        """
        Apply Canny edge detection to an image
        
        Args:
            image (PIL.Image or numpy.ndarray): Input image
            low_threshold (int): Lower threshold for the hysteresis procedure
            high_threshold (int): Higher threshold for the hysteresis procedure
            resolution (int): Resolution for resizing the image
            
        Returns:
            PIL.Image: Canny edge detection result
        """
        if isinstance(image, Image.Image):
            img = pil_to_cv2(image)
        else:
            img = image
            
        # Convert to RGB
        img = HWC3(img)
        
        # Resize image
        img = resize_image(img, resolution)
        
        # Apply Canny edge detection
        canny = cv2.Canny(img, low_threshold, high_threshold)
        
        # Convert to RGB
        canny = HWC3(canny)
        
        # Convert back to PIL
        return cv2_to_pil(canny) 