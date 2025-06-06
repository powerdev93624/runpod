import cv2
import numpy as np
from PIL import Image

def HWC3(x):
    """Convert image to HWC3 format"""
    if x.ndim == 2:
        x = x[:, :, None]
    if x.shape[2] == 1:
        x = np.concatenate([x, x, x], axis=2)
    return x

def resize_image(input_image, resolution):
    """Resize image to target resolution"""
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def pil_to_cv2(pil_image):
    """Convert PIL image to cv2 image"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Convert cv2 image to PIL image"""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)) 