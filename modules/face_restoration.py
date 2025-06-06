import os
import numpy as np
import torch
from PIL import Image
import cv2

class CodeFormerRestorer:
    def __init__(self, device="cuda", model_path=None):
        """Initialize CodeFormer face restorer
        
        Args:
            device (str): Device to run on ('cuda' or 'cpu')
            model_path (str): Path to model directory
        """
        self.device = device
        self.model_path = model_path
        self.model = None
        
        # Initialize model
        self.initialize()
    
    def initialize(self):
        """Initialize CodeFormer model"""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
            from torchvision.transforms.functional import normalize
            from basicsr.utils.registry import ARCH_REGISTRY
            
            # Paths
            if self.model_path:
                model_path = os.path.join(self.model_path, "codeformer.pth")
                detection_model = os.path.join(self.model_path, "detection_Resnet50_Final.pth")
                parsing_model = os.path.join(self.model_path, "parsing_parsenet.pth")
            else:
                model_path = "codeformer.pth"
                detection_model = "detection_Resnet50_Final.pth"
                parsing_model = "parsing_parsenet.pth"
            
            # Load model
            self.model = ARCH_REGISTRY.get("CodeFormer")(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=["32", "64", "128", "256"],
            ).to(self.device)
            
            # Load checkpoint
            checkpoint = torch.load(model_path)["params_ema"]
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            
            # Face detection and alignment
            self.face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model=detection_model,
                save_ext="png",
                use_parse=True,
                device=self.device,
                model_rootpath=self.model_path,
            )
            
        except Exception as e:
            print(f"Error initializing CodeFormer: {e}")
            print("CodeFormer face restoration will not be available.")
    
    def restore(self, image, strength=0.5):
        """Restore faces in an image
        
        Args:
            image (PIL.Image): Input image
            strength (float): Strength of restoration (0.0 to 1.0)
            
        Returns:
            PIL.Image: Image with restored faces
        """
        if self.model is None:
            print("CodeFormer not initialized. Returning original image.")
            return image
            
        # Convert PIL to numpy/BGR for facexlib
        if isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image.copy()
        
        # Extract and align faces
        self.face_helper.clean_all()
        self.face_helper.read_image(img)
        self.face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        self.face_helper.align_warp_face()
        
        # If no face detected, return original image
        if len(self.face_helper.cropped_faces) == 0:
            print("No face detected in the image")
            return image
        
        # Process each detected face
        for i, cropped_face in enumerate(self.face_helper.cropped_faces):
            # Convert to RGB and normalize
            cropped_face_t = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            cropped_face_t = torch.from_numpy(cropped_face_t).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            try:
                with torch.no_grad():
                    output = self.model(cropped_face_t, w=strength)[0]
                    # Convert back to BGR
                    restored_face = output.permute(1, 2, 0).cpu().numpy() * 255.0
                    restored_face = restored_face.astype("uint8")
                    restored_face = cv2.cvtColor(restored_face, cv2.COLOR_RGB2BGR)
                self.face_helper.add_restored_face(restored_face)
            except Exception as e:
                print(f"Error restoring face: {e}")
                self.face_helper.add_restored_face(cropped_face)
        
        # Paste faces back
        self.face_helper.get_inverse_affine(None)
        restored_img = self.face_helper.paste_faces_to_input_image()
        
        # Convert back to PIL
        return Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))


class GFPGANRestorer:
    def __init__(self, device="cuda", model_path=None):
        """Initialize GFPGAN face restorer
        
        Args:
            device (str): Device to run on ('cuda' or 'cpu')
            model_path (str): Path to model directory
        """
        self.device = device
        self.model_path = model_path
        self.model = None
        
        # Initialize model
        self.initialize()
    
    def initialize(self):
        """Initialize GFPGAN model"""
        try:
            import gfpgan
            from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
            
            # Paths
            if self.model_path:
                model_path = os.path.join(self.model_path, "GFPGANv1.4.pth")
                detection_model = os.path.join(self.model_path, "detection_Resnet50_Final.pth")
                parsing_model = os.path.join(self.model_path, "parsing_parsenet.pth")
            else:
                model_path = "GFPGANv1.4.pth"
                detection_model = "detection_Resnet50_Final.pth"
                parsing_model = "parsing_parsenet.pth"
            
            # Load model
            self.model = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=2,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True
            ).to(self.device)
            
            # Load checkpoint
            checkpoint = torch.load(model_path)["params_ema"]
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            
            # Face detection and alignment
            self.face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model=detection_model,
                save_ext="png",
                use_parse=True,
                device=self.device,
                model_rootpath=self.model_path,
            )
            
        except Exception as e:
            print(f"Error initializing GFPGAN: {e}")
            print("GFPGAN face restoration will not be available.")
    
    def restore(self, image, strength=0.5):
        """Restore faces in an image
        
        Args:
            image (PIL.Image): Input image
            strength (float): Strength of restoration (0.0 to 1.0)
            
        Returns:
            PIL.Image: Image with restored faces
        """
        if self.model is None:
            print("GFPGAN not initialized. Returning original image.")
            return image
            
        # Convert PIL to numpy/BGR for facexlib
        if isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image.copy()
        
        # Extract and align faces
        self.face_helper.clean_all()
        self.face_helper.read_image(img)
        self.face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        self.face_helper.align_warp_face()
        
        # If no face detected, return original image
        if len(self.face_helper.cropped_faces) == 0:
            print("No face detected in the image")
            return image
        
        # Process each detected face
        for i, cropped_face in enumerate(self.face_helper.cropped_faces):
            # Convert to RGB and normalize
            cropped_face_t = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            cropped_face_t = torch.from_numpy(cropped_face_t).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            try:
                with torch.no_grad():
                    _, _, output = self.model(cropped_face_t, return_rgb=True, weight=strength)
                    # Convert back to BGR
                    restored_face = output.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
                    restored_face = restored_face.astype("uint8")
                    restored_face = cv2.cvtColor(restored_face, cv2.COLOR_RGB2BGR)
                self.face_helper.add_restored_face(restored_face)
            except Exception as e:
                print(f"Error restoring face: {e}")
                self.face_helper.add_restored_face(cropped_face)
        
        # Paste faces back
        self.face_helper.get_inverse_affine(None)
        restored_img = self.face_helper.paste_faces_to_input_image()
        
        # Convert back to PIL
        return Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)) 