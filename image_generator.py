import torch
from diffusers import (
    StableDiffusionPipeline, 
    DPMSolverMultistepScheduler, 
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import torch.nn as nn
import numpy as np
import random
import os
import PIL
from PIL import Image
import cv2
from typing import Dict, List, Optional, Tuple, Union
import insightface
from insightface.app import FaceAnalysis
import annotator.util as util
from annotator.canny import CannyDetector
from annotator.midas import MidasDetector
from annotator.openpose import OpenposeDetector
from annotator.uniformer import UniformerDetector
from annotator.densepose import DenseposeDetector
from modules.reactor_swapper import ReActorSwapper
from modules.ip_adapter import IPAdapter
from modules.hyperdreambooth import HyperDreamBooth
from modules.face_restoration import CodeFormerRestorer, GFPGANRestorer
from modules.textual_inversion import TextualInversionLoader
from modules.instantid import InstantIDModel

class RealisticVisionGenerator:
    def __init__(self):
        """
        Initialize the Realistic Vision image generator with local model and addons
        """
        print("Loading custom Realistic Vision model from local directory...")
        
        # Path to local model
        model_path = "./model"
        
        # Check if model directory exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory {model_path} not found. Please ensure your custom model is saved in this location.")
        
        try:
            # Load components from local model
            self.text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
            self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
            self.unet = torch.load(os.path.join(model_path, "unet/diffusion_pytorch_model.bin"))
            self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
            self.feature_extractor = CLIPFeatureExtractor.from_pretrained(model_path, subfolder="feature_extractor")
            
            # Initialize main pipeline
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Initialize specialized pipelines
            self.img2img_pipe = StableDiffusionImg2ImgPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
                safety_checker=None,
                feature_extractor=self.pipe.feature_extractor,
                requires_safety_checker=False
            )
            
            self.inpaint_pipe = StableDiffusionInpaintPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
                safety_checker=None,
                feature_extractor=self.pipe.feature_extractor,
                requires_safety_checker=False
            )
            
        except Exception as e:
            print(f"Error loading model components: {e}")
            print("Falling back to loading full pipeline...")
            
            # Fallback to loading the full pipeline
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Initialize specialized pipelines from the main one
            self.img2img_pipe = StableDiffusionImg2ImgPipeline(**self.pipe.components)
            self.inpaint_pipe = StableDiffusionInpaintPipeline(**self.pipe.components)
        
        # Set the scheduler to DPM++ SDE Karras (recommended for Realistic Vision)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True
        )
        self.img2img_pipe.scheduler = self.pipe.scheduler
        self.inpaint_pipe.scheduler = self.pipe.scheduler
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.device = "cuda"
            self.pipe = self.pipe.to(self.device)
            self.img2img_pipe = self.img2img_pipe.to(self.device)
            self.inpaint_pipe = self.inpaint_pipe.to(self.device)
            
            # Try to enable xformers for memory efficiency
            try:
                import xformers
                print("xformers is installed. Enabling memory efficient attention.")
                self.pipe.enable_xformers_memory_efficient_attention()
                self.img2img_pipe.enable_xformers_memory_efficient_attention()
                self.inpaint_pipe.enable_xformers_memory_efficient_attention()
            except ImportError:
                print("xformers is not installed. Using default attention mechanism.")
                print("To enable xformers for better performance, install it with:")
                print("python install_xformers.py")
        else:
            self.device = "cpu"
            print("Warning: Running on CPU, which will be very slow!")
        
        print(f"Base model loaded successfully on {self.device}")
        
        # Initialize addons
        self.init_addons()
    
    def init_addons(self):
        """Initialize addon models and components"""
        self.addons = {}
        
        # ControlNet models
        print("Loading ControlNet models...")
        try:
            controlnet_models = {
                "canny": "./addons/controlnet/control_sd15_canny",
                "depth": "./addons/controlnet/control_sd15_depth",
                "pose": "./addons/controlnet/control_sd15_openpose",
                "seg": "./addons/controlnet/control_sd15_seg"
            }
            
            self.addons["controlnet"] = {}
            self.controlnet_pipes = {}
            
            for name, path in controlnet_models.items():
                if os.path.exists(path):
                    controlnet = ControlNetModel.from_pretrained(
                        path,
                        torch_dtype=torch.float16
                    )
                    self.addons["controlnet"][name] = controlnet
                    
                    # Create ControlNet pipeline
                    self.controlnet_pipes[name] = StableDiffusionControlNetPipeline(
                        vae=self.pipe.vae,
                        text_encoder=self.pipe.text_encoder,
                        tokenizer=self.pipe.tokenizer,
                        unet=self.pipe.unet,
                        scheduler=self.pipe.scheduler,
                        safety_checker=None,
                        feature_extractor=self.pipe.feature_extractor,
                        controlnet=controlnet
                    ).to(self.device)
                    
                    if torch.cuda.is_available():
                        try:
                            self.controlnet_pipes[name].enable_xformers_memory_efficient_attention()
                        except:
                            pass
            
            # ControlNet preprocessors
            self.addons["preprocessors"] = {
                "canny": CannyDetector(),
                "depth": MidasDetector(),
                "pose": OpenposeDetector(),
                "seg": UniformerDetector(),
                "densepose": DenseposeDetector()
            }
        except Exception as e:
            print(f"Error loading ControlNet models: {e}")
            print("Some ControlNet features may not be available.")
        
        # InstantID
        print("Loading InstantID...")
        try:
            if os.path.exists("./addons/instantid"):
                self.addons["instantid"] = InstantIDModel(
                    device=self.device,
                    model_path="./addons/instantid"
                )
        except Exception as e:
            print(f"Error loading InstantID: {e}")
        
        # InsightFace
        print("Loading InsightFace...")
        try:
            if os.path.exists("./addons/insightface"):
                self.addons["insightface"] = FaceAnalysis(
                    name="buffalo_l",
                    root="./addons/insightface",
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] 
                    if self.device == "cuda" else ['CPUExecutionProvider']
                )
                self.addons["insightface"].prepare(ctx_id=0 if self.device == "cuda" else -1)
        except Exception as e:
            print(f"Error loading InsightFace: {e}")
        
        # IP-Adapter
        print("Loading IP-Adapter...")
        try:
            if os.path.exists("./addons/ip-adapter"):
                self.addons["ip-adapter"] = IPAdapter(
                    device=self.device,
                    model_path="./addons/ip-adapter"
                )
        except Exception as e:
            print(f"Error loading IP-Adapter: {e}")
        
        # HyperDreamBooth
        print("Loading HyperDreamBooth...")
        try:
            if os.path.exists("./addons/hyperdreambooth"):
                self.addons["hyperdreambooth"] = HyperDreamBooth(
                    device=self.device,
                    model_path="./addons/hyperdreambooth"
                )
        except Exception as e:
            print(f"Error loading HyperDreamBooth: {e}")
        
        # ReActor
        print("Loading ReActor face swapper...")
        try:
            if os.path.exists("./addons/reactor"):
                self.addons["reactor"] = ReActorSwapper(
                    device=self.device,
                    model_path="./addons/reactor"
                )
        except Exception as e:
            print(f"Error loading ReActor: {e}")
        
        # Face Restoration
        print("Loading face restoration models...")
        try:
            if os.path.exists("./addons/codeformer"):
                self.addons["codeformer"] = CodeFormerRestorer(
                    device=self.device,
                    model_path="./addons/codeformer"
                )
            
            if os.path.exists("./addons/gfpgan"):
                self.addons["gfpgan"] = GFPGANRestorer(
                    device=self.device,
                    model_path="./addons/gfpgan"
                )
        except Exception as e:
            print(f"Error loading face restoration models: {e}")
        
        # Textual Inversion
        print("Loading Textual Inversion...")
        try:
            if os.path.exists("./addons/textual_inversion"):
                self.addons["textual_inversion"] = TextualInversionLoader(
                    pipe=self.pipe,
                    model_path="./addons/textual_inversion"
                )
                # Load all embeddings
                self.addons["textual_inversion"].load_embeddings()
        except Exception as e:
            print(f"Error loading Textual Inversion: {e}")
    
    def apply_controlnet(self, image, prompt, negative_prompt, controlnet_type, 
                        controlnet_conditioning_scale=0.8, **kwargs):
        """Apply ControlNet to guide image generation"""
        if controlnet_type not in self.addons["controlnet"]:
            raise ValueError(f"ControlNet type {controlnet_type} not available")
        
        # Prepare control image based on type
        if controlnet_type == "canny":
            control_image = self.addons["preprocessors"]["canny"].detect(image)
        elif controlnet_type == "depth":
            control_image = self.addons["preprocessors"]["depth"].detect(image)
        elif controlnet_type == "pose":
            control_image = self.addons["preprocessors"]["pose"].detect(image)
        elif controlnet_type == "seg":
            control_image = self.addons["preprocessors"]["seg"].detect(image)
        elif controlnet_type == "densepose":
            control_image = self.addons["preprocessors"]["densepose"].detect(image)
        else:
            control_image = image
        
        # Generate image with ControlNet
        result = self.controlnet_pipes[controlnet_type](
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            **kwargs
        ).images[0]
        
        return result
    
    def apply_instantid(self, face_image, prompt, negative_prompt, 
                        instantid_strength=0.8, **kwargs):
        """Apply InstantID for face-guided generation"""
        if "instantid" not in self.addons:
            raise ValueError("InstantID is not available")
        
        return self.addons["instantid"].generate(
            face_image=face_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=instantid_strength,
            pipe=self.pipe,
            **kwargs
        )
    
    def apply_ip_adapter(self, reference_image, prompt, negative_prompt, 
                         ip_adapter_scale=0.8, **kwargs):
        """Apply IP-Adapter for image-prompted generation"""
        if "ip-adapter" not in self.addons:
            raise ValueError("IP-Adapter is not available")
        
        return self.addons["ip-adapter"].generate(
            reference_image=reference_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            scale=ip_adapter_scale,
            pipe=self.pipe,
            **kwargs
        )
    
    def apply_face_swap(self, source_image, target_image, face_index=0, 
                        swap_strength=0.8):
        """Apply ReActor face swapper"""
        if "reactor" not in self.addons:
            raise ValueError("ReActor is not available")
        
        return self.addons["reactor"].swap_face(
            source_image=source_image,
            target_image=target_image,
            face_index=face_index,
            swap_strength=swap_strength
        )
    
    def apply_face_restoration(self, image, method="codeformer", strength=0.8):
        """Apply face restoration using CodeFormer or GFPGAN"""
        if method not in ["codeformer", "gfpgan"] or method not in self.addons:
            raise ValueError(f"Face restoration method {method} not available")
        
        return self.addons[method].restore(
            image=image,
            strength=strength
        )
    
    def generate(self, prompt, negative_prompt=None, width=768, height=768, 
                num_inference_steps=30, guidance_scale=7.5, seed=None, output_path=None,
                control_image=None, controlnet_type=None, controlnet_conditioning_scale=0.8,
                face_image=None, instantid_strength=0.8,
                reference_image=None, ip_adapter_scale=0.8,
                init_image=None, strength=0.7,
                mask_image=None,
                source_face_image=None, face_index=0, swap_strength=0.8,
                restore_face=False, face_restoration_method="codeformer", face_restoration_strength=0.5):
        """
        Generate an image using the Realistic Vision model with various addons
        
        Args:
            prompt (str): The prompt to generate an image from
            negative_prompt (str, optional): Things to avoid in the image
            width (int): Width of the generated image
            height (int): Height of the generated image
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Guidance scale for generation
            seed (int, optional): Random seed for reproducibility
            output_path (str, optional): Path to save the generated image
            
            # ControlNet parameters
            control_image (PIL.Image, optional): Image for ControlNet guidance
            controlnet_type (str, optional): Type of ControlNet to use ("canny", "depth", "pose", "seg", "densepose")
            controlnet_conditioning_scale (float): Strength of ControlNet guidance
            
            # InstantID parameters
            face_image (PIL.Image, optional): Face image for InstantID
            instantid_strength (float): Strength of InstantID guidance
            
            # IP-Adapter parameters
            reference_image (PIL.Image, optional): Reference image for IP-Adapter
            ip_adapter_scale (float): Strength of IP-Adapter guidance
            
            # img2img parameters
            init_image (PIL.Image, optional): Initial image for img2img
            strength (float): Strength of denoising for img2img
            
            # inpainting parameters
            mask_image (PIL.Image, optional): Mask for inpainting
            
            # ReActor parameters
            source_face_image (PIL.Image, optional): Source face for face swapping
            face_index (int): Index of face to swap
            swap_strength (float): Strength of face swapping
            
            # Face restoration parameters
            restore_face (bool): Whether to apply face restoration
            face_restoration_method (str): Method for face restoration ("codeformer" or "gfpgan")
            face_restoration_strength (float): Strength of face restoration
            
        Returns:
            PIL.Image: The generated image
        """
        # Set a random seed if not provided
        if seed is None:
            seed = random.randint(0, 2147483647)
        
        # Set the generator for reproducibility
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Default negative prompt for Realistic Vision
        if negative_prompt is None:
            negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
        
        # Enhance prompt with recommended settings
        enhanced_prompt = f"RAW photo, {prompt}, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
        
        # Apply Textual Inversion if available
        if "textual_inversion" in self.addons:
            enhanced_prompt = self.addons["textual_inversion"].process_prompt(enhanced_prompt)
        
        # Initialize the image variable
        image = None
        
        # Main generation logic with different paths based on inputs
        generation_params = {
            "prompt": enhanced_prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator
        }
        
        # 1. ControlNet path
        if control_image is not None and controlnet_type is not None and controlnet_type in self.addons.get("controlnet", {}):
            print(f"Using {controlnet_type} ControlNet for generation...")
            image = self.apply_controlnet(
                image=control_image,
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                controlnet_type=controlnet_type,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                **{k: v for k, v in generation_params.items() if k not in ["prompt", "negative_prompt"]}
            )
        
        # 2. InstantID path
        elif face_image is not None and "instantid" in self.addons:
            print("Using InstantID for face-guided generation...")
            image = self.apply_instantid(
                face_image=face_image,
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                instantid_strength=instantid_strength,
                **{k: v for k, v in generation_params.items() if k not in ["prompt", "negative_prompt"]}
            )
        
        # 3. IP-Adapter path
        elif reference_image is not None and "ip-adapter" in self.addons:
            print("Using IP-Adapter for reference-guided generation...")
            image = self.apply_ip_adapter(
                reference_image=reference_image,
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                ip_adapter_scale=ip_adapter_scale,
                **{k: v for k, v in generation_params.items() if k not in ["prompt", "negative_prompt"]}
            )
        
        # 4. Inpainting path
        elif init_image is not None and mask_image is not None:
            print("Using inpainting...")
            image = self.inpaint_pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                mask_image=mask_image,
                **{k: v for k, v in generation_params.items() if k not in ["prompt", "negative_prompt", "width", "height"]}
            ).images[0]
        
        # 5. Img2img path
        elif init_image is not None:
            print("Using img2img...")
            image = self.img2img_pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=strength,
                **{k: v for k, v in generation_params.items() if k not in ["prompt", "negative_prompt", "width", "height"]}
            ).images[0]
        
        # 6. Default text-to-image path
        else:
            print("Using standard text-to-image generation...")
            image = self.pipe(
                **generation_params
            ).images[0]
        
        # Post-processing
        
        # 1. ReActor face swapping
        if image is not None and source_face_image is not None and "reactor" in self.addons:
            print("Applying ReActor face swapping...")
            image = self.apply_face_swap(
                source_image=source_face_image,
                target_image=image,
                face_index=face_index,
                swap_strength=swap_strength
            )
        
        # 2. Face restoration
        if image is not None and restore_face and face_restoration_method in self.addons:
            print(f"Applying {face_restoration_method} face restoration...")
            image = self.apply_face_restoration(
                image=image,
                method=face_restoration_method,
                strength=face_restoration_strength
            )
        
        # Save the image if output path is provided
        if output_path and image is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path)
        
        return image 