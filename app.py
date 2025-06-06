from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import uuid
import time
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import shutil
from PIL import Image
import io
import numpy as np
import base64

from image_generator import RealisticVisionGenerator

app = FastAPI(
    title="Realistic Vision API",
    description="API server for generating realistic images using a custom Realistic Vision model with various addons",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the image generator
generator = RealisticVisionGenerator()

class TextToImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: Optional[int] = 768
    height: Optional[int] = 768
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = None

class ControlNetRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: Optional[int] = 768
    height: Optional[int] = 768
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = None
    controlnet_type: str = Field(..., description="Type of ControlNet to use: canny, depth, pose, seg, densepose")
    controlnet_conditioning_scale: Optional[float] = 0.8
    control_image_url: str = Field(..., description="URL to the control image (previously uploaded)")

class InstantIDRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: Optional[int] = 768
    height: Optional[int] = 768
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = None
    instantid_strength: Optional[float] = 0.8
    face_image_url: str = Field(..., description="URL to the face image (previously uploaded)")

class IPAdapterRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: Optional[int] = 768
    height: Optional[int] = 768
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = None
    ip_adapter_scale: Optional[float] = 0.8
    reference_image_url: str = Field(..., description="URL to the reference image (previously uploaded)")

class Img2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = None
    strength: Optional[float] = 0.7
    init_image_url: str = Field(..., description="URL to the initial image (previously uploaded)")

class InpaintRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = None
    init_image_url: str = Field(..., description="URL to the initial image (previously uploaded)")
    mask_image_url: str = Field(..., description="URL to the mask image (previously uploaded)")

class FaceSwapRequest(BaseModel):
    image_url: str = Field(..., description="URL to the target image (previously generated)")
    source_face_url: str = Field(..., description="URL to the source face image (previously uploaded)")
    face_index: Optional[int] = 0
    swap_strength: Optional[float] = 0.8

class FaceRestorationRequest(BaseModel):
    image_url: str = Field(..., description="URL to the image to restore (previously generated)")
    method: str = Field(..., description="Restoration method: codeformer or gfpgan")
    strength: Optional[float] = 0.5

class ImageResponse(BaseModel):
    image_url: str
    generation_time: float
    parameters: Dict[str, Any]

class AddonListResponse(BaseModel):
    controlnet: List[str] = []
    face_restoration: List[str] = []
    other_addons: List[str] = []

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

def get_image_from_url(url: str) -> Image.Image:
    """Get an image from a local URL"""
    if not url.startswith("/uploads/") and not url.startswith("/images/"):
        raise HTTPException(status_code=400, detail="Invalid image URL")
    
    # Remove leading slash and get the path
    path = url[1:]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Image not found: {url}")
    
    return Image.open(path)

@app.post("/upload", response_model=Dict[str, str])
async def upload_image(file: UploadFile = File(...)):
    """Upload an image to use with addons"""
    try:
        # Generate a unique filename
        file_extension = os.path.splitext(file.filename)[1]
        filename = f"uploads/{uuid.uuid4()}{file_extension}"
        
        # Save the file
        with open(filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {"url": f"/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/addons", response_model=AddonListResponse)
def list_addons():
    """List available addon models"""
    response = AddonListResponse()
    
    # List ControlNet models
    if hasattr(generator, "addons") and "controlnet" in generator.addons:
        response.controlnet = list(generator.addons["controlnet"].keys())
    
    # List face restoration methods
    face_restoration = []
    if hasattr(generator, "addons"):
        if "codeformer" in generator.addons:
            face_restoration.append("codeformer")
        if "gfpgan" in generator.addons:
            face_restoration.append("gfpgan")
    response.face_restoration = face_restoration
    
    # List other addons
    other_addons = []
    if hasattr(generator, "addons"):
        if "instantid" in generator.addons:
            other_addons.append("instantid")
        if "ip-adapter" in generator.addons:
            other_addons.append("ip-adapter")
        if "reactor" in generator.addons:
            other_addons.append("reactor")
        if "hyperdreambooth" in generator.addons:
            other_addons.append("hyperdreambooth")
        if "textual_inversion" in generator.addons:
            other_addons.append("textual_inversion")
        if "insightface" in generator.addons:
            other_addons.append("insightface")
    response.other_addons = other_addons
    
    return response

@app.post("/generate", response_model=ImageResponse)
async def generate_image(request: TextToImageRequest):
    """
    Generate an image based on the provided text prompt using the custom Realistic Vision model
    """
    try:
        start_time = time.time()
        
        # Generate a unique filename
        image_filename = f"output/{uuid.uuid4()}.png"
        
        # Generate the image
        generator.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            output_path=image_filename
        )
        
        generation_time = time.time() - start_time
        
        # Return the URL to the generated image
        return {
            "image_url": f"/images/{os.path.basename(image_filename)}",
            "generation_time": generation_time,
            "parameters": {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "width": request.width,
                "height": request.height,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/controlnet", response_model=ImageResponse)
async def generate_with_controlnet(request: ControlNetRequest):
    """
    Generate an image with ControlNet guidance
    """
    try:
        start_time = time.time()
        
        # Get control image
        control_image = get_image_from_url(request.control_image_url)
        
        # Generate a unique filename
        image_filename = f"output/{uuid.uuid4()}.png"
        
        # Generate the image with ControlNet
        generator.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            output_path=image_filename,
            control_image=control_image,
            controlnet_type=request.controlnet_type,
            controlnet_conditioning_scale=request.controlnet_conditioning_scale
        )
        
        generation_time = time.time() - start_time
        
        # Return the URL to the generated image
        return {
            "image_url": f"/images/{os.path.basename(image_filename)}",
            "generation_time": generation_time,
            "parameters": {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "width": request.width,
                "height": request.height,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed,
                "controlnet_type": request.controlnet_type,
                "controlnet_conditioning_scale": request.controlnet_conditioning_scale
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/instantid", response_model=ImageResponse)
async def generate_with_instantid(request: InstantIDRequest):
    """
    Generate an image with InstantID face guidance
    """
    try:
        start_time = time.time()
        
        # Get face image
        face_image = get_image_from_url(request.face_image_url)
        
        # Generate a unique filename
        image_filename = f"output/{uuid.uuid4()}.png"
        
        # Generate the image with InstantID
        generator.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            output_path=image_filename,
            face_image=face_image,
            instantid_strength=request.instantid_strength
        )
        
        generation_time = time.time() - start_time
        
        # Return the URL to the generated image
        return {
            "image_url": f"/images/{os.path.basename(image_filename)}",
            "generation_time": generation_time,
            "parameters": {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "width": request.width,
                "height": request.height,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed,
                "instantid_strength": request.instantid_strength
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/ip-adapter", response_model=ImageResponse)
async def generate_with_ip_adapter(request: IPAdapterRequest):
    """
    Generate an image with IP-Adapter reference guidance
    """
    try:
        start_time = time.time()
        
        # Get reference image
        reference_image = get_image_from_url(request.reference_image_url)
        
        # Generate a unique filename
        image_filename = f"output/{uuid.uuid4()}.png"
        
        # Generate the image with IP-Adapter
        generator.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            output_path=image_filename,
            reference_image=reference_image,
            ip_adapter_scale=request.ip_adapter_scale
        )
        
        generation_time = time.time() - start_time
        
        # Return the URL to the generated image
        return {
            "image_url": f"/images/{os.path.basename(image_filename)}",
            "generation_time": generation_time,
            "parameters": {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "width": request.width,
                "height": request.height,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed,
                "ip_adapter_scale": request.ip_adapter_scale
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/img2img", response_model=ImageResponse)
async def generate_img2img(request: Img2ImgRequest):
    """
    Generate an image using img2img
    """
    try:
        start_time = time.time()
        
        # Get init image
        init_image = get_image_from_url(request.init_image_url)
        
        # Generate a unique filename
        image_filename = f"output/{uuid.uuid4()}.png"
        
        # Generate the image with img2img
        generator.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            output_path=image_filename,
            init_image=init_image,
            strength=request.strength
        )
        
        generation_time = time.time() - start_time
        
        # Return the URL to the generated image
        return {
            "image_url": f"/images/{os.path.basename(image_filename)}",
            "generation_time": generation_time,
            "parameters": {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed,
                "strength": request.strength
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/inpaint", response_model=ImageResponse)
async def generate_inpaint(request: InpaintRequest):
    """
    Generate an image using inpainting
    """
    try:
        start_time = time.time()
        
        # Get init and mask images
        init_image = get_image_from_url(request.init_image_url)
        mask_image = get_image_from_url(request.mask_image_url)
        
        # Generate a unique filename
        image_filename = f"output/{uuid.uuid4()}.png"
        
        # Generate the image with inpainting
        generator.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            output_path=image_filename,
            init_image=init_image,
            mask_image=mask_image
        )
        
        generation_time = time.time() - start_time
        
        # Return the URL to the generated image
        return {
            "image_url": f"/images/{os.path.basename(image_filename)}",
            "generation_time": generation_time,
            "parameters": {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/post-process/face-swap", response_model=ImageResponse)
async def apply_face_swap(request: FaceSwapRequest):
    """
    Apply face swapping to an image
    """
    try:
        start_time = time.time()
        
        # Get target and source face images
        target_image = get_image_from_url(request.image_url)
        source_face_image = get_image_from_url(request.source_face_url)
        
        # Generate a unique filename
        image_filename = f"output/{uuid.uuid4()}.png"
        
        # Apply face swapping
        result_image = generator.apply_face_swap(
            source_image=source_face_image,
            target_image=target_image,
            face_index=request.face_index,
            swap_strength=request.swap_strength
        )
        
        # Save the image
        result_image.save(image_filename)
        
        generation_time = time.time() - start_time
        
        # Return the URL to the processed image
        return {
            "image_url": f"/images/{os.path.basename(image_filename)}",
            "generation_time": generation_time,
            "parameters": {
                "face_index": request.face_index,
                "swap_strength": request.swap_strength
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/post-process/face-restoration", response_model=ImageResponse)
async def apply_face_restoration(request: FaceRestorationRequest):
    """
    Apply face restoration to an image
    """
    try:
        start_time = time.time()
        
        # Get image
        image = get_image_from_url(request.image_url)
        
        # Generate a unique filename
        image_filename = f"output/{uuid.uuid4()}.png"
        
        # Apply face restoration
        result_image = generator.apply_face_restoration(
            image=image,
            method=request.method,
            strength=request.strength
        )
        
        # Save the image
        result_image.save(image_filename)
        
        generation_time = time.time() - start_time
        
        # Return the URL to the processed image
        return {
            "image_url": f"/images/{os.path.basename(image_filename)}",
            "generation_time": generation_time,
            "parameters": {
                "method": request.method,
                "strength": request.strength
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/{image_name}")
async def get_image(image_name: str):
    """
    Retrieve a generated image by name
    """
    image_path = f"output/{image_name}"
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)

@app.get("/uploads/{filename}")
async def get_uploaded_file(filename: str):
    """
    Retrieve an uploaded file by name
    """
    file_path = f"uploads/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.get("/health")
def health_check():
    """
    Health check endpoint
    """
    addon_status = {}
    
    # Check addon availability
    if hasattr(generator, "addons"):
        addon_status = {
            "controlnet": list(generator.addons.get("controlnet", {}).keys()),
            "instantid": "instantid" in generator.addons,
            "ip_adapter": "ip-adapter" in generator.addons,
            "reactor": "reactor" in generator.addons,
            "face_restoration": {
                "codeformer": "codeformer" in generator.addons,
                "gfpgan": "gfpgan" in generator.addons
            },
            "textual_inversion": "textual_inversion" in generator.addons,
            "hyperdreambooth": "hyperdreambooth" in generator.addons,
            "insightface": "insightface" in generator.addons
        }
    
    return {
        "status": "healthy",
        "model": "custom_realistic_vision",
        "device": generator.device,
        "addons": addon_status
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 