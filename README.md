# Realistic Vision API

An API server for generating realistic images using a custom Stable Diffusion model with various addons.

## Features

- Text-to-image generation with a custom Realistic Vision model
- ControlNet guidance (Canny, Depth, OpenPose, Segmentation, DensePose)
- InstantID for face-guided generation
- IP-Adapter for image-prompted generation
- Image-to-image transformation
- Inpainting
- Face swapping with ReActor
- Face restoration with CodeFormer and GFPGAN
- Textual Inversion for custom styles

## Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Custom Stable Diffusion model in `./model` directory

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/realistic-vision-api.git
cd realistic-vision-api
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download addon models
```bash
python download_addons.py --output_dir ./addons
```

You can also download specific addons:
```bash
python download_addons.py --output_dir ./addons --addons controlnet instantid ip-adapter
```

Available addon options:
- `controlnet`: ControlNet models for Canny, Depth, OpenPose, and Segmentation
- `instantid`: InstantID for face-guided generation
- `ip-adapter`: IP-Adapter for image-prompted generation
- `insightface`: InsightFace for face analysis
- `face-restoration`: CodeFormer and GFPGAN for face restoration
- `reactor`: ReActor for face swapping
- `textual-inversion`: Textual Inversion embeddings

4. Place your custom Stable Diffusion model in the `./model` directory

## Running the Server

Start the API server:
```bash
python app.py
```

The server will run at `http://localhost:8000` by default.

## API Endpoints

### Main Endpoints

- `POST /generate`: Text-to-image generation
- `POST /generate/controlnet`: Generation with ControlNet
- `POST /generate/instantid`: Generation with InstantID
- `POST /generate/ip-adapter`: Generation with IP-Adapter
- `POST /generate/img2img`: Image-to-image generation
- `POST /generate/inpaint`: Inpainting

### Post-processing Endpoints

- `POST /post-process/face-swap`: Face swapping with ReActor
- `POST /post-process/face-restoration`: Face restoration with CodeFormer or GFPGAN

### Utility Endpoints

- `POST /upload`: Upload images for processing
- `GET /addons`: List available addon models
- `GET /health`: Check server health and available addons

## Web Interface

The API includes a built-in web interface for easy interaction. Access it by navigating to `http://localhost:8000` in your web browser.

The interface allows you to:
- Generate images with text prompts
- Use ControlNet for guided generation
- Apply InstantID for face-guided generation
- Use IP-Adapter for image-prompted generation
- Perform image-to-image transformations
- Do inpainting
- Apply post-processing (face swapping and restoration)

## Example Usage (Python)

```python
import requests
import json

# Text-to-image generation
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "A beautiful landscape with mountains and a lake, realistic",
        "negative_prompt": "blurry, low quality",
        "width": 768,
        "height": 768,
        "num_inference_steps": 30,
        "guidance_scale": 7.5
    }
)

result = response.json()
print(f"Image generated at: {result['image_url']}")
```

## License

This project is provided for research and personal use only. Please ensure you comply with the licenses of all underlying models and components. 