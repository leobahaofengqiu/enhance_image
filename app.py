import os
import shutil
import uuid
import logging
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, handle_file
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Allow all CORS origins (adjust for production if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Allowed image extensions
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

@app.get("/")
def root():
    return {"message": "Image Enhancer API is running"}

@app.post("/enhance/")
async def enhance_image(file: UploadFile = File(...)):
    # Generate unique filenames
    temp_input_filename = f"temp_input_{uuid.uuid4().hex}"
    temp_output_filename = f"temp_output_{uuid.uuid4().hex}.png"

    # Get original extension (e.g., .jpg, .png)
    input_ext = os.path.splitext(file.filename)[-1].lower()
    input_path = temp_input_filename + input_ext

    # Validate extension
    if input_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        # Save uploaded file to disk
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logging.info(f"File saved: {input_path}")

        # Get Hugging Face token
        HF_TOKEN = os.getenv("HF_TOKEN")
        if not HF_TOKEN:
            raise HTTPException(status_code=500, detail="HF_TOKEN not set in environment")

        # Initialize Gradio client
        client = Client("gokaygokay/Tile-Upscaler", hf_token=HF_TOKEN)

        try:
            result_urls = client.predict(
                param_0=handle_file(input_path),
                param_1=512,   # Resolution
                param_2=20,    # Inference steps
                param_3=0.07,   # Strength
                param_4=0,     # HDR effect
                param_5=3,     # Guidance scale
                api_name="/wrapper"
            )
        except Exception as model_error:
            raise HTTPException(status_code=500, detail=f"Gradio API call failed: {model_error}")

        # Handle result: could be list or str
        if isinstance(result_urls, list):
            result_url = result_urls[0]  # Use first result
        else:
            result_url = result_urls

        logging.info(f"Downloading result from: {result_url}")

        # Download enhanced image
        response = requests.get(result_url)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to download enhanced image")

        # Save result
        with open(temp_output_filename, "wb") as f:
            f.write(response.content)

        # Return image as response
        with open(temp_output_filename, "rb") as f:
            return Response(content=f.read(), media_type="image/png")

    except Exception as e:
        logging.error(f"Enhancement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up all temp files
        for path in [input_path, temp_output_filename]:
            if os.path.exists(path):
                os.remove(path)
