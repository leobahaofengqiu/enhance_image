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

# Enable CORS for all origins (for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Allowed image formats
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

@app.get("/")
def root():
    return {"message": "Image Enhancer API is running"}

@app.post("/enhance/")
async def enhance_image(file: UploadFile = File(...)):
    # Generate temporary filenames
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    input_filename = f"temp_input_{uuid.uuid4().hex}{ext}"
    output_filename = f"temp_output_{uuid.uuid4().hex}.png"

    try:
        # Save the uploaded image to disk
        with open(input_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"Uploaded file saved as: {input_filename}")

        # Load Hugging Face token
        HF_TOKEN = os.getenv("HF_TOKEN")
        if not HF_TOKEN:
            raise HTTPException(status_code=500, detail="HF_TOKEN not set in environment")

        # Connect to Hugging Face Gradio model
        client = Client("smartfeed/image_hd", hf_token=HF_TOKEN)

        # Run prediction
        try:
            result = client.predict(
                input_image=handle_file(input_filename),
                scale=2,
                enhance_mode="Face Enhance + Image Enhance",
                api_name="/enhance_image"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model API failed: {e}")

        # Validate model response
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            raise HTTPException(status_code=500, detail="Invalid response from model")

        enhanced_image_path = result[1]

        # Convert relative path to full URL
        if enhanced_image_path.startswith("/"):
            enhanced_image_url = f"https://smartfeed-image-hd.hf.space/file={enhanced_image_path}"
        else:
            enhanced_image_url = enhanced_image_path

        logging.info(f"Downloading enhanced image from: {enhanced_image_url}")

        # Download the enhanced image
        img_response = requests.get(enhanced_image_url)
        if img_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch enhanced image")

        with open(output_filename, "wb") as out:
            out.write(img_response.content)

        # Read and return enhanced image to frontend
        with open(output_filename, "rb") as final_image:
            return Response(content=final_image.read(), media_type="image/png")

    except Exception as err:
        logging.error(f"Processing failed: {err}")
        raise HTTPException(status_code=500, detail=str(err))

    finally:
        # Always clean up temp files
        for path in [input_filename, output_filename]:
            if os.path.exists(path):
                os.remove(path)
