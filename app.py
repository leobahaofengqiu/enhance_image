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

# Allow all CORS origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

@app.get("/")
def root():
    return {"message": "Image Enhancer API is running"}

@app.post("/enhance/")
async def enhance_image(file: UploadFile = File(...)):
    temp_input_filename = f"temp_input_{uuid.uuid4().hex}"
    temp_output_filename = f"temp_output_{uuid.uuid4().hex}.png"

    input_ext = os.path.splitext(file.filename)[-1].lower()
    input_path = temp_input_filename + input_ext

    if input_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        # Save uploaded image to disk
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"Uploaded file saved: {input_path}")

        # Load HF token
        HF_TOKEN = os.getenv("HF_TOKEN")
        if not HF_TOKEN:
            raise HTTPException(status_code=500, detail="HF_TOKEN not set in environment")

        # Connect to smartfeed/image_hd Gradio client
        client = Client("smartfeed/image_hd", hf_token=HF_TOKEN)

        try:
            # Call the model
            result_tuple = client.predict(
                input_image=handle_file(input_path),
                scale=2,
                enhance_mode="Face Enhance + Image Enhance",
                api_name="/enhance_image"
            )
        except Exception as model_error:
            raise HTTPException(status_code=500, detail=f"Gradio API call failed: {model_error}")

        # Ensure result is tuple or list
        if not isinstance(result_tuple, (tuple, list)) or len(result_tuple) < 2:
            raise HTTPException(status_code=500, detail="Unexpected result format from Gradio API")

        # Extract enhanced image URL (second element)
        enhanced_image_path = result_tuple[1]

        # Build full URL if relative path
        if enhanced_image_path.startswith("/"):
            enhanced_image_url = f"https://smartfeed-image-hd.hf.space/file={enhanced_image_path}"
        else:
            enhanced_image_url = enhanced_image_path

        logging.info(f"Downloading enhanced image from: {enhanced_image_url}")

        # Download enhanced image
        response = requests.get(enhanced_image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to download enhanced image")

        with open(temp_output_filename, "wb") as f:
            f.write(response.content)

        # Return image as response
        with open(temp_output_filename, "rb") as f:
            return Response(content=f.read(), media_type="image/png")

    except Exception as e:
        logging.error(f"Enhancement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up
        for path in [input_path, temp_output_filename]:
            if os.path.exists(path):
                os.remove(path)
