import os
import shutil
import uuid
import base64
import logging
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Enable logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Allowed image types
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

@app.get("/")
def root():
    return {"message": "Image Enhancer API is running"}

@app.post("/enhance/")
async def enhance_image(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    input_filename = f"temp_input_{uuid.uuid4().hex}{ext}"
    output_filename = f"temp_output_{uuid.uuid4().hex}.png"

    try:
        # Save the uploaded file
        with open(input_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logging.info(f"Uploaded file saved as: {input_filename}")

        # Convert image to base64
        with open(input_filename, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode("utf-8")
        base64_url = f"data:image/{ext[1:]};base64,{encoded_image}"

        # Payload for POST request
        payload = {
            "data": [
                base64_url,       # Input image in base64 URL
                "Version 1",      # Default version; change if needed
                2.0               # Rescaling factor
            ]
        }

        # Call the HF Space API directly
        api_url = "https://rakibulbd030-old-photo-restoration.hf.space/api/predict"
        response = requests.post(api_url, json=payload)
        response.raise_for_status()

        output_data = response.json().get("data")
        if not output_data or not isinstance(output_data, list):
            raise Exception("Unexpected response structure")

        output_base64_url = output_data[0]  # Base64 output image

        # Decode base64 image
        base64_data = output_base64_url.split(",", 1)[1]
        image_bytes = base64.b64decode(base64_data)

        # Save output image
        with open(output_filename, "wb") as f:
            f.write(image_bytes)

        # Return image
        with open(output_filename, "rb") as f:
            return Response(content=f.read(), media_type="image/png")

    except Exception as e:
        logging.error(f"Enhancement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup
        for path in [input_filename, output_filename]:
            if os.path.exists(path):
                os.remove(path)
