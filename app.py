import os
import shutil
import uuid
import logging
import requests
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)

# FastAPI app
app = FastAPI()

# Enable CORS
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
async def enhance_image(
    file: UploadFile = File(...),
    version: str = Form("v1.4"),
    scale: float = Form(20)
):
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    input_path = f"temp_input_{uuid.uuid4().hex}{ext}"

    try:
        # Save uploaded image
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"Image saved at: {input_path}")

        # Prepare request to HuggingFace Space
        api_url = "https://rakibulbd030-old-photo-restoration.hf.space/api/predict"
        payload = {
            "data": [
                input_path,  # file path
                version,     # model version string
                scale        # rescaling factor
            ]
        }

        response = requests.post(api_url, json=payload)
        response.raise_for_status()

        result = response.json()
        output_image_path = result.get("data", [None, None])[1]

        if not output_image_path or not os.path.exists(output_image_path):
            raise Exception("Output image not generated or path missing.")

        # Return image as response
        with open(output_image_path, "rb") as out_file:
            image_bytes = out_file.read()

        return Response(content=image_bytes, media_type="image/png")

    except Exception as e:
        logging.error(f"Enhancement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up input and output files
        for path in [input_path]:
            if os.path.exists(path):
                os.remove(path)
