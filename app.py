import os
import shutil
import uuid
import logging
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, handle_file
from dotenv import load_dotenv

# Load .env variables (e.g., HF_TOKEN)
load_dotenv()

# Enable logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# CORS for frontend access
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
    # Validate and prepare temp filenames
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    input_filename = f"temp_input_{uuid.uuid4().hex}{ext}"
    output_filename = f"temp_output_{uuid.uuid4().hex}.png"

    try:
        # Save uploaded image locally
        with open(input_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"Uploaded file saved as: {input_filename}")

        # Load Hugging Face token
        HF_TOKEN = os.getenv("HF_TOKEN")
        if not HF_TOKEN:
            raise HTTPException(status_code=500, detail="HF_TOKEN not set in environment")

        # Initialize Gradio Client
        client = Client("smartfeed/image_hd", hf_token=HF_TOKEN)

        # Submit job to model queue
        logging.info("Submitting job to model...")
        job = client.submit(
            input_image=handle_file(input_filename),
            scale=2,
            enhance_mode="Face Enhance + Image Enhance",
            api_name="/enhance_image"
        )

        # Wait for result (timeout after 5 min)
        result = job.result(timeout=300)
        logging.info("Received result from model.")

        # Expecting a list/tuple with at least 2 values
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            raise HTTPException(status_code=500, detail="Invalid response from model")

        enhanced_image_path = result[1]

        # Build absolute URL if path is relative
        if enhanced_image_path.startswith("/"):
            enhanced_image_url = f"https://smartfeed-image-hd.hf.space/file={enhanced_image_path}"
        else:
            enhanced_image_url = enhanced_image_path

        logging.info(f"Downloading from: {enhanced_image_url}")
        response = requests.get(enhanced_image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to download enhanced image")

        # Save final image
        with open(output_filename, "wb") as f:
            f.write(response.content)

        # Return image as binary response
        with open(output_filename, "rb") as f:
            return Response(content=f.read(), media_type="image/png")

    except Exception as e:
        logging.error(f"Enhancement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temporary files
        for path in [input_filename, output_filename]:
            if os.path.exists(path):
                os.remove(path)
