import os
import uuid
import shutil
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from gradio_client import Client, handle_file

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

# Initialize Gradio client
client = Client("smartfeed/image_hd")

@app.get("/")
def root():
    return {"message": "Image Enhancer API is running"}

@app.post("/enhance/")
async def enhance_image(
    file: UploadFile = File(...),
    scale: float = Form(2),
    enhance_mode: str = Form("Face Enhance + Image Enhance")  # Options: Only Face Enhance, Only Image Enhance, Face Enhance + Image Enhance
):
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    input_path = f"temp_input_{uuid.uuid4().hex}{ext}"

    try:
        # Save uploaded image temporarily
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"Image saved at: {input_path}")

        # Use gradio_client to enhance image
        result = client.predict(
            input_image=handle_file(input_path),
            scale=scale,
            enhance_mode=enhance_mode,
            api_name="/enhance_image"
        )

        output_url = result[0].get("url")
        if not output_url:
            raise Exception("No output image returned from API.")

        # Fetch the enhanced image from URL
        response = requests.get(output_url)
        response.raise_for_status()

        return Response(content=response.content, media_type="image/png")

    except Exception as e:
        logging.error(f"Enhancement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup input file
        if os.path.exists(input_path):
            os.remove(input_path)
