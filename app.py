import os
import uuid
import shutil
import logging
import requests
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

# Initialize Gradio client for CodeFormer
client = Client("sczhou/CodeFormer")

@app.get("/")
def root():
    return {"message": "CodeFormer Image Enhancement API is running"}

@app.post("/enhance/")
async def enhance_image(
    file: UploadFile = File(...),
    upscale: float = Form(2),
    codeformer_fidelity: float = Form(0.5),
    face_align: bool = Form(True),
    background_enhance: bool = Form(True),
    face_upsample: bool = Form(True),
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

        # Use CodeFormer Gradio client to enhance image
        result = client.predict(
            image=handle_file(input_path),
            face_align=face_align,
            background_enhance=background_enhance,
            face_upsample=face_upsample,
            upscale=upscale,
            codeformer_fidelity=codeformer_fidelity,
            api_name="/predict"
        )

        # result is a filepath (local or URL), fetch the content
        if not result:
            raise Exception("No output returned from CodeFormer API.")

        if result.startswith("http"):
            response = requests.get(result)
            response.raise_for_status()
            content = response.content
        else:
            with open(result, "rb") as f:
                content = f.read()

        return Response(content=content, media_type="image/png")

    except Exception as e:
        logging.error(f"Enhancement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup input file
        if os.path.exists(input_path):
            os.remove(input_path)
