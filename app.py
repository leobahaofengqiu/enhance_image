import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, handle_file
from dotenv import load_dotenv
import requests

# Load environment variables from .env
load_dotenv()

app = FastAPI()

# Allow all CORS origins (adjust for production if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/enhance/")
async def enhance_image(file: UploadFile = File(...)):
    # Generate unique filenames
    temp_input_filename = f"temp_input_{uuid.uuid4().hex}"
    temp_output_filename = f"temp_output_{uuid.uuid4().hex}.png"

    try:
        # Get original extension (e.g., .jpg, .png)
        input_ext = os.path.splitext(file.filename)[-1].lower()
        input_path = temp_input_filename + input_ext

        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Get Hugging Face token
        HF_TOKEN = os.getenv("HF_TOKEN")
        if not HF_TOKEN:
            raise HTTPException(status_code=500, detail="HF_TOKEN not set in environment")

        # Hugging Face Gradio Client
        client = Client("gokaygokay/Tile-Upscaler", hf_token=HF_TOKEN)

        # Call the model
        result_url = client.predict(
            param_0=handle_file(input_path),
            param_1=512,   # Resolution
            param_2=20,    # Inference steps
            param_3=0.4,   # Strength
            param_4=0,     # HDR effect
            param_5=3,     # Guidance scale
            api_name="/wrapper"
        )

        # Download the result image from the URL
        response = requests.get(result_url)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to download enhanced image")

        # Save result to temp file
        with open(temp_output_filename, "wb") as f:
            f.write(response.content)

        # Return the image as response
        with open(temp_output_filename, "rb") as f:
            return Response(content=f.read(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up all temp files
        for path in [input_path, temp_output_filename]:
            if os.path.exists(path):
                os.remove(path)
