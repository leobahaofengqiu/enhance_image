import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, handle_file
from dotenv import load_dotenv

# Load environment variables from .env (optional for local testing)
load_dotenv()

app = FastAPI()

# CORS settings (allow all origins for Flutter app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/enhance/")
async def enhance_image(file: UploadFile = File(...)):
    try:
        # Save uploaded file to a temp file
        temp_filename = f"temp_{uuid.uuid4().hex}.png"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Get Hugging Face API Token from env
        HF_TOKEN = os.getenv("HF_TOKEN")
        if not HF_TOKEN:
            raise HTTPException(status_code=500, detail="HF_TOKEN not set")

        # Create Hugging Face client
        client = Client("gokaygokay/Tile-Upscaler", hf_token=HF_TOKEN)

        # Call the model
        result = client.predict(
            param_0=handle_file(temp_filename),
            param_1=512,   # Resolution
            param_2=20,    # Inference steps
            param_3=0.4,   # Strength
            param_4=0,     # HDR effect
            param_5=3,     # Guidance scale
            api_name="/wrapper"
        )

        return {"enhanced_url": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
