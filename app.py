import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, handle_file
import shutil
import uuid

app = FastAPI()

# Allow all origins for Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/enhance/")
async def enhance_image(file: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4().hex}.png"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ✅ Get token from Railway environment variable
    HF_TOKEN = os.getenv("HF_TOKEN")

    # Create Hugging Face client with token
    client = Client("gokaygokay/Tile-Upscaler", hf_token=HF_TOKEN)

    result = client.predict(
        param_0=handle_file(temp_filename),
        param_1=512,
        param_2=20,
        param_3=0.4,
        param_4=0,
        param_5=3,
        api_name="/wrapper"
    )

    os.remove(temp_filename)
    return {"enhanced_url": result}
