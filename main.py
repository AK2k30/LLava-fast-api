import torch
from transformers import pipeline, BitsAndBytesConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from PIL import Image
from io import BytesIO
from mangum import Mangum  

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    print("GPU found. Using 4-bit quantization.")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
else:
    print("GPU not found. Using CPU with default settings.")
    quantization_config = None

model_id = "bczhou/tiny-llava-v1-hf"
if device == "cuda":
    pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config}, device=device)
else:
    pipe = pipeline("image-to-text", model=model_id, device=device)

print(f"Using device: {device}")

app = FastAPI()

class ImagePromptInput(BaseModel):
    image_url: str
    prompt: str

@app.post("/generate")
async def generate_text(input_data: ImagePromptInput):
    try:
        response = requests.get(input_data.image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image = image.resize((750, 500)) 

        full_prompt = f"USER: <image>\n{input_data.prompt}\nASSISTANT: "

        outputs = pipe(image, prompt=full_prompt, generate_kwargs={"max_new_tokens": 200})

        generated_text = outputs[0]['generated_text']
        return {"response": generated_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

handler = Mangum(app)
