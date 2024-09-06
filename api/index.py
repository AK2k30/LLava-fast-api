import torch
from transformers import pipeline, BitsAndBytesConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from PIL import Image
from io import BytesIO

# Set up device (CPU or GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configure quantization if using GPU
if device == "cuda":
    print("GPU found. Using 4-bit quantization.")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
else:
    print("GPU not found. Using CPU with default settings.")
    quantization_config = None

# Load model pipeline
model_id = "bczhou/tiny-llava-v1-hf"
pipe = pipeline("image-to-text", model=model_id, device=device)

print(f"Using device: {device}")

# Initialize FastAPI application
app = FastAPI()

# Define Pydantic model for request input
class ImagePromptInput(BaseModel):
    image_url: str
    prompt: str

# FastAPI route for generating text from an image
@app.post("/generate")
async def generate_text(input_data: ImagePromptInput):
    try:
        # Download and process the image
        response = requests.get(input_data.image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image = image.resize((750, 500))  # Resize image to fixed dimensions

        # Create a full prompt to pass to the model
        full_prompt = f"USER: <image>\n{input_data.prompt}\nASSISTANT: "

        # Generate response using the model pipeline
        outputs = pipe(image, prompt=full_prompt, generate_kwargs={"max_new_tokens": 200})

        # Return generated text
        generated_text = outputs[0]['generated_text'] #type: ignore
        return {"response": generated_text}

    except Exception as e:
        # Return error if something goes wrong
        raise HTTPException(status_code=500, detail=str(e))
