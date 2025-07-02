import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = FastAPI()

# Load HF token
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
MODEL_NAME = "neuraxcompany/gpt2-botfolio"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=hf_token)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 150

@app.post("/generate")
def generate_text(request: GenerateRequest):
    output = generator(
        request.prompt,
        max_length=request.max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    return {"generated_text": output[0]["generated_text"]}
