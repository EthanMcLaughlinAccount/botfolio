from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = FastAPI()

# Public model — no token needed
MODEL_NAME = "neuraxcompany/gpt2-botfolio"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Request body schema
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 150

# Root endpoint (optional health check)
@app.get("/")
def root():
    return {"status": "Botfolio is running ✅"}

# Generation endpoint
@app.post("/generate")
def generate_text(request: GenerateRequest):
    output = generator(
        request.prompt,
        max_length=request.max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    return {"generated_text": output[0]["generated_text"]}
