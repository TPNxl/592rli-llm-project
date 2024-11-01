import torch
from transformers import pipeline
from huggingface_hub import login

model_id = "meta-llama/Llama-3.2-3B"

with open("./tokens/hf_token.txt", 'r') as f:
    token = f.read().strip()

login(token=token)

pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
)

print(pipe("The key to life is"))
