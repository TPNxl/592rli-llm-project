from message_generator import *
from transformers import pipeline
from huggingface_hub import login
from openai import OpenAI

with open("./tokens/hf_token.txt", 'r') as f:
    hf_token = f.read().strip()
with open("./tokens/openai_token.txt", 'r') as f:
    openai_token = f.read().strip()
with open("./prompts/first_prompt.txt", 'r') as f:
    first_prompt = f.read().replace("\n", " ")
with open("./prompts/continue_prompt.txt", 'r') as f:
    continue_prompt = f.read().replace("\n", " ")
with open("./prompts/feedback_prompt.txt", 'r') as f:
    feedback_prompt = f.read().replace("\n", " ")
with open("./prompts/eval_prompt.txt", 'r') as f:
    eval_prompt = f.read().replace("\n", " ")
with open("./prompts/quality_buf.txt", 'r') as f:
    quality_buf = f.readlines()

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def llama_generate(messages: dict):
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    return outputs[0]["generated_text"][-1]

client = OpenAI(api_key=openai_token)

def gpt_generate(messages: dict):
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
    )
    out = chat_completion.choices[0].message.content
    out = out.split(": ")[-1]
    print(out)
    return out

m = MessageGenerator(topic = "fruits")

m.new_agent('A', 'bananas > oranges', gpt_generate)
m.new_agent('B', 'oranges > bananas', gpt_generate)


