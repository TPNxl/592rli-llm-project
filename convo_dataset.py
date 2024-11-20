from message_generator import *
from text_dataset import *
import torch
import numpy as np
import os
from transformers import pipeline
from huggingface_hub import login
from openai import OpenAI
from datetime import datetime

with open("./tokens/hf_token.txt", 'r') as f:
    hf_token = f.read().strip()
with open("./tokens/openai_token.txt", 'r') as f:
    openai_token = f.read().strip()
with open("./prompts/first_prompt.txt", 'r') as f:
    FIRST_PROMPT = f.read().replace("\n", " ")
with open("./prompts/continue_prompt.txt", 'r') as f:
    CONTINUE_PROMPT = f.read().replace("\n", " ")
with open("./prompts/feedback_prompt.txt", 'r') as f:
    FEEDBACK_PROMPT = f.read().replace("\n", " ")
with open("./prompts/eval_prompt.txt", 'r') as f:
    EVAL_PROMPT = f.read().replace("\n", " ")
with open("./prompts/quality_buf.txt", 'r') as f:
    QUALITY_BUF = f.readlines()
    QUALITY_LEN = len(QUALITY_BUF)

class Convo_Dataset(torch.utils.data.Dataset):
    def __init__(self, item_path="./convos/", llama_generate_func=None, gpt_generate_func=None, topics_path="./prompts/topics.txt", seed=12345):
        self.item_path = item_path
        if not os.path.exists(item_path):
            os.mkdir(item_path)
        self.llama_generate_func = llama_generate_func
        self.gpt_generate_func = gpt_generate_func
        self.topics_dataset = TextDataset(topics_path)
        self.TOPICS_LEN = len(self.topics_dataset)
        self.seed = seed
        self.r = np.random.RandomState(seed)
        self.convos = []
           
    def generate(self, n, header_name="chat", debate_len = 10):
        alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        for _ in range(n):
            item = self.topics_dataset[self.r.randint(0, self.TOPICS_LEN)]
            topic = item[0]
            v = item[1]
            m = MessageGenerator()
            m.reset()
            m.topic = topic
            random_letters = self.r.choice(alphabet, 2, replace=False)
            m.new_agent(random_letters[0], v[0], self.llama_generate_func)
            m.new_agent(random_letters[1], v[1], self.llama_generate_func)
            if r.binomial(1, 0.5) < 0.5:
                quality = None
                evaluation = None
            else:
                quality = self.r.choice(QUALITY_BUF)
                evaluation = f"Who was more {quality}?"

            m.start(FIRST_PROMPT)
            m.generate_debate(CONTINUE_PROMPT, s_prompt_2=None, debate_len=debate_len)
            m.generate_ranking(EVAL_PROMPT, self.llama_generate_func, quality, evaluation)
            m.generate_feedback(FEEDBACK_PROMPT, self.gpt_generate_func, quality, evaluation)
            while True:
                now = datetime.now()
                formatted_string = now.strftime("%m%d%H%M")
                index = self.r.randint(0, np.power(2,8)+100)
                path = os.path.join(self.item_path, f"{header_name}{formatted_string}{index}.hlst")
                if not os.path.exists(path):
                    m.save(path)
                    break
            
            del m
            del random_letters
            del v
            del topic

    def load(self):
        for file in os.listdir(self.item_path):
            if file.endswith(".hlst"):
                m = MessageGenerator()
                m.load(os.path.join(self.item_path, file))
                self.convos.append(m)

    def __len__(self):
        return len(self.convos)
    
    def __getitem__(self, idx):
        return self.convos[idx]
        
if __name__ == "__main__":
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
            pad_token_id=None,
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
        return out
    
    c = Convo_Dataset(llama_generate_func=llama_generate, gpt_generate_func=gpt_generate)
    c.generate(5)