from message_generator import *
from text_dataset import *
import torch
import numpy as np
import os

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
    QUALITY_BUF_LEN = len(QUALITY_BUF)

class Convo_Dataset(torch.utils.data.Dataset):
    def __init__(self, item_path="./convos/", llama_generate_func=None, gpt_generate_func=None, topics_path="./prompts/topics.txt", seed=12345):
        self.item_path = item_path
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
            m = MessageGenerator(topic = topic)
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
            m.generate_debate(CONTINUE_PROMPT, debate_len)
            m.generate_ranking(EVAL_PROMPT, self.llama_generate_func, quality, evaluation)
            m.generate_feedback(FEEDBACK_PROMPT, self.gpt_generate_func, quality, evaluation)
            while True:
                index = self.r.randint(0, np.power(2,16))
                path = os.path.join(self.item_path, f"{header_name}{index}.hlst")
                if not os.path.exists(path):
                    m.save(path)
                    break

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
        
        