import torch
import os
from transformers import pipeline
from huggingface_hub import login
from openai import OpenAI
import numpy as np
from typing import Callable, List
from history_list import *

r = np.random.RandomState(42)

MODEL_ID = "meta-llama/Llama-3.2-3B"

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

client = OpenAI(api_key=openai_token)
login(token=hf_token)

def llama(text: str) -> str:
    pipe = pipeline(
        "text-generation", 
        model=MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        max_length=1024,
        truncation=True,
    )
    return pipe(text)

def gpt(text: str) -> str:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": text,
            }
        ],
        model="gpt-4o-mini",
    )
    return chat_completion.choices[0].message.content

def trans_func(x: int) -> int:
    return (x + 1) % 2

def conversation_inference(M: List[Callable], 
                           E: Callable, 
                           T: Callable, 
                           topic: str,
                           v0: str, 
                           v1: str, 
                           feedback_buf: 
                           list[str], 
                           quality_buf: list[str], 
                           pi=0.5, epsilon=0.1,
                           get_feedback = True,
                           save_to_file = False,
                           debug_print = True,
                           epoch_name = None):
    # Feedback buffer
    f = []
    if len(feedback_buf) > 0:
        idx = r.randint(0, len(feedback_buf))
        prob = r.rand()
        if prob < epsilon:
            f.append(feedback_buf[idx])
        else:
            f.append("")
    else:
        f.append("")
    f.append("")

    # Views
    v = [v0, v1]

    # Text storage
    H = HistoryList()
    
    # Init
    i = r.randint(0, 2)
    names = [r.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')) for _ in range(2)]
    for j in range(2):
        prompt = first_prompt.replace("<agent_name>", names[(i+j) % 2]).replace("<opp_name>", names[(i+j+1) % 2]).replace("<topic>", topic).replace("<view>", v[(i+j) % 2])
        print("Prompt:", prompt)
        name = names[(i+j) % 2]
        print("i+j % 2 =", (i+j) % 2)
        response = M[(i+j) % 2](prompt)["generated_text"]
        response = response.split("\"")[-2]
        H.append(name, response)
    # Argue
    for _ in range(10):
        i = r.randint(0, 2)
        prompt = continue_prompt.replace("<agent_name>", names[i]).replace("<opp_name>", names[(i+1) % 2]).replace("<topic>", topic).replace("<view>", v[i]).replace("<conversation>", H.to_string())
        print("Prompt:", prompt)
        name = names[i]
        response = M[i](prompt)["generated_text"]
        response = response.split("\"")[-2]
        H.append(name, response)
    # Feedback
    if r.rand() < pi:
        H.q = "convince the other of their perspective"
        H.e = "Which model better convinced the other of their perspective?"
    else:
        idx = r.randint(0, len(quality_buf))
        q = quality_buf[idx]
        e = f"Which model was more {q} in the conversation?"
    win_name = E(eval_prompt.replace("<conversation>", H.to_string()).replace("<eval>", e)).split("The better-performing agent was ")[0].split("\"")[0]
    H.w = names.index(win_name)
    loser_name = names[(H.w + 1) % 2]

    if get_feedback and not T is None:
        H.feedback = T(feedback_prompt.replace("<loser_name>", loser_name).replace("<quality>", q))
    else:
        H.feedback = None

    H.names = names
    H.views = v

    if save_to_file:
        if get_feedback:
            save_datapoint_to_file(H, epoch_name)

    if debug_print:
        print("Names:", names)
        print("Views:", v)
        print("Quality:", q)
        print("Winner:", win_name)
        print("Feedback:", H.feedback)
    
    return H

def save_datapoint_to_file(H, epoch_name = "epoch_0"):
    n = r.randint(0, 10000000)
    filename = f"./data/{epoch_name}_{n}.txt"
    H.save(filename)

class RLHF_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, epoch_name):
        self.data_dir = data_dir
        self.epoch_name = epoch_name
        self.data = []
        self.load_data()
    
    def load_data(self):
        for filename in os.listdir(self.data_dir):
            if filename.startswith(self.epoch_name):
                H = HistoryList()
                H.load(os.path.join(self.data_dir, filename))
                self.data.append(H)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class RLHF_Streaming_Dataset(torch.utils.data.IterableDataset):
    def __init__(self, data_dir: str, epoch_name: str, get_topics: Callable, topic: str, M: List[Callable], E: Callable, T: Callable, feedback_buf: list[str], quality_buf: list[str], pi=0.5, epsilon=0.1):
        self.data_dir = data_dir
        self.epoch_name = epoch_name
        self.get_topics = get_topics
        self.topic = topic
        self.M = M
        self.E = E
        self.T = T
        self.feedback_buf = feedback_buf
        self.quality_buf = quality_buf
        self.pi = pi
        self.epsilon = epsilon
    
    def __iter__(self):
        while True:
            v0, v1 = self.get_topics(self.topic)
            yield conversation_inference(self.M, 
                                         self.E, 
                                         self.T, 
                                         self.v0, 
                                         self.v1, 
                                         self.feedback_buf, 
                                         self.quality_buf, 
                                         self.pi, self.epsilon, 
                                         get_feedback=True, 
                                         save_to_file=True, 
                                         epoch_name=self.epoch_name)
        


        
def main_loop():
    # Models
    M = [llama, llama]
    E = llama

    # Parameters
    pi = 0.5
    epsilon = 0.1
    gamma = 0.2

    # Buffers
    feedback_buf = []
    new_feedback_buf = []
    replay_buf = []

    #while True:
        # Main loop
        # for _ in range(100):
    T = gpt if r.rand() < gamma else None
    H = conversation_inference(M, E, T, "fruits", "apples are better than oranges", "oranges are better than apples", feedback_buf, quality_buf, pi, epsilon, save_to_file=True, epoch_name="epoch_0")
    replay_buf.append(H)
    if not H.feedback is None:
        new_feedback_buf.append(H.feedback)

    feedback_buf = new_feedback_buf

    print(H)

        # Placeholder for do training with replay buffer

if __name__ == "__main__":
    main_loop()





    
    