from message_generator import *
from text_dataset import *
import torch
import numpy as np
import os
from transformers import pipeline
from huggingface_hub import login
from openai import OpenAI, AsyncOpenAI
import asyncio

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
    def __init__(self, name="epoch1", item_path="./convos/", llama_generate_func=None, gpt_generate_func=None, topics_path="./prompts/topics.txt"):
        self.name = name
        self.item_path = item_path
        if not os.path.exists(item_path):
            os.mkdir(item_path)
        self.llama_generate_func = llama_generate_func
        self.gpt_generate_func = gpt_generate_func
        self.topics_dataset = TextDataset(topics_path)
        self.TOPICS_LEN = len(self.topics_dataset)
        self.r = np.random.RandomState()
        self.convos = []
        self.feedback_buf = []

    # Asynchronous function to handle each request
    async def get_response(prompt):
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
           
    def generate(self, header_name="chat", debate_len = 10, bsz=16):
        alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        bsz_queue = []
        n = self.TOPICS_LEN
        while n > 0:
            bsz_queue.append(min(bsz, n))
            n -= bsz
        dataset_elem = 0
        for b in bsz_queue:
            gens = []
            # Setup
            for i in b:
                item = self.topics_dataset[dataset_elem]
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
                m.desc = str(dataset_elem)
                gens.append(m)
                dataset_elem += 1
            
            print(g.topic for g in gens)

            # Start
            prompts = []
            for i in range(len(gens)):
                g = gens[i]
                for i in range(len(g.agents)):
                    agent = g.agents[i]
                    msgs = g.generate_starting_view(FIRST_PROMPT, agent, g.agents[(i+1)%len(g.agents)], g.agent_views[agent])
                    prompts.append(msgs)
            outputs = self.llama_generate_func(prompts)
            for i in range(len(gens)):
                g = gens[i]
                talking_order = r.permutation(range(0,len(g.agents)))
                for i in talking_order:
                    agent = self.agents[i]
                    g.append(agent, outputs.pop(0))
            
            # Generate debate
            for n in range(debate_len):
                prompts = []
                for i in range(len(gens)):
                    g = gens[i]
                    agent_name = g.agents[g.curr_agent]
                    opp_name = g.agents[(g.curr_agent+1)%len(g.agents)]
                    msgs = g.generate_debate_prompt(CONTINUE_PROMPT, agent_name, opp_name, s_prompt_2=None)
                    prompts.append(msgs)
                outputs = self.llama_generate_func(prompts)
                for i in range(len(gens)):
                    g = gens[i]
                    agent_output = outputs.pop(0)
                    agent_str = agent_output['content']
                    agent_str = agent_str.replace(f"{agent_name}: ", "")
                    agent_str = agent_str.replace("\"", "")
                    agent_output['content'] = agent_str
                    g.append(agent_name, agent_output)
                    g.curr_agent = (self.curr_agent + 1) % len(g.agents)

            # Generate ranking
            prompts = []
            for i in range(len(gens)):
                g = gens[i]
                msgs = g.generate_ranking_prompt(EVAL_PROMPT, quality, evaluation)
                prompts.append(msgs)
            outputs = self.llama_generate_func(prompts)
            for i in range(len(gens)):
                g = gens[i]
                out = self.gpt_generate_func(msgs)
                out['role'] = "system"
                g.set_winner_from_prompt(out['content'])
                g.append("Judge", out)

            # Generate feedback
            prompts = []
            for i in range(len(g.agents)):
                agent = g.agents[i]
                msgs = g.generate_feedback_prompt(FEEDBACK_PROMPT, agent, quality, evaluation)
                prompts.append(msgs)
            outputs = self.gpt_generate_func(prompts)
            for i in range(len(g.agents)):
                agent = g.agents[i]
                out = outputs.pop(0)
                out['role'] = "system"
                g.append("Judge", out)

            # Save
            for g in gens:
                while True:
                    path = os.path.join(self.item_path, f"{header_name}_{self.name}_{g.desc}.hlst")
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
        return outputs[:]["generated_text"][-1]

    client = OpenAI(api_key=openai_token)

    def gpt_generate(messages: dict):
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="gpt-4o-mini",
        )
        out = chat_completion.choices[0].message.content
        out = out.split(": ")[-1]
        return out
    
    client = AsyncOpenAI(api_key=openai_token)
    
    
    
    c = Convo_Dataset(llama_generate_func=llama_generate, gpt_generate_func=gpt_generate)
    c.generate(5)