from message_generator import *
from text_dataset import *
import torch
import numpy as np
import os
from transformers import pipeline, AutoTokenizer
from huggingface_hub import login
from openai import OpenAI, AsyncOpenAI
import asyncio
from datetime import datetime
import time

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

a_client = AsyncOpenAI(api_key=openai_token)
c_client = OpenAI(api_key=openai_token)

# Asynchronous function to handle each request
async def gpt_response(messages: dict, model="gpt-4o-mini"):
    response = await a_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content

# Main function to gather all responses asynchronously
async def batch_requests(prompts, model):
    tasks = [gpt_response(prompt, model) for prompt in prompts]
    return await asyncio.gather(*tasks)

# Run the asynchronous function to get responses for all prompts
def gpt_generate(prompts, model="gpt-4o-mini", print_debug=False):
    if print_debug:
        print(f"Input to {model}: {prompts}")
    outputs = asyncio.run(batch_requests(prompts, model))
    if print_debug:
        print(f"Output from {model}: {outputs}")
    return outputs

def gpt_generate_old(messages: dict, model="gpt-4o-mini", print_debug=False):
    if print_debug:
        print(f"Input to {model}: {messages}")
    chat_completion = c_client.chat.completions.create(
        messages=messages,
        model=model,
    )
    out = chat_completion.choices[0].message.content
    if print_debug:
        print(f"Output from {model}: {chat_completion.choices[0].message.content}")
    return out

class Convo_Dataset(torch.utils.data.Dataset):
    def __init__(self, item_path="./convos/", 
                 llama_new=None, 
                 llama_base=None, 
                 feedback_buf=None,
                 topics_path="./prompts/topics.txt",
                 generate_ranking=True,
                 generate_feedback=True,
                 print_debug=False,
                 p_feedback_out = 0.2,
                 device="cuda:0"):
        self.item_path = item_path
        if not os.path.exists(item_path):
            os.mkdir(item_path)

        self.llama_base = llama_base
        self.tokenizer = None if self.llama_base is None else AutoTokenizer.from_pretrained(self.llama_base)
        self.llama_base_pipe = None if self.llama_base is None else pipeline(
            "text-generation",
            model=self.llama_base,
            torch_dtype=torch.bfloat16,
            device=device,
            pad_token_id=self.tokenizer.eos_token_id
        )

        self.llama_new = llama_new if llama_new is not None else self.llama_base
        self.llama_new_pipe = self.llama_base_pipe if llama_new is None else pipeline(
            "text-generation",
            model=self.llama_new,
            torch_dtype=torch.bfloat16,
            device=device,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        self.topics_dataset = TextDataset(topics_path)
        self.TOPICS_LEN = len(self.topics_dataset)

        self.generate_ranking = generate_ranking
        self.generate_feedback = generate_feedback

        self.feedback_buf = feedback_buf
        if feedback_buf is not None:
            self.FEEDBACK_BUF_LEN = len(feedback_buf)
        self.p_feedback_out = p_feedback_out
        self.out_feedback = []
        now = datetime.now()
        self.r = np.random.RandomState(int(now.strftime("%m%d%H%M")))
        self.convos = []

        self.print_debug = print_debug

    def gen_base(self, messages, length=256, print_debug=False):
        if print_debug:
            print(f"Input to llama_base: {messages}")
        outputs = self.llama_base_pipe(
            messages,
            max_new_tokens=length,
        )
        out = [None] * len(outputs)
        for i in range(len(outputs)):
            out[i] = outputs[i][0]["generated_text"][-1]
        if print_debug:
            print(f"Output from llama_base: {out}")
        return out

    def gen_new(self, messages, length=256, print_debug=False):
        if print_debug:
            print(f"Input to llama_new: {messages}")
        outputs = self.llama_new_pipe(
            messages,
            max_new_tokens=length,
        )
        out = [None] * len(outputs)
        for i in range(len(outputs)):
            out[i] = outputs[i][0]["generated_text"][-1]
        if print_debug:
            print(f"Output from llama_new: {out}")
        return out
           
    def generate(self, num_elems=None, debate_len = 10, bsz=16, print_debug=None):
        debate_len = debate_len - debate_len % 2
        if print_debug is None:
            print_debug = self.print_debug
        alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        bsz_queue = []
        n = min(num_elems, self.TOPICS_LEN) if num_elems is not None else self.TOPICS_LEN
        while n > 0:
            bsz_queue.append(min(bsz, n))
            n -= bsz
        dataset_elem = r.randint(0, self.TOPICS_LEN)
        if print_debug:
            print(f"bsz_queue: len={len(bsz_queue)} | {bsz_queue}")
        for b in bsz_queue:
            t = time.time()
            if print_debug:
                print(f"starting generation on dataset_elem: {dataset_elem}")
            gens = []
            NUM_AGENTS=2
            # Setup
            for i in range(b):
                item = self.topics_dataset[dataset_elem]
                topic = item[0]
                v = item[1]
                m = MessageGenerator()
                m.reset()
                m.topic = topic
                random_letters = self.r.choice(alphabet, 2, replace=False)
                if self.feedback_buf is not None and self.FEEDBACK_BUF_LEN > 0:
                    m.new_agent(random_letters[0], v[0], None, self.feedback_buf[self.r.randint(0, self.FEEDBACK_BUF_LEN)])
                else:
                    m.new_agent(random_letters[0], v[0], None)
                m.new_agent(random_letters[1], v[1], None)
                if r.binomial(1, 0.5) < 0.5:
                    quality = None
                    evaluation = None
                else:
                    quality = self.r.choice(QUALITY_BUF)
                    evaluation = f"Who was more {quality}?"
                while True:
                    m.desc = str(dataset_elem)
                    if os.path.isfile(os.path.join(self.item_path, f"{m.desc}.hlst")):
                        dataset_elem += 1
                        dataset_elem %= self.TOPICS_LEN
                    else:
                        break
                gens.append(m)
                dataset_elem += 1
                dataset_elem %= self.TOPICS_LEN
            
            if print_debug:
                print(f"picking up generation next time on dataset_elem: {dataset_elem}")
                print()
                print("Debate topics")
                print([g.topic for g in gens])

            # Start
            for i in range(NUM_AGENTS):
                prompts = []
                for j in range(b):
                    g = gens[j]
                    agent = g.agents[i]
                    msgs = g.generate_starting_view(FIRST_PROMPT, agent, g.agents[(i+1)%NUM_AGENTS], g.agent_views[agent])
                    prompts.append(msgs)
                if i == 0:
                    outputs = self.gen_base(prompts)
                else:
                    outputs = self.gen_new(prompts)
                for j in range(b):
                    g = gens[j]
                    agent = g.agents[i]
                    g.append(agent, outputs.pop(0))
            
            # Generate debate
            for n in range(debate_len):
                print("Debate round", n)
                prompts = []
                for i in range(b):
                    g = gens[i]
                    agent_name = g.agents[n%NUM_AGENTS]
                    opp_name = g.agents[(n+1)%NUM_AGENTS]
                    msgs = g.generate_debate_prompt(CONTINUE_PROMPT, agent_name, opp_name, s_prompt_2=None)
                    prompts.append(msgs)
                if n % 2 == 0:
                    outputs = self.gen_base(prompts)
                else:
                    outputs = self.gen_new(prompts)
                for i in range(b):
                    g = gens[i]
                    agent_name = g.agents[n%NUM_AGENTS]
                    agent_output = outputs.pop(0)
                    agent_output['content'] = agent_output['content'].replace(f"{agent_name}: ", "").replace("\"", "")
                    g.append(agent_name, agent_output)

            # Generate ranking
            if self.generate_ranking:
                t3 = time.time()
                if print_debug:
                    print("Generating ranking")
                prompts = []
                for i in range(b):
                    g = gens[i]
                    msgs = g.generate_ranking_prompt(EVAL_PROMPT, quality, evaluation)
                    prompts.append(msgs)
                outputs = self.gen_new(prompts, length=512)
                for i in range(b):
                    g = gens[i]
                    out = outputs.pop(0)
                    out['content'] = out['content'].split(": ")[-1]
                    out['role'] = "system"
                    g.set_winner_from_prompt(out['content'])
                    g.append("Judge", out)
                
                t4 = time.time()

                # Generate feedback
                if self.generate_feedback and self.r.binomial(1, self.p_feedback_out) == 1:
                    if print_debug:
                        print("Generating feedback")
                    prompts = []
                    for i in range(b):
                        msgs = g.generate_feedback_prompt(FEEDBACK_PROMPT, quality, evaluation)
                        prompts.append(msgs)
                    outputs = gpt_generate(prompts, print_debug=print_debug)
                    self.out_feedback.extend(outputs)
                    for i in range(b):
                        g.feedback = outputs.pop(0)
                        g.append("Judge", {"role": "system", "content": g.feedback})
                else:
                    for i in range(b):
                        g.feedback = ""
                        g.append("Judge", {"role": "system", "content": "No feedback given."})
                t5 = time.time()

            # Save
            for g in gens:
                path = os.path.join(self.item_path, f"{g.desc}.hlst")
                print(f"Saving item {g.desc} to {path}")
                g.save(path)

            path = os.path.join(self.item_path, f"feedback.fbuf")
            if os.path.isfile(path):
                os.remove(path)
            with open(path, "wb") as f:
                torch.save(self.out_feedback, f)
                    
            t2 = time.time()
            print(f"Time to generate {b} conversations: {t2-t} seconds")
            print(f"Average time per conversation: {(t2-t)/b} seconds")
            if self.generate_ranking:
                print(f"Time to generate ranking: {t4-t3} seconds")
                if self.generate_feedback:
                    print(f"Time to generate feedback: {t5-t4} seconds")
       

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
    cd = Convo_Dataset()
    path = "./datasets/convo_test/"
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))
    cd = Convo_Dataset(item_path=path, llama_base="meta-llama/Llama-3.2-3B-Instruct", print_debug=False)
    cd.generate(num_elems=2, debate_len=2)