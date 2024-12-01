import asyncio
import random
import os
import torch
from convo_dataset_batched import Convo_Dataset
from process_reward import *
import subprocess

def get_last_trained_model():
    max_n = -1
    for fn in os.listdir("./models"):
        if fn.endswith(".pt").startswith("llama_trained_epoch_"):
            epoch = int(fn.split("_")[3].split(".")[0])
            max_n = max(max_n, epoch)
    return os.path.join("./models", f"llama_trained_epoch_{max_n}.pt")

def get_last_feedback_buf():
    max_n = -1
    for fn in os.listdir("./feedback_bufs"):
        if fn.endswith(".pt").startswith("feedback_buf_epoch_"):
            epoch = int(fn.split("_")[3].split(".")[0])
            max_n = max(max_n, epoch)
    return os.path.join("./feedback_bufs", f"feedback_buf_epoch_{max_n}.pt")

def get_last_wsd():
    max_n = -1
    for fn in os.listdir("./datasets/"):
        if fn.endswith(".pt").startswith("wsd_epoch_"):
            epoch = int(fn.split("_")[3].split(".")[0])
            max_n = max(max_n, epoch)
    return os.path.join("./datasets/", f"wsd_epoch_{max_n}.pt")

def get_base_model():
    return "meta-llama/Llama-3.2-3B-Instruct"

def gpt_generate(messages: dict):
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
    )
    out = chat_completion.choices[0].message.content
    out = out.split(": ")[-1]
    return out

# Dummy function to simulate cmd execution
async def dataset_generation(epoch):
    curr_pipe = pipeline(
        "text-generation",
        model=get_last_trained_model(),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    base_pipe = pipeline(
        "text-generation",
        model=get_base_model(),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    def curr_pipe_generate(messages: dict):
        outputs = curr_pipe(
            messages,
            max_new_tokens=256,
            pad_token_id=None,
        )
        return outputs[0]["generated_text"][-1]
    
    def base_pipe_generate(messages: dict):
        outputs = base_pipe(
            messages,
            max_new_tokens=256,
            pad_token_id=None,
        )
        return outputs[0]["generated_text"][-1]
    
    with open(get_last_feedback_buf(), "rb") as f:
        feedback_buf = torch.load(f)
    
    print(f"Starting dataset generation, epoch {epoch}")
    cd = Convo_Dataset(item_path=f"./datasets/convo_epoch{epoch}/", 
                       llama_generate_func_new=curr_pipe_generate, 
                       llama_generate_func_base=base_pipe_generate, 
                       gpt_generate_func=gpt_generate, 
                       feedback_buf=feedback_buf)
    cd.generate()

    with open(f"./feedback_bufs/feedback_buf_epoch_{epoch}.pt", "wb") as f:
        torch.save(cd.feedback_buf, f)

    wsd = WinnerSeparatedDataset(cd)
    wsd.process()
    wsd.save(f"./run{epoch}/epoch{epoch}/wsd_epoch_{epoch}.pt")

async def training(epoch):
    # Reward model training

    subprocess.run([
        "python", "reward_modeling.py",
        "--model_name_or_path", "Qwen/Qwen2-0.5B-Instruct",
        "--dataset_name", "trl-lib/ultrafeedback_binarized",
        "--output_dir", "Qwen2-0.5B-Reward",
        "--per_device_train_batch_size", "8",
        "--num_train_epochs", "1",
        "--gradient_checkpointing", "True",
        "--learning_rate", "1.0e-5",
        "--logging_steps", "25",
        "--eval_strategy", "steps",
        "--eval_steps", "50",
        "--max_length", "2048"
    ])

    subprocess.run([
        "python", "ppo_tldr.py",
        "--dataset_name", "trl-internal-testing/tldr-preference-sft-trl-style",
        "--learning_rate", "3e-6",
        "--output_dir", "models/minimal/ppo_tldr",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "64",
        "--total_episodes", "30000",
        "--model_name_or_path", "EleutherAI/pythia-1b-deduped",
        "--sft_model_path", "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr",
        "--reward_model_path", "cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr",
        "--missing_eos_penalty", "1.0",
        "--stop_token", "eos",
        "--response_length", "53",
        "--eval_strategy", "steps",
        "--eval_steps", "100"
    ])




# Function to manage sequential execution for a single input
async def process_input(input_number, cmd1_semaphore, cmd2_semaphore):
    async with cmd1_semaphore:
        await dataset_generation()
    async with cmd2_semaphore:
        await training()

# Main function to handle the processing pipeline
async def main(input_numbers):
    # Semaphores to manage concurrency
    cmd1_semaphore = asyncio.Semaphore(1)  # Only one cmd1 runs at a time
    cmd2_semaphore = asyncio.Semaphore(2)  # cmd2 can overlap with cmd1 of the next number

    # Schedule all tasks
    tasks = [
        process_input(num, cmd1_semaphore, cmd2_semaphore)
        for num in input_numbers
    ]
    await asyncio.gather(*tasks)

# Input data
input_numbers = [1, 2, 3, 4, 5]

# Run the main loop
asyncio.run(main(input_numbers))