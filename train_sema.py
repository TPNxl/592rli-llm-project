import asyncio
import random
import os
import torch
from convo_dataset_batched import Convo_Dataset
from process_reward import *
import subprocess

MODEL_DIR = "./model_weights"
DATA_DIR = "./datasets"

def get_last_trained_model():
    max_n = -1
    for fn in os.listdir(MODEL_DIR):
        if fn.endswith(".pt").startswith("llama_trained_epoch_"):
            epoch = int(fn.split("_")[3].split(".")[0])
            max_n = max(max_n, epoch)
    return os.path.join(MODEL_DIR, f"llama_trained_epoch_{max_n}.pt") if max_n != -1 else get_base_model()

def get_last_reward_model():
    max_n = -1
    for fn in os.listdir(MODEL_DIR):
        if fn.endswith(".pt").startswith("reward_model_epoch_"):
            epoch = int(fn.split("_")[3].split(".")[0])
            max_n = max(max_n, epoch)
    return os.path.join(MODEL_DIR, f"reward_model_epoch_{max_n}.pt") if max_n != -1 else get_base_reward_model()

def get_last_value_model():
    max_n = -1
    for fn in os.listdir(MODEL_DIR):
        if fn.endswith(".pt").startswith("value_model_epoch_"):
            epoch = int(fn.split("_")[3].split(".")[0])
            max_n = max(max_n, epoch)
    return os.path.join(MODEL_DIR, f"value_model_epoch_{max_n}.pt") if max_n != -1 else get_base_reward_model()

def get_last_feedback_buf():
    max_n = -1
    for fn in os.listdir(DATA_DIR):
        if fn.endswith(".fbuf"):
            epoch = int(fn.split("_")[3].split(".")[0])
            max_n = max(max_n, epoch)
    return os.path.join(DATA_DIR, f"feedback_buf_epoch_{max_n}.pt") if max_n != -1 else None

def get_last_wsd():
    max_n = -1
    for fn in os.listdir(DATA_DIR):
        if fn.endswith(".pt").startswith("wsd_epoch_"):
            epoch = int(fn.split("_")[3].split(".")[0])
            max_n = max(max_n, epoch)
    return os.path.join(DATA_DIR, f"wsd_epoch_{max_n}.pt") if max_n != -1 else None

def get_base_model():
    return "meta-llama/Llama-3.2-3B-Instruct"

def get_base_reward_model():
    return "Qwen/Qwen2-0.5B-Instruct"

# Dummy function to simulate cmd execution
# def dataset_generation(epoch):
async def dataset_generation(epoch):
    fb_path = get_last_feedback_buf()
    if fb_path is None:
        feedback_buf = []
    else:
        with open(get_last_feedback_buf(), "rb") as f:
            feedback_buf = torch.load(f)
        
    print(f"Starting dataset generation, epoch {epoch}")
    cd = Convo_Dataset(item_path=os.path.join(DATA_DIR, f"convo_epoch{epoch}/"), 
                       llama_new=get_last_trained_model(), 
                       llama_base=get_base_model(),
                       feedback_buf=feedback_buf,
                       generate_feedback=True,
                       generate_ranking=True,
                       device="cuda:0")
    await cd.generate(num_elems=20, bsz=10)

    wsd = WinnerSeparatedDataset(cd)
    await wsd.process()
    wsd.save(os.path.join(DATA_DIR, f"wsd_epoch{epoch}/wsd_epoch_{epoch}.pt"))

# def training(epoch):
async def training(epoch):
    # Reward model training
    await subprocess.run([
        "python", "reward_modeling.py",
        "--model_name_or_path", get_last_reward_model(),
        "--dataset_path", get_last_wsd(),
        "--output_dir", os.path.join(MODEL_DIR, f"reward_model_epoch_{epoch}"),
        "--per_device_train_batch_size", "8",
        "--num_train_epochs", "1",
        "--gradient_checkpointing", "True",
        "--learning_rate", "1.0e-5",
        "--logging_steps", "25",
        "--eval_strategy", "steps",
        "--eval_steps", "50",
        "--max_length", "2048"
    ])

    await subprocess.run([
        "python", "ppo_tldr.py",
        "--dataset_path", get_last_wsd(),
        "--learning_rate", "3e-6",
        "--output_dir", os.path.join(MODEL_DIR, f"llama_trained_epoch_{epoch}"),
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "64",
        "--total_episodes", "30000",
        "--model_name_or_path", get_last_trained_model(),
        "--sft_model_path", get_base_model(),
        "--reward_model_path", get_last_reward_model(),
        "--missing_eos_penalty", "1.0",
        "--stop_token", "eos",
        "--response_length", "53",
        "--eval_strategy", "steps",
        "--eval_steps", "100"
    ])

# Function to manage sequential execution for a single input
async def process_input(num, cmd1_semaphore, cmd2_semaphore):
    print(f"Epoch {num}")
    async with cmd1_semaphore:
        await dataset_generation(num)
    async with cmd2_semaphore:
        await training(num)

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

# Run the main loop
if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)  
    asyncio.run(main(range(5)))