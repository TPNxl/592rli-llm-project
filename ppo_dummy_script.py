import asyncio
import os
import torch
from convo_dataset_batched import Convo_Dataset
from process_reward import *
from reward_processor import *
import subprocess

MODEL_DIR = "./temp_model_weights"
REWARD_DIR = "./reward_models"
DATA_DIR = "./datasets"
NUM_ITEMS = 250
DEBATE_LEN = 6

def get_last_trained_model(dir = MODEL_DIR):
    max_n = 0
    for fn in os.listdir(dir):
        if fn.startswith("epoch_"):
            epoch = int(fn.split("_")[1].split(".")[0])
            max_n = max(max_n, epoch)
    return os.path.join(dir, f"epoch_{max_n}") if max_n >= 1 else get_base_model()

def get_base_model():
    return "meta-llama/Llama-3.2-1B-Instruct"

def get_base_reward_model():
    return "Qwen/Qwen2-0.5B-Instruct"

def run(command):
    # Run the command
    result = subprocess.run(command)

    # Print the output and errors
    print("Output:")
    print(result.stdout)
    print("Errors:")
    print(result.stderr)

    return result.stderr

# def training(epoch):
def training(epoch):
    print("#################################### Training PPO model")
    bsz = 1
    while bsz >= 1:
        run([
            "python", "ppo_custom.py",
            "--dataset_name", os.path.join(DATA_DIR, f"dataset_epoch_{epoch}.csv"),
            "--learning_rate", "1e-6",
            "--output_dir", MODEL_DIR,
            "--per_device_train_batch_size", "4",
            "--gradient_accumulation_steps", "1",
            "--total_episodes", "1200",
            "--model_name_or_path", get_base_model(),
            "--sft_model_path", get_base_model(),
            "--reward_model_path", get_base_reward_model(),
            "--missing_eos_penalty", "5.0"
        ])
        if os.path.exists(os.path.join(MODEL_DIR, f"epoch_{epoch}")):
            return
        bsz /= 2
        if bsz < 1:
            print("Failed to train PPO model")
            return


# Run the main loop
if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(REWARD_DIR):
        os.makedirs(REWARD_DIR)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)  

    training(1)