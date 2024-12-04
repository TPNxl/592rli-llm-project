import asyncio
import os
import torch
from convo_dataset_batched import Convo_Dataset
from process_reward import *
from reward_processor_v2 import *
import subprocess

MODEL_DIR = "./model_weights"
DATA_DIR = "./datasets"

def get_last_trained_model():
    max_n = -1
    for fn in os.listdir(MODEL_DIR):
        if fn.endswith(".pt") and fn.startswith("llama_trained_epoch_"):
            epoch = int(fn.split("_")[3].split(".")[0])
            max_n = max(max_n, epoch)
    return os.path.join(MODEL_DIR, f"llama_trained_epoch_{max_n}.pt") if max_n != -1 else get_base_model()

def get_last_reward_model():
    max_n = -1
    for fn in os.listdir(MODEL_DIR):
        if fn.endswith(".pt") and fn.startswith("reward_model_epoch_"):
            epoch = int(fn.split("_")[3].split(".")[0])
            max_n = max(max_n, epoch)
    return os.path.join(MODEL_DIR, f"reward_model_epoch_{max_n}.pt") if max_n != -1 else get_base_reward_model()

def get_last_value_model():
    max_n = -1
    for fn in os.listdir(MODEL_DIR):
        if fn.endswith(".pt") and fn.startswith("value_model_epoch_"):
            epoch = int(fn.split("_")[3].split(".")[0])
            max_n = max(max_n, epoch)
    return os.path.join(MODEL_DIR, f"value_model_epoch_{max_n}.pt") if max_n != -1 else get_base_reward_model()

def get_prev_feedback_buf(epoch):
    p = os.path.join(DATA_DIR, f"convo_epoch{epoch-1}/feedback.fbuf")
    return p if os.path.isfile(p) else None

# def get_last_wsd():
#     max_n = -1
#     for fn in os.listdir(DATA_DIR):
#         if fn.endswith(".pt") and fn.startswith("wsd_epoch_"):
#             epoch = int(fn.split("_")[2].split(".")[0])
#             max_n = max(max_n, epoch)
#     return os.path.join(DATA_DIR, f"wsd_epoch_{max_n}.pt") if max_n != -1 else None

def get_base_model():
    return "meta-llama/Llama-3.2-3B-Instruct"

def get_base_reward_model():
    return "Qwen/Qwen2-0.5B-Instruct"

# Dummy function to simulate cmd execution
# def dataset_generation(epoch):
async def dataset_generation(epoch):
    fb_path = get_prev_feedback_buf(epoch)
    feedback_buf = [] if fb_path is None else torch.load(fb_path)

    print(f"Starting dataset generation, epoch {epoch}")
    cd = Convo_Dataset(item_path=os.path.join(DATA_DIR, f"convo_epoch{epoch}/"), 
                       llama_new=get_last_trained_model(), 
                       llama_base=get_base_model(),
                       feedback_buf=feedback_buf,
                       generate_feedback=True,
                       generate_ranking=True,
                       device="cuda:0")
    cd.load()
    if not len(os.listdir(os.path.join(DATA_DIR, f"convo_epoch{epoch}/"))) > 301:
        await asyncio.to_thread(cd.generate, num_elems=300, debate_len=8)

    # if not os.path.exists(os.path.join(DATA_DIR, f"wsd_epoch_{epoch}.pt")):
    #     wsd = WinnerSeparatedDataset(cd)
    #     await asyncio.to_thread(wsd.process)
    #     wsd.save(os.path.join(DATA_DIR, f"wsd_epoch_{epoch}.pt"))
    rp = RewardProcessor(cd)
    rp.process(os.path.join(DATA_DIR, f"dataset_epoch_{epoch}.csv"))

# def training(epoch):
async def training(epoch):
    # Reward model training
    print(f"Starting reward training, epoch {epoch}")
    await asyncio.wait(5)
    # process_reward = await asyncio.create_subprocess_exec(
    #     "python", "reward_modeling.py",
    #     "--model_name_or_path", get_last_reward_model(),
    #     "--dataset_name", os.path.join(DATA_DIR, f"dataset_epoch_{epoch}"),
    #     "--output_dir", os.path.join(MODEL_DIR, f"reward_model_epoch_{epoch}"),
    #     "--per_device_train_batch_size", "8",
    #     "--num_train_epochs", "1",
    #     "--gradient_checkpointing", "True",
    #     "--learning_rate", "1.0e-5",
    #     "--logging_steps", "25",
    #     "--eval_strategy", "steps",
    #     "--eval_steps", "50",
    #     "--max_length", "2048"
    # )
    # print(f"Starting PPO training, epoch {epoch}")
    # process_ppo = await asyncio.create_subprocess_exec(
    #     "python", "-i", "ppo.py",
    #     "--dataset_name", os.path.join(DATA_DIR, f"dataset_epoch_{epoch}",
    #     "--learning_rate", "3e-6",
    #     "--output_dir", os.path.join(MODEL_DIR, f"model_{epoch}"),
    #     "--per_device_train_batch_size", "64",
    #     "--gradient_accumulation_steps", "1",
    #     "--total_episodes", "10000",
    #     "--model_name_or_path", "EleutherAI/pythia-1b-deduped",
    #     "--missing_eos_penalty", "1.0"
    # )
    print("#################################### Training reward model")
    process_reward = await asyncio.create_subprocess_exec(
        "python", "reward_modeling.py",
        "--model_name_or_path", "Qwen/Qwen2-0.5B-Instruct",
        "--dataset_name", "convos2.csv",
        "--output_dir", "./reward_models",
        "--per_device_train_batch_size", "8",
        "--num_train_epochs", "1",
        "--gradient_checkpointing", "True",
        "--learning_rate", "1.0e-4",
        "--logging_steps", "25",
        "--max_length", "2048",
        "--use_peft",
        "--lora_r", "32",
        "--lora_alpha", "16"
    )

    print("#################################### Training PPO model")

    process_ppo = await asyncio.create_subprocess_exec(
        "python", "ppo.py",
        "--dataset_name", "convos2.csv",
        # "--dataset_name", "trl-internal-testing/descriptiveness-sentiment-trl-style",
        # "--dataset_train_split", "descriptiveness",
        "--learning_rate", "3e-6",
        "--output_dir", "./model_weights",
        "--per_device_train_batch_size", "64",
        "--gradient_accumulation_steps", "1",
        "--total_episodes", "10000",
        "--model_name_or_path", "EleutherAI/pythia-1b-deduped",
        "--sft_model_path", "EleutherAI/pythia-1b-deduped",
        "--reward_model_path", "Qwen/Qwen2-0.5B-Instruct",
        "--missing_eos_penalty", "1.0"
    )


# Function to manage sequential execution for a single input
async def process_input(epoch, cmd1_semaphore, cmd2_semaphore):
    async with cmd1_semaphore:
        await dataset_generation(epoch)
    async with cmd2_semaphore:
        await training(epoch)

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

    asyncio.run(main(range(1,6)))