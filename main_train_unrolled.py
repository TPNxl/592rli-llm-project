import asyncio
import os
import torch
from convo_dataset_batched import Convo_Dataset
from process_reward import *
from reward_processor import *
import subprocess

MODEL_DIR = "./model_weights"
REWARD_DIR = "./reward_models"
DATA_DIR = "./datasets"
NUM_ITEMS = 250
DEBATE_LEN = 6

def get_last_trained_model():
    max_n = 0
    for fn in os.listdir(MODEL_DIR):
        if fn.startswith("epoch_"):
            epoch = int(fn.split("_")[1].split(".")[0])
            max_n = max(max_n, epoch)
    return os.path.join(MODEL_DIR, f"epoch_{max_n}") if max_n >= 1 else get_base_model()

def get_prev_feedback_buf(epoch):
    p = os.path.join(DATA_DIR, f"convo_epoch_{epoch-1}/feedback.fbuf")
    return p if os.path.isfile(p) else None

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

# Dummy function to simulate cmd execution
# def dataset_generation(epoch):
def dataset_generation(epoch):
    fb_path = get_prev_feedback_buf(epoch)
    feedback_buf = [] if fb_path is None else torch.load(fb_path)
    
    print(f"Starting dataset generation, epoch {epoch}")
    model_path = get_last_trained_model()
    print(f"New model path: {model_path}")
    data_path = os.path.join(DATA_DIR, f"convo_epoch_{epoch}/")
    print(f"Data path: {data_path}")
    cd = Convo_Dataset(item_path=data_path, 
                       llama_new=model_path, 
                       llama_base=get_base_model(),
                       feedback_buf=feedback_buf,
                       generate_feedback=True,
                       generate_ranking=True,
                       device="cuda:0")
    cd.load()
    if not len(os.listdir(os.path.join(DATA_DIR, f"convo_epoch_{epoch}/"))) > NUM_ITEMS + 1:
        cd.generate(num_elems=NUM_ITEMS, debate_len=DEBATE_LEN)
    else:
        print("Skipping dataset generation")

    # if not os.path.exists(os.path.join(DATA_DIR, f"wsd_epoch_{epoch}.pt")):
    #     wsd = WinnerSeparatedDataset(cd)
    #     await asyncio.to_thread(wsd.process)
    #     wsd.save(os.path.join(DATA_DIR, f"wsd_epoch_{epoch}.pt"))
    rp = RewardProcessor(cd)
    out_path = os.path.join(DATA_DIR, f"dataset_epoch_{epoch}.csv")
    print(f"Saving CSV to: {out_path}")
    rp.process(out_path)

    del rp
    del cd

# def training(epoch):
def training(epoch):
    # Reward model training
    print(f"Starting training, epoch {epoch}")

    # If MODEL_DIR/epoch_{epoch} exists, skip reward model
    if not os.path.exists(os.path.join(REWARD_DIR, f"epoch_{epoch}")): 
        print("#################################### Training reward model")
        run([
            "python", "reward_modeling.py",
            "--model_name_or_path", get_base_reward_model(),
            "--dataset_name", os.path.join(DATA_DIR, f"dataset_epoch_{epoch}.csv"),
            "--output_dir", REWARD_DIR,
            "--per_device_train_batch_size", "8",
            "--num_train_epochs", "1",
            "--gradient_checkpointing", "True",
            "--learning_rate", "1.0e-4",
            "--logging_steps", "25",
            "--max_length", "2048",
            "--use_peft",
            "--lora_r", "32",
            "--lora_alpha", "16"
        ])
    else:
        print("Skipping reward model training")

    time.sleep(5)

    if not os.path.exists(os.path.join(MODEL_DIR, f"epoch_{epoch}")):
        print("#################################### Training PPO model")
        run([
            "python", "ppo.py",
            "--dataset_name", os.path.join(DATA_DIR, f"dataset_epoch_{epoch}.csv"),
            "--learning_rate", "1e-6",
            "--output_dir", MODEL_DIR,
            "--per_device_train_batch_size", "4",
            "--gradient_accumulation_steps", "1",
            "--total_episodes", "1200",
            "--model_name_or_path", get_base_model(),
            "--sft_model_path", get_base_model(),
            "--reward_model_path", get_base_reward_model(),
            "--missing_eos_penalty", "1.0"
        ])
    else:
        print("Skipping PPO model training")


# Run the main loop
if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)  

    i = 1
    try:
        while True:
            print(f"######################################################################################\nStarting epoch {i}")
            dataset_generation(i)
            training(i)
            i += 1
    except KeyboardInterrupt:
        print("Interrupted")
        pass