import asyncio
import os
import torch
from convo_dataset_batched_new import Convo_Dataset
from process_reward import *
from reward_processor_2 import *
import subprocess
from aggregate import *

MODEL_DIR = "./model_weights_v2"
REWARD_DIR = "./reward_models_v2"
DATA_DIR = "./datasets_v2"
COMPLETE_DATASET_DIR = "./complete_dataset"
NUM_ITEMS = 250
DEBATE_LEN = 6

def get_last_trained_model(dir = MODEL_DIR):
    max_n = 0
    for fn in os.listdir(dir):
        if fn.startswith("epoch_"):
            epoch = int(fn.split("_")[1].split(".")[0])
            max_n = max(max_n, epoch)
    return os.path.join(dir, f"epoch_{max_n}") if max_n >= 1 else get_base_model()

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

    return result.stderr

# Dummy function to simulate cmd execution
# def dataset_generation(epoch):
def generation(epoch, path=None, items=NUM_ITEMS):
    fb_path = get_prev_feedback_buf(epoch)
    feedback_buf = [] if fb_path is None else torch.load(fb_path, weights_only=False)
    
    print(f"Starting dataset generation, epoch {epoch}")
    model_path = os.path.join(get_last_trained_model(), "policy")
    print(f"New model path: {model_path}")
    data_path = os.path.join(DATA_DIR, f"convo_epoch_{epoch}/") if path is None else path
    print(f"Data path: {data_path}")
    cd = Convo_Dataset(item_path=data_path, 
                       llama_new=model_path, 
                       llama_base=get_base_model(),
                       feedback_buf=feedback_buf,
                       generate_feedback=True,
                       generate_ranking=True,
                       device="cuda:0")
    cd.load()
    if not len(os.listdir(os.path.join(DATA_DIR, f"convo_epoch_{epoch}/"))) >= items:
        cd.generate(num_elems=items, debate_len=DEBATE_LEN)
    else:
        print("Skipping dataset generation")

    del cd

def aggregate_dataset():
    merge_datasets_folder(DATA_DIR, COMPLETE_DATASET_DIR)
    cd = Convo_Dataset(item_path=COMPLETE_DATASET_DIR, 
                       feedback_buf=None,
                       generate_feedback=True,
                       generate_ranking=True,
                       device="cuda:0")
    cd.load()

    rp = RewardProcessor(cd)
    out_path = os.path.join(COMPLETE_DATASET_DIR, f"dataset.csv")
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
            "--model_name_or_path", get_last_trained_model(REWARD_DIR),
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
        bsz = 32
        while bsz >= 1:
            run([
                "python", "ppo.py",
                "--dataset_name", os.path.join(DATA_DIR, f"dataset_epoch_{epoch}.csv"),
                "--learning_rate", "1e-6",
                "--output_dir", MODEL_DIR,
                "--per_device_train_batch_size", str(bsz),
                "--gradient_accumulation_steps", "1",
                "--total_episodes", "1200",
                "--model_name_or_path", get_base_model(),
                "--sft_model_path", get_base_model(),
                "--reward_model_path", get_base_reward_model(),
                "--missing_eos_penalty", "5.0"
            ])
            if os.path.exists(os.path.join(MODEL_DIR, f"epoch_{epoch}")):
                break
            bsz /= 2
            if bsz < 1:
                print("Failed to train PPO model")
                return
    else:
        print("Skipping PPO model training")


# Run the main loop
if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(REWARD_DIR):
        os.makedirs(REWARD_DIR)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)  
    if not os.path.exists(COMPLETE_DATASET_DIR):
        os.makedirs(COMPLETE_DATASET_DIR)

    i = 1
    # generation(epoch=0, items=1000)
    try:
        while True:
            print(f"######################################################################################\nStarting epoch {i}")
            # generation(i)
            aggregate_dataset()
            training(i)
            i += 1
    except KeyboardInterrupt:
        print("Interrupted")
        pass