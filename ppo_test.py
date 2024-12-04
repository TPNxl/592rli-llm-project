import subprocess
import os
from convo_dataset_batched import Convo_Dataset
from reward_processor_v2 import *
import time

MODEL_DIR = "./model_weights"
DATA_DIR = "./datasets"
cd = Convo_Dataset(item_path="/scratch/gilbreth/tnadolsk/592rli-llm-project/datasets/convo_epoch_2")
cd.load()
rp = RewardProcessor(cd)

rp.process("/scratch/gilbreth/tnadolsk/592rli-llm-project/datasets/dataset_epoch_2.csv")

# def run(command):
#     # Run the command
#     result = subprocess.run(command)

#     # Print the output and errors
#     print("Output:")
#     print(result.stdout)
#     print("Errors:")
#     print(result.stderr)

# for i in range(1, 6):
#     # print("#################################### Training reward model")
#     # run([
#     #     "python", "reward_modeling.py",
#     #     "--model_name_or_path", "Qwen/Qwen2-0.5B-Instruct",
#     #     "--dataset_name", "convos2.csv",
#     #     "--output_dir", "./reward_models",
#     #     "--per_device_train_batch_size", "8",
#     #     "--num_train_epochs", "1",
#     #     "--gradient_checkpointing", "True",
#     #     "--learning_rate", "1.0e-4",
#     #     "--logging_steps", "25",
#     #     "--max_length", "2048",
#     #     "--use_peft",
#     #     "--lora_r", "32",
#     #     "--lora_alpha", "16"
#     # ])

#     print("#################################### Training PPO model")

#     run([
#         "python", "ppo.py",
#         "--dataset_name", "convos2.csv",
#         # "--dataset_name", "trl-internal-testing/descriptiveness-sentiment-trl-style",
#         # "--dataset_train_split", "descriptiveness",
#         "--learning_rate", "3e-6",
#         "--output_dir", "./model_weights",
#         "--per_device_train_batch_size", "8",
#         "--gradient_accumulation_steps", "1",
#         "--total_episodes", "1000",
#         "--model_name_or_path", "meta-llama/Llama-3.2-1B-Instruct",
#         "--sft_model_path", "meta-llama/Llama-3.2-1B-Instruct",
#         "--reward_model_path", "Qwen/Qwen2-0.5B-Instruct",
#         "--missing_eos_penalty", "1.0"
#     ])

#     # run([
#     #     "accelerate", "launch",
#     #     "--config_file", "deepspeed_zero3.yaml",
#     #     "ppo.py",
#     #     "--dataset_name", "convos2.csv",
#     #     # "--dataset_name", "trl-internal-testing/descriptiveness-sentiment-trl-style",
#     #     # "--dataset_train_split", "descriptiveness",
#     #     "--output_dir", "./model_weights",
#     #     "--learning_rate", "3e-6",
#     #     "--per_device_train_batch_size", "16",
#     #     "--gradient_accumulation_steps", "1",
#     #     "--total_episodes", "5000",
#     #     "--model_name_or_path", "meta-llama/Llama-3.2-1B-Instruct",
#     #     "--sft_model_path", "meta-llama/Llama-3.2-1B-Instruct",
#     #     "--reward_model_path", "Qwen/Qwen2-0.5B-Instruct",
#     #     "--missing_eos_penalty", "1.0"
#     # ])
