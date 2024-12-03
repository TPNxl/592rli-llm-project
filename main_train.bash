#!/bin/bash

ppo="python ppo.py --dataset_name convos2.csv --learning_rate 3e-6 --output_dir ./model_weights --per_device_train_batch_size 64 --gradient_accumulation_steps 1 --total_episodes 10000 --model_name_or_path EleutherAI/pythia-1b-deduped --sft_model_path EleutherAI/pythia-1b-deduped --reward_model_path Qwen/Qwen2-0.5B-Instruct --missing_eos_penalty 1.0"

for i in {1..2}; do
    doub=$((i * 2))
    python3 main_train.py "$doub"
    $ppo
done
