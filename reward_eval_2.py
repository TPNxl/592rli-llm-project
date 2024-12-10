# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Full training:
python examples/scripts/reward_modeling.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --output_dir Qwen2-0.5B-Reward \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-5 \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 2048

LoRA:
python examples/scripts/reward_modeling.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --output_dir Qwen2-0.5B-Reward-LoRA \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-4 \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 2048 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
"""

import warnings

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)

import os

MODEL_DIR = "./win_eval_datasets"
REWARD_DIR = "./reward_models"
DATA_DIR = "./datasets"
NUM_ITEMS = 30
DEBATE_LEN = 6
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, RewardConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    
    for fn in os.listdir(REWARD_DIR):
        if fn.startswith("epoch_"):
            epoch = int(fn.split("_")[1].split(".")[0])
            rm_path = os.path.join(MODEL_DIR, f"epoch_{epoch}")
            data_path = os.path.join(DATA_DIR, f"convo_epoch_{epoch}/")
            print(f"Reward Model Path: {rm_path}")
            model = AutoModelForSequenceClassification.from_pretrained(
                rm_path, num_labels=1, trust_remote_code=model_config.trust_remote_code, **model_kwargs
            )
            # Align padding tokens between tokenizer and model
            model.config.pad_token_id = tokenizer.pad_token_id

            # If post-training a base model, use ChatML as the default template
            if tokenizer.chat_template is None:
                model, tokenizer = setup_chat_format(model, tokenizer)

            if model_config.use_peft and model_config.lora_task_type != "SEQ_CLS":
                warnings.warn(
                    "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
                    " Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT."
                )

    
