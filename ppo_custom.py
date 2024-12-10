# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import shutil

from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import ScriptArguments, ModelConfig, PPOConfig
from ppo_trainer_custom import CustomPPOTrainer
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

import os


"""
python ppo.py \
    --dataset_name convos.csv \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --output_dir model_weights \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path EleutherAI/pythia-1b-deduped \
    --reward_model_path EleutherAI/pythia-1b-deduped \
    --missing_eos_penalty 1.0

accelerate launch --config_file deepspeed_zero3.yaml \
    ppo.py \
    --dataset_name convos.csv \
    --dataset_train_split descriptiveness \
    --output_dir models/minimal/ppo \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path EleutherAI/pythia-1b-deduped \
    --reward_model_path EleutherAI/pythia-1b-deduped \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0
"""


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    # shutil.rmtree(training_args.output_dir, ignore_errors=True)

    # Get last epoch
    max_n = 0
    for fn in os.listdir(training_args.output_dir):
        if fn.startswith("epoch_") and fn.split("_")[-1].isdigit():  # Ensure valid format
            n = int(fn.split("_")[-1])
            max_n = max(max_n, n)
    print(f"Last value epoch: {max_n}")

    # Get last model paths
    vm_path = os.path.join(training_args.output_dir, f"epoch_{max_n}/value_model") if max_n > 0 else training_args.reward_model_path
    policy_path = os.path.join(training_args.output_dir, f"epoch_{max_n}/policy") if max_n > 0 else training_args.sft_model_path

    max_r = 0
    for fn in os.listdir("./reward_models/"):
        if fn.startswith("epoch_") and fn.split("_")[-1].isdigit():  # Ensure valid format
            n = int(fn.split("_")[-1])
            max_r = max(max_r, n)
    print(f"Last reward epoch: {max_r}")

    rm_path = os.path.join("./reward_models/" f"epoch_{max_r}/") if max_r > 0 else training_args.reward_model_path

    # Debugging print statements (optional)
    print(f"Reward Model Path: {rm_path}")
    print(f"Value Model Path: {vm_path}")
    print(f"Policy Path: {policy_path}")

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_config.trust_remote_code,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    value_model = AutoModelForSequenceClassification.from_pretrained(
        rm_path, trust_remote_code=model_config.trust_remote_code, num_labels=1
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        vm_path, trust_remote_code=model_config.trust_remote_code, num_labels=1
    )
    ref_policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_config.trust_remote_code
    )
    policy = AutoModelForCausalLM.from_pretrained(
        policy_path, trust_remote_code=model_config.trust_remote_code
    )
    ################
    # Dataset
    ################
    dataset = load_dataset('csv', data_files=script_args.dataset_name)['train']
    eval_samples = 100
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    dataset_text_field = "prompt"

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    ################
    # Training
    ################

    trainer = CustomPPOTrainer(
        config=training_args,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train(length_penalty=0.01, bad_token_ids=[25, 101060, 487], bad_token_penalty=1)

    # Save and push to hub
    print("Saving model")
    # Save each model to the output dir/epoch_{max_n+1}/the respective path
    epoch_dir = os.path.join(training_args.output_dir, f"epoch_{max_n+1}")
    value_model.save_pretrained(os.path.join(epoch_dir, "value_model"))
    policy.save_pretrained(os.path.join(epoch_dir, "policy"))
    tokenizer.save_pretrained(os.path.join(epoch_dir, "policy"))
    print("Models saved")
    