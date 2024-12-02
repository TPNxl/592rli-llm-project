import torch
from convo_dataset import Convo_Dataset
from process_reward import RewardProcessedDataset
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSequenceClassification
from openai import OpenAI

"""
CHECK TRAIN_SEMA.PY INSTEAD

So basic pipeline:

train_loop() - synchronous version
- Load base model
- Load curr trained model = base model to start, then load from disk
- If training reward model, load reward model
- Load value model...? Question: What is it for?
- For each epoch:
    - Generate dataset of conversation
    - If training reward model, train reward model using dataset separated into chosen/rejected by LLaMA
    - Else, use PairRM to process rewards and separate conversations 
    - Train curr model using PPO
    - Save each model to disk for easy resume

train_loop() - asynchronous version
- Load base model
- Load curr trained model = base model to start, then load from disk
- If training reward model, load reward model
- Load value model...? Question: What is it for?
Thread 1:
- For each epoch:
    - Grab curr trained model
    - Generate dataset of conversation using curr trained model
    - If training reward model, train reward model using dataset separated into chosen/rejected by LLaMA
    - Save each model to disk for easy resume
Thread 2:
- For each epoch:
    - Grab previous dataset from thread 1
    - PPO train curr trained model
    - Save each model to disk for easy resume
"""

with open("./tokens/hf_token.txt", 'r') as f:
    hf_token = f.read().strip()
with open("./tokens/openai_token.txt", 'r') as f:
    openai_token = f.read().strip()
client = OpenAI(api_key=openai_token)

def gpt_generate(messages: dict):
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
    )
    out = chat_completion.choices[0].message.content
    out = out.split(": ")[-1]
    return out

def train_loop():

    num_epochs = 5

    training_run = int(np.random.random() * 1000000)

    base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

    curr_trained_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    feedback_buf = []

    curr_reward_model = AutoModelForSequenceClassification.from_pretrained(
        "Qwen/Qwen2-0.5B-Instruct", num_labels=1,
    )

    value_model = AutoModelForSequenceClassification.from_pretrained(
        "Qwen/Qwen2-0.5B-Instruct", num_labels=1,
    )

    model_peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=modules,
    )

    for epoch in range(num_epochs):
        # Training code here
        print(f"Epoch {epoch+1}")

        curr_pipe = pipeline(
            "text-generation",
            model=curr_trained_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        base_pipe = pipeline(
            "text-generation",
            model=base_model,
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
        
        # Generate dataset on cuda:0 and cuda:1
        cd = Convo_Dataset(f"./run{training_run}/epoch{epoch}/", curr_pipe_generate, base_pipe_generate, gpt_generate, feedback_buf)
        cd.generate()

        # Process rewards
        rpd = RewardProcessedDataset(cd, curr_trained_model)
        rpd.process()

        # Use asyncio for these tasks!

        # If needed, train reward model
        # <some code calling reward_modeling.py>
        # reward_model = <new model>
        reward_trainer = RewardTrainer(
            model=curr_reward_model,
            processing_class=tokenizer,
            args=reward_training_args,
            train_dataset=cd,
            eval_dataset=None,
            peft_config=get_peft_config(model_config),
        )
        reward_trainer.train()

        # Train curr model
        # <some code calling ppo>
        # curr_trained_model = <new model>
        model_trainer = PPOTrainer(
            args=model_training_args,
            processing_class=tokenizer,
            model=curr_trained_model,
            ref_model=base_model,
            reward_model=curr_reward_model,
            value_model=value_model,
            train_dataset=cd,
            eval_dataset=None,
            peft_config=model_peft_config,
        )
        model_trainer.train()

        # Save and push to hub
        model_trainer.save_model(training_args.output_dir)


        # Save checkpoints