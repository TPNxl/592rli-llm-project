import pandas as pd
from convo_dataset_batched import Convo_Dataset
import os
from reward_processor import winEval

MODEL_DIR = "./model_weights"
REWARD_DIR = "./reward_models"
DATA_DIR = "./win_eval_datasets"
NUM_ITEMS = 30
DEBATE_LEN = 6
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

def test(model_path, data_path):
    print(f"Evaluating with model path: {model_path}, data path: {data_path}")
    cd = Convo_Dataset(item_path=data_path, 
                    llama_new=model_path, 
                    llama_base="meta-llama/Llama-3.2-1B-Instruct",
                    feedback_buf=None,
                    generate_feedback=False,
                    generate_ranking=True,
                    device="cuda:0")
    cd.load()
    if not len(os.listdir(data_path)) >= NUM_ITEMS:
        cd.generate(num_elems=NUM_ITEMS, debate_len=DEBATE_LEN)

    we = winEval(cd)
    we.process()

if __name__ == "__main__":
    test(None, os.path.join(DATA_DIR, f"convo_epoch_0/"))
    for fn in os.listdir(MODEL_DIR):
        if fn.startswith("epoch_"):
            epoch = int(fn.split("_")[1].split(".")[0])
            test(os.path.join(MODEL_DIR, f"epoch_{epoch}", "policy"), os.path.join(DATA_DIR, f"convo_epoch_{epoch}/"))

