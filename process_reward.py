import torch
import torch.utils.data.dataset
import torch.utils.data.dataset
import llm_blender
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from llm_blender.pair_ranker.pairrm import DebertaV2PairRM
from transformers import AutoTokenizer
from typing import List

from convo_dataset_batched import *

class RewardProcessedDataset(torch.utils.data.Dataset):
    def __init__(self, convo_dataset: Convo_Dataset):
        self.convo_dataset = convo_dataset
        self.datapoints = []

    def process(self):
        blender = llm_blender.Blender()
        blender.loadranker("llm-blender/PairRM") # load ranker checkpoint
        pairrm = DebertaV2PairRM.from_pretrained("llm-blender/PairRM-hf", device_map="cuda:0").eval()
        tokenizer = AutoTokenizer.from_pretrained('llm-blender/PairRM-hf')

        source_prefix = "<|source|>"  # Should be our topic
        cand1_prefix = "<|candidate1|>" # LLM1 response (on training)
        cand2_prefix = "<|candidate2|>" # LLM2 response (frozen)

        datapoints = []

        def tokenize_pair(sources:List[str], candidate1s:List[str], candidate2s:List[str], source_max_length=1224, candidate_max_length=412):
            ids = []
            assert len(sources) == len(candidate1s) == len(candidate2s)
            max_length = source_max_length + 2 * candidate_max_length
            for i in range(len(sources)):
                source_ids = tokenizer.encode(source_prefix + sources[i], max_length=source_max_length, truncation=True)
                candidate_max_length = (max_length - len(source_ids)) // 2
                candidate1_ids = tokenizer.encode(cand1_prefix + candidate1s[i], max_length=candidate_max_length, truncation=True)
                candidate2_ids = tokenizer.encode(cand2_prefix + candidate2s[i], max_length=candidate_max_length, truncation=True)
                ids.append(source_ids + candidate1_ids + candidate2_ids)
            encodings = tokenizer.pad({"input_ids": ids}, return_tensors="pt", padding="max_length", max_length=max_length)
            return encodings

        i = 0
        for convo in self.convo_dataset.convos:
            if i % 100 == 0:
                print(f"Processing conversation {i}")
            agent_A = convo.agents[0]
            agent_B = convo.agents[1]
            candidates_A = [item[1] for item in convo.history if item[0] == agent_A]
            candidates_B = [item[1] for item in convo.history if item[0] == agent_B]
            L = min(len(candidates_A), len(candidates_B))
            inputs = [convo.topic] * L

            encodings = tokenize_pair(inputs, candidates_A, candidates_B)
            encodings = {k:v.to(pairrm.device) for k,v in encodings.items()}
            outputs = pairrm(**encodings)
            values = torch.cumsum(outputs.logits, dim=1).tolist()
            
            for i in range(L):
                datapoints.append((inputs[i], candidates_A[i], candidates_B[i], agent_A, agent_B) 
                                  if values[i][0] > values[i][1] 
                                  else (inputs[i], candidates_B[i], candidates_A[i], agent_B, agent_A))

            i += 1

        self.datapoints = datapoints

    def __len__(self):
        return len(self.datapoints)
    
    def __getitem__(self, idx):
        return self.datapoints[idx]

class WinnerSeparatedDataset(torch.utils.data.Dataset):
    def __init__(self, convo_dataset: Convo_Dataset):
        self.convo_dataset = convo_dataset
        self.datapoints = []

    def save(self, path):
        with open(path, "wb") as f:
            torch.save(self.datapoints, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.datapoints = torch.load(f)

    def process(self):
        datapoints = []
        count = 0
        for convo in self.convo_dataset.convos:
            if count % 100 == 0:
                print(f"Processing conversation {count}")

            if convo.winner == "" or convo.loser == "":
                continue
            agent_A = convo.winner
            agent_B = convo.loser
            candidates_A = [item[1]['content'] for item in convo.history if item[0] == agent_A]
            candidates_B = [item[1]['content'] for item in convo.history if item[0] == agent_B]
            L = min(len(candidates_A), len(candidates_B))
            inputs = [convo.topic] * L
            for i in range(L):
                self.datapoints.append((inputs[i], candidates_A[i], candidates_B[i], agent_A, agent_B))

            if count % 100 == 0:
                print(self.datapoints[-1])
            
            count += 1
            
        self.datapoints = datapoints
            
    def __len__(self):
        return len(self.datapoints)
    
    def __getitem__(self, idx):
        return self.datapoints[idx]


if __name__ == "__main__":
    convo_dataset = Convo_Dataset("./datasets/convo_test/")
    convo_dataset.load()
    # reward_dataset = RewardProcessedDataset(convo_dataset)
    # reward_dataset.process()
    # reward_dataset.save("reward_dataset.pkl")
    winner_dataset = WinnerSeparatedDataset(convo_dataset)
    winner_dataset.process()
    if os.path.exists("./datasets/wsd_test.pt"):
        os.remove("./datasets/wsd_test.pt")
    winner_dataset.save("./datasets/wsd_test.pt")