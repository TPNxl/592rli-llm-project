import pandas as pd
from convo_dataset_batched import Convo_Dataset

class RewardProcessor:
    def __init__(self, cd: Convo_Dataset):
        self.cd = cd

    def process(self, output_path: str):
        processed_convos = []
        count = 0
        for convo in self.cd.convos:
            if convo.winner == "" or convo.loser == "":
                continue
            agent_A = convo.winner
            agent_B = convo.loser
            candidates_A = [item[1] for item in convo.history if item[0] == agent_A]
            candidates_B = [item[1] for item in convo.history if item[0] == agent_B]
            L = min(len(candidates_A), len(candidates_B))
            inputs = [convo.overall_view()] * L
            for i in range(L):
                processed_convos.append({
                    'prompt': inputs[i],
                    'chosen': candidates_A[i],
                    'rejected': candidates_B[i],
                })
                count += 1
            if count > 500:
                break

        # Create a DataFrame
        df = pd.DataFrame(processed_convos)

        # Save to CSV
        df.to_csv(output_path, index=False)