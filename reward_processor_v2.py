import pandas as pd
from convo_dataset_batched import Convo_Dataset
from message_generator import MessageGenerator

with open("./prompts/continue_prompt.txt", 'r') as f:
    CONTINUE_PROMPT = f.read().replace("\n", " ")
NUM_AGENTS = 2

class RewardProcessor:
    def __init__(self, cd: Convo_Dataset):
        self.cd = cd

    def process(self, output_path: str):
        processed_convos = []
        count = 0
        for convo in self.cd.convos:
            if convo.winner == "" or convo.loser == "":
                continue
            prompt = convo.overall_view()

            g = MessageGenerator()
            g.copy(convo)

            for n in range(len(convo.history) - 4):
                g.history = convo.history[:n+2]
                agent_name = g.agents[n%NUM_AGENTS]
                opp_name = g.agents[(n+1)%NUM_AGENTS]
                prompt = g.generate_debate_prompt(CONTINUE_PROMPT, agent_name, opp_name, s_prompt_2=None)

                if convo.winner == agent_name:
                    chosen = convo.history[n+2]
                    rejected = convo.history[n+3]
                else:
                    chosen = convo.history[n+3]
                    rejected = convo.history[n+2]
                chosen = chosen[1]
                rejected = rejected[1]
                chosen['content'] = chosen['content'].replace("\n", " ").replace("\"", "")
                rejected['content'] = rejected['content'].replace("\n", " ").replace("\"", "")

                prompt_str = "".join(msg['content'].replace("\n", " ").replace("\"", "") + "\n" for msg in prompt)

                processed_convos.append({
                    'prompt': prompt_str,
                    'chosen': chosen,
                    'rejected': rejected,
                })
                count += 1

            if count > 1000:
                break

        # Create a DataFrame
        df = pd.DataFrame(processed_convos)

        # Save to CSV
        df.to_csv(output_path, index=False)