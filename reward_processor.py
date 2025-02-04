import pandas as pd
from convo_dataset_batched_old import Convo_Dataset
from message_generator_old import MessageGenerator

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
                elif convo.winner == opp_name:
                    chosen = convo.history[n+3]
                    rejected = convo.history[n+2]
                else:
                    continue
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

class winEval:
    def __init__(self, cd: Convo_Dataset):
        self.cd = cd

    def process(self):
        wins = 0
        bad_evals = 0
        for convo in self.cd.convos:
            if convo.winner == "" or convo.loser == "":
                bad_evals += 1
                continue
            
            wins += 1 if convo.winner == convo.agents[0] else 0
        
        count = len(self.cd.convos)
        print(f"Total convos: {count}")
        print(f"Wins: {wins} = {wins/count*100:.2f}%")
        print(f"Losses: {count-wins-bad_evals} = {(count-wins-bad_evals)/count*100:.2f}%")
        print(f"Bad evaluations: {bad_evals} = {bad_evals/count*100:.2f}%")
        print(f"Win rate across good evals: {wins/(count-bad_evals)*100:.2f}%")
        return count, wins, count-wins-bad_evals, bad_evals
    
class lenEval:
    def __init__(self, cd: Convo_Dataset):
        self.cd = cd

    def process(self):
        chars = 0
        for convo in self.cd.convos:
            for msg in convo.history:
                chars += len(msg[1]['content'])
        print(f"Total characters: {chars}")
        print(f"Average characters per message: {chars/len(self.cd.convos)/len(self.cd.convos[0].history)}")


if __name__ == "__main__":
    while True:
        print("Welcome to the WinEval client!")
        fn = input("Please enter the directory of the ConvoDataset you would like to load: ")
        cd = Convo_Dataset(item_path=fn)
        cd.load()
        we = lenEval(cd)
        we.process()