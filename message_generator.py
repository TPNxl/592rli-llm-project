import pickle

class MessageGenerator():
    def __init__(self):
        self.conversation_name = ""
        self.agent_views = {}
        self.history = []
        self.quality = ""
        self.evaluation = ""
        self.winner = ""
        self.feedback = ""
    
    def append(self, agent_name, val):
        self.history.append([agent_name, val])

    def generate_debate_history(self, system_prompt, agent_name, opp_name):
        messages = []
        system_prompt = system_prompt.replace("<agent_name>", agent_name).replace("<opp_name>", opp_name)
        messages.append({"role": "system", "content": system_prompt})
        for h in self.history:
            if h[0] == agent_name:
                messages.append({"role": "assistant", "content": h[1]})
            elif h[0] == opp_name:
                messages.append({"role": "user", "content": h[1]})
            else:
                messages.append({"role": "user", "content": f"Third party named {h[0]} said {h[1]}"})
        return messages
    
    def generate_continue_debate_prompt(self, continue_prompt, agent_name, opp_name):
        messages = []
        continue_prompt = continue_prompt.replace("<agent_name>", agent_name).replace("<opp_name>", opp_name)
        messages.append({"role": "system", "content": continue_prompt})
        for h in self.history:
            if h[0] == agent_name:
                messages.append({"role": "assistant", "content": h[1]})
            elif h[0] == opp_name:
                messages.append({"role": "user", "content": h[1]})
            else:
                messages.append({"role": "user", "content": f"Third party named {h[0]} said {h[1]}"})
        messages.append({"role": "system", "content": f"Please continue the debate - remember, you are {agent_name}."})
        return messages
    
    def generate_ranking_prompt(self, ranking_prompt, quality, evaluation):
        messages = []
        messages.append({"role": "system", "content": ranking_prompt.replace("<quality>", quality).replace("<evaluation>", evaluation)})
        for h in self.history:
            messages.append({"role": "user", "content": f"{h[0]}: {h[1]}"})
        messages.append({"role": "system", "content": "Your analysis starts now. Please end your analysis with the name of the winner on a line by itself."})
        return messages
    
    def generate_feedback_prompt(self, feedback_prompt, quality, evaluation):
        messages = []
        messages.append({"role": "system", "content": feedback_prompt.replace("<quality>", quality).replace("<evaluation>", evaluation)})
        for h in self.history:
            messages.append({"role": "user", "content": f"{h[0]}: {h[1]}"})
        messages.append(f"The agent that was {evaluation} was declared to be {self.winner}.")
        messages.append({"role": "system", "content": "Please provide feedback on the debate."})
        return messages
    
    def get(self, name):
        for h in self.history:
            if h[0] == name:
                return h[1]
        return None

    def save(self, filename):
        with open(filename, 'wb') as f:
            x = (self.history, self.prompt, self.names, self.views, self.q, self.e, self.w, self.feedback)
            pickle.dump(x, f)
    
    def load(self, filename):
        with open(filename, 'rb') as f:
            self.history, self.prompt, self.names, self.views, self.q, self.e, self.w, self.feedback = pickle.load(f)
            self.history_index = len(self.history) - 1