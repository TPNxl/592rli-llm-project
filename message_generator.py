import pickle
import numpy as np

r = np.random.RandomState(123)

class MessageGenerator():
    def __init__(self, 
                 filename = None,
                 topic = "", 
                 agents = [],
                 agent_views = {}, 
                 agent_funcs = {},
                 quality = "convince the other of their perspective", 
                 evaluation = "Who convinced the other of their perspective?"):
        if filename is not None:
            self.load(filename)
        else:
            self.topic = topic
            self.agents = agents
            self.agent_views = agent_views
            self.agent_funcs = agent_funcs
            self.history = []
            self.quality = quality
            self.evaluation = evaluation
            self.winner = ""
            self.loser = ""
            self.feedback = ""

    def new_agent(self, agent_name, agent_view, agent_func):
        self.agents.append(agent_name)
        self.agent_views[agent_name] = agent_view
        self.agent_funcs[agent_name] = agent_func
    
    def append(self, agent_name, val):
        self.history.append([agent_name, val])

    def format_string(self, format_str, agent_name = "", opp_name = ""):
        format_str = format_str.replace("<agent_name>", agent_name).replace("<opp_name>", opp_name)
        for i, agent in enumerate(self.agents):
            format_str = format_str.replace(f"<agent_{i+1}>", agent)
        format_str = format_str.replace("<quality>", self.quality).replace("<evaluation>", self.evaluation)
        format_str = format_str.replace("<topic>", self.topic)
        format_str = format_str.replace("<winner>", self.winner).replace("<loser>", self.loser)
        return format_str

    def generate_starting_view(self, system_prompt, agent_name, opp_name, agent_view = None):
        messages = []
        agent_view = agent_view.lower()
        if agent_view is None:
            self.agent_views[agent_name] = agent_view
        messages.append({"role": "system", "content": self.format_string(system_prompt, agent_name, opp_name)})
        messages.append({"role": "system", "content": f"Please provide your initial view on the topic in one sentence, which is that {agent_view}."})
        return messages
    
    def start(self, system_prompt):
        assert len(self.agents) == 2
        talking_order = r.permutation(range(0,2))
        for i in talking_order:
            agent = self.agents[i]
            msgs = self.generate_starting_view(system_prompt, agent, self.agents[(i+1)%2], self.agent_views[agent])
            self.append(agent, self.agent_funcs[agent](msgs))

    def generate_debate_history(self,
                               agent_name = None):
        messages = []
        for h in self.history:
            role = "user" if agent_name is None or not h[0] == agent_name else "assistant"
            role = "system" if h[0] == "Judge" or h[0] == "Feedback" else role
            messages.append({"role": role, "content": f"{h[0]}: {h[1]}"})
        return messages

    def generate_debate_prompt(self, 
                                s_prompt_1, 
                                agent_name, 
                                opp_name,
                                s_prompt_2 = "Please continue the debate using at most one sentence - remember, you are <agent_name>.",
                                ):
        messages = []
        messages.append({"role": "system", "content": self.format_string(s_prompt_1, agent_name, opp_name)})
        messages.extend(self.generate_debate_history(agent_name))
        if not s_prompt_2 is None:
            messages.append({"role": "system", "content": self.format_string(s_prompt_2, agent_name, opp_name)})
        return messages

    
    def generate_debate(self,
                        s_prompt_1,
                        s_prompt_2 = None,
                        debate_len = 10,
                        ):
        talking_order = r.randint(0,len(self.agents), size = debate_len)
        for i in talking_order:
            agent_name = self.agents[i]
            opp_name = self.agents[(i+1)%2]
            msgs = self.generate_debate_prompt(s_prompt_1, agent_name, opp_name, s_prompt_2)
            self.append(agent_name, self.agent_funcs[agent_name](msgs))
    
    def generate_ranking_prompt(self, ranking_prompt, quality = None, evaluation = None):
        messages = []
        if quality is None:
            quality = self.quality
        if evaluation is None:
            evaluation = self.evaluation
        ranking_prompt = self.format_string(ranking_prompt)
        messages.append({"role": "system", "content": ranking_prompt})
        messages.extend(self.generate_debate_history(agent_name = None))
        messages.append({"role": "system", "content": "Your analysis starts now. Please end your analysis with the name of the winner on a line by itself."})
        return messages
    
    def set_winner_from_prompt(self, prompt):
        lines = prompt.split("\n")
        for l in lines:
            if l in self.agents:
                self.winner = l
                self.loser = (self.agents[1] if self.agents[0] == self.winner else self.agents[0]) if self.winner != "" else ""
                return True
        return False
    
    def generate_ranking(self, ranking_prompt, judge_func, quality = None, evaluation = None):
        msgs = self.generate_ranking_prompt(ranking_prompt, quality, evaluation)
        out = judge_func(msgs)
        self.set_winner_from_prompt(out)
        self.append("Judge", out)
    
    def generate_feedback_prompt(self, feedback_prompt, quality = None, evaluation = None):
        if quality is None:
            quality = self.quality
        if evaluation is None:
            evaluation = self.evaluation
        messages = []
        feedback_prompt = self.format_string(feedback_prompt)
        messages.append({"role": "system", "content": feedback_prompt})
        messages.extend(self.generate_debate_history(agent_name = None))
        messages.append({"role": "system", "content": f"The agent that {self.quality} was declared to be {self.winner}, not {self.loser}."})
        messages.append({"role": "system", "content": f"Please provide up to 1 sentence of feedback for {self.loser} on how to better {self.quality}. Do not mention agent names or topics."})
        return messages
    
    def generate_feedback(self, feedback_prompt, judge_func, quality = None, evaluation = None):
        msgs = self.generate_feedback_prompt(feedback_prompt, quality, evaluation)
        out = judge_func(msgs)
        self.set_winner_from_prompt(out)
        self.append("Feedback", out)
    
    def get(self, name):
        for h in self.history:
            if h[0] == name:
                return h[1]
        return None

    def save(self, filename):
        with open(filename, 'wb') as f:
            x = (self.topic, self.agents, self.agent_views, self.history, self.quality, self.evaluation, self.winner, self.loser, self.feedback)
            pickle.dump(x, f)
    
    def load(self, filename):
        with open(filename, 'rb') as f:
            self.topic, self.agents, self.agent_views, self.history, self.quality, self.evaluation, self.winner, self.loser, self.feedback = pickle.load(f)