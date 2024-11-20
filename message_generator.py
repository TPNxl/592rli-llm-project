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
            self.rating = ""
            self.winner = ""
            self.loser = ""
            self.feedback = ""
            self.curr_agent = 0
    
    def reset(self):
        self.topic = ""
        self.agents = []
        self.agent_views = {}
        self.agent_funcs = {}
        self.history = []
        self.quality = ""
        self.evaluation = ""
        self.rating = ""
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
        if agent_name is not None and agent_name != "":
            format_str = format_str.replace("<agent_view>", self.agent_views[agent_name])
        if opp_name is not None and opp_name != "":
            format_str = format_str.replace("<opp_view>", self.agent_views[opp_name])
        return format_str

    def generate_starting_view(self, system_prompt, agent_name, opp_name, agent_view = None):
        messages = []
        agent_view = agent_view.lower()
        if agent_view is None:
            self.agent_views[agent_name] = agent_view
        messages.append({"role": "system", "content": self.format_string(system_prompt, agent_name, opp_name)})
        return messages
    
    def start(self, system_prompt):
        assert len(self.agents) == 2
        self.curr_agent = r.randint(0, len(self.agents))
        for i in range(0, len(self.agents)):
            agent = self.agents[i]
            msgs = self.generate_starting_view(system_prompt, agent, self.agents[(i+1)%2], self.agent_views[agent])
            self.append(agent, self.agent_funcs[agent](msgs))

    def generate_debate_history(self,
                               agent_name = None):
        messages = []
        for h in self.history:
            role = "user" if agent_name is None or not h[0] == agent_name else "assistant"
            role = "system" if h[0] == "Judge" or h[0] == "Feedback" else role
            messages.append({"role": role, "content": f"{h[0]}: \"{h[1]['content']}\""})
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
        for n in range(debate_len):
            agent_name = self.agents[self.curr_agent]
            opp_name = self.agents[(self.curr_agent+1)%len(self.agents)]
            msgs = self.generate_debate_prompt(s_prompt_1, agent_name, opp_name, s_prompt_2)
            agent_output = self.agent_funcs[agent_name](msgs)
            agent_str = agent_output['content']
            agent_str = agent_str.replace(f"{agent_name}: ", "")
            agent_str = agent_str.replace("\"", "")
            agent_output['content'] = agent_str
            self.append(agent_name, agent_output)
            self.curr_agent = (self.curr_agent + 1) % len(self.agents)
    
    def generate_ranking_prompt(self, ranking_prompt, quality = None, evaluation = None):
        messages = []
        if quality is None:
            quality = self.quality
        if evaluation is None:
            evaluation = self.evaluation
        ranking_prompt = self.format_string(ranking_prompt)
        messages.append({"role": "system", "content": ranking_prompt})
        messages.extend(self.generate_debate_history(agent_name = None))
        messages.append({"role": "system", "content": "Your analysis starts now. Please end your analysis with the line: \"Winner: Agent <name>\""})
        return messages
    
    def set_winner_from_prompt(self, prompt):
        winner_line = prompt.split("\n")[-1]
        try:
            self.winner = winner_line.split("Winner: Agent ")[1]
            self.loser = (self.agents[1] if self.agents[0] == self.winner else self.agents[0]) if self.winner != "" else ""
            return True
        except Exception:
            print("Error setting winner from prompt")
            print("Winner line:", winner_line)
        return False
    
    def generate_ranking(self, ranking_prompt, judge_func, quality = None, evaluation = None):
        msgs = self.generate_ranking_prompt(ranking_prompt, quality, evaluation)
        out = judge_func(msgs)
        out['role'] = "system"
        self.set_winner_from_prompt(out['content'])
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
        self.append("Feedback", {"role": "system", "content": out})
    
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

if __name__ == "__main__":
    """
    A client to open message files and print their contents in a nice manner with labels.
    """
    print("Welcome to the MessageGenerator client!")
    fn = input("Please enter the filename of the message file you would like to load: ")
    m = MessageGenerator(filename = fn)
    print("Topic:", m.topic)
    print("Agents:", m.agents)
    print("Agent Views:", m.agent_views)
    print("Quality:", m.quality)
    print("Evaluation:", m.evaluation)
    print("History:")
    for h in m.history:
        print(h)
    print("Winner:", m.winner)
    print("Loser:", m.loser)
    print("Feedback:", m.feedback)
   