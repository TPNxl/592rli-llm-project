import torch
from torch.utils.data import Dataset
import ast
import re

class TextDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.topics = []
    
        curr_top = ""
        
        with open(self.file_path, "r") as file:
            for line in file:
                line = line.strip().lower()  # Strip whitespace around the line
                if not line:  # Skip empty lines
                    continue
                elif line[0].isalpha():  # If the line starts with a letter, it's a topic line
                    curr_top = line
                else:
                    modified_line = re.sub(r"^(100|[1-9][0-9]?)\.\s", "", line)  # Remove leading numbering
                    try:
                        content = ast.literal_eval(modified_line)
                        self.topics.append((curr_top.strip(), content))
                    except (SyntaxError, ValueError):
                        print(f"Skipping line due to parsing error: {line}")

        self.index_list = torch.randperm(len(self.topics)).tolist()

    def __len__(self):
        return len(self.topics)

    def __getitem__(self, idx):
        return self.topics[self.index_list[idx]]