import ast
import re

curr_top = ""

with open("topics") as file:
    for line in file:
        if line[0].isalpha():
            curr_top = line
        elif line == "\n" or line == "":
            continue
        else:
            modified_line = re.sub(r"^(100|[1-9][0-9]?)\.\s", "", line)
            k = ast.literal_eval(modified_line)
            act_tup = (curr_top.strip(), k)
            # Make any changes you want here, or alternatively pull the modified_line to a broader scope if you want to work with it outside the with and for loop
            print(act_tup)