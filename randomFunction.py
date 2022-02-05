import random
import json

from pathlib import Path

base_path = Path(__file__).parent

with open(base_path / "function_list.json") as f:
    funcs = json.load(f)

def dict_to_list(some_dict: dict=funcs):
    dict_as_list = []

    for val in some_dict.values():
        for item in val:
            dict_as_list.append(item)

    return dict_as_list

def create_possible_inputs(varlist: list, funclist: list, num_terms: int):
    all_list = varlist + funclist
    remaining_terms = num_terms - len(varlist)

    weights = tuple((random.randrange(0, 100) for _ in range(remaining_terms)))
    sum_weights = sum(weights)
    weights = [100 * w / sum_weights for w in weights]

    rest_inputs = random.choices(all_list, weights=weights, k=remaining_terms)

    return varlist + rest_inputs

def add_roles(varlist, poss_inputs):
    for idx, input in enumerate(poss_inputs):
        if input in varlist:
            poss_inputs[idx] = (input, random.choice(['alone', 'terminator']))
    
    return poss_inputs

def construct_function(poss_inputs):
    funcstr = ''
    funcinputs = [input for input in poss_inputs if type(input) != tuple]
    varinputs = [input for input in poss_inputs if type(input) == tuple]

    for input in varinputs:
        if type(input) == tuple:
            if input[1] == 'alone':
                funcstr = funcstr + input[0] + random.choice(
                    [
                        f'^{random.choice(range(10))}', 
                        f'*{random.choice(range(10))}'
                    ]
                ) + '+'
            elif input[1] == 'terminator':
                num_passes = 1
                term = ''
                while len(funcinputs) > 0:
                    funcidx = random.choice(range(len(funcinputs)))
                    if num_passes == 1:
                        term = term + funcinputs[funcidx] + f'({input[0]})'
                    else:
                        term = funcinputs[funcidx] + '(' + term

                    num_passes += 1
                    funcinputs.pop(funcidx)
                funcstr = funcstr + term + '+'

    return funcstr


varlist = ['x', 'y', 'z']
funclist = dict_to_list(funcs)

poss_inputs = create_possible_inputs(varlist=varlist, funclist=funclist, num_terms=15)
poss_inputs = add_roles(varlist, poss_inputs)
funcstr = construct_function(poss_inputs)

print(poss_inputs)
print(funcstr)





