from collections.abc import Iterable
from dataclasses import dataclass
import itertools
import math
import random
from typing import Literal, Union

import numpy as np

rational = Union[int, float]
collection = Union[list, set]

FUNCS = [
    "sin",
    "cos",
    "tan",
    "cos",
    "sec",
    "cot",
    "exp",
    "ln",
    "log"
]

BASIC_FUNCS = [
    "+",
    "-",
    "*",
    "/",
    "^"
]

DEFAULT_MIN = -9223372036854775808
DEFAULT_MAX = 9223372036854775807

@dataclass
class TreeEntry:
    const: rational = None
    input: rational = None
    operator: Literal["+", "-", "/", "*", "^"] = None
    func: str = None
    side: Literal["left", "right"] = None
    type: Literal["const", "func", "var"] = None
    var: Union[rational, str] = None

def random_exclude(min: rational=DEFAULT_MIN, max: rational=DEFAULT_MAX, exclude=[0]):
    term_type = random.choice(["int", "float"])

    if term_type == "int":
        randfunc = random.randint
    elif term_type == "float":
        randfunc = random.uniform

    x = exclude[0]
    while x in exclude:
        x = randfunc(min, max)

    return x

def find_constant(inputs: list):
    for input in inputs:
        if type(input) in {int, float}:
            constant = input

            break
    else:
        constant = None

    return constant

def initialize_tree(inputs: list):
    tree = {}
    for idx, input in enumerate(inputs):
        tree[idx] = [input]

    return tree

def try_round(x):
    try:
        x = round(x, 2)
    except TypeError:
        pass

    return x

def set_terminal_inputs(
    vars: list, 
    num_terms: int, 
    min: rational=DEFAULT_MIN, 
    max: rational=DEFAULT_MAX
) -> list:
    terminal_inputs = []
    choicelist = vars + [random_exclude(min, max, exclude=[0])]

    for _ in range(num_terms):
        term_base = random.choice(choicelist)
        
        if type(term_base) in {float, int}: # we only want one constant
            choicelist = vars
            
        terminal_inputs.append(term_base)

    return terminal_inputs

def create_connectors(num_connectors: int, connector_funcs: collection=BASIC_FUNCS):
    return [random.choice(connector_funcs) for _ in range(num_connectors)]

def add_operators(
    vars: list, 
    inputs: list,
    num_operators: int,
    funcs: list=FUNCS,
    basic_funcs: list=BASIC_FUNCS,
    nothing_is_operator: bool=True
) -> tuple[list, dict]:
    inputlist = [x for x in inputs if type(x) is str] # remove constant from list
    function_tree = initialize_tree(inputlist)
    actions = ["nothing", "basic", "apply_func"]
    subactions = ["constant", "variable"]
    sides = ["left", "right"]

    counter = 0
    while num_operators > 0:
        tree_entry = TreeEntry()
        action = random.choice(actions)
        idx = random.randint(0, len(inputlist) - 1)
        input = inputlist[idx]

        if action == "nothing":
            if nothing_is_operator == True:
                tree_entry = None
            elif nothing_is_operator == False:
                continue
        elif action == "apply_func":
            func = random.choice(funcs)

            if func == "log":
                base = random_exclude(1, 20, exclude=[0])
                func = f"logb{round(base, 2)}"

            inputlist[idx] = f"({func}({input}))"
            tree_entry.func = func
            tree_entry.type = "func"
        elif action == "basic":
            basic_func = random.choice(basic_funcs)
            subaction = random.choice(subactions)
            side = random.choice(sides)

            if subaction == "constant":
                if basic_func in {"*", "/", "^"}:
                    term = random_exclude(-3, 3, exclude=[0,1])
                else:
                    term = random_exclude(-3, 3, exclude=[0])

                tree_entry.const = term
                tree_entry.type = "const"

                if side == "right":
                    if term < 0 and basic_func == "-":
                        basic_func = "+"
                        term = -1 * term
                    elif term < 0 and basic_func == "+":
                        basic_func = "-"
                        term = -1 * term
            elif subaction == "variable":
                valid_vars = [var for var in vars if var != input]
                term = random.choice(valid_vars)

                tree_entry.type = "var"
                tree_entry.var = term

            if side == "right":
                inputlist[idx] = f"({input}{basic_func}{try_round(term)})"
            elif side == "left":
                inputlist[idx] = f"({try_round(term)}{basic_func}{input})"

            tree_entry.operator = basic_func
            tree_entry.side = side

        function_tree[idx].append(tree_entry)
        num_operators -= 1
        counter += 1

    return inputlist, function_tree

def generate_ranges(vars: list, min: rational, max: rational) -> dict:
    intervals = {}
    for var in vars:
        range = sorted([random.randint(min, max), random.randint(min, max)])
            
        intervals[var] = (range[0], range[1])

    return intervals

def generate_inputs(vars: list, intervals: dict=None) -> Iterable:
    if intervals is None:
        intervals = generate_ranges(vars, -10, 10)

    input_dict = {}
    for var in vars:
        input_dict[var] = np.linspace(intervals[var][0], intervals[var][1], 5)

    if len(input_dict) > 1:
        input_tuples = itertools.product(*input_dict.values())
        inputs = [dict(zip(input_dict.keys(), x)) for x in input_tuples]
    else:
        inputs = [{vars[0]: x} for x in input_dict[vars[0]]]
    
    return inputs

def evaluate_function(
    function_tree: dict, 
    connectors: list=None,
    inputs: list[dict]=None,
    min: rational=-50, 
    max: rational=50
):
    for entry in function_tree:
        print(type(function_tree[entry]))

    return None

terminal_inputs = set_terminal_inputs(vars=["x", "y", "z"], num_terms=3, min=-5, max=5)
inputs_w_operators, function_tree = add_operators(
    vars=["x", "y", "z"], 
    inputs=terminal_inputs, 
    num_operators=10
)

print(terminal_inputs)
print(inputs_w_operators)
print(function_tree)


x = generate_inputs(vars=["x"])
for elem in x:
    print(elem)

print(type(x))

evaluate_function(function_tree)