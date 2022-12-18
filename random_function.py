# Imports

import cmath
from collections.abc import Iterable
from dataclasses import dataclass
import itertools
import matplotlib.pyplot as plt
import operator
import random
from typing import Literal, Union
import warnings

import numpy as np
import pandas as pd

# Settings

warnings.filterwarnings("error")

# Types

rational = Union[int, float]
collection = Union[list, set]

# Globals

FUNCS = [
    "sin",
    "cos",
    "tan",
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
    operator: Literal["+", "-", "/", "*", "^"] = None
    func: str = None
    side: Literal["left", "right"] = None
    type: Literal["const", "func", "init", "var"] = None
    var: Union[rational, str] = None

class NotPlottable(Exception):
    pass

def random_exclude(min: rational=DEFAULT_MIN, max: rational=DEFAULT_MAX, exclude=[0]):
    term_type = random.choice(["int", "float"])

    if term_type == "int":
        randfunc = random.randint
    elif term_type == "float":
        randfunc = random.uniform

    if exclude is None:
        x = randfunc(min, max)
    else:
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

def translate_ops(op):
    translation = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "^": operator.pow
    }

    return translation[op]

def parse_log(log_str: str):
    log_lst = log_str.split("b")
    base = float(log_lst[1])

    def log_func(x: rational):
        return cmath.log(x, base)

    return log_func

def translate_funcs(func):
    translation = {
        "sin": cmath.sin,
        "cos": cmath.cos,
        "tan": cmath.tan,
        "exp": cmath.exp,
        "ln": cmath.log
    }

    if "log" in func:
        true_func = "log"
        translation["log"] = parse_log(func)
    else:
        true_func = func

    return translation[true_func]

def initialize_tree(inputs: list):
    tree = {}
    for idx, input in enumerate(inputs):
        tree_entry = TreeEntry()
        tree_entry.type = "init"
        tree_entry.var = input
        tree[idx] = [tree_entry]

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

def add_operators(
    vars: list, 
    inputs: list,
    num_operators: int,
    funcs: list=FUNCS,
    basic_funcs: list=BASIC_FUNCS,
    base_min: rational=DEFAULT_MIN,
    base_max: rational=DEFAULT_MAX,
    c_min: rational=DEFAULT_MIN,
    c_max: rational=DEFAULT_MAX,
    nothing_is_operator: bool=True
) -> tuple[list, dict]:
    inputlist = [x for x in inputs if type(x) is str] # remove constant from list
    function_tree = initialize_tree(inputlist)
    actions = ["nothing", "basic", "apply_func"]
    if len(vars) > 1:
        subactions = ["constant", "variable"]
    else:
        subactions = ["constant"]
    sides = ["left", "right"]

    counter = 0
    while num_operators > 0:
        tree_entry = TreeEntry()
        action = random.choice(actions)
        idx = random.randint(0, len(inputlist) - 1)
        input = inputlist[idx]

        if action == "nothing":
            if nothing_is_operator == True:
                pass
            elif nothing_is_operator == False:
                continue
        elif action == "apply_func":
            func = random.choice(funcs)

            if func == "log":
                base = random_exclude(base_min, base_max, exclude=[0])
                func = f"logb{base}"

            tree_entry.func = func
            tree_entry.type = "func"
        elif action == "basic":
            basic_func = random.choice(basic_funcs)
            subaction = random.choice(subactions)
            side = random.choice(sides)

            if subaction == "constant":
                if basic_func in {"*", "/", "^"}:
                    term = random_exclude(c_min, c_max, exclude=[0,1])
                else:
                    term = random_exclude(c_min, c_max, exclude=[0])

                tree_entry.const = term
                tree_entry.type = "const"
            elif subaction == "variable":
                valid_vars = [var for var in vars if var != input]
                term = random.choice(valid_vars)

                tree_entry.type = "var"
                tree_entry.var = term

            tree_entry.operator = basic_func
            tree_entry.side = side

        function_tree[idx].append(tree_entry)
        num_operators -= 1
        counter += 1

    return function_tree

def generate_ranges(vars: list, min: rational, max: rational) -> dict:
    intervals = {}
    for var in vars:
        range1 = random_exclude(min, max, exclude=None)
        range2 = random_exclude(min, max, exclude=[range1]) # ensure range1 != range2
        range = sorted([range1, range2])
            
        intervals[var] = (range[0], range[1])

    return intervals

def generate_inputs(
    vars: list, 
    min: rational, 
    max: rational,
    density: int=100,
    intervals: dict=None
) -> Iterable:
    if intervals is None:
        intervals = generate_ranges(vars, min, max)

    input_dict = {}
    for var in vars:
        input_dict[var] = np.linspace(intervals[var][0], intervals[var][1], density)

    if len(input_dict) > 1:
        input_tuples = itertools.product(*input_dict.values())
        inputs = [dict(zip(input_dict.keys(), x)) for x in input_tuples]
    else:
        inputs = [{vars[0]: x} for x in input_dict[vars[0]]]
    
    return inputs

def create_results_dict(function_tree: dict, inputs: list[dict]=None):
    results = {key: [] for key in function_tree.keys()}

    for entry in function_tree.keys():
        branch = function_tree[entry]
        pruned_branch = branch[1:len(branch)]

        for input in inputs:
            prev_val = input[branch[0].var]

            for tree_entry in pruned_branch:
                type = tree_entry.type
                op = tree_entry.operator
                func = tree_entry.func
                side = tree_entry.side
                
                if op is not None:
                    opfunc = translate_ops(op)

                    if type == "var":
                        val = input[tree_entry.var]
                    elif type == "const":
                        val = tree_entry.const

                    if side == "left":
                        args = [val, prev_val]
                    elif side == "right":
                        args = [prev_val, val]

                    try:
                        new_val = opfunc(*args)
                    except ZeroDivisionError:
                        break
                    except RuntimeWarning: # this happens when large or small value
                        new_val = 0
                elif type == "func":
                    tfunc = translate_funcs(func)

                    try:
                        new_val = tfunc(prev_val)
                    except ValueError:
                        break
                    except OverflowError:
                        new_val = float("inf")
                elif type is None:
                    continue

                prev_val = new_val
            
            results[entry].append(prev_val)

    return results

def strip_complex(num: complex):
    if num.imag == 0:
        num = num.real
    
    return num

def exists_complex(lst):
    return any(type(num) is complex for num in lst)

def evaluate_function(results: dict, terminal_inputs: list):
    lhs = [strip_complex(sum(item)) for item in zip(*results.values())]
    constant = find_constant(terminal_inputs)

    if constant is not None:
        lhs = [item + constant for item in lhs]

    return lhs

def create_output_table(inputs: list[dict], lhs: list):
    df = pd.DataFrame(inputs)
    df["lhs"] = lhs

    return df

def plot_function(df):
    num_vars = len(df.columns) - 1
    lhs = df["lhs"].to_list()
    has_complex = exists_complex(lhs)

    ax = plt.axes(projection="3d")

    if (has_complex and num_vars > 1) or num_vars > 2:
        raise NotPlottable("This function is currently not plottable.")

    if has_complex and num_vars == 1:
        var_col = df.iloc[:, 0].to_list()
        lhs_complex = df.iloc[:, 1].to_list()
        lhs_real = [x.real for x in lhs_complex]
        lhs_imag = [complex(x).imag for x in lhs_complex]

        ax.scatter(var_col, lhs_real, lhs_imag)
    elif has_complex == False and num_vars == 2:
        var1 = df.iloc[:, 0].to_list()
        var2 = df.iloc[:, 1].to_list()
        lhs = df.iloc[:, 2].to_list()

        ax.scatter(var1, var2, lhs)
    elif has_complex == False and num_vars == 1:
        var = df.iloc[:, 0].to_list()
        lhs = df.iloc[:, 1].to_list()

        ax.scatter(var, lhs)

    plt.show()

    return None
