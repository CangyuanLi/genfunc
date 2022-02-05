import random

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

DEFAULT_MIN = -9223372036854775808
DEFAULT_MAX = 9223372036854775807

def random_exclude(min, max, exclude=[0]):
    term_type = random.choice(["int", "float"])

    if term_type == "int":
        randfunc = random.randint
    elif term_type == "float":
        randfunc = random.uniform

    x = exclude[0]
    while x in exclude:
        x = randfunc(min, max)

    return x

def find_constant(inputs):
    for input in inputs:
        if type(input) in {int, float}:
            constant = [input]

            break
    else:
        constant = []

    return constant

def set_terminal_inputs(vars, num_terms, min=DEFAULT_MIN, max=DEFAULT_MAX):
    terminal_inputs = []
    choicelist = vars + [random_exclude(min, max, exclude=[0])]

    for _ in range(num_terms):
        term_base = random.choice(choicelist)
        
        if type(term_base) in {float, int}: # we only want one constant
            choicelist = vars
            
        terminal_inputs.append(term_base)

    return terminal_inputs

def add_operators(vars, inputs, num_operators, nothing_is_operator: bool=True):
    inputlist = [x for x in inputs if type(x) is str] # remove constant from list
    constant = find_constant(inputs)
    actions = ["nothing", "+", "-", "/", "*", "^", "apply_func"]
    subactions = ["constant", "variable"]

    while num_operators > 0:
        action = random.choice(actions)
        idx = random.randint(0, len(inputlist) - 1)
        input = inputlist[idx]

        if action == "nothing":
            if nothing_is_operator == True:
                pass
            elif nothing_is_operator == False:
                continue
        elif action == "apply_func":
            func = random.choice(FUNCS)

            if func == "log":
                base = random_exclude(1, 20, exclude=[0])
                func = f"logb{base}"

            inputlist[idx] = f"({func}({input}))"
        else:
            subaction = random.choice(subactions)

            if subaction == "constant":
                if action in {"*", "/", "^"}:
                    term = random_exclude(-3, 3, exclude=[0,1])
                else:
                    term = random_exclude(-3, 3, exclude=[0])

                if term < 0 and action == "-":
                    action = "+"
                    term = -1 * term
                elif term < 0 and action == "+":
                    action = "-"
                    term = -1 * term
            else:
                valid_vars = [var for var in vars if var != input]
                term = random.choice(valid_vars)

            inputlist[idx] = f"({input}{action}{term})"

        num_operators -= 1

    return inputlist + constant

terminal_inputs = set_terminal_inputs(vars=["x", "y", "z"], num_terms=3, min=-5, max=5)
inputs_w_operators = add_operators(vars=["x", "y", "z"], inputs=terminal_inputs, num_operators=10)

print(terminal_inputs)
print(inputs_w_operators)