import random_function as randfunc

VARS = ["x", "y"]

def main():
    terminal_inputs = randfunc.set_terminal_inputs(vars=VARS, num_terms=4, min=-100, max=100)
    function_tree = randfunc.add_operators(
        vars=VARS, 
        inputs=terminal_inputs, 
        num_operators=10,
        c_min=-100,
        c_max=100,
        nothing_is_operator=True
    )
    x = randfunc.generate_inputs(vars=VARS, min=-100, max=100)
    res = randfunc.create_results_dict(function_tree, inputs=x)
    lhs = randfunc.evaluate_function(res, terminal_inputs)
    df = randfunc.create_output_table(x, lhs)
    randfunc.plot_function(df)


if __name__ == "__main__":
    main()
