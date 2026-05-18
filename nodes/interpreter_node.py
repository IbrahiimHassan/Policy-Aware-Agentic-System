from interpreter.llm_interpreter import llm_interpreter


def interpreter_node(state):
    print("\n [NODE] Interpreter")
    result = llm_interpreter(state["input"])

    return result