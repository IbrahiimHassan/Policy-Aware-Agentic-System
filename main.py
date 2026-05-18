from graph.builder import build_graph, save_graph_png

import os
import platform


def visualize_graph(app):
    del app  # runtime graph unchanged; PNG uses documentation mermaid (all branch labels)
    save_graph_png("graph.png")

    # print("\n Opening graph image...")

   # os.startfile(file_path)

def run():
    app = build_graph()
    visualize_graph(app)

    print("Press Ctrl+C to exit.")

    try:
        while True:
            user_input = input("Enter your request: ")
            state = {
                "input": user_input
            }
            result = app.invoke(state)
            if isinstance(result, dict) and "explanation" in result:
                print(result["explanation"])
            else:
                print(result)
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    run()