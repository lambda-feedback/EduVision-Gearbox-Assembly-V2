
from lf_toolkit import create_server, run

from .evaluation import evaluation_function
from .preview import preview_function

def main():
    """Run the IPC server with the evaluation and preview functions.
    """
    server = create_server()

    server.eval(evaluation_function)
    server.preview(preview_function)

    run(server)

if __name__ == "__main__":
    main()