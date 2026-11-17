from lf_toolkit import create_server, run
from lf_toolkit.io.file_server import FileServer

from .evaluation import evaluation_function, prestart
from .preview import preview_function

def main():
    """Run the IPC server with the evaluation and preview functions.
    """
    server = create_server()

    server.eval(evaluation_function)
    server.preview(preview_function)

    # Long-running server (rpc): the analysis process (libraries + YOLO
    # model) loads in the background while the server starts answering.
    # The "file" interface handles a single request per process.
    prestart(long_lived=not isinstance(server, FileServer))

    run(server)

if __name__ == "__main__":
    main()
