FROM ghcr.io/lambda-feedback/evaluation-function-base/python:3.12 AS builder

RUN pip install poetry==2.2.1

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

COPY pyproject.toml poetry.lock ./

RUN --mount=type=cache,target=$POETRY_CACHE_DIR \
    poetry install --without dev --no-root


FROM ghcr.io/lambda-feedback/evaluation-function-base/python:3.12

# System dependencies required by OpenCV, Open3D and OpenMP.
RUN if command -v apt-get >/dev/null 2>&1; then \
        apt-get update && apt-get install -y --no-install-recommends \
            libgl1 \
            libegl1 \
            libglib2.0-0 \
            libgomp1 \
            libusb-1.0-0 \
        && rm -rf /var/lib/apt/lists/*; \
    elif command -v dnf >/dev/null 2>&1; then \
        dnf install -y mesa-libGL mesa-libEGL glib2 libgomp libusbx && dnf clean all; \
    elif command -v yum >/dev/null 2>&1; then \
        yum install -y mesa-libGL mesa-libEGL glib2 libgomp libusbx && yum clean all; \
    else \
        echo "Unsupported package manager: cannot install system dependencies" >&2; \
        exit 1; \
    fi

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Copy application before compilation.
COPY evaluation_function ./evaluation_function

# Precompile Python files to reduce compilation overhead during cold starts.
RUN python -m compileall -q /app/evaluation_function

# Writable scratch directories (/tmp is the only writable path on AWS Lambda
# and an in-memory filesystem on Cloud Run).
ENV EDUVISION_WORK_DIR=/tmp/eduvision \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_OFFLINE=1 \
    YOLO_VERBOSE=False

# Analysis: CAD fitting and multi-view fusion (the slowest stages with
# parts) run in a process pool forked at the start of each request
# (EDUVISION_WORKERS=auto: one process per available CPU, limited by free
# memory). The pool uses no /dev/shm semaphores, so it works on AWS Lambda
# too; results are identical to a single process. 2 image threads, and a
# time limit below FUNCTION_WORKER_SEND_TIMEOUT so the student gets a
# message instead of a server error.
#
# For Cloud Run with GPU: EDUVISION_WORKERS=2, GEARBOX_IMAGE_THREADS=1.
ENV EDUVISION_WORKERS=auto \
    GEARBOX_IMAGE_THREADS=2 \
    EDUVISION_TIME_LIMIT_S=270

# Analysis process: with the rpc interface (a long-running server) a zygote
# (worker.py --zygote) imports the libraries and warms up the YOLO model
# once, when the server starts; every request is analysed in a fork of it
# (no ~10-15 s start per request; memory is still returned after each
# request, because the fork exits). The zygote is idle and single-threaded
# between requests, so freezing the environment (AWS Lambda) does not
# affect it; a fork-safety self-test runs at start. If the zygote cannot
# start, or a fork fails, the request is analysed in a fresh process, as
# before (EDUVISION_WORKER_MODE=once forces that for every request).
# EDUVISION_WORKER_REUSE=1 (one long-lived analysis process) is not used: it
# was seen to abort (SIGABRT) on AWS Lambda after the environment was
# frozen between invocations.
ENV EDUVISION_WORKER_REUSE=0

# Function startup configuration. AWS Lambda timeout is 300 s:
# EDUVISION_TIME_LIMIT_S (270) < FUNCTION_WORKER_SEND_TIMEOUT (285) < 300.
# Lambda sends one request at a time to each instance, so the rpc interface
# is enough. The server imports only lf_toolkit (~2 s); heavy libraries load
# in the zygote, in the background (the first request waits for it).
#
# For Cloud Run with GPU: FUNCTION_INTERFACE="file", FUNCTION_MAX_PROCS="2",
# no FUNCTION_RPC_TRANSPORT, and Cloud Run concurrency equal to
# FUNCTION_MAX_PROCS (rpc serves one request at a time per instance).
ENV FUNCTION_COMMAND="python" \
    FUNCTION_ARGS="-m,evaluation_function.main" \
    FUNCTION_INTERFACE="rpc" \
    FUNCTION_RPC_TRANSPORT="stdio" \
    FUNCTION_WORKER_SEND_TIMEOUT="285s" \
    FUNCTION_WORKER_START_TIMEOUT="15s" \
    FUNCTION_WORKER_STOP_TIMEOUT="5s"

# Enable temporarily when investigating cold starts.
# Remove or override in production after profiling.
# ENV PYTHONPROFILEIMPORTTIME=1

ENV LOG_LEVEL="debug"
