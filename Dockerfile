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

# Writable scratch directories (/tmp is an in-memory filesystem on Cloud Run).
ENV EDUVISION_WORK_DIR=/tmp/eduvision \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_OFFLINE=1 \
    YOLO_VERBOSE=False

# Analysis: worker-process pools for CAD fitting and fusion, and a time limit
# below FUNCTION_WORKER_SEND_TIMEOUT so the student gets a message instead of
# a server error.
#
# Sized for Cloud Run with 6 vCPU / 16 GiB / 1 L4 running FUNCTION_MAX_PROCS
# (2) submissions at once: each submission gets a pool of 2 processes and 1
# image thread, so 2 x (1 analysis + 2 pool) ~= 6 busy processes on 6 vCPU.
# Memory, not CPU or GPU, is the limit: estimated ~10 GiB at peak for 2
# concurrent submissions (/tmp is in-memory and counts too). Check
# memory_after_mb / worker_rss_mb in the debug section before raising
# FUNCTION_MAX_PROCS. On AWS Lambda (no /dev/shm semaphores) use
# EDUVISION_WORKERS=auto, which falls back to a single process.
ENV EDUVISION_WORKERS=2 \
    GEARBOX_IMAGE_THREADS=1 \
    EDUVISION_TIME_LIMIT_S=270

# Every request is analysed in a fresh process (safest: memory is returned
# after each request). EDUVISION_WORKER_REUSE=1 would keep the process
# loaded between requests (~10-15 s faster). It was seen to abort (SIGABRT)
# on AWS Lambda after the container was frozen between invocations; Cloud Run
# also throttles CPU between requests by default, so test before enabling.
ENV EDUVISION_WORKER_REUSE=0

# Function startup configuration. Cloud Run request timeout should be set to
# at least 300 s (set in the console):
# EDUVISION_TIME_LIMIT_S (270) < FUNCTION_WORKER_SEND_TIMEOUT (285) < 300.
#
# File interface: shimmy starts a fresh server process per request (it sets
# EVAL_IO=FILE, which lf_toolkit picks up) and runs up to FUNCTION_MAX_PROCS
# of them at once. The rpc/stdio interface serves one request at a time per
# instance, so concurrent requests queued behind a running analysis and could
# hit the send timeout. The server imports only lf_toolkit (~2 s); heavy
# libraries load in the per-request analysis process.
#
# Set Cloud Run concurrency equal to FUNCTION_MAX_PROCS (2) so no request
# waits inside shimmy for a free slot.
ENV FUNCTION_COMMAND="python" \
    FUNCTION_ARGS="-m,evaluation_function.main" \
    FUNCTION_INTERFACE="file" \
    FUNCTION_MAX_PROCS="2" \
    FUNCTION_WORKER_SEND_TIMEOUT="285s" \
    FUNCTION_WORKER_START_TIMEOUT="15s" \
    FUNCTION_WORKER_STOP_TIMEOUT="5s"

# Enable temporarily when investigating cold starts.
# Remove or override in production after profiling.
# ENV PYTHONPROFILEIMPORTTIME=1

ENV LOG_LEVEL="debug"
