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

# AWS Lambda writable directories.
ENV EDUVISION_WORK_DIR=/tmp/eduvision \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_OFFLINE=1 \
    YOLO_VERBOSE=False

# Function startup configuration.
ENV FUNCTION_COMMAND="python" \
    FUNCTION_ARGS="-m,evaluation_function.main" \
    FUNCTION_INTERFACE="rpc" \
    FUNCTION_RPC_TRANSPORT="ipc" \
    FUNCTION_WORKER_SEND_TIMEOUT="90s"

# Enable temporarily when investigating cold starts.
# Remove or override in production after profiling.
# ENV PYTHONPROFILEIMPORTTIME=1

ENV LOG_LEVEL="debug"
