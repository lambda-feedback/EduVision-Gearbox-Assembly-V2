FROM ghcr.io/lambda-feedback/evaluation-function-base/python:3.12 AS builder

# Must match the Poetry major version that generated poetry.lock
RUN pip install "poetry>=2,<3"

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

COPY pyproject.toml poetry.lock ./

RUN --mount=type=cache,target=$POETRY_CACHE_DIR \
    poetry install --without dev --no-root

FROM ghcr.io/lambda-feedback/evaluation-function-base/python:3.12

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Copy the evaluation function to the app directory
COPY evaluation_function ./evaluation_function

# Precompile function code too: the deployed filesystem is read-only, so
# anything not compiled here would be recompiled on every cold start.
RUN python -m compileall -q .

# Command to start the evaluation function with
ENV FUNCTION_COMMAND="python"

# Args to start the evaluation function with
ENV FUNCTION_ARGS="-m,evaluation_function.main"

# Stdio avoids waiting for Python imports before the Lambda init response.
ENV FUNCTION_INTERFACE="rpc"
ENV FUNCTION_RPC_TRANSPORT="stdio"

# Torch and the analysis pipeline can take longer than shimmy's 30 s default.
ENV FUNCTION_WORKER_SEND_TIMEOUT="90s"

# TEMPORARY: log per-module import times to stderr while measuring cold starts.
ENV PYTHONPROFILEIMPORTTIME=1

ENV LOG_LEVEL="debug"
