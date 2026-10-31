import tempfile
from pathlib import Path
from typing import Any
import traceback

from lf_toolkit.evaluation import Params, Result

from . import feedback, inputs, pipeline
from .attachments import Attacher


def _error(text):
    return Result(is_correct=False,
                  feedback_items=[("error", f"\n\n{text}\n\n")])


def evaluation_function(
    response: Any,
    answer: Any,
    params: Params,
) -> Result:

    params = dict(params or {})
    diagnostics = []

    try:
        stages = inputs.stage_definitions(answer, params)
    except ValueError as exc:
        return _error(f"Task configuration error: {feedback._e(exc)}")

    try:
        images, problems = inputs.load_images(response)
    except inputs.SubmissionError as exc:
        return _error(feedback._e(exc))

    try:

        with tempfile.TemporaryDirectory(
            prefix="eduvision_stages_"
        ) as folder:

            stage_paths = []

            for index, (name, text) in enumerate(stages):
                path = Path(folder) / f"{index:02d}_{name}"
                path.write_text(text, encoding="utf-8")
                stage_paths.append(path)

            run_result = pipeline.run(
                images,
                stage_paths,
                diagnostics=diagnostics
            )

        diagnostic_text = "\n".join(
            (
                f"- {d.get('event')} | "
                f"step={d.get('step')} | "
                f"ok={d.get('ok')} | "
                f"time={d.get('seconds')}s | "
                f"error={d.get('error')}"
            )
            for d in diagnostics
        )

        log = (run_result.get("log") or "")[-6000:]

        debug = (
            "### PIPELINE DIAGNOSTICS\n\n"
            f"**Mode:** `{run_result.get('mode')}`\n\n"
            f"**Failed step:** `{run_result.get('failed_step')}`\n\n"
            f"**Executor:** `{run_result.get('executor')}`\n\n"
            f"**Duration:** `{run_result.get('seconds')}` seconds\n\n"
            "### Step history\n\n"
            f"```text\n{diagnostic_text or 'No steps recorded.'}\n```\n\n"
            "### Captured log\n\n"
            f"```text\n{log}\n```"
        )

        return Result(
            is_correct=False,
            feedback_items=[("debug", debug)]
        )

    except Exception:
        error = traceback.format_exc()

        diagnostic_text = repr(diagnostics)

        return Result(
            is_correct=False,
            feedback_items=[
                (
                    "debug_error",
                    "### PIPELINE ERROR\n\n"
                    f"**Recorded steps:**\n\n"
                    f"```text\n{diagnostic_text}\n```\n\n"
                    f"**Traceback:**\n\n"
                    f"```text\n{error[-8000:]}\n```"
                )
            ]
        )
