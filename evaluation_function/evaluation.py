
import json
import tempfile
import traceback
from pathlib import Path
from typing import Any

from lf_toolkit.evaluation import Params, Result

from . import feedback, inputs, pipeline
from .attachments import Attacher


def _error(text):
    return Result(
        is_correct=False,
        feedback_items=[("error", f"\n\n{text}\n\n")]
    )


def _format_diagnostics(data):
    if not data:
        return "No previous interrupted run detected."

    return json.dumps(
        data,
        indent=2,
        ensure_ascii=True,
        default=str
    )[-12000:]


def evaluation_function(
    response: Any,
    answer: Any,
    params: Params,
) -> Result:

    params = dict(params or {})

    # Read the previous diagnostic report BEFORE starting a new pipeline.
    # pipeline.run() resets the diagnostic file at the beginning.
    try:
        previous_diagnostics = pipeline.read_previous_diagnostics()
    except Exception as exc:
        previous_diagnostics = {
            "diagnostic_read_error": repr(exc)
        }

    # Load stage definitions.
    try:
        stages = inputs.stage_definitions(answer, params)

    except ValueError as exc:
        return _error(
            f"Task configuration error: {feedback._e(exc)}"
        )

    # Load submitted images.
    try:
        images, problems = inputs.load_images(response)

    except inputs.SubmissionError as exc:
        return _error(feedback._e(exc))

    # Run the GearBox pipeline.
    try:
        with tempfile.TemporaryDirectory(
            prefix="eduvision_stages_"
        ) as folder:

            stage_paths = []

            for index, (name, text) in enumerate(stages):
                path = Path(folder) / f"{index:02d}_{name}"

                path.write_text(
                    text,
                    encoding="utf-8"
                )

                stage_paths.append(path)

            # Diagnostic logging is handled directly by pipeline.py.
            # Do NOT pass diagnostics=[] here.
            run_result = pipeline.run(
                images,
                stage_paths
            )

        # Current execution summary.
        steps = run_result.get("steps") or []

        step_text = "\n".join(
            f"- {key}: ok={ok}, duration={seconds}s"
            for key, ok, seconds in steps
        )

        log = (run_result.get("log") or "")[-6000:]
        log = log.replace("```", "'''")

        previous_text = _format_diagnostics(
            previous_diagnostics
        )

        debug = (
            "### PIPELINE DIAGNOSTICS\n\n"

            "**Current run:**\n\n"
            f"- Mode: `{run_result.get('mode')}`\n"
            f"- Failed step: `{run_result.get('failed_step')}`\n"
            f"- Executor: `{run_result.get('executor')}`\n"
            f"- Duration: `{run_result.get('seconds')}` seconds\n\n"

            "### Current step history\n\n"
            f"```text\n{step_text or 'No steps recorded.'}\n```\n\n"

            "### Previous interrupted run\n\n"
            f"```json\n{previous_text}\n```\n\n"

            "### Current captured log\n\n"
            f"```text\n{log}\n```"
        )

        return Result(
            is_correct=False,
            feedback_items=[
                ("debug", debug)
            ]
        )

    except Exception:
        error = traceback.format_exc()

        # Read the diagnostic file from the failed current execution.
        try:
            failed_diagnostics = pipeline.read_previous_diagnostics()

        except Exception as exc:
            failed_diagnostics = {
                "diagnostic_read_error": repr(exc)
            }

        diagnostic_text = _format_diagnostics(
            failed_diagnostics
        )

        previous_text = _format_diagnostics(
            previous_diagnostics
        )

        return Result(
            is_correct=False,
            feedback_items=[
                (
                    "debug_error",
                    "### PIPELINE ERROR\n\n"

                    "### Current execution diagnostics\n\n"
                    f"```json\n{diagnostic_text}\n```\n\n"

                    "### Previous execution diagnostics\n\n"
                    f"```json\n{previous_text}\n```\n\n"

                    "### Python traceback\n\n"
                    f"```text\n{error[-8000:]}\n```"
                )
            ]
        )
