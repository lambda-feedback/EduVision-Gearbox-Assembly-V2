
"""
EduVision - gearbox assembly assessment based on student photos.

Inputs:
    response  photos (URLs / data URI / base64; supported formats: inputs.py)
    answer    stage to check: the name of a stage attached to the function
              (e.g. "etap_pelny") or a YAML/JSON definition (as in
              assemblies/*.yaml in the GearBox application); a list means
              that all stages must be passed
    params    optional:
                  stage        stage to use when answer is empty
                  images       ["3d", "3d_top", "top_view", "photos",
                               "detections"],
                               true (default) or false
                  max_photos   number of photos for "photos"/"detections" (3)
                  image_mode   "upload" (default) | "inline" | "none"
                  image_folder folder in the image storage ("eduvision")
                  debug        true - include analysis log in feedback

Output:
    is_correct = all stages PASSED;
    feedback (Markdown) - verdict, reasons, parts, connections,
    photo issues and images.
"""

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


def evaluation_function(
    response: Any,
    answer: Any,
    params: Params,
) -> Result:

    params = dict(params or {})

    # Load stage definitions
    try:
        stages = inputs.stage_definitions(answer, params)
    except ValueError as exc:
        return _error(
            f"Task configuration error: {feedback._e(exc)}"
        )

    # Load submitted images
    try:
        images, problems = inputs.load_images(response)
    except inputs.SubmissionError as exc:
        return _error(feedback._e(exc))

    # Generate feedback from pipeline results
    def make_feedback(run_result):
        try:
            is_correct, items = feedback.build(
                run_result,
                problems,
                Attacher(params),
                params
            )

            debug = (
                "### DEBUG\n\n"
                f"- is_correct: `{is_correct}`\n"
                f"- items type: `{type(items).__name__}`\n"
                f"- items count: `{len(items)}`\n"
                f"- items: `{repr(items)[:4000]}`"
            )

            return False, [("debug", debug)]

        except Exception:
            error = traceback.format_exc()

            return False, [
                (
                    "debug_error",
                    f"### FEEDBACK.BUILD ERROR\n\n{error[-5000:]}"
                )
            ]

    # Create temporary stage files and run the pipeline
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

            # IMPORTANT:
            # pipeline.run() must execute inside the TemporaryDirectory
            # context because it needs access to the temporary stage files.

            is_correct, items = pipeline.run(
                images,
                stage_paths,
                finish=make_feedback
            )

    except Exception:
        error = traceback.format_exc()

        return Result(
            is_correct=False,
            feedback_items=[
                (
                    "pipeline_error",
                    f"### PIPELINE ERROR\n\n{error[-5000:]}"
                )
            ],
        )

    return Result(
        is_correct=is_correct,
        feedback_items=items
    )
