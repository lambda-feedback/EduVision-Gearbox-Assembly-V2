"""
EduVision — ocena montażu przekładni ze zdjęć studenta.

    response  zdjęcia (adresy URL / data URI / base64; formaty: inputs.py)
    answer    etap do sprawdzenia: nazwa etapu dołączonego do funkcji
              (np. "etap_pelny") albo definicja YAML/JSON (jak
              assemblies/*.yaml w aplikacji GearBox); lista = wszystkie
              etapy muszą być zaliczone
    params    opcjonalnie:
                stage        etap, gdy answer jest pusty
                images       ["3d", "3d_top", "top_view", "photos",
                             "detections"],
                             true (domyślne) lub false
                max_photos   ile zdjęć dla "photos"/"detections" (3)
                image_mode   "upload" (domyślnie) | "inline" | "none"
                image_folder katalog w magazynie obrazów ("eduvision")
                debug        true — log analizy w feedbacku

Wynik: is_correct = wszystkie etapy ZALICZONE; feedback (Markdown) — ocena,
powody, części, połączenia, problemy ze zdjęciami, obrazy.
"""

import tempfile
from pathlib import Path
from typing import Any

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

    try:
        stages = inputs.stage_definitions(answer, params)
    except ValueError as exc:
        return _error(f"Task configuration error: {feedback._e(exc)}")

    try:
        images, problems = inputs.load_images(response)
    except inputs.SubmissionError as exc:
        return _error(feedback._e(exc))

    with tempfile.TemporaryDirectory(prefix="eduvision_stages_") as folder:
        stage_paths = []
        for index, (name, text) in enumerate(stages):
            path = Path(folder) / f"{index:02d}_{name}"
            path.write_text(text, encoding="utf-8")
            stage_paths.append(path)

        # is_correct, items = pipeline.run(
        #     images, stage_paths,
        #     finish=lambda run: feedback.build(run, problems,
        #                                       Attacher(params), params))

        import traceback
        def make_feedback(run_result):
            try:
                is_correct, items = feedback.build(
                    run_result, problems, Attacher(params), params
                )
                # debug_info = (
                #     "### DEBUG PIPELINE\n\n"
                #     f"- **Steps:** `{run_result.get('steps')}`\n"
                #     f"- **Failed step:** `{run_result.get('failed_step')}`\n"
                #     f"- **Executor:** `{run_result.get('executor')}`\n"
                #     f"- **Czas analizy:** `{run_result.get('seconds')}`\n"
                #     f"- **is_correct:** `{is_correct}`\n"
                #     f"- **Liczba sekcji feedbacku:** `{len(items)}`\n"
                # )
                # items.insert(0, ("debug", debug_info))
                items.insert(0, (
                    "debug",
                    f"### DEBUG\n\n"
                    f"- is_correct: `{is_correct}`\n"
                    f"- items type: `{type(items).__name__}`\n"
                    f"- items count: `{len(items)}`\n"
                    f"- items: `{repr(items)[:4000]}`"
                ))

                return is_correct, items
            except Exception:
                error = traceback.format_exc()
                return False, [("debug_error", f"### BŁĄD FEEDBACK.BUILD\n\n{error[-5000:]}")]

        try:
            is_correct, items = pipeline.run(
                images, stage_paths, finish=make_feedback
            )
        except Exception:
            error = traceback.format_exc()
            return Result(
                is_correct=False,
                feedback_items=[
                    ("pipeline_error", f"### BŁĄD PIPELINE\n\n{error[-5000:]}")
                ],
            )


    ##############################################################################################

    # stage = params.get("stage", "Nie określono etapu")
    #
    # is_correct = False
    # items = [
    #     (
    #         "verdict",
    #         f"""
    # ### Wynik analizy
    #
    # - **Sprawdzany etap:** {stage}
    # - **Komponent:** Servo
    # - **Pewność wykrycia:** 95%
    #
    # Komponent został poprawnie rozpoznany.
    #
    # **Dobra robota!**
    # """
    #     )
    # ]

    return Result(is_correct=is_correct, feedback_items=items)
