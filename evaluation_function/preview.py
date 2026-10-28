from typing import Any

from lf_toolkit.preview import Params, Preview, Result

from .inputs import collect_sources


def preview_function(response: Any, params: Params) -> Result:
    """
    Podgląd przed wysłaniem: tylko liczba zdjęć (bez pobierania i analizy,
    ta trwa kilkadziesiąt sekund i odbywa się po wysłaniu).
    """
    try:
        count = len(collect_sources(response))
        if count == 0:
            text = "No photos yet."
        elif count < 4:
            text = (f"{count} photo(s) — 4-10 photos from different sides "
                    f"are recommended.")
        else:
            text = f"{count} photos ready to submit."
        return Result(preview=Preview(latex="", sympy="", feedback=text))
    except Exception as e:
        return Result(preview=Preview(latex="", sympy="", feedback=str(e)))
