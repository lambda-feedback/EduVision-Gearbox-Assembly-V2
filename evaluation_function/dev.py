import argparse
import json
import os
import sys
import time
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dev():
    """Run the evaluation function from the command line for development.

    Usage:
        python -m evaluation_function.dev <answer> <response> [options]

        answer    bundled stage name (e.g. etap_pelny), YAML/JSON text or a
                  path to a .yaml file; "-" = empty (stage from --params)
        response  folder with photos, photo files / URLs (comma separated)
                  or the raw response JSON
        --params  params JSON, e.g. '{"images": ["3d", "photos"]}'
        --html    save an HTML preview of the feedback (images inline;
                  the Markdown is rendered by marked.js from cdnjs)
        --md      save the feedback Markdown as it is sent to the platform
    """
    parser = argparse.ArgumentParser(
        prog="python -m evaluation_function.dev")
    parser.add_argument("answer")
    parser.add_argument("response")
    parser.add_argument("--params", default="{}")
    parser.add_argument("--html", default=None)
    parser.add_argument("--md", default=None)
    args = parser.parse_args()
    # Konsola Windows (cp1250) nie zna emoji z feedbacku.
    sys.stdout.reconfigure(errors="replace")

    # Lokalne pliki dozwolone tylko tutaj (nie w serwerze).
    os.environ["EDUVISION_ALLOW_LOCAL_FILES"] = "1"
    from .evaluation import evaluation_function

    answer = "" if args.answer == "-" else args.answer
    if answer.lower().endswith((".yaml", ".yml")) and Path(answer).is_file():
        answer = Path(answer).read_text(encoding="utf-8")

    response = args.response
    if Path(response).is_dir():
        response = [str(p) for p in sorted(Path(response).iterdir())
                    if p.suffix.lower() in IMAGE_EXTENSIONS]

    params = json.loads(args.params)
    if args.html or args.md:
        params.setdefault("image_mode", "inline")

    started = time.time()
    result = evaluation_function(response, answer, params)
    result_dict = result.to_dict(include_test_data=True)
    print(json.dumps({k: v for k, v in result_dict.items()
                      if k != "feedback"}, indent=2))
    print(f"time: {time.time() - started:.1f} s")

    feedback = result_dict["feedback"]
    if args.md:
        Path(args.md).write_text(feedback, encoding="utf-8")
        print(f"feedback (Markdown): {Path(args.md).resolve()}")
    if args.html:
        # Podgląd: Markdown zamieniany na HTML w przeglądarce (marked.js).
        source = json.dumps(feedback).replace("</", "<\\/")
        Path(args.html).write_text(
            "<!doctype html><meta charset='utf-8'><title>EduVision feedback"
            "</title><body style='font-family:sans-serif;max-width:900px;"
            "margin:auto'><div id='fb'></div><script src='https://cdnjs."
            "cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js'></script>"
            f"<script>document.getElementById('fb').innerHTML = "
            f"marked.parse({source});</script>"
            "<style>img{max-width:100%}</style></body>",
            encoding="utf-8")
        print(f"feedback (HTML preview): {Path(args.html).resolve()}")
    if not args.md and not args.html:
        print(feedback)


if __name__ == "__main__":
    sys.exit(dev())
