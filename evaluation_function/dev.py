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
                  path to a .yaml file
        response  folder with photos, photo files / URLs (comma separated)
                  or the raw response JSON
        --params  params JSON, e.g. '{"images": ["3d", "photos"]}'
        --html    save the feedback as an HTML page (images inline)
    """
    parser = argparse.ArgumentParser(
        prog="python -m evaluation_function.dev")
    parser.add_argument("answer")
    parser.add_argument("response")
    parser.add_argument("--params", default="{}")
    parser.add_argument("--html", default=None)
    args = parser.parse_args()

    # Lokalne pliki dozwolone tylko tutaj (nie w serwerze).
    os.environ["EDUVISION_ALLOW_LOCAL_FILES"] = "1"
    from .evaluation import evaluation_function

    answer = args.answer
    if answer.lower().endswith((".yaml", ".yml")) and Path(answer).is_file():
        answer = Path(answer).read_text(encoding="utf-8")

    response = args.response
    if Path(response).is_dir():
        response = [str(p) for p in sorted(Path(response).iterdir())
                    if p.suffix.lower() in IMAGE_EXTENSIONS]

    params = json.loads(args.params)
    if args.html:
        params.setdefault("image_mode", "inline")

    started = time.time()
    result = evaluation_function(response, answer, params)
    result_dict = result.to_dict(include_test_data=True)
    print(json.dumps({k: v for k, v in result_dict.items()
                      if k != "feedback"}, indent=2))
    print(f"time: {time.time() - started:.1f} s")

    if args.html:
        Path(args.html).write_text(
            "<!doctype html><meta charset='utf-8'><title>EduVision feedback"
            "</title><body style='font-family:sans-serif;max-width:900px;"
            f"margin:auto'>{result_dict['feedback']}</body>",
            encoding="utf-8")
        print(f"feedback: {Path(args.html).resolve()}")
    else:
        print(result_dict["feedback"])


if __name__ == "__main__":
    sys.exit(dev())
