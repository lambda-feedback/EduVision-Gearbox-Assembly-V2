# EduVision — gearbox assembly check

Checks a student's gearbox assembly from 4–10 phone photos. The photos go
through the GearBox analysis pipeline (AprilTag world frame, YOLO part
segmentation, CAD pose fitting, multi-view 3D fusion, assembly check), and
the result is compared with the stage given as the answer.

## Layout

| Path | Content |
|---|---|
| `evaluation_function/evaluation.py` | entry point: parse input, run analysis, build feedback |
| `evaluation_function/inputs.py` | photos from the response, stage from the answer |
| `evaluation_function/pipeline.py` | runs the GearBox stages in-process, one request at a time |
| `evaluation_function/feedback.py` | HTML feedback from the analysis reports |
| `evaluation_function/render.py` | 3D image of the assembly without OpenGL |
| `evaluation_function/attachments.py` | images in the feedback (upload / inline) |
| `evaluation_function/gearbox/` | **copy** of the GearBox application code, YOLO model, CAD models and stage files |
| `tools/sync_gearbox.py` | refreshes `gearbox/` from the GearBox application |

Do not edit `evaluation_function/gearbox/` by hand: change the GearBox
application and run `python tools/sync_gearbox.py [path to the app]`.

## Inputs

### `response` — photos

Any of these, also nested:

```json
["https://.../IMG_1.jpg", "https://.../IMG_2.jpg"]
[{"url": "https://...", "name": "IMG_1.jpg"}]
[{"data": "<base64>", "name": "IMG_1.jpg"}]
{"images": ["data:image/jpeg;base64,..."]}
"https://.../IMG_1.jpg, https://.../IMG_2.jpg"
```

Keys recognised in objects: `url`/`src`/`href`/`uri`/`link`,
`data`/`base64`/`content`/`dataUrl`, `name`/`filename`/`fileName`.
JPEG, PNG, BMP and WEBP are analysed (HEIC only if `pillow-heif` is
installed). Original files are best: EXIF focal length improves the camera
calibration. At most 30 photos (`EDUVISION_MAX_IMAGES`), 40 MB each
(`EDUVISION_MAX_IMAGE_MB`).

### Stage to check — `params.stage` (or `answer`)

Set the stage in the question's **Evaluation Function Parameters** as
`stage`, e.g. `{"stage": "etap_2"}`; the answer field can stay empty. If
`answer` is not empty it is used instead of `params.stage`. Accepted values:

* name of a bundled stage (`gearbox/assemblies/*.yaml`), e.g. `"etap_pelny"`,
* a stage definition as YAML/JSON text or an object (same format as the
  GearBox stage files),
* a list of the above — all stages must pass.

```yaml
name: "Stage 2 - motor + two gears"
parts:                       # exact number of each class
  engine: 1
  driving_gear: 1
  gear_big: 2
relations:                   # none = "parts only" stage (works without the tag)
  - mesh: [driving_gear, gear_big#1]
    sections: [main, big]    # big | pinion | main | any
  - mesh: [gear_big#1, gear_big#2]
    sections: [pinion, big]
  - driven: gear_big#2
```

### `params` (all optional)

| Key | Default | Meaning |
|---|---|---|
| `stage` | – | stage to check (see above) |
| `images` | `true` | images in the feedback: list of `3d`, `top_view`, `photos`, `detections`; `true` = `3d` + `top_view` (no tag: `detections`); `false` = none |
| `max_photos` | `3` | how many photos for `photos` / `detections` |
| `image_mode` | `upload` | `upload` (storage URL), `inline` (data URI), `none` |
| `image_folder` | `eduvision` | folder in the image storage |
| `debug` | `false` | add the analysis log to the feedback |

## Output

`is_correct` is `true` when every stage is PASSED. `feedback` is Markdown
(headings, lists, images as `![caption](url)`) with these parts (tags):
`verdict` (result, what is wrong, parts, connections), `photos` (rejected
photos and why, quality warnings, viewpoint coverage), `parts` (parts found,
what the motor drives), `images`, `error`. lf_toolkit joins the parts with
`<br>` and shows only their text (tags are not displayed), so every part
starts and ends with a blank line.

A missing stage or a submission without readable photos returns
`is_correct: false` with an explanation (no error status).

## Environment

| Variable | Meaning |
|---|---|
| `EDUVISION_WORK_DIR` | work directory (`/tmp/eduvision` in the image) |
| `EDUVISION_WORKERS` | `auto` (processes if available — not on AWS Lambda), `0` (single process), or a number |
| `EDUVISION_IMAGE_MODE` | default `image_mode` |
| `S3_BUCKET_URI`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | image upload to S3 (`lf_toolkit.upload_image`); or `GCS_BUCKET` for GCS |

Without image storage the feedback says that images could not be attached;
the rest of the feedback is unaffected.

## Running locally

```bash
poetry install
poetry run python -m evaluation_function.dev etap_pelny path/to/photos --md feedback.md --html feedback.html
# stage from params instead of the answer ("-" = empty answer):
poetry run python -m evaluation_function.dev - path/to/photos --params '{"stage": "etap_2"}' --html feedback.html
poetry run pytest
# full analysis test (about 30-60 s):
EDUVISION_TEST_IMAGES=path/to/photos EDUVISION_TEST_EXPECT=1 poetry run pytest
```

## Performance

11 photos (3000×4000): about 30 s with a GPU and worker processes, about
50 s on a desktop CPU in a single process. The server image uses CPU-only
PyTorch; the first request also loads the libraries and the model.
