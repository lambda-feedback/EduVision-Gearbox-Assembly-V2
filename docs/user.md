# EduVision — gearbox assembly check

Students upload 4–10 photos of their gearbox assembly. The function
reconstructs the assembly in 3D and checks it against the stage you set
for the question.

## Setting up a question

1. Add an **Image Input** to the question so students can upload photos.
2. Select this evaluation function.
3. In **Evaluation Function Parameters** add the key `stage` with the stage
   to check, e.g. `etap_2`. The answer field can stay empty (if it is
   filled in, it is used instead of `stage`).

**Stage** — one of:

* a bundled stage name, e.g. `etap_pelny` (full 5-gear assembly),
  `etap_1_duza`, `etap_1_zebnik`, `etap_2`, `etap_wspolna_os`,
  `etap_elementy`,
* or your own stage definition in YAML:

```yaml
name: "Stage 2 - motor + two gears"
description: "The motor drives gear 1, gear 1 drives gear 2."
parts:
  engine: 1
  driving_gear: 1
  gear_big: 2
relations:
  - mesh: [driving_gear, gear_big#1]
    sections: [main, big]
    description: "motor gear meshes with the large disc of gear 1"
  - mesh: [gear_big#1, gear_big#2]
    sections: [pinion, big]
  - driven: gear_big#2
```

Part counts must match exactly. A stage without `relations` only checks
which parts are present, and it also works when the AprilTag is not
visible.

**Params** (optional), e.g. `{"images": ["3d", "photos"], "max_photos": 2}`
to show the 3D model and two photos with the recognised parts overlaid.

## What students see

* PASSED / FAILED / CANNOT ASSESS for the stage, with what is wrong
  (missing or extra parts, gears that do not mesh, gears that do not turn),
* which photos were rejected and why (tag not visible, too far, blurred,
  too dark) and whether the photos cover enough sides,
* the parts found and which of them the motor drives,
* a 3D image of the recognised assembly.

## Photo guidelines for students

* the AprilTag on the base must be fully visible and sharp in every photo,
* 4–10 photos from different sides (walk around the model), from above at
  an angle, close enough that the tag is clearly visible,
* good, even light; original photos (not screenshots or messenger copies).
