import base64
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from . import feedback, inputs
from .attachments import Attacher
from .evaluation import Params, evaluation_function


def _jpeg_bytes():
    ok, data = cv2.imencode(".jpg", np.full((40, 60, 3), 128, np.uint8))
    return data.tobytes()


class TestInputs(unittest.TestCase):

    def test_collect_sources_formats(self):
        response = {
            "images": [
                "https://example.com/a.jpg",
                {"url": "https://example.com/b", "name": "IMG_2.jpg"},
                {"data": "AAAA", "filename": "c.png"},
            ],
            "comment": "not an image",
        }
        sources = inputs.collect_sources(response)
        self.assertEqual(sources[0], (None, "https://example.com/a.jpg"))
        self.assertEqual(sources[1], ("IMG_2.jpg", "https://example.com/b"))
        self.assertEqual(sources[2], ("c.png", "base64:AAAA"))
        self.assertEqual(len(sources), 3)

    def test_collect_sources_json_text_and_list_text(self):
        self.assertEqual(len(inputs.collect_sources(
            '[{"url": "https://x/1.jpg"}, {"url": "https://x/2.jpg"}]')), 2)
        self.assertEqual(len(inputs.collect_sources(
            "https://x/1.jpg, https://x/2.jpg")), 2)

    def test_load_images_data_uri(self):
        uri = ("data:image/jpeg;base64,"
               + base64.b64encode(_jpeg_bytes()).decode())
        images, problems = inputs.load_images(
            [uri, {"data": base64.b64encode(_jpeg_bytes()).decode(),
                   "name": "photo 1.jpg"}])
        self.assertEqual([n for n, _ in images],
                         ["image_01.jpg", "photo_1.jpg"])
        self.assertEqual(problems, [])

    def test_load_images_rejects_non_images(self):
        uri = "data:text/plain;base64," + base64.b64encode(b"hello").decode()
        with self.assertRaises(inputs.SubmissionError):
            inputs.load_images([uri])
        with self.assertRaises(inputs.SubmissionError):
            inputs.load_images("")

    def test_local_files_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.jpg"
            path.write_bytes(_jpeg_bytes())
            saved = os.environ.pop("EDUVISION_ALLOW_LOCAL_FILES", None)
            try:
                self.assertEqual(inputs.collect_sources(str(path)), [])
            finally:
                if saved is not None:
                    os.environ["EDUVISION_ALLOW_LOCAL_FILES"] = saved

    def test_stage_bundled_name(self):
        stages = inputs.stage_definitions("etap_pelny", {})
        self.assertEqual(stages[0][0], "etap_pelny.yaml")
        self.assertIn("parts:", stages[0][1])

    def test_stage_yaml_dict_and_list(self):
        text = "name: Test\nparts:\n  gear_big: 2\n"
        self.assertEqual(len(inputs.stage_definitions(text, {})), 1)
        stages = inputs.stage_definitions(
            [{"name": "A", "parts": {"engine": 1}}, "etap_2"], {})
        self.assertEqual([s[0] for s in stages], ["A.yaml", "etap_2.yaml"])
        self.assertEqual(len(inputs.stage_definitions(
            "[etap_2, etap_pelny]", {})), 2)

    def test_stage_from_params(self):
        stages = inputs.stage_definitions("", {"stage": "etap_2"})
        self.assertEqual(stages[0][0], "etap_2.yaml")

    def test_stage_errors(self):
        for answer in ("", "no_such_stage", {"name": "x"}):
            with self.assertRaises(ValueError):
                inputs.stage_definitions(answer, {})


def _run(verdict, mode="3d"):
    return {
        "mode": mode,
        "failed_step": None,
        "log": "",
        "output_dir": Path(tempfile.gettempdir()),
        "apriltag": [
            {"image": "a.jpg", "status": "ACCEPTED", "reason": ["OK"],
             "warnings": ["Image is dark"]},
            {"image": "b.jpg", "status": "REJECTED",
             "reason": ["AprilTag ID 0 not detected"], "warnings": []},
        ],
        "camera": {"images": [], "coverage": {
            "warnings": ["Only 1 usable image"]}},
        "check": {
            "inventory": {"source": "3D", "parts": {"gear_big": 1},
                          "uncertain_classes": []},
            "drive_train": [],
            "assemblies": [{
                "name": "Stage <1>", "kind": "assembly", "verdict": verdict,
                "reasons": ["missing engine: found 0, expected 1"],
                "counts": [{"class": "engine", "expected": 1, "found": 0,
                            "result": "ERROR", "uncertain": False}],
                "relations": [{"type": "mesh", "roles": ["a", "b"],
                               "description": "mesh a-b", "result": "ERROR",
                               "detail": "missing part: engine#1"}],
                "uncertain": [],
            }],
        },
    }


class TestFeedback(unittest.TestCase):

    def test_failed(self):
        ok, items = feedback.build(_run("FAILED"), ["x.gif: not supported"],
                                   Attacher({"image_mode": "none"}),
                                   {"images": False})
        self.assertFalse(ok)
        md = "".join(h for _, h in items)
        self.assertIn("### ❌ FAILED", md)
        self.assertIn("Stage \\<1\\>", md)               # escaped
        self.assertIn("- missing engine", md)
        self.assertIn("`b.jpg`: AprilTag ID 0 not detected", md)
        self.assertIn("Image is dark", md)
        self.assertIn("Only 1 usable image", md)
        self.assertIn("x.gif", md)
        self.assertNotIn("<p>", md)
        # Każdy element otoczony pustymi liniami (łączone przez "<br>").
        for _, text in items:
            self.assertTrue(text.startswith("\n\n") and text.endswith("\n\n"))
        self.assertEqual([t for t, _ in items], ["verdict", "photos",
                                                 "parts"])

    def test_passed(self):
        ok, _ = feedback.build(_run("PASSED"), [], Attacher({}),
                               {"images": False})
        self.assertTrue(ok)

    def test_no_tag_message(self):
        _, items = feedback.build(_run("CANNOT ASSESS", mode="notag"), [],
                                  Attacher({}), {"images": False})
        self.assertIn("AprilTag was not found", dict(items)["photos"])


class TestEvaluationFunction(unittest.TestCase):

    def test_configuration_error(self):
        result = evaluation_function("https://x/1.jpg", "", Params())
        data = result.to_dict()
        self.assertFalse(data["is_correct"])
        self.assertIn("configuration", data["feedback"])

    def test_no_photos(self):
        result = evaluation_function("", "etap_pelny", Params())
        data = result.to_dict()
        self.assertFalse(data["is_correct"])
        self.assertIn("No photos", data["feedback"])

    @unittest.skipUnless(os.environ.get("EDUVISION_TEST_IMAGES"),
                         "set EDUVISION_TEST_IMAGES=<folder with photos>")
    def test_full_analysis(self):
        """Pełna analiza (kilkadziesiąt sekund): EDUVISION_TEST_IMAGES =
        folder ze zdjęciami, EDUVISION_TEST_STAGE = etap (etap_pelny),
        EDUVISION_TEST_EXPECT = 1/0 — oczekiwany wynik."""
        folder = Path(os.environ["EDUVISION_TEST_IMAGES"])
        response = [
            {"data": base64.b64encode(p.read_bytes()).decode(),
             "name": p.name}
            for p in sorted(folder.iterdir())
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        stage = os.environ.get("EDUVISION_TEST_STAGE", "etap_pelny")
        result = evaluation_function(response, stage,
                                     {"image_mode": "none"}).to_dict()
        self.assertIn("### Photos", result["feedback"])
        expect = os.environ.get("EDUVISION_TEST_EXPECT")
        if expect is not None:
            self.assertEqual(result["is_correct"], expect == "1")
