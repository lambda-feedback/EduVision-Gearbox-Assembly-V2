"""
pipeline: filtr obrazów diagnostycznych (pomijany odczyt i zapis, gdy
feedback ich nie pokazuje) i dobór rozmiaru puli procesów.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from . import pipeline


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.image = self.folder / "IMG_0.jpg"
        cv2.imwrite(str(self.image), np.zeros((30, 40, 3), np.uint8))

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_filtered_cv2(self):
        proxy = pipeline._FilteredCv2(
            cv2, reads={self.folder: "photos"},
            writes={self.folder: "photos"})
        saved = dict(pipeline._OUTPUTS)
        try:
            pipeline._OUTPUTS["photos"] = False
            self.assertIsNone(proxy.imread(str(self.image)))
            self.assertTrue(proxy.imwrite(str(self.folder / "x.jpg"),
                                          np.zeros((4, 4, 3), np.uint8)))
            self.assertFalse((self.folder / "x.jpg").exists())
            pipeline._OUTPUTS["photos"] = True
            self.assertIsNotNone(proxy.imread(str(self.image)))
        finally:
            pipeline._OUTPUTS.update(saved)

    def test_pool_size(self):
        with mock.patch.dict(os.environ, {"EDUVISION_WORKERS": "0"}):
            self.assertEqual(pipeline.pool_size(), 0)
        with mock.patch.dict(os.environ, {"EDUVISION_WORKERS": "3"}):
            expected = 3 if pipeline.fork_available() else 0
            self.assertEqual(pipeline.pool_size(), expected)
        self.assertGreaterEqual(pipeline.effective_cpus(), 1)


if __name__ == "__main__":
    unittest.main()
