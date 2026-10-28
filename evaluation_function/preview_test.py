import unittest

from .preview import Params, preview_function


class TestPreviewFunction(unittest.TestCase):

    def test_no_photos(self):
        result = preview_function("", Params())
        self.assertIn("preview", result)
        self.assertEqual(result["preview"]["feedback"], "No photos yet.")

    def test_counts_photos(self):
        response = [{"url": f"https://example.com/{i}.jpg"} for i in range(5)]
        result = preview_function(response, Params())
        self.assertIn("5 photos", result["preview"]["feedback"])

    def test_few_photos(self):
        result = preview_function("https://example.com/a.jpg", Params())
        self.assertIn("recommended", result["preview"]["feedback"])
