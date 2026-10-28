import unittest

from .preview import Params, preview_function

class TestPreviewFunction(unittest.TestCase):
    """
    TestCase Class used to test the algorithm.
    ---
    Tests are used here to check that the algorithm written
    is working as it should.

    It's best practice to write these tests first to get a
    kind of 'specification' for how your algorithm should
    work, and you should run these tests before committing
    your code to AWS.

    Read the docs on how to use unittest here:
    https://docs.python.org/3/library/unittest.html

    Use preview_function() to check your algorithm works
    as it should.
    """

    def test_preview(self):
        response, params = "A", Params()
        result = preview_function(response, params)

        self.assertIn("preview", result)
        self.assertIsNotNone(result["preview"])
