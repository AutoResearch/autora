import unittest

import aer


class SmokeTest(unittest.TestCase):
    def test_import(self):
        self.assertIsNotNone(aer)


if __name__ == "__main__":
    unittest.main()
