import unittest

import autora


class SmokeTest(unittest.TestCase):
    def test_import(self):
        self.assertIsNotNone(autora)


if __name__ == "__main__":
    unittest.main()
