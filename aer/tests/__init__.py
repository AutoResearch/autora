import unittest

class TestTheoristExperimentalist(unittest.TestCase):

    def test_weber_init(self):
        self.assertEqual('foo'.upper(), 'FOO')

if __name__ == '__main__':
    unittest.main()