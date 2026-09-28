import unittest
from utils import normalize_score


class TestNormalizeScore(unittest.TestCase):
    def test_normalize_score(self):
        self.assertEqual(normalize_score(50, 100), 0.5)

    def test_full_score(self):
        self.assertEqual(normalize_score(100, 100), 1.0)

    def test_invalid_maximum(self):
        with self.assertRaises(ValueError):
            normalize_score(50, 0)


if __name__ == "__main__":
    unittest.main()