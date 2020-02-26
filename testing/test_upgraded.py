import unittest
from my_sum import sum


class TestSum(unittest.TestCase):

    def setUp(self):
        self.data = [1, 2, 3]

    def test_sum(self):
        result = sum(self.data)
        self.assertEqual(result, 6)

    def test_sum_tuple(self):
        data = tuple(self.data)
        result = sum(data)
        self.assertEqual(result, 6)

    def test_bad_type(self):
        data = "banana"
        with self.assertRaises(TypeError):
            result = sum(data)


if __name__ == '__main__':
    unittest.main()
