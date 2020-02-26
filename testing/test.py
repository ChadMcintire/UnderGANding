import unittest
from my_sum import sum

class TestSum(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSum, self).__init__(*args, **kwargs)
        self.data = [1,2,3]
    def test_sum(self):
        result = sum(self.data)
        self.assertEqual(result, 6)

    def test_sum_tuple(self):
        data = tuple(self.data)
        result = sum(data)
        self.assertEqual(result, 6)

if __name__ == '__main__':
    unittest.main()
