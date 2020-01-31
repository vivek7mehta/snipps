"""
test cases for train_test_split
"""
import unittest
import numpy as np
from sklearn.model_selection import train_test_split as sk_tts
from src.train_test_split import train_test_split

class TestTrainTestSplit(unittest.TestCase):

    def test_size_mismatch(self):
        x = np.random.randint(0, 100, size=(1000, 65, 56, 3)).tolist()
        y = np.random.randint(0, 100, size=(1000, 10)).tolist()
        z = np.random.randint(0, 100, size=(500, 10)).tolist()
        with self.assertRaises(AssertionError):
            _ , _, _, _ = train_test_split(x, y, z)

    def test_list_split(self):
        x = np.random.randint(0, 100, size=(1000, 65, 56, 3)).tolist()
        y = np.random.randint(0, 100, size=(1000, 10)).tolist()

        x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, random_seed=2)
        x_train2, x_test2, y_train2, y_test2 = sk_tts(x, y, random_state=2)

        self.assertEqual(x_train1, x_train2)
        self.assertEqual(x_test1, x_test2)
        self.assertEqual(y_train1, y_train2)
        self.assertEqual(y_test1, y_test2)

    def test_np_array_split(self):
        x = np.random.randint(0, 100, size=(1000, 65, 56, 3)).tolist()
        y = np.random.randint(0, 100, size=(1000, 10))

        x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, random_seed=2)
        x_train2, x_test2, y_train2, y_test2 = sk_tts(x, y, random_state=2)

        self.assertEqual(x_train1, x_train2)
        self.assertEqual(x_test1, x_test2)
        self.assertTrue(np.array_equal(y_train1,y_train2))
        self.assertTrue(np.array_equal(y_test1, y_test2))

def run_test():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTrainTestSplit)
    unittest.TextTestRunner(verbosity=2).run(suite)
