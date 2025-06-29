import unittest
from src.evaluate import evaluate_model

class TestEvaluation(unittest.TestCase):
    def test_accuracy_above_threshold(self):
        accuracy = evaluate_model()
        self.assertGreaterEqual(accuracy, 0.7)

if __name__ == '__main__':
    unittest.main()
